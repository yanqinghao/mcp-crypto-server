# -*- coding: utf-8 -*-
"""
multi_loop.py
单进程调度 15m+1h 与 1h+4h 两套策略：
- 每个 15 分钟边界：跑 M15
- 每个整点边界：同时跑 M15 和 H1，并“合并成一条消息”（分段展示）
- 启动后立刻先跑一次 M15
- 风控路由：M15 标记 [L1]，H1 标记 [L2] 且小幅加权（score_boost）
- 需要 your_package 内已有：signals_pro, universe, formatter, notifier, exchange, strategies
"""

import time
from typing import Dict, List, Tuple

from .loggingx import dbg, ts_now
from .config import (
    SLEEP_MS,
    POLL_SEC,
    PER_MESSAGE_LIMIT,
    TITLE_PREFIX,
    SEPARATOR_LINE,
    ONLY_PUSH_EXPLODE,
    MIN_SIGNAL_SCORE,
    EXPLODE_QUIET_EXTRA_SCORE,
    MODE,
    CANDIDATE_REFRESH_SEC,
    FRAME_SEC,
    AUTO_DELETE_HOURS,
    AUTO_DELETE_GRACE,
)
from .strategies import Strategy
from .strategies.m15 import M15
from .strategies.h1_with_4h import H1_4H
from .exchange import build_exchange
from .candidates import hourly_refresh_candidates
from .detect_pro import detect_signal  # 若要用基础版改成 .signals
from .notifier import telegram_send, schedule_delete, cleanup_pending_deletes
from .formatter import format_signal_cn


def crossed_boundary(prev_ts: int, now_ts: int, frame_sec: int) -> bool:
    return (prev_ts // frame_sec) != (now_ts // frame_sec)


# —— 风控路由：按策略加权 & 等级标签 —— #
LEVEL_LABEL = {
    M15.name: "[L1]",  # 15m
    H1_4H.name: "[L2]",  # 1h
}
LEVEL_SCORE_BOOST = {
    M15.name: 0.00,  # 不加权
    H1_4H.name: 0.05,  # 1h 信号略微加权，利于排序优先
}


def _collect_for_strategy(
    ex,
    strategy: Strategy,
    candidates: List[str],
    strong_up_map: Dict[str, bool],
    strong_dn_map: Dict[str, bool],
    last_alert_at: Dict[Tuple[str, str, str], float],
) -> List[dict]:
    """
    扫描一个策略，收集 payload 列表（不发送）。
    """
    if not candidates:
        dbg(f"[{strategy.name}] No candidates; skip.")
        return []

    collected = []
    dbg(f"[{strategy.name}] Scanning {len(candidates)} symbols")
    for sym in candidates:
        try:
            ok, payload = detect_signal(ex, sym, strong_up_map, strong_dn_map, strategy)
            ex.sleep(SLEEP_MS)
            if not ok:
                continue

            # 冷却：粒度 (策略, 品种, 信号)
            key = (strategy.name, payload["symbol"], payload["kind"])
            if strategy.overrides:
                cd_override = strategy.overrides.get("ALERT_COOLDOWN_SEC", None)
            else:
                cd_override = None
            from .config import ALERT_COOLDOWN_SEC as GLOBAL_CD

            cooldown = (
                cd_override
                if isinstance(cd_override, (int, float)) and cd_override > 0
                else GLOBAL_CD
            )
            if time.time() - last_alert_at.get(key, 0.0) < cooldown:
                continue

            # 评分 & MODE 门槛
            score_now = float(payload.get("score", 0.0)) + LEVEL_SCORE_BOOST.get(
                strategy.name, 0.0
            )
            kind_now = str(payload.get("kind", ""))

            if score_now < MIN_SIGNAL_SCORE:
                continue
            if MODE == "QUIET":
                if "explode" in kind_now and score_now < (
                    MIN_SIGNAL_SCORE + EXPLODE_QUIET_EXTRA_SCORE
                ):
                    continue
            else:
                if ONLY_PUSH_EXPLODE and ("explode" not in kind_now):
                    continue

            # 标记来源策略 / 等级标签
            payload["_strategy_name"] = strategy.name
            payload["_timeframe_fast"] = strategy.timeframe_fast
            payload["_timeframe_htf"] = strategy.timeframe_htf
            payload["_level_label"] = LEVEL_LABEL.get(strategy.name, "")
            payload["_score_routed"] = score_now  # 用于排序
            collected.append(payload)

        except Exception as e:
            print(f"[ERROR] detect {sym} @ {strategy.name}: {e}")

    return collected


def _format_batches_for_strategy(strategy_name: str, items: List[dict]) -> List[str]:
    """
    将某策略的 payload 列表，转成多段消息文本（每段 ≤ PER_MESSAGE_LIMIT）。
    """
    if not items:
        return []

    # 排序：先按“策略内”优先级，再按加权评分
    priority = {
        "ema_rebound_long": 0,
        "ema_rebound_short": 0,
        "pullback_long": 1,
        "pullback_short": 1,
        "bb_squeeze_long": 2,
        "bb_squeeze_short": 2,
        "ema_stack_bull": 2,
        "ema_stack_bear": 2,
        "trend_break_up": 2,
        "trend_break_down": 2,
        "equilibrium_break": 2,
        "equilibrium_reject": 2,
        "volume_shift_long": 3,
        "volume_shift_short": 3,
        "cap_long": 3,
        "cap_short": 3,
        "climax_bottom": 3,
        "climax_top": 3,
        "rsi_div_long": 4,
        "rsi_div_short": 4,
        "explode_up": 5,
        "explode_down": 5,
    }
    items.sort(key=lambda x: x.get("_score_routed", 0.0), reverse=True)
    items.sort(key=lambda p: priority.get(p["kind"], 9))

    # 分批
    chunks = [
        items[i : i + PER_MESSAGE_LIMIT]
        for i in range(0, len(items), PER_MESSAGE_LIMIT)
    ]
    segments = []
    for idx, group in enumerate(chunks, 1):
        level = LEVEL_LABEL.get(strategy_name, "")
        hdr = f"— {level} {strategy_name}（{idx}/{len(chunks)}）—"
        lines = [
            hdr,
            "本批："
            + "，".join(f"{p['symbol']}·{p.get('kind_cn', p['kind'])}" for p in group),
        ]
        for p in group:
            label = p.get("_level_label", "")
            lines.append(f"{label} " + format_signal_cn(p))
        segments.append("\n".join(lines))
    return segments


def _send_segments_as_one_message(segments: List[str]) -> None:
    """
    将若干段文本拼成一条消息发送（用于整点合并）。
    """
    if not segments:
        return
    text = (
        f"{TITLE_PREFIX}｜合并批｜{ts_now()}\n"
        + ("\n" + SEPARATOR_LINE + "\n").join(segments)
        + f"\n{SEPARATOR_LINE}"
    )
    res = telegram_send(text)
    if res:
        chat_id, msg_id = res
        schedule_delete(
            chat_id,
            msg_id,
            int(time.time()) + AUTO_DELETE_HOURS * 3600 + AUTO_DELETE_GRACE,
        )


def run_fused_loop():
    """
    单进程调度：
    - 启动立即跑一次 M15
    - 每个 15m 边界：跑 M15（单独推送）
    - 每个 1h 边界：同时跑 M15 和 H1，并“合并成一条消息”
    - 候选列表按策略分别维护；刷新由 4h 边界或超时触发
    """
    ex = build_exchange()

    # 候选 & 强弱映射
    def _refresh_for(strategy: Strategy):
        cands, up_map, dn_map = hourly_refresh_candidates(ex, strategy)
        return cands, up_map, dn_map

    m15_candidates, m15_up, m15_dn = _refresh_for(M15)
    h1_candidates, h1_up, h1_dn = _refresh_for(H1_4H)

    last_alert_at: Dict[Tuple[str, str, str], float] = {}
    last_candidates_refresh_ts = int(time.time())

    last_ts = int(time.time())
    first_run_done = False

    while True:
        loop_start = time.time()
        try:
            now_ts = int(time.time())

            # 候选刷新（超时或 4h 边界）
            if (
                now_ts - last_candidates_refresh_ts >= CANDIDATE_REFRESH_SEC
            ) or crossed_boundary(last_ts, now_ts, FRAME_SEC["4h"]):
                m15_candidates, m15_up, m15_dn = _refresh_for(M15)
                h1_candidates, h1_up, h1_dn = _refresh_for(H1_4H)

                res = telegram_send(
                    f"🧭 <b>候选列表已刷新</b>\n"
                    f"M15数量：<b>{len(m15_candidates)}</b>\n"
                    f"H1 数量：<b>{len(h1_candidates)}</b>\n"
                    f"周期：<b>{CANDIDATE_REFRESH_SEC // 3600} 小时</b>"
                )
                if res:
                    chat_id, msg_id = res
                    schedule_delete(
                        chat_id,
                        msg_id,
                        int(time.time()) + AUTO_DELETE_HOURS * 3600 + AUTO_DELETE_GRACE,
                    )
                last_candidates_refresh_ts = now_ts

            # 边界判定
            do_m15 = crossed_boundary(last_ts, now_ts, FRAME_SEC["15m"])
            do_h1 = crossed_boundary(last_ts, now_ts, FRAME_SEC["1h"])

            # 启动立即跑一次 M15
            if not first_run_done:
                do_m15 = True

            if not (do_m15 or do_h1):
                elapsed = time.time() - loop_start
                dbg(f"[FUSED] Idle tick ({elapsed:.2f}s)")
                cleanup_pending_deletes(int(time.time()))
                time.sleep(POLL_SEC)
                last_ts = now_ts
                continue

            # —— 整点特殊：合并推送 —— #
            if do_h1:
                # 收集两套
                m15_payloads = _collect_for_strategy(
                    ex, M15, m15_candidates, m15_up, m15_dn, last_alert_at
                )
                h1_payloads = _collect_for_strategy(
                    ex, H1_4H, h1_candidates, h1_up, h1_dn, last_alert_at
                )

                # 生成分段文本
                segs = []
                segs += _format_batches_for_strategy(M15.name, m15_payloads)
                segs += _format_batches_for_strategy(H1_4H.name, h1_payloads)

                if segs:
                    _send_segments_as_one_message(segs)
                    # 更新冷却时间
                    now = time.time()
                    for p in m15_payloads:
                        last_alert_at[(M15.name, p["symbol"], p["kind"])] = now
                    for p in h1_payloads:
                        last_alert_at[(H1_4H.name, p["symbol"], p["kind"])] = now

            # —— 仅 15m 边界（非整点）：单独推送 —— #
            elif do_m15:
                m15_payloads = _collect_for_strategy(
                    ex, M15, m15_candidates, m15_up, m15_dn, last_alert_at
                )
                segs = _format_batches_for_strategy(M15.name, m15_payloads)
                if segs:
                    # 每个 15m 批次单独发
                    for seg in segs:
                        res = telegram_send(
                            f"{TITLE_PREFIX}｜{ts_now()}\n{seg}\n{SEPARATOR_LINE}"
                        )
                        if res:
                            chat_id, msg_id = res
                            schedule_delete(
                                chat_id,
                                msg_id,
                                int(time.time())
                                + AUTO_DELETE_HOURS * 3600
                                + AUTO_DELETE_GRACE,
                            )
                    # 更新冷却
                    now = time.time()
                    for p in m15_payloads:
                        last_alert_at[(M15.name, p["symbol"], p["kind"])] = now

            first_run_done = True

        except Exception as e:
            print("[FUSED LOOP ERROR]", e)

        # 维护
        elapsed = time.time() - loop_start
        dbg(f"[FUSED] Loop {elapsed:.2f}s")
        cleanup_pending_deletes(int(time.time()))
        time.sleep(POLL_SEC)
        last_ts = int(time.time())


if __name__ == "__main__":
    run_fused_loop()
