# -*- coding: utf-8 -*-
"""
multi_loop.py
单进程调度 15m+1h 与 1h+4h 两套策略：
- 每个 15 分钟边界：跑 M15（分页：一段=一条消息）
- 每个整点边界：先跑 M15 再跑 H1，二者各自分页（每段=一条消息）
- 启动后立刻先跑一次（M15）
- 风控路由：M15 标记 [L1]，H1 标记 [L2] 且小幅加权（score_boost）
- 🔧 预跑提前量细分（按你要求）：
    · 15m 边界：提前 2 分钟预跑
    · 1h 边界（不刷新候选）：提前 3 分钟预跑
    · 1h 边界（与 4h 刷新重合/即将刷新）：提前 5 分钟预跑
    · 候选列表刷新（4h 边界或到期）：提前 5 分钟预跑
"""

import time
from typing import Dict, List, Tuple

from .loggingx import dbg, ts_now
from .config import (
    SLEEP_MS,
    POLL_SEC,
    PER_MESSAGE_LIMIT,  # 每段内最多几个 payload
    MESSAGE_DELAY_SEC,  # 条间延迟（秒）
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
from .detect_pro import detect_signal  # 若用基础版改为 .detect
from .notifier import telegram_send, schedule_delete, cleanup_pending_deletes
from .formatter import format_signal_cn

# —— 等级标签 / 加权 —— #
LEVEL_LABEL = {M15.name: "[L1]", H1_4H.name: "[L2]"}
LEVEL_SCORE_BOOST = {M15.name: 0.00, H1_4H.name: 0.05}

# —— 发送控制：每个策略每次最多发几条 —— #
MAX_MSGS_PER_STRATEGY = 3

# —— Telegram 安全长度（双保险，基本用不到，因为一段=一条） —— #
TELEGRAM_MAX_CHARS = 3900

# —— 边界预跑提前量（秒，细分） —— #
PRE_RUN_LEAD_SEC_15M = 120  # 15m 边界：提前 2 分钟（按你要求保留）
PRE_RUN_LEAD_SEC_1H_NR = 180  # 1h 边界（不刷新候选）：提前 3 分钟
PRE_RUN_LEAD_SEC_4H = 300  # 4h 刷新（含与整点重合时）：提前 5 分钟


def crossed_boundary(prev_ts: int, now_ts: int, frame_sec: int) -> bool:
    """兜底：若错过预跑窗口，边界后触发一次"""
    return (prev_ts // frame_sec) != (now_ts // frame_sec)


def next_boundary_ts(now_ts: int, frame_sec: int) -> int:
    """下一个周期边界的时间戳（秒）"""
    return ((now_ts // frame_sec) + 1) * frame_sec


def approaching_boundary(now_ts: int, frame_sec: int, lead_sec: int) -> bool:
    """
    是否进入“边界前 lead_sec 秒”的预跑窗口。
    仅判断是否接近；去重由 last_slot_fired 控制（预跑记 next_slot，跨槽记 current_slot）。
    """
    remaining = next_boundary_ts(now_ts, frame_sec) - now_ts
    return 0 <= remaining <= max(1, int(lead_sec))


def _send_text_with_delete(text: str):
    """统一发送 + 自动删除封装；超长截断双保险"""
    if len(text) > TELEGRAM_MAX_CHARS:
        text = text[: TELEGRAM_MAX_CHARS - 20] + "\n... (truncated)"
    res = telegram_send(text)
    if res:
        chat_id, msg_id = res
        schedule_delete(
            chat_id,
            msg_id,
            int(time.time()) + AUTO_DELETE_HOURS * 3600 + AUTO_DELETE_GRACE,
        )


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

            cd_override = (strategy.overrides or {}).get("ALERT_COOLDOWN_SEC")
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
    注意：后续发送逻辑保证“一段=一条消息”，不再拼接。
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

    # 每段 PER_MESSAGE_LIMIT 个 payload，后续“一段=一条消息”
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
            lines.append(SEPARATOR_LINE)
        segments.append("\n".join(lines))
    return segments


def _send_segments_paginated(segments: List[str], title_prefix: str, max_msgs: int):
    """
    严格分页发送：一段=一条消息；条间 sleep；最多发送 max_msgs 条，多余丢弃（可改为入队列）。
    """
    if not segments:
        return

    total = len(segments)
    to_send = segments[:max_msgs]
    for i, seg in enumerate(to_send, 1):
        # 每条都有独立标题 + 分页计数
        text = f"{title_prefix}｜{ts_now()}｜{i}/{min(total, max_msgs)}\n{seg}"
        _send_text_with_delete(text)
        if i < len(to_send):
            time.sleep(max(0, MESSAGE_DELAY_SEC))


def run_fused_loop():
    """
    单进程调度（含边界预跑）：
    - 启动立即跑一次 M15
    - 每个 15m 边界：M15 分页，一段=一条消息，最多 3 条（15m 边界前 2 分钟预跑）
    - 每个 1h 边界：先 M15 再 H1，各自分页，一段=一条消息，各自最多 3 条
      · 若本轮也会触发 4h 刷新：整点前 5 分钟预跑
      · 否则：整点前 3 分钟预跑
    - 候选列表按策略分别维护；刷新由 4h 边界（前 5 分钟预跑）或超时触发
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

    # —— 槽位去重：确保一个周期只触发一次预跑 —— #
    last_slot_fired = {
        "15m": None,  # 记录：最近一次“触发”的槽位编号（预跑=next 槽；兜底=current 槽）
        "1h": None,
        "4h": None,
    }
    # —— 额外：M15 实际执行目标槽位去重（避免 15m 预跑 + 1h 预跑重复跑同一 15m 槽） —— #
    last_m15_target_slot = None  # 仅用于“真正执行 M15 前”的二次去重

    while True:
        loop_start = time.time()
        try:
            now_ts = int(time.time())

            # 是否处于 4h 预跑窗口（供 1h 预跑提前量复用）
            win_4h = approaching_boundary(now_ts, FRAME_SEC["4h"], PRE_RUN_LEAD_SEC_4H)

            # 候选刷新（超时或「4h 边界前」预跑）
            do_refresh_4h = False
            slot_4h_cur = now_ts // FRAME_SEC["4h"]
            slot_4h_next = slot_4h_cur + 1
            if win_4h:
                if last_slot_fired["4h"] != slot_4h_next:
                    do_refresh_4h = True
                    last_slot_fired["4h"] = slot_4h_next
            elif crossed_boundary(last_ts, now_ts, FRAME_SEC["4h"]):
                if last_slot_fired["4h"] != slot_4h_cur:
                    do_refresh_4h = True
                    last_slot_fired["4h"] = slot_4h_cur

            if do_refresh_4h or (
                (now_ts - last_candidates_refresh_ts) >= CANDIDATE_REFRESH_SEC
            ):
                m15_candidates, m15_up, m15_dn = _refresh_for(M15)
                h1_candidates, h1_up, h1_dn = _refresh_for(H1_4H)

                _send_text_with_delete(
                    f"🧭 <b>候选列表已刷新</b>\n"
                    f"M15数量：<b>{len(m15_candidates)}</b>\n"
                    f"H1 数量：<b>{len(h1_candidates)}</b>\n"
                    f"周期：<b>{CANDIDATE_REFRESH_SEC // 3600} 小时</b>"
                )
                last_candidates_refresh_ts = now_ts

            # —— 边界判定（优先预跑窗口，其次兜底 crossed_boundary） —— #
            do_m15 = False
            do_h1 = False

            # 启动立即跑一次 M15
            if not first_run_done:
                do_m15 = True
            else:
                # 15m（固定提前 2 分钟；预跑=next_slot、兜底=current_slot）
                slot_15_cur = now_ts // FRAME_SEC["15m"]
                slot_15_next = slot_15_cur + 1
                if approaching_boundary(now_ts, FRAME_SEC["15m"], PRE_RUN_LEAD_SEC_15M):
                    if last_slot_fired["15m"] != slot_15_next:
                        do_m15 = True
                        last_slot_fired["15m"] = slot_15_next
                elif crossed_boundary(last_ts, now_ts, FRAME_SEC["15m"]):
                    if last_slot_fired["15m"] != slot_15_cur:
                        do_m15 = True
                        last_slot_fired["15m"] = slot_15_cur

                # 1h（根据是否 4h 预跑窗口选择 3 分钟或 5 分钟；预跑=next_slot、兜底=current_slot）
                slot_1h_cur = now_ts // FRAME_SEC["1h"]
                slot_1h_next = slot_1h_cur + 1
                lead_1h = PRE_RUN_LEAD_SEC_4H if win_4h else PRE_RUN_LEAD_SEC_1H_NR
                if approaching_boundary(now_ts, FRAME_SEC["1h"], lead_1h):
                    if last_slot_fired["1h"] != slot_1h_next:
                        do_h1 = True
                        last_slot_fired["1h"] = slot_1h_next
                elif crossed_boundary(last_ts, now_ts, FRAME_SEC["1h"]):
                    if last_slot_fired["1h"] != slot_1h_cur:
                        do_h1 = True
                        last_slot_fired["1h"] = slot_1h_cur

            # —— 没活儿，休眠 —— #
            if not (do_m15 or do_h1):
                elapsed = time.time() - loop_start
                dbg(f"[FUSED] Idle tick ({elapsed:.2f}s)")
                cleanup_pending_deletes(int(time.time()))
                time.sleep(POLL_SEC)
                last_ts = now_ts
                continue

            # ===== 执行阶段：对 M15 增加“目标槽位”二次去重，避免同一 15m 槽被跑两次 =====
            def _run_m15_if_needed(tag_from: str):
                nonlocal last_m15_target_slot
                # 确定此次 M15 的“目标槽位”（预跑=next，兜底=current）
                slot_cur = now_ts // FRAME_SEC["15m"]
                is_pre = approaching_boundary(
                    now_ts, FRAME_SEC["15m"], PRE_RUN_LEAD_SEC_15M
                )
                slot_target = (slot_cur + 1) if is_pre else slot_cur

                if last_m15_target_slot == slot_target:
                    dbg(
                        f"[{M15.name}] Skip duplicate run for 15m slot={slot_target} (from {tag_from})"
                    )
                    return None  # 表示跳过

                # 真正执行
                payloads = _collect_for_strategy(
                    ex, M15, m15_candidates, m15_up, m15_dn, last_alert_at
                )
                if payloads:
                    segs = _format_batches_for_strategy(M15.name, payloads)
                    if segs:
                        _send_segments_paginated(
                            segs,
                            f"{TITLE_PREFIX}｜{LEVEL_LABEL[M15.name]} {M15.name}",
                            MAX_MSGS_PER_STRATEGY,
                        )
                        now_mark = time.time()
                        for p in payloads:
                            last_alert_at[(M15.name, p["symbol"], p["kind"])] = now_mark
                # 标记本次已覆盖此目标槽位
                last_m15_target_slot = slot_target
                return payloads

            # —— 整点：先 M15 后 H1；各自分页（每段=一条），各自最多 3 条 —— #
            if do_h1:
                _run_m15_if_needed("hourly-branch")
                h1_payloads = _collect_for_strategy(
                    ex, H1_4H, h1_candidates, h1_up, h1_dn, last_alert_at
                )

                if h1_payloads:
                    h1_segs = _format_batches_for_strategy(H1_4H.name, h1_payloads)
                    if h1_segs:
                        _send_segments_paginated(
                            h1_segs,
                            f"{TITLE_PREFIX}｜{LEVEL_LABEL[H1_4H.name]} {H1_4H.name}",
                            MAX_MSGS_PER_STRATEGY,
                        )
                        now_mark = time.time()
                        for p in h1_payloads:
                            last_alert_at[(H1_4H.name, p["symbol"], p["kind"])] = (
                                now_mark
                            )

            # —— 非整点，仅 15m 边界：M15 分页 —— #
            elif do_m15:
                _run_m15_if_needed("15m-branch")

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
