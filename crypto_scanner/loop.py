# -*- coding: utf-8 -*-
"""
multi_loop.py (FUSED 合并发送版 · 统一候选集)
- 🔧 将 M15 与 H1_4H 的 candidates 先合并成一份候选集（去重），再对同一份集合分别跑两遍检测
- 每个 15 分钟槽位：同时跑 M15 与 H1_4H（在同一候选集上）→ 双边互证/单边/冲突 → 按 symbol 合并只发一次
- 统一分页为一块 [FUSED] 合并发送
- 预跑提前量：
    · 普通 15m 槽：提前 3 分钟
    · 整点槽（每小时的 00 分槽）：提前 5 分钟
    · 4h 候选刷新：提前 5 分钟（或边界/超时）
"""

import time
from typing import Dict, List, Tuple
from collections import defaultdict

from .loggingx import dbg, ts_now
from .config import (
    SLEEP_MS,
    POLL_SEC,
    PER_MESSAGE_LIMIT,  # 每段内最多几个 payload
    MESSAGE_DELAY_SEC,  # 条间延迟（秒）
    TITLE_PREFIX,
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
from .detect_pro import detect_signal
from .notifier import telegram_send, schedule_delete, cleanup_pending_deletes
from .formatter import format_signal_cn

# —— 等级标签 / 加权 —— #
LEVEL_LABEL = {M15.name: "[L1]", H1_4H.name: "[L2]"}
LEVEL_SCORE_BOOST = {M15.name: 0.00, H1_4H.name: 0.05}

# —— 交叉确认参数 —— #
CROSS_CONFIRM_ENABLED = True
CROSS_CONFIRM_SCORE_BOOST = 0.12  # 互证附加加分
CROSS_CONFIRM_BADGE = "✅CONFIRM"
RISK_BADGE_SINGLE = "⚠️RISK·单边"
RISK_BADGE_OPPOSE = "⚠️RISK·冲突"

# —— 发送控制：每批最多发几段（“一段=一条消息”） —— #
MAX_MSGS_PER_STRATEGY = 3  # 对 FUSED 也沿用这个上限

# —— Telegram 安全长度（双保险） —— #
TELEGRAM_MAX_CHARS = 3900

# —— 预跑提前量（秒） —— #
PRE_RUN_LEAD_SEC_15M_NORMAL = 180  # 普通 15m 槽提前 3 分钟
PRE_RUN_LEAD_SEC_15M_ON_HOURLY = 300  # 整点槽提前 5 分钟
PRE_RUN_LEAD_SEC_4H = 300  # 4h 刷新提前 5 分钟

# —— 视觉分隔 —— #
BAR_HEAVY = "═══════════════════════════════════════"
BAR_LIGHT = "───────────────────────────────────────"
BAR_DOTTED = "⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯"
BULLET = "·"


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


def _side_of(kind: str) -> str:
    k = (kind or "").lower()
    return "long" if any(s in k for s in ("_up", "bull", "long", "bottom")) else "short"


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
            payload["_side"] = _side_of(kind_now)
            collected.append(payload)

        except Exception as e:
            print(f"[ERROR] detect {sym} @ {strategy.name}: {e}")

    return collected


def _cross_confirm(m15_items: List[dict], h1_items: List[dict]) -> None:
    """
    原地修改两侧 payload，添加互证/风险标记与加分。
    规则：
      - 同 symbol 且 side 一致 => 互证；两侧都加分 + 标记
      - 只在一侧出现 => 单边风险
      - 两侧方向相反 => 冲突风险
    """
    if not CROSS_CONFIRM_ENABLED:
        return

    idx_m15 = defaultdict(list)
    for p in m15_items:
        idx_m15[p["symbol"]].append(p)

    idx_h1 = defaultdict(list)
    for p in h1_items:
        idx_h1[p["symbol"]].append(p)

    symbols = set(idx_m15.keys()) | set(idx_h1.keys())
    for s in symbols:
        group_a = idx_m15.get(s, [])
        group_b = idx_h1.get(s, [])
        if not group_a or not group_b:
            # 单边
            for p in group_a or group_b:
                p["_risk_tag"] = RISK_BADGE_SINGLE
            continue

        # 有相同方向则互证，否则冲突
        confirmed = False
        for pa in group_a:
            for pb in group_b:
                if pa["_side"] == pb["_side"]:
                    confirmed = True
                    pa["_confirm_tag"] = CROSS_CONFIRM_BADGE
                    pb["_confirm_tag"] = CROSS_CONFIRM_BADGE
                    pa["_score_routed"] = (
                        float(pa.get("_score_routed", 0.0)) + CROSS_CONFIRM_SCORE_BOOST
                    )
                    pb["_score_routed"] = (
                        float(pb.get("_score_routed", 0.0)) + CROSS_CONFIRM_SCORE_BOOST
                    )

        if not confirmed:
            for p in group_a + group_b:
                p["_risk_tag"] = RISK_BADGE_OPPOSE


def _merge_by_symbol(m15_items: List[dict], h1_items: List[dict]) -> List[dict]:
    """
    将两侧结果按 symbol 合并，每个 symbol 只保留一个代表 payload：
    - 先按“种类优先级”再按 _score_routed 选择代表
    - 代表继承：确认/风险徽标、来源策略列表、(strategy, kind) 列表
    - 若同 symbol 有“互证”与“风险”混杂，优先保留“互证”徽标
    """
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
        "equilibrium_break_up": 2,
        "equilibrium_break_down": 2,
        "equilibrium_reject_up": 2,
        "equilibrium_reject_down": 2,
        "equilibrium_persist_up": 2,
        "equilibrium_persist_down": 2,
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

    by_sym: Dict[str, List[dict]] = defaultdict(list)
    for x in m15_items or []:
        by_sym[x["symbol"]].append(x)
    for x in h1_items or []:
        by_sym[x["symbol"]].append(x)

    merged: List[dict] = []
    for sym, lst in by_sym.items():
        lst.sort(key=lambda p: p.get("_score_routed", 0.0), reverse=True)
        lst.sort(key=lambda p: priority.get(p["kind"], 9))
        rep = dict(lst[0])  # 浅拷贝代表

        rep["_sources"] = [p["_strategy_name"] for p in lst]
        rep["_merged_kinds"] = [(p["_strategy_name"], p["kind"]) for p in lst]
        rep["_merged"] = True

        # 徽标：互证优先于风险
        has_confirm = any(p.get("_confirm_tag") for p in lst)
        has_risk = any(p.get("_risk_tag") for p in lst)
        if has_confirm:
            rep["_confirm_tag"] = CROSS_CONFIRM_BADGE
            rep.pop("_risk_tag", None)
        elif has_risk:
            has_conflict = any(p.get("_risk_tag") == RISK_BADGE_OPPOSE for p in lst)
            rep["_risk_tag"] = RISK_BADGE_OPPOSE if has_conflict else RISK_BADGE_SINGLE

        merged.append(rep)

    return merged


def _format_batches_for_strategy(strategy_name: str, items: List[dict]) -> List[str]:
    """
    将 payload 列表转成多段消息文本（每段 ≤ PER_MESSAGE_LIMIT）。
    strategy_name 固定 "FUSED"。
    """
    if not items:
        return []

    # 优先级→评分排序
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
        "equilibrium_break_up": 2,
        "equilibrium_break_down": 2,
        "equilibrium_reject_up": 2,
        "equilibrium_reject_down": 2,
        "equilibrium_persist_up": 2,
        "equilibrium_persist_down": 2,
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

    # 分段
    chunks = [
        items[i : i + PER_MESSAGE_LIMIT]
        for i in range(0, len(items), PER_MESSAGE_LIMIT)
    ]

    segments = []
    for idx, group in enumerate(chunks, 1):
        hdr = (
            f"{BAR_HEAVY}\n"
            f"🧩 <b>[FUSED] 合并发送</b>  <i>(第 {idx}/{len(chunks)} 批)</i>\n"
            f"{BAR_LIGHT}"
        )

        # 概览（限 10 项）
        overview_items = []
        show_max = 10
        for i, p in enumerate(group[:show_max], 1):
            tag1 = p.get("_confirm_tag", "")
            tag2 = p.get("_risk_tag", "")
            tag = (" " + tag1 if tag1 else "") + (" " + tag2 if tag2 else "")
            srcs = p.get("_sources") or []
            src_txt = f" 〔{', '.join(srcs)}〕" if srcs else ""
            overview_items.append(
                f"{i}. <b>{p['symbol']}</b> {BULLET} {p.get('kind_cn', p['kind'])}{tag}{src_txt}"
            )
        extra = len(group) - show_max
        if extra > 0:
            overview_items.append(f"… 以及 <b>+{extra}</b> 个信号")

        overview = "🗂️ <b>批次概览</b>\n" + "\n".join(overview_items)

        # 详情
        details_lines = ["📣 <b>详情</b>"]
        for p in group:
            tags = " ".join(t for t in (p.get("_confirm_tag"), p.get("_risk_tag")) if t)
            prefix = f"{tags} " if tags else ""
            details_lines.append(BAR_DOTTED)
            details_lines.append(prefix + format_signal_cn(p))
        details = "\n".join(details_lines)

        segments.append("\n".join([hdr, overview, BAR_LIGHT, details, BAR_HEAVY]))

    return segments


def _send_segments_paginated(segments: List[str], title_prefix: str, max_msgs: int):
    """
    严格分页发送：一段=一条消息；条间 sleep；最多发送 max_msgs 条。
    """
    if not segments:
        return

    total = len(segments)
    to_send = segments[:max_msgs]
    for i, seg in enumerate(to_send, 1):
        title_line = f"{title_prefix}｜{ts_now()}｜{i}/{min(total, max_msgs)}"
        text = f"{title_line}\n{seg}"
        _send_text_with_delete(text)
        if i < len(to_send):
            time.sleep(max(0, MESSAGE_DELAY_SEC))


def run_fused_loop():
    """
    单进程调度（15m统一预跑 + 统一候选集 + 双边互证 + 合并发送）：
    - 每到 15m 槽位（预跑或边界后兜底）：
        🔧 获取/维护 M15/H1_4H 各自候选 → 合并去重成 unified_candidates →
        在 unified_candidates 上同时跑 M15 与 H1_4H → 互证 → 按 symbol 合并 → [FUSED] 分段发送
      · 普通 15m 槽提前 3 分钟预跑；整点槽提前 5 分钟
    - 4h 候选刷新：整点前 5 分钟预跑或边界/周期超时
    """
    ex = build_exchange()

    # 候选 & 强弱映射（分别维护，以便跑 detect 时传对应映射）
    def _refresh_for(strategy: Strategy):
        cands, up_map, dn_map = hourly_refresh_candidates(ex, strategy)
        return cands, up_map, dn_map

    # m15_candidates, m15_up, m15_dn = _refresh_for(M15)
    m15_candidates, m15_up, m15_dn = [], {}, {}
    h1_candidates, h1_up, h1_dn = _refresh_for(H1_4H)

    last_alert_at: Dict[Tuple[str, str, str], float] = {}
    last_candidates_refresh_ts = int(time.time())

    last_ts = int(time.time())
    first_run_done = False

    # 槽位去重：确保一个目标 15m 槽只跑一次
    last_15m_target_slot = None
    # 4h 槽去重
    last_slot_fired_4h = None

    while True:
        loop_start = time.time()
        try:
            now_ts = int(time.time())

            # 4h 预跑窗口（候选刷新）
            slot_4h_cur = now_ts // FRAME_SEC["4h"]
            slot_4h_next = slot_4h_cur + 1
            win_4h = approaching_boundary(now_ts, FRAME_SEC["4h"], PRE_RUN_LEAD_SEC_4H)

            do_refresh_4h = False
            if win_4h:
                if last_slot_fired_4h != ("next", slot_4h_next):
                    do_refresh_4h = True
                    last_slot_fired_4h = ("next", slot_4h_next)
            elif crossed_boundary(last_ts, now_ts, FRAME_SEC["4h"]):
                if last_slot_fired_4h != ("cur", slot_4h_cur):
                    do_refresh_4h = True
                    last_slot_fired_4h = ("cur", slot_4h_cur)

            if do_refresh_4h or (
                (now_ts - last_candidates_refresh_ts) >= CANDIDATE_REFRESH_SEC
            ):
                # m15_candidates, m15_up, m15_dn = _refresh_for(M15)
                m15_candidates, m15_up, m15_dn = [], {}, {}
                h1_candidates, h1_up, h1_dn = _refresh_for(H1_4H)

                _send_text_with_delete(
                    f"🧭 <b>候选列表已刷新</b>\n"
                    f"M15数量：<b>{len(m15_candidates)}</b>\n"
                    f"H1 数量：<b>{len(h1_candidates)}</b>\n"
                    f"周期：<b>{CANDIDATE_REFRESH_SEC // 3600} 小时</b>"
                )
                last_candidates_refresh_ts = now_ts

            # —— 15m 统一预跑 —— #
            slot_15_cur = now_ts // FRAME_SEC["15m"]
            slot_15_next = slot_15_cur + 1
            ts_next_15 = next_boundary_ts(now_ts, FRAME_SEC["15m"])
            is_hour_slot = (
                ts_next_15 % FRAME_SEC["1h"] == 0
            )  # 下一个 15m 边界是否整点槽
            lead_15m = (
                PRE_RUN_LEAD_SEC_15M_ON_HOURLY
                if is_hour_slot
                else PRE_RUN_LEAD_SEC_15M_NORMAL
            )

            do_fused = False
            target_slot = None
            if approaching_boundary(now_ts, FRAME_SEC["15m"], lead_15m):
                target_slot = ("next", slot_15_next)
                if last_15m_target_slot != target_slot:
                    do_fused = True
            elif not first_run_done and not approaching_boundary(
                now_ts, FRAME_SEC["15m"], lead_15m
            ):
                # 启动立即跑
                target_slot = ("cur", slot_15_cur)
                if last_15m_target_slot != target_slot:
                    do_fused = True
            elif crossed_boundary(last_ts, now_ts, FRAME_SEC["15m"]):
                target_slot = ("cur", slot_15_cur)
                if last_15m_target_slot != target_slot:
                    do_fused = True

            if not do_fused:
                elapsed = time.time() - loop_start
                dbg(f"[FUSED] Idle tick ({elapsed:.2f}s)")
                cleanup_pending_deletes(int(time.time()))
                time.sleep(POLL_SEC)
                last_ts = now_ts
                continue

            # ===== 执行阶段 =====
            # 🔧 合并候选集：M15+H1_4H（去重）
            unified_candidates = sorted((h1_candidates or []))
            dbg(
                f"[FUSED] unified candidates: {len(unified_candidates)} (m15={len(m15_candidates)}, h1={len(h1_candidates)})"
            )

            # 在统一候选集上分别跑两遍
            m15_payloads = _collect_for_strategy(
                ex, M15, unified_candidates, m15_up, m15_dn, last_alert_at
            )
            h1_payloads = _collect_for_strategy(
                ex, H1_4H, unified_candidates, h1_up, h1_dn, last_alert_at
            )

            # 交叉确认（双边/单边/冲突）
            _cross_confirm(m15_payloads or [], h1_payloads or [])

            # 合并为每 symbol 一条
            fused_payloads = _merge_by_symbol(m15_payloads or [], h1_payloads or [])

            # 发送（统一 FUSED 分页）
            if fused_payloads:
                segs = _format_batches_for_strategy("FUSED", fused_payloads)
                if segs:
                    _send_segments_paginated(
                        segs,
                        f"{TITLE_PREFIX}｜[FUSED] 合并发送",
                        MAX_MSGS_PER_STRATEGY,
                    )
                    now_mark = time.time()
                    # 代表推进两侧冷却（对每个 fused payload，将其 _merged_kinds 中的 (strategy, kind) 全部标记冷却）
                    for p in fused_payloads:
                        mk = p.get("_merged_kinds") or []
                        for src, kind in mk:
                            key = (src, p["symbol"], kind)
                            last_alert_at[key] = now_mark

            # 标记此目标槽位已跑
            last_15m_target_slot = target_slot
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
