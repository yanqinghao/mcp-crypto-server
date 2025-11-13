# -*- coding: utf-8 -*-
"""
multi_loop.py (FUSED 合并发送版 · 统一候选集 + SR邻近汇总 · 精简5列)
- 同一候选集上分别跑 M15 与 H1_4H；双周期共识/仅单周期/相反 → 合并按 symbol 只发一次
- 末尾附带 “SR邻近汇总”（仅临近；5列：Symbol/现价/SR价/Dist%/VolR；Symbol 截 /USDT 前）

变更点：
- SR 邻近判定不依赖方向：只要 price<r 判“临近突破”，price>s 判“临近跌破”
- 邻近阈值默认 1.2%（可调）；汇总全空时打印轻量调试日志
"""

import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .loggingx import dbg, ts_now
from .config import (
    SLEEP_MS,
    POLL_SEC,
    PER_MESSAGE_LIMIT,
    MESSAGE_DELAY_SEC,
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

# ===== 视觉与发送控制 =====
BAR_HEAVY = "═══════════════════════════════════════"
BAR_LIGHT = "───────────────────────────────────────"
BAR_DOTTED = "⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯"
BULLET = "·"
MAX_MSGS_PER_STRATEGY = 3
TELEGRAM_MAX_CHARS = 3900

# ===== 预跑提前量（秒） =====
PRE_RUN_LEAD_SEC_15M_NORMAL = 180
PRE_RUN_LEAD_SEC_15M_ON_HOURLY = 300
PRE_RUN_LEAD_SEC_4H = 300

# ===== 周期/等级 & 徽标 =====
LEVEL_LABEL = {M15.name: "[L1]", H1_4H.name: "[L2]"}
LEVEL_SCORE_BOOST = {M15.name: 0.00, H1_4H.name: 0.05}
SIDE_TF_LABEL = {M15.name: "短线", H1_4H.name: "长线"}

CROSS_CONFIRM_ENABLED = True
CROSS_CONFIRM_SCORE_BOOST = 0.12
CONFIRM_BADGE = "✅双周期共识"
SINGLE_BADGE = "⚠️仅单周期"
OPPOSE_BADGE = "⚠️双周期相反"

# ===== SR 邻近汇总参数 =====
SUMMARY_MAX_ROWS = 30
NEAR_TH_RESIST_PCT = 1.2  # 最近阻力 距离≤1.2% 视为“临近突破”
NEAR_TH_SUPPOR_PCT = 1.2  # 最近支撑 距离≤1.2% 视为“临近跌破”

# ===== 其他（保留给非SR逻辑使用）=====
BREAKUP_KINDS_UP = {
    "equilibrium_break_up",
    "equilibrium_persist_up",
    "trend_break_up",
    "explode_up",
    "bb_squeeze_long",
    "ema_stack_bull",
    "ema_rebound_long",
    "pullback_long",
    "volume_shift_long",
    "rsi_div_long",
    "climax_bottom",
}
BREAKUP_KINDS_DN = {
    "equilibrium_break_down",
    "equilibrium_persist_down",
    "trend_break_down",
    "explode_down",
    "bb_squeeze_short",
    "ema_stack_bear",
    "ema_rebound_short",
    "pullback_short",
    "volume_shift_short",
    "rsi_div_short",
    "climax_top",
}


# ===== 数值与文本工具 =====
def _num(x, prec=2, compact=False):
    try:
        if x is None:
            return "—"
        if isinstance(x, (int, float)):
            if compact and abs(x) >= 1000:
                return f"${x:,.0f}"
            return f"{x:.{prec}f}"
        return str(x)
    except Exception:
        return str(x)


def _compact_money(x: Optional[float]) -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        if abs(v) >= 1_000_000:
            return f"${v / 1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"${v / 1_000:.0f}k"
        return f"${int(v):d}"
    except Exception:
        return "—"


def _tf_cn(strategy_name: str) -> str:
    return SIDE_TF_LABEL.get(strategy_name, strategy_name)


def _confidence_score(p: dict) -> float:
    base = float(p.get("_score_routed", p.get("score", 0.0)) or 0.0)
    return max(0.0, min(100.0, base / 1.6 * 100.0))


def _side_of(kind: str) -> str:
    k = (kind or "").lower()
    return "long" if any(s in k for s in ("_up", "bull", "long", "bottom")) else "short"


def _collect_side_lines(items: list) -> list:
    lines = []
    for p in items:
        tf = _tf_cn(p.get("_strategy_name", ""))
        k = p.get("kind_cn", p.get("kind", "—"))
        sc = _num(p.get("_score_routed", p.get("score", 0.0)), 2)
        vr = _num(p.get("volr_now"), 2)
        bu = _compact_money(p.get("eq_now_bar_usd"))
        lines.append(f"· {tf}: {k} (score {sc}, VolR {vr}, BarUSD {bu})")
    return lines


# ===== SR 邻近工具 =====
def _dist_pct(price: Optional[float], level: Optional[float]) -> Optional[float]:
    try:
        if price is None or level is None or level == 0:
            return None
        return abs(price - float(level)) / abs(float(level)) * 100.0
    except Exception:
        return None


def _gap_abs(price: Optional[float], level: Optional[float]) -> Optional[float]:
    try:
        if price is None or level is None:
            return None
        return float(level) - float(price)
    except Exception:
        return None


def _clean_sym(s: str) -> str:
    """提取 /USDT 前的 symbol（兼容 :USDT / 末尾 USDT）"""
    if not s:
        return "—"
    s = s.upper()
    if "/USDT" in s:
        return s.split("/USDT")[0]
    if ":USDT" in s:
        return s.split(":USDT")[0]
    if s.endswith("USDT"):
        return s[:-4]
    return s


# ===== SR 邻近阶段判定（与方向无关，仅临近）=====
def _stage_break_resistance(p: dict):
    """
    临近突破（与方向无关）：
      - r 存在 且 price < r
      - dist_pct(price, r) ≤ NEAR_TH_RESIST_PCT
      - 返回 (label, rank=2, r, dist_pct, gap_abs)
    """
    r = p.get("sr_near_resistance")
    price = p.get("last_price") or p.get("price_now")
    if not isinstance(r, (int, float)) or not isinstance(price, (int, float)):
        return None, 0, None, None, None
    if not (price < r):
        return None, 0, None, None, None
    dist = _dist_pct(price, r)
    if dist is not None and dist <= NEAR_TH_RESIST_PCT:
        return "临近突破", 2, r, dist, _gap_abs(price, r)
    return None, 0, None, None, None


def _stage_break_support(p: dict):
    """
    临近跌破（与方向无关）：
      - s 存在 且 price > s
      - dist_pct(price, s) ≤ NEAR_TH_SUPPOR_PCT
      - 返回 (label, rank=2, s, dist_pct, gap_abs)
    """
    s = p.get("sr_near_support")
    price = p.get("last_price") or p.get("price_now")
    if not isinstance(s, (int, float)) or not isinstance(price, (int, float)):
        return None, 0, None, None, None
    if not (price > s):
        return None, 0, None, None, None
    dist = _dist_pct(price, s)
    if dist is not None and dist <= NEAR_TH_SUPPOR_PCT:
        return "临近跌破", 2, s, dist, _gap_abs(price, s)
    return None, 0, None, None, None


# ===== SR 邻近汇总（5列：Symbol/现价/SR价/Dist%/VolR）=====
def _build_breakout_summary(fused_payloads: List[dict]) -> Optional[str]:
    """
    仅展示“临近突破 / 临近跌破”
    排序：距离%(升序) → VolR(降序)
    """
    rows_up, rows_dn = [], []
    for p in fused_payloads or []:
        st1, rk1, r1, d1, g1 = _stage_break_resistance(p)
        st2, rk2, s2, d2, g2 = _stage_break_support(p)
        price = p.get("last_price") or p.get("price_now")
        volr = p.get("volr_now")
        sym = _clean_sym(p.get("symbol"))

        if rk1 > 0:
            rows_up.append(
                {"sym": sym, "px": price, "sr": r1, "dist": d1, "volr": volr}
            )
        if rk2 > 0:
            rows_dn.append(
                {"sym": sym, "px": price, "sr": s2, "dist": d2, "volr": volr}
            )

    if not rows_up and not rows_dn:
        dbg("[SR SUMMARY] no near items (check thresholds or sr fields)")
        return None

    def _sort(rows):
        rows.sort(
            key=lambda r: (
                r["dist"] if r["dist"] is not None else 9e9,
                -(r["volr"] or 0.0),
            )
        )
        return rows

    rows_up = _sort(rows_up)
    rows_dn = _sort(rows_dn)
    total = len(rows_up) + len(rows_dn)
    if total > SUMMARY_MAX_ROWS:
        keep_up = max(5, int(SUMMARY_MAX_ROWS * (len(rows_up) / max(1, total))))
        keep_dn = SUMMARY_MAX_ROWS - keep_up
        rows_up, rows_dn = rows_up[:keep_up], rows_dn[:keep_dn]

    def _render(title, rows):
        if not rows:
            return ""
        th = f"{'Symbol':<10} {'现价':>10} {'SR价':>10} {'Dist%':>7} {'VolR':>6}"
        lines = [th]
        for r in rows:
            lines.append(
                f"{(r['sym'] or '—')[:10]:<10} "
                f"{_num(r['px'], 6):>10} "
                f"{_num(r['sr'], 6):>10} "
                f"{_num(r['dist'], 2):>7} "
                f"{_num(r['volr'], 2):>6}"
            )
        return f"<b>{title}</b>\n<pre><code>\n" + "\n".join(lines) + "\n</code></pre>"

    header = "📊 <b>[FUSED] SR邻近汇总</b>（按距离升序）"
    part_up = _render("↑ 临近突破（阻力）", rows_up)
    part_dn = _render("↓ 临近跌破（支撑）", rows_dn)
    return "\n".join(x for x in [header, part_up, part_dn] if x)


# ===== 时间槽工具 =====
def crossed_boundary(prev_ts: int, now_ts: int, frame_sec: int) -> bool:
    return (prev_ts // frame_sec) != (now_ts // frame_sec)


def next_boundary_ts(now_ts: int, frame_sec: int) -> int:
    return ((now_ts // frame_sec) + 1) * frame_sec


def approaching_boundary(now_ts: int, frame_sec: int, lead_sec: int) -> bool:
    remaining = next_boundary_ts(now_ts, frame_sec) - now_ts
    return 0 <= remaining <= max(1, int(lead_sec))


# ===== 统一发送封装 =====
def _send_text_with_delete(text: str):
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


# ===== 收集 / 互证 / 合并 / 格式化 =====
def _collect_for_strategy(
    ex,
    strategy: Strategy,
    candidates: List[str],
    strong_up_map: Dict[str, bool],
    strong_dn_map: Dict[str, bool],
    last_alert_at: Dict[Tuple[str, str, str], float],
) -> List[dict]:
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

            payload["_strategy_name"] = strategy.name
            payload["_timeframe_fast"] = strategy.timeframe_fast
            payload["_timeframe_htf"] = strategy.timeframe_htf
            payload["_level_label"] = LEVEL_LABEL.get(strategy.name, "")
            payload["_score_routed"] = score_now
            payload["_side"] = _side_of(kind_now)
            collected.append(payload)

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[ERROR] detect {sym} @ {strategy.name}: {e}")

    return collected


def _cross_confirm(m15_items: List[dict], h1_items: List[dict]) -> None:
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
            for p in group_a or group_b:
                p["_risk_tag"] = SINGLE_BADGE
            continue

        confirmed = False
        for pa in group_a:
            for pb in group_b:
                if pa["_side"] == pb["_side"]:
                    confirmed = True
                    pa["_confirm_tag"] = CONFIRM_BADGE
                    pb["_confirm_tag"] = CONFIRM_BADGE
                    pa["_score_routed"] = (
                        float(pa.get("_score_routed", 0.0)) + CROSS_CONFIRM_SCORE_BOOST
                    )
                    pb["_score_routed"] = (
                        float(pb.get("_score_routed", 0.0)) + CROSS_CONFIRM_SCORE_BOOST
                    )

        if not confirmed:
            for p in group_a + group_b:
                p["_risk_tag"] = OPPOSE_BADGE


def _merge_by_symbol(m15_items: List[dict], h1_items: List[dict]) -> List[dict]:
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
        lst.sort(key=lambda p: priority.get(p["kind"], 9))
        lst.sort(key=lambda p: p.get("_score_routed", 0.0), reverse=True)
        rep = dict(lst[0])

        rep["_sources"] = [p["_strategy_name"] for p in lst]
        rep["_merged_kinds"] = [(p["_strategy_name"], p["kind"]) for p in lst]
        rep["_merged"] = True
        rep["_sides_lines"] = _collect_side_lines(lst)

        has_confirm = any(p.get("_confirm_tag") for p in lst)
        if has_confirm:
            rep["_confirm_tag"] = CONFIRM_BADGE
            rep.pop("_risk_tag", None)
        else:
            badges = set(p.get("_risk_tag") for p in lst if p.get("_risk_tag"))
            if OPPOSE_BADGE in badges:
                rep["_risk_tag"] = OPPOSE_BADGE
            elif SINGLE_BADGE in badges:
                rep["_risk_tag"] = SINGLE_BADGE

        merged.append(rep)
    return merged


def _format_batches_for_strategy(strategy_name: str, items: List[dict]) -> List[str]:
    if not items:
        return []
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
    items.sort(key=lambda p: priority.get(p["kind"], 9))
    items.sort(key=lambda x: x.get("_score_routed", 0.0), reverse=True)

    chunks = [
        items[i : i + PER_MESSAGE_LIMIT]
        for i in range(0, len(items), PER_MESSAGE_LIMIT)
    ]
    segments = []
    for idx, group in enumerate(chunks, 1):
        hdr = f"{BAR_HEAVY}\n🧩 <b>[FUSED] 合并发送</b>  <i>(第 {idx}/{len(chunks)} 批)</i>\n{BAR_LIGHT}"

        overview_items = []
        show_max = 10
        for i, p in enumerate(group[:show_max], 1):
            tag1 = p.get("_confirm_tag", "")
            tag2 = p.get("_risk_tag", "")
            tag = (" " + tag1 if tag1 else "") + (" " + tag2 if tag2 else "")
            srcs = p.get("_sources") or []
            src_cn = [_tf_cn(s) for s in srcs]
            src_txt = f" 〔{', '.join(src_cn)}〕" if src_cn else ""
            overview_items.append(
                f"{i}. <b>{p['symbol']}</b> {BULLET} {p.get('kind_cn', p['kind'])}{tag}{src_txt}"
            )
        extra = len(group) - show_max
        if extra > 0:
            overview_items.append(f"… 以及 <b>+{extra}</b> 个信号")
        overview = "🗂️ <b>批次概览</b>\n" + "\n".join(overview_items)

        details_lines = ["📣 <b>详情</b>"]
        for p in group:
            tags = " ".join(t for t in (p.get("_confirm_tag"), p.get("_risk_tag")) if t)
            prefix = f"{tags} " if tags else ""
            details_lines.append(BAR_DOTTED)
            details_lines.append(prefix + format_signal_cn(p))
            side_lines = p.get("_sides_lines") or []
            if side_lines:
                details_lines.append("🔎 <b>周期明细</b>")
                details_lines.extend(side_lines)
        details = "\n".join(details_lines)
        segments.append("\n".join([hdr, overview, BAR_LIGHT, details, BAR_HEAVY]))
    return segments


def _send_segments_paginated(segments: List[str], title_prefix: str, max_msgs: int):
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


# ===== 主循环 =====
def run_fused_loop():
    ex = build_exchange()

    def _refresh_for(strategy: Strategy):
        cands, up_map, dn_map = hourly_refresh_candidates(ex, strategy)
        return cands, up_map, dn_map

    # m15 候选暂不启用
    m15_candidates, m15_up, m15_dn = [], {}, {}
    h1_candidates, h1_up, h1_dn = _refresh_for(H1_4H)

    last_alert_at: Dict[Tuple[str, str, str], float] = {}
    last_candidates_refresh_ts = int(time.time())

    last_ts = int(time.time())
    first_run_done = False
    last_15m_slot_id: Optional[int] = None
    last_4h_slot_id: Optional[int] = None

    while True:
        loop_start = time.time()
        try:
            now_ts = int(time.time())

            # ---- 4h 候选刷新 ----
            slot_4h_cur = now_ts // FRAME_SEC["4h"]
            slot_4h_next = slot_4h_cur + 1
            win_4h = approaching_boundary(now_ts, FRAME_SEC["4h"], PRE_RUN_LEAD_SEC_4H)

            do_refresh_4h = False
            target_4h_slot_id: Optional[int] = None

            if win_4h:
                target_4h_slot_id = slot_4h_next
            else:
                if crossed_boundary(last_ts, now_ts, FRAME_SEC["4h"]):
                    target_4h_slot_id = slot_4h_cur

            if target_4h_slot_id is not None and last_4h_slot_id != target_4h_slot_id:
                do_refresh_4h = True
                last_4h_slot_id = target_4h_slot_id

            if do_refresh_4h or (
                (now_ts - last_candidates_refresh_ts) >= CANDIDATE_REFRESH_SEC
            ):
                m15_candidates, m15_up, m15_dn = [], {}, {}
                h1_candidates, h1_up, h1_dn = _refresh_for(H1_4H)
                _send_text_with_delete(
                    f"🧭 <b>候选列表已刷新</b>\n"
                    f"M15数量：<b>{len(m15_candidates)}</b>\n"
                    f"H1 数量：<b>{len(h1_candidates)}</b>\n"
                    f"周期：<b>{CANDIDATE_REFRESH_SEC // 3600} 小时</b>"
                )
                last_candidates_refresh_ts = now_ts

            # ---- 15m 统一预跑 ----
            slot_15_cur = now_ts // FRAME_SEC["15m"]
            slot_15_next = slot_15_cur + 1
            ts_next_15 = next_boundary_ts(now_ts, FRAME_SEC["15m"])
            is_hour_slot = ts_next_15 % FRAME_SEC["1h"] == 0
            lead_15m = (
                PRE_RUN_LEAD_SEC_15M_ON_HOURLY
                if is_hour_slot
                else PRE_RUN_LEAD_SEC_15M_NORMAL
            )

            do_fused = False
            target_15m_slot_id: Optional[int] = None

            if approaching_boundary(now_ts, FRAME_SEC["15m"], lead_15m):
                target_15m_slot_id = slot_15_next
            elif not first_run_done:
                target_15m_slot_id = slot_15_cur
            elif crossed_boundary(last_ts, now_ts, FRAME_SEC["15m"]):
                target_15m_slot_id = slot_15_cur

            if (
                target_15m_slot_id is not None
                and last_15m_slot_id != target_15m_slot_id
            ):
                do_fused = True

            if not do_fused:
                elapsed = time.time() - loop_start
                dbg(f"[FUSED] Idle tick ({elapsed:.2f}s)")
                cleanup_pending_deletes(int(time.time()))
                time.sleep(POLL_SEC)
                last_ts = now_ts
                continue

            # ---- 执行阶段 ----
            unified_candidates = sorted((h1_candidates or []))
            dbg(
                f"[FUSED] unified candidates: {len(unified_candidates)} (m15={len(m15_candidates)}, h1={len(h1_candidates)})"
            )

            m15_payloads = _collect_for_strategy(
                ex, M15, unified_candidates, m15_up, m15_dn, last_alert_at
            )
            h1_payloads = _collect_for_strategy(
                ex, H1_4H, unified_candidates, h1_up, h1_dn, last_alert_at
            )

            _cross_confirm(m15_payloads or [], h1_payloads or [])
            fused_payloads = _merge_by_symbol(m15_payloads or [], h1_payloads or [])

            if fused_payloads:
                segs = _format_batches_for_strategy("FUSED", fused_payloads)
                if segs:
                    _send_segments_paginated(
                        segs, f"{TITLE_PREFIX}｜[FUSED] 合并发送", MAX_MSGS_PER_STRATEGY
                    )

                    summary_text = _build_breakout_summary(fused_payloads)
                    if summary_text:
                        _send_text_with_delete(summary_text)

                    now_mark = time.time()
                    for p in fused_payloads:
                        mk = p.get("_merged_kinds") or []
                        for src, kind in mk:
                            key = (src, p["symbol"], kind)
                            last_alert_at[key] = now_mark

            last_15m_slot_id = target_15m_slot_id
            first_run_done = True

        except Exception as e:
            print("[FUSED LOOP ERROR]", e)

        elapsed = time.time() - loop_start
        dbg(f"[FUSED] Loop {elapsed:.2f}s")
        cleanup_pending_deletes(int(time.time()))
        time.sleep(POLL_SEC)
        last_ts = int(time.time())


if __name__ == "__main__":
    run_fused_loop()
