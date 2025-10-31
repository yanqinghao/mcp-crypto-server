# -*- coding: utf-8 -*-
"""
detect.py  (Fused, patched)
—— Base 信号 + Pro 扩展信号，统一进入 ChaseGuard 多数表决
要点：
- 保留原 Base 五类：explode / pullback / capitulation / ema_rebound / bb_squeeze
- 增加 Pro 六类：trend_break_* / volume_shift_* / ema_stack_* / rsi_div_* / climax_* / equilibrium_*
- 新增 Pro 扩展：equilibrium_persist_up/down（平衡区突破后 N 根内不反包）
- Guard 只看“信号本身与环境”票据：不再用 RR 作为票据
- 新票据：最低量能 VolR、HTF 趋势错位、动量豁免收紧(仅当 volr>=2.2 & ADX>=18)、扁平无量、squeeze 带宽过宽
- MODE=QUIET → 使用 HARD 档；否则 MEDIUM 档；可通过 strategy.overrides 覆盖细项
- （本版补丁）
  · 修复 equilibrium_persist_* NaN 判断 & 放宽反包定义
  · 结构类票数 3→2，ema_stack_* 候选前置硬条件 + 35min 冷却
  · RSI 背离放宽容差，HTF 不再直接禁（交由 Guard）
  · climax_* 轻度放宽
  · Guard: VolR 阈值按 ADX 自适应；ema_stack_* 的 flat & no volume 不计票；
    对回踩/反弹类放宽“接近日内极值”的票据
  · CLASS_ADJ：按 SYMBOL_CLASS 自适应（MEGA/GOLD 放宽量能等）
"""

import time
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import talib as ta

from .strategies import Strategy
from .loggingx import dbg
from .config import (
    MODE,
    TIMEFRAME_HOURLY,
    LOOKBACK_VOL,
    BREAKOUT_WINDOW,
    BASELINE_BARS,
    ADX_MIN_TREND,
    TREND_LOOKBACK_BARS,
    TREND_MIN_SPEARMAN,
    TREND_MIN_NET_UP,
    TREND_MIN_NET_DN,
    PRICE_DN_TH,
    NO_FOLLOW_UP_TH,
    REQUIRE_TREND_ALIGN,
    PB_RSI_TH,
    PB_WR_TH,
    PB_MIN_BOUNCE_PCT,
    CAP_VOLR,
    CAP_WICK_RATIO,
    CAP_ALLOW_BOUNCE,
    BO_REQUIRE_HHV,
    BO_WINDOW,
    ENABLE_PULLBACK,
    ENABLE_CAPITULATION,
    ENABLE_EMA_REBOUND,
    ENABLE_BB_SQUEEZE,
    SCALE_ABS_PCT,
    EXPLODE_MAX_BAR_AGE_FRAC,
    EXPLODE_MIN_DIST_DAYEXT_PCT,
    EXPLODE_REQUIRE_CONTRACTION,
    SAFE_MODE_ALWAYS,
    SAFE_SL_PCT,
    SAFE_TP_PCTS,
    SAFE_ATR_MULT,
    SAFE_FIB_RATIO,
    PRINT_FULL_REASONS,
    MAX_REASONS_IN_MSG,
    FRAME_SEC,
    MIN_QV5M_USD,
    KINDS_CN,
)
from .exchange import fetch_ohlcv_df, get_tick_size, round_to_tick
from .ta_utils import (
    trend_scores,
    last_rsi,
    last_willr,
    last_adx,
    last_atr_pct,
    percent_change,
    compute_atr,
)
from .liquidity import get_day_stats
from .candidates import resolve_params_for_symbol, SYMBOL_CLASS


# =========================
# 配置：平衡区突破自适应参数 & 持续窗口
# =========================
EQ_BREAK_BUF_BASE = 0.003  # 0.30% 基础越界缓冲
EQ_MIN_BODY_PCT_BASE = 0.003  # 0.30% 最小实体百分比（相对开盘）
EQ_ATR_REF = 2.0  # 参考 ATR%，用于自适应
EQ_PERSIST_N = 3  # 突破后 N 根内不反包，即触发 equilibrium_persist_*

# =========================
# 近爆发做空锁（与原版一致）
# =========================
LAST_EXPLODE_UP: Dict[str, int] = {}  # {symbol: ts}
EXPLODE_LOCK_MIN = 15 * 60
EXPLODE_LOCK_RET_PCT = 0.9

# =========================
# EMA 堆叠冷却
# =========================
LAST_EMA_STACK_TS: Dict[str, int] = {}


# =========================
# TP/SL 与入场价工具
# =========================
def _tps_from_entry(side: str, entry: float, tick: float):
    tps = []
    for pct in SAFE_TP_PCTS:
        if side == "long":
            tps.append(round_to_tick(entry * (1.0 + pct / 100.0), tick))
        else:
            tps.append(round_to_tick(entry * (1.0 - pct / 100.0), tick))
    return tps


def _sl_from_entry(side: str, entry: float, tick: float, atr_abs: float | None):
    if entry <= 0:
        return entry
    pct_sl_abs = entry * (SAFE_SL_PCT / 100.0)
    if atr_abs is not None and atr_abs > 0:
        atr_sl_abs = atr_abs * SAFE_ATR_MULT
        slip = max(pct_sl_abs, atr_sl_abs)
    else:
        slip = pct_sl_abs
    if side == "long":
        sl = max(1e-12, entry - slip)
    else:
        sl = entry + slip
    return round_to_tick(sl, tick)


def dynamic_targets(
    symbol: str, side: str, entry_price: float, df_closed: pd.DataFrame, tick: float
):
    atr = compute_atr(df_closed, period=14)
    tps = _tps_from_entry(side, entry_price, tick)
    sl = _sl_from_entry(side, entry_price, tick, atr)
    return (*tps, sl)


def conservative_entry(
    symbol: str,
    side: str,
    close_last: float,
    df_closed: pd.DataFrame,
    tick: float,
    hhv_prev: float | None,
    llv_prev: float | None,
) -> float:
    entry = close_last
    if side == "long":
        if isinstance(hhv_prev, (int, float)) and hhv_prev > 0 and close_last > 0:
            target = close_last - (close_last - hhv_prev) * SAFE_FIB_RATIO
            entry = min(close_last, target)
        else:
            entry = close_last * (1.0 - SAFE_FIB_RATIO * 0.15)
    else:
        if isinstance(llv_prev, (int, float)) and llv_prev > 0 and close_last > 0:
            target = close_last + (llv_prev - close_last) * SAFE_FIB_RATIO
            entry = max(close_last, target)
        else:
            entry = close_last * (1.0 + SAFE_FIB_RATIO * 0.15)
    return round_to_tick(entry, tick)


def tpsl_for_safe_entry(side: str, entry_safe: float, tick: float):
    tps = _tps_from_entry(side, entry_safe, tick)
    sl = _sl_from_entry(side, entry_safe, tick, atr_abs=None)
    return (*tps, sl)


# =========================
# 本地中文名（Pro 新增）
# =========================
LOCAL_KINDS_CN = {
    "trend_break_up": "趋势线突破 · 上破",
    "trend_break_down": "趋势线突破 · 下破",
    "volume_shift_long": "横盘放量 · 吸筹",
    "volume_shift_short": "横盘放量 · 出货",
    "ema_stack_bull": "EMA 多头排列 · 确认",
    "ema_stack_bear": "EMA 空头排列 · 确认",
    "rsi_div_long": "RSI 背离 · 多",
    "rsi_div_short": "RSI 背离 · 空",
    "climax_bottom": "量能峰值 · 见底",
    "climax_top": "量能峰值 · 见顶",
    "equilibrium_break_up": "平衡区上破",
    "equilibrium_break_down": "平衡区下破",
    "equilibrium_reject_up": "假上破 · 回收",
    "equilibrium_reject_down": "假下破 · 回收",
    "equilibrium_persist_up": "平衡区上破 · 持续",
    "equilibrium_persist_down": "平衡区下破 · 持续",
}


def _kind_cn(kind: str) -> str:
    return KINDS_CN.get(kind, LOCAL_KINDS_CN.get(kind, kind))


# =========================
# Guard：分级 + 多数表决（无 RR 票据）
# =========================
_GUARD_STATS = defaultdict(int)

STRICT_DEFAULTS = {
    "NO_CHASE_DAYEXT_GAP_PCT": 1.0,  # 接近日内高/低的最小安全距离%
    "NO_CHASE_EMA20_PREMIUM_PCT": 2.0,  # EMA20 溢价（价偏离）
    "CLOSE_CONFIRM_FRAC": 0.60,  # 强势类需收盘确认比例
    "MAX_BODY_FOR_LONG": 2.5,
    "MAX_BODY_FOR_SHORT": 2.5,
    "SKIP_LONG_NEAR_DHIGH_FOR": [
        "explode_up",
        "ema_rebound_long",
        "bb_squeeze_long",
        "pullback_long",
        "ema_stack_bull",
        "trend_break_up",
        "volume_shift_long",
        "equilibrium_break_up",
        "equilibrium_persist_up",
    ],
    "SKIP_SHORT_NEAR_DLOW_FOR": [
        "explode_down",
        "ema_rebound_short",
        "bb_squeeze_short",
        "pullback_short",
        "ema_stack_bear",
        "trend_break_down",
        "volume_shift_short",
        "equilibrium_break_down",
        "equilibrium_persist_down",
    ],
    "RSI_CAP_LONG": 68,
    "RSI_CAP_SHORT": 32,
}

GUARD_LEVELS = {
    "OFF": dict(
        need_close_frac=0.0,
        max_body=99.0,
        gap_pct=0.0,
        ema20_prem=99.0,
        min_volr=0.0,
        votes_need=99,
    ),
    "SOFT": dict(
        need_close_frac=0.0,
        max_body=3.5,
        gap_pct=0.6,
        ema20_prem=2.2,
        min_volr=1.3,
        votes_need=2,
    ),
    "MEDIUM": dict(
        need_close_frac=0.0,
        max_body=2.5,
        gap_pct=1.0,
        ema20_prem=2.0,
        min_volr=1.4,
        votes_need=2,
    ),
    "HARD": dict(
        need_close_frac=0.0,
        max_body=2.0,
        gap_pct=1.2,
        ema20_prem=2.0,
        min_volr=1.6,
        votes_need=2,
    ),
}

# 某些信号需要更高/更低 VolR 底线（种类修正）
MIN_VOLR_BY_KIND = {
    # 收紧：确认类需要更扎实量能
    "ema_stack_bull": 1.8,
    "ema_stack_bear": 1.8,
    # 保持原值
    "volume_shift_long": 1.6,
    "volume_shift_short": 1.6,
    "bb_squeeze_long": 1.6,
    "bb_squeeze_short": 1.6,
    # 放宽：结构类允许稍低量能（以免两票秒拦）
    "equilibrium_break_up": 1.3,
    "equilibrium_break_down": 1.3,
    "equilibrium_reject_up": 1.3,
    "equilibrium_reject_down": 1.3,
    "equilibrium_persist_up": 1.3,
    "equilibrium_persist_down": 1.3,
    "trend_break_up": 1.4,
    "trend_break_down": 1.4,
}

# 提高部分“易泛滥信号”的票数门槛（结构类 3 -> 2）
VOTES_NEED_BY_KIND = {
    "volume_shift_long": 2,
    "volume_shift_short": 2,
    "bb_squeeze_long": 2,
    "bb_squeeze_short": 2,
    "equilibrium_break_up": 2,
    "equilibrium_break_down": 2,
    "equilibrium_reject_up": 2,
    "equilibrium_reject_down": 2,
    "equilibrium_persist_up": 2,
    "equilibrium_persist_down": 2,
    "trend_break_up": 2,
    "trend_break_down": 2,
}

# === Symbol-class adjustments (optional) ===
CLASS_ADJ = {
    "MEGA": dict(volr_scale=0.85, gap_add=+0.2, need_close_add=0),  # BTC/ETH/DOGE
    "GOLD": dict(volr_scale=0.80, gap_add=+0.3, need_close_add=0),  # PAXG/金锚品
    "ALTS": dict(volr_scale=1.00, gap_add=0.0, need_close_add=0.00),
    "MICRO": dict(volr_scale=1.10, gap_add=-0.2, need_close_add=0),
}


def _eval_guard(
    kind: str,
    side: str,  # "long"/"short"
    *,
    symbol: str,  # <— 新增：用于类目自适应
    elapsed: int,
    BAR_SEC: int,
    body_pct: float,
    rsi: float | None,
    ema20_val: float | None,
    c: float,
    day_high: float | None,
    day_low: float | None,
    level: str = "MEDIUM",
    shadow: bool = False,
    # 新增用于票据的上下文
    volr_now: float = 1.0,
    adx_val: float = 0.0,
    htf_bull: bool = False,
    htf_bear: bool = False,
    squeeze_width: float | None = None,
):
    """
    Guard 内部做“类目自适应”而不污染全局 GUARD_LEVELS。
    """
    # 基础档位参数
    P = dict(GUARD_LEVELS[level])

    # 类目自适应
    klass = SYMBOL_CLASS.get(symbol, "ALTS")
    adj = CLASS_ADJ.get(klass, CLASS_ADJ["ALTS"])
    P["min_volr"] = P["min_volr"] * adj["volr_scale"]
    P["gap_pct"] = P["gap_pct"] + adj["gap_add"]
    P["need_close_frac"] = float(
        np.clip(P["need_close_frac"] + adj.get("need_close_add", 0.0), 0.0, 0.80)
    )

    votes = 0
    reasons: List[str] = []

    # 1) 收盘确认（强波动/追价风险类）
    need_close = kind in (
        "explode_up",
        "explode_down",
        "ema_rebound_long",
        "ema_rebound_short",
        "bb_squeeze_long",
        "bb_squeeze_short",
        "pullback_long",
        "pullback_short",
        "ema_stack_bull",
        "ema_stack_bear",
        "trend_break_up",
        "trend_break_down",
        "volume_shift_long",
        "volume_shift_short",
        "equilibrium_break_up",
        "equilibrium_break_down",
        "equilibrium_persist_up",
        "equilibrium_persist_down",
        "equilibrium_reject_up",
        "equilibrium_reject_down",
    )
    if need_close and elapsed < int(BAR_SEC * P["need_close_frac"]):
        votes += 1
        reasons.append(f"close<{int(P['need_close_frac'] * 100)}%")

    # 2) 实体过大拒绝（避免追大阳/大阴）
    if body_pct > P["max_body"]:
        votes += 1
        reasons.append(f"body>{P['max_body']}%")

    # 3) 接近日内极值（不追高/不抄底）——对回踩/反弹类放宽
    NEAR_CAP_FOR_LONG = STRICT_DEFAULTS["SKIP_LONG_NEAR_DHIGH_FOR"]
    NEAR_CAP_FOR_SHORT = STRICT_DEFAULTS["SKIP_SHORT_NEAR_DLOW_FOR"]
    relax_for = {
        "pullback_long",
        "pullback_short",
        "ema_rebound_long",
        "ema_rebound_short",
    }
    if side == "long":
        if (
            (day_high and c > 0)
            and (kind in NEAR_CAP_FOR_LONG)
            and (kind not in relax_for)
        ):
            gap = (day_high - c) / c * 100.0
            if gap < P["gap_pct"]:
                votes += 1
                reasons.append(f"nearDHigh<{P['gap_pct']}%")
    else:
        if (
            (day_low and c > 0)
            and (kind in NEAR_CAP_FOR_SHORT)
            and (kind not in relax_for)
        ):
            gap = (c - day_low) / c * 100.0
            if gap < P["gap_pct"]:
                votes += 1
                reasons.append(f"nearDLow<{P['gap_pct']}%")

    # 4) EMA20 溢价限制 + 动量豁免收紧
    prem = None
    if ema20_val is not None and not pd.isna(ema20_val) and ema20_val > 0:
        prem = (c / ema20_val - 1.0) * 100.0
        momo_exempt = volr_now >= 2.2 and adx_val >= 18.0  # 仅强动量才豁免

        def _ema_soft_vote_allowed(kind_: str) -> bool:
            return (
                kind_.startswith("ema_stack_")
                or kind_.startswith("bb_squeeze_")
                or kind_.startswith("ema_rebound_")
                or kind_.startswith("volume_shift_")
            )

        if not momo_exempt:
            if side == "long" and prem > P["ema20_prem"]:
                votes += 1
                reasons.append(f"ema20+{prem:.2f}%>{P['ema20_prem']}%")
            elif side == "short" and (-prem) > P["ema20_prem"]:
                votes += 1
                reasons.append(f"ema20-{(-prem):.2f}%>{P['ema20_prem']}%")
            else:
                if _ema_soft_vote_allowed(kind) and abs(prem) >= 0.4:
                    votes += 1
                    reasons.append(f"ema20±{abs(prem):.2f}%≤{P['ema20_prem']:.1f}%")
        else:
            reasons.append(f"ema20±{abs(prem or 0):.2f}% (momo_exempt)")

    # 5) 量能底线（档位 + 种类修正 + ADX 自适应）
    base_min = max(P["min_volr"], MIN_VOLR_BY_KIND.get(kind, 0.0))
    adj_adx = np.clip((18.0 - adx_val) * 0.02, -0.3, 0.4)  # ADX 高→门槛降，低→升
    min_volr_need = max(1.2, base_min + adj_adx)
    if volr_now < min_volr_need:
        votes += 1
        reasons.append(f"VolR<{min_volr_need:.1f}x")

    # 6) 趋势闸门错位（与 HTF 相反向）——reject 不施加
    if not kind.startswith("cap_") and "equilibrium_reject" not in kind:
        if side == "long" and htf_bear:
            votes += 1
            reasons.append("trend-align offset")
        if side == "short" and htf_bull:
            votes += 1
            reasons.append("trend-align offset")

    # 7) 扁平/无量（结构类不施加；ema_stack_* 不计票，仅提示）
    if "equilibrium" not in kind and "trend_break" not in kind:
        if c > 0:
            if volr_now < (P["min_volr"] + 0.1) and body_pct < 0.15:
                if kind.startswith("ema_stack_"):
                    reasons.append("flat & no volume")
                else:
                    votes += 1
                    reasons.append("flat & no volume")

    # 8) squeeze 带宽过宽
    if "bb_squeeze" in kind and (squeeze_width is not None):
        if squeeze_width > 0.018:  # 1.8%
            votes += 1
            reasons.append("squeeze range too wide")

    # 9) 收紧 ema_stack（ADX/RSI）
    if kind in ("ema_stack_bull", "ema_stack_bear"):
        if adx_val < 20.0:
            votes += 1
            reasons.append("weak-trend ADX<20")
        if rsi is not None and not pd.isna(rsi):
            if kind == "ema_stack_bull" and rsi >= 68:
                votes += 1
                reasons.append("RSI>68 (overheat)")
            if kind == "ema_stack_bear" and rsi <= 32:
                votes += 1
                reasons.append("RSI<32 (oversold)")

    need_votes = max(P["votes_need"], VOTES_NEED_BY_KIND.get(kind, P["votes_need"]))
    block = votes >= need_votes
    if shadow:
        block = False

    _GUARD_STATS["total_seen"] += 1
    for r in reasons:
        _GUARD_STATS["reason_" + r] += 1
    if block:
        _GUARD_STATS["blocked_" + level.lower()] += 1

    return block, reasons, votes, need_votes


# =========================
# 主函数：detect_signal（融合 Base+Pro，统一 Guard）
# =========================
def detect_signal(
    ex,
    symbol: str,
    strong_up_map: dict,
    strong_dn_map: dict,
    strategy: Strategy,
):
    """
    统一检测：产生 Base+Pro 的所有候选 kind，计算 score & reasons，
    进入 Guard 多数表决后，按优先级+评分选 1 个最佳 payload 返回。
    """
    TIMEFRAME_FAST = strategy.timeframe_fast
    BAR_SEC = FRAME_SEC[TIMEFRAME_FAST]

    # 策略可覆盖量能阈值与评分标尺
    MIN_EQBAR = (
        strategy.overrides.get("MIN_QV5M_USD", MIN_QV5M_USD)
        if strategy.overrides
        else MIN_QV5M_USD
    )
    W_VOLR_NOW = strategy.w_volr_now
    W_EQ_NOW_USD = strategy.w_eq_now_usd
    W_ABS_PCT = strategy.w_abs_pct
    W_TREND_ALIGN = strategy.w_trend_align
    SCALE_EQ5M_USD = strategy.scale_eq_bar_usd
    PB_LOOKBACK_HI = strategy.pb_lookback_hi

    # 数据
    df = fetch_ohlcv_df(
        ex,
        symbol,
        strategy.timeframe_fast,
        limit=max(LOOKBACK_VOL, BREAKOUT_WINDOW, BASELINE_BARS, 320) + 8,
    )
    if len(df) < (BASELINE_BARS + 6):
        return False, None

    df_closed = df.iloc[:-1].copy()
    cur = df.iloc[-1]
    if len(df_closed) < (BASELINE_BARS + 3):
        return False, None

    o = float(cur["open"])
    h = float(cur["high"])
    l = float(cur["low"])
    c = float(cur["close"])
    vol = float(cur["volume"])
    cur_open_ts = int(cur["ts"]) // 1000
    now_ts = int(time.time())
    elapsed = max(1, min(BAR_SEC, now_ts - cur_open_ts))  # 本bar已进行秒数
    vps_now = vol / elapsed

    recent_closed = df_closed.tail(BASELINE_BARS).copy()
    vps_hist = (recent_closed["volume"].astype(float) / BAR_SEC).tolist()
    vps_base = float(sum(vps_hist) / len(vps_hist)) if len(vps_hist) else 0.0
    if vps_base <= 0:
        vps_base = float(df_closed.iloc[-1]["volume"]) / BAR_SEC

    volr_now = vps_now / max(1e-12, vps_base)
    eq_base_bar_usd = vps_base * BAR_SEC * c
    eq_now_bar_usd = vps_now * BAR_SEC * c

    # 最低等效额
    if eq_now_bar_usd < MIN_EQBAR:
        return False, None

    # 趋势与波动
    tr = trend_scores(df_closed, bars=TREND_LOOKBACK_BARS)
    adx = last_adx(df_closed, period=14)
    atrp = last_atr_pct(df_closed, period=14)

    trending_up = (
        (adx >= ADX_MIN_TREND)
        and (tr["spearman"] >= TREND_MIN_SPEARMAN)
        and (tr["net_pct"] >= TREND_MIN_NET_UP)
    )
    trending_dn = (
        (adx >= ADX_MIN_TREND)
        and (tr["spearman"] <= -TREND_MIN_SPEARMAN)
        and (-tr["net_pct"] >= TREND_MIN_NET_DN)
    )
    is_trending = trending_up or trending_dn

    loose_trending_up = (tr["spearman"] >= (TREND_MIN_SPEARMAN - 0.05)) and (
        tr["net_pct"] >= (TREND_MIN_NET_UP - 0.05)
    )
    loose_trending_dn = (tr["spearman"] <= -(TREND_MIN_SPEARMAN - 0.05)) and (
        -tr["net_pct"] >= (TREND_MIN_NET_DN - 0.05)
    )

    rsi = last_rsi(df_closed, period=14)
    wr = last_willr(df_closed, period=14)

    hhv_prev = df_closed["high"].iloc[:-1].tail(BO_WINDOW).max()
    llv_prev = df_closed["low"].iloc[:-1].tail(BO_WINDOW).min()

    # EMA 15m + 1h
    close_15m = df_closed["close"].astype(float).values
    ema8_15 = (
        ta.EMA(close_15m, timeperiod=8)
        if len(close_15m) >= 8
        else np.array([np.nan, np.nan])
    )
    ema20_15 = ta.EMA(close_15m, timeperiod=20)[-1] if len(close_15m) >= 20 else np.nan
    ema50_15 = ta.EMA(close_15m, timeperiod=50)[-1] if len(close_15m) >= 50 else np.nan

    try:
        df_1h = fetch_ohlcv_df(ex, symbol, TIMEFRAME_HOURLY, limit=120)
        close_1h = df_1h["close"].astype(float).values
        ema20_1h = (
            ta.EMA(close_1h, timeperiod=20)[-1] if len(close_1h) >= 20 else np.nan
        )
        ema50_1h = (
            ta.EMA(close_1h, timeperiod=50)[-1] if len(close_1h) >= 50 else np.nan
        )
    except Exception:
        ema20_1h = ema50_1h = np.nan

    HTF_BULL = (
        pd.notna(ema20_15)
        and pd.notna(ema50_15)
        and pd.notna(ema20_1h)
        and pd.notna(ema50_1h)
        and ema20_15 >= ema50_15
        and ema20_1h >= ema50_1h
        and c >= ema20_15
        and c >= ema20_1h
    )
    HTF_BEAR = (
        pd.notna(ema20_15)
        and pd.notna(ema50_15)
        and pd.notna(ema20_1h)
        and pd.notna(ema50_1h)
        and ema20_15 <= ema50_15
        and ema20_1h <= ema50_1h
        and c <= ema20_15
        and c <= ema20_1h
    )

    # ===== Base: explode / pullback / cap / ema_rebound / bb_squeeze =====
    pct_now = percent_change(c, o)

    Pm = resolve_params_for_symbol(symbol)
    EXP_VOLR = Pm["EXPLODE_VOLR"]
    UP_TH = Pm["PRICE_UP_TH"]
    PB_HI_PCT = Pm["PB_LOOKBACK_HI_PCT"]
    MIN_EQBAR = Pm["MIN_QV5M_USD"]  # 名称沿用

    # explode
    explode_up = (volr_now >= EXP_VOLR) and (pct_now >= UP_TH)
    explode_down = (volr_now >= EXP_VOLR) and (
        pct_now <= min(NO_FOLLOW_UP_TH, PRICE_DN_TH) or pct_now <= PRICE_DN_TH
    )
    if REQUIRE_TREND_ALIGN:
        if explode_up and not trending_up:
            explode_up = False
        if explode_down and not trending_dn:
            explode_down = False

    # 结构确认
    STRUCT_TOL_PCT = 0.001
    STRUCT_TOL_TICKS = 2
    tick_tmp = get_tick_size(ex, symbol)
    struct_tol = max(c * STRUCT_TOL_PCT, STRUCT_TOL_TICKS * tick_tmp)
    if BO_REQUIRE_HHV:
        if explode_up:
            explode_up = explode_up and (c >= max(hhv_prev - struct_tol, o))
        if explode_down:
            explode_down = explode_down and (c <= min(llv_prev + struct_tol, o))

    # QUIET 下 explode 额外门槛
    if MODE == "QUIET" and (explode_up or explode_down):
        max_age = int(BAR_SEC * EXPLODE_MAX_BAR_AGE_FRAC)
        if elapsed > max_age:
            explode_up = explode_down = False
        try:
            _dh, _dl, _pct24, _last_price = get_day_stats(ex, symbol, tick_tmp)
        except Exception:
            _dh, _dl, _pct24, _last_price = None, None, None, None
        if explode_up and (isinstance(_dh, (int, float)) and c > 0):
            if (_dh - c) / c * 100.0 < EXPLODE_MIN_DIST_DAYEXT_PCT:
                explode_up = False
        if explode_down and (isinstance(_dl, (int, float)) and c > 0):
            if (c - _dl) / c * 100.0 < EXPLODE_MIN_DIST_DAYEXT_PCT:
                explode_down = False
        if EXPLODE_REQUIRE_CONTRACTION and (explode_up or explode_down):
            _u, _m, _l = ta.BBANDS(
                close_15m, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            if pd.notna(_u[-1]) and pd.notna(_l[-1]) and pd.notna(_m[-1]):
                _width_series = (_u - _l) / np.maximum(_m, 1e-12)
                _recent_min = float(pd.Series(_width_series[-12:]).min())
                if not (_recent_min <= 0.015):
                    explode_up = explode_down = False

    # 自适应
    PB_MIN_BOUNCE_PCT_dyn = max(
        0.03, min(0.08, PB_MIN_BOUNCE_PCT * (0.7 if adx >= ADX_MIN_TREND else 0.85))
    )
    PB_HI_PCT_dyn = max(0.6, min(2.5, PB_HI_PCT * (0.75 if atrp <= 2.0 else 0.95)))
    CAPV_dyn = max(2.2, CAP_VOLR * (0.85 if not is_trending else 1.0))
    CAPW_dyn = max(0.40, min(0.75, CAP_WICK_RATIO * (0.9 if not is_trending else 1.0)))

    # pullback
    def recent_high_distance_pct(df_closed_, lookback: int, price_now: float) -> float:
        if len(df_closed_) < lookback or price_now <= 0:
            return 0.0
        hhv = float(df_closed_["high"].tail(lookback).max())
        return max(0.0, (hhv - price_now) / price_now * 100.0)

    def recent_low_distance_pct(df_closed_, lookback: int, price_now: float) -> float:
        if len(df_closed_) < lookback or price_now <= 0:
            return 0.0
        llv = float(df_closed_["low"].tail(lookback).min())
        return max(0.0, (price_now - llv) / price_now * 100.0)

    pullback_long = pullback_short = False
    if ENABLE_PULLBACK:
        if loose_trending_up and not explode_up:
            dist_hi = recent_high_distance_pct(df_closed, PB_LOOKBACK_HI, c)
            pullback_long = (
                (rsi <= PB_RSI_TH or wr <= PB_WR_TH)
                and (dist_hi >= PB_HI_PCT_dyn)
                and (pct_now >= PB_MIN_BOUNCE_PCT_dyn)
            )
        if loose_trending_dn and not explode_down:
            dist_lo = recent_low_distance_pct(df_closed, PB_LOOKBACK_HI, c)
            pullback_short = (
                (rsi >= (100 - PB_RSI_TH) or wr >= -100 - PB_WR_TH)
                and (dist_lo >= PB_HI_PCT_dyn)
                and (pct_now <= -PB_MIN_BOUNCE_PCT_dyn)
            )

    # capitulation
    def lower_wick_ratio(o_, h_, l_, c_):
        total = max(h_ - l_, 1e-12)
        lower = max(min(o_, c_) - l_, 0.0)
        return lower / total

    def upper_wick_ratio(o_, h_, l_, c_):
        total = max(h_ - l_, 1e-12)
        upper = max(h_ - max(o_, c_), 0.0)
        return upper / total

    cap_long = cap_short = False
    weak_trend = adx < ADX_MIN_TREND + 2
    if ENABLE_CAPITULATION and (not is_trending or weak_trend):
        lw = lower_wick_ratio(o, h, l, c)
        if (
            (volr_now >= max(2.0, CAPV_dyn * 0.95))
            and (lw >= max(0.33, CAPW_dyn * 0.9))
            and (((c - l) / max(h - l, 1e-12) >= 0.6) or (pct_now >= CAP_ALLOW_BOUNCE))
        ):
            cap_long = True
    if ENABLE_CAPITULATION and (not is_trending or weak_trend):
        uw = upper_wick_ratio(o, h, l, c)
        if (
            (volr_now >= max(2.0, CAPV_dyn * 0.95))
            and (uw >= max(0.33, CAPW_dyn * 0.9))
            and (((h - c) / max(h - l, 1e-12) >= 0.6) or (pct_now <= -CAP_ALLOW_BOUNCE))
        ):
            cap_short = True

    # ema_rebound（15m）
    ema8 = ema8_15
    ema20_series = (
        ta.EMA(close_15m, timeperiod=20)
        if len(close_15m) >= 20
        else np.array([np.nan, np.nan])
    )
    ema50_series = (
        ta.EMA(close_15m, timeperiod=50)
        if len(close_15m) >= 50
        else np.array([np.nan, np.nan])
    )

    ema_rebound_long = ema_rebound_short = False
    if (
        ENABLE_EMA_REBOUND
        and len(close_15m) >= 50
        and pd.notna(ema8[-2])
        and pd.notna(ema8[-1])
        and pd.notna(ema20_series[-1])
        and pd.notna(ema50_series[-1])
    ):
        up_trend_ok = ema20_series[-1] >= ema50_series[-1]
        dn_trend_ok = ema20_series[-1] <= ema50_series[-1]
        volr_soft = max(1.25, EXP_VOLR * 0.5)
        cross_up = (df_closed["close"].iloc[-1] <= float(ema8[-2]) * 1.001) and (
            c >= float(ema8[-1]) * 0.997
        )
        cross_down = (df_closed["close"].iloc[-1] >= float(ema8[-2]) * 0.999) and (
            c <= float(ema8[-1]) * 1.003
        )
        long_momo = rsi >= 48 or wr >= -55
        short_momo = rsi <= 52 or wr <= -45
        ema_rebound_long = (
            up_trend_ok and cross_up and (volr_now >= volr_soft) and long_momo
        )
        ema_rebound_short = (
            dn_trend_ok and cross_down and (volr_now >= volr_soft) and short_momo
        )

    # bb_squeeze
    bb_upper, bb_mid, bb_lower = (
        ta.BBANDS(close_15m, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        if len(close_15m) >= 20
        else (np.nan, np.nan, np.nan)
    )
    bb_width = (
        (bb_upper[-1] - bb_lower[-1]) / max(1e-12, bb_mid[-1])
        if (isinstance(bb_mid, np.ndarray) and pd.notna(bb_mid[-1]))
        else np.nan
    )
    squeeze = ENABLE_BB_SQUEEZE and (pd.notna(bb_width) and bb_width <= 0.015)
    bb_reversal_long = bb_reversal_short = False
    if squeeze:
        volr_bb = max(1.6, EXP_VOLR * 0.6)
        bb_reversal_long = (
            (c >= bb_upper[-1] * 0.998) and (volr_now >= volr_bb) and (rsi >= 48)
        )
        bb_reversal_short = (
            (c <= bb_lower[-1] * 1.002) and (volr_now >= volr_bb) and (rsi <= 52)
        )

    # HTF 闸门
    if HTF_BULL:
        explode_down = pullback_short = ema_rebound_short = bb_reversal_short = False
    if HTF_BEAR:
        explode_up = pullback_long = ema_rebound_long = bb_reversal_long = False

    def below(x, y, tol=0.005):
        return (x is not None and y is not None) and (x <= y * (1 + tol))

    if HTF_BULL:
        if explode_down:
            explode_down = (volr_now >= EXP_VOLR * 1.35) and (
                pct_now <= min(NO_FOLLOW_UP_TH - 0.06, PRICE_DN_TH - 0.05)
            )
        if pullback_short:
            pullback_short = below(c, ema20_15) and (
                pct_now <= -max(PB_MIN_BOUNCE_PCT_dyn * 1.6, 0.25)
            )
        if ema_rebound_short:
            ema8_last = float(ema8[-1]) if pd.notna(ema8[-1]) else None
            ema8_prev = float(ema8[-2]) if pd.notna(ema8[-2]) else None
            ema_rebound_short = below(c, ema20_15) and (
                ema8_prev is not None
                and ema8_last is not None
                and c <= ema8_last * 1.003
            )

    STRUCT_TOL_PCT_HTF = 0.0015
    struct_tol_htf = max(c * STRUCT_TOL_PCT_HTF, 2 * tick_tmp)

    def broke_15m_LL(x):
        return (llv_prev is not None) and (x <= (llv_prev + struct_tol_htf))

    if pullback_short:
        pullback_short = pullback_short and (below(c, ema20_15) or broke_15m_LL(c))
    if ema_rebound_short:
        ema_rebound_short = ema_rebound_short and (
            below(c, ema20_15) or broke_15m_LL(c)
        )
    if bb_reversal_short:
        bb_reversal_short = bb_reversal_short and (
            below(c, ema20_15) or broke_15m_LL(c)
        )

    # 爆发后做空锁
    lock_ok = True
    last_up_ts = LAST_EXPLODE_UP.get(symbol)
    if last_up_ts and (int(time.time()) - last_up_ts < EXPLODE_LOCK_MIN):
        lock_ok = pct_now <= -EXPLODE_LOCK_RET_PCT
    if not lock_ok:
        explode_down = pullback_short = ema_rebound_short = bb_reversal_short = False

    # ===== Pro: trend_break / volume_shift / ema_stack / rsi_div / climax / equilibrium =====
    pro_kinds: List[str] = []
    pro_reasons: List[str] = []

    def _poly_slope(arr):
        x = np.arange(len(arr))
        try:
            return float(np.polyfit(x, arr, 1)[0])
        except Exception:
            return 0.0

    # ema arrays (for pro)
    ema8_arr = (
        ta.EMA(close_15m, timeperiod=8)
        if len(close_15m) >= 8
        else np.array([np.nan, np.nan])
    )
    ema20_arr = (
        ta.EMA(close_15m, timeperiod=20)
        if len(close_15m) >= 20
        else np.array([np.nan, np.nan])
    )
    ema50_arr = (
        ta.EMA(close_15m, timeperiod=50)
        if len(close_15m) >= 50
        else np.array([np.nan, np.nan])
    )

    # 1) 趋势线突破
    TB_LOOKBACK = 10
    TB_UP_BUF = 0.002
    TB_DN_BUF = 0.002
    if len(df_closed) >= TB_LOOKBACK + 2:
        highs_ = df_closed["high"].tail(TB_LOOKBACK).values
        lows_ = df_closed["low"].tail(TB_LOOKBACK).values
        s_h = _poly_slope(highs_)
        s_l = _poly_slope(lows_)
        break_up_tb = (s_h < 0) and (c >= highs_[-1] * (1.0 + TB_UP_BUF))
        break_dn_tb = (s_l > 0) and (c <= lows_[-1] * (1.0 - TB_DN_BUF))
        if HTF_BULL:
            break_dn_tb = False
        if HTF_BEAR:
            break_up_tb = False
        if break_up_tb:
            pro_kinds.append("trend_break_up")
            pro_reasons.append(
                f"trend_break_up: slope(H)<0 & c≥H_last*(1+{TB_UP_BUF:.3f})"
            )
        if break_dn_tb:
            pro_kinds.append("trend_break_down")
            pro_reasons.append(
                f"trend_break_down: slope(L)>0 & c≤L_last*(1-{TB_DN_BUF:.3f})"
            )

    # 2) 横盘放量
    VS_VOLR = 2.2
    VS_PCT_ABS = 0.12
    body_pct_now = abs(c - o) / max(o, 1e-12) * 100.0
    rng_pct_now = (h - l) / max(o, 1e-12) * 100.0
    flat = (body_pct_now <= VS_PCT_ABS) and (rng_pct_now <= max(VS_PCT_ABS * 2.2, 0.35))
    if flat and volr_now >= VS_VOLR:
        if rsi >= 52 or (pd.notna(ema20_15) and c >= float(ema20_15)):
            if not HTF_BEAR:
                pro_kinds.append("volume_shift_long")
                pro_reasons.append(
                    f"volume_shift_long: flat({body_pct_now:.2f}%/{rng_pct_now:.2f}%) & volr {volr_now:.2f}≥{VS_VOLR:.1f}"
                )
        elif rsi <= 48 or (pd.notna(ema20_15) and c <= float(ema20_15)):
            if not HTF_BULL:
                pro_kinds.append("volume_shift_short")
                pro_reasons.append(
                    f"volume_shift_short: flat({body_pct_now:.2f}%/{rng_pct_now:.2f}%) & volr {volr_now:.2f}≥{VS_VOLR:.1f}"
                )

    # 3) EMA 多/空头排列（前置硬条件 + 冷却）
    EMA_STACK_ADX = 16
    EMA_GAP_MIN_PCT = 0.25  # EMA8-20 与 EMA20-50 至少 0.25%
    EMA20_SLOPE_WIN = 6  # 近 6 根 EMA20 斜率方向一致
    RNG_MIN_PCT = 0.60  # 当根振幅至少 0.60%（避免横摆假堆叠）
    EMA_STACK_COOLDOWN = 35 * 60  # 35 分钟冷却

    if (
        len(close_15m) >= 50
        and isinstance(ema8_arr, np.ndarray)
        and pd.notna(ema8_arr[-1])
        and isinstance(ema20_arr, np.ndarray)
        and pd.notna(ema20_arr[-1])
        and isinstance(ema50_arr, np.ndarray)
        and pd.notna(ema50_arr[-1])
    ):
        gap_8_20 = abs(ema8_arr[-1] - ema20_arr[-1]) / max(ema20_arr[-1], 1e-12) * 100.0
        gap_20_50 = (
            abs(ema20_arr[-1] - ema50_arr[-1]) / max(ema50_arr[-1], 1e-12) * 100.0
        )
        ema20_slope = _poly_slope(ema20_arr[-EMA20_SLOPE_WIN:])
        rng_pct_now2 = (h - l) / max(o, 1e-12) * 100.0

        pre_ok_bull = (
            (ema8_arr[-1] >= ema20_arr[-1] >= ema50_arr[-1])
            and (adx >= EMA_STACK_ADX)
            and (c >= ema20_arr[-1])
            and (gap_8_20 >= EMA_GAP_MIN_PCT)
            and (gap_20_50 >= EMA_GAP_MIN_PCT)
            and (ema20_slope > 0.0)
            and (rng_pct_now2 >= RNG_MIN_PCT)
        )
        pre_ok_bear = (
            (ema8_arr[-1] <= ema20_arr[-1] <= ema50_arr[-1])
            and (adx >= EMA_STACK_ADX)
            and (c <= ema20_arr[-1])
            and (gap_8_20 >= EMA_GAP_MIN_PCT)
            and (gap_20_50 >= EMA_GAP_MIN_PCT)
            and (ema20_slope < 0.0)
            and (rng_pct_now2 >= RNG_MIN_PCT)
        )

        nowi = int(time.time())
        last_ts = LAST_EMA_STACK_TS.get(symbol, 0)
        cooldown_ok = (nowi - last_ts) >= EMA_STACK_COOLDOWN

        if HTF_BULL:
            pre_ok_bear = False
        if HTF_BEAR:
            pre_ok_bull = False

        if pre_ok_bull and cooldown_ok:
            pro_kinds.append("ema_stack_bull")
            pro_reasons.append(
                f"ema_stack_bull: EMA8≥20≥50 & gaps≥{EMA_GAP_MIN_PCT:.2f}% & ADX {adx:.1f} & rng≥{RNG_MIN_PCT:.2f}%"
            )
            LAST_EMA_STACK_TS[symbol] = nowi
        if pre_ok_bear and cooldown_ok:
            pro_kinds.append("ema_stack_bear")
            pro_reasons.append(
                f"ema_stack_bear: EMA8≤20≤50 & gaps≥{EMA_GAP_MIN_PCT:.2f}% & ADX {adx:.1f} & rng≥{RNG_MIN_PCT:.2f}%"
            )
            LAST_EMA_STACK_TS[symbol] = nowi

    # 4) RSI 背离（放宽容差 & 不再被 HTF 直接禁掉，交给 Guard）
    DIV_LOOKBACK = 18
    DIV_TOL_PCT = 0.22  # from 0.15 -> 0.22
    if len(df_closed) >= DIV_LOOKBACK + 2:
        lows_ = df_closed["low"].tail(DIV_LOOKBACK + 1).values
        highs_ = df_closed["high"].tail(DIV_LOOKBACK + 1).values
        rsi_series = ta.RSI(df_closed["close"].astype(float).values, timeperiod=14)
        rsi_prev = (
            float(rsi_series[-(DIV_LOOKBACK + 1)])
            if pd.notna(rsi_series[-(DIV_LOOKBACK + 1)])
            else rsi
        )
        bull_div = (l <= np.min(lows_[:-1]) * (1 + DIV_TOL_PCT / 100.0)) and (
            rsi >= rsi_prev * (1 - DIV_TOL_PCT / 100.0)
        )
        bear_div = (h >= np.max(highs_[:-1]) * (1 - DIV_TOL_PCT / 100.0)) and (
            rsi <= rsi_prev * (1 + DIV_TOL_PCT / 100.0)
        )
        if bull_div:
            pro_kinds.append("rsi_div_long")
            pro_reasons.append(
                f"rsi_div_long: price LL vs {DIV_LOOKBACK} 但 RSI 未创新低"
            )
        if bear_div:
            pro_kinds.append("rsi_div_short")
            pro_reasons.append(
                f"rsi_div_short: price HH vs {DIV_LOOKBACK} 但 RSI 未创新高"
            )

    # 5) 量能峰值反转（轻度放宽）
    CLIMAX_VOLR = 4.2  # from 5.0
    CLIMAX_RSI_H = 70  # from 72
    CLIMAX_RSI_L = 30  # from 28
    total_rng = max(h - l, 1e-12)
    lower_w = max(min(o, c) - l, 0.0) / total_rng
    upper_w = max(h - max(o, c), 0.0) / total_rng
    if (
        volr_now >= CLIMAX_VOLR
        and lower_w >= 0.50  # from 0.55
        and rsi <= CLIMAX_RSI_L
        and not HTF_BEAR
    ):
        pro_kinds.append("climax_bottom")
        pro_reasons.append(
            f"climax_bottom: volr {volr_now:.2f}≥{CLIMAX_VOLR:.1f} & 下影 {lower_w:.2f} & RSI≤{CLIMAX_RSI_L}"
        )
    if (
        volr_now >= CLIMAX_VOLR
        and upper_w >= 0.50
        and rsi >= CLIMAX_RSI_H
        and not HTF_BULL
    ):
        pro_kinds.append("climax_top")
        pro_reasons.append(
            f"climax_top: volr {volr_now:.2f}≥{CLIMAX_VOLR:.1f} & 上影 {upper_w:.2f} & RSI≥{CLIMAX_RSI_H}"
        )

    # 6) 平衡区突破/回收 + 持续（不反包）确认
    EQ_WIN = 20
    # 自适应阈值（随 ATR% 缩放）
    atr_scale = max(0.7, min(1.3, atrp / EQ_ATR_REF))
    EQ_BREAK_BUF = max(0.002, min(0.005, EQ_BREAK_BUF_BASE * (1.0 / atr_scale)))
    EQ_MIN_BODY_PCT = max(
        0.0025, min(0.0040, EQ_MIN_BODY_PCT_BASE * (1.0 / atr_scale))
    )  # 0.25%~0.40%

    if len(df_closed) >= EQ_WIN + 2:
        eq_h = float(df_closed["high"].tail(EQ_WIN).max())
        eq_l = float(df_closed["low"].tail(EQ_WIN).min())
        body_pct_now_abs = abs(c - o) / max(o, 1e-12) * 100.0

        break_up = (c >= eq_h * (1.0 + EQ_BREAK_BUF)) and (
            body_pct_now_abs >= EQ_MIN_BODY_PCT * 100
        )
        break_dn = (c <= eq_l * (1.0 - EQ_BREAK_BUF)) and (
            body_pct_now_abs >= EQ_MIN_BODY_PCT * 100
        )

        # 假突破回收：实体部分≥一定比例
        EQ_REJ_BODY = 0.6
        reject_up = (
            (h > eq_h * (1.0 + EQ_BREAK_BUF))
            and (c <= eq_h)
            and (abs(c - o) / max(h - l, 1e-12) >= EQ_REJ_BODY)
        )
        reject_dn = (
            (l < eq_l * (1.0 - EQ_BREAK_BUF))
            and (c >= eq_l)
            and (abs(o - c) / max(h - l, 1e-12) >= EQ_REJ_BODY)
        )

        if HTF_BULL:
            break_dn = reject_up = False
        if HTF_BEAR:
            break_up = reject_dn = False

        if break_up:
            pro_kinds.append("equilibrium_break_up")
            pro_reasons.append(
                f"equilibrium_break_up: [{eq_l:.6g}, {eq_h:.6g}] buf {EQ_BREAK_BUF:.3f} & body≥{EQ_MIN_BODY_PCT * 100:.2f}%"
            )
        if break_dn:
            pro_kinds.append("equilibrium_break_down")
            pro_reasons.append(
                f"equilibrium_break_down: [{eq_l:.6g}, {eq_h:.6g}] buf {EQ_BREAK_BUF:.3f} & body≥{EQ_MIN_BODY_PCT * 100:.2f}%"
            )
        if reject_up:
            pro_kinds.append("equilibrium_reject_up")
            pro_reasons.append(
                f"equilibrium_reject_up: 假上破后实体≥{EQ_REJ_BODY:.2f} 回收入区间"
            )
        if reject_dn:
            pro_kinds.append("equilibrium_reject_down")
            pro_reasons.append(
                f"equilibrium_reject_down: 假下破后实体≥{EQ_REJ_BODY:.2f} 回收入区间"
            )

        # ===== 新增：突破后 N 根内不反包（持续） =====
        if len(df_closed) >= (EQ_WIN + EQ_PERSIST_N + 2):
            closes = df_closed["close"].astype(float).values
            opens = df_closed["open"].astype(float).values
            highs = df_closed["high"].astype(float).values
            lows = df_closed["low"].astype(float).values

            # 滚动窗口的区间上下界
            eq_h_series = pd.Series(highs).rolling(EQ_WIN).max().values
            eq_l_series = pd.Series(lows).rolling(EQ_WIN).min().values

            # 取最后 EQ_PERSIST_N+5 根做回顾，找“最近一次突破”的收盘位置 idx_b
            lookback = min(EQ_PERSIST_N + 5, len(df_closed) - 2)
            idx_end = len(df_closed) - 1
            idx_start = idx_end - lookback
            idx_b_up = None
            idx_b_dn = None

            for j in range(idx_end, idx_start - 1, -1):
                if j < EQ_WIN:
                    break
                body_pct_j = abs(closes[j] - opens[j]) / max(opens[j], 1e-12) * 100.0
                # 上破确认（修复 NaN 判断：用 pd.isna）
                if (
                    not pd.isna(eq_h_series[j])
                    and closes[j] >= eq_h_series[j] * (1.0 + EQ_BREAK_BUF)
                    and body_pct_j >= EQ_MIN_BODY_PCT * 100
                ):
                    idx_b_up = j
                    break
            for j in range(idx_end, idx_start - 1, -1):
                if j < EQ_WIN:
                    break
                body_pct_j = abs(closes[j] - opens[j]) / max(opens[j], 1e-12) * 100.0
                # 下破确认（修复 NaN 判断：用 pd.isna）
                if (
                    not pd.isna(eq_l_series[j])
                    and closes[j] <= eq_l_series[j] * (1.0 - EQ_BREAK_BUF)
                    and body_pct_j >= EQ_MIN_BODY_PCT * 100
                ):
                    idx_b_dn = j
                    break

            # 放宽“反包定义”：实体覆盖 ≥ 70% + 收盘越过中点
            def bearish_engulf_relaxed(prev_open, prev_close, o1, c1) -> bool:
                prev_low = min(prev_open, prev_close)
                prev_high = max(prev_open, prev_close)
                prev_mid = (prev_open + prev_close) / 2.0
                covered = max(0.0, (o1 - c1)) >= 0.7 * max(
                    1e-12, (prev_high - prev_low)
                )
                return (o1 > prev_high) and (c1 < prev_mid) and covered

            def bullish_engulf_relaxed(prev_open, prev_close, o1, c1) -> bool:
                prev_low = min(prev_open, prev_close)
                prev_high = max(prev_open, prev_close)
                prev_mid = (prev_open + prev_close) / 2.0
                covered = max(0.0, (c1 - o1)) >= 0.7 * max(
                    1e-12, (prev_high - prev_low)
                )
                return (o1 < prev_low) and (c1 > prev_mid) and covered

            # 检查上破后的 N 根
            if idx_b_up is not None and not HTF_BEAR:
                ok = True
                end_chk = min(idx_b_up + EQ_PERSIST_N, len(df_closed) - 1)
                for j in range(idx_b_up + 1, end_chk + 1):
                    if bearish_engulf_relaxed(
                        opens[idx_b_up], closes[idx_b_up], opens[j], closes[j]
                    ):
                        ok = False
                        break
                if ok:
                    pro_kinds.append("equilibrium_persist_up")
                    pro_reasons.append(
                        f"equilibrium_persist_up: 上破后 {EQ_PERSIST_N} 根内无反向吞没(放宽判定)"
                    )

            # 检查下破后的 N 根
            if idx_b_dn is not None and not HTF_BULL:
                ok = True
                end_chk = min(idx_b_dn + EQ_PERSIST_N, len(df_closed) - 1)
                for j in range(idx_b_dn + 1, end_chk + 1):
                    if bullish_engulf_relaxed(
                        opens[idx_b_dn], closes[idx_b_dn], opens[j], closes[j]
                    ):
                        ok = False
                        break
                if ok:
                    pro_kinds.append("equilibrium_persist_down")
                    pro_reasons.append(
                        f"equilibrium_persist_down: 下破后 {EQ_PERSIST_N} 根内无反向吞没(放宽判定)"
                    )

    # ===== 聚合：Base + Pro → 候选清单 =====
    candidates: List[Tuple[str, List[str]]] = []

    # Base
    if explode_up:
        candidates.append(("explode_up", []))
    if explode_down:
        candidates.append(("explode_down", []))
    if pullback_long:
        candidates.append(("pullback_long", []))
    if pullback_short:
        candidates.append(("pullback_short", []))
    if cap_long:
        candidates.append(("cap_long", []))
    if cap_short:
        candidates.append(("cap_short", []))
    if bb_reversal_long:
        candidates.append(("bb_squeeze_long", []))
    if bb_reversal_short:
        candidates.append(("bb_squeeze_short", []))
    if ema_rebound_long:
        candidates.append(("ema_rebound_long", []))
    if ema_rebound_short:
        candidates.append(("ema_rebound_short", []))

    # Pro（附带理由）
    for k in pro_kinds:
        candidates.append((k, pro_reasons))

    if not candidates:
        dbg(
            f"[FILTERED {symbol}] eq_bar=${eq_now_bar_usd:,.0f} volr={volr_now:.2f} pct={pct_now:.2f}% ADX={adx:.1f} no-signal"
        )
        return False, None

    # ===== 对每个候选 kind 计算评分 & Guard =====
    base_bonus = {
        "ema_rebound_long": 0.12,
        "ema_rebound_short": 0.12,
        "bb_squeeze_long": 0.10,
        "bb_squeeze_short": 0.10,
        "pullback_long": 0.10,
        "pullback_short": 0.10,
        "cap_long": 0.15,
        "cap_short": 0.15,
        "explode_up": 0.15,
        "explode_down": 0.15,
        "trend_break_up": 0.12,
        "trend_break_down": 0.12,
        "equilibrium_break_up": 0.12,
        "equilibrium_break_down": 0.12,
        "equilibrium_reject_up": 0.12,
        "equilibrium_reject_down": 0.12,
        "equilibrium_persist_up": 0.13,
        "equilibrium_persist_down": 0.13,
        "volume_shift_long": 0.10,
        "volume_shift_short": 0.10,
        "ema_stack_bull": 0.08,
        "ema_stack_bear": 0.08,
        "rsi_div_long": 0.12,
        "rsi_div_short": 0.12,
    }

    # 优先级：数值越小越靠前
    priority = {
        "trend_break_up": 0,
        "trend_break_down": 0,
        "equilibrium_break_up": 1,
        "equilibrium_break_down": 1,
        "equilibrium_reject_up": 1,
        "equilibrium_reject_down": 1,
        "equilibrium_persist_up": 1,
        "equilibrium_persist_down": 1,
        "ema_stack_bull": 2,
        "ema_stack_bear": 2,
        "ema_rebound_long": 2,
        "ema_rebound_short": 2,
        "bb_squeeze_long": 2,
        "bb_squeeze_short": 2,
        "pullback_long": 3,
        "pullback_short": 3,
        "volume_shift_long": 3,
        "volume_shift_short": 3,
        "cap_long": 4,
        "cap_short": 4,
        "rsi_div_long": 5,
        "rsi_div_short": 5,
        "explode_up": 6,
        "explode_down": 6,
    }

    # 日内信息
    tick = get_tick_size(ex, symbol)
    day_high, day_low, pct24, last_price = get_day_stats(ex, symbol, tick)
    dist_day_high_pct = (
        ((day_high - c) / c * 100.0)
        if (isinstance(day_high, (int, float)) and day_high and c > 0)
        else None
    )
    dist_day_low_pct = (
        ((c - day_low) / c * 100.0)
        if (isinstance(day_low, (int, float)) and day_low and c > 0)
        else None
    )

    # 统一 Guard 档位与影子模式
    level = "HARD" if MODE == "QUIET" else "MEDIUM"
    SHADOW = False  # 观察期使用 True；要拦截则改 False

    evaluated = []
    for kind, extra_reasons in candidates:
        side = (
            "long"
            if ("_up" in kind or "bull" in kind or "long" in kind or "bottom" in kind)
            else "short"
        )
        sc = (
            W_VOLR_NOW * max(0.0, min(1.0, (volr_now - 1.0) / 4.0))
            + W_EQ_NOW_USD
            * max(0.0, min(1.0, eq_now_bar_usd / max(1.0, SCALE_EQ5M_USD)))
            + W_ABS_PCT * max(0.0, min(1.0, abs(pct_now) / max(1e-9, SCALE_ABS_PCT)))
            + W_TREND_ALIGN
            * (
                1.0
                if ((HTF_BULL and side == "long") or (HTF_BEAR and side == "short"))
                else 0.0
            )
        ) + base_bonus.get(kind, 0.0)

        tr2 = trend_scores(df_closed, bars=TREND_LOOKBACK_BARS)
        adx2 = last_adx(df_closed, period=14)
        atrp2 = last_atr_pct(df_closed, period=14)
        trend_text = f"Class={SYMBOL_CLASS.get(symbol, '?')} | ADX≈{adx2:.1f}, ρ={tr2['spearman']:.2f}, net%={tr2['net_pct']:.2f}, ATR%≈{atrp2:.2f} | HTF:{'BULL' if HTF_BULL else ('BEAR' if HTF_BEAR else '—')}"

        reasons = [f"class={SYMBOL_CLASS.get(symbol, '?')}; mode={MODE}"]
        if HTF_BULL:
            reasons.append("HTF=15m/1h 多头闸门")
        if HTF_BEAR:
            reasons.append("HTF=15m/1h 空头闸门")
        reasons += extra_reasons
        # base 的简述
        if kind == "explode_up":
            reasons += [
                f"explode_up: volr {volr_now:.2f}≥{EXP_VOLR:.2f}",
                f"pct_now {pct_now:.2f}%≥{UP_TH:.2f}%",
                f"struct c≥HHV({BO_WINDOW})-tol",
            ]
        elif kind == "explode_down":
            reasons += [
                f"explode_down: volr {volr_now:.2f}≥{EXP_VOLR:.2f}",
                f"pct_now {pct_now:.2f}%≤{min(NO_FOLLOW_UP_TH, PRICE_DN_TH):.2f}%",
                f"struct c≤LLV({BO_WINDOW})+tol",
            ]
            last_up_ts = LAST_EXPLODE_UP.get(symbol)
            if last_up_ts and (int(time.time()) - last_up_ts < EXPLODE_LOCK_MIN):
                reasons.append(
                    f"post-explode lock checked: {int(time.time()) - last_up_ts}s"
                )

        entry_safe = conservative_entry(
            "NA",
            side,
            c,
            df_closed,
            tick,
            hhv_prev if side == "long" else None,
            llv_prev if side == "short" else None,
        )
        tp1_s, tp2_s, tp3_s, sl_s = tpsl_for_safe_entry(side, entry_safe, tick)
        tp1_i, tp2_i, tp3_i, sl_i = dynamic_targets("NA", side, c, df_closed, tick)

        cmd_immd = (
            f"/forcelong {symbol} 10 10 {tp1_i} {tp2_i} {tp3_i} {sl_i} {c:.6g}"
            if side == "long"
            else f"/forceshort {symbol} 10 10 {tp1_i} {tp2_i} {tp3_i} {sl_i} {c:.6g}"
        )
        cmd_safe = (
            f"/forcelong {symbol} 10 10 {tp1_s} {tp2_s} {tp3_s} {sl_s} {entry_safe}"
            if side == "long"
            else f"/forceshort {symbol} 10 10 {tp1_s} {tp2_s} {tp3_s} {sl_s} {entry_safe}"
        )

        block, reasons_guard, votes, need_votes = _eval_guard(
            kind,
            side,
            symbol=symbol,
            elapsed=elapsed,
            BAR_SEC=BAR_SEC,
            body_pct=body_pct_now,
            rsi=rsi,
            ema20_val=float(ema20_15) if pd.notna(ema20_15) else None,
            c=c,
            day_high=day_high,
            day_low=day_low,
            level=level,
            shadow=False,
            volr_now=volr_now,
            adx_val=adx,
            htf_bull=HTF_BULL,
            htf_bear=HTF_BEAR,
            squeeze_width=bb_width
            if "bb_squeeze" in kind and pd.notna(bb_width)
            else None,
        )
        status = "BLOCK" if block else "PASS"
        dbg(
            f"[GUARD {status}] {symbol} {kind} votes={votes}/{need_votes} reasons={reasons_guard}"
        )

        if block:
            continue

        reasons_show = reasons if PRINT_FULL_REASONS else reasons[:MAX_REASONS_IN_MSG]
        text_core = [
            f"Symbol: <b>{symbol}</b>",
            f"Price: <code>{c:.6g}</code>",
            f"Now {TIMEFRAME_FAST}: <b>{pct_now:.2f}%</b> | VolR: <b>{volr_now:.2f}x</b> | EqBar≈<b>${eq_now_bar_usd:,.0f}</b>",
            f"Base vps: <b>{vps_base:.4f}</b> | Trend: {trend_text} {'✅' if ((HTF_BULL and side == 'long') or (HTF_BEAR and side == 'short')) else '—'}",
            "Why: " + " ; ".join(reasons_show),
        ]
        if not SAFE_MODE_ALWAYS:
            text_core.append(f"<code>{cmd_immd}</code>")
        text_core += [
            f"<b>Conservative Entry:</b> <code>{entry_safe}</code> | SL(≈{SAFE_SL_PCT:.1f}%): <code>{tpsl_for_safe_entry(side, entry_safe, tick)[-1]}</code>",
            f"<code>{cmd_safe}</code>",
        ]

        evaluated.append(
            {
                "kind": kind,
                "side": side,
                "score": float(sc),
                "priority": priority.get(kind, 9),
                "text_core": text_core,
                "trend_text": trend_text,
                "cmd_immd": (None if SAFE_MODE_ALWAYS else text_core[-2]),
                "cmd_safe": text_core[-1],
            }
        )

        if kind == "explode_up":
            LAST_EXPLODE_UP[symbol] = int(time.time())

    if not evaluated:
        return False, None

    # 选择：优先级小者优先，再按分数高者
    evaluated.sort(key=lambda x: x["score"], reverse=True)
    evaluated.sort(key=lambda x: x["priority"])

    best = evaluated[0]
    kind = best["kind"]
    side = best["side"]
    sc = best["score"]
    text_core = best["text_core"]
    trend_text = best["trend_text"]

    # —— 结构化的 HTF 状态 —— #
    htf_gate = "BULL" if HTF_BULL else ("BEAR" if HTF_BEAR else "")

    payload = {
        "symbol": symbol,
        "kind": kind,
        "kind_cn": _kind_cn(kind),
        "title": _kind_cn(kind),
        "timeframe_fast": strategy.timeframe_fast,
        "pct_now": float(pct_now),
        "volr_now": float(volr_now),
        "vps_now": float(vps_now),
        "vps_base": float(vps_base),
        "eq_base_bar_usd": float(eq_base_bar_usd),
        "eq_now_bar_usd": float(eq_now_bar_usd),
        # —— HTF 相关（新增）——
        "htf_gate": htf_gate,  # "BULL" / "BEAR" / ""（formatter可直接用）
        "htf_bull": bool(HTF_BULL),  # 可选：布尔
        "htf_bear": bool(HTF_BEAR),  # 可选：布尔
        # 兼容旧模板：保留 trend_text（人读友好）
        "trend_text": trend_text,
        # —— 触发原因（新增：给 formatter 用）——
        # 注意：reasons_show 已在上文构建（受 PRINT_FULL_REASONS / MAX_REASONS_IN_MSG 控制）
        "reasons": reasons_show,
        "trend_align": bool(
            (HTF_BULL and side == "long") or (HTF_BEAR and side == "short")
        ),
        "text_core": text_core,  # 保持原有长文描述
        "score": float(sc),
        # 日内信息
        "day_high": day_high,
        "day_low": day_low,
        "pct24": pct24,
        "dist_day_high_pct": dist_day_high_pct,
        "dist_day_low_pct": dist_day_low_pct,
        "last_price": last_price or c,
        # 命令
        "cmd_safe": best["cmd_safe"],
    }
    if not SAFE_MODE_ALWAYS and best.get("cmd_immd") is not None:
        payload["cmd_immd"] = best["cmd_immd"]

    return True, payload
