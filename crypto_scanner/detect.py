# -*- coding: utf-8 -*-
"""
detect.py
基础 15m / 1h 检测逻辑（你现有的版本）+ ChaseGuard 防追价与严格确认层

改动要点：
- 保留你原有的 explode / pullback / capitulation / ema_rebound / bb_squeeze 全部规则与评分、payload 结构
- 追加 ChaseGuard：收盘进度门槛、实体过大、接近日内极值拒绝、EMA20 溢价限制、最小 R:R 底线，多数表决（默认≥2条理由拦截）
- MODE=QUIET 时自动走更严格的 HARD 档；普通模式 MEDIUM 档
- 保守入场/立即入场 TP/SL 与你原版一致
"""

import time
from typing import Dict
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
    EXPLODE_CONTRACTION_BB_WIDTH,
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
# 爆发后反向封控（与原版一致）
# =========================
LAST_EXPLODE_UP: Dict[str, int] = {}  # {symbol: ts}
EXPLODE_LOCK_MIN = 15 * 60  # 爆发后15分钟内不做空
EXPLODE_LOCK_RET_PCT = 0.9  # 做空前要求当根回撤 ≥ 0.9%


# =========================
# TP/SL 与入场价工具（与原版一致）
# =========================
def _tps_from_entry(side: str, entry: float, tick: float):
    """
    用 config 中的 SAFE_TP_PCTS 生成三个分批 TP，统一按照绝对百分比。
    """
    tps = []
    for pct in SAFE_TP_PCTS:
        if side == "long":
            tps.append(round_to_tick(entry * (1.0 + pct / 100.0), tick))
        else:
            tps.append(round_to_tick(entry * (1.0 - pct / 100.0), tick))
    return tps


def _sl_from_entry(side: str, entry: float, tick: float, atr_abs: float | None):
    """
    SL 优先使用百分比 SAFE_SL_PCT；如有 ATR，则在百分比与 ATR*SAFE_ATR_MULT 之间取“更宽”一档。
    """
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
    """
    “立即入场”的 TP/SL（以当前价为 entry）。若能计算 ATR，则 SL 与 ATR 结合。
    """
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
    """
    保守入场：基于 15m 的结构点位 + Fib 比例回测（多：靠近前高回踩；空：靠近前低回抽）。
    无法获取结构时，fallback 为按 Fib 调整的价位。
    """
    entry = close_last
    if side == "long":
        if isinstance(hhv_prev, (int, float)) and hhv_prev > 0 and close_last > 0:
            # 回测到 前高 与 当前之间的 Fib 位置
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
    """
    以保守入场价生成 TP/SL。
    """
    tps = _tps_from_entry(side, entry_safe, tick)
    sl = _sl_from_entry(side, entry_safe, tick, atr_abs=None)
    return (*tps, sl)


# =========================
# ChaseGuard：防追价 / 收盘确认 / 最小R:R 多数表决
# 可通过 Strategy.overrides 覆盖阈值
# =========================
_GUARD_STATS = defaultdict(int)

# 默认全局（可被每个 Strategy.overrides 覆盖）
STRICT_DEFAULTS = {
    # 接近日内极值时禁止追多/追空的最小“安全距离”（%）
    "NO_CHASE_DAYEXT_GAP_PCT": 0.8,  # 与日内高/低的距离 < 0.8% 则拦截
    # 价格相对 EMA20 的“溢价”限制（%）
    "NO_CHASE_EMA20_PREMIUM_PCT": 1.2,
    # 需要当前K线至少走完的比例（收盘确认），强势类信号才需要
    "CLOSE_CONFIRM_FRAC": 0.60,
    # 单根实体过大直接拒绝（%）
    "MAX_BODY_FOR_LONG": 2.5,
    "MAX_BODY_FOR_SHORT": 2.5,
    # 最小风险收益比（R:R）底线（基于保守入场价）
    "MIN_RR": 1.25,
    # 哪些多/空类信号需要接近日内高/低时跳过
    "SKIP_LONG_NEAR_DHIGH_FOR": [
        "explode_up",
        "ema_rebound_long",
        "bb_squeeze_long",
        "pullback_long",
    ],
    "SKIP_SHORT_NEAR_DLOW_FOR": [
        "explode_down",
        "ema_rebound_short",
        "bb_squeeze_short",
        "pullback_short",
    ],
    # RSI 顶/底保护（可按需使用；当前未强制启用为票据，只保留在 overrides 里可读取）
    "RSI_CAP_LONG": 68,
    "RSI_CAP_SHORT": 32,
}

GUARD_LEVELS = {
    "OFF": dict(
        need_close_frac=0.0,
        max_body=99.0,
        gap_pct=0.0,
        ema20_prem=99.0,
        min_rr=0.0,
        votes_need=99,
    ),
    "SOFT": dict(
        need_close_frac=0.50,
        max_body=3.5,
        gap_pct=0.5,
        ema20_prem=1.8,
        min_rr=1.10,
        votes_need=2,
    ),
    "MEDIUM": dict(
        need_close_frac=0.60,
        max_body=2.5,
        gap_pct=0.8,
        ema20_prem=1.5,
        min_rr=1.25,
        votes_need=2,
    ),
    # ↓↓↓ 关键：把 HARD 放宽一些
    "HARD": dict(
        need_close_frac=0.70,
        max_body=2.0,
        gap_pct=1.0,
        ema20_prem=1.6,
        min_rr=1.25,
        votes_need=2,
    ),
}


def _eval_guard(
    kind: str,
    side: str,  # "long" / "short"
    *,
    elapsed: int,
    BAR_SEC: int,
    body_pct: float,
    rsi: float | None,
    ema20_val: float | None,
    c: float,
    day_high: float | None,
    day_low: float | None,
    entry_safe: float | None,
    tp1_s: float | None,
    sl_s: float | None,
    level: str = "MEDIUM",
    shadow: bool = False,
):
    """
    根据分级参数做“多数表决”：满足 >= votes_need 即拦截。
    shadow=True 时只统计不拦截（影子模式）。
    """
    P = GUARD_LEVELS[level]
    votes = 0
    reasons = []

    # 1) 收盘确认（强波动类/追价风险类信号）
    need_close = kind in (
        "explode_up",
        "explode_down",
        "ema_rebound_long",
        "ema_rebound_short",
        "bb_squeeze_long",
        "bb_squeeze_short",
        "pullback_long",
        "pullback_short",
    )
    if need_close and elapsed < int(BAR_SEC * P["need_close_frac"]):
        votes += 1
        reasons.append(f"close<{int(P['need_close_frac'] * 100)}%")

    # 2) 实体过大（避免追大阳/大阴）
    if body_pct > P["max_body"]:
        votes += 1
        reasons.append(f"body>{P['max_body']}%")

    # 3) 接近日内极值（不追高/不追底）
    if side == "long":
        if day_high and c > 0:
            gap = (day_high - c) / c * 100.0
            if (
                gap < P["gap_pct"]
                and kind in STRICT_DEFAULTS["SKIP_LONG_NEAR_DHIGH_FOR"]
            ):
                votes += 1
                reasons.append(f"nearDHigh<{P['gap_pct']}%")
    else:
        if day_low and c > 0:
            gap = (c - day_low) / c * 100.0
            if (
                gap < P["gap_pct"]
                and kind in STRICT_DEFAULTS["SKIP_SHORT_NEAR_DLOW_FOR"]
            ):
                votes += 1
                reasons.append(f"nearDLow<{P['gap_pct']}%")

    # 4) EMA20 溢价限制（价位偏离均线太多，容易追高/抄底失败）
    if ema20_val is not None and not pd.isna(ema20_val) and ema20_val > 0:
        prem = (c / ema20_val - 1.0) * 100.0
        if side == "long":
            if prem > P["ema20_prem"]:
                votes += 1
                reasons.append(f"ema20+{prem:.2f}%>{P['ema20_prem']}%")
        else:
            if (-prem) > P["ema20_prem"]:
                votes += 1
                reasons.append(f"ema20-{(-prem):.2f}%>{P['ema20_prem']}%")

    # 5) 最小 R:R 底线（基于保守入场）
    rr_ok = False
    rr = None
    if entry_safe and sl_s and tp1_s and entry_safe != sl_s:
        if side == "long" and tp1_s > entry_safe and sl_s < entry_safe:
            rr = (tp1_s - entry_safe) / (entry_safe - sl_s)
        elif side == "short" and tp1_s < entry_safe and sl_s > entry_safe:
            rr = (entry_safe - tp1_s) / (sl_s - entry_safe)
        rr_ok = (rr is not None) and (rr >= P["min_rr"])
    if rr_ok is False:
        votes += 1
        reasons.append(f"RR<{P['min_rr']}")

    block = votes >= P["votes_need"]
    if shadow:
        block = False

    _GUARD_STATS["total_seen"] += 1
    for r in reasons:
        _GUARD_STATS["reason_" + r] += 1
    if block:
        _GUARD_STATS["blocked_" + level.lower()] += 1

    return block, reasons, votes, P["votes_need"]


# =========================
# 主函数：detect_signal
# =========================
def detect_signal(
    ex,
    symbol: str,
    strong_up_map: dict,
    strong_dn_map: dict,
    strategy: Strategy,
):
    """
    15m / 1h 核心检测逻辑：形成统一 payload，供 formatter / notifier 使用。
    返回 (ok, payload|None)
    """
    # 1) 周期与权重由策略提供
    TIMEFRAME_FAST = strategy.timeframe_fast
    BAR_SEC = FRAME_SEC[TIMEFRAME_FAST]

    # 2) 覆盖部分全局参数（如果策略提供）
    MIN_EQBAR = (
        strategy.overrides.get("MIN_QV5M_USD", MIN_QV5M_USD)
        if strategy.overrides
        else MIN_QV5M_USD
    )

    # 3) 评分权重/标尺使用策略参数
    W_VOLR_NOW = strategy.w_volr_now
    W_EQ_NOW_USD = strategy.w_eq_now_usd
    W_ABS_PCT = strategy.w_abs_pct
    W_TREND_ALIGN = strategy.w_trend_align
    SCALE_EQ5M_USD = strategy.scale_eq_bar_usd

    # 4) Pullback 的回看根数
    PB_LOOKBACK_HI = strategy.pb_lookback_hi

    # —— 数据获取 —— #
    df = fetch_ohlcv_df(
        ex,
        symbol,
        strategy.timeframe_fast,
        limit=max(LOOKBACK_VOL, BREAKOUT_WINDOW, BASELINE_BARS, 300) + 8,
    )
    if len(df) < (BASELINE_BARS + 6):
        return False, None

    df_closed = df.iloc[:-1].copy()
    cur = df.iloc[-1]
    if len(df_closed) < (BASELINE_BARS + 3):
        return False, None

    o = float(cur["open"])
    h = float(cur["high"])
    low = float(cur["low"])
    c = float(cur["close"])
    vol = float(cur["volume"])
    cur_open_ts = int(cur["ts"]) // 1000
    now_ts = int(time.time())
    elapsed = max(1, min(BAR_SEC, now_ts - cur_open_ts))  # 本 bar 已进行秒数
    vps_now = vol / elapsed

    recent_closed = df_closed.tail(BASELINE_BARS).copy()
    vps_hist = (recent_closed["volume"].astype(float) / BAR_SEC).tolist()
    vps_base = float(sum(vps_hist) / len(vps_hist)) if len(vps_hist) else 0.0
    if vps_base <= 0:
        vps_base = float(df_closed.iloc[-1]["volume"]) / BAR_SEC
    volr_now = vps_now / max(1e-12, vps_base)

    eq_base_bar_usd = vps_base * BAR_SEC * c
    eq_now_bar_usd = vps_now * BAR_SEC * c

    # —— 符号参数（因子/门槛） —— #
    Pm = resolve_params_for_symbol(symbol)
    EXP_VOLR = Pm["EXPLODE_VOLR"]
    UP_TH = Pm["PRICE_UP_TH"]
    PB_HI_PCT = Pm["PB_LOOKBACK_HI_PCT"]
    MIN_EQBAR = Pm["MIN_QV5M_USD"]  # 名称沿用，含义：本 bar 等效成交额

    if eq_now_bar_usd < MIN_EQBAR:
        return False, None

    # —— 趋势与波动 —— #
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

    # 宽松顺势
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

    # 15m 自身 EMA；HTF 取真实 1h EMA
    close_15m = df_closed["close"].astype(float).values
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

    # —— 爆发（15m） —— #
    pct_now = percent_change(c, o)
    explode_up = (volr_now >= EXP_VOLR) and (pct_now >= UP_TH)
    explode_down = (volr_now >= EXP_VOLR) and (
        pct_now <= min(NO_FOLLOW_UP_TH, PRICE_DN_TH) or pct_now <= PRICE_DN_TH
    )
    if REQUIRE_TREND_ALIGN:
        if explode_up and not trending_up:
            explode_up = False
        if explode_down and not trending_dn:
            explode_down = False

    # 15m 结构确认
    STRUCT_TOL_PCT = 0.001
    STRUCT_TOL_TICKS = 2
    tick_tmp = get_tick_size(ex, symbol)
    struct_tol = max(c * STRUCT_TOL_PCT, STRUCT_TOL_TICKS * tick_tmp)
    if BO_REQUIRE_HHV:
        if explode_up:
            explode_up = explode_up and (c >= max(hhv_prev - struct_tol, o))
        if explode_down:
            explode_down = explode_down and (c <= min(llv_prev + struct_tol, o))

    # QUIET 下爆发额外门槛
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
                df_closed["close"].astype(float).values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0,
            )
            if pd.notna(_u[-1]) and pd.notna(_l[-1]) and pd.notna(_m[-1]):
                _width_series = (_u - _l) / np.maximum(_m, 1e-12)
                _recent_min = float(pd.Series(_width_series[-12:]).min())
                if not (_recent_min <= EXPLODE_CONTRACTION_BB_WIDTH):
                    explode_up = explode_down = False

    # —— 自适应（放宽/收紧） —— #
    PB_MIN_BOUNCE_PCT_dyn = max(
        0.03, min(0.08, PB_MIN_BOUNCE_PCT * (0.7 if adx >= ADX_MIN_TREND else 0.85))
    )
    PB_HI_PCT_dyn = max(0.6, min(2.5, PB_HI_PCT * (0.75 if atrp <= 2.0 else 0.95)))
    CAPV_dyn = max(2.2, CAP_VOLR * (0.85 if not is_trending else 1.0))
    CAPW_dyn = max(0.40, min(0.75, CAP_WICK_RATIO * (0.9 if not is_trending else 1.0)))

    # —— Pullback（顺势回撤） —— #
    def recent_high_distance_pct(
        df_closed_: pd.DataFrame, lookback: int, price_now: float
    ) -> float:
        if len(df_closed_) < lookback or price_now <= 0:
            return 0.0
        hhv = float(df_closed_["high"].tail(lookback).max())
        return max(0.0, (hhv - price_now) / price_now * 100.0)

    def recent_low_distance_pct(
        df_closed_: pd.DataFrame, lookback: int, price_now: float
    ) -> float:
        if len(df_closed_) < lookback or price_now <= 0:
            return 0.0
        llv = float(df_closed_["low"].tail(lookback).min())
        return max(0.0, (price_now - llv) / price_now * 100.0)

    pullback_long = False
    pullback_short = False
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

    # —— Capitulation（允许弱趋势） —— #
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
        lw = lower_wick_ratio(o, h, low, c)
        if (
            (volr_now >= max(2.0, CAPV_dyn * 0.95))
            and (lw >= max(0.33, CAPW_dyn * 0.9))
            and (
                ((c - low) / max(h - low, 1e-12) >= 0.6)
                or (pct_now >= CAP_ALLOW_BOUNCE)
            )
        ):
            cap_long = True
    if ENABLE_CAPITULATION and (not is_trending or weak_trend):
        uw = upper_wick_ratio(o, h, low, c)
        if (
            (volr_now >= max(2.0, CAPV_dyn * 0.95))
            and (uw >= max(0.33, CAPW_dyn * 0.9))
            and (
                ((h - c) / max(h - low, 1e-12) >= 0.6) or (pct_now <= -CAP_ALLOW_BOUNCE)
            )
        ):
            cap_short = True

    # —— EMA 回踩（15m） —— #
    ema8 = (
        ta.EMA(close_15m, timeperiod=8)
        if len(close_15m) >= 8
        else np.array([np.nan, np.nan])
    )
    ema20 = (
        ta.EMA(close_15m, timeperiod=20)
        if len(close_15m) >= 20
        else np.array([np.nan, np.nan])
    )
    ema50 = (
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
        and pd.notna(ema20[-1])
        and pd.notna(ema50[-1])
    ):
        up_trend_ok = ema20[-1] >= ema50[-1]
        dn_trend_ok = ema20[-1] <= ema50[-1]
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

    # —— BB 挤压（15m） —— #
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

    # —— 方向闸门（15m 自身 + 1h） —— #
    if HTF_BULL:
        explode_down = False
        pullback_short = False
        ema_rebound_short = False
        bb_reversal_short = False
    if HTF_BEAR:
        explode_up = False
        pullback_long = False
        ema_rebound_long = False
        bb_reversal_long = False

    # —— 在强势多头下对空头类抬门槛 —— #
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

    # —— 空头结构确认 —— #
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

    # —— 爆发后做空锁 —— #
    lock_ok = True
    last_up_ts = LAST_EXPLODE_UP.get(symbol)
    if last_up_ts and (int(time.time()) - last_up_ts < EXPLODE_LOCK_MIN):
        lock_ok = pct_now <= -EXPLODE_LOCK_RET_PCT
    if not lock_ok:
        explode_down = pullback_short = ema_rebound_short = bb_reversal_short = False

    # —— 有无信号 —— #
    any_long = (
        explode_up or pullback_long or cap_long or ema_rebound_long or bb_reversal_long
    )
    any_short = (
        explode_down
        or pullback_short
        or cap_short
        or ema_rebound_short
        or bb_reversal_short
    )
    if not (any_long or any_short):
        dbg(
            f"[FILTERED {symbol}] eq_bar=${eq_now_bar_usd:,.0f} "
            f"PB={pullback_long or pullback_short} EMA={('L' if ema_rebound_long else '')}{('S' if ema_rebound_short else '')} "
            f"CAP={('L' if cap_long else '')}{('S' if cap_short else '')} "
            f"BB={('L' if bb_reversal_long else '')}{('S' if bb_reversal_short else '')} "
            f"EXP={explode_up or explode_down} volr={volr_now:.2f}"
        )
        return False, None

    # —— 获取 tick & 日内信息 —— #
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

    # —— 生成 kind、命令与中文标题 —— #
    if any_long:
        if explode_up:
            kind = "explode_up"
        elif ema_rebound_long:
            kind = "ema_rebound_long"
        elif pullback_long:
            kind = "pullback_long"
        elif bb_reversal_long:
            kind = "bb_squeeze_long"
        else:
            kind = "cap_long"

        tp1_i, tp2_i, tp3_i, sl_i = dynamic_targets(
            symbol="NA", side="long", entry_price=c, df_closed=df_closed, tick=tick
        )
        entry_safe = conservative_entry(
            symbol="NA",
            side="long",
            close_last=c,
            df_closed=df_closed,
            tick=tick,
            hhv_prev=hhv_prev,
            llv_prev=None,
        )
        tp1_s, tp2_s, tp3_s, sl_s = tpsl_for_safe_entry("long", entry_safe, tick)
        cmd_immd = f"/forcelong {symbol} 10 10 {tp1_i} {tp2_i} {tp3_i} {sl_i} {c:.6g}"
        cmd_safe = (
            f"/forcelong {symbol} 10 10 {tp1_s} {tp2_s} {tp3_s} {sl_s} {entry_safe}"
        )

        trend_align = trending_up
        if kind == "explode_up":
            LAST_EXPLODE_UP[symbol] = int(time.time())
    else:
        if explode_down:
            kind = "explode_down"
        elif ema_rebound_short:
            kind = "ema_rebound_short"
        elif pullback_short:
            kind = "pullback_short"
        elif bb_reversal_short:
            kind = "bb_squeeze_short"
        else:
            kind = "cap_short"

        tp1_i, tp2_i, tp3_i, sl_i = dynamic_targets(
            symbol="NA", side="short", entry_price=c, df_closed=df_closed, tick=tick
        )
        entry_safe = conservative_entry(
            symbol="NA",
            side="short",
            close_last=c,
            df_closed=df_closed,
            tick=tick,
            hhv_prev=None,
            llv_prev=llv_prev,
        )
        tp1_s, tp2_s, tp3_s, sl_s = tpsl_for_safe_entry("short", entry_safe, tick)
        cmd_immd = f"/forceshort {symbol} 10 10 {tp1_i} {tp2_i} {tp3_i} {sl_i} {c:.6g}"
        cmd_safe = (
            f"/forceshort {symbol} 10 10 {tp1_s} {tp2_s} {tp3_s} {sl_s} {entry_safe}"
        )

        trend_align = trending_dn

    kind_cn = KINDS_CN.get(kind, kind)

    # —— 趋势文字 —— #
    tr2 = trend_scores(df_closed, bars=TREND_LOOKBACK_BARS)
    adx2 = last_adx(df_closed, period=14)
    atrp2 = last_atr_pct(df_closed, period=14)
    trend_text = (
        f"Class={SYMBOL_CLASS.get(symbol, '?')} | ADX≈{adx2:.1f}, ρ={tr2['spearman']:.2f}, "
        f"net%={tr2['net_pct']:.2f}, ATR%≈{atrp2:.2f} | HTF:{'BULL' if HTF_BULL else ('BEAR' if HTF_BEAR else '—')}"
    )

    # —— 触发原因（基础） —— #
    reasons = [f"class={SYMBOL_CLASS.get(symbol, '?')}; mode={MODE}"]
    if HTF_BULL:
        reasons.append("HTF=15m/1h 多头闸门")
    if HTF_BEAR:
        reasons.append("HTF=15m/1h 空头闸门")

    if kind == "explode_up":
        reasons += [
            f"explode_up: volr {volr_now:.2f}≥{EXP_VOLR:.2f}",
            f"pct_now {pct_now:.2f}%≥{UP_TH:.2f}%",
            f"struct c≥HHV({BO_WINDOW})-tol",
            f"trend_up ADX {adx:.1f}≥{ADX_MIN_TREND}, ρ {tr['spearman']:.2f}≥{TREND_MIN_SPEARMAN}",
        ]
        if MODE == "QUIET":
            reasons.append("QUIET: 早期/离日内极值远/前置收缩")
    elif kind == "explode_down":
        reasons += [
            f"explode_down: volr {volr_now:.2f}≥{EXP_VOLR:.2f}",
            f"pct_now {pct_now:.2f}%≤{min(NO_FOLLOW_UP_TH, PRICE_DN_TH):.2f}%",
            f"struct c≤LLV({BO_WINDOW})+tol",
            f"trend_dn ADX {adx:.1f}≥{ADX_MIN_TREND}, ρ {tr['spearman']:.2f}≤{-TREND_MIN_SPEARMAN}",
        ]
        if last_up_ts and (int(time.time()) - last_up_ts < EXPLODE_LOCK_MIN):
            reasons.append(
                f"post-explode lock checked: {int(time.time()) - last_up_ts}s"
            )
        if MODE == "QUIET":
            reasons.append("QUIET: 早期/离日内极值远/前置收缩")
    elif kind == "pullback_long":
        dist_hi = (
            float(df_closed["high"].tail(PB_LOOKBACK_HI).max() - c)
            / max(c, 1e-12)
            * 100.0
        )
        reasons += [
            f"pullback_long: dist_HH({PB_LOOKBACK_HI}) {dist_hi:.2f}%≥{PB_HI_PCT_dyn:.2f}%",
            f"RSI {rsi:.1f}≤{PB_RSI_TH} 或 W%R {wr:.1f}≤{PB_WR_TH}",
            f"bounce pct_now {pct_now:.2f}%≥{PB_MIN_BOUNCE_PCT_dyn:.2f}%",
        ]
    elif kind == "pullback_short":
        dist_lo = (
            float(c - df_closed["low"].tail(PB_LOOKBACK_HI).min())
            / max(c, 1e-12)
            * 100.0
        )
        reasons += [
            f"pullback_short: dist_LL({PB_LOOKBACK_HI}) {dist_lo:.2f}%≥{PB_HI_PCT_dyn:.2f}%",
            f"RSI {rsi:.1f}≥{100 - PB_RSI_TH} 或 W%R {wr:.1f}≥{-100 - PB_WR_TH}",
            f"bounce pct_now {pct_now:.2f}%≤{-PB_MIN_BOUNCE_PCT_dyn:.2f}%",
        ]
    elif kind == "cap_long":
        reasons += [
            f"cap_long: wick_low≥{CAPW_dyn:.2f}, volr {volr_now:.2f}≥{max(2.0, CAPV_dyn * 0.95):.2f}"
        ]
    elif kind == "cap_short":
        reasons += [
            f"cap_short: wick_high≥{CAPW_dyn:.2f}, volr {volr_now:.2f}≥{max(2.0, CAPV_dyn * 0.95):.2f}"
        ]
    elif kind == "ema_rebound_long":
        reasons.append(
            f"ema_rebound_long: EMA20≥EMA50 & cross_up EMA8 ; volr {volr_now:.2f}≥{max(1.25, EXP_VOLR * 0.5):.2f}"
        )
    elif kind == "ema_rebound_short":
        reasons.append(
            f"ema_rebound_short: EMA20≤EMA50 & cross_down EMA8 ; volr {volr_now:.2f}≥{max(1.25, EXP_VOLR * 0.5):.2f}"
        )
    elif "bb_squeeze" in kind:
        if "long" in kind:
            reasons.append(
                f"bb_squeeze_long: width={bb_width:.3%}≤1.50% & c≈≥Upper ; volr {volr_now:.2f}≥{max(1.6, EXP_VOLR * 0.6):.2f}"
            )
        else:
            reasons.append(
                f"bb_squeeze_short: width={bb_width:.3%}≤1.50% & c≈≤Lower ; volr {volr_now:.2f}≥{max(1.6, EXP_VOLR * 0.6):.2f}"
            )

    # —— 评分 —— #
    sc = (
        W_VOLR_NOW * max(0.0, min(1.0, (volr_now - 1.0) / 4.0))
        + W_EQ_NOW_USD * max(0.0, min(1.0, eq_now_bar_usd / max(1.0, SCALE_EQ5M_USD)))
        + W_ABS_PCT * max(0.0, min(1.0, abs(pct_now) / max(1e-9, SCALE_ABS_PCT)))
        + W_TREND_ALIGN * (1.0 if trend_align else 0.0)
    )
    if "ema_rebound" in kind:
        sc += 0.12
    elif "bb_squeeze" in kind:
        sc += 0.10
    elif "pullback" in kind:
        sc += 0.10
    elif "cap_" in kind:
        sc += 0.15
    elif "explode" in kind:
        sc += 0.15

    if HTF_BULL and ("_short" in kind or "explode_down" in kind):
        sc = max(0.0, sc - 0.20)
    elif HTF_BEAR and ("_long" in kind or "explode_up" in kind):
        sc = max(0.0, sc - 0.20)

    # —— 文本核心 + 命令 —— #
    reasons_show = reasons if PRINT_FULL_REASONS else reasons[:MAX_REASONS_IN_MSG]
    text_core = [
        f"Symbol: <b>{symbol}</b>",
        f"Price: <code>{c:.6g}</code>",
        f"Now {TIMEFRAME_FAST}: <b>{pct_now:.2f}%</b> | VolR: <b>{volr_now:.2f}x</b> | EqBar≈<b>${eq_now_bar_usd:,.0f}</b>",
        f"Base vps: <b>{vps_base:.4f}</b> | Trend: {trend_text} {'✅' if trend_align else '—'}",
        "Why: " + " ; ".join(reasons_show),
    ]
    if not SAFE_MODE_ALWAYS:
        text_core.append(f"<code>{cmd_immd}</code>")
    text_core += [
        f"<b>Conservative Entry:</b> <code>{entry_safe}</code> | SL(≈{SAFE_SL_PCT:.1f}%): <code>{tpsl_for_safe_entry('long' if 'long' in kind else 'short', entry_safe, tick)[-1]}</code>",
        f"<code>{cmd_safe}</code>",
    ]

    # =========================
    # 应用 ChaseGuard（分级/多数表决）
    # =========================
    # 合并 overrides
    S = dict(STRICT_DEFAULTS)
    if getattr(strategy, "overrides", None):
        for k in S.keys():
            if k in strategy.overrides:
                S[k] = strategy.overrides[k]

    body_pct = abs(c - o) / max(o, 1e-12) * 100.0
    side_now = (
        "long"
        if ("_long" in kind or "explode_up" in kind or kind in ("cap_long",))
        else "short"
    )
    level = "HARD" if MODE == "QUIET" else "MEDIUM"
    SHADOW = True  # 如需先观察数据不拦截，可改 True

    block, reasons_guard, votes, need_votes = _eval_guard(
        kind,
        side_now,
        elapsed=elapsed,
        BAR_SEC=BAR_SEC,
        body_pct=body_pct,
        rsi=rsi,
        ema20_val=float(ema20_15) if pd.notna(ema20_15) else None,
        c=c,
        day_high=day_high,
        day_low=day_low,
        entry_safe=entry_safe,
        tp1_s=tp1_s,
        sl_s=sl_s,
        level=level,
        shadow=SHADOW,
    )
    # ✅ 无论是否阻断，都打印一条日志（SHADOW 下会显示 SHADOW）
    status = "SHADOW" if SHADOW else ("BLOCK" if block else "PASS")
    from .loggingx import dbg

    dbg(
        f"[GUARD {status}] {symbol} {kind} votes={votes}/{need_votes} reasons={reasons_guard}"
    )

    # 仅在非影子模式且被阻断时才返回
    if (not SHADOW) and block:
        return False, None

    # —— payload —— #
    payload = {
        "symbol": symbol,
        "kind": kind,
        "kind_cn": KINDS_CN.get(kind, kind),
        "title": KINDS_CN.get(kind, kind),
        "timeframe_fast": strategy.timeframe_fast,
        "pct_now": float(pct_now),
        "volr_now": float(volr_now),
        "vps_now": float(vps_now),
        "vps_base": float(vps_base),
        "eq_base_bar_usd": float(eq_base_bar_usd),
        "eq_now_bar_usd": float(eq_now_bar_usd),
        "trend_align": bool(trend_align),
        "trend_text": trend_text,
        "text_core": text_core,
        "score": float(sc),
        "reasons": reasons,
        "day_high": day_high,
        "day_low": day_low,
        "pct24": pct24,
        "dist_day_high_pct": dist_day_high_pct,
        "dist_day_low_pct": dist_day_low_pct,
        "last_price": last_price or c,
        "cmd_safe": text_core[-1],
    }
    if not SAFE_MODE_ALWAYS:
        payload["cmd_immd"] = text_core[-2]

    return True, payload
