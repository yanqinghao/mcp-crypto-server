# -*- coding: utf-8 -*-
"""
signals_pro.py
增强版信号检测：
- 若基础 signals.detect_signal 命中 => 直接返回基础结果（完全不改变行为）
- 若基础未命中 => 尝试以下“专业级”新增信号：
  1) trend_break_up/down         趋势线突破
  2) volume_shift_long/short     横盘放量（吸筹/出货）
  3) ema_stack_bull/bear         EMA 多/空头排列确认
  4) rsi_div_long/short          RSI 背离（多/空）
  5) climax_bottom/top           量能峰值反转（底/顶）
  6) equilibrium_break/reject    平衡区突破 / 假突破回收
"""

import time
import numpy as np
import pandas as pd
import talib as ta

# 先引入你已有的“基础检测”
from .detect import detect_signal as _base_detect_signal

from .config import (
    MODE,
    TIMEFRAME_HOURLY,
    FRAME_SEC,
    LOOKBACK_VOL,
    BREAKOUT_WINDOW,
    BASELINE_BARS,
    TREND_LOOKBACK_BARS,
    BO_WINDOW,
    SCALE_EQ5M_USD,
    SCALE_ABS_PCT,
    W_VOLR_NOW,
    W_EQ_NOW_USD,
    W_ABS_PCT,
    W_TREND_ALIGN,
    SAFE_MODE_ALWAYS,
    SAFE_SL_PCT,
    SAFE_TP_PCTS,
    SAFE_ATR_MULT,
    SAFE_FIB_RATIO,
    PRINT_FULL_REASONS,
    MAX_REASONS_IN_MSG,
    MIN_QV5M_USD,
    KINDS_CN,  # 用于已有的中文映射
)
from .exchange import fetch_ohlcv_df, get_tick_size, round_to_tick
from .ta_utils import (
    trend_scores,
    last_rsi,
    last_adx,
    last_atr_pct,
    compute_atr,
    percent_change,
)
from .candidates import resolve_params_for_symbol, SYMBOL_CLASS
from .liquidity import get_day_stats

# ==== 本模块的新增信号开关（默认启用，若你想集中管理也可搬到 config.py） ====
ENABLE_TREND_BREAK = True
ENABLE_VOLUME_SHIFT = True
ENABLE_EMA_STACK = True
ENABLE_RSI_DIVERGENCE = True
ENABLE_CLIMAX = True
ENABLE_EQUILIBRIUM = True

# ---- 各自的轻量阈值（与策略强弱无关，仅用于是否触发）----
TB_LOOKBACK = 10  # 趋势线回归回看
TB_UP_BUF = 0.002  # 向上突破缓冲（0.2%）
TB_DN_BUF = 0.002
VS_VOLR = 2.2  # 横盘放量的量能倍数
VS_PCT_ABS = 0.12  # 横盘时当根涨跌幅阈值（%）
EMA_STACK_ADX = 16  # EMA 排列时的最小 ADX
DIV_LOOKBACK = 18  # 背离对比距离
DIV_TOL_PCT = 0.15  # 背离价格/RSI比较的宽容度
CLIMAX_VOLR = 5.0  # 量能峰值倍数
CLIMAX_RSI_H = 72  # 顶部 RSI
CLIMAX_RSI_L = 28  # 底部 RSI
EQ_WIN = 20  # 平衡区窗口
EQ_BREAK_BUF = 0.003  # 突破缓冲 0.3%
EQ_REJ_BODY = 0.6  # 收回时实体占比要求

# ==== 本模块新增信号的中文映射（不改你的 config 也能显示中文） ====
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
    "equilibrium_break": "平衡区突破",
    "equilibrium_reject": "平衡区假突破 · 回收",
}


# === 复用基础模块里的目标价计算 ===
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


def _kind_cn(kind: str) -> str:
    return KINDS_CN.get(kind, LOCAL_KINDS_CN.get(kind, kind))


def detect_signal(ex, symbol, strong_up_map: dict, strong_dn_map: dict, strategy):
    """
    先走基础检测；基础未命中时，再尝试新增信号。
    返回 (ok, payload|None)
    """
    # 1) 先尝试基础 signals
    try:
        ok, payload = _base_detect_signal(
            ex, symbol, strong_up_map, strong_dn_map, strategy
        )
        if ok:
            return ok, payload
    except Exception as e:
        # 基础检测异常不影响扩展检测继续进行
        print(f"[WARN] base detect_signal error on {symbol}: {e}")

    # 2) 扩展检测（以下为轻量实现，尽量少引入额外代价）
    TIMEFRAME_FAST = strategy.timeframe_fast
    # TIMEFRAME_HTF = strategy.timeframe_htf
    BAR_SEC = FRAME_SEC[TIMEFRAME_FAST]

    df = fetch_ohlcv_df(
        ex,
        symbol,
        strategy.timeframe_fast,
        limit=max(LOOKBACK_VOL, BREAKOUT_WINDOW, BASELINE_BARS, 220) + 8,
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
    elapsed = max(1, min(BAR_SEC, now_ts - cur_open_ts))
    vps_now = vol / elapsed

    recent_closed = df_closed.tail(BASELINE_BARS).copy()
    vps_hist = (recent_closed["volume"].astype(float) / BAR_SEC).tolist()
    vps_base = float(sum(vps_hist) / len(vps_hist)) if len(vps_hist) else 0.0
    if vps_base <= 0:
        vps_base = float(df_closed.iloc[-1]["volume"]) / BAR_SEC
    volr_now = vps_now / max(1e-12, vps_base)

    eq_now_bar_usd = vps_now * BAR_SEC * c
    P = resolve_params_for_symbol(symbol)
    MIN_EQBAR = P.get("MIN_QV5M_USD", MIN_QV5M_USD)
    if eq_now_bar_usd < MIN_EQBAR:
        return False, None

    # 趋势与波动
    # tr = trend_scores(df_closed, bars=TREND_LOOKBACK_BARS)
    adx = last_adx(df_closed, period=14)
    # atrp = last_atr_pct(df_closed, period=14)
    rsi = last_rsi(df_closed, period=14)

    # HTF/MA 门槛
    close_ = df_closed["close"].astype(float).values
    ema8 = (
        ta.EMA(close_, timeperiod=8) if len(close_) >= 8 else np.array([np.nan, np.nan])
    )
    ema20 = (
        ta.EMA(close_, timeperiod=20)
        if len(close_) >= 20
        else np.array([np.nan, np.nan])
    )
    ema50 = (
        ta.EMA(close_, timeperiod=50)
        if len(close_) >= 50
        else np.array([np.nan, np.nan])
    )

    # 1h/4h EMA 闸门与基础版保持一致
    try:
        df_htf = fetch_ohlcv_df(ex, symbol, TIMEFRAME_HOURLY, limit=120)
        close_htf = df_htf["close"].astype(float).values
        ema20_htf = (
            ta.EMA(close_htf, timeperiod=20)[-1] if len(close_htf) >= 20 else np.nan
        )
        ema50_htf = (
            ta.EMA(close_htf, timeperiod=50)[-1] if len(close_htf) >= 50 else np.nan
        )
    except Exception:
        ema20_htf = ema50_htf = np.nan

    HTF_BULL = (
        pd.notna(ema20[-1])
        and pd.notna(ema50[-1])
        and pd.notna(ema20_htf)
        and pd.notna(ema50_htf)
        and ema20[-1] >= ema50[-1]
        and ema20_htf >= ema50_htf
        and c >= ema20[-1]
        and c >= ema20_htf
    )
    HTF_BEAR = (
        pd.notna(ema20[-1])
        and pd.notna(ema50[-1])
        and pd.notna(ema20_htf)
        and pd.notna(ema50_htf)
        and ema20[-1] <= ema50[-1]
        and ema20_htf <= ema50_htf
        and c <= ema20[-1]
        and c <= ema20_htf
    )

    # ---- 新增信号判定 ----
    pct_now = percent_change(c, o)

    def _poly_slope(arr):
        x = np.arange(len(arr))
        try:
            return float(np.polyfit(x, arr, 1)[0])
        except Exception:
            return 0.0

    kinds = []
    reasons = []

    # 1) 趋势线突破
    if ENABLE_TREND_BREAK and len(df_closed) >= TB_LOOKBACK + 2:
        highs = df_closed["high"].tail(TB_LOOKBACK).values
        lows = df_closed["low"].tail(TB_LOOKBACK).values
        s_h = _poly_slope(highs)
        s_l = _poly_slope(lows)
        break_up = (s_h < 0) and (c >= highs[-1] * (1.0 + TB_UP_BUF))
        break_down = (s_l > 0) and (c <= lows[-1] * (1.0 - TB_DN_BUF))
        if HTF_BULL:
            break_down = False
        if HTF_BEAR:
            break_up = False
        if break_up:
            kinds.append("trend_break_up")
            reasons.append(f"trend_break_up: slope(H)<0 & c≥H_last*(1+{TB_UP_BUF:.3f})")
        if break_down:
            kinds.append("trend_break_down")
            reasons.append(
                f"trend_break_down: slope(L)>0 & c≤L_last*(1-{TB_DN_BUF:.3f})"
            )

    # 2) 横盘放量（吸筹/出货）
    if ENABLE_VOLUME_SHIFT:
        # 横盘定义：当根振幅小且 pct_now 很小，但量能放大
        body = abs(c - o) / max(o, 1e-12) * 100.0
        rng = (h - low) / max(o, 1e-12) * 100.0
        flat = (body <= VS_PCT_ABS) and (rng <= max(VS_PCT_ABS * 2.2, 0.35))
        if flat and volr_now >= VS_VOLR:
            # 方向用 RSI/EMA20 简单区分
            if rsi >= 52 or (pd.notna(ema20[-1]) and c >= float(ema20[-1])):
                if HTF_BEAR:
                    pass
                else:
                    kinds.append("volume_shift_long")
                    reasons.append(
                        f"volume_shift_long: flat({body:.2f}%/{rng:.2f}%) & volr {volr_now:.2f}≥{VS_VOLR:.1f}"
                    )
            elif rsi <= 48 or (pd.notna(ema20[-1]) and c <= float(ema20[-1])):
                if HTF_BULL:
                    pass
                else:
                    kinds.append("volume_shift_short")
                    reasons.append(
                        f"volume_shift_short: flat({body:.2f}%/{rng:.2f}%) & volr {volr_now:.2f}≥{VS_VOLR:.1f}"
                    )

    # 3) EMA 多/空头排列确认
    if (
        ENABLE_EMA_STACK
        and len(close_) >= 50
        and pd.notna(ema8[-1])
        and pd.notna(ema20[-1])
        and pd.notna(ema50[-1])
    ):
        stack_bull = (
            (ema8[-1] >= ema20[-1] >= ema50[-1])
            and (adx >= EMA_STACK_ADX)
            and (c >= ema20[-1])
        )
        stack_bear = (
            (ema8[-1] <= ema20[-1] <= ema50[-1])
            and (adx >= EMA_STACK_ADX)
            and (c <= ema20[-1])
        )
        if HTF_BULL:
            stack_bear = False
        if HTF_BEAR:
            stack_bull = False
        if stack_bull:
            kinds.append("ema_stack_bull")
            reasons.append(
                f"ema_stack_bull: EMA8≥20≥50 & ADX {adx:.1f}≥{EMA_STACK_ADX} & c≥EMA20"
            )
        if stack_bear:
            kinds.append("ema_stack_bear")
            reasons.append(
                f"ema_stack_bear: EMA8≤20≤50 & ADX {adx:.1f}≥{EMA_STACK_ADX} & c≤EMA20"
            )

    # 4) RSI 背离（简化）
    if ENABLE_RSI_DIVERGENCE and len(df_closed) >= DIV_LOOKBACK + 2:
        lows = df_closed["low"].tail(DIV_LOOKBACK + 1).values
        highs = df_closed["high"].tail(DIV_LOOKBACK + 1).values
        rsi_series = ta.RSI(df_closed["close"].astype(float).values, timeperiod=14)
        rsi_prev = (
            float(rsi_series[-(DIV_LOOKBACK + 1)])
            if pd.notna(rsi_series[-(DIV_LOOKBACK + 1)])
            else rsi
        )
        # 多头背离：新低但 RSI 没创新低
        bull_div = (low <= np.min(lows[:-1]) * (1 + DIV_TOL_PCT / 100.0)) and (
            rsi >= rsi_prev * (1 - DIV_TOL_PCT / 100.0)
        )
        # 空头背离：新高但 RSI 没创新高
        bear_div = (h >= np.max(highs[:-1]) * (1 - DIV_TOL_PCT / 100.0)) and (
            rsi <= rsi_prev * (1 + DIV_TOL_PCT / 100.0)
        )
        if HTF_BULL:
            bear_div = False
        if HTF_BEAR:
            bull_div = False
        if bull_div:
            kinds.append("rsi_div_long")
            reasons.append(f"rsi_div_long: price LL vs {DIV_LOOKBACK} 但 RSI 未创新低")
        if bear_div:
            kinds.append("rsi_div_short")
            reasons.append(f"rsi_div_short: price HH vs {DIV_LOOKBACK} 但 RSI 未创新高")

    # 5) 量能峰值反转
    if ENABLE_CLIMAX:
        # 峰值量能 + 长影线 + RSI 极值
        total = max(h - low, 1e-12)
        lower_w = max(min(o, c) - low, 0.0) / total
        upper_w = max(h - max(o, c), 0.0) / total
        if (
            volr_now >= CLIMAX_VOLR
            and lower_w >= 0.55
            and rsi <= CLIMAX_RSI_L
            and not HTF_BEAR
        ):
            kinds.append("climax_bottom")
            reasons.append(
                f"climax_bottom: volr {volr_now:.2f}≥{CLIMAX_VOLR:.1f} & 下影 {lower_w:.2f} & RSI≤{CLIMAX_RSI_L}"
            )
        if (
            volr_now >= CLIMAX_VOLR
            and upper_w >= 0.55
            and rsi >= CLIMAX_RSI_H
            and not HTF_BULL
        ):
            kinds.append("climax_top")
            reasons.append(
                f"climax_top: volr {volr_now:.2f}≥{CLIMAX_VOLR:.1f} & 上影 {upper_w:.2f} & RSI≥{CLIMAX_RSI_H}"
            )

    # 6) 平衡区突破 / 假突破回收
    if ENABLE_EQUILIBRIUM and len(df_closed) >= EQ_WIN + 2:
        eq_h = float(df_closed["high"].tail(EQ_WIN).max())
        eq_l = float(df_closed["low"].tail(EQ_WIN).min())
        # 突破
        break_up = c >= eq_h * (1.0 + EQ_BREAK_BUF)
        break_dn = c <= eq_l * (1.0 - EQ_BREAK_BUF)
        # 假突破回收（当根有明显越界但收回到区间内大部）
        reject_up = (
            (h > eq_h * (1.0 + EQ_BREAK_BUF))
            and (c <= eq_h)
            and (abs(c - o) / max(h - low, 1e-12) >= EQ_REJ_BODY)
        )
        reject_dn = (
            (low < eq_l * (1.0 - EQ_BREAK_BUF))
            and (c >= eq_l)
            and (abs(o - c) / max(h - low, 1e-12) >= EQ_REJ_BODY)
        )
        if HTF_BULL:
            break_dn = reject_up = False
        if HTF_BEAR:
            break_up = reject_dn = False
        if break_up or break_dn:
            kinds.append("equilibrium_break")
            reasons.append(
                f"equilibrium_break: [{eq_l:.6g}, {eq_h:.6g}] with buf {EQ_BREAK_BUF:.3f}"
            )
        if reject_up or reject_dn:
            kinds.append("equilibrium_reject")
            reasons.append(
                f"equilibrium_reject: 假突破后实体≥{EQ_REJ_BODY:.2f} 回收入区间"
            )

    # 没有新增信号
    if not kinds:
        return False, None

    # 选择一个“最有意义”的信号（简单优先级）
    priority = {
        "trend_break_up": 0,
        "trend_break_down": 0,
        "equilibrium_break": 1,
        "ema_stack_bull": 2,
        "ema_stack_bear": 2,
        "volume_shift_long": 3,
        "volume_shift_short": 3,
        "climax_bottom": 4,
        "climax_top": 4,
        "rsi_div_long": 5,
        "rsi_div_short": 5,
    }
    kinds_sorted = sorted(kinds, key=lambda k: priority.get(k, 9))
    kind = kinds_sorted[0]
    side = (
        "long"
        if (
            "_up" in kind
            or "bull" in kind
            or "long" in kind
            or "bottom" in kind
            or "break" in kind
            and HTF_BULL
        )
        else "short"
    )

    # 文本/命令与评分
    tick = get_tick_size(ex, symbol)
    hhv_prev = df_closed["high"].iloc[:-1].tail(BO_WINDOW).max()
    llv_prev = df_closed["low"].iloc[:-1].tail(BO_WINDOW).min()

    tp1_i, tp2_i, tp3_i, sl_i = dynamic_targets(
        symbol="NA", side=side, entry_price=c, df_closed=df_closed, tick=tick
    )
    entry_safe = conservative_entry(
        symbol="NA",
        side=side,
        close_last=c,
        df_closed=df_closed,
        tick=tick,
        hhv_prev=hhv_prev if side == "long" else None,
        llv_prev=llv_prev if side == "short" else None,
    )
    tp1_s, tp2_s, tp3_s, sl_s = tpsl_for_safe_entry(side, entry_safe, tick)
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

    # 趋势文本
    tr2 = trend_scores(df_closed, bars=TREND_LOOKBACK_BARS)
    adx2 = last_adx(df_closed, period=14)
    atrp2 = last_atr_pct(df_closed, period=14)
    trend_text = f"Class={SYMBOL_CLASS.get(symbol, '?')} | ADX≈{adx2:.1f}, ρ={tr2['spearman']:.2f}, net%={tr2['net_pct']:.2f}, ATR%≈{atrp2:.2f} | HTF:{'BULL' if HTF_BULL else ('BEAR' if HTF_BEAR else '—')}"

    # 评分（复用你的权重体系）
    sc = (
        W_VOLR_NOW * max(0.0, min(1.0, (volr_now - 1.0) / 4.0))
        + W_EQ_NOW_USD * max(0.0, min(1.0, eq_now_bar_usd / max(1.0, SCALE_EQ5M_USD)))
        + W_ABS_PCT * max(0.0, min(1.0, abs(pct_now) / max(1e-9, SCALE_ABS_PCT)))
        + W_TREND_ALIGN
        * (
            1.0
            if (HTF_BULL and side == "long") or (HTF_BEAR and side == "short")
            else 0.0
        )
    )
    # 新信号加分（与原体系接近）
    if "trend_break" in kind:
        sc += 0.12
    elif "equilibrium_break" in kind:
        sc += 0.12
    elif "ema_stack" in kind:
        sc += 0.08
    elif "volume_shift" in kind:
        sc += 0.10
    elif "climax" in kind:
        sc += 0.15
    elif "rsi_div" in kind:
        sc += 0.12

    # 方向闸门惩罚与原一致
    if HTF_BULL and side == "short":
        sc = max(0.0, sc - 0.20)
    elif HTF_BEAR and side == "long":
        sc = max(0.0, sc - 0.20)

    # 触发原因（限长）
    reasons = [f"class={SYMBOL_CLASS.get(symbol, '?')}; mode={MODE}"] + reasons
    reasons_show = reasons if PRINT_FULL_REASONS else reasons[:MAX_REASONS_IN_MSG]

    text_core = [
        f"Symbol: <b>{symbol}</b>",
        f"Price: <code>{c:.6g}</code>",
        f"Now {strategy.timeframe_fast}: <b>{pct_now:.2f}%</b> | VolR: <b>{volr_now:.2f}x</b> | EqBar≈<b>${eq_now_bar_usd:,.0f}</b>",
        f"Base vps: <b>{vps_base:.4f}</b> | Trend: {trend_text}",
        "Why: " + " ; ".join(reasons_show),
    ]
    if not SAFE_MODE_ALWAYS:
        text_core.append(f"<code>{cmd_immd}</code>")
    text_core += [
        f"<b>Conservative Entry:</b> <code>{entry_safe}</code> | SL(≈{SAFE_SL_PCT:.1f}%): <code>{tpsl_for_safe_entry(side, entry_safe, tick)[-1]}</code>",
        f"<code>{cmd_safe}</code>",
    ]
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
        "eq_base_bar_usd": float(vps_base * FRAME_SEC[strategy.timeframe_fast] * c),
        "eq_now_bar_usd": float(eq_now_bar_usd),
        "trend_align": bool(
            (HTF_BULL and side == "long") or (HTF_BEAR and side == "short")
        ),
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
