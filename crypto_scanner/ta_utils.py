# -*- coding: utf-8 -*-
import pandas as pd
import talib as ta

from .config import TIMEFRAME_HOURLY


def rank_corr(x, y):
    xr = pd.Series(x).rank(method="average")
    yr = pd.Series(y).rank(method="average")
    if xr.std() == 0 or yr.std() == 0:
        return 0.0
    return float(((xr - xr.mean()) * (yr - yr.mean())).mean() / (xr.std() * yr.std()))


def trend_scores(df_closed, bars=12):
    if len(df_closed) < (bars + 1):
        return {"spearman": 0.0, "net_pct": 0.0, "ma_up": False, "ma_down": False}
    w = df_closed.tail(bars).copy()
    closes = w["close"].astype(float).tolist()
    t = list(range(bars))
    spearman = rank_corr(t, closes)
    net_pct = (closes[-1] - closes[0]) / closes[0] * 100.0 if closes[0] else 0.0
    s = pd.Series(closes)
    sma_short = s.rolling(4, min_periods=4).mean().iloc[-1]
    sma_long = s.rolling(12, min_periods=12).mean().iloc[-1]
    ma_up = pd.notna(sma_short) and pd.notna(sma_long) and (sma_short >= sma_long)
    ma_down = pd.notna(sma_short) and pd.notna(sma_long) and (sma_short <= sma_long)
    return {
        "spearman": float(spearman),
        "net_pct": float(net_pct),
        "ma_up": bool(ma_up),
        "ma_down": bool(ma_down),
    }


def compute_atr(df_closed: pd.DataFrame, period: int = 14):
    if len(df_closed) < period + 1:
        return None
    h = df_closed["high"].astype(float).values
    low = df_closed["low"].astype(float).values
    c = df_closed["close"].astype(float).values
    atr = ta.ATR(h, low, c, timeperiod=period)
    val = atr[-1]
    return float(val) if pd.notna(val) else None


def last_rsi(df_closed: pd.DataFrame, period: int = 14) -> float:
    close = df_closed["close"].astype(float).values
    if close.size < period + 1:
        return 50.0
    out = ta.RSI(close, timeperiod=period)
    v = out[-1]
    return float(v) if pd.notna(v) else 50.0


def last_willr(df_closed: pd.DataFrame, period: int = 14) -> float:
    high = df_closed["high"].astype(float).values
    low = df_closed["low"].astype(float).values
    close = df_closed["close"].astype(float).values
    if close.size < period + 1:
        return -50.0
    out = ta.WILLR(high, low, close, timeperiod=period)
    v = out[-1]
    return float(v) if pd.notna(v) else -50.0


def last_adx(df_closed: pd.DataFrame, period: int = 14) -> float:
    high = df_closed["high"].astype(float).values
    low = df_closed["low"].astype(float).values
    close = df_closed["close"].astype(float).values
    if close.size < period + 1:
        return 0.0
    out = ta.ADX(high, low, close, timeperiod=period)
    v = out[-1]
    return float(v) if pd.notna(v) else 0.0


def last_atr_pct(df_closed: pd.DataFrame, period: int = 14) -> float:
    atr = compute_atr(df_closed, period=period)
    if not atr:
        return 0.0
    last_close = float(df_closed["close"].iloc[-1])
    return float(atr / last_close * 100.0) if last_close > 0 else 0.0


def recent_high_distance_pct(
    df_closed: pd.DataFrame, lookback: int, price_now: float
) -> float:
    if len(df_closed) < lookback or price_now <= 0:
        return 0.0
    hhv = float(df_closed["high"].tail(lookback).max())
    return max(0.0, (hhv - price_now) / price_now * 100.0)


def recent_low_distance_pct(
    df_closed: pd.DataFrame, lookback: int, price_now: float
) -> float:
    if len(df_closed) < lookback or price_now <= 0:
        return 0.0
    llv = float(df_closed["low"].tail(lookback).min())
    return max(0.0, (price_now - llv) / price_now * 100.0)


def lower_wick_ratio(o, h, low, c) -> float:
    total = max(h - low, 1e-12)
    lower = max(min(o, c) - low, 0.0)
    return lower / total


def upper_wick_ratio(o, h, low, c) -> float:
    total = max(h - low, 1e-12)
    upper = max(h - max(o, c), 0.0)
    return upper / total


# ====== 补充缺失的通用函数 ======


def percent_change(a: float, b: float) -> float:
    """安全的百分比变动计算：(a - b) / b * 100."""
    try:
        return (a - b) / b * 100.0 if b else 0.0
    except Exception:
        return 0.0


def compute_atr_pct_on_1h(
    ex, symbol: str, lookback: int = 48, period: int = 14
) -> float:
    """
    使用交易所实例 ex 直接抓 1h K 计算 ATR%，避免跨模块依赖。
    返回 ATR(14)/close * 100（最后一根）。
    """
    try:
        # 直接 fetch_ohlcv，避免依赖 exchange.py
        ohlcv = ex.fetch_ohlcv(
            symbol, timeframe=TIMEFRAME_HOURLY, limit=max(lookback, period + 2)
        )
        if not ohlcv or len(ohlcv) < period + 2:
            return 0.0
        df1h = pd.DataFrame(
            ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        atr = compute_atr(df1h, period=period)
        last_close = float(df1h["close"].iloc[-1])
        return float(atr / last_close * 100.0) if (atr and last_close > 0) else 0.0
    except Exception:
        return 0.0
