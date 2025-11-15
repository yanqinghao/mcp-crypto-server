# liquidity.py
from .exchange import round_to_tick
from .loggingx import dbg


def _fetch_funding_rate_safe(ex, symbol):
    """
    尝试通过交易所对象获取资金费率（兼容 ccxt 风格）：
    - 优先使用 ex.fetch_funding_rate(symbol)
    - 失败则返回 (None, None, None)
    返回:
        rate: float|None  例如 0.0001 ≈ 0.01%
        ts:   int|None    时间戳（毫秒）
        raw:  dict|None   原始返回（调试用）
    """
    rate = None
    ts = None
    raw = None
    try:
        if hasattr(ex, "fetch_funding_rate"):
            fr = ex.fetch_funding_rate(symbol)
            raw = fr or {}
            # 不同交易所字段名可能不一样，尽量兼容几种常见写法
            v = (
                raw.get("fundingRate")
                or raw.get("fundingRateLast")
                or raw.get("fundingRate8h")
            )
            if v is not None:
                rate = float(v)
            ts = raw.get("timestamp") or raw.get("info", {}).get("fundingTime")
            dbg(f"[DETECT] {symbol}: funding_rate={rate}, ts={ts}")
        else:
            dbg(f"[DETECT] {symbol}: ex has no fetch_funding_rate")
    except Exception as e:
        dbg(f"[DETECT] {symbol}: fetch_funding_rate failed: {e}")
    return rate, ts, raw


def get_day_stats(ex, symbol, tick):
    """
    返回: (day_high, day_low, pct24, last_price)
    - pct24 优先用 ticker['percentage']，若缺失尝试用 (last-open)/open 计算
    - 所有价格按 tick 对齐
    """
    try:
        t = ex.fetch_ticker(symbol) or {}
        last = t.get("last") or t.get("close")
        hi = t.get("high")
        lo = t.get("low")
        pct = t.get("percentage")

        # 尝试用 open 计算 pct24
        if pct is None:
            op = t.get("open")
            try:
                if op is not None and last is not None and float(op) > 0:
                    pct = (float(last) - float(op)) / float(op) * 100.0
            except Exception:
                pct = None

        # 标准化 / 对齐 tick
        try:
            last = round_to_tick(float(last), tick) if last is not None else None
        except Exception:
            last = None
        try:
            hi = round_to_tick(float(hi), tick) if hi is not None else None
        except Exception:
            hi = None
        try:
            lo = round_to_tick(float(lo), tick) if lo is not None else None
        except Exception:
            lo = None
        try:
            pct = float(pct) if pct is not None else None
        except Exception:
            pct = None

        return hi, lo, pct, last
    except Exception as e:
        dbg(f"get_day_stats error {symbol}: {e}")
        return None, None, None, None
