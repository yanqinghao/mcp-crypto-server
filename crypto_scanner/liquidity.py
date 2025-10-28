# liquidity.py
from .exchange import round_to_tick
from .loggingx import dbg


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
