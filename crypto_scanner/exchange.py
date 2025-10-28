import math
import pandas as pd
import ccxt
import time
from typing import Tuple, Dict
from .config import EXCHANGE_ID, MARKET_TYPE, QUOTE
from .loggingx import dbg


def build_exchange():
    dbg(f"Connecting {EXCHANGE_ID} (type={MARKET_TYPE})")
    ex = getattr(ccxt, EXCHANGE_ID)(
        {"enableRateLimit": True, "options": {"defaultType": MARKET_TYPE}}
    )
    ex.load_markets()
    dbg(f"Loaded {len(ex.markets)} markets")
    return ex


def market_is_usdt_swap(ex, symbol: str) -> bool:
    m = ex.markets.get(symbol)
    return bool(
        m
        and m.get("swap", False)
        and m.get("linear", False)
        and m.get("quote") == QUOTE
    )


def is_good_symbol(ex, symbol: str) -> bool:
    return symbol.endswith(":USDT") and market_is_usdt_swap(ex, symbol)


def get_tick_size(ex, symbol: str) -> float:
    m = ex.markets.get(symbol, {})
    info = m.get("info") or {}
    for f in info.get("filters") or []:
        if f.get("filterType") == "PRICE_FILTER":
            tsz = float(f.get("tickSize", 0) or 0)
            if tsz > 0:
                return tsz
    prec = (m.get("precision") or {}).get("price")
    if isinstance(prec, (int, float)) and prec >= 0:
        return 10 ** (-int(prec))
    return 1e-8


def round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return x
    decimals = int(abs(math.log10(tick))) if tick < 1 else 0
    return round(round(x / tick) * tick, decimals)


# (symbol, timeframe, limit) -> (ts, df)
_OHLCV_CACHE: Dict[Tuple[str, str, int], Tuple[float, "pd.DataFrame"]] = {}
# 默认 TTL：5 秒；如果你更激进，可改为 2~3
OHLCV_CACHE_TTL_SEC = 5


def _get_cached_ohlcv(symbol: str, timeframe: str, limit: int):
    key = (symbol, timeframe, int(limit or 0))
    item = _OHLCV_CACHE.get(key)
    if not item:
        return None
    ts, df = item
    if (time.time() - ts) <= OHLCV_CACHE_TTL_SEC:
        return df
    # 过期清理（惰性）
    try:
        del _OHLCV_CACHE[key]
    except Exception:
        pass
    return None


def _set_cached_ohlcv(symbol: str, timeframe: str, limit: int, df):
    key = (symbol, timeframe, int(limit or 0))
    _OHLCV_CACHE[key] = (time.time(), df)


# ====== 你的原有 fetch_ohlcv_df 上加缓存 ======
def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int = 200):
    """
    包装 ccxt.fetch_ohlcv 为 DataFrame，增加 5s 内存缓存以减少重复请求。
    缓存键： (symbol, timeframe, limit)
    """

    # 1) 命中缓存直接返回（深拷贝避免被上游修改）
    cached = _get_cached_ohlcv(symbol, timeframe, limit)
    if cached is not None:
        return cached.copy()

    # 2) 真正去拉取
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    # raw: [[ts, open, high, low, close, volume], ...]
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])

    # 3) 写缓存（浅存，返回前 copy）
    _set_cached_ohlcv(symbol, timeframe, limit, df)
    return df.copy()
