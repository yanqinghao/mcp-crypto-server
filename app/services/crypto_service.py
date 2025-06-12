import ccxt.async_support as ccxt
import asyncio
from typing import List, Optional, Dict, Any
from fastmcp import Context
from datetime import datetime

from config import settings


class DataFetchError(Exception):
    """数据获取异常"""

    pass


class ExchangeManager:
    """交易所管理器，用于管理不同交易所的连接"""

    def __init__(self):
        self._exchanges = {}
        self._default_exchange = None

    def add_exchange(
        self, name: str, exchange_config: Dict[str, Any], is_default: bool = False
    ):
        """添加交易所配置"""
        try:
            exchange_class = getattr(ccxt, name.lower())
            exchange = exchange_class(exchange_config)
            self._exchanges[name] = exchange

            if is_default or self._default_exchange is None:
                self._default_exchange = exchange

        except AttributeError:
            raise DataFetchError(f"Unsupported exchange: {name}")
        except Exception as e:
            raise DataFetchError(f"Failed to initialize exchange {name}: {e}")

    def get_exchange(self, name: Optional[str] = None):
        """获取交易所实例"""
        if name:
            return self._exchanges.get(name)
        return self._default_exchange

    def get_available_exchanges(self) -> List[str]:
        """获取可用的交易所列表"""
        return list(self._exchanges.keys())


# 全局交易所管理器
exchange_manager = ExchangeManager()


def init_default_exchanges():
    """初始化默认交易所配置"""
    try:
        # 初始化一些常用的交易所
        exchanges_config = {
            settings.DEFAULT_EXCHANGE_ID: {
                "apiKey": settings.API_KEY,  # 可选，用于私有API
                "secret": settings.SECRET_KEY,
                "sandbox": settings.SANDBOX_MODE,
                "rateLimit": 1200,
                "enableRateLimit": True,
            }
        }

        # 添加交易所，Binance作为默认
        for name, config in exchanges_config.items():
            try:
                exchange_manager.add_exchange(name, config, name == "binance")
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")

    except Exception as e:
        print(f"Warning: Failed to initialize default exchanges: {e}")


# 初始化默认交易所
init_default_exchanges()


async def fetch_ohlcv_data(
    ctx: Context,
    symbol: str,
    timeframe: str,
    limit: int = 1000,
    exchange_name: Optional[str] = None,
    since: Optional[int] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[List[List[float]]]:
    """
    获取OHLCV数据的通用函数

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号 (例如: 'BTC/USDT')
        timeframe: 时间框架 (例如: '1h', '4h', '1d')
        limit: 获取的K线数量
        exchange_name: 交易所名称，None表示使用默认交易所
        since: 开始时间戳（毫秒）
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）

    Returns:
        OHLCV数据列表，格式为 [[timestamp, open, high, low, close, volume], ...]
        如果失败返回None
    """
    exchange = exchange_manager.get_exchange(exchange_name)
    if not exchange:
        await ctx.error(f"Exchange not available: {exchange_name or 'default'}")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(
                f"Fetching OHLCV data for {symbol} on {timeframe} (attempt {attempt + 1})"
            )

            # 检查交易所是否支持该交易对
            if symbol not in exchange.symbols:
                # 尝试加载市场信息
                try:
                    await exchange.load_markets()
                    if symbol not in exchange.symbols:
                        await ctx.error(f"Symbol {symbol} not found on exchange")
                        return None
                except Exception as e:
                    await ctx.error(f"Failed to load markets: {e}")
                    return None

            # 获取OHLCV数据
            ohlcv_data = await exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=since, limit=limit
            )

            if not ohlcv_data:
                await ctx.warning(f"No OHLCV data returned for {symbol}")
                return None

            # 数据质量检查
            valid_data = []
            for candle in ohlcv_data:
                if len(candle) >= 6 and all(
                    isinstance(x, (int, float)) for x in candle[:6]
                ):
                    # 检查数据合理性
                    timestamp, open_price, high, low, close, volume = candle[:6]
                    if (
                        high >= max(open_price, close)
                        and low <= min(open_price, close)
                        and all(x > 0 for x in [open_price, high, low, close])
                        and volume >= 0
                    ):
                        valid_data.append(candle)

            if len(valid_data) < limit * 0.8:  # 如果有效数据太少，发出警告
                await ctx.warning(
                    f"Only {len(valid_data)} valid candles out of {len(ohlcv_data)} for {symbol}"
                )

            await ctx.info(
                f"Successfully fetched {len(valid_data)} OHLCV candles for {symbol}"
            )
            return valid_data

        except ccxt.NetworkError as e:
            await ctx.warning(
                f"Network error fetching {symbol} data (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2**attempt))  # 指数退避
            else:
                await ctx.error(f"Max retries exceeded for {symbol}")
                return None

        except ccxt.ExchangeError as e:
            await ctx.error(f"Exchange error fetching {symbol} data: {e}")
            return None

        except Exception as e:
            await ctx.error(f"Unexpected error fetching {symbol} data: {e}")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            else:
                return None

    return None


async def fetch_ticker_data(
    ctx: Context, symbol: str, exchange_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    获取ticker数据

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号
        exchange_name: 交易所名称

    Returns:
        ticker数据字典或None
    """
    exchange = exchange_manager.get_exchange(exchange_name)
    if not exchange:
        await ctx.error(f"Exchange not available: {exchange_name or 'default'}")
        return None

    try:
        ticker = await exchange.fetch_ticker(symbol)
        await ctx.info(f"Successfully fetched ticker for {symbol}")
        return ticker

    except Exception as e:
        await ctx.error(f"Error fetching ticker for {symbol}: {e}")
        return None


async def fetch_order_book(
    ctx: Context, symbol: str, limit: int = 100, exchange_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    获取订单簿数据

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号
        limit: 订单簿深度
        exchange_name: 交易所名称

    Returns:
        订单簿数据字典或None
    """
    exchange = exchange_manager.get_exchange(exchange_name)
    if not exchange:
        await ctx.error(f"Exchange not available: {exchange_name or 'default'}")
        return None

    try:
        order_book = await exchange.fetch_order_book(symbol, limit)
        await ctx.info(f"Successfully fetched order book for {symbol}")
        return order_book

    except Exception as e:
        await ctx.error(f"Error fetching order book for {symbol}: {e}")
        return None


async def fetch_trades(
    ctx: Context,
    symbol: str,
    limit: int = 100,
    since: Optional[int] = None,
    exchange_name: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取最近交易数据

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号
        limit: 交易记录数量
        since: 开始时间戳
        exchange_name: 交易所名称

    Returns:
        交易数据列表或None
    """
    exchange = exchange_manager.get_exchange(exchange_name)
    if not exchange:
        await ctx.error(f"Exchange not available: {exchange_name or 'default'}")
        return None

    try:
        trades = await exchange.fetch_trades(symbol, since, limit)
        await ctx.info(f"Successfully fetched {len(trades)} trades for {symbol}")
        return trades

    except Exception as e:
        await ctx.error(f"Error fetching trades for {symbol}: {e}")
        return None


def get_timeframe_duration_ms(timeframe: str) -> int:
    """
    获取时间框架对应的毫秒数

    Args:
        timeframe: 时间框架字符串

    Returns:
        时间框架对应的毫秒数
    """
    timeframe_map = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }
    return timeframe_map.get(timeframe, 60 * 60 * 1000)  # 默认1小时


async def fetch_historical_ohlcv(
    ctx: Context,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    exchange_name: Optional[str] = None,
) -> Optional[List[List[float]]]:
    """
    获取历史OHLCV数据

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号
        timeframe: 时间框架
        start_date: 开始日期
        end_date: 结束日期
        exchange_name: 交易所名称

    Returns:
        历史OHLCV数据列表或None
    """
    exchange = exchange_manager.get_exchange(exchange_name)
    if not exchange:
        await ctx.error(f"Exchange not available: {exchange_name or 'default'}")
        return None

    try:
        # 计算时间范围
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        timeframe_ms = get_timeframe_duration_ms(timeframe)

        all_data = []
        current_timestamp = start_timestamp

        while current_timestamp < end_timestamp:
            # 计算这次请求的限制
            remaining_time = end_timestamp - current_timestamp
            limit = min(1000, remaining_time // timeframe_ms + 1)

            if limit <= 0:
                break

            # 获取数据
            data = await fetch_ohlcv_data(
                ctx, symbol, timeframe, limit, exchange_name, current_timestamp
            )

            if not data:
                break

            # 过滤重复数据并添加到结果中
            for candle in data:
                if candle[0] <= end_timestamp:
                    all_data.append(candle)

            # 更新时间戳
            if data:
                current_timestamp = int(data[-1][0]) + timeframe_ms
            else:
                break

            # 避免请求过快
            await asyncio.sleep(0.1)

        # 按时间戳排序并去重
        all_data.sort(key=lambda x: x[0])
        unique_data = []
        seen_timestamps = set()

        for candle in all_data:
            if candle[0] not in seen_timestamps:
                unique_data.append(candle)
                seen_timestamps.add(candle[0])

        await ctx.info(
            f"Successfully fetched {len(unique_data)} historical candles for {symbol}"
        )
        return unique_data

    except Exception as e:
        await ctx.error(f"Error fetching historical data for {symbol}: {e}")
        return None


# 向后兼容的函数别名
async def _fetch_and_prepare_ohlcv_single_series(
    ctx: Context,
    symbol: str,
    timeframe: str,
    required_candles: int,
    series_index: int = 4,
    exchange_name: Optional[str] = None,
) -> Optional[List[float]]:
    """
    向后兼容的单序列数据获取函数
    """
    ohlcv_data = await fetch_ohlcv_data(
        ctx, symbol, timeframe, required_candles, exchange_name
    )
    if not ohlcv_data:
        return None

    return [candle[series_index] for candle in ohlcv_data]


async def _fetch_and_prepare_multi_data_ohlcv(
    ctx: Context,
    symbol: str,
    timeframe: str,
    required_candles_primary: int,
    required_series_map: Dict[str, int],
    exchange_name: Optional[str] = None,
) -> Optional[Dict[str, List[float]]]:
    """
    向后兼容的多序列数据获取函数
    """
    ohlcv_data = await fetch_ohlcv_data(
        ctx, symbol, timeframe, required_candles_primary, exchange_name
    )
    if not ohlcv_data:
        return None

    result = {}
    for series_name, series_index in required_series_map.items():
        result[series_name] = [candle[series_index] for candle in ohlcv_data]

    return result


async def search_crypto_by_name(
    ctx: Context, base_currency: str, exchange_name: Optional[str] = None
) -> Optional[List[str]]:
    """根据币种名称搜索交易对"""
    exchange = exchange_manager.get_exchange(exchange_name)
    if not exchange:
        return None

    try:
        await exchange.load_markets()

        results = []
        for symbol, market in exchange.markets.items():
            if market["base"].lower() == base_currency.lower():
                results.append(symbol)

        return results

    except Exception as e:
        await ctx.error(f"Error searching crypto: {e}")
        return None
