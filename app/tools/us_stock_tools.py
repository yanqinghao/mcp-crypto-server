import numpy as np
import talib
import json
from typing import Dict, Optional, List, Any
from datetime import datetime

from config import settings
from models.analysis import (
    SmaInput,
    SmaOutput,
    RsiInput,
    RsiOutput,
    MacdInput,
    MacdOutput,
    BbandsInput,
    BbandsOutput,
    AtrInput,
    AtrOutput,
    ComprehensiveAnalysisInput,
    ComprehensiveAnalysisOutput,
)
from models.market_data import (
    CandlesInput,
    CandlesOutput,
    PriceInput,
    PriceOutput,
    TickerInput,
    TickerOutput,
    OHLCVCandle,
)
from fastmcp import FastMCP, Context

# 导入更新的yfinance服务函数
from services.yfinance_service import (
    fetch_us_stock_hist_data,
    fetch_us_stock_financials,
    search_us_stock_by_name,
    fetch_dividend_history,
    fetch_market_indices_data,
)

mcp = FastMCP()

# ==================== 数据获取辅助函数 ====================


async def _fetch_single_series_data(
    ctx: Context,
    symbol: str,
    period: str,
    interval: str,
    required_candles: int,
    series_type: str = "close",
) -> Optional[np.ndarray]:
    """
    获取单个数据序列

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        period: 时间周期
        interval: 数据间隔
        required_candles: 需要的K线数量
        series_type: 数据类型 (open, high, low, close, adj_close, volume)

    Returns:
        numpy数组或None
    """
    try:
        # 获取美股历史数据
        stock_data = await fetch_us_stock_hist_data(ctx, symbol, period, interval)

        if not stock_data or len(stock_data) < required_candles:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(stock_data) if stock_data else 0}, need {required_candles}"
            )
            return None

        # 提取指定类型的数据
        valid_fields = ["open", "high", "low", "close", "adj_close", "volume"]
        if series_type not in valid_fields:
            await ctx.error(f"Invalid series type: {series_type}")
            return None

        data_series = np.array([item[series_type] for item in stock_data])

        # 确保数据质量
        if np.any(np.isnan(data_series)) or np.any(np.isinf(data_series)):
            await ctx.warning(
                f"Found NaN or Inf values in {series_type} data for {symbol}"
            )
            # 移除NaN和Inf值
            data_series = data_series[~(np.isnan(data_series) | np.isinf(data_series))]

        return data_series

    except Exception as e:
        await ctx.error(f"Error fetching {series_type} data for {symbol}: {e}")
        return None


async def _fetch_multi_series_data(
    ctx: Context,
    symbol: str,
    period: str,
    interval: str,
    required_candles: int,
    series_types: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    """
    获取多个数据序列

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        period: 时间周期
        interval: 数据间隔
        required_candles: 需要的K线数量
        series_types: 需要的数据类型列表

    Returns:
        包含各数据序列的字典或None
    """
    try:
        # 获取美股历史数据
        stock_data = await fetch_us_stock_hist_data(ctx, symbol, period, interval)

        if not stock_data or len(stock_data) < required_candles:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(stock_data) if stock_data else 0}, need {required_candles}"
            )
            return None

        valid_fields = ["open", "high", "low", "close", "adj_close", "volume"]
        result = {}

        for series_type in series_types:
            if series_type not in valid_fields:
                await ctx.error(f"Invalid series type: {series_type}")
                return None

            data_series = np.array([item[series_type] for item in stock_data])

            # 确保数据质量
            if np.any(np.isnan(data_series)) or np.any(np.isinf(data_series)):
                await ctx.warning(
                    f"Found NaN or Inf values in {series_type} data for {symbol}"
                )
                data_series = data_series[
                    ~(np.isnan(data_series) | np.isinf(data_series))
                ]

            result[series_type] = data_series

        # 确保所有序列长度一致
        min_length = min(len(series) for series in result.values())
        if min_length < required_candles:
            await ctx.error(
                f"Insufficient clean data after filtering: {min_length} < {required_candles}"
            )
            return None

        # 截取到相同长度
        for key in result:
            result[key] = result[key][-min_length:]

        return result

    except Exception as e:
        await ctx.error(f"Error fetching multi-series data for {symbol}: {e}")
        return None


def _extract_valid_values(values: np.ndarray, history_len: int) -> List[float]:
    """
    从计算结果中提取有效值

    Args:
        values: TA-Lib计算结果
        history_len: 需要的历史数据长度

    Returns:
        有效值列表
    """
    # 移除NaN值
    valid_values = values[~np.isnan(values)]

    # 返回最后history_len个值
    if len(valid_values) >= history_len:
        return [float(x) for x in valid_values[-history_len:]]
    else:
        return [float(x) for x in valid_values]


# ==================== 美股数据获取工具 ====================


@mcp.tool()
async def get_us_stock_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    获取美股K线数据

    Args:
        inputs.symbol: 股票代码 (如: AAPL, MSFT, TSLA)
        inputs.timeframe: 时间框架 (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M)
        inputs.limit: 数据条数限制
        inputs.period: 时间周期 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        inputs.prepost: 是否包含盘前盘后数据
    """
    await ctx.info(
        f"Fetching US stock candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        # 转换时间框架为yfinance格式
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo",
        }
        interval = interval_map.get(inputs.timeframe.value, "1d")

        # 获取时间周期
        period = getattr(inputs, "period", "1y") or "1y"

        # 使用现有的yfinance服务获取美股数据
        stock_data = await fetch_us_stock_hist_data(
            ctx=ctx, symbol=inputs.symbol, period=period, interval=interval
        )

        if stock_data:
            # 限制返回的数据量
            limited_data = stock_data[-inputs.limit :] if inputs.limit else stock_data

            candles = []
            for item in limited_data:
                # 转换为OHLCVCandle格式
                date_str = item["date"]
                if isinstance(date_str, str):
                    # 尝试不同的日期格式
                    try:
                        timestamp = int(
                            datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000
                        )
                    except ValueError:
                        timestamp = int(
                            datetime.fromisoformat(
                                date_str.replace("Z", "+00:00")
                            ).timestamp()
                            * 1000
                        )
                else:
                    timestamp = int(date_str.timestamp() * 1000)

                candle = OHLCVCandle(
                    timestamp=timestamp,
                    open=item["open"],
                    high=item["high"],
                    low=item["low"],
                    close=item["close"],
                    volume=item["volume"],
                )
                candles.append(candle)

            await ctx.info(
                f"Successfully fetched {len(candles)} US stock candles for {inputs.symbol}"
            )

            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe.value,
                candles=candles,
                count=len(candles),
            )
        else:
            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe.value,
                error=f"No US stock candle data available for {inputs.symbol}",
            )

    except Exception as e:
        await ctx.error(f"Error fetching US stock candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"US stock data error: {str(e)}",
        )


@mcp.tool()
async def get_us_stock_price(ctx: Context, inputs: PriceInput) -> PriceOutput:
    """
    获取美股当前价格

    Args:
        inputs.symbol: 股票代码 (如: AAPL)
    """
    await ctx.info(f"Fetching US stock current price for {inputs.symbol}")

    try:
        # 获取最新的日线数据作为当前价格
        stock_data = await fetch_us_stock_hist_data(ctx, inputs.symbol, "1d", "1d")

        if stock_data and len(stock_data) > 0:
            latest = stock_data[-1]
            date_str = latest["date"]

            if isinstance(date_str, str):
                timestamp = int(
                    datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000
                )
            else:
                timestamp = int(date_str.timestamp() * 1000)

            await ctx.info(
                f"US stock current price for {inputs.symbol}: ${latest['close']:.2f}"
            )
            return PriceOutput(
                symbol=inputs.symbol,
                price=latest["close"],
                timestamp=timestamp,
            )
        else:
            return PriceOutput(
                symbol=inputs.symbol, error="US stock price data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching US stock price for {inputs.symbol}: {e}")
        return PriceOutput(
            symbol=inputs.symbol, error=f"US stock price error: {str(e)}"
        )


# @mcp.tool()
# async def get_us_stock_info(ctx: Context, symbol: str) -> dict:
#     """
#     获取美股详细信息

#     Args:
#         symbol: 股票代码 (如: AAPL)
#     """
#     await ctx.info(f"Fetching US stock info for {symbol}")

#     try:
#         stock_info = await fetch_us_stock_info(ctx, symbol)

#         if stock_info:
#             return {"success": True, "symbol": symbol, "info": stock_info}
#         else:
#             return {
#                 "success": False,
#                 "symbol": symbol,
#                 "error": "US stock info not available",
#             }

#     except Exception as e:
#         await ctx.error(f"Error fetching US stock info: {e}")
#         return {"success": False, "symbol": symbol, "error": str(e)}


@mcp.tool()
async def search_us_stock_symbols(ctx: Context, query: str, limit: int = 10) -> dict:
    """
    搜索美股股票代码（使用更新的API）

    Args:
        query: 搜索关键词 (如: apple, AAPL, microsoft)
        limit: 结果数量限制
    """
    await ctx.info(f"Searching US stock symbols for '{query}'")

    try:
        # 使用更新的搜索API
        results = await search_us_stock_by_name(
            ctx, query, max_results=limit, include_news=False
        )

        if results and results.get("quotes"):
            quotes = results["quotes"]
            return {
                "success": True,
                "query": query,
                "market_type": "us_stock",
                "results": quotes,
                "count": len(quotes),
            }
        else:
            return {
                "success": False,
                "query": query,
                "market_type": "us_stock",
                "error": "No matching stocks found",
            }

    except Exception as e:
        await ctx.error(f"Error searching US stock symbols: {e}")
        return {
            "success": False,
            "query": query,
            "market_type": "us_stock",
            "error": str(e),
        }


# @mcp.tool()
# async def lookup_us_securities(
#     ctx: Context,
#     query: str,
#     security_type: str = "stock",
#     count: int = 20
# ) -> dict:
#     """
#     按类型查找美股证券（新增工具）

#     Args:
#         query: 查询关键词
#         security_type: 证券类型 (stock, etf, mutualfund, index, etc.)
#         count: 返回结果数量
#     """
#     await ctx.info(f"Looking up {security_type} securities for '{query}'")

#     try:
#         results = await lookup_us_stock_symbol(ctx, query, security_type, count)

#         if results:
#             return {
#                 "success": True,
#                 "query": query,
#                 "security_type": security_type,
#                 "results": results,
#                 "count": len(results),
#             }
#         else:
#             return {
#                 "success": False,
#                 "query": query,
#                 "security_type": security_type,
#                 "error": f"No {security_type} securities found",
#             }

#     except Exception as e:
#         await ctx.error(f"Error looking up {security_type} securities: {e}")
#         return {
#             "success": False,
#             "query": query,
#             "security_type": security_type,
#             "error": str(e),
#         }


# @mcp.tool()
# async def validate_us_stock_symbol_tool(ctx: Context, symbol: str) -> dict:
#     """
#     验证美股代码是否有效（新增工具）

#     Args:
#         symbol: 股票代码
#     """
#     await ctx.info(f"Validating US stock symbol: {symbol}")

#     try:
#         result = await validate_us_stock_symbol(ctx, symbol)
#         return result if result else {"symbol": symbol, "valid": False}

#     except Exception as e:
#         await ctx.error(f"Error validating symbol {symbol}: {e}")
#         return {"symbol": symbol, "valid": False, "error": str(e)}


@mcp.tool()
async def get_us_stock_ticker(ctx: Context, inputs: TickerInput) -> TickerOutput:
    """
    获取美股详细行情数据

    Args:
        inputs.symbol: 股票代码
    """
    await ctx.info(f"Fetching US stock ticker for {inputs.symbol}")

    try:
        # 获取最近几天的数据来计算涨跌
        stock_data = await fetch_us_stock_hist_data(ctx, inputs.symbol, "5d", "1d")

        if stock_data and len(stock_data) > 0:
            latest = stock_data[-1]
            previous = stock_data[-2] if len(stock_data) > 1 else latest

            current_price = latest["close"]
            previous_price = previous["close"]

            change = current_price - previous_price
            percentage = (change / previous_price * 100) if previous_price != 0 else 0

            date_str = latest["date"]
            if isinstance(date_str, str):
                timestamp = int(
                    datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000
                )
            else:
                timestamp = int(date_str.timestamp() * 1000)

            await ctx.info(f"Successfully fetched US stock ticker for {inputs.symbol}")
            return TickerOutput(
                symbol=inputs.symbol,
                last=current_price,
                open=latest["open"],
                high=latest["high"],
                low=latest["low"],
                close=current_price,
                volume=latest["volume"],
                change=change,
                percentage=percentage,
                timestamp=timestamp,
            )
        else:
            return TickerOutput(
                symbol=inputs.symbol, error="US stock ticker data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching US stock ticker for {inputs.symbol}: {e}")
        return TickerOutput(
            symbol=inputs.symbol, error=f"US stock ticker error: {str(e)}"
        )


@mcp.tool()
async def get_us_stock_financials(
    ctx: Context, symbol: str, statement_type: str = "income", quarterly: bool = False
) -> dict:
    """
    获取美股财务数据

    Args:
        symbol: 股票代码
        statement_type: 财务报表类型 (income, balance, cashflow)
        quarterly: 是否获取季度数据
    """
    await ctx.info(f"Fetching US stock financials for {symbol}")

    try:
        financials = await fetch_us_stock_financials(
            ctx, symbol, statement_type, quarterly
        )

        if financials:
            return {
                "success": True,
                "symbol": symbol,
                "statement_type": statement_type,
                "quarterly": quarterly,
                "data": financials,
            }
        else:
            return {
                "success": False,
                "symbol": symbol,
                "error": f"No {statement_type} data available",
            }

    except Exception as e:
        await ctx.error(f"Error fetching US stock financials: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


# @mcp.tool()
# async def get_us_stock_options(
#     ctx: Context, symbol: str, expiration_date: Optional[str] = None
# ) -> dict:
#     """
#     获取美股期权数据

#     Args:
#         symbol: 股票代码
#         expiration_date: 到期日期 (可选)
#     """
#     await ctx.info(f"Fetching US stock options for {symbol}")

#     try:
#         options = await fetch_options_data(ctx, symbol, expiration_date)

#         if options:
#             return {"success": True, "symbol": symbol, "options": options}
#         else:
#             return {
#                 "success": False,
#                 "symbol": symbol,
#                 "error": "No options data available",
#             }

#     except Exception as e:
#         await ctx.error(f"Error fetching US stock options: {e}")
#         return {"success": False, "symbol": symbol, "error": str(e)}


# ==================== 新增：市场数据工具 ====================


@mcp.tool()
async def get_market_indices(
    ctx: Context,
    indices: Optional[List[str]] = None,
    period: str = "1mo",
    interval: str = "1d",
) -> dict:
    """
    获取美股市场指数数据（新增工具）

    Args:
        indices: 指数代码列表 (默认: S&P500, Dow Jones, NASDAQ)
        period: 数据周期
        interval: 数据间隔
    """
    if indices is None:
        indices = ["^GSPC", "^DJI", "^IXIC"]  # S&P500, Dow Jones, NASDAQ

    await ctx.info(f"Fetching market indices data for {indices}")

    try:
        results = await fetch_market_indices_data(ctx, indices, period, interval)

        if results:
            return {
                "success": True,
                "indices": list(results.keys()),
                "period": period,
                "interval": interval,
                "data": results,
                "count": len(results),
            }
        else:
            return {
                "success": False,
                "error": "No market indices data available",
            }

    except Exception as e:
        await ctx.error(f"Error fetching market indices: {e}")
        return {"success": False, "error": str(e)}


# @mcp.tool()
# async def get_realtime_stock_data(
#     ctx: Context, symbols: List[str]
# ) -> dict:
#     """
#     获取多只股票的实时数据（新增工具）

#     Args:
#         symbols: 股票代码列表
#     """
#     await ctx.info(f"Fetching real-time data for {len(symbols)} stocks")

#     try:
#         results = await fetch_us_stock_realtime_data(ctx, symbols)

#         if results:
#             return {
#                 "success": True,
#                 "symbols": list(results.keys()),
#                 "data": results,
#                 "count": len(results),
#                 "timestamp": datetime.now().isoformat(),
#             }
#         else:
#             return {
#                 "success": False,
#                 "error": "No real-time data available",
#             }

#     except Exception as e:
#         await ctx.error(f"Error fetching real-time data: {e}")
#         return {"success": False, "error": str(e)}


# @mcp.tool()
# async def get_analyst_recommendations(ctx: Context, symbol: str) -> dict:
#     """
#     获取分析师评级和推荐（新增工具）

#     Args:
#         symbol: 股票代码
#     """
#     await ctx.info(f"Fetching analyst recommendations for {symbol}")

#     try:
#         results = await fetch_analyst_recommendations(ctx, symbol)

#         if results:
#             return {
#                 "success": True,
#                 "symbol": symbol,
#                 "recommendations": results.get("recommendations", []),
#                 "price_targets": results.get("price_targets", {}),
#             }
#         else:
#             return {
#                 "success": False,
#                 "symbol": symbol,
#                 "error": "No analyst recommendations available",
#             }

#     except Exception as e:
#         await ctx.error(f"Error fetching analyst recommendations: {e}")
#         return {"success": False, "symbol": symbol, "error": str(e)}


@mcp.tool()
async def get_dividend_history(ctx: Context, symbol: str, period: str = "5y") -> dict:
    """
    获取股票分红历史（新增工具）

    Args:
        symbol: 股票代码
        period: 数据周期
    """
    await ctx.info(f"Fetching dividend history for {symbol}")

    try:
        results = await fetch_dividend_history(ctx, symbol, period)

        if results:
            return {
                "success": True,
                "symbol": symbol,
                "period": period,
                "dividends": results,
                "count": len(results),
            }
        else:
            return {
                "success": False,
                "symbol": symbol,
                "error": "No dividend history available",
            }

    except Exception as e:
        await ctx.error(f"Error fetching dividend history: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


# ==================== 新增：更多技术指标工具 ====================


@mcp.tool()
async def calculate_us_stock_ema(
    ctx: Context,
    symbol: str,
    period: int = 20,
    timeframe: str = "1d",
    history_len: int = 50,
) -> dict:
    """
    计算美股指数移动平均线 (EMA)

    Args:
        symbol: 股票代码
        period: EMA周期
        timeframe: 时间框架
        history_len: 历史数据长度
    """
    await ctx.info(f"Calculating EMA for {symbol}, period: {period}")

    try:
        required_candles = period + history_len
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(timeframe, "1d")
        data_period = "2y" if required_candles > 252 else "1y"

        close_prices = await _fetch_single_series_data(
            ctx, symbol, data_period, interval, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Insufficient data for EMA calculation",
            }

        ema_values = talib.EMA(close_prices, timeperiod=period)
        valid_ema = _extract_valid_values(ema_values, history_len)

        if not valid_ema:
            return {
                "success": False,
                "symbol": symbol,
                "error": "EMA calculation resulted in no valid data",
            }

        return {
            "success": True,
            "symbol": symbol,
            "indicator": "EMA",
            "period": period,
            "timeframe": timeframe,
            "values": valid_ema,
            "latest": valid_ema[-1],
            "count": len(valid_ema),
        }

    except Exception as e:
        await ctx.error(f"Error calculating EMA for {symbol}: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


@mcp.tool()
async def calculate_us_stock_stochastic(
    ctx: Context,
    symbol: str,
    k_period: int = 14,
    d_period: int = 3,
    timeframe: str = "1d",
    history_len: int = 50,
) -> dict:
    """
    计算美股随机指标 (Stochastic Oscillator)

    Args:
        symbol: 股票代码
        k_period: %K周期
        d_period: %D周期
        timeframe: 时间框架
        history_len: 历史数据长度
    """
    await ctx.info(f"Calculating Stochastic for {symbol}")

    try:
        required_candles = k_period + d_period + history_len
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(timeframe, "1d")
        data_period = "2y" if required_candles > 252 else "1y"

        price_data = await _fetch_multi_series_data(
            ctx,
            symbol,
            data_period,
            interval,
            required_candles,
            ["high", "low", "close"],
        )

        if not price_data:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Failed to fetch HLC data for Stochastic",
            }

        slowk, slowd = talib.STOCH(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0,
        )

        valid_k = _extract_valid_values(slowk, history_len)
        valid_d = _extract_valid_values(slowd, history_len)

        if not valid_k or not valid_d:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Stochastic calculation resulted in no valid data",
            }

        return {
            "success": True,
            "symbol": symbol,
            "indicator": "Stochastic",
            "k_period": k_period,
            "d_period": d_period,
            "timeframe": timeframe,
            "k_values": valid_k,
            "d_values": valid_d,
            "latest_k": valid_k[-1],
            "latest_d": valid_d[-1],
            "count": len(valid_k),
        }

    except Exception as e:
        await ctx.error(f"Error calculating Stochastic for {symbol}: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


@mcp.tool()
async def calculate_us_stock_williams_r(
    ctx: Context,
    symbol: str,
    period: int = 14,
    timeframe: str = "1d",
    history_len: int = 50,
) -> dict:
    """
    计算美股威廉指标 (Williams %R)

    Args:
        symbol: 股票代码
        period: 计算周期
        timeframe: 时间框架
        history_len: 历史数据长度
    """
    await ctx.info(f"Calculating Williams %R for {symbol}")

    try:
        required_candles = period + history_len
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(timeframe, "1d")
        data_period = "2y" if required_candles > 252 else "1y"

        price_data = await _fetch_multi_series_data(
            ctx,
            symbol,
            data_period,
            interval,
            required_candles,
            ["high", "low", "close"],
        )

        if not price_data:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Failed to fetch HLC data for Williams %R",
            }

        willr = talib.WILLR(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            timeperiod=period,
        )

        valid_willr = _extract_valid_values(willr, history_len)

        if not valid_willr:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Williams %R calculation resulted in no valid data",
            }

        return {
            "success": True,
            "symbol": symbol,
            "indicator": "Williams %R",
            "period": period,
            "timeframe": timeframe,
            "values": valid_willr,
            "latest": valid_willr[-1],
            "count": len(valid_willr),
        }

    except Exception as e:
        await ctx.error(f"Error calculating Williams %R for {symbol}: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


@mcp.tool()
async def calculate_us_stock_volume_indicators(
    ctx: Context,
    symbol: str,
    timeframe: str = "1d",
    history_len: int = 50,
) -> dict:
    """
    计算美股成交量指标 (OBV, Volume SMA)

    Args:
        symbol: 股票代码
        timeframe: 时间框架
        history_len: 历史数据长度
    """
    await ctx.info(f"Calculating volume indicators for {symbol}")

    try:
        required_candles = history_len + 20
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(timeframe, "1d")
        data_period = "2y" if required_candles > 252 else "1y"

        price_data = await _fetch_multi_series_data(
            ctx, symbol, data_period, interval, required_candles, ["close", "volume"]
        )

        if not price_data:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Failed to fetch price and volume data",
            }
        import numpy as np

        # 计算OBV (On Balance Volume)
        obv = talib.OBV(
            np.array([float(i) for i in price_data["close"]]),
            np.array([float(i) for i in price_data["volume"]]),
        )
        valid_obv = _extract_valid_values(obv, history_len)

        # 计算成交量移动平均
        volume_sma = talib.SMA(
            np.array([float(i) for i in price_data["volume"]]), timeperiod=20
        )
        valid_volume_sma = _extract_valid_values(volume_sma, history_len)

        if not valid_obv or not valid_volume_sma:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Volume indicators calculation resulted in no valid data",
            }

        # 计算当前成交量相对于平均成交量的比例
        current_volume = price_data["volume"][-1]
        avg_volume = valid_volume_sma[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "obv": valid_obv,
            "volume_sma": valid_volume_sma,
            "latest_obv": valid_obv[-1],
            "latest_volume_sma": valid_volume_sma[-1],
            "current_volume": current_volume,
            "volume_ratio": volume_ratio,
            "count": len(valid_obv),
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(f"Error calculating volume indicators for {symbol}: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


# ==================== 风险管理工具 ====================


@mcp.tool()
async def calculate_risk_metrics(
    ctx: Context,
    symbol: str,
    period: str = "1y",
    timeframe: str = "1d",
    benchmark_symbol: str = "^GSPC",
) -> dict:
    """
    计算风险指标 (Beta, Volatility, Sharpe Ratio等)

    Args:
        symbol: 股票代码
        period: 计算周期
        timeframe: 时间框架
        benchmark_symbol: 基准指数
    """
    await ctx.info(f"Calculating risk metrics for {symbol}")

    try:
        # 获取股票数据
        stock_data = await fetch_us_stock_hist_data(ctx, symbol, period, timeframe)
        if not stock_data or len(stock_data) < 50:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Insufficient stock data for risk calculation",
            }

        # 获取基准数据
        benchmark_data = await fetch_us_stock_hist_data(
            ctx, benchmark_symbol, period, timeframe
        )
        if not benchmark_data or len(benchmark_data) < 50:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Insufficient benchmark data for risk calculation",
            }

        # 计算收益率
        stock_prices = np.array([item["close"] for item in stock_data])
        benchmark_prices = np.array([item["close"] for item in benchmark_data])

        # 对齐数据长度
        min_length = min(len(stock_prices), len(benchmark_prices))
        stock_prices = stock_prices[-min_length:]
        benchmark_prices = benchmark_prices[-min_length:]

        # 计算日收益率
        stock_returns = np.diff(stock_prices) / stock_prices[:-1]
        benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]

        # 计算风险指标
        volatility = np.std(stock_returns) * np.sqrt(252)  # 年化波动率
        # benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)

        # Beta计算
        covariance = np.cov(stock_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # 年化收益率
        annual_return = (stock_prices[-1] / stock_prices[0]) - 1
        benchmark_annual_return = (benchmark_prices[-1] / benchmark_prices[0]) - 1

        # Sharpe比率 (假设无风险利率为2%)
        risk_free_rate = 0.02
        sharpe_ratio = (
            (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        )

        # 最大回撤
        rolling_max = np.maximum.accumulate(stock_prices)
        drawdown = (stock_prices - rolling_max) / rolling_max
        max_drawdown = np.min(drawdown)

        return {
            "success": True,
            "symbol": symbol,
            "benchmark": benchmark_symbol,
            "period": period,
            "volatility": volatility,
            "beta": beta,
            "annual_return": annual_return,
            "benchmark_return": benchmark_annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "data_points": len(stock_returns),
        }

    except Exception as e:
        await ctx.error(f"Error calculating risk metrics for {symbol}: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


# ==================== 原有技术分析工具 ====================
# (保持原有的SMA, RSI, MACD, BBANDS, ATR函数不变)


@mcp.tool()
async def calculate_us_stock_sma(ctx: Context, inputs: SmaInput) -> SmaOutput:
    """计算美股简单移动平均线 (SMA)"""
    await ctx.info(
        f"Calculating US stock SMA for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # SMA需要period + history_len - 1个数据点
        required_candles = inputs.period + inputs.history_len - 1

        # 转换时间框架
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(inputs.timeframe, "1d")

        # 根据需要的数据量选择合适的period
        period = "2y" if required_candles > 252 else "1y"

        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, period, interval, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient US stock data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)
        valid_sma = _extract_valid_values(sma_values, inputs.history_len)

        if not valid_sma:
            return SmaOutput(
                **output_base,
                error="US stock SMA calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated US stock SMA for {inputs.symbol}: {len(valid_sma)} values, latest: ${valid_sma[-1]:.2f}"
        )
        return SmaOutput(**output_base, sma=valid_sma)

    except Exception as e:
        await ctx.error(f"Error in US stock SMA calculation for {inputs.symbol}: {e}")
        return SmaOutput(**output_base, error="US stock SMA calculation error.")


@mcp.tool()
async def calculate_us_stock_rsi(ctx: Context, inputs: RsiInput) -> RsiOutput:
    """计算美股相对强弱指数 (RSI)"""
    await ctx.info(
        f"Calculating US stock RSI for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # RSI需要period + history_len个数据点
        required_candles = inputs.period + inputs.history_len

        # 转换时间框架
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(inputs.timeframe, "1d")

        # 根据需要的数据量选择合适的period
        period = "2y" if required_candles > 252 else "1y"

        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, period, interval, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient US stock data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)
        valid_rsi = _extract_valid_values(rsi_values, inputs.history_len)

        if not valid_rsi:
            return RsiOutput(
                **output_base,
                error="US stock RSI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated US stock RSI for {inputs.symbol}: {len(valid_rsi)} values, latest: {valid_rsi[-1]:.2f}"
        )
        return RsiOutput(**output_base, rsi=valid_rsi)

    except Exception as e:
        await ctx.error(f"Error in US stock RSI calculation for {inputs.symbol}: {e}")
        return RsiOutput(**output_base, error="US stock RSI calculation error.")


@mcp.tool()
async def calculate_us_stock_macd(ctx: Context, inputs: MacdInput) -> MacdOutput:
    """计算美股MACD指标"""
    await ctx.info(
        f"Calculating US stock MACD for {inputs.symbol}, Periods: {inputs.fast_period}/{inputs.slow_period}/{inputs.signal_period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "fast_period": inputs.fast_period,
        "slow_period": inputs.slow_period,
        "signal_period": inputs.signal_period,
    }
    try:
        required_candles = (
            inputs.slow_period + inputs.signal_period + inputs.history_len + 20
        )

        # 转换时间框架
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(inputs.timeframe, "1d")

        # 根据需要的数据量选择合适的period
        period = "5y" if required_candles > 500 else "2y"

        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, period, interval, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return MacdOutput(
                **output_base,
                error="Failed to fetch sufficient US stock data for MACD.",
            )

        macd, signal, hist = talib.MACD(
            close_prices,
            fastperiod=inputs.fast_period,
            slowperiod=inputs.slow_period,
            signalperiod=inputs.signal_period,
        )

        valid_macd = _extract_valid_values(macd, inputs.history_len)
        valid_signal = _extract_valid_values(signal, inputs.history_len)
        valid_hist = _extract_valid_values(hist, inputs.history_len)

        if not valid_macd or not valid_signal or not valid_hist:
            return MacdOutput(
                **output_base,
                error="US stock MACD calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated US stock MACD for {inputs.symbol}: latest MACD: {valid_macd[-1]:.4f}"
        )
        return MacdOutput(
            **output_base, macd=valid_macd, signal=valid_signal, histogram=valid_hist
        )

    except Exception as e:
        await ctx.error(
            f"Error in US stock MACD calculation for {inputs.symbol}: {e}",
        )
        return MacdOutput(**output_base, error="US stock MACD calculation error.")


@mcp.tool()
async def calculate_us_stock_bbands(ctx: Context, inputs: BbandsInput) -> BbandsOutput:
    """计算美股布林带 (Bollinger Bands)"""
    await ctx.info(
        f"Calculating US stock Bollinger Bands for {inputs.symbol}, Period: {inputs.period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
        "nbdevup": inputs.nbdevup,
        "nbdevdn": inputs.nbdevdn,
        "matype": inputs.matype,
    }
    try:
        required_candles = inputs.period + inputs.history_len - 1

        # 转换时间框架
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(inputs.timeframe, "1d")

        # 根据需要的数据量选择合适的period
        period = "2y" if required_candles > 252 else "1y"

        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, period, interval, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return BbandsOutput(
                **output_base,
                error="Failed to fetch sufficient US stock data for Bollinger Bands.",
            )

        upper, middle, lower = talib.BBANDS(
            close_prices,
            timeperiod=inputs.period,
            nbdevup=inputs.nbdevup,
            nbdevdn=inputs.nbdevdn,
            matype=inputs.matype,
        )

        valid_upper = _extract_valid_values(upper, inputs.history_len)
        valid_middle = _extract_valid_values(middle, inputs.history_len)
        valid_lower = _extract_valid_values(lower, inputs.history_len)

        if not valid_upper or not valid_middle or not valid_lower:
            return BbandsOutput(
                **output_base,
                error="US stock Bollinger Bands calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated US stock Bollinger Bands for {inputs.symbol}: latest upper: ${valid_upper[-1]:.2f}"
        )
        return BbandsOutput(
            **output_base,
            upper_band=valid_upper,
            middle_band=valid_middle,
            lower_band=valid_lower,
        )

    except Exception as e:
        await ctx.error(
            f"Error in US stock Bollinger Bands calculation for {inputs.symbol}: {e}",
        )
        return BbandsOutput(
            **output_base, error="US stock Bollinger Bands calculation error."
        )


@mcp.tool()
async def calculate_us_stock_atr(ctx: Context, inputs: AtrInput) -> AtrOutput:
    """计算美股平均真实波幅 (ATR)"""
    await ctx.info(
        f"Calculating US stock ATR for {inputs.symbol}, Period: {inputs.period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        required_candles = inputs.period + inputs.history_len - 1

        # 转换时间框架
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        interval = interval_map.get(inputs.timeframe, "1d")

        # 根据需要的数据量选择合适的period
        period = "2y" if required_candles > 252 else "1y"

        price_data = await _fetch_multi_series_data(
            ctx,
            inputs.symbol,
            period,
            interval,
            required_candles,
            ["high", "low", "close"],
        )

        if not price_data:
            return AtrOutput(**output_base, error="Failed to fetch HLC data for ATR.")

        high_prices = price_data["high"]
        low_prices = price_data["low"]
        close_prices = price_data["close"]

        if len(high_prices) < required_candles:
            return AtrOutput(
                **output_base,
                error=f"Insufficient HLC data points for ATR. Need at least {required_candles}.",
            )

        atr_values = talib.ATR(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )

        valid_atr = _extract_valid_values(atr_values, inputs.history_len)

        if not valid_atr:
            return AtrOutput(
                **output_base,
                error="US stock ATR calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated US stock ATR for {inputs.symbol}: {len(valid_atr)} values, latest: ${valid_atr[-1]:.2f}"
        )
        return AtrOutput(**output_base, atr=valid_atr)

    except Exception as e:
        await ctx.error(f"Error in US stock ATR calculation for {inputs.symbol}: {e}")
        return AtrOutput(**output_base, error="US stock ATR calculation error.")


# ==================== 综合分析工具 ====================


@mcp.tool()
async def generate_us_stock_comprehensive_report(
    ctx: Context, inputs: ComprehensiveAnalysisInput
) -> ComprehensiveAnalysisOutput:
    """
    生成美股综合技术分析报告
    """
    await ctx.info(
        f"Generating US stock comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
    )
    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # 确定要运行的指标
    default_indicators = ["SMA", "RSI", "MACD", "BBANDS", "ATR"]  # 美股常用指标
    indicators_to_run = (
        inputs.indicators_to_include
        if inputs.indicators_to_include is not None
        else default_indicators
    )

    try:
        # SMA分析
        if "SMA" in indicators_to_run:
            sma_period = inputs.sma_period or settings.DEFAULT_SMA_PERIOD
            sma_input = SmaInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=sma_period,
            )
            sma_output = await calculate_us_stock_sma.run(
                {"ctx": ctx, "inputs": sma_input}
            )
            indicator_results_structured["sma"] = json.loads(
                sma_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["sma"]["sma"] is not None
                and len(indicator_results_structured["sma"]["sma"]) > 0
            ):
                latest_sma = indicator_results_structured["sma"]["sma"][-1]
                report_sections.append(
                    f"- US Stock SMA({indicator_results_structured['sma']['period']}): ${latest_sma:.2f} (Latest)"
                )
                if len(indicator_results_structured["sma"]["sma"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["sma"]["sma"][-1]
                        > indicator_results_structured["sma"]["sma"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Trend: {trend} ({len(indicator_results_structured['sma']['sma'])} data points)"
                    )
            elif indicator_results_structured["sma"]["error"]:
                report_sections.append(
                    f"- US Stock SMA: Error - {indicator_results_structured['sma']['error']}"
                )

        # RSI分析
        if "RSI" in indicators_to_run:
            rsi_period = inputs.rsi_period or settings.DEFAULT_RSI_PERIOD
            rsi_input = RsiInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=rsi_period,
            )
            rsi_output = await calculate_us_stock_rsi.run(
                {"ctx": ctx, "inputs": rsi_input}
            )
            indicator_results_structured["rsi"] = json.loads(
                rsi_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["rsi"]["rsi"] is not None
                and len(indicator_results_structured["rsi"]["rsi"]) > 0
            ):
                latest_rsi = indicator_results_structured["rsi"]["rsi"][-1]
                report_sections.append(
                    f"- US Stock RSI({indicator_results_structured['rsi']['period']}): {latest_rsi:.2f}"
                )
                if latest_rsi > 70:
                    report_sections.append(
                        "  - Note: RSI suggests overbought conditions (>70)"
                    )
                elif latest_rsi < 30:
                    report_sections.append(
                        "  - Note: RSI suggests oversold conditions (<30)"
                    )

                if len(indicator_results_structured["rsi"]["rsi"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["rsi"]["rsi"][-1]
                        > indicator_results_structured["rsi"]["rsi"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Trend: {trend} ({len(indicator_results_structured['rsi']['rsi'])} data points)"
                    )
            elif indicator_results_structured["rsi"]["error"]:
                report_sections.append(
                    f"- US Stock RSI: Error - {indicator_results_structured['rsi']['error']}"
                )

        # MACD分析
        if "MACD" in indicators_to_run:
            macd_input = MacdInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                fast_period=inputs.macd_fast_period or settings.DEFAULT_MACD_FAST,
                slow_period=inputs.macd_slow_period or settings.DEFAULT_MACD_SLOW,
                signal_period=inputs.macd_signal_period or settings.DEFAULT_MACD_SIGNAL,
            )
            macd_output = await calculate_us_stock_macd.run(
                {"ctx": ctx, "inputs": macd_input}
            )
            indicator_results_structured["macd"] = json.loads(
                macd_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["macd"]["macd"] is not None
                and len(indicator_results_structured["macd"]["macd"]) > 0
                and indicator_results_structured["macd"]["signal"] is not None
                and len(indicator_results_structured["macd"]["signal"]) > 0
                and indicator_results_structured["macd"]["histogram"] is not None
                and len(indicator_results_structured["macd"]["histogram"]) > 0
            ):
                latest_macd = indicator_results_structured["macd"]["macd"][-1]
                latest_signal = indicator_results_structured["macd"]["signal"][-1]
                latest_hist = indicator_results_structured["macd"]["histogram"][-1]

                report_sections.append(
                    f"- US Stock MACD({indicator_results_structured['macd']['fast_period']},{indicator_results_structured['macd']['slow_period']},{indicator_results_structured['macd']['signal']}): "
                    f"MACD: {latest_macd:.4f}, Signal: {latest_signal:.4f}, Histogram: {latest_hist:.4f}"
                )

                if latest_hist > 0 and latest_macd > latest_signal:
                    report_sections.append(
                        "  - Note: MACD histogram positive, potentially bullish momentum"
                    )
                elif latest_hist < 0 and latest_macd < latest_signal:
                    report_sections.append(
                        "  - Note: MACD histogram negative, potentially bearish momentum"
                    )

            elif indicator_results_structured["macd"]["error"]:
                report_sections.append(
                    f"- US Stock MACD: Error - {indicator_results_structured['macd']['error']}"
                )

        # Bollinger Bands分析
        if "BBANDS" in indicators_to_run:
            bbands_input = BbandsInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.bbands_period or settings.DEFAULT_BBANDS_PERIOD,
            )
            bbands_output = await calculate_us_stock_bbands.run(
                {"ctx": ctx, "inputs": bbands_input}
            )
            indicator_results_structured["bbands"] = json.loads(
                bbands_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["bbands"]["upper_band"] is not None
                and len(indicator_results_structured["bbands"]["upper_band"]) > 0
                and indicator_results_structured["bbands"]["middle_band"] is not None
                and len(indicator_results_structured["bbands"]["middle_band"]) > 0
                and indicator_results_structured["bbands"]["lower_band"] is not None
                and len(indicator_results_structured["bbands"]["lower_band"]) > 0
            ):
                latest_upper = indicator_results_structured["bbands"]["upper_band"][-1]
                latest_middle = indicator_results_structured["bbands"]["middle_band"][
                    -1
                ]
                latest_lower = indicator_results_structured["bbands"]["lower_band"][-1]

                report_sections.append(
                    f"- US Stock Bollinger Bands({indicator_results_structured['bbands']['period']}): "
                    f"Upper: ${latest_upper:.2f}, Middle: ${latest_middle:.2f}, Lower: ${latest_lower:.2f}"
                )
                report_sections.append(
                    f"  - Band Width: ${latest_upper - latest_lower:.2f}"
                )

            elif indicator_results_structured["bbands"]["error"]:
                report_sections.append(
                    f"- US Stock Bollinger Bands: Error - {indicator_results_structured['bbands']['error']}"
                )

        # ATR分析
        if "ATR" in indicators_to_run:
            atr_input = AtrInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.atr_period or settings.DEFAULT_ATR_PERIOD,
            )
            atr_output = await calculate_us_stock_atr.run(
                {"ctx": ctx, "inputs": atr_input}
            )
            indicator_results_structured["atr"] = json.loads(
                atr_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["atr"]["atr"] is not None
                and len(indicator_results_structured["atr"]["atr"]) > 0
            ):
                latest_atr = indicator_results_structured["atr"]["atr"][-1]
                report_sections.append(
                    f"- US Stock ATR({indicator_results_structured['atr']['period']}): ${latest_atr:.2f} (Volatility Measure)"
                )

                if len(indicator_results_structured["atr"]["atr"]) > 1:
                    vol_trend = (
                        "↗"
                        if indicator_results_structured["atr"]["atr"][-1]
                        > indicator_results_structured["atr"]["atr"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Volatility Trend: {vol_trend} ({len(indicator_results_structured['atr']['atr'])} data points)"
                    )

            elif indicator_results_structured["atr"]["error"]:
                report_sections.append(
                    f"- US Stock ATR: Error - {indicator_results_structured['atr']['error']}"
                )

        # 合成报告
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base,
                error="No indicator data could be calculated",
            )

        report_title = f"US Stock Technical Analysis Report - {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} Data Points:\n"
        report_text = report_title + "\n".join(report_sections)

        # 添加趋势总结
        summary_sections = []
        trend_indicators = []

        for indicator_name, indicator_data in indicator_results_structured.items():
            if isinstance(indicator_data, dict) and not indicator_data.get("error"):
                if indicator_name == "sma" and indicator_data.get("sma"):
                    sma_data = indicator_data["sma"]
                    if len(sma_data) > 1:
                        trend_indicators.append(
                            f"SMA: {'↗' if sma_data[-1] > sma_data[-2] else '↘'}"
                        )

                elif indicator_name == "rsi" and indicator_data.get("rsi"):
                    rsi_data = indicator_data["rsi"]
                    if len(rsi_data) > 1:
                        trend_indicators.append(
                            f"RSI: {'↗' if rsi_data[-1] > rsi_data[-2] else '↘'}"
                        )

        if trend_indicators:
            summary_sections.append("\nTrend Summary:")
            summary_sections.extend([f"  {trend}" for trend in trend_indicators])
            report_text += "\n" + "\n".join(summary_sections)

        await ctx.info(
            f"Successfully generated US stock comprehensive report for {inputs.symbol}"
        )
        return ComprehensiveAnalysisOutput(
            **output_base,
            report_text=report_text,
            structured_data=indicator_results_structured,
        )

    except Exception as e:
        await ctx.error(
            f"Error in US stock comprehensive report for {inputs.symbol}: {e}",
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"US stock comprehensive analysis error: {str(e)}"
        )


# ==================== 新增：股票筛选工具 ====================


# @mcp.tool()
# async def screen_stocks_by_criteria(
#     ctx: Context,
#     symbols: List[str],
#     min_price: Optional[float] = None,
#     max_price: Optional[float] = None,
#     min_volume: Optional[int] = None,
#     min_market_cap: Optional[float] = None,
#     max_pe_ratio: Optional[float] = None,
# ) -> dict:
#     """
#     根据条件筛选股票（新增工具）

#     Args:
#         symbols: 股票代码列表
#         min_price: 最低价格
#         max_price: 最高价格
#         min_volume: 最小成交量
#         min_market_cap: 最小市值
#         max_pe_ratio: 最大市盈率
#     """
#     await ctx.info(f"Screening {len(symbols)} stocks with criteria")

#     try:
#         screened_stocks = []

#         for symbol in symbols:
#             try:
#                 # 获取股票信息
#                 stock_info = await fetch_us_stock_info(ctx, symbol)
#                 if not stock_info:
#                     continue

#                 # 获取当前价格
#                 stock_data = await fetch_us_stock_hist_data(ctx, symbol, "1d", "1d")
#                 if not stock_data:
#                     continue

#                 current_price = stock_data[-1]["close"]
#                 current_volume = stock_data[-1]["volume"]

#                 # 应用筛选条件
#                 if min_price is not None and current_price < min_price:
#                     continue
#                 if max_price is not None and current_price > max_price:
#                     continue
#                 if min_volume is not None and current_volume < min_volume:
#                     continue
#                 if min_market_cap is not None and stock_info.get("market_cap", 0) < min_market_cap:
#                     continue
#                 if max_pe_ratio is not None and stock_info.get("trailing_pe", float('inf')) > max_pe_ratio:
#                     continue

#                 screened_stocks.append({
#                     "symbol": symbol,
#                     "name": stock_info.get("company_name", ""),
#                     "price": current_price,
#                     "volume": current_volume,
#                     "market_cap": stock_info.get("market_cap", 0),
#                     "pe_ratio": stock_info.get("trailing_pe", None),
#                     "sector": stock_info.get("sector", ""),
#                 })

#             except Exception as e:
#                 await ctx.warning(f"Error screening {symbol}: {e}")
#                 continue

#         return {
#             "success": True,
#             "screened_stocks": screened_stocks,
#             "original_count": len(symbols),
#             "screened_count": len(screened_stocks),
#             "criteria": {
#                 "min_price": min_price,
#                 "max_price": max_price,
#                 "min_volume": min_volume,
#                 "min_market_cap": min_market_cap,
#                 "max_pe_ratio": max_pe_ratio,
#             },
#         }

#     except Exception as e:
#         await ctx.error(f"Error in stock screening: {e}")
#         return {"success": False, "error": str(e)}
