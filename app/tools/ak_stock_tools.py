import numpy as np
import talib
import json
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta, time

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
    RecommendationCriteria,
    StockRecommendationInput,
    StockRecommendationOutput,
    StockScore,
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

# 导入现有的AKShare服务函数
from services.akshare_service import (
    fetch_stock_hist_data,
    fetch_stock_realtime_data,
    fetch_hk_stock_data,
    search_stock_by_name,
    search_hk_stock_by_name,
    fetch_hk_stock_symbol_mapping,
)

mcp = FastMCP()

# ==================== 数据获取辅助函数 ====================


async def _fetch_single_series_data(
    ctx: Context,
    symbol: str,
    period: str,
    required_candles: int,
    series_type: str = "close",
    market_type: str = "a_stock",
) -> Optional[np.ndarray]:
    """
    获取单个数据序列

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        period: 周期
        required_candles: 需要的K线数量
        series_type: 数据类型 (open, high, low, close, volume)
        market_type: 市场类型 (a_stock, hk_stock)

    Returns:
        numpy数组或None
    """
    try:
        # 计算需要的天数，加上缓冲区
        buffer_days = max(20, required_candles // 4)
        total_days = required_candles + buffer_days

        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=total_days + 50)).strftime(
            "%Y%m%d"
        )

        # 根据市场类型获取数据
        if market_type == "hk_stock":
            stock_data = await fetch_hk_stock_data(
                ctx, symbol, period, start_date, end_date
            )
        else:  # a_stock
            stock_data = await fetch_stock_hist_data(
                ctx, symbol, period, start_date, end_date
            )

        if not stock_data:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(stock_data) if stock_data else 0}, need {required_candles}"
            )
            return None

        # 提取指定类型的数据
        if series_type not in ["open", "high", "low", "close", "volume"]:
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
    required_candles: int,
    series_types: List[str],
    market_type: str = "a_stock",
) -> Optional[Dict[str, np.ndarray]]:
    """
    获取多个数据序列

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        period: 周期
        required_candles: 需要的K线数量
        series_types: 需要的数据类型列表
        market_type: 市场类型

    Returns:
        包含各数据序列的字典或None
    """
    try:
        # 计算需要的天数
        buffer_days = max(20, required_candles // 4)
        total_days = required_candles + buffer_days

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=total_days + 50)).strftime(
            "%Y%m%d"
        )

        # 根据市场类型获取数据
        if market_type == "hk_stock":
            stock_data = await fetch_hk_stock_data(ctx, symbol, start_date, end_date)
        else:  # a_stock
            stock_data = await fetch_stock_hist_data(
                ctx, symbol, period, start_date, end_date
            )

        if not stock_data:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(stock_data) if stock_data else 0}, need {required_candles}"
            )
            return None

        result = {}
        for series_type in series_types:
            if series_type not in ["open", "high", "low", "close", "volume"]:
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


# ==================== A股数据获取工具 ====================


@mcp.tool()
async def get_a_stock_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    获取A股K线数据

    Args:
        ctx: FastMCP上下文对象
        inputs: A股K线输入参数，包含symbol（A股代码，6位数字）、timeframe（时间框架）、limit（数据条数，默认24）、adjust（复权类型，默认qfq）

    Returns:
        CandlesOutput: K线数据输出对象，包含symbol、timeframe、candles列表、count和可能的error信息
    """
    await ctx.info(
        f"Fetching A-stock candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")

        # 获取复权类型
        adjust = getattr(inputs, "adjust", "qfq") or "qfq"

        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=inputs.limit + 50)).strftime(
            "%Y%m%d"
        )

        # 使用现有的AKShare服务获取A股数据
        stock_data = await fetch_stock_hist_data(
            ctx=ctx,
            symbol=inputs.symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if stock_data:
            candles = []
            for item in stock_data[-inputs.limit :]:  # 只取需要的数量
                # 转换为OHLCVCandle格式
                date_str = item["date"]
                if isinstance(date_str, str):
                    timestamp = int(
                        datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000
                    )
                else:
                    timestamp = int(
                        datetime.combine(date_str, time()).timestamp() * 1000
                    )

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
                f"Successfully fetched {len(candles)} A-stock candles for {inputs.symbol}"
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
                error=f"No A-stock candle data available for {inputs.symbol}",
            )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(f"Error fetching A-stock candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"A-stock data error: {str(e)}",
        )


@mcp.tool()
async def get_a_stock_price(ctx: Context, inputs: PriceInput) -> PriceOutput:
    """
    获取A股当前价格

    Args:
        ctx: FastMCP上下文对象
        inputs: A股价格输入参数，包含symbol（A股代码，6位数字）

    Returns:
        PriceOutput: 价格输出对象，包含symbol、price、timestamp和可能的error信息
    """
    await ctx.info(f"Fetching A-stock current price for {inputs.symbol}")

    try:
        # 使用实时数据API
        realtime_data = await fetch_stock_realtime_data(ctx, [inputs.symbol])

        if realtime_data and len(realtime_data) > 0:
            stock_info = realtime_data[0]  # 取第一个结果
            timestamp = int(datetime.now().timestamp() * 1000)

            await ctx.info(
                f"A-stock current price for {inputs.symbol}: ¥{stock_info['price']}"
            )
            return PriceOutput(
                symbol=inputs.symbol,
                price=stock_info["price"],
                timestamp=timestamp,
            )
        else:
            # 如果实时数据不可用，尝试获取最新的历史数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

            stock_data = await fetch_stock_hist_data(
                ctx, inputs.symbol, "daily", start_date, end_date
            )

            if stock_data and len(stock_data) > 0:
                latest = stock_data[-1]
                timestamp = int(
                    datetime.strptime(latest["date"], "%Y-%m-%d").timestamp() * 1000
                )

                return PriceOutput(
                    symbol=inputs.symbol,
                    price=latest["close"],
                    timestamp=timestamp,
                )

            return PriceOutput(
                symbol=inputs.symbol, error="A-stock price data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching A-stock price for {inputs.symbol}: {e}")
        return PriceOutput(symbol=inputs.symbol, error=f"A-stock price error: {str(e)}")


@mcp.tool()
async def search_a_stock_symbols(ctx: Context, query: str, limit: int = 10) -> dict:
    """
    搜索A股股票代码

    Args:
        ctx: FastMCP上下文对象
        query: 搜索关键词（公司名称、股票代码或简称）
        limit: 返回结果数量，默认10

    Returns:
        dict: 包含success状态、query查询词、market_type市场类型、results结果列表、count结果数量或error错误信息的字典
    """
    await ctx.info(f"Searching A-stock symbols for '{query}'")

    try:
        results = await search_stock_by_name(ctx, query)

        if results:
            # 限制结果数量
            limited_results = results[:limit]
            return {
                "success": True,
                "query": query,
                "market_type": "a_stock",
                "results": limited_results,
                "count": len(limited_results),
            }
        else:
            return {
                "success": False,
                "query": query,
                "market_type": "a_stock",
                "error": "No matching stocks found",
            }

    except Exception as e:
        await ctx.error(f"Error searching A-stock symbols: {e}")
        return {
            "success": False,
            "query": query,
            "market_type": "a_stock",
            "error": str(e),
        }


@mcp.tool()
async def search_hk_stock_symbols(ctx: Context, query: str, limit: int = 10) -> dict:
    """
    搜索A股股票代码

    Args:
        ctx: FastMCP上下文对象
        query: 搜索关键词（公司名称、股票代码或简称）
        limit: 返回结果数量，默认10

    Returns:
        dict: 包含success状态、query查询词、market_type市场类型、results结果列表、count结果数量或error错误信息的字典
    """
    await ctx.info(f"Searching A-stock symbols for '{query}'")

    try:
        results = await search_hk_stock_by_name(ctx, query)

        if results:
            # 限制结果数量
            limited_results = results[:limit]
            return {
                "success": True,
                "query": query,
                "market_type": "a_stock",
                "results": limited_results,
                "count": len(limited_results),
            }
        else:
            return {
                "success": False,
                "query": query,
                "market_type": "a_stock",
                "error": "No matching stocks found",
            }

    except Exception as e:
        await ctx.error(f"Error searching A-stock symbols: {e}")
        return {
            "success": False,
            "query": query,
            "market_type": "a_stock",
            "error": str(e),
        }


@mcp.tool()
async def get_a_stock_ticker(ctx: Context, inputs: TickerInput) -> TickerOutput:
    """
    获取A股详细行情数据

    Args:
        ctx: FastMCP上下文对象
        inputs: A股行情输入参数，包含symbol（A股代码）

    Returns:
        TickerOutput: 详细行情输出对象，包含symbol、last、open、high、low、close、volume、change、percentage、timestamp和可能的error信息
    """
    await ctx.info(f"Fetching A-stock ticker for {inputs.symbol}")

    try:
        # 获取实时数据
        realtime_data = await fetch_stock_realtime_data(ctx, [inputs.symbol])

        if realtime_data and len(realtime_data) > 0:
            stock_info = realtime_data[0]
            timestamp = int(datetime.now().timestamp() * 1000)

            await ctx.info(f"Successfully fetched A-stock ticker for {inputs.symbol}")
            return TickerOutput(
                symbol=inputs.symbol,
                last=stock_info["price"],
                open=stock_info["open"],
                high=stock_info["high"],
                low=stock_info["low"],
                close=stock_info["price"],
                volume=stock_info["volume"],
                change=stock_info["change"],
                percentage=stock_info["pct_change"],
                timestamp=timestamp,
            )
        else:
            return TickerOutput(
                symbol=inputs.symbol, error="A-stock ticker data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching A-stock ticker for {inputs.symbol}: {e}")
        return TickerOutput(
            symbol=inputs.symbol, error=f"A-stock ticker error: {str(e)}"
        )


# ==================== 港股数据获取工具 ====================


@mcp.tool()
async def get_hk_stock_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    获取港股K线数据

    Args:
        ctx: FastMCP上下文对象
        inputs: 港股K线输入参数，包含symbol（港股代码，5位数字）、timeframe（时间框架）、limit（数据条数）

    Returns:
        CandlesOutput: K线数据输出对象，包含symbol、timeframe、candles列表、count和可能的error信息
    """
    await ctx.info(
        f"Fetching HK stock candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")

        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=inputs.limit + 50)).strftime(
            "%Y%m%d"
        )

        # 使用现有的港股服务获取数据
        stock_data = await fetch_hk_stock_data(
            ctx=ctx,
            period=period,
            symbol=inputs.symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if stock_data:
            candles = []
            for item in stock_data[-inputs.limit :]:  # 只取需要的数量
                # 转换为OHLCVCandle格式
                date_str = item["date"]
                if isinstance(date_str, str):
                    timestamp = int(
                        datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000
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
                f"Successfully fetched {len(candles)} HK stock candles for {inputs.symbol}"
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
                error=f"No HK stock candle data available for {inputs.symbol}",
            )

    except Exception as e:
        await ctx.error(f"Error fetching HK stock candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"HK stock data error: {str(e)}",
        )


@mcp.tool()
async def get_hk_stock_price(ctx: Context, inputs: PriceInput) -> PriceOutput:
    """
    获取港股当前价格

    Args:
        ctx: FastMCP上下文对象
        inputs: 港股价格输入参数，包含symbol（港股代码）

    Returns:
        PriceOutput: 价格输出对象，包含symbol、price、timestamp和可能的error信息
    """
    await ctx.info(f"Fetching HK stock current price for {inputs.symbol}")

    try:
        # 获取最新的历史数据作为当前价格
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

        stock_data = await fetch_hk_stock_data(
            ctx, inputs.symbol, "daily", start_date, end_date
        )

        if stock_data and len(stock_data) > 0:
            latest = stock_data[-1]
            timestamp = int(
                datetime.strptime(latest["date"], "%Y-%m-%d").timestamp() * 1000
            )

            await ctx.info(
                f"HK stock current price for {inputs.symbol}: HK${latest['close']}"
            )
            return PriceOutput(
                symbol=inputs.symbol,
                price=latest["close"],
                timestamp=timestamp,
            )
        else:
            return PriceOutput(
                symbol=inputs.symbol, error="HK stock price data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching HK stock price for {inputs.symbol}: {e}")
        return PriceOutput(
            symbol=inputs.symbol, error=f"HK stock price error: {str(e)}"
        )


# ==================== A股技术分析工具 ====================


@mcp.tool()
async def calculate_a_stock_sma(ctx: Context, inputs: SmaInput) -> SmaOutput:
    """
    计算A股简单移动平均线（SMA）

    Args:
        ctx: FastMCP上下文对象
        inputs: SMA输入参数，包含symbol（A股代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为20）、history_len（历史数据长度，默认5）

    Returns:
        SmaOutput: SMA输出对象，包含symbol、timeframe、period、sma指标值或error错误信息
    """
    await ctx.info(
        f"Calculating A-stock SMA for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        # SMA需要period + history_len - 1个数据点
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "a_stock"
        )
        if close_prices is None:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient A-stock data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)
        valid_sma = _extract_valid_values(sma_values, inputs.history_len)

        if not valid_sma:
            return SmaOutput(
                **output_base,
                error="A-stock SMA calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock SMA for {inputs.symbol}: {len(valid_sma)} values, latest: ¥{valid_sma[-1]:.2f}"
        )
        return SmaOutput(**output_base, sma=valid_sma)

    except Exception as e:
        await ctx.error(f"Error in A-stock SMA calculation for {inputs.symbol}: {e}")
        return SmaOutput(**output_base, error="A-stock SMA calculation error.")


@mcp.tool()
async def calculate_a_stock_rsi(ctx: Context, inputs: RsiInput) -> RsiOutput:
    """
    计算A股相对强弱指数（RSI）

    Args:
        ctx: FastMCP上下文对象
        inputs: RSI输入参数，包含symbol（A股代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为14）、history_len（历史数据长度，默认5）

    Returns:
        RsiOutput: RSI输出对象，包含symbol、timeframe、period、rsi指标值或error错误信息
    """
    await ctx.info(
        f"Calculating A-stock RSI for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        # RSI需要period + history_len个数据点
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "a_stock"
        )

        if close_prices is None:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient A-stock data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)
        valid_rsi = _extract_valid_values(rsi_values, inputs.history_len)

        if not valid_rsi:
            return RsiOutput(
                **output_base,
                error="A-stock RSI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock RSI for {inputs.symbol}: {len(valid_rsi)} values, latest: {valid_rsi[-1]:.2f}"
        )
        return RsiOutput(**output_base, rsi=valid_rsi)

    except Exception as e:
        await ctx.error(f"Error in A-stock RSI calculation for {inputs.symbol}: {e}")
        return RsiOutput(**output_base, error="A-stock RSI calculation error.")


@mcp.tool()
async def calculate_a_stock_macd(ctx: Context, inputs: MacdInput) -> MacdOutput:
    """
    计算A股MACD指标

    Args:
        ctx: FastMCP上下文对象
        inputs: MACD输入参数，包含symbol（A股代码）、timeframe（时间框架，默认1h）、fast_period（快线周期，默认为12）、slow_period（慢线周期，默认为26）、signal_period（信号线周期，默认为9）、history_len（历史数据长度，默认5）

    Returns:
        MacdOutput: MACD输出对象，包含symbol、timeframe、fast_period、slow_period、signal_period、macd主线、signal信号线、histogram柱状图或error错误信息
    """
    await ctx.info(
        f"Calculating A-stock MACD for {inputs.symbol}, Periods: {inputs.fast_period}/{inputs.slow_period}/{inputs.signal_period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "fast_period": inputs.fast_period,
        "slow_period": inputs.slow_period,
        "signal_period": inputs.signal_period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = (
            inputs.slow_period + inputs.signal_period + inputs.history_len + 10
        )
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "a_stock"
        )

        if close_prices is None:
            return MacdOutput(
                **output_base, error="Failed to fetch sufficient A-stock data for MACD."
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
                error="A-stock MACD calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock MACD for {inputs.symbol}: latest MACD: {valid_macd[-1]:.4f}"
        )
        return MacdOutput(
            **output_base, macd=valid_macd, signal=valid_signal, histogram=valid_hist
        )

    except Exception as e:
        await ctx.error(f"Error in A-stock MACD calculation for {inputs.symbol}: {e}")
        return MacdOutput(**output_base, error="A-stock MACD calculation error.")


@mcp.tool()
async def calculate_a_stock_bbands(ctx: Context, inputs: BbandsInput) -> BbandsOutput:
    """
    计算A股布林带（Bollinger Bands）

    Args:
        ctx: FastMCP上下文对象
        inputs: 布林带输入参数，包含symbol（A股代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为20）、nbdevup（上轨标准差倍数，默认为2.0）、nbdevdn（下轨标准差倍数，默认为2.0）、matype（移动平均类型，默认为0）、history_len（历史数据长度，默认5）

    Returns:
        BbandsOutput: 布林带输出对象，包含symbol、timeframe、period、nbdevup、nbdevdn、matype、upper_band上轨、middle_band中轨、lower_band下轨或error错误信息
    """
    await ctx.info(
        f"Calculating A-stock Bollinger Bands for {inputs.symbol}, Period: {inputs.period}"
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
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "a_stock"
        )

        if close_prices is None:
            return BbandsOutput(
                **output_base,
                error="Failed to fetch sufficient A-stock data for Bollinger Bands.",
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
                error="A-stock Bollinger Bands calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock Bollinger Bands for {inputs.symbol}: latest upper: ¥{valid_upper[-1]:.2f}"
        )
        return BbandsOutput(
            **output_base,
            upper_band=valid_upper,
            middle_band=valid_middle,
            lower_band=valid_lower,
        )

    except Exception as e:
        await ctx.error(
            f"Error in A-stock Bollinger Bands calculation for {inputs.symbol}: {e}",
        )
        return BbandsOutput(
            **output_base, error="A-stock Bollinger Bands calculation error."
        )


@mcp.tool()
async def calculate_a_stock_atr(ctx: Context, inputs: AtrInput) -> AtrOutput:
    """
    计算A股平均真实波幅（ATR）

    Args:
        ctx: FastMCP上下文对象
        inputs: ATR输入参数，包含symbol（A股代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为14）、history_len（历史数据长度，默认5）

    Returns:
        AtrOutput: ATR输出对象，包含symbol、timeframe、period、atr指标值或error错误信息
    """
    await ctx.info(
        f"Calculating A-stock ATR for {inputs.symbol}, Period: {inputs.period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len - 1

        price_data = await _fetch_multi_series_data(
            ctx,
            inputs.symbol,
            timeframe,
            required_candles,
            ["high", "low", "close"],
            "a_stock",
        )

        if not price_data:
            return AtrOutput(**output_base, error="Failed to fetch HLC data for ATR.")

        high_prices = price_data["high"]
        low_prices = price_data["low"]
        close_prices = price_data["close"]

        atr_values = talib.ATR(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )

        valid_atr = _extract_valid_values(atr_values, inputs.history_len)

        if not valid_atr:
            return AtrOutput(
                **output_base,
                error="A-stock ATR calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock ATR for {inputs.symbol}: {len(valid_atr)} values, latest: ¥{valid_atr[-1]:.2f}"
        )
        return AtrOutput(**output_base, atr=valid_atr)

    except Exception as e:
        await ctx.error(f"Error in A-stock ATR calculation for {inputs.symbol}: {e}")
        return AtrOutput(**output_base, error="A-stock ATR calculation error.")


# ==================== 港股技术分析工具 ====================


@mcp.tool()
async def calculate_hk_stock_sma(ctx: Context, inputs: SmaInput) -> SmaOutput:
    """
    计算港股简单移动平均线（SMA）

    Args:
        ctx: FastMCP上下文对象
        inputs: SMA输入参数，包含symbol（港股代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为20）、history_len（历史数据长度，默认5）

    Returns:
        SmaOutput: SMA输出对象，包含symbol、timeframe、period、sma指标值或error错误信息
    """
    await ctx.info(
        f"Calculating HK stock SMA for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "hk_stock"
        )

        if close_prices is None:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient HK stock data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)
        valid_sma = _extract_valid_values(sma_values, inputs.history_len)

        if not valid_sma:
            return SmaOutput(
                **output_base,
                error="HK stock SMA calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated HK stock SMA for {inputs.symbol}: {len(valid_sma)} values, latest: HK${valid_sma[-1]:.2f}"
        )
        return SmaOutput(**output_base, sma=valid_sma)

    except Exception as e:
        await ctx.error(f"Error in HK stock SMA calculation for {inputs.symbol}: {e}")
        return SmaOutput(**output_base, error="HK stock SMA calculation error.")


@mcp.tool()
async def calculate_hk_stock_rsi(ctx: Context, inputs: RsiInput) -> RsiOutput:
    """
    计算港股相对强弱指数（RSI）

    Args:
        ctx: FastMCP上下文对象
        inputs: RSI输入参数，包含symbol（港股代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为14）、history_len（历史数据长度，默认5）

    Returns:
        RsiOutput: RSI输出对象，包含symbol、timeframe、period、rsi指标值或error错误信息
    """
    await ctx.info(
        f"Calculating HK stock RSI for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "hk_stock"
        )

        if close_prices is None:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient HK stock data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)
        valid_rsi = _extract_valid_values(rsi_values, inputs.history_len)

        if not valid_rsi:
            return RsiOutput(
                **output_base,
                error="HK stock RSI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated HK stock RSI for {inputs.symbol}: {len(valid_rsi)} values, latest: {valid_rsi[-1]:.2f}"
        )
        return RsiOutput(**output_base, rsi=valid_rsi)

    except Exception as e:
        await ctx.error(f"Error in HK stock RSI calculation for {inputs.symbol}: {e}")
        return RsiOutput(**output_base, error="HK stock RSI calculation error.")


# ==================== 综合分析工具 ====================


@mcp.tool()
async def generate_a_stock_comprehensive_report(
    ctx: Context, inputs: ComprehensiveAnalysisInput
) -> ComprehensiveAnalysisOutput:
    """
    生成A股综合技术分析报告

    Args:
        ctx: FastMCP上下文对象
        inputs: 综合分析输入参数，包含symbol（A股代码）、timeframe（时间框架，默认1h）、history_len（历史数据长度，默认5）、indicators_to_include（要包含的指标列表，默认全部）以及各指标的可选周期参数（sma_period默认为20、rsi_period默认为14、macd_fast_period默认为12等）

    Returns:
        ComprehensiveAnalysisOutput: 综合分析输出对象，包含symbol、timeframe、report_text报告文本、structured_data结构化数据或error错误信息
    """
    await ctx.info(
        f"Generating A-stock comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
    )
    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # 确定要运行的指标
    default_indicators = ["SMA", "RSI", "MACD", "BBANDS"]  # A股常用指标
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
            sma_output = await calculate_a_stock_sma.run(
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
                    f"- A股SMA({indicator_results_structured['sma']['period']}): ¥{latest_sma:.2f} (最新值)"
                )
                if len(indicator_results_structured["sma"]["sma"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["sma"]["sma"][-1]
                        > indicator_results_structured["sma"]["sma"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(indicator_results_structured['sma']['sma'])} 个数据点)"
                    )
            elif indicator_results_structured["sma"]["error"]:
                report_sections.append(
                    f"- A股SMA: 错误 - {indicator_results_structured['sma']['error']}"
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
            rsi_output = await calculate_a_stock_rsi.run(
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
                    f"- A股RSI({indicator_results_structured['rsi']['period']}): {latest_rsi:.2f}"
                )
                if latest_rsi > 70:
                    report_sections.append("  - 注意: RSI表明超买状态 (>70)")
                elif latest_rsi < 30:
                    report_sections.append("  - 注意: RSI表明超卖状态 (<30)")

                if len(indicator_results_structured["rsi"]["rsi"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["rsi"]["rsi"][-1]
                        > indicator_results_structured["rsi"]["rsi"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(indicator_results_structured['rsi']['rsi'])} 个数据点)"
                    )
            elif indicator_results_structured["rsi"]["error"]:
                report_sections.append(
                    f"- A股RSI: 错误 - {indicator_results_structured['rsi']['error']}"
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
            macd_output = await calculate_a_stock_macd.run(
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
                    f"- A股MACD({indicator_results_structured['macd']['fast_period']},{indicator_results_structured['macd']['slow_period']},{indicator_results_structured['macd']['signal_period']}): "
                    f"MACD: {latest_macd:.4f}, 信号线: {latest_signal:.4f}, 柱状图: {latest_hist:.4f}"
                )

                if latest_hist > 0 and latest_macd > latest_signal:
                    report_sections.append("  - 注意: MACD柱状图为正，可能有看涨动量")
                elif latest_hist < 0 and latest_macd < latest_signal:
                    report_sections.append("  - 注意: MACD柱状图为负，可能有看跌动量")

            elif indicator_results_structured["macd"]["error"]:
                report_sections.append(
                    f"- A股MACD: 错误 - {indicator_results_structured['macd']['error']}"
                )

        # Bollinger Bands分析
        if "BBANDS" in indicators_to_run:
            bbands_input = BbandsInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.bbands_period or settings.DEFAULT_BBANDS_PERIOD,
            )
            bbands_output = await calculate_a_stock_bbands.run(
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
                    f"- A股布林带({indicator_results_structured['bbands']['period']}): "
                    f"上轨: ¥{latest_upper:.2f}, 中轨: ¥{latest_middle:.2f}, 下轨: ¥{latest_lower:.2f}"
                )
                report_sections.append(f"  - 带宽: ¥{latest_upper - latest_lower:.2f}")

            elif indicator_results_structured["bbands"]["error"]:
                report_sections.append(
                    f"- A股布林带: 错误 - {indicator_results_structured['bbands']['error']}"
                )

        # 合成报告
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base,
                error="无法计算任何指标数据",
            )

        report_title = f"A股技术分析报告 - {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} 个数据点:\n"
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
            summary_sections.append("\n趋势总结:")
            summary_sections.extend([f"  {trend}" for trend in trend_indicators])
            report_text += "\n" + "\n".join(summary_sections)

        await ctx.info(
            f"Successfully generated A-stock comprehensive report for {inputs.symbol}"
        )
        return ComprehensiveAnalysisOutput(
            **output_base,
            report_text=report_text,
            structured_data=indicator_results_structured,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(
            f"Error in A-stock comprehensive report for {inputs.symbol}: {e}",
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"A股综合分析报告生成错误: {str(e)}"
        )


@mcp.tool()
async def generate_hk_stock_comprehensive_report(
    ctx: Context, inputs: ComprehensiveAnalysisInput
) -> ComprehensiveAnalysisOutput:
    """
    生成港股综合技术分析报告

    Args:
        ctx: FastMCP上下文对象
        inputs: 综合分析输入参数，包含symbol（港股代码）、timeframe（时间框架，默认1h）、history_len（历史数据长度，默认5）、indicators_to_include（要包含的指标列表，默认全部）以及各指标的可选周期参数（sma_period默认为20、rsi_period默认为14等）

    Returns:
        ComprehensiveAnalysisOutput: 综合分析输出对象，包含symbol、timeframe、report_text报告文本、structured_data结构化数据或error错误信息
    """
    await ctx.info(
        f"Generating HK stock comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
    )
    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # 确定要运行的指标
    default_indicators = ["SMA", "RSI"]  # 港股常用指标
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
            sma_output = await calculate_hk_stock_sma.run(
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
                    f"- 港股SMA({indicator_results_structured['sma']['period']}): HK${latest_sma:.2f} (最新值)"
                )
                if len(indicator_results_structured["sma"]["sma"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["sma"]["sma"][-1]
                        > indicator_results_structured["sma"]["sma"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(indicator_results_structured['sma']['sma'])} 个数据点)"
                    )
            elif indicator_results_structured["sma"]["error"]:
                report_sections.append(
                    f"- 港股SMA: 错误 - {indicator_results_structured['sma']['error']}"
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
            rsi_output = await calculate_hk_stock_rsi.run(
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
                    f"- 港股RSI({indicator_results_structured['rsi']['period']}): {latest_rsi:.2f}"
                )
                if latest_rsi > 70:
                    report_sections.append("  - 注意: RSI表明超买状态 (>70)")
                elif latest_rsi < 30:
                    report_sections.append("  - 注意: RSI表明超卖状态 (<30)")

                if len(indicator_results_structured["rsi"]["rsi"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["rsi"]["rsi"][-1]
                        > indicator_results_structured["rsi"]["rsi"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(indicator_results_structured['rsi']['rsi'])} 个数据点)"
                    )
            elif indicator_results_structured["rsi"]["error"]:
                report_sections.append(
                    f"- 港股RSI: 错误 - {indicator_results_structured['rsi']['error']}"
                )

        # 合成报告
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base,
                error="无法计算任何指标数据",
            )

        report_title = f"港股技术分析报告 - {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} 个数据点:\n"
        report_text = report_title + "\n".join(report_sections)

        await ctx.info(
            f"Successfully generated HK stock comprehensive report for {inputs.symbol}"
        )
        return ComprehensiveAnalysisOutput(
            **output_base,
            report_text=report_text,
            structured_data=indicator_results_structured,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(
            f"Error in HK stock comprehensive report for {inputs.symbol}: {e}",
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"港股综合分析报告生成错误: {str(e)}"
        )


async def _pre_filter_a_stocks(
    realtime_data: List[Dict[str, Any]], criteria: RecommendationCriteria
) -> List[Dict[str, Any]]:
    """
    A股数据预筛选

    Args:
        realtime_data: 实时数据列表
        criteria: 筛选条件

    Returns:
        预筛选后的数据列表
    """
    pre_filtered = []

    for stock_data in realtime_data:
        symbol = stock_data.get("symbol", "")
        name = stock_data.get("name", "")
        price = stock_data.get("price", 0)
        change_percent = stock_data.get("pct_change", 0)
        market_cap = stock_data.get("market_cap", 0) / 100000000  # 转换为亿
        volume = stock_data.get("volume", 0)
        pe_ratio = stock_data.get("pe_ratio", 0)
        pb_ratio = stock_data.get("pb_ratio", 0)

        # 基础数据有效性检查
        if not symbol or not name or price <= 0 or volume <= 0:
            continue

        # 排除ST、退市风险等股票
        if any(flag in name for flag in ["ST", "*ST", "退", "N ", "C "]):
            continue

        # 粗筛选条件
        passed_pre_filter = True

        # 涨跌幅粗筛选
        if (
            criteria.price_change_min is not None
            and change_percent < criteria.price_change_min
        ):
            passed_pre_filter = False
        if (
            criteria.price_change_max is not None
            and change_percent > criteria.price_change_max
        ):
            passed_pre_filter = False

        # 市值粗筛选
        if criteria.market_cap_min is not None and market_cap < criteria.market_cap_min:
            passed_pre_filter = False
        if criteria.market_cap_max is not None and market_cap > criteria.market_cap_max:
            passed_pre_filter = False

        # PE粗筛选
        if (
            criteria.pe_ratio_min is not None
            and pe_ratio > 0
            and pe_ratio < criteria.pe_ratio_min
        ):
            passed_pre_filter = False
        if (
            criteria.pe_ratio_max is not None
            and pe_ratio > 0
            and pe_ratio > criteria.pe_ratio_max
        ):
            passed_pre_filter = False

        # 排除异常数据
        if abs(change_percent) > 11:  # A股涨跌停限制
            passed_pre_filter = False
        if market_cap > 0 and market_cap < 10:  # 排除过小市值
            passed_pre_filter = False
        if pe_ratio > 300:  # 排除异常高PE
            passed_pre_filter = False
        if pb_ratio > 50:  # 排除异常高PB
            passed_pre_filter = False

        # 优先选择有一定流动性的股票
        if price * volume < 1000000:  # 成交额太小的股票
            passed_pre_filter = False

        if passed_pre_filter:
            # 添加一些质量评分用于后续排序
            quality_score = 0

            # 市值评分
            if market_cap >= 100:
                quality_score += 30
            elif market_cap >= 50:
                quality_score += 20
            elif market_cap >= 20:
                quality_score += 10

            # 估值评分
            if 0 < pe_ratio <= 30:
                quality_score += 20
            elif 30 < pe_ratio <= 50:
                quality_score += 10

            # 成交活跃度评分
            turnover = price * volume
            if turnover >= 50000000:  # 5000万以上成交额
                quality_score += 20
            elif turnover >= 10000000:  # 1000万以上成交额
                quality_score += 10

            stock_data["quality_score"] = quality_score
            pre_filtered.append(stock_data)

    return pre_filtered


async def _select_candidate_hk_stocks(
    hk_mapping: Dict[str, str], max_count: int = 100
) -> List[tuple]:
    """
    智能选择港股候选股票

    Args:
        hk_mapping: 港股名称到代码的映射
        max_count: 最大候选数量

    Returns:
        候选股票列表 [(name, symbol), ...]
    """
    candidate_stocks = []

    # 优先级规则
    for name, symbol in hk_mapping.items():
        if not symbol or len(symbol) != 5:
            continue

        try:
            symbol_int = int(symbol)
            priority_score = 0

            # 1. 主板蓝筹股（代码小）- 最高优先级
            if symbol_int <= 1000:
                priority_score += 100

            # 2. 知名公司关键词
            high_priority_keywords = [
                "腾讯",
                "阿里",
                "美团",
                "小米",
                "比亚迪",
                "建设银行",
                "工商银行",
                "中国平安",
                "汇丰",
                "友邦",
                "恒生",
                "中国移动",
                "港交所",
            ]
            if any(keyword in name for keyword in high_priority_keywords):
                priority_score += 80

            # 3. 行业关键词
            industry_keywords = [
                "银行",
                "保险",
                "电信",
                "石油",
                "科技",
                "地产",
                "医药",
                "汽车",
            ]
            if any(keyword in name for keyword in industry_keywords):
                priority_score += 50

            # 4. 新经济股票
            if 3000 <= symbol_int <= 3999 or 9000 <= symbol_int <= 9999:
                priority_score += 60

            # 5. 包含"中国"、"香港"等的公司
            if any(keyword in name for keyword in ["中国", "香港", "国际"]):
                priority_score += 40

            # 6. 避免过于小众的股票
            avoid_keywords = ["发展", "投资", "集团", "控股", "有限"]
            if all(keyword not in name for keyword in avoid_keywords):
                priority_score += 20

            # 7. 代码规律性加分
            if symbol_int <= 2000:  # 早期上市的公司
                priority_score += 30

            candidate_stocks.append((name, symbol, priority_score))

        except ValueError:
            continue

    # 按优先级排序
    candidate_stocks.sort(key=lambda x: x[2], reverse=True)

    # 返回前max_count个，去掉优先级分数
    return [(name, symbol) for name, symbol, _ in candidate_stocks[:max_count]]


async def _calculate_stock_score(
    ctx: Context,
    symbol: str,
    name: str,
    realtime_data: Dict[str, Any],
    market_type: str = "a_stock",
) -> Optional[StockScore]:
    """
    计算单只股票的综合评分

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        name: 股票名称
        realtime_data: 实时数据
        market_type: 市场类型

    Returns:
        StockScore对象或None
    """
    try:
        # 基础数据
        current_price = realtime_data.get("price", 0)
        change_percent = realtime_data.get("pct_change", 0)
        volume = realtime_data.get("volume", 0)
        market_cap = realtime_data.get("market_cap", 0) / 100000000  # 转换为亿
        pe_ratio = realtime_data.get("pe_ratio", 0)
        pb_ratio = realtime_data.get("pb_ratio", 0)

        recommendation_reasons = []
        technical_score = 0
        fundamental_score = 0

        # 计算技术指标
        try:
            # RSI计算
            rsi_input = RsiInput(
                symbol=symbol, timeframe="1d", period=14, history_len=1
            )
            if market_type == "a_stock":
                rsi_output = await calculate_a_stock_rsi.fn(ctx, rsi_input)
            else:
                rsi_output = await calculate_hk_stock_rsi.fn(ctx, rsi_input)

            rsi_value = None
            if (
                hasattr(rsi_output, "rsi")
                and rsi_output.rsi
                and len(rsi_output.rsi) > 0
            ):
                rsi_value = rsi_output.rsi[-1]

                # RSI评分
                if 30 <= rsi_value <= 70:  # 中性区域
                    technical_score += 25
                    recommendation_reasons.append(f"RSI处于健康区域({rsi_value:.1f})")
                elif rsi_value < 30:  # 超卖
                    technical_score += 35
                    recommendation_reasons.append(f"RSI超卖，可能反弹({rsi_value:.1f})")
                elif rsi_value > 80:  # 过度超买
                    technical_score -= 10

        except Exception as e:
            await ctx.warning(f"计算{symbol} RSI时出错: {e}")
            rsi_value = None

        # MACD信号
        macd_signal = None
        try:
            macd_input = MacdInput(symbol=symbol, timeframe="1d", history_len=2)
            if market_type == "a_stock":
                macd_output = await calculate_a_stock_macd.fn(ctx, macd_input)
            else:
                # 港股暂时跳过MACD
                macd_output = None

            if (
                macd_output
                and hasattr(macd_output, "macd")
                and macd_output.macd
                and len(macd_output.macd) >= 2
                and hasattr(macd_output, "signal")
                and macd_output.signal
                and len(macd_output.signal) >= 2
            ):
                current_macd = macd_output.macd[-1]
                prev_macd = macd_output.macd[-2]
                current_signal = macd_output.signal[-1]
                prev_signal = macd_output.signal[-2]

                # 判断金叉死叉
                if prev_macd <= prev_signal and current_macd > current_signal:
                    macd_signal = "金叉"
                    technical_score += 30
                    recommendation_reasons.append("MACD出现金叉信号")
                elif prev_macd >= prev_signal and current_macd < current_signal:
                    macd_signal = "死叉"
                    technical_score -= 20
                elif current_macd > current_signal:
                    macd_signal = "多头"
                    technical_score += 10
                else:
                    macd_signal = "空头"

        except Exception as e:
            await ctx.warning(f"计算{symbol} MACD时出错: {e}")

        # SMA位置
        sma_position = None
        try:
            sma_input = SmaInput(
                symbol=symbol, timeframe="1d", period=20, history_len=1
            )
            if market_type == "a_stock":
                sma_output = await calculate_a_stock_sma.fn(ctx, sma_input)
            else:
                sma_output = await calculate_hk_stock_sma.fn(ctx, sma_input)
            if (
                hasattr(sma_output, "sma")
                and sma_output.sma
                and len(sma_output.sma) > 0
            ):
                sma_value = sma_output.sma[-1]
                if current_price > sma_value:
                    sma_position = "上方"
                    technical_score += 15
                    recommendation_reasons.append("价格位于20日均线上方")
                else:
                    sma_position = "下方"

        except Exception as e:
            await ctx.warning(f"计算{symbol} SMA时出错: {e}")

        # 成交量评分
        volume_ratio = None
        try:
            # 简单的成交量评估（实际中可以计算与历史平均的比值）
            if volume > 0:
                # 假设正常成交量，这里可以改进为与历史平均比较
                volume_ratio = 1.0
                if change_percent > 0 and volume > 0:  # 放量上涨
                    technical_score += 10
                    recommendation_reasons.append("放量上涨")
        except Exception:
            volume_ratio = None

        # 基本面评分
        if pe_ratio > 0:
            if 10 <= pe_ratio <= 25:  # 合理估值
                fundamental_score += 30
                recommendation_reasons.append(f"估值合理(PE={pe_ratio:.1f})")
            elif pe_ratio < 10:  # 低估值
                fundamental_score += 40
                recommendation_reasons.append(f"低估值(PE={pe_ratio:.1f})")
            elif pe_ratio > 50:  # 高估值
                fundamental_score -= 20

        if pb_ratio > 0:
            if pb_ratio < 2:  # 低PB
                fundamental_score += 20
                recommendation_reasons.append(f"低PB值({pb_ratio:.1f})")

        # 市值评分
        if market_cap > 0:
            if 50 <= market_cap <= 1000:  # 中等市值
                fundamental_score += 15
            elif market_cap > 1000:  # 大市值
                fundamental_score += 10

        # 涨跌幅评分
        if -3 <= change_percent <= 7:  # 适度变化
            technical_score += 10
        elif change_percent > 9.5:  # 涨幅过大
            technical_score -= 20
        elif change_percent < -8:  # 跌幅过大
            technical_score -= 10

        # 确保评分在合理范围内
        technical_score = max(0, min(100, technical_score))
        fundamental_score = max(0, min(100, fundamental_score))
        overall_score = technical_score * 0.6 + fundamental_score * 0.4

        return StockScore(
            symbol=symbol,
            name=name,
            current_price=current_price,
            change_percent=change_percent,
            volume_ratio=volume_ratio,
            rsi=rsi_value,
            macd_signal=macd_signal,
            sma_position=sma_position,
            market_cap=market_cap if market_cap > 0 else None,
            pe_ratio=pe_ratio if pe_ratio > 0 else None,
            pb_ratio=pb_ratio if pb_ratio > 0 else None,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            overall_score=overall_score,
            recommendation_reason=recommendation_reasons or ["基础数据正常"],
        )

    except Exception as e:
        await ctx.error(f"计算{symbol}评分时出错: {e}")
        return None


async def _filter_stocks_by_criteria(
    stocks: List[StockScore], criteria: RecommendationCriteria
) -> List[StockScore]:
    """
    根据筛选条件过滤股票

    Args:
        stocks: 股票评分列表
        criteria: 筛选条件

    Returns:
        过滤后的股票列表
    """
    filtered_stocks = []

    for stock in stocks:
        # RSI筛选
        if criteria.rsi_min is not None and (
            stock.rsi is None or stock.rsi < criteria.rsi_min
        ):
            continue
        if criteria.rsi_max is not None and (
            stock.rsi is None or stock.rsi > criteria.rsi_max
        ):
            continue

        # 涨跌幅筛选
        if (
            criteria.price_change_min is not None
            and stock.change_percent < criteria.price_change_min
        ):
            continue
        if (
            criteria.price_change_max is not None
            and stock.change_percent > criteria.price_change_max
        ):
            continue

        # 市值筛选
        if criteria.market_cap_min is not None and (
            stock.market_cap is None or stock.market_cap < criteria.market_cap_min
        ):
            continue
        if criteria.market_cap_max is not None and (
            stock.market_cap is None or stock.market_cap > criteria.market_cap_max
        ):
            continue

        # PE筛选
        if criteria.pe_ratio_min is not None and (
            stock.pe_ratio is None or stock.pe_ratio < criteria.pe_ratio_min
        ):
            continue
        if criteria.pe_ratio_max is not None and (
            stock.pe_ratio is None or stock.pe_ratio > criteria.pe_ratio_max
        ):
            continue

        # 技术形态筛选
        if criteria.require_golden_cross and stock.macd_signal != "金叉":
            continue
        if criteria.require_above_sma and stock.sma_position != "上方":
            continue

        filtered_stocks.append(stock)

    return filtered_stocks


async def recommend_a_stocks(
    ctx: Context, inputs: StockRecommendationInput
) -> StockRecommendationOutput:
    """
    推荐A股股票

    Args:
        ctx: FastMCP上下文对象
        inputs: 推荐输入参数，包含market_type（市场类型，默认a_stock）、criteria（筛选条件）、limit（返回数量，默认20）、timeframe（时间框架，默认1d）

    Returns:
        StockRecommendationOutput: 推荐结果，包含market_type、total_analyzed、recommendations推荐列表、criteria_used使用条件或error错误信息
    """
    await ctx.info(f"开始推荐A股股票，限制数量: {inputs.limit}")

    try:
        # 获取实时股票数据
        realtime_data = await fetch_stock_realtime_data(ctx, None)  # 获取所有股票

        if not realtime_data:
            return StockRecommendationOutput(
                market_type="a_stock",
                total_analyzed=0,
                recommendations=[],
                criteria_used=inputs.criteria.model_dump(),
                error="无法获取股票实时数据",
            )

        await ctx.info(f"获取到{len(realtime_data)}只A股数据，开始初步筛选...")

        # 第一步：基于汇总数据进行粗筛选
        pre_filtered = []
        for stock_data in realtime_data:
            symbol = stock_data.get("symbol", "")
            name = stock_data.get("name", "")
            price = stock_data.get("price", 0)
            change_percent = stock_data.get("pct_change", 0)
            market_cap = stock_data.get("market_cap", 0) / 100000000  # 转换为亿
            volume = stock_data.get("volume", 0)
            pe_ratio = stock_data.get("pe_ratio", 0)

            # 基础数据有效性检查
            if not symbol or not name or price <= 0:
                continue

            # 粗筛选条件
            passed_pre_filter = True

            # 涨跌幅粗筛选
            if (
                inputs.criteria.price_change_min is not None
                and change_percent < inputs.criteria.price_change_min
            ):
                passed_pre_filter = False
            if (
                inputs.criteria.price_change_max is not None
                and change_percent > inputs.criteria.price_change_max
            ):
                passed_pre_filter = False

            # 市值粗筛选
            if (
                inputs.criteria.market_cap_min is not None
                and market_cap < inputs.criteria.market_cap_min
            ):
                passed_pre_filter = False
            if (
                inputs.criteria.market_cap_max is not None
                and market_cap > inputs.criteria.market_cap_max
            ):
                passed_pre_filter = False

            # PE粗筛选
            if (
                inputs.criteria.pe_ratio_min is not None
                and pe_ratio > 0
                and pe_ratio < inputs.criteria.pe_ratio_min
            ):
                passed_pre_filter = False
            if (
                inputs.criteria.pe_ratio_max is not None
                and pe_ratio > 0
                and pe_ratio > inputs.criteria.pe_ratio_max
            ):
                passed_pre_filter = False

            # 排除异常数据
            if change_percent > 10 or change_percent < -10:  # 排除异常涨跌幅
                passed_pre_filter = False
            if market_cap > 0 and market_cap < 10:  # 排除过小市值
                passed_pre_filter = False
            if volume <= 0:  # 排除无成交量
                passed_pre_filter = False

            if passed_pre_filter:
                pre_filtered.append(stock_data)

        await ctx.info(
            f"粗筛选完成，从{len(realtime_data)}只股票中筛选出{len(pre_filtered)}只候选股票"
        )

        # 按成交量和市值排序，优先分析活跃度高的股票
        pre_filtered.sort(
            key=lambda x: (x.get("volume", 0) * x.get("price", 0)), reverse=True
        )

        # 限制详细分析的数量（取前150只最活跃的股票）
        analysis_data = pre_filtered[: min(100, len(pre_filtered))]

        # 第二步：对粗筛选后的股票进行详细技术分析
        scored_stocks = []
        for i, stock_data in enumerate(analysis_data):
            if i % 30 == 0:  # 每处理30只股票报告一次进度
                await ctx.info(f"详细分析进度：{i}/{len(analysis_data)}只股票...")

            symbol = stock_data.get("symbol", "")
            name = stock_data.get("name", "")

            score = await _calculate_stock_score(
                ctx, symbol, name, stock_data, "a_stock"
            )
            if score and score.overall_score > 25:  # 降低门槛以获得更多候选
                scored_stocks.append(score)
            await asyncio.sleep(1)

        await ctx.info(f"详细分析完成，筛选到{len(scored_stocks)}只高质量候选股票")

        # 第三步：应用剩余的精细筛选条件
        filtered_stocks = await _filter_stocks_by_criteria(
            scored_stocks, inputs.criteria
        )

        await ctx.info(f"精细筛选后剩余{len(filtered_stocks)}只股票")

        # 按综合评分排序
        filtered_stocks.sort(key=lambda x: x.overall_score, reverse=True)

        # 限制返回数量
        recommendations = filtered_stocks[: inputs.limit]

        await ctx.info(f"A股推荐完成，返回{len(recommendations)}只股票")

        return StockRecommendationOutput(
            market_type="a_stock",
            total_analyzed=len(realtime_data),
            recommendations=recommendations,
            criteria_used=inputs.criteria.model_dump(),
        )

    except Exception as e:
        await ctx.error(f"A股推荐过程出错: {e}")
        return StockRecommendationOutput(
            market_type="a_stock",
            total_analyzed=0,
            recommendations=[],
            criteria_used=inputs.criteria.model_dump(),
            error=f"推荐过程出错: {str(e)}",
        )


async def recommend_hk_stocks(
    ctx: Context, inputs: StockRecommendationInput
) -> StockRecommendationOutput:
    """
    推荐港股股票

    Args:
        ctx: FastMCP上下文对象
        inputs: 推荐输入参数，包含market_type（市场类型，默认hk_stock）、criteria（筛选条件）、limit（返回数量，默认20）、timeframe（时间框架，默认1d）

    Returns:
        StockRecommendationOutput: 推荐结果，包含market_type、total_analyzed、recommendations推荐列表、criteria_used使用条件或error错误信息
    """
    await ctx.info(f"开始推荐港股股票，限制数量: {inputs.limit}")

    try:
        # 获取港股映射表
        hk_mapping = await fetch_hk_stock_symbol_mapping(ctx)
        if not hk_mapping:
            return StockRecommendationOutput(
                market_type="hk_stock",
                total_analyzed=0,
                recommendations=[],
                criteria_used=inputs.criteria.model_dump(),
                error="无法获取港股代码映射",
            )

        await ctx.info(f"获取到{len(hk_mapping)}只港股代码，开始动态筛选...")

        # 第一步：智能选择候选股票
        candidate_stocks = await _select_candidate_hk_stocks(hk_mapping, max_count=120)

        await ctx.info(f"筛选出{len(candidate_stocks)}只候选港股，开始获取实时数据...")

        # 第二步：获取候选股票的基础数据进行粗筛选
        pre_filtered = []
        batch_size = 20  # 分批处理，避免一次性处理太多

        for i in range(
            0, min(100, len(candidate_stocks)), batch_size
        ):  # 最多处理100只股票
            batch = candidate_stocks[i : i + batch_size]
            await ctx.info(f"处理第{i // batch_size + 1}批港股数据 ({len(batch)}只)...")

            for name, symbol in batch:
                try:
                    # 获取最近几天的数据
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

                    stock_data_list = await fetch_hk_stock_data(
                        ctx, symbol, "daily", start_date, end_date
                    )

                    if stock_data_list and len(stock_data_list) > 0:
                        latest_data = stock_data_list[-1]

                        # 计算涨跌幅（如果有多天数据）
                        change_percent = latest_data.get("pct_change", 0)
                        if change_percent == 0 and len(stock_data_list) > 1:
                            prev_close = stock_data_list[-2]["close"]
                            current_close = latest_data["close"]
                            if prev_close > 0:
                                change_percent = (
                                    (current_close - prev_close) / prev_close
                                ) * 100

                        # 构造类似实时数据的格式
                        mock_realtime_data = {
                            "symbol": symbol,
                            "name": name,
                            "price": latest_data["close"],
                            "pct_change": change_percent,
                            "open": latest_data["open"],
                            "high": latest_data["high"],
                            "low": latest_data["low"],
                            "volume": latest_data["volume"],
                            "amount": latest_data.get("amount", 0),
                            "market_cap": 0,  # 港股市值数据需要单独获取
                            "pe_ratio": 0,
                            "pb_ratio": 0,
                        }

                        # 基础数据有效性和粗筛选
                        price = mock_realtime_data["price"]
                        volume = mock_realtime_data["volume"]

                        if price > 0 and volume > 0:
                            # 涨跌幅筛选
                            if (
                                inputs.criteria.price_change_min is not None
                                and change_percent < inputs.criteria.price_change_min
                            ):
                                continue
                            if (
                                inputs.criteria.price_change_max is not None
                                and change_percent > inputs.criteria.price_change_max
                            ):
                                continue

                            # 排除异常数据
                            if -15 <= change_percent <= 15:  # 港股正常涨跌幅范围
                                pre_filtered.append(mock_realtime_data)

                except Exception as e:
                    await ctx.warning(f"获取港股{symbol}数据时出错: {e}")
                    continue

        await ctx.info(f"港股粗筛选完成，筛选出{len(pre_filtered)}只候选股票")

        if not pre_filtered:
            return StockRecommendationOutput(
                market_type="hk_stock",
                total_analyzed=len(candidate_stocks),
                recommendations=[],
                criteria_used=inputs.criteria.model_dump(),
                error="没有符合基础条件的港股数据",
            )

        # 按成交额排序（价格*成交量），优先分析活跃股票
        pre_filtered.sort(key=lambda x: x["price"] * x["volume"], reverse=True)

        # 第三步：对筛选后的股票进行详细技术分析
        scored_stocks = []
        analysis_limit = min(50, len(pre_filtered))  # 最多详细分析50只

        for i, stock_data in enumerate(pre_filtered[:analysis_limit]):
            await ctx.info(
                f"详细分析港股进度：{i + 1}/{analysis_limit} - {stock_data['name']}({stock_data['symbol']})"
            )

            symbol = stock_data["symbol"]
            name = stock_data["name"]

            score = await _calculate_stock_score(
                ctx, symbol, name, stock_data, "hk_stock"
            )
            if score and score.overall_score > 15:  # 港股门槛较低
                scored_stocks.append(score)
            await asyncio.sleep(1)

        await ctx.info(f"港股详细分析完成，筛选到{len(scored_stocks)}只高质量候选股票")

        # 第四步：应用剩余的精细筛选条件
        filtered_stocks = await _filter_stocks_by_criteria(
            scored_stocks, inputs.criteria
        )

        # 按综合评分排序
        filtered_stocks.sort(key=lambda x: x.overall_score, reverse=True)

        # 限制返回数量
        recommendations = filtered_stocks[: inputs.limit]

        await ctx.info(f"港股推荐完成，返回{len(recommendations)}只股票")

        return StockRecommendationOutput(
            market_type="hk_stock",
            total_analyzed=len(candidate_stocks),
            recommendations=recommendations,
            criteria_used=inputs.criteria.model_dump(),
        )

    except Exception as e:
        await ctx.error(f"港股推荐过程出错: {e}")
        return StockRecommendationOutput(
            market_type="hk_stock",
            total_analyzed=0,
            recommendations=[],
            criteria_used=inputs.criteria.model_dump(),
            error=f"推荐过程出错: {str(e)}",
        )


# async def get_stock_recommendation_presets(ctx: Context) -> Dict[str, Any]:
#     """
#     获取预设的推荐筛选条件

#     Args:
#         ctx: FastMCP上下文对象

#     Returns:
#         dict: 包含各种预设筛选条件的字典
#     """
#     await ctx.info("获取股票推荐预设条件")

#     presets = {
#         "value_stocks": {
#             "name": "价值股推荐",
#             "description": "寻找低估值、基本面良好的股票",
#             "criteria": RecommendationCriteria(
#                 pe_ratio_min=5,
#                 pe_ratio_max=20,
#                 rsi_min=30,
#                 rsi_max=60,
#                 market_cap_min=50,
#             ),
#         },
#         "growth_stocks": {
#             "name": "成长股推荐",
#             "description": "寻找技术形态良好、有上涨动能的股票",
#             "criteria": RecommendationCriteria(
#                 rsi_min=40,
#                 rsi_max=75,
#                 require_above_sma=True,
#                 price_change_min=-2,
#                 price_change_max=8,
#             ),
#         },
#         "oversold_bounce": {
#             "name": "超卖反弹",
#             "description": "寻找超卖后可能反弹的股票",
#             "criteria": RecommendationCriteria(
#                 rsi_min=20, rsi_max=35, price_change_min=-10, price_change_max=2
#             ),
#         },
#         "momentum_stocks": {
#             "name": "动量股推荐",
#             "description": "寻找技术指标向好的强势股",
#             "criteria": RecommendationCriteria(
#                 require_golden_cross=True,
#                 require_above_sma=True,
#                 rsi_min=50,
#                 rsi_max=80,
#             ),
#         },
#         "large_cap_stable": {
#             "name": "大盘稳健股",
#             "description": "大市值稳健型股票",
#             "criteria": RecommendationCriteria(
#                 market_cap_min=500,
#                 pe_ratio_min=8,
#                 pe_ratio_max=25,
#                 rsi_min=35,
#                 rsi_max=65,
#             ),
#         },
#     }

#     # 转换为可序列化的格式
#     serializable_presets = {}
#     for key, preset in presets.items():
#         serializable_presets[key] = {
#             "name": preset["name"],
#             "description": preset["description"],
#             "criteria": preset["criteria"].model_dump(),
#         }

#     return {
#         "success": True,
#         "presets": serializable_presets,
#         "total_presets": len(presets),
#     }


@mcp.tool()
async def query_stock_recommendations_db(
    ctx: Context,
    preset_name: str = "value_stocks",
    market_type: str = "a_stock",
    limit: int = 10,
) -> dict:
    """
    从数据库查询股票推荐结果

    Args:
        ctx: FastMCP上下文对象
        preset_name: 策略名称 (value_stocks, growth_stocks, oversold_bounce, momentum_stocks, large_cap_stable)
                    value_stocks - 价值股推荐, growth_stocks - 成长股推荐, oversold_bounce - 超卖反弹, momentum_stocks - 动量股推荐, large_cap_stable - 大盘稳健股
        market_type: 市场类型 (a_stock, not support hk_stock us_stock crypto)
        limit: 返回数量限制

    Returns:
        dict: 推荐结果
    """
    from utils.stock_analysis_scheduler import stock_scheduler

    await ctx.info(f"查询{market_type}的{preset_name}推荐结果")

    try:
        results = stock_scheduler.db.get_recommendations(
            preset_name, market_type, limit
        )

        if not results:
            return {
                "success": False,
                "preset_name": preset_name,
                "market_type": market_type,
                "message": "未找到推荐数据，请先运行分析任务",
            }

        # 格式化推荐结果
        recommendations = []
        for rec in results:
            formatted_rec = {
                "symbol": rec["symbol"],
                "name": rec["name"],
                "current_price": rec["current_price"],
                "change_percent": rec["change_percent"],
                "overall_score": rec["overall_score"],
                "technical_score": rec["technical_score"],
                "fundamental_score": rec["fundamental_score"],
                "rsi": rec["rsi_14"],
                "macd_signal": rec["macd_signal_type"],
                "sma_position": rec["sma_position"],
                "market_cap": rec["market_cap"],
                "pe_ratio": rec["pe_ratio"],
                "pb_ratio": rec["pb_ratio"],
                "recommendation_reasons": rec["recommendation_reasons"],
                "analysis_date": rec["analysis_date"],
            }
            recommendations.append(formatted_rec)

        result = {
            "success": True,
            "preset_name": preset_name,
            "market_type": market_type,
            "total_found": len(results),
            "recommendations": recommendations,
            "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        await ctx.info(f"返回{len(recommendations)}个推荐结果")
        return result

    except Exception as e:
        error_msg = f"查询推荐失败: {str(e)}"
        await ctx.error(error_msg)
        return {
            "success": False,
            "preset_name": preset_name,
            "market_type": market_type,
            "error": error_msg,
        }
