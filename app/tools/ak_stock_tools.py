import numpy as np
import talib
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta

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

# 导入现有的AKShare服务函数
from services.akshare_service import (
    fetch_stock_hist_data,
    fetch_stock_realtime_data,
    fetch_hk_stock_data,
    search_stock_by_name,
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
            stock_data = await fetch_hk_stock_data(ctx, symbol, start_date, end_date)
        else:  # a_stock
            stock_data = await fetch_stock_hist_data(
                ctx, symbol, period, start_date, end_date
            )

        if not stock_data or len(stock_data) < required_candles:
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

        if not stock_data or len(stock_data) < required_candles:
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


# ==================== A股数据获取工具 ====================


@mcp.tool()
async def get_a_stock_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    获取A股K线数据

    Args:
        inputs.symbol: 股票代码 (如: 000001, 600519)
        inputs.timeframe: 时间框架 (daily, weekly, monthly)
        inputs.limit: 数据条数限制
        inputs.adjust: 复权类型 (qfq前复权, hfq后复权, ""不复权)
    """
    await ctx.info(
        f"Fetching A-stock candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe.value, "daily")

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
        inputs.symbol: 股票代码 (如: 000001)
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
        query: 搜索关键词 (如: 平安银行, 000001, 茅台)
        limit: 结果数量限制
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
async def get_a_stock_ticker(ctx: Context, inputs: TickerInput) -> TickerOutput:
    """
    获取A股详细行情数据

    Args:
        inputs.symbol: 股票代码
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
        inputs.symbol: 股票代码 (如: 00700, 09988)
        inputs.timeframe: 时间框架 (daily, weekly, monthly)
        inputs.limit: 数据条数限制
    """
    await ctx.info(
        f"Fetching HK stock candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=inputs.limit + 50)).strftime(
            "%Y%m%d"
        )

        # 使用现有的港股服务获取数据
        stock_data = await fetch_hk_stock_data(
            ctx=ctx, symbol=inputs.symbol, start_date=start_date, end_date=end_date
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
        inputs.symbol: 股票代码 (如: 00700)
    """
    await ctx.info(f"Fetching HK stock current price for {inputs.symbol}")

    try:
        # 获取最新的历史数据作为当前价格
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

        stock_data = await fetch_hk_stock_data(ctx, inputs.symbol, start_date, end_date)

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
    """计算A股简单移动平均线 (SMA)"""
    await ctx.info(
        f"Calculating A-stock SMA for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # SMA需要period + history_len - 1个数据点
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, "daily", required_candles, "close", "a_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
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
    """计算A股相对强弱指数 (RSI)"""
    await ctx.info(
        f"Calculating A-stock RSI for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # RSI需要period + history_len个数据点
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, "daily", required_candles, "close", "a_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
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
    """计算A股MACD指标"""
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
        required_candles = (
            inputs.slow_period + inputs.signal_period + inputs.history_len + 10
        )
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, "daily", required_candles, "close", "a_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
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
    """计算A股布林带 (Bollinger Bands)"""
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
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, "daily", required_candles, "close", "a_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
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
    """计算A股平均真实波幅 (ATR)"""
    await ctx.info(
        f"Calculating A-stock ATR for {inputs.symbol}, Period: {inputs.period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        required_candles = inputs.period + inputs.history_len - 1

        price_data = await _fetch_multi_series_data(
            ctx,
            inputs.symbol,
            "daily",
            required_candles,
            ["high", "low", "close"],
            "a_stock",
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
    """计算港股简单移动平均线 (SMA)"""
    await ctx.info(
        f"Calculating HK stock SMA for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, "daily", required_candles, "close", "hk_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
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
    """计算港股相对强弱指数 (RSI)"""
    await ctx.info(
        f"Calculating HK stock RSI for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, "daily", required_candles, "close", "hk_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
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
            sma_output = await calculate_a_stock_sma(ctx, sma_input)
            indicator_results_structured["sma"] = sma_output.model_dump()

            if sma_output.sma is not None and len(sma_output.sma) > 0:
                latest_sma = sma_output.sma[-1]
                report_sections.append(
                    f"- A股SMA({sma_output.period}): ¥{latest_sma:.2f} (最新值)"
                )
                if len(sma_output.sma) > 1:
                    trend = "↗" if sma_output.sma[-1] > sma_output.sma[-2] else "↘"
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(sma_output.sma)} 个数据点)"
                    )
            elif sma_output.error:
                report_sections.append(f"- A股SMA: 错误 - {sma_output.error}")

        # RSI分析
        if "RSI" in indicators_to_run:
            rsi_period = inputs.rsi_period or settings.DEFAULT_RSI_PERIOD
            rsi_input = RsiInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=rsi_period,
            )
            rsi_output = await calculate_a_stock_rsi(ctx, rsi_input)
            indicator_results_structured["rsi"] = rsi_output.model_dump()

            if rsi_output.rsi is not None and len(rsi_output.rsi) > 0:
                latest_rsi = rsi_output.rsi[-1]
                report_sections.append(
                    f"- A股RSI({rsi_output.period}): {latest_rsi:.2f}"
                )
                if latest_rsi > 70:
                    report_sections.append("  - 注意: RSI表明超买状态 (>70)")
                elif latest_rsi < 30:
                    report_sections.append("  - 注意: RSI表明超卖状态 (<30)")

                if len(rsi_output.rsi) > 1:
                    trend = "↗" if rsi_output.rsi[-1] > rsi_output.rsi[-2] else "↘"
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(rsi_output.rsi)} 个数据点)"
                    )
            elif rsi_output.error:
                report_sections.append(f"- A股RSI: 错误 - {rsi_output.error}")

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
            macd_output = await calculate_a_stock_macd(ctx, macd_input)
            indicator_results_structured["macd"] = macd_output.model_dump()

            if (
                macd_output.macd is not None
                and len(macd_output.macd) > 0
                and macd_output.signal is not None
                and len(macd_output.signal) > 0
                and macd_output.histogram is not None
                and len(macd_output.histogram) > 0
            ):
                latest_macd = macd_output.macd[-1]
                latest_signal = macd_output.signal[-1]
                latest_hist = macd_output.histogram[-1]

                report_sections.append(
                    f"- A股MACD({macd_output.fast_period},{macd_output.slow_period},{macd_output.signal_period}): "
                    f"MACD: {latest_macd:.4f}, 信号线: {latest_signal:.4f}, 柱状图: {latest_hist:.4f}"
                )

                if latest_hist > 0 and latest_macd > latest_signal:
                    report_sections.append("  - 注意: MACD柱状图为正，可能有看涨动量")
                elif latest_hist < 0 and latest_macd < latest_signal:
                    report_sections.append("  - 注意: MACD柱状图为负，可能有看跌动量")

            elif macd_output.error:
                report_sections.append(f"- A股MACD: 错误 - {macd_output.error}")

        # Bollinger Bands分析
        if "BBANDS" in indicators_to_run:
            bbands_input = BbandsInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.bbands_period or settings.DEFAULT_BBANDS_PERIOD,
            )
            bbands_output = await calculate_a_stock_bbands(ctx, bbands_input)
            indicator_results_structured["bbands"] = bbands_output.model_dump()

            if (
                bbands_output.upper_band is not None
                and len(bbands_output.upper_band) > 0
                and bbands_output.middle_band is not None
                and len(bbands_output.middle_band) > 0
                and bbands_output.lower_band is not None
                and len(bbands_output.lower_band) > 0
            ):
                latest_upper = bbands_output.upper_band[-1]
                latest_middle = bbands_output.middle_band[-1]
                latest_lower = bbands_output.lower_band[-1]

                report_sections.append(
                    f"- A股布林带({bbands_output.period}): "
                    f"上轨: ¥{latest_upper:.2f}, 中轨: ¥{latest_middle:.2f}, 下轨: ¥{latest_lower:.2f}"
                )
                report_sections.append(f"  - 带宽: ¥{latest_upper - latest_lower:.2f}")

            elif bbands_output.error:
                report_sections.append(f"- A股布林带: 错误 - {bbands_output.error}")

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
            sma_output = await calculate_hk_stock_sma(ctx, sma_input)
            indicator_results_structured["sma"] = sma_output.model_dump()

            if sma_output.sma is not None and len(sma_output.sma) > 0:
                latest_sma = sma_output.sma[-1]
                report_sections.append(
                    f"- 港股SMA({sma_output.period}): HK${latest_sma:.2f} (最新值)"
                )
                if len(sma_output.sma) > 1:
                    trend = "↗" if sma_output.sma[-1] > sma_output.sma[-2] else "↘"
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(sma_output.sma)} 个数据点)"
                    )
            elif sma_output.error:
                report_sections.append(f"- 港股SMA: 错误 - {sma_output.error}")

        # RSI分析
        if "RSI" in indicators_to_run:
            rsi_period = inputs.rsi_period or settings.DEFAULT_RSI_PERIOD
            rsi_input = RsiInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=rsi_period,
            )
            rsi_output = await calculate_hk_stock_rsi(ctx, rsi_input)
            indicator_results_structured["rsi"] = rsi_output.model_dump()

            if rsi_output.rsi is not None and len(rsi_output.rsi) > 0:
                latest_rsi = rsi_output.rsi[-1]
                report_sections.append(
                    f"- 港股RSI({rsi_output.period}): {latest_rsi:.2f}"
                )
                if latest_rsi > 70:
                    report_sections.append("  - 注意: RSI表明超买状态 (>70)")
                elif latest_rsi < 30:
                    report_sections.append("  - 注意: RSI表明超卖状态 (<30)")

                if len(rsi_output.rsi) > 1:
                    trend = "↗" if rsi_output.rsi[-1] > rsi_output.rsi[-2] else "↘"
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(rsi_output.rsi)} 个数据点)"
                    )
            elif rsi_output.error:
                report_sections.append(f"- 港股RSI: 错误 - {rsi_output.error}")

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
        await ctx.error(
            f"Error in HK stock comprehensive report for {inputs.symbol}: {e}",
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"港股综合分析报告生成错误: {str(e)}"
        )
