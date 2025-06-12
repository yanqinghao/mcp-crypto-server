import numpy as np
import talib
from typing import Dict, Optional, List, Any

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
    AdxInput,
    AdxOutput,
    ObvInput,
    ObvOutput,
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
    OrderBookInput,
    OrderBookOutput,
    TradesInput,
    TradesOutput,
    OHLCVCandle,
    OrderBookLevel,
    TradeData,
)

# Import the refined data fetching helpers
from services.crypto_service import (
    fetch_ohlcv_data,
    fetch_ticker_data,
    fetch_order_book,
    fetch_trades,
)
import ccxt  # For exception types
from fastmcp import FastMCP, Context

mcp = FastMCP()


async def _fetch_single_series_data(
    ctx: Context,
    symbol: str,
    timeframe: str,
    required_candles: int,
    series_type: str = "close",
) -> Optional[np.ndarray]:
    """
    获取单个数据序列的通用函数

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号
        timeframe: 时间框架
        required_candles: 需要的K线数量
        series_type: 数据类型 (open, high, low, close, volume)

    Returns:
        numpy数组或None
    """
    try:
        # 获取OHLCV数据，加上缓冲区以确保有足够的数据
        buffer_size = max(50, required_candles // 2)
        fetch_size = required_candles + buffer_size

        ohlcv_data = await fetch_ohlcv_data(ctx, symbol, timeframe, fetch_size)

        if not ohlcv_data or len(ohlcv_data) < required_candles:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(ohlcv_data) if ohlcv_data else 0}, need {required_candles}"
            )
            return None

        # 选择对应的数据列
        series_index_map = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}

        if series_type not in series_index_map:
            await ctx.error(f"Invalid series type: {series_type}")
            return None

        series_index = series_index_map[series_type]
        data_series = np.array([candle[series_index] for candle in ohlcv_data])

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
    timeframe: str,
    required_candles: int,
    series_types: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    """
    获取多个数据序列的通用函数

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号
        timeframe: 时间框架
        required_candles: 需要的K线数量
        series_types: 需要的数据类型列表

    Returns:
        包含各数据序列的字典或None
    """
    try:
        # 获取OHLCV数据
        buffer_size = max(50, required_candles // 2)
        fetch_size = required_candles + buffer_size

        ohlcv_data = await fetch_ohlcv_data(ctx, symbol, timeframe, fetch_size)

        if not ohlcv_data or len(ohlcv_data) < required_candles:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(ohlcv_data) if ohlcv_data else 0}, need {required_candles}"
            )
            return None

        series_index_map = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}

        result = {}
        for series_type in series_types:
            if series_type not in series_index_map:
                await ctx.error(f"Invalid series type: {series_type}")
                return None

            series_index = series_index_map[series_type]
            data_series = np.array([candle[series_index] for candle in ohlcv_data])

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


# --- SMA Tool ---
@mcp.tool()
async def calculate_sma(ctx: Context, inputs: SmaInput) -> SmaOutput:
    """Calculates the Simple Moving Average (SMA) for a trading pair."""
    await ctx.info(
        f"Executing calculate_sma for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # SMA需要period + history_len - 1个数据点来计算history_len个SMA值
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)

        # 提取有效的历史值
        valid_sma = _extract_valid_values(sma_values, inputs.history_len)

        if not valid_sma:
            return SmaOutput(
                **output_base,
                error="SMA calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated SMA for {inputs.symbol}: {len(valid_sma)} values, latest: {valid_sma[-1]}"
        )
        return SmaOutput(**output_base, sma=valid_sma)

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_sma for {inputs.symbol}: {e}")
        return SmaOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_sma for {inputs.symbol}: {e}", exc_info=True
        )
        return SmaOutput(**output_base, error="An unexpected server error occurred.")


# --- RSI Tool ---
@mcp.tool()
async def calculate_rsi(ctx: Context, inputs: RsiInput) -> RsiOutput:
    """Calculates the Relative Strength Index (RSI) for a trading pair."""
    await ctx.info(
        f"Executing calculate_rsi for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
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
            ctx, inputs.symbol, inputs.timeframe, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)

        # 提取有效的历史值
        valid_rsi = _extract_valid_values(rsi_values, inputs.history_len)

        if not valid_rsi:
            return RsiOutput(
                **output_base,
                error="RSI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated RSI for {inputs.symbol}: {len(valid_rsi)} values, latest: {valid_rsi[-1]}"
        )
        return RsiOutput(**output_base, rsi=valid_rsi)

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_rsi for {inputs.symbol}: {e}")
        return RsiOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_rsi for {inputs.symbol}: {e}", exc_info=True
        )
        return RsiOutput(**output_base, error="An unexpected server error occurred.")


# --- MACD Tool ---
@mcp.tool()
async def calculate_macd(ctx: Context, inputs: MacdInput) -> MacdOutput:
    """Calculates the Moving Average Convergence Divergence (MACD) for a trading pair."""
    await ctx.info(
        f"Executing calculate_macd for {inputs.symbol}, Periods: {inputs.fast_period}/{inputs.slow_period}/{inputs.signal_period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "fast_period": inputs.fast_period,
        "slow_period": inputs.slow_period,
        "signal_period": inputs.signal_period,
    }
    try:
        # MACD需要足够的数据用于慢速EMA和信号线计算，加上历史长度
        required_candles = (
            inputs.slow_period + inputs.signal_period + inputs.history_len + 10
        )  # 增加缓冲

        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return MacdOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for MACD."
            )

        macd, signal, hist = talib.MACD(
            close_prices,
            fastperiod=inputs.fast_period,
            slowperiod=inputs.slow_period,
            signalperiod=inputs.signal_period,
        )

        # 提取有效的历史值
        valid_macd = _extract_valid_values(macd, inputs.history_len)
        valid_signal = _extract_valid_values(signal, inputs.history_len)
        valid_hist = _extract_valid_values(hist, inputs.history_len)

        if not valid_macd or not valid_signal or not valid_hist:
            return MacdOutput(
                **output_base,
                error="MACD calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated MACD for {inputs.symbol}: {len(valid_macd)} values, latest MACD: {valid_macd[-1]}"
        )
        return MacdOutput(
            **output_base, macd=valid_macd, signal=valid_signal, histogram=valid_hist
        )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_macd for {inputs.symbol}: {e}")
        return MacdOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_macd for {inputs.symbol}: {e}",
            exc_info=True,
        )
        return MacdOutput(**output_base, error="An unexpected server error occurred.")


# --- Bollinger Bands (BBANDS) Tool ---
@mcp.tool()
async def calculate_bbands(ctx: Context, inputs: BbandsInput) -> BbandsOutput:
    """Calculates Bollinger Bands (BBANDS) for a trading pair."""
    await ctx.info(
        f"Executing calculate_bbands for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
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
        # BBANDS需要period + history_len - 1个数据点
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return BbandsOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for BBANDS."
            )

        upper, middle, lower = talib.BBANDS(
            close_prices,
            timeperiod=inputs.period,
            nbdevup=inputs.nbdevup,
            nbdevdn=inputs.nbdevdn,
            matype=inputs.matype,
        )

        # 提取有效的历史值
        valid_upper = _extract_valid_values(upper, inputs.history_len)
        valid_middle = _extract_valid_values(middle, inputs.history_len)
        valid_lower = _extract_valid_values(lower, inputs.history_len)

        if not valid_upper or not valid_middle or not valid_lower:
            return BbandsOutput(
                **output_base,
                error="BBANDS calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated BBANDS for {inputs.symbol}: {len(valid_upper)} values, latest upper: {valid_upper[-1]}"
        )
        return BbandsOutput(
            **output_base,
            upper_band=valid_upper,
            middle_band=valid_middle,
            lower_band=valid_lower,
        )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_bbands for {inputs.symbol}: {e}")
        return BbandsOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_bbands for {inputs.symbol}: {e}",
            exc_info=True,
        )
        return BbandsOutput(**output_base, error="An unexpected server error occurred.")


# --- Average True Range (ATR) Tool ---
@mcp.tool()
async def calculate_atr(ctx: Context, inputs: AtrInput) -> AtrOutput:
    """Calculates the Average True Range (ATR) for a trading pair."""
    await ctx.info(
        f"Executing calculate_atr for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # ATR需要period + history_len - 1个数据点
        required_candles = inputs.period + inputs.history_len - 1

        price_data = await _fetch_multi_series_data(
            ctx,
            inputs.symbol,
            inputs.timeframe,
            required_candles,
            ["high", "low", "close"],
        )

        if not price_data:
            return AtrOutput(**output_base, error="Failed to fetch HLC data for ATR.")

        high_prices = price_data["high"]
        low_prices = price_data["low"]
        close_prices = price_data["close"]

        # 确保数据长度足够
        if len(high_prices) < required_candles:
            return AtrOutput(
                **output_base,
                error=f"Insufficient HLC data points for ATR. Need at least {required_candles}.",
            )

        atr_values = talib.ATR(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )

        # 提取有效的历史值
        valid_atr = _extract_valid_values(atr_values, inputs.history_len)

        if not valid_atr:
            return AtrOutput(
                **output_base,
                error="ATR calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated ATR for {inputs.symbol}: {len(valid_atr)} values, latest: {valid_atr[-1]}"
        )
        return AtrOutput(**output_base, atr=valid_atr)

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_atr for {inputs.symbol}: {e}")
        return AtrOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_atr for {inputs.symbol}: {e}", exc_info=True
        )
        return AtrOutput(**output_base, error="An unexpected server error occurred.")


# --- Average Directional Index (ADX) Tool ---
@mcp.tool()
async def calculate_adx(ctx: Context, inputs: AdxInput) -> AdxOutput:
    """Calculates the Average Directional Index (ADX), +DI, and -DI."""
    await ctx.info(
        f"Executing calculate_adx for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # ADX需要更多数据来稳定计算，加上历史长度
        required_candles = inputs.period * 3 + inputs.history_len

        price_data = await _fetch_multi_series_data(
            ctx,
            inputs.symbol,
            inputs.timeframe,
            required_candles,
            ["high", "low", "close"],
        )

        if not price_data:
            return AdxOutput(**output_base, error="Failed to fetch HLC data for ADX.")

        high_prices = price_data["high"]
        low_prices = price_data["low"]
        close_prices = price_data["close"]

        if len(high_prices) < required_candles:
            return AdxOutput(
                **output_base,
                error=f"Insufficient HLC data points for ADX. Need at least {required_candles}.",
            )

        adx_values = talib.ADX(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )
        plus_di_values = talib.PLUS_DI(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )
        minus_di_values = talib.MINUS_DI(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )

        # 提取有效的历史值
        valid_adx = _extract_valid_values(adx_values, inputs.history_len)
        valid_plus_di = _extract_valid_values(plus_di_values, inputs.history_len)
        valid_minus_di = _extract_valid_values(minus_di_values, inputs.history_len)

        if not valid_adx or not valid_plus_di or not valid_minus_di:
            return AdxOutput(
                **output_base,
                error="ADX/DI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated ADX for {inputs.symbol}: {len(valid_adx)} values, latest ADX: {valid_adx[-1]}"
        )
        return AdxOutput(
            **output_base, adx=valid_adx, plus_di=valid_plus_di, minus_di=valid_minus_di
        )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_adx for {inputs.symbol}: {e}")
        return AdxOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_adx for {inputs.symbol}: {e}", exc_info=True
        )
        return AdxOutput(**output_base, error="An unexpected server error occurred.")


# --- On-Balance Volume (OBV) Tool ---
@mcp.tool()
async def calculate_obv(ctx: Context, inputs: ObvInput) -> ObvOutput:
    """Calculates On-Balance Volume (OBV) for a trading pair."""
    await ctx.info(
        f"Executing calculate_obv for {inputs.symbol}, Data points: {inputs.data_points}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "data_points": inputs.data_points,
    }
    try:
        # OBV需要data_points个数据点来计算，确保至少有history_len个有效结果
        required_candles = max(inputs.data_points, inputs.history_len)

        price_data = await _fetch_multi_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, ["close", "volume"]
        )

        if not price_data:
            return ObvOutput(
                **output_base, error="Failed to fetch close and volume data for OBV."
            )

        close_prices = price_data["close"]
        volume_data = price_data["volume"]

        if len(close_prices) < required_candles or len(volume_data) < required_candles:
            return ObvOutput(
                **output_base, error="Insufficient close/volume data points for OBV."
            )

        obv_values = talib.OBV(close_prices, volume_data)

        # 提取有效的历史值
        valid_obv = _extract_valid_values(obv_values, inputs.history_len)

        if not valid_obv:
            return ObvOutput(
                **output_base,
                error="OBV calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated OBV for {inputs.symbol}: {len(valid_obv)} values, latest: {valid_obv[-1]}"
        )
        return ObvOutput(**output_base, obv=valid_obv)

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_obv for {inputs.symbol}: {e}")
        return ObvOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_obv for {inputs.symbol}: {e}", exc_info=True
        )
        return ObvOutput(**output_base, error="An unexpected server error occurred.")


@mcp.tool()
async def get_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    Fetches OHLCV candle data for a given trading pair with specified timeframe and limit.
    """
    await ctx.info(
        f"Fetching {inputs.limit} candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        # Fetch OHLCV data from exchange
        ohlcv_data = await fetch_ohlcv_data(
            ctx=ctx,
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            limit=inputs.limit,
            since=inputs.since,
        )

        if ohlcv_data:
            candles = [OHLCVCandle.from_list(candle) for candle in ohlcv_data]
            await ctx.info(
                f"Successfully fetched {len(candles)} candles for {inputs.symbol}"
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
                error=f"No candle data available for {inputs.symbol}",
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"Exchange/Network Error: {str(e)}",
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching candles for {inputs.symbol}: {e}", exc_info=True
        )
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error="An unexpected server error occurred.",
        )


@mcp.tool()
async def get_current_price(inputs: PriceInput, ctx: Context) -> PriceOutput:
    """
    Fetches the current market price for a given trading pair symbol.
    """
    await ctx.info(f"Fetching current price for {inputs.symbol}")

    try:
        ticker = await fetch_ticker_data(ctx, inputs.symbol)
        if ticker and "last" in ticker and ticker["last"] is not None:
            await ctx.info(f"Current price for {inputs.symbol}: {ticker['last']}")
            return PriceOutput(
                symbol=inputs.symbol,
                price=ticker["last"],
                timestamp=ticker.get("timestamp"),
            )
        else:
            return PriceOutput(symbol=inputs.symbol, error="Price data not available")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching price for {inputs.symbol}: {e}")
        return PriceOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching price for {inputs.symbol}: {e}", exc_info=True
        )
        return PriceOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_ticker(ctx: Context, inputs: TickerInput) -> TickerOutput:
    """
    Fetches comprehensive ticker data for a given trading pair.
    """
    await ctx.info(f"Fetching ticker data for {inputs.symbol}")

    try:
        ticker = await fetch_ticker_data(ctx, inputs.symbol)
        if ticker:
            await ctx.info(f"Successfully fetched ticker for {inputs.symbol}")
            return TickerOutput(
                symbol=inputs.symbol,
                bid=ticker.get("bid"),
                ask=ticker.get("ask"),
                last=ticker.get("last"),
                open=ticker.get("open"),
                high=ticker.get("high"),
                low=ticker.get("low"),
                close=ticker.get("close"),
                volume=ticker.get("baseVolume"),
                percentage=ticker.get("percentage"),
                change=ticker.get("change"),
                timestamp=ticker.get("timestamp"),
            )
        else:
            return TickerOutput(symbol=inputs.symbol, error="Ticker data not available")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching ticker for {inputs.symbol}: {e}")
        return TickerOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching ticker for {inputs.symbol}: {e}", exc_info=True
        )
        return TickerOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_order_book(ctx: Context, inputs: OrderBookInput) -> OrderBookOutput:
    """
    Fetches order book data for a given trading pair.
    """
    await ctx.info(f"Fetching order book for {inputs.symbol} (limit: {inputs.limit})")

    try:
        order_book = await fetch_order_book(ctx, inputs.symbol, inputs.limit)
        if order_book:
            bids = [
                OrderBookLevel(price=bid[0], amount=bid[1])
                for bid in order_book.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=ask[0], amount=ask[1])
                for ask in order_book.get("asks", [])
            ]

            await ctx.info(f"Successfully fetched order book for {inputs.symbol}")
            return OrderBookOutput(
                symbol=inputs.symbol,
                bids=bids,
                asks=asks,
                timestamp=order_book.get("timestamp"),
            )
        else:
            return OrderBookOutput(
                symbol=inputs.symbol, error="Order book data not available"
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching order book for {inputs.symbol}: {e}")
        return OrderBookOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching order book for {inputs.symbol}: {e}",
            exc_info=True,
        )
        return OrderBookOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_recent_trades(ctx: Context, inputs: TradesInput) -> TradesOutput:
    """
    Fetches recent trades data for a given trading pair.
    """
    await ctx.info(f"Fetching {inputs.limit} recent trades for {inputs.symbol}")

    try:
        trades_data = await fetch_trades(ctx, inputs.symbol, inputs.limit)
        if trades_data:
            trades = [
                TradeData(
                    id=trade.get("id"),
                    timestamp=trade["timestamp"],
                    price=trade["price"],
                    amount=trade["amount"],
                    side=trade["side"],
                )
                for trade in trades_data
            ]

            await ctx.info(
                f"Successfully fetched {len(trades)} trades for {inputs.symbol}"
            )
            return TradesOutput(symbol=inputs.symbol, trades=trades, count=len(trades))
        else:
            return TradesOutput(
                symbol=inputs.symbol, error="Recent trades data not available"
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching trades for {inputs.symbol}: {e}")
        return TradesOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching trades for {inputs.symbol}: {e}", exc_info=True
        )
        return TradesOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def generate_comprehensive_market_report(
    ctx: Context, inputs: ComprehensiveAnalysisInput
) -> ComprehensiveAnalysisOutput:
    """
    Generates a comprehensive market analysis report by combining multiple technical indicators.
    Returns historical data for each indicator based on history_len parameter.
    """
    await ctx.info(
        f"Generating comprehensive report for {inputs.symbol} ({inputs.timeframe}) with {inputs.history_len} data points."
    )
    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # Determine which indicators to run
    default_indicators = ["SMA", "RSI", "MACD", "BBANDS", "ATR", "ADX", "OBV"]
    indicators_to_run = (
        inputs.indicators_to_include
        if inputs.indicators_to_include is not None
        else default_indicators
    )

    try:
        # --- SMA ---
        if "SMA" in indicators_to_run:
            sma_period = inputs.sma_period or settings.DEFAULT_SMA_PERIOD
            sma_input = SmaInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=sma_period,
            )
            sma_output = await calculate_sma(ctx, sma_input)
            indicator_results_structured["sma"] = sma_output.model_dump()

            if sma_output.sma is not None and len(sma_output.sma) > 0:
                latest_sma = sma_output.sma[-1]
                report_sections.append(
                    f"- SMA({sma_output.period}): {latest_sma:.4f} (Latest)"
                )
                if len(sma_output.sma) > 1:
                    trend = "↗" if sma_output.sma[-1] > sma_output.sma[-2] else "↘"
                    report_sections.append(
                        f"  - Trend: {trend} ({len(sma_output.sma)} data points)"
                    )
            elif sma_output.error:
                report_sections.append(f"- SMA: Error - {sma_output.error}")

        # --- RSI ---
        if "RSI" in indicators_to_run:
            rsi_period = inputs.rsi_period or settings.DEFAULT_RSI_PERIOD
            rsi_input = RsiInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=rsi_period,
            )
            rsi_output = await calculate_rsi(ctx, rsi_input)
            indicator_results_structured["rsi"] = rsi_output.model_dump()

            if rsi_output.rsi is not None and len(rsi_output.rsi) > 0:
                latest_rsi = rsi_output.rsi[-1]
                report_sections.append(f"- RSI({rsi_output.period}): {latest_rsi:.2f}")
                if latest_rsi > 70:
                    report_sections.append(
                        "  - Note: RSI suggests overbought conditions (>70)."
                    )
                elif latest_rsi < 30:
                    report_sections.append(
                        "  - Note: RSI suggests oversold conditions (<30)."
                    )

                # Add trend analysis if we have multiple data points
                if len(rsi_output.rsi) > 1:
                    trend = "↗" if rsi_output.rsi[-1] > rsi_output.rsi[-2] else "↘"
                    report_sections.append(
                        f"  - Trend: {trend} ({len(rsi_output.rsi)} data points)"
                    )
            elif rsi_output.error:
                report_sections.append(f"- RSI: Error - {rsi_output.error}")

        # --- MACD ---
        if "MACD" in indicators_to_run:
            macd_input = MacdInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                fast_period=inputs.macd_fast_period or settings.DEFAULT_MACD_FAST,
                slow_period=inputs.macd_slow_period or settings.DEFAULT_MACD_SLOW,
                signal_period=inputs.macd_signal_period or settings.DEFAULT_MACD_SIGNAL,
            )
            macd_output = await calculate_macd(ctx, macd_input)
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
                    f"- MACD({macd_output.fast_period},{macd_output.slow_period},{macd_output.signal_period}): "
                    f"MACD: {latest_macd:.4f}, Signal: {latest_signal:.4f}, Hist: {latest_hist:.4f}"
                )

                if latest_hist > 0 and latest_macd > latest_signal:
                    report_sections.append(
                        "  - Note: MACD histogram positive, potentially bullish momentum."
                    )
                elif latest_hist < 0 and latest_macd < latest_signal:
                    report_sections.append(
                        "  - Note: MACD histogram negative, potentially bearish momentum."
                    )

                # Add trend analysis
                if len(macd_output.histogram) > 1:
                    hist_trend = (
                        "↗"
                        if macd_output.histogram[-1] > macd_output.histogram[-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Histogram Trend: {hist_trend} ({len(macd_output.histogram)} data points)"
                    )

            elif macd_output.error:
                report_sections.append(f"- MACD: Error - {macd_output.error}")

        # --- Bollinger Bands (BBANDS) ---
        if "BBANDS" in indicators_to_run:
            bbands_input = BbandsInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.bbands_period or settings.DEFAULT_BBANDS_PERIOD,
            )
            bbands_output = await calculate_bbands(ctx, bbands_input)
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
                    f"- Bollinger Bands({bbands_output.period}, {bbands_output.nbdevup}dev): "
                    f"Upper: {latest_upper:.4f}, Middle: {latest_middle:.4f}, Lower: {latest_lower:.4f}"
                )
                report_sections.append(
                    f"  - Band Width: {latest_upper - latest_lower:.4f}"
                )

                # Add trend analysis
                if len(bbands_output.middle_band) > 1:
                    trend = (
                        "↗"
                        if bbands_output.middle_band[-1] > bbands_output.middle_band[-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Middle Band Trend: {trend} ({len(bbands_output.middle_band)} data points)"
                    )

            elif bbands_output.error:
                report_sections.append(f"- BBANDS: Error - {bbands_output.error}")

        # --- Average True Range (ATR) ---
        if "ATR" in indicators_to_run:
            atr_input = AtrInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.atr_period or settings.DEFAULT_ATR_PERIOD,
            )
            atr_output = await calculate_atr(ctx, atr_input)
            indicator_results_structured["atr"] = atr_output.model_dump()

            if atr_output.atr is not None and len(atr_output.atr) > 0:
                latest_atr = atr_output.atr[-1]
                report_sections.append(
                    f"- ATR({atr_output.period}): {latest_atr:.4f} (Volatility Measure)"
                )

                # Add trend analysis for volatility
                if len(atr_output.atr) > 1:
                    vol_trend = "↗" if atr_output.atr[-1] > atr_output.atr[-2] else "↘"
                    report_sections.append(
                        f"  - Volatility Trend: {vol_trend} ({len(atr_output.atr)} data points)"
                    )

            elif atr_output.error:
                report_sections.append(f"- ATR: Error - {atr_output.error}")

        # --- Average Directional Index (ADX) ---
        if "ADX" in indicators_to_run:
            adx_input = AdxInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.adx_period or settings.DEFAULT_ADX_PERIOD,
            )
            adx_output = await calculate_adx(ctx, adx_input)
            indicator_results_structured["adx"] = adx_output.model_dump()

            if (
                adx_output.adx is not None
                and len(adx_output.adx) > 0
                and adx_output.plus_di is not None
                and len(adx_output.plus_di) > 0
                and adx_output.minus_di is not None
                and len(adx_output.minus_di) > 0
            ):
                latest_adx = adx_output.adx[-1]
                latest_plus_di = adx_output.plus_di[-1]
                latest_minus_di = adx_output.minus_di[-1]

                report_sections.append(
                    f"- ADX({adx_output.period}): {latest_adx:.2f}, +DI: {latest_plus_di:.2f}, -DI: {latest_minus_di:.2f}"
                )

                if latest_adx > 25:
                    report_sections.append(
                        f"  - Note: ADX ({latest_adx:.2f}) suggests a trending market."
                    )
                else:
                    report_sections.append(
                        f"  - Note: ADX ({latest_adx:.2f}) suggests a weak or non-trending market."
                    )

                # Directional analysis
                if latest_plus_di > latest_minus_di:
                    report_sections.append("  - Direction: Bullish (+DI > -DI)")
                else:
                    report_sections.append("  - Direction: Bearish (-DI > +DI)")

                # Add trend analysis
                if len(adx_output.adx) > 1:
                    trend_strength = (
                        "↗" if adx_output.adx[-1] > adx_output.adx[-2] else "↘"
                    )
                    report_sections.append(
                        f"  - Trend Strength: {trend_strength} ({len(adx_output.adx)} data points)"
                    )

            elif adx_output.error:
                report_sections.append(f"- ADX: Error - {adx_output.error}")

        # --- On-Balance Volume (OBV) ---
        if "OBV" in indicators_to_run:
            obv_input = ObvInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                data_points=inputs.obv_data_points or settings.DEFAULT_OBV_DATA_POINTS,
            )
            obv_output = await calculate_obv(ctx, obv_input)
            indicator_results_structured["obv"] = obv_output.model_dump()

            if obv_output.obv is not None and len(obv_output.obv) > 0:
                latest_obv = obv_output.obv[-1]
                report_sections.append(
                    f"- OBV (using {obv_output.data_points} points): {latest_obv:.2f}"
                )

                # Add trend analysis for volume flow
                if len(obv_output.obv) > 1:
                    volume_trend = (
                        "↗" if obv_output.obv[-1] > obv_output.obv[-2] else "↘"
                    )
                    report_sections.append(
                        f"  - Volume Flow: {volume_trend} ({len(obv_output.obv)} data points)"
                    )

            elif obv_output.error:
                report_sections.append(f"- OBV: Error - {obv_output.error}")

        # --- Synthesize the report ---
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base,
                error="No indicator data could be calculated or selected.",
            )

        report_title = f"Comprehensive Technical Analysis for {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} Data Points:\n"
        report_text = report_title + "\n".join(report_sections)

        # Add summary section with overall trend analysis
        summary_sections = []
        trend_indicators = []

        # Collect trend information from indicators that have it
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
            f"Successfully generated comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
        )
        return ComprehensiveAnalysisOutput(
            **output_base,
            report_text=report_text,
            structured_data=indicator_results_structured,
        )

    except Exception as e:
        await ctx.error(
            f"Unexpected error in generate_comprehensive_market_report for {inputs.symbol}: {e}",
            exc_info=True,
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"An unexpected server error occurred: {str(e)}"
        )
