import numpy as np
import talib
from typing import Dict, Optional, List

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
)

# Import the refined data fetching helpers
from services.exchange_service import fetch_ohlcv_data
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
