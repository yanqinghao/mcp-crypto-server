# tools/market_analysis.py
import numpy as np
import talib
from fastmcp import FastMCP, Context

# Uses corrected Pydantic models
from models.analysis import (
    SmaInput,
    SmaOutput,
    RsiInput,
    RsiOutput,
    MacdInput,
    MacdOutput,
)
from services.exchange_service import exchange_service
import ccxt
from typing import Optional

mcp = FastMCP()


# --- Helper Function ---
async def _fetch_and_prepare_ohlcv(
    ctx: Context, symbol: str, timeframe: str, required_candles: int
) -> Optional[np.ndarray]:
    """Fetches OHLCV data and returns closing prices as a NumPy array."""
    await ctx.info(f"Fetching ~{required_candles} candles for {symbol} ({timeframe})")
    try:
        # Fetch slightly more candles for indicator calculation stability
        limit = required_candles + 30  # Increased buffer for TA-Lib needs
        ohlcv = await exchange_service.get_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv:
            await ctx.warning(f"No OHLCV data returned for {symbol} ({timeframe}).")
            return None

        # Ensure we have enough data *after* potential initial NaNs from TA-Lib
        if len(ohlcv) < required_candles:
            await ctx.warning(
                f"Insufficient OHLCV data for {symbol} ({timeframe}). Required: {required_candles}, Got: {len(ohlcv)}"
            )
            return None

        # Extract closing prices (index 4)
        close_prices = np.array([candle[1] for candle in ohlcv], dtype=float)

        # Check for NaN/None in closing prices which can break TA-Lib
        if np.isnan(close_prices).any():
            await ctx.warning(
                f"NaN values found in closing prices for {symbol} ({timeframe}). Attempting to use valid subset."
            )
            # Filter out NaNs - this might reduce the effective number of candles
            close_prices = close_prices[~np.isnan(close_prices)]
            if len(close_prices) < required_candles:
                await ctx.warning(
                    f"Insufficient valid (non-NaN) closing prices for {symbol} ({timeframe}) after filtering. Required: {required_candles}, Got: {len(close_prices)}"
                )
                return None

        # Return only the required number of *most recent* valid candles
        return close_prices[-required_candles:]

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching OHLCV for {symbol} ({timeframe}): {e}")
        raise
    except Exception as e:
        await ctx.error(
            f"Unexpected error preparing OHLCV for {symbol} ({timeframe}): {e}",
            exc_info=True,
        )
        raise


# --- SMA Tool ---
@mcp.tool()
async def calculate_sma(
    ctx: Context, inputs: SmaInput
) -> SmaOutput:  # Correctly uses SmaInput
    """Calculates the Simple Moving Average (SMA) for a trading pair."""
    await ctx.info(
        f"Executing calculate_sma for {inputs.symbol}, Period: {inputs.period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }

    try:
        # SMA needs 'period' candles
        close_prices = await _fetch_and_prepare_ohlcv(
            ctx, inputs.symbol, inputs.timeframe, inputs.period
        )
        if close_prices is None or len(close_prices) < inputs.period:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)
        last_sma = sma_values[-1]

        if np.isnan(last_sma):
            await ctx.warning(
                f"SMA calculation resulted in NaN for {inputs.symbol} (Period: {inputs.period})."
            )
            return SmaOutput(
                **output_base,
                error="SMA calculation resulted in NaN (insufficient data for period).",
            )

        await ctx.info(f"Calculated SMA for {inputs.symbol}: {last_sma}")
        return SmaOutput(**output_base, sma=last_sma)

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
async def calculate_rsi(
    ctx: Context, inputs: RsiInput
) -> RsiOutput:  # Correctly uses RsiInput
    """Calculates the Relative Strength Index (RSI) for a trading pair."""
    await ctx.info(
        f"Executing calculate_rsi for {inputs.symbol}, Period: {inputs.period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }

    try:
        # RSI needs 'period' candles for calculation, plus one for the initial difference
        required_candles = inputs.period + 1
        close_prices = await _fetch_and_prepare_ohlcv(
            ctx, inputs.symbol, inputs.timeframe, required_candles
        )
        if close_prices is None or len(close_prices) < required_candles:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)
        last_rsi = rsi_values[-1]

        if np.isnan(last_rsi):
            await ctx.warning(
                f"RSI calculation resulted in NaN for {inputs.symbol} (Period: {inputs.period})."
            )
            return RsiOutput(
                **output_base,
                error="RSI calculation resulted in NaN (insufficient data for period).",
            )

        await ctx.info(f"Calculated RSI for {inputs.symbol}: {last_rsi}")
        return RsiOutput(**output_base, rsi=last_rsi)

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
async def calculate_macd(
    ctx: Context, inputs: MacdInput
) -> MacdOutput:  # Correctly uses MacdInput
    """Calculates the Moving Average Convergence Divergence (MACD) for a trading pair."""
    await ctx.info(
        f"Executing calculate_macd for {inputs.symbol}, Periods: {inputs.fast_period}/{inputs.slow_period}/{inputs.signal_period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "fast_period": inputs.fast_period,
        "slow_period": inputs.slow_period,
        "signal_period": inputs.signal_period,
    }

    try:
        # MACD needs enough data for the slow EMA plus the signal EMA lookback period
        required_candles = (
            inputs.slow_period + inputs.signal_period - 1
        )  # Minimum needed by TA-Lib
        close_prices = await _fetch_and_prepare_ohlcv(
            ctx, inputs.symbol, inputs.timeframe, required_candles + 50
        )  # Fetch ample buffer
        if (
            close_prices is None or len(close_prices) < required_candles
        ):  # Check against minimum
            return MacdOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for MACD."
            )

        macd, signal, hist = talib.MACD(
            close_prices,
            fastperiod=inputs.fast_period,
            slowperiod=inputs.slow_period,
            signalperiod=inputs.signal_period,
        )

        # Get the most recent non-NaN values
        last_macd = macd[~np.isnan(macd)][-1] if not np.all(np.isnan(macd)) else None
        last_signal = (
            signal[~np.isnan(signal)][-1] if not np.all(np.isnan(signal)) else None
        )
        last_hist = hist[~np.isnan(hist)][-1] if not np.all(np.isnan(hist)) else None

        if last_macd is None or last_signal is None or last_hist is None:
            await ctx.warning(f"MACD calculation resulted in NaN for {inputs.symbol}.")
            return MacdOutput(
                **output_base,
                error="MACD calculation resulted in NaN (insufficient data for periods).",
            )

        await ctx.info(
            f"Calculated MACD for {inputs.symbol}: MACD={last_macd}, Signal={last_signal}, Hist={last_hist}"
        )
        return MacdOutput(
            **output_base, macd=last_macd, signal=last_signal, histogram=last_hist
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
