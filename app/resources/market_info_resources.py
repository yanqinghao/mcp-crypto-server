from typing import Dict, List, Optional
from config import settings
from fastmcp import FastMCP, Context

mcp = FastMCP()


@mcp.resource("docs://{indicator_name}/explanation")  # URI for LLM to request
async def get_indicator_explanation(ctx: Context, indicator_name: str) -> str:
    """Provides a brief explanation of a specified technical indicator."""
    explanations = {
        "SMA": "Simple Moving Average (SMA) calculates the average of a selected range of prices, usually closing prices, by the number of periods in that range. It's a lagging indicator used to identify trend direction.",
        "RSI": "Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. RSI oscillates between 0 and 100. Traditionally, RSI is considered overbought when above 70 and oversold when below 30.",
        "MACD": "Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. A nine-day EMA of the MACD, called the 'signal line', is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals.",
        "BBANDS": "Bollinger Bands consist of a middle band (typically an SMA) and upper and lower bands set at a number of standard deviations (usually two) above and below the middle band. They measure price volatility and can help identify overbought or oversold conditions.",
        "ATR": "Average True Range (ATR) measures market volatility. It does not indicate price direction but rather the degree of price movement. High ATR values indicate high volatility, and low ATR values indicate low volatility.",
        "ADX": "Average Directional Index (ADX) measures trend strength, not direction. Values above 25 often indicate a trending market, while values below 20-25 suggest a weak trend or ranging market. It's used with +DI (Plus Directional Indicator) and -DI (Minus Directional Indicator) to gauge trend direction.",
        "OBV": "On-Balance Volume (OBV) is a cumulative momentum indicator that relates volume to price change. It's used to confirm price trends and spot potential reversals through divergences.",
    }
    return explanations.get(
        indicator_name.upper(), f"No explanation available for {indicator_name}."
    )


@mcp.resource("config://indicators/default_periods")
async def get_default_indicator_periods(ctx: Context) -> Dict[str, int | float]:
    """Returns default period settings for common indicators."""
    return {
        "SMA_period": settings.DEFAULT_SMA_PERIOD,
        "RSI_period": settings.DEFAULT_RSI_PERIOD,
        "MACD_fast_period": settings.DEFAULT_MACD_FAST,
        "MACD_slow_period": settings.DEFAULT_MACD_SLOW,
        "MACD_signal_period": settings.DEFAULT_MACD_SIGNAL,
        "BBANDS_period": settings.DEFAULT_BBANDS_PERIOD,
        "BBANDS_nbdevup": settings.DEFAULT_BBANDS_NBDEVUP,
        "BBANDS_nbdevdn": settings.DEFAULT_BBANDS_NBDEVDN,
        "ATR_period": settings.DEFAULT_ATR_PERIOD,
        "ADX_period": settings.DEFAULT_ADX_PERIOD,
        "OBV_data_points": settings.DEFAULT_OBV_DATA_POINTS,
    }


@mcp.resource("exchange://supported_timeframes")
async def get_supported_timeframes(ctx: Context) -> List[str]:
    """Returns a list of commonly supported timeframes."""
    # This could eventually query the exchange_service for actual supported timeframes
    return [
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    ]


@mcp.resource("config://analysis/time_horizon_map")
async def get_time_horizon_to_timeframe_map(ctx: Context) -> Dict[str, List[str]]:
    """
    Provides a mapping from general time horizons (e.g., ultra-short-term, short-term,
    medium-term, long-term) to suggested timeframes.
    """
    return {
        "ultra_short_term": ["5m"],  # e.g., for scalping
        "short_term": ["15m", "30m", "1h", "2h"],  # e.g., for day trading
        "medium_term": ["4h", "6h", "8h", "12h", "1d"],  # e.g., for swing trading
        "long_term": ["3d", "1w", "1M"],  # e.g., for position trading or investment
    }


@mcp.resource("docs://{indicator_name}/data_length_guidelines")
async def get_indicator_data_length_guidelines(
    ctx: Context, indicator_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Provides suggested minimum historical data lengths or number of periods
    for various technical indicators or analysis types.
    For example, an SMA200 needs at least 200 data points for the given timeframe.
    A general guideline is to have 2-3 times the longest period length.
    """
    guidelines = {
        "general_rule": "Aim for at least 2-3 times the number of periods of the longest indicator being used. For example, for a 200-period SMA, aim for 400-600 data points for that timeframe.",
        "SMA_short": "For short SMAs (e.g., SMA10, SMA20), ensure at least 30-50 periods of data.",
        "SMA_long": "For long SMAs (e.g., SMA100, SMA200), ensure at least 2-3 times the SMA period (e.g., 200-300 periods for SMA100, 400-600 for SMA200).",
        "RSI": "Typically needs at least 3-4 times its period. For RSI(14), aim for 40-60 data points.",
        "MACD": "As it uses EMAs (e.g., 12, 26), ensure enough data for these EMAs to stabilize, typically 100+ data points.",
        "BBANDS": "Similar to SMAs, ensure 2-3 times the period used for the middle band.",
        "PATTERN_RECOGNITION": "Chart patterns (e.g., head and shoulders, triangles) may require viewing a significant amount of historical data, often 100-500+ candles depending on the pattern's typical formation time relative to the timeframe.",
        "TREND_ANALYSIS_SHORT_TERM": "For short-term trends (days to weeks), viewing several weeks to a few months of data on hourly/daily charts is common (e.g., 60-100 candles).",
        "TREND_ANALYSIS_LONG_TERM": "For long-term trends (months to years), viewing several months to years of data on daily/weekly charts is common (e.g., 100-300+ candles).",
    }
    if indicator_name:
        return {
            indicator_name: guidelines.get(
                indicator_name.upper(), guidelines.get("general_rule", "")
            )
        }
    return guidelines


@mcp.resource("config://analysis/timeframe_lookback_map")
async def get_timeframe_lookback_recommendations(ctx: Context) -> Dict[str, str]:
    """
    Provides **ultra-concise** recommended historical data lookback periods
    (as a string descriptor, e.g., '6 hours', '5 days') for different analysis timeframes.
    The primary goal is to **minimize data packet size** for a general, recent overview.
    The actual number of candles would depend on the timeframe.
    For example, '6-12 hours' of '5m' data is 72-144 candles.
    '5-7 days' of '1h' data is 120-168 candles.
    These are very conservative guidelines. Specific analyses or indicators requiring
    longer historical data (e.g., long-period Moving Averages) should prompt the LLM
    to refer to 'docs://indicators/data_length_guidelines' for more tailored data length needs,
    potentially requesting more data than suggested here for those specific calculations.
    """
    return {
        # Ultra-Short-Term / Scalping Context (Minimal View)
        "5m": "Last 6-12 hours",  # Approx. 72-144 candles
        # Day Trading / Very Short-Term Swings (Minimal View)
        "15m": "Last 1-2 days",  # Approx. 96-192 candles
        "30m": "Last 2-3 days",  # Approx. 96-144 candles
        # Short-Term Analysis / Swing Trading (Minimal View)
        "1h": "Last 5-7 days",  # Approx. 120-168 candles
        "2h": "Last 7-10 days",  # Approx. 84-120 candles
        # Medium-Term Analysis / Swing Trading (Minimal View)
        "4h": "Last 15-20 days",  # Approx. 90-120 candles
        "6h": "Last 20-30 days",  # Approx. 80-120 candles
        "8h": "Last 30-40 days",  # Approx. 90-120 candles
        "12h": "Last 1.5-2 months",  # Approx. 90-120 candles (e.g., ~45-60 trading days * 2 candles/day)
        # Long-Term Analysis / Position Trading (Minimal View)
        "1d": "Last 4-6 months",  # Approx. 84-126 trading days/candles
        "3d": "Last 9-12 months",  # Approx. 63-84 candles (based on ~7 of 3D-candles/month)
        "1w": "Last 1.5-2 years",  # Approx. 78-104 candles
        # Very Long-Term / Macro Trend Analysis (Minimal View)
        "1M": "Last 3-5 years",  # Approx. 36-60 candles
    }


# Add more resources as needed
