from pydantic import BaseModel, Field
from typing import Optional, List
from config import settings  # Import defaults from your config


class IndicatorInput(BaseModel):
    """Base input model for technical indicators."""

    symbol: str = Field(..., description="Trading pair symbol, e.g., 'BTC/USDT'")
    timeframe: str = Field(
        default="1h",
        description="Candlestick timeframe, e.g., '1m', '5m', '1h', '1d', default is '1h'",
    )
    history_len: int = Field(
        default=5, description="Candlestick data length, default is 5 candles"
    )


class IndicatorOutputBase(BaseModel):
    """Base output model for indicators, including common fields."""

    symbol: str
    timeframe: str
    error: Optional[str] = None


class SmaInput(IndicatorInput):
    """Input model for calculating Simple Moving Average (SMA)."""

    period: int = Field(
        default=settings.DEFAULT_SMA_PERIOD,
        gt=0,
        description="SMA calculation period (number of candles)",
    )


class SmaOutput(IndicatorOutputBase):
    """Output model for the SMA calculation tool."""

    period: int
    sma: Optional[List[float]] = None


class RsiInput(IndicatorInput):
    """Input model for calculating Relative Strength Index (RSI)."""

    period: int = Field(
        default=settings.DEFAULT_RSI_PERIOD,
        gt=1,  # RSI typically needs period > 1
        description="RSI calculation period (number of candles)",
    )


class RsiOutput(IndicatorOutputBase):
    """Output model for the RSI calculation tool."""

    period: int
    rsi: Optional[List[float]] = None


class MacdInput(IndicatorInput):
    """Input model for calculating Moving Average Convergence Divergence (MACD)."""

    fast_period: int = Field(
        default=settings.DEFAULT_MACD_FAST, gt=0, description="MACD fast EMA period"
    )
    slow_period: int = Field(
        default=settings.DEFAULT_MACD_SLOW, gt=0, description="MACD slow EMA period"
    )
    signal_period: int = Field(
        default=settings.DEFAULT_MACD_SIGNAL,
        gt=0,
        description="MACD signal line EMA period",
    )


class MacdOutput(IndicatorOutputBase):
    """Output model for the MACD calculation tool."""

    fast_period: int
    slow_period: int
    signal_period: int
    macd: Optional[List[float]] = None
    signal: Optional[List[float]] = None
    histogram: Optional[List[float]] = None


class BbandsInput(IndicatorInput):
    """Input model for calculating Bollinger Bands (BBANDS)."""

    period: int = Field(
        default=settings.DEFAULT_BBANDS_PERIOD,
        gt=1,
        description="BBANDS calculation period",
    )
    nbdevup: float = Field(
        default=settings.DEFAULT_BBANDS_NBDEVUP,
        gt=0,
        description="Number of standard deviations for upper band",
    )
    nbdevdn: float = Field(
        default=settings.DEFAULT_BBANDS_NBDEVDN,
        gt=0,
        description="Number of standard deviations for lower band",
    )
    matype: int = Field(
        default=settings.DEFAULT_BBANDS_MATYPE,
        description="Moving average type (e.g., 0 for SMA)",
    )


class BbandsOutput(IndicatorOutputBase):
    """Output model for the BBANDS calculation tool."""

    period: int
    nbdevup: float
    nbdevdn: float
    matype: int
    upper_band: Optional[List[float]] = None
    middle_band: Optional[List[float]] = None
    lower_band: Optional[List[float]] = None


class AtrInput(IndicatorInput):
    """Input model for calculating Average True Range (ATR)."""

    period: int = Field(
        default=settings.DEFAULT_ATR_PERIOD, gt=1, description="ATR calculation period"
    )


class AtrOutput(IndicatorOutputBase):
    """Output model for the ATR calculation tool."""

    period: int
    atr: Optional[List[float]] = None


class AdxInput(IndicatorInput):
    """Input model for calculating Average Directional Index (ADX)."""

    period: int = Field(
        default=settings.DEFAULT_ADX_PERIOD, gt=1, description="ADX calculation period"
    )


class AdxOutput(IndicatorOutputBase):
    """Output model for the ADX calculation tool."""

    period: int
    adx: Optional[List[float]] = None
    plus_di: Optional[List[float]] = None
    minus_di: Optional[List[float]] = None


class ObvInput(IndicatorInput):
    """Input model for calculating On-Balance Volume (OBV)."""

    # OBV is cumulative, so period is less about lookback for the formula
    # and more about how much data to fetch for a meaningful starting point.
    data_points: int = Field(
        default=settings.DEFAULT_OBV_DATA_POINTS,
        gt=1,
        description="Number of data points to fetch for OBV calculation",
    )


class ObvOutput(IndicatorOutputBase):
    """Output model for the OBV calculation tool."""

    data_points: int  # Reflects the amount of data used
    obv: Optional[List[float]] = None


# --- Models for Comprehensive Report Tool ---
class ComprehensiveAnalysisInput(IndicatorInput):
    """Input model for the comprehensive market report."""

    # Allow specifying which indicators to include, or use defaults
    indicators_to_include: Optional[List[str]] = Field(
        default=None,  # None means include all configured defaults
        description="List of indicators to include, e.g. ['SMA', 'RSI', 'MACD', 'BBANDS', 'ATR', 'ADX', 'OBV']",
    )
    # Optionally allow overriding default periods for this specific report
    sma_period: Optional[int] = None
    rsi_period: Optional[int] = None
    macd_fast_period: Optional[int] = None
    macd_slow_period: Optional[int] = None
    macd_signal_period: Optional[int] = None
    bbands_period: Optional[int] = None
    atr_period: Optional[int] = None
    adx_period: Optional[int] = None
    obv_data_points: Optional[int] = None
    # Add other indicator-specific parameters if needed for overrides


class ComprehensiveAnalysisOutput(IndicatorOutputBase):
    """Output model for the comprehensive market report."""

    report_text: Optional[str] = Field(
        default=None, description="Narrative market analysis report"
    )
    structured_data: Optional[dict] = Field(
        default=None, description="Dictionary of all calculated indicator values"
    )
