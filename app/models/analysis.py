from pydantic import BaseModel, Field  # Import from Pydantic
from typing import Optional
from config import settings  # Import defaults


class IndicatorInput(BaseModel):  # Inherit from BaseModel
    """Base input model for technical indicators."""

    symbol: str = Field(..., description="Trading pair symbol, e.g., 'BTC/USDT'")
    timeframe: str = Field(
        default="1h", description="Candlestick timeframe, e.g., '1m', '5m', '1h', '1d'"
    )


class SmaInput(IndicatorInput):
    """Input model for calculating Simple Moving Average (SMA)."""

    period: int = Field(
        default=settings.DEFAULT_SMA_PERIOD,
        gt=0,
        description="SMA calculation period (number of candles)",
    )


class SmaOutput(BaseModel):  # Inherit from BaseModel
    """Output model for the SMA calculation tool."""

    symbol: str
    timeframe: str
    period: int
    sma: Optional[float] = None
    error: Optional[str] = None


class RsiInput(IndicatorInput):
    """Input model for calculating Relative Strength Index (RSI)."""

    period: int = Field(
        default=settings.DEFAULT_RSI_PERIOD,
        gt=1,
        description="RSI calculation period (number of candles)",
    )


class RsiOutput(BaseModel):  # Inherit from BaseModel
    """Output model for the RSI calculation tool."""

    symbol: str
    timeframe: str
    period: int
    rsi: Optional[float] = None
    error: Optional[str] = None


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


class MacdOutput(BaseModel):  # Inherit from BaseModel
    """Output model for the MACD calculation tool."""

    symbol: str
    timeframe: str
    fast_period: int
    slow_period: int
    signal_period: int
    macd: Optional[float] = None
    signal: Optional[float] = None
    histogram: Optional[float] = None
    error: Optional[str] = None
