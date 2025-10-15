from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
from enum import Enum


class TimeFrame(str, Enum):
    """Available timeframes for OHLCV data"""

    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class OHLCVCandle(BaseModel):
    """Represents a single OHLCV candle.
    Standard format: [timestamp, open, high, low, close, volume]
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_list(cls, data: List[Union[int, float]]):
        if len(data) >= 6:
            return cls(
                timestamp=int(data[0]),
                open=float(data[1]),
                high=float(data[2]),
                low=float(data[3]),
                close=float(data[4]),
                volume=float(data[5]),
            )
        raise ValueError("OHLCV list must have at least 6 elements.")


# Input Models
class CandlesInput(BaseModel):
    """Input model for requesting OHLCV candle data."""

    symbol: str = Field(..., description="Trading pair symbol, e.g., 'BTC/USDT'")
    timeframe: TimeFrame = Field(
        default=TimeFrame.ONE_HOUR, description="Timeframe for candles"
    )
    limit: int = Field(
        default=24, ge=1, le=1000, description="Number of candles to fetch (1-1000)"
    )
    since: Optional[int] = Field(
        default=None, description="Timestamp to fetch data from (optional)"
    )


class PriceInput(BaseModel):
    """Input model for requesting the current price of a trading pair."""

    symbol: str = Field(..., description="Trading pair symbol, e.g., 'BTC/USDT'")


class TickerInput(BaseModel):
    """Input model for requesting ticker data."""

    symbol: str = Field(..., description="Trading pair symbol, e.g., 'BTC/USDT'")


class OrderBookInput(BaseModel):
    """Input model for requesting order book data."""

    symbol: str = Field(..., description="Trading pair symbol, e.g., 'BTC/USDT'")
    limit: int = Field(
        default=20, ge=1, le=100, description="Number of bid/ask levels to fetch"
    )


class TradesInput(BaseModel):
    """Input model for requesting recent trades data."""

    symbol: str = Field(..., description="Trading pair symbol, e.g., 'BTC/USDT'")
    limit: int = Field(
        default=50, ge=1, le=500, description="Number of recent trades to fetch"
    )


# Output Models
class CandlesOutput(BaseModel):
    """Output model for OHLCV candle data."""

    symbol: str
    timeframe: str
    candles: List[OHLCVCandle] = []
    count: int = 0
    error: Optional[str] = None


class PriceOutput(BaseModel):
    """Output model for the current price tool."""

    symbol: str
    price: Optional[float] = None
    timestamp: Optional[int] = None
    error: Optional[str] = None


class TickerOutput(BaseModel):
    """Output model for ticker data."""

    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    percentage: Optional[float] = None
    change: Optional[float] = None
    timestamp: Optional[int] = None
    error: Optional[str] = None


class OrderBookLevel(BaseModel):
    """Represents a single order book level (price and amount)."""

    price: float
    amount: float


class OrderBookOutput(BaseModel):
    """Output model for order book data."""

    symbol: str
    bids: List[OrderBookLevel] = []
    asks: List[OrderBookLevel] = []
    timestamp: Optional[int] = None
    error: Optional[str] = None


class TradeData(BaseModel):
    """Represents a single trade."""

    id: Optional[str] = None
    timestamp: int
    price: float
    amount: float
    side: str  # 'buy' or 'sell'


class TradesOutput(BaseModel):
    """Output model for recent trades data."""

    symbol: str
    trades: List[TradeData] = []
    count: int = 0
    error: Optional[str] = None


class FundingRateInput(BaseModel):
    symbol: str
    include_history: bool = True
    limit: int = 50
    since: Optional[int] = None  # ms timestamp


class FundingRatePoint(BaseModel):
    timestamp: int
    rate: float
    info: Optional[Dict[str, Any]] = None


class FundingRateOutput(BaseModel):
    symbol: str
    current_rate: Optional[float] = None  # e.g., 0.0001 == 0.01%
    next_funding_time: Optional[int] = None  # ms timestamp
    funding_interval: Optional[str] = None  # e.g., '8h'
    history: Optional[List[FundingRatePoint]] = None
    error: Optional[str] = None


class OpenInterestInput(BaseModel):
    symbol: str
    timeframe: str = (
        "1h"  # depends on exchange support, e.g. BinFutures: 5m/15m/1h/4h/1d
    )
    limit: int = 100
    since: Optional[int] = None  # ms timestamp


class OpenInterestPoint(BaseModel):
    timestamp: int
    open_interest: float
    currency: Optional[str] = None
    info: Optional[Dict[str, Any]] = None


class OpenInterestOutput(BaseModel):
    symbol: str
    latest: Optional[OpenInterestPoint] = None
    series: Optional[List[OpenInterestPoint]] = None
    error: Optional[str] = None
