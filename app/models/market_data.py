from pydantic import BaseModel, Field  # Import from Pydantic
from typing import Optional


class PriceInput(BaseModel):  # Inherit from BaseModel
    """Input model for requesting the current price of a trading pair."""

    symbol: str = Field(..., description="Trading pair symbol, e.g., 'BTC/USDT'")


class PriceOutput(BaseModel):  # Inherit from BaseModel
    """Output model for the current price tool."""

    symbol: str
    price: Optional[float] = None
    error: Optional[str] = None
