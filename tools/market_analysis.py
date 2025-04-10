import json
from datetime import datetime, timedelta

import pandas as pd

from config.settings import settings

# Get the mcp server instance
from main import mcp
from services.database_service import DatabaseService
from services.exchange_service import ExchangeService

# Initialize services
exchange_service = ExchangeService(settings.exchanges)
db_service = DatabaseService(settings.database_url)


@mcp.tool()
def analyze_price_trend(exchange: str, symbol: str, days: int = 30) -> str:
    """
    Analyze price trend for a cryptocurrency

    Parameters:
    - exchange: Name of the exchange (e.g., 'binance', 'coinbase')
    - symbol: Trading pair symbol (e.g., 'BTC/USDT')
    - days: Number of days to analyze (default: 30)

    Returns JSON with trend analysis results
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    # Get OHLCV data
    ohlcv_data = db_service.get_ohlcv_data(
        exchange=exchange, symbol=symbol, timeframe="1d", start_time=start_time, end_time=end_time
    )

    # Convert to pandas DataFrame
    df = pd.DataFrame(ohlcv_data)

    # Calculate moving averages
    df["ma7"] = df["close"].rolling(window=7).mean()
    df["ma25"] = df["close"].rolling(window=25).mean()

    # Calculate RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Determine trend
    last_price = df["close"].iloc[-1]
    ma7_last = df["ma7"].iloc[-1]
    ma25_last = df["ma25"].iloc[-1]

    if ma7_last > ma25_last:
        trend = "bullish"
    elif ma7_last < ma25_last:
        trend = "bearish"
    else:
        trend = "neutral"

    # Prepare result
    result = {
        "exchange": exchange,
        "symbol": symbol,
        "period_days": days,
        "last_price": float(last_price),
        "trend": trend,
        "rsi": float(df["rsi"].iloc[-1]),
        "ma7": float(ma7_last),
        "ma25": float(ma25_last),
        "price_change_percent": float((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100),
    }

    return json.dumps(result)


@mcp.tool()
def get_market_depth(exchange: str, symbol: str, depth: int = 10) -> str:
    """
    Get market depth (order book) for a cryptocurrency

    Parameters:
    - exchange: Name of the exchange (e.g., 'binance', 'coinbase')
    - symbol: Trading pair symbol (e.g., 'BTC/USDT')
    - depth: Depth of the order book to retrieve (default: 10)

    Returns JSON with order book data
    """
    orderbook = exchange_service.get_orderbook(exchange, symbol, depth)
    return json.dumps(orderbook)
