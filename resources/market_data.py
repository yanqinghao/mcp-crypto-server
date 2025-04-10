import json
from datetime import datetime, timedelta

from config.settings import settings

# Get the mcp server instance
from main import mcp
from services.database_service import DatabaseService
from services.exchange_service import ExchangeService

# Initialize services
exchange_service = ExchangeService(settings.exchanges)
db_service = DatabaseService(settings.database_url)


@mcp.resource("markets://exchanges")
def get_available_exchanges() -> str:
    """Get list of available cryptocurrency exchanges"""
    exchanges = exchange_service.get_available_exchanges()
    return json.dumps(exchanges)


@mcp.resource("markets://symbols/{exchange}")
def get_exchange_symbols(exchange: str) -> str:
    """Get list of available symbols for a specific exchange"""
    symbols = exchange_service.get_symbols(exchange)
    return json.dumps(symbols)


@mcp.resource("markets://price/{exchange}/{symbol}")
def get_current_price(exchange: str, symbol: str) -> str:
    """Get current price for a specific symbol on an exchange"""
    price = exchange_service.get_current_price(exchange, symbol)
    return json.dumps(price)


@mcp.resource("markets://ohlcv/{exchange}/{symbol}/{timeframe}")
def get_ohlcv_data(exchange: str, symbol: str, timeframe: str) -> str:
    """Get OHLCV data for a specific symbol on an exchange"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)  # Last 7 days by default

    ohlcv_data = db_service.get_ohlcv_data(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
    )

    return json.dumps(ohlcv_data)
