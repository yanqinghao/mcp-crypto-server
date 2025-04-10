from datetime import datetime
from typing import Any

import ccxt


class ExchangeService:
    """Service for interacting with cryptocurrency exchanges"""

    def __init__(self, exchange_configs: dict[str, dict[str, str]]):
        """
        Initialize exchange service with configuration

        Args:
            exchange_configs: Dictionary mapping exchange names to their configs
        """
        self.exchanges = {}
        for name, config in exchange_configs.items():
            exchange_class = getattr(ccxt, name)
            self.exchanges[name] = exchange_class(config)

    def get_available_exchanges(self) -> list[str]:
        """Get list of available exchanges"""
        return list(self.exchanges.keys())

    def get_symbols(self, exchange: str) -> list[str]:
        """Get list of available symbols for a specific exchange"""
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange {exchange} not configured")

        self.exchanges[exchange].load_markets()
        return list(self.exchanges[exchange].markets.keys())

    def get_current_price(self, exchange: str, symbol: str) -> dict[str, Any]:
        """Get current price for a specific symbol on an exchange"""
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange {exchange} not configured")

        ticker = self.exchanges[exchange].fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "last": ticker["last"],
            "bid": ticker["bid"],
            "ask": ticker["ask"],
            "timestamp": ticker["timestamp"],
        }

    def get_orderbook(self, exchange: str, symbol: str, depth: int = 10) -> dict[str, Any]:
        """Get order book for a specific symbol on an exchange"""
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange {exchange} not configured")

        orderbook = self.exchanges[exchange].fetch_order_book(symbol, depth)
        return {
            "symbol": symbol,
            "bids": orderbook["bids"][:depth],
            "asks": orderbook["asks"][:depth],
            "timestamp": orderbook["timestamp"],
        }

    def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch OHLCV data from exchange

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            since: Timestamp in milliseconds (optional)
            limit: Maximum number of records (optional)

        Returns:
            List of OHLCV data points
        """
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange {exchange} not configured")

        ohlcv = self.exchanges[exchange].fetch_ohlcv(symbol, timeframe, since, limit)

        # Convert to list of dictionaries
        result = []
        for candle in ohlcv:
            result.append(
                {
                    "timestamp": datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                }
            )

        return result
