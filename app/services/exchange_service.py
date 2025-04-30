import ccxt.async_support as ccxt
from config import settings
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ExchangeService:
    """Handles interactions with the cryptocurrency exchange via CCXT."""

    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.exchange_id = exchange_id
        self.exchange = self._initialize_exchange(api_key, secret_key)
        logger.info(f"Initialized CCXT exchange: {self.exchange_id}")

    def _initialize_exchange(
        self, api_key: Optional[str] = None, secret_key: Optional[str] = None
    ):
        """Initializes the CCXT exchange instance."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            config = {
                "enableRateLimit": True,  # [3, 4]
                "options": {
                    "adjustForTimeDifference": True,
                },
            }
            if api_key and secret_key:
                config["apiKey"] = api_key
                config["secret"] = secret_key
                logger.info(f"API keys loaded for {self.exchange_id}.")
            else:
                logger.info(
                    f"No API keys provided for {self.exchange_id}. Using public endpoints only."
                )

            return exchange_class(config)
        except AttributeError:
            logger.error(f"Exchange '{self.exchange_id}' not found in CCXT.")
            raise ValueError(f"Unsupported exchange ID: {self.exchange_id}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange '{self.exchange_id}': {e}")
            raise

    async def get_ticker(self, symbol: str) -> Optional[dict]:
        """Fetches ticker information for a given symbol."""
        try:
            # Check if fetchTicker is supported [3]
            if not self.exchange.has.get("fetchTicker"):
                logger.warning(
                    f"Exchange {self.exchange_id} does not support fetchTicker."
                )
                return None
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except ccxt.NetworkError as e:
            logger.error(
                f"Network error fetching ticker for {symbol} from {self.exchange_id}: {e}"
            )
            raise
        except ccxt.ExchangeError as e:
            logger.error(
                f"Exchange error fetching ticker for {symbol} from {self.exchange_id}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error fetching ticker for {symbol} from {self.exchange_id}: {e}"
            )
            raise

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Optional[list]:
        """Fetches OHLCV (candlestick) data."""
        if not self.exchange.has.get("fetchOHLCV"):  # [3]
            logger.warning(f"Exchange {self.exchange_id} does not support fetchOHLCV.")
            return None
        try:
            # Check if the timeframe is supported by the exchange [3]
            if timeframe not in self.exchange.timeframes:
                logger.warning(
                    f"Timeframe '{timeframe}' may not be supported by {self.exchange_id}. Available: {list(self.exchange.timeframes.keys())}"
                )
                # Proceed anyway, CCXT might handle aliases or raise a specific error

            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=limit
            )
            return ohlcv
        except ccxt.NetworkError as e:
            logger.error(
                f"Network error fetching OHLCV for {symbol} ({timeframe}) from {self.exchange_id}: {e}"
            )
            raise
        except ccxt.ExchangeError as e:
            logger.error(
                f"Exchange error fetching OHLCV for {symbol} ({timeframe}) from {self.exchange_id}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error fetching OHLCV for {symbol} ({timeframe}) from {self.exchange_id}: {e}"
            )
            raise

    async def close_connection(self):
        """Closes the underlying CCXT connection."""
        if self.exchange:
            try:
                await self.exchange.close()
                logger.info(f"Closed connection to {self.exchange_id}")
            except Exception as e:
                logger.error(f"Error closing connection to {self.exchange_id}: {e}")


# --- Singleton Instance ---
exchange_service = ExchangeService(
    settings.EXCHANGE_ID, settings.API_KEY, settings.SECRET_KEY
)
