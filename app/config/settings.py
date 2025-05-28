import os
import dotenv

dotenv.load_dotenv(override=True)


class Settings:
    # Default periods for indicators
    DEFAULT_SMA_PERIOD: int = 20
    DEFAULT_RSI_PERIOD: int = 14
    DEFAULT_MACD_FAST: int = 12
    DEFAULT_MACD_SLOW: int = 26
    DEFAULT_MACD_SIGNAL: int = 9
    DEFAULT_BBANDS_PERIOD: int = 20
    DEFAULT_BBANDS_NBDEVUP: float = 2.0
    DEFAULT_BBANDS_NBDEVDN: float = 2.0
    DEFAULT_BBANDS_MATYPE: int = 0  # MA_Type.SMA
    DEFAULT_ATR_PERIOD: int = 14
    DEFAULT_ADX_PERIOD: int = 14
    DEFAULT_OBV_DATA_POINTS: int = 200  # Number of data points for OBV calculation

    # Data fetching
    DEFAULT_CANDLE_BUFFER: int = 30  # Buffer for TA-Lib needs in basic fetcher
    DEFAULT_MULTI_DATA_CANDLE_BUFFER: int = (
        50  # Ample buffer for complex indicators in multi-fetcher
    )
    DEFAULT_MULTI_DATA_SAFETY_MARGIN: int = 30  # Additional safety margin for TA-Lib

    # Exchange settings (placeholders - use environment variables or a secure vault in production)
    DEFAULT_EXCHANGE_ID: str = os.getenv("DEFAULT_EXCHANGE_ID", "binance")
    API_KEY: str = os.getenv("API_KEY", "")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    SANDBOX_MODE: bool = os.getenv("SANDBOX_MODE", "false") == "true"

    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000


settings = Settings()
