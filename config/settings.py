from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Database settings
    database_url: str = "influxdb://localhost:8086/crypto_data"

    # Exchange configurations
    exchanges: dict[str, dict[str, str]] = {
        "binance": {
            "apiKey": "",
            "secret": "",
        },
        "coinbase": {
            "apiKey": "",
            "secret": "",
        },
    }

    # LLM settings
    llm_api_key: str = ""
    llm_model: str = "gpt-4"

    # Data collection settings
    default_symbols: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    default_timeframes: list[str] = ["1m", "15m", "1h", "4h", "1d"]

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
