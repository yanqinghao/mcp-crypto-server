import os

# --- Exchange Configuration ---
EXCHANGE_ID = os.getenv("MCP_EXCHANGE_ID", "binance")

# --- API Keys (Optional) ---
API_KEY = os.getenv("MCP_API_KEY", None)
SECRET_KEY = os.getenv("MCP_SECRET_KEY", None)

# --- Server Configuration ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# --- TA-Lib Configuration ---
DEFAULT_SMA_PERIOD = 14
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
