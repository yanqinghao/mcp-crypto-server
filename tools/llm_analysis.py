import json
from datetime import datetime, timedelta

from config.settings import settings

# Get the mcp server instance
from main import mcp
from services.database_service import DatabaseService
from services.llm_service import LLMService

# Initialize services
llm_service = LLMService(settings.llm_api_key, settings.llm_model)
db_service = DatabaseService(settings.database_url)


@mcp.tool()
def analyze_with_llm(
    exchange: str, symbol: str, days: int = 30, specific_question: str = None
) -> str:
    """
    Analyze cryptocurrency data using a Large Language Model

    Parameters:
    - exchange: Name of the exchange (e.g., 'binance', 'coinbase')
    - symbol: Trading pair symbol (e.g., 'BTC/USDT')
    - days: Number of days to analyze (default: 30)
    - specific_question: Specific aspect to analyze (optional)

    Returns JSON with LLM analysis results
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    # Get OHLCV data
    ohlcv_data = db_service.get_ohlcv_data(
        exchange=exchange, symbol=symbol, timeframe="1d", start_time=start_time, end_time=end_time
    )

    # Format data for LLM
    data_description = []
    for entry in ohlcv_data[-10:]:  # Last 10 entries for brevity
        data_description.append(
            f"Date: {entry['timestamp']}, Open: {entry['open']}, High: {entry['high']}, "
            f"Low: {entry['low']}, Close: {entry['close']}, Volume: {entry['volume']}"
        )

    data_text = "\n".join(data_description)

    # Create prompt for LLM
    if specific_question:
        prompt = f"""
        Analyze the following cryptocurrency data for {symbol} on {exchange} and answer this specific question: {specific_question}

        Data for the last {days} days (showing last 10 entries):
        {data_text}
        """
    else:
        prompt = f"""
        Analyze the following cryptocurrency data for {symbol} on {exchange}.
        Provide insights on price trends, volatility, and potential factors affecting the price.

        Data for the last {days} days (showing last 10 entries):
        {data_text}
        """

    # Get analysis from LLM
    analysis = llm_service.generate_analysis(prompt)

    # Prepare result
    result = {
        "exchange": exchange,
        "symbol": symbol,
        "period_days": days,
        "analysis": analysis,
        "timestamp": datetime.utcnow().isoformat(),
    }

    return json.dumps(result)
