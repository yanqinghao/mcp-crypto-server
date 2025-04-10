from main import mcp


@mcp.prompt()
def market_trend_prompt(exchange: str, symbol: str, days: int = 30) -> str:
    """Create a prompt for analyzing market trends"""
    return f"""
    Please analyze the cryptocurrency market trends for {symbol} on {exchange} over the past {days} days.

    Consider:
    1. Price movements and key support/resistance levels
    2. Trading volume patterns
    3. Market sentiment indicators
    4. Correlation with other major cryptocurrencies
    5. External factors that might be influencing the price

    Use the market data resources to support your analysis.
    """


@mcp.prompt()
def technical_analysis_prompt(exchange: str, symbol: str, days: int = 30) -> str:
    """Create a prompt for technical analysis"""
    return f"""
    Please perform a technical analysis for {symbol} on {exchange} using data from the past {days} days.

    Include these technical indicators in your analysis:
    1. Moving Averages (MA)
    2. Relative Strength Index (RSI)
    3. Moving Average Convergence Divergence (MACD)
    4. Bollinger Bands
    5. Volume analysis

    What patterns do you observe, and what might they indicate about future price movements?
    """
