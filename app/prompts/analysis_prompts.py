from fastmcp import FastMCP, Context

mcp = FastMCP()


@mcp.prompt()
async def prompt_for_indicator_summary(
    ctx: Context,
    symbol: str,
    timeframe: str,
    indicator_name: str,
    indicator_values: str,
) -> str:
    """
    Asks the LLM, in its role as an AI financial analysis assistant,
    to summarize a specific indicator's values and comment on data sufficiency.
    """
    return (
        f"As an AI assistant specialized in financial market analysis, I will interpret the provided technical indicator data.\n"
        f"Please provide a concise technical interpretation for {symbol} on the {timeframe} timeframe, "
        f"based on the following {indicator_name} indicator values: {indicator_values}.\n\n"
        f"What does this suggest about the current market state (e.g., trend, momentum, volatility, overbought/oversold)?\n\n"
        f"Additionally, please briefly comment on whether the data period implied by these values seems adequate for a reliable interpretation of {indicator_name}. "
        f"You can refer to 'docs://indicators/data_length_guidelines' for typical data requirements for {indicator_name}. "
        f"Indicate if caution is warranted due to potentially limited data for {indicator_name} on this {timeframe}."
    )


@mcp.prompt()
async def prompt_for_data_requirements_with_limits(
    ctx: Context,
    symbol: str,
    analysis_type: str,  # e.g., "Short-term trend analysis", "RSI overbought/oversold check", "Long-term investment assessment"
    target_time_horizon: str,  # e.g., "short-term", "ultra-short-term", "long-term"
) -> str:
    """
    Asks the LLM, in its role as an AI financial analysis assistant, to determine
    appropriate timeframes and data lengths for a given analysis type and target horizon,
    emphasizing reasonable and concise data requests by consulting available resources.
    The LLM should refer to resources like:
    - 'docs://indicators/explanation'
    - 'config://indicators/default_periods'
    - 'config://analysis/time_horizon_map'
    - 'docs://indicators/data_length_guidelines'
    - 'config://analysis/timeframe_lookback_map'
    - 'exchange://supported_timeframes'
    """
    return (
        f"As an AI assistant specialized in financial market analysis, my task is to help determine the optimal data parameters for analyzing {symbol} for the purpose of '{analysis_type}', focusing on a '{target_time_horizon}' horizon.\n"
        f"To proceed efficiently and accurately, please specify the following data parameters:\n\n"
        f"1.  **Appropriate Timeframe**: Considering the '{target_time_horizon}' goal, consult 'config://analysis/time_horizon_map' (for translating horizon to timeframes) and 'exchange://supported_timeframes' (for available options) to select the most suitable timeframe.\n\n"
        f"2.  **Required Historical Data Length (Number of Candles or Lookback Period)**:\n"
        f"    a.  Refer to 'docs://indicators/data_length_guidelines' (for indicator-specific needs) and 'config://analysis/timeframe_lookback_map' (for general timeframe lookbacks) to estimate the necessary number of data points or the lookback period.\n"
        f"    b.  **Crucially**: Your data request should ensure the analysis is robust, yet **you must avoid requesting excessively long historical data.** Focus on the most recent data that is directly relevant to the current '{analysis_type}' and '{target_time_horizon}'. Prioritize efficiency.\n"
        f"    c.  If you deem a data length significantly beyond the typical recommendations (as found in the aforementioned resources) essential, briefly justify this necessity for this *specific* analysis.\n\n"
        f"3.  **(Optional) Relevant Indicators and Their Parameters**: If this analysis involves specific technical indicators (e.g., RSI, MACD), list them and their proposed parameters. You can reference 'config://indicators/default_periods' for standard settings and 'docs://indicators/explanation' for their use cases.\n\n"
        f"Please provide your recommendations in a clear JSON format, including keys: 'timeframe', 'num_candles' (or 'lookback_period_description' if more appropriate for expressing duration), and optionally 'indicator_params'.\n"
        f'Example: {{ "timeframe": "1h", "num_candles": 250, "indicator_params": {{ "RSI": {{ "period": 14 }} }} }}'
    )


@mcp.prompt()
async def prompt_for_strategy_suggestion(
    ctx: Context, symbol: str, timeframe: str, comprehensive_report_summary: str
) -> str:
    """
    Asks the LLM, in its role as an AI financial analysis assistant,
    for a trading strategy suggestion based on a market report,
    instructing it to assume the report's data basis is sound unless stated otherwise.
    """
    return (
        f"As an AI assistant specialized in financial market analysis, my role is to help formulate potential trading approaches based on analytical summaries.\n"
        f"Considering the following market analysis summary for {symbol} on the {timeframe} timeframe:\n"
        f"```text\n{comprehensive_report_summary}\n```\n\n"
        f"Based on this summary, what kind of trading strategy (e.g., trend-following, range-bound, breakout) might be appropriate in this context? "
        f"Please briefly explain your reasoning.\n"
        f"For the purpose of this request, please assume that the provided summary is based on an adequate and relevant data period, unless the summary explicitly indicates a data limitation. Focus your strategic suggestions on the information given.\n"
        f"This is for informational and educational purposes only and does not constitute financial advice."
    )


# Add more prompts as needed
