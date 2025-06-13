from fastmcp import Client

import sys
from pathlib import Path

# 添加上一层目录到路径
parent_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(parent_dir))

from models.market_data import (  # noqa
    CandlesInput,
    PriceInput,
    TickerInput,
    OrderBookInput,
    TradesInput,
)
from models.analysis import (  # noqa
    SmaInput,
    RsiInput,
    MacdInput,
    BbandsInput,
    AtrInput,
    AdxInput,
    ObvInput,
    ComprehensiveAnalysisInput,
)


async def main():
    # Connect via SSE
    client = Client(transport="http://localhost:8000/sse")
    async with client:
        # ... use the client
        # current_price = await client.call_tool_mcp(
        #     "crypto_tools_get_current_price", {"inputs": PriceInput(symbol="BTC/USDT")}
        # )
        # print(current_price.model_dump_json())
        # candles = await client.call_tool_mcp(
        #     "crypto_tools_get_candles", {"inputs": CandlesInput(symbol="BTC/USDT")}
        # )
        # print(candles.model_dump_json())
        # ticker = await client.call_tool_mcp(
        #     "crypto_tools_get_ticker", {"inputs": TickerInput(symbol="BTC/USDT")}
        # )
        # print(ticker.model_dump_json())
        # order_book = await client.call_tool_mcp(
        #     "crypto_tools_get_order_book", {"inputs": OrderBookInput(symbol="BTC/USDT")}
        # )
        # print(order_book.model_dump_json())
        # trades = await client.call_tool_mcp(
        #     "crypto_tools_get_recent_trades", {"inputs": TradesInput(symbol="BTC/USDT")}
        # )
        # print(trades.model_dump_json())

        # sma = await client.call_tool_mcp(
        #     "crypto_tools_calculate_sma", {"inputs": SmaInput(symbol="BTC/USDT")}
        # )
        # print(sma.model_dump_json())
        # rsi = await client.call_tool_mcp(
        #     "crypto_tools_calculate_rsi", {"inputs": RsiInput(symbol="BTC/USDT")}
        # )
        # print(rsi.model_dump_json())
        # macd = await client.call_tool_mcp(
        #     "crypto_tools_calculate_macd", {"inputs": MacdInput(symbol="BTC/USDT")}
        # )
        # print(macd.model_dump_json())
        # bbands = await client.call_tool_mcp(
        #     "crypto_tools_calculate_bbands", {"inputs": BbandsInput(symbol="BTC/USDT")}
        # )
        # print(bbands.model_dump_json())
        # atr = await client.call_tool_mcp(
        #     "crypto_tools_calculate_atr", {"inputs": AtrInput(symbol="BTC/USDT")}
        # )
        # print(atr.model_dump_json())
        # adx = await client.call_tool_mcp(
        #     "crypto_tools_calculate_adx", {"inputs": AdxInput(symbol="BTC/USDT")}
        # )
        # print(adx.model_dump_json())
        # obv = await client.call_tool_mcp(
        #     "crypto_tools_calculate_obv", {"inputs": ObvInput(symbol="BTC/USDT")}
        # )
        # print(obv.model_dump_json())

        # market_report = await client.call_tool_mcp(
        #     "crypto_tools_generate_comprehensive_market_report",
        #     {"inputs": ComprehensiveAnalysisInput(symbol="BTC/USDT")},
        # )
        # print(market_report.model_dump_json())

        # indicator_summary = await client.get_prompt(
        #     "analysis_prompts_prompt_for_indicator_summary",
        #     {
        #         "symbol": "BTC/USDT",
        #         "timeframe": "1h",
        #         "indicator_name": "MACD",
        #         "indicator_values": "1,2,3,4,5",
        #     },
        # )
        # print(indicator_summary.model_dump_json())
        # data_requirements = await client.get_prompt(
        #     "analysis_prompts_prompt_for_data_requirements_with_limits",
        #     {
        #         "symbol": "BTC/USDT",
        #         "analysis_type": "Short-term trend analysis",
        #         "target_time_horizon": "short-term",
        #     },
        # )
        # print(data_requirements.model_dump_json())
        # strategy_suggestion = await client.get_prompt(
        #     "analysis_prompts_prompt_for_strategy_suggestion",
        #     {
        #         "symbol": "BTC/USDT",
        #         "timeframe": "1h",
        #         "comprehensive_report_summary": "Comprehensive market analysis report summary",
        #     },
        # )
        # print(strategy_suggestion.model_dump_json())

        # explanation = await client.read_resource_mcp(
        #     "docs://market_info_resources/MACD/explanation"
        # )
        # print(explanation.model_dump_json())
        # default_periods = await client.read_resource_mcp(
        #     "config://market_info_resources/indicators/default_periods"
        # )
        # print(default_periods.model_dump_json())
        # supported_timeframes = await client.read_resource_mcp(
        #     "exchange://market_info_resources/supported_timeframes"
        # )
        # print(supported_timeframes.model_dump_json())
        # time_horizon_map = await client.read_resource_mcp(
        #     "config://market_info_resources/analysis/time_horizon_map"
        # )
        # print(time_horizon_map.model_dump_json())
        # data_length_guidelines = await client.read_resource_mcp(
        #     "docs://market_info_resources/MACD/data_length_guidelines"
        # )
        # print(data_length_guidelines.model_dump_json())
        timeframe_lookback_map = await client.read_resource_mcp(
            "config://market_info_resources/analysis/timeframe_lookback_map"
        )
        print(timeframe_lookback_map.model_dump_json())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
