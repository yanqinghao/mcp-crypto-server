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


async def test_crypto_tools(client):
    """测试加密货币工具"""
    print("=== 测试加密货币工具 ===")

    # 加密货币数据获取测试
    print("\n--- 加密货币数据获取 ---")

    current_price = await client.call_tool_mcp(
        "crypto_tools_get_current_price", {"inputs": PriceInput(symbol="BTC/USDT")}
    )
    print("BTC当前价格:", current_price.model_dump_json(indent=2))

    # candles = await client.call_tool_mcp(
    #     "crypto_tools_get_candles", {"inputs": CandlesInput(symbol="BTC/USDT")}
    # )
    # print("BTC K线数据:", candles.model_dump_json(indent=2))

    # 加密货币技术分析测试
    print("\n--- 加密货币技术分析 ---")

    sma = await client.call_tool_mcp(
        "crypto_tools_calculate_sma", {"inputs": SmaInput(symbol="BTC/USDT")}
    )
    print("BTC SMA:", sma.model_dump_json(indent=2))

    # 综合分析报告
    print("\n--- 加密货币综合分析 ---")
    market_report = await client.call_tool_mcp(
        "crypto_tools_generate_comprehensive_market_report",
        {"inputs": ComprehensiveAnalysisInput(symbol="BTC/USDT")},
    )
    print("BTC综合分析报告:", market_report.model_dump_json(indent=2))


async def test_a_stock_tools(client):
    """测试A股工具"""
    print("\n=== 测试A股工具 ===")

    # A股数据获取测试
    print("\n--- A股数据获取 ---")

    # 搜索股票
    search_result = await client.call_tool_mcp(
        "a_hk_stock_tools_search_a_stock_symbols", {"query": "平安", "limit": 5}
    )
    print("A股搜索结果:", search_result.model_dump_json(indent=2))

    # 获取A股价格
    a_price = await client.call_tool_mcp(
        "a_hk_stock_tools_get_a_stock_price", {"inputs": PriceInput(symbol="000001")}
    )
    print("平安银行当前价格:", a_price.model_dump_json(indent=2))

    # 获取A股K线数据
    a_candles = await client.call_tool_mcp(
        "a_hk_stock_tools_get_a_stock_candles",
        {"inputs": CandlesInput(symbol="000001", limit=30)},
    )
    print("平安银行K线数据:", a_candles.model_dump_json(indent=2))

    # 获取A股详细行情
    a_ticker = await client.call_tool_mcp(
        "a_hk_stock_tools_get_a_stock_ticker", {"inputs": TickerInput(symbol="000001")}
    )
    print("平安银行详细行情:", a_ticker.model_dump_json(indent=2))

    # A股技术分析测试
    print("\n--- A股技术分析 ---")

    # A股SMA
    a_sma = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_a_stock_sma",
        {"inputs": SmaInput(symbol="000001", period=20, history_len=10)},
    )
    print("平安银行SMA:", a_sma.model_dump_json(indent=2))

    # A股RSI
    a_rsi = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_a_stock_rsi",
        {"inputs": RsiInput(symbol="000001", period=14, history_len=10)},
    )
    print("平安银行RSI:", a_rsi.model_dump_json(indent=2))

    # A股MACD
    a_macd = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_a_stock_macd",
        {"inputs": MacdInput(symbol="000001", history_len=10)},
    )
    print("平安银行MACD:", a_macd.model_dump_json(indent=2))

    # A股布林带
    a_bbands = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_a_stock_bbands",
        {"inputs": BbandsInput(symbol="000001", period=20, history_len=10)},
    )
    print("平安银行布林带:", a_bbands.model_dump_json(indent=2))

    # A股ATR
    a_atr = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_a_stock_atr",
        {"inputs": AtrInput(symbol="000001", period=14, history_len=10)},
    )
    print("平安银行ATR:", a_atr.model_dump_json(indent=2))

    # A股综合分析报告
    print("\n--- A股综合分析 ---")
    a_report = await client.call_tool_mcp(
        "a_hk_stock_tools_generate_a_stock_comprehensive_report",
        {"inputs": ComprehensiveAnalysisInput(symbol="000001", history_len=10)},
    )
    print("平安银行综合分析报告:", a_report.model_dump_json(indent=2))


async def test_hk_stock_tools(client):
    """测试港股工具"""
    print("\n=== 测试港股工具 ===")

    # 港股数据获取测试
    print("\n--- 港股数据获取 ---")

    # 获取港股价格
    hk_price = await client.call_tool_mcp(
        "a_hk_stock_tools_get_hk_stock_price", {"inputs": PriceInput(symbol="00700")}
    )
    print("腾讯控股当前价格:", hk_price.model_dump_json(indent=2))

    # 获取港股K线数据
    hk_candles = await client.call_tool_mcp(
        "a_hk_stock_tools_get_hk_stock_candles",
        {"inputs": CandlesInput(symbol="00700", limit=30)},
    )
    print("腾讯控股K线数据:", hk_candles.model_dump_json(indent=2))

    # 港股技术分析测试
    print("\n--- 港股技术分析 ---")

    # 港股SMA
    hk_sma = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_hk_stock_sma",
        {"inputs": SmaInput(symbol="00700", period=20, history_len=10)},
    )
    print("腾讯控股SMA:", hk_sma.model_dump_json(indent=2))

    # 港股RSI
    hk_rsi = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_hk_stock_rsi",
        {"inputs": RsiInput(symbol="00700", period=14, history_len=10)},
    )
    print("腾讯控股RSI:", hk_rsi.model_dump_json(indent=2))

    # 港股综合分析报告
    print("\n--- 港股综合分析 ---")
    hk_report = await client.call_tool_mcp(
        "a_hk_stock_tools_generate_hk_stock_comprehensive_report",
        {"inputs": ComprehensiveAnalysisInput(symbol="00700", history_len=10)},
    )
    print("腾讯控股综合分析报告:", hk_report.model_dump_json(indent=2))


async def test_us_stock_tools(client):
    """测试美股工具"""
    print("\n=== 测试美股工具 ===")

    # 美股数据获取测试
    print("\n--- 美股数据获取 ---")

    # 搜索美股
    us_search_result = await client.call_tool_mcp(
        "us_stock_tools_search_us_stock_symbols", {"query": "apple", "limit": 5}
    )
    print("美股搜索结果:", us_search_result.model_dump_json(indent=2))

    # 获取美股价格
    us_price = await client.call_tool_mcp(
        "us_stock_tools_get_us_stock_price", {"inputs": PriceInput(symbol="AAPL")}
    )
    print("苹果当前价格:", us_price.model_dump_json(indent=2))

    # 获取美股K线数据
    us_candles = await client.call_tool_mcp(
        "us_stock_tools_get_us_stock_candles",
        {"inputs": CandlesInput(symbol="AAPL", limit=30)},
    )
    print("苹果K线数据:", us_candles.model_dump_json(indent=2))

    # 获取美股详细行情
    us_ticker = await client.call_tool_mcp(
        "us_stock_tools_get_us_stock_ticker", {"inputs": TickerInput(symbol="AAPL")}
    )
    print("苹果详细行情:", us_ticker.model_dump_json(indent=2))

    # 获取美股公司信息
    us_info = await client.call_tool_mcp(
        "us_stock_tools_get_us_stock_info", {"symbol": "AAPL"}
    )
    print("苹果公司信息:", us_info.model_dump_json(indent=2))

    # 获取美股财务数据
    us_financials = await client.call_tool_mcp(
        "us_stock_tools_get_us_stock_financials",
        {"symbol": "AAPL", "statement_type": "income", "quarterly": False},
    )
    print("苹果财务数据:", us_financials.model_dump_json(indent=2))

    # 获取美股期权数据
    us_options = await client.call_tool_mcp(
        "us_stock_tools_get_us_stock_options", {"symbol": "AAPL"}
    )
    print("苹果期权数据:", us_options.model_dump_json(indent=2))

    # 美股技术分析测试
    print("\n--- 美股技术分析 ---")

    # 美股SMA
    us_sma = await client.call_tool_mcp(
        "us_stock_tools_calculate_us_stock_sma",
        {"inputs": SmaInput(symbol="AAPL", period=20, history_len=10)},
    )
    print("苹果SMA:", us_sma.model_dump_json(indent=2))

    # 美股RSI
    us_rsi = await client.call_tool_mcp(
        "us_stock_tools_calculate_us_stock_rsi",
        {"inputs": RsiInput(symbol="AAPL", period=14, history_len=10)},
    )
    print("苹果RSI:", us_rsi.model_dump_json(indent=2))

    # 美股MACD
    us_macd = await client.call_tool_mcp(
        "us_stock_tools_calculate_us_stock_macd",
        {"inputs": MacdInput(symbol="AAPL", history_len=10)},
    )
    print("苹果MACD:", us_macd.model_dump_json(indent=2))

    # 美股布林带
    us_bbands = await client.call_tool_mcp(
        "us_stock_tools_calculate_us_stock_bbands",
        {"inputs": BbandsInput(symbol="AAPL", period=20, history_len=10)},
    )
    print("苹果布林带:", us_bbands.model_dump_json(indent=2))

    # 美股ATR
    us_atr = await client.call_tool_mcp(
        "us_stock_tools_calculate_us_stock_atr",
        {"inputs": AtrInput(symbol="AAPL", period=14, history_len=10)},
    )
    print("苹果ATR:", us_atr.model_dump_json(indent=2))

    # 美股综合分析报告
    print("\n--- 美股综合分析 ---")
    us_report = await client.call_tool_mcp(
        "us_stock_tools_generate_us_stock_comprehensive_report",
        {"inputs": ComprehensiveAnalysisInput(symbol="AAPL", history_len=10)},
    )
    print("苹果综合分析报告:", us_report.model_dump_json(indent=2))


async def test_prompts_and_resources(client):
    """测试提示符和资源"""
    print("\n=== 测试提示符和资源 ===")

    # 测试提示符
    print("\n--- 测试提示符 ---")

    indicator_summary = await client.get_prompt(
        "analysis_prompts_prompt_for_indicator_summary",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "indicator_name": "MACD",
            "indicator_values": "1,2,3,4,5",
        },
    )
    print("指标总结提示符:", indicator_summary.model_dump_json(indent=2))

    data_requirements = await client.get_prompt(
        "analysis_prompts_prompt_for_data_requirements_with_limits",
        {
            "symbol": "BTC/USDT",
            "analysis_type": "Short-term trend analysis",
            "target_time_horizon": "short-term",
        },
    )
    print("数据需求提示符:", data_requirements.model_dump_json(indent=2))

    strategy_suggestion = await client.get_prompt(
        "analysis_prompts_prompt_for_strategy_suggestion",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "comprehensive_report_summary": "Comprehensive market analysis report summary",
        },
    )
    print("策略建议提示符:", strategy_suggestion.model_dump_json(indent=2))

    # 测试资源
    print("\n--- 测试资源 ---")

    explanation = await client.read_resource_mcp(
        "docs://market_info_resources/MACD/explanation"
    )
    print("MACD解释:", explanation.model_dump_json(indent=2))

    default_periods = await client.read_resource_mcp(
        "config://market_info_resources/indicators/default_periods"
    )
    print("默认周期:", default_periods.model_dump_json(indent=2))

    supported_timeframes = await client.read_resource_mcp(
        "exchange://market_info_resources/supported_timeframes"
    )
    print("支持的时间框架:", supported_timeframes.model_dump_json(indent=2))

    time_horizon_map = await client.read_resource_mcp(
        "config://market_info_resources/analysis/time_horizon_map"
    )
    print("时间范围映射:", time_horizon_map.model_dump_json(indent=2))

    data_length_guidelines = await client.read_resource_mcp(
        "docs://market_info_resources/MACD/data_length_guidelines"
    )
    print("数据长度指南:", data_length_guidelines.model_dump_json(indent=2))

    timeframe_lookback_map = await client.read_resource_mcp(
        "config://market_info_resources/analysis/timeframe_lookback_map"
    )
    print("时间框架回看映射:", timeframe_lookback_map.model_dump_json(indent=2))


async def main():
    """主测试函数"""
    # Connect via SSE
    client = Client(transport="http://localhost:8888/sse")
    async with client:
        try:
            # 选择要测试的模块
            test_crypto = False  # 测试加密货币工具
            test_a_stock = False  # 测试A股工具
            test_hk_stock = True  # 测试港股工具
            test_us_stock = False  # 测试美股工具
            test_prompts = False  # 测试提示符和资源

            if test_crypto:
                await test_crypto_tools(client)

            if test_a_stock:
                await test_a_stock_tools(client)

            if test_hk_stock:
                await test_hk_stock_tools(client)

            if test_us_stock:
                await test_us_stock_tools(client)

            if test_prompts:
                await test_prompts_and_resources(client)

        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
