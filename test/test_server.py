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
    # search_result = await client.call_tool_mcp(
    #     "a_hk_stock_tools_search_a_stock_symbols", {"query": "平安", "limit": 5}
    # )
    # print("A股搜索结果:", search_result.model_dump_json(indent=2))

    # # 获取A股价格
    # a_price = await client.call_tool_mcp(
    #     "a_hk_stock_tools_get_a_stock_price", {"inputs": PriceInput(symbol="000001")}
    # )
    # print("平安银行当前价格:", a_price.model_dump_json(indent=2))

    # # 获取A股K线数据
    # a_candles = await client.call_tool_mcp(
    #     "a_hk_stock_tools_get_a_stock_candles",
    #     {"inputs": CandlesInput(symbol="000001", limit=30)},
    # )
    # print("平安银行K线数据:", a_candles.model_dump_json(indent=2))

    # # 获取A股详细行情
    # a_ticker = await client.call_tool_mcp(
    #     "a_hk_stock_tools_get_a_stock_ticker", {"inputs": TickerInput(symbol="000001")}
    # )
    # print("平安银行详细行情:", a_ticker.model_dump_json(indent=2))

    # # A股技术分析测试
    # print("\n--- A股技术分析 ---")

    # # A股SMA
    # a_sma = await client.call_tool_mcp(
    #     "a_hk_stock_tools_calculate_a_stock_sma",
    #     {"inputs": SmaInput(symbol="000001", period=20, history_len=10)},
    # )
    # print("平安银行SMA:", a_sma.model_dump_json(indent=2))

    # # A股RSI
    # a_rsi = await client.call_tool_mcp(
    #     "a_hk_stock_tools_calculate_a_stock_rsi",
    #     {"inputs": RsiInput(symbol="000001", period=14, history_len=10)},
    # )
    # print("平安银行RSI:", a_rsi.model_dump_json(indent=2))

    # # A股MACD
    # a_macd = await client.call_tool_mcp(
    #     "a_hk_stock_tools_calculate_a_stock_macd",
    #     {"inputs": MacdInput(symbol="000001", history_len=10)},
    # )
    # print("平安银行MACD:", a_macd.model_dump_json(indent=2))

    # # A股布林带
    # a_bbands = await client.call_tool_mcp(
    #     "a_hk_stock_tools_calculate_a_stock_bbands",
    #     {"inputs": BbandsInput(symbol="000001", period=20, history_len=10)},
    # )
    # print("平安银行布林带:", a_bbands.model_dump_json(indent=2))

    # # A股ATR
    # a_atr = await client.call_tool_mcp(
    #     "a_hk_stock_tools_calculate_a_stock_atr",
    #     {"inputs": AtrInput(symbol="000001", period=14, history_len=10)},
    # )
    # print("平安银行ATR:", a_atr.model_dump_json(indent=2))

    # A股综合分析报告
    print("\n--- A股综合分析 ---")
    a_report = await client.call_tool_mcp(
        "a_hk_stock_tools_generate_a_stock_comprehensive_report",
        {
            "inputs": ComprehensiveAnalysisInput(
                symbol="510050", history_len=1000, timeframe="daily"
            )
        },
    )
    print("平安银行综合分析报告:", a_report.model_dump_json(indent=2))


async def test_hk_stock_tools(client):
    """测试港股工具"""
    print("\n=== 测试港股工具 ===")

    # 港股数据获取测试
    print("\n--- 港股数据获取 ---")

    # 搜索股票
    search_result = await client.call_tool_mcp(
        "a_hk_stock_tools_search_hk_stock_symbols", {"query": "比亚迪", "limit": 5}
    )
    print("港股搜索结果:", search_result.model_dump_json(indent=2))

    # 获取港股价格
    hk_price = await client.call_tool_mcp(
        "a_hk_stock_tools_get_hk_stock_price", {"inputs": PriceInput(symbol="01211")}
    )
    print("腾讯控股当前价格:", hk_price.model_dump_json(indent=2))

    # 获取港股K线数据
    hk_candles = await client.call_tool_mcp(
        "a_hk_stock_tools_get_hk_stock_candles",
        {"inputs": CandlesInput(symbol="01211", limit=30)},
    )
    print("腾讯控股K线数据:", hk_candles.model_dump_json(indent=2))

    # 港股技术分析测试
    print("\n--- 港股技术分析 ---")

    # 港股SMA
    hk_sma = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_hk_stock_sma",
        {"inputs": SmaInput(symbol="01211", period=20, history_len=10)},
    )
    print("腾讯控股SMA:", hk_sma.model_dump_json(indent=2))

    # 港股RSI
    hk_rsi = await client.call_tool_mcp(
        "a_hk_stock_tools_calculate_hk_stock_rsi",
        {"inputs": RsiInput(symbol="01211", period=14, history_len=10)},
    )
    print("腾讯控股RSI:", hk_rsi.model_dump_json(indent=2))

    # 港股综合分析报告
    print("\n--- 港股综合分析 ---")
    hk_report = await client.call_tool_mcp(
        "a_hk_stock_tools_generate_hk_stock_comprehensive_report",
        {"inputs": ComprehensiveAnalysisInput(symbol="01211", history_len=10)},
    )
    print("腾讯控股综合分析报告:", hk_report.model_dump_json(indent=2))


async def test_us_stock_tools(client):
    """测试美股工具 - 完整版本"""
    print("\n=== 测试美股工具 ===")

    # ==================== 美股数据获取测试 ====================
    print("\n--- 美股数据获取 ---")

    # # 搜索美股
    # print("1. 搜索美股...")
    # us_search_result = await client.call_tool_mcp(
    #     "us_stock_tools_search_us_stock_symbols", {"query": "apple", "limit": 5}
    # )
    # print("美股搜索结果:", us_search_result.model_dump_json(indent=2))

    # # 验证股票代码
    # print("\n2. 验证股票代码...")
    # validation_result = await client.call_tool_mcp(
    #     "us_stock_tools_validate_us_stock_symbol_tool", {"symbol": "AAPL"}
    # )
    # print("AAPL验证结果:", validation_result.model_dump_json(indent=2))

    # # 按类型查找证券
    # print("\n3. 查找ETF...")
    # etf_lookup = await client.call_tool_mcp(
    #     "us_stock_tools_lookup_us_securities",
    #     {"query": "technology", "security_type": "etf", "count": 10}
    # )
    # print("科技ETF查找结果:", etf_lookup.model_dump_json(indent=2))

    # # 获取美股价格
    # print("\n4. 获取美股价格...")
    # us_price = await client.call_tool_mcp(
    #     "us_stock_tools_get_us_stock_price",
    #     {"inputs": {"symbol": "AAPL"}}
    # )
    # print("苹果当前价格:", us_price.model_dump_json(indent=2))

    # # 获取美股K线数据
    # print("\n5. 获取美股K线数据...")
    # us_candles = await client.call_tool_mcp(
    #     "us_stock_tools_get_us_stock_candles",
    #     {"inputs": {"symbol": "AAPL", "timeframe": "1d", "limit": 30}},
    # )
    # print("苹果K线数据:", us_candles.model_dump_json(indent=2))

    # # 获取美股详细行情
    # print("\n6. 获取美股详细行情...")
    # us_ticker = await client.call_tool_mcp(
    #     "us_stock_tools_get_us_stock_ticker",
    #     {"inputs": {"symbol": "AAPL"}}
    # )
    # print("苹果详细行情:", us_ticker.model_dump_json(indent=2))

    # # 获取美股公司信息
    # print("\n7. 获取美股公司信息...")
    # us_info = await client.call_tool_mcp(
    #     "us_stock_tools_get_us_stock_info", {"symbol": "AAPL"}
    # )
    # print("苹果公司信息:", us_info.model_dump_json(indent=2))

    # # 获取美股财务数据
    # print("\n8. 获取美股财务数据...")
    # us_financials = await client.call_tool_mcp(
    #     "us_stock_tools_get_us_stock_financials",
    #     {"symbol": "AAPL", "statement_type": "income", "quarterly": False},
    # )
    # print("苹果财务数据:", us_financials.model_dump_json(indent=2))

    # # 获取美股期权数据
    # print("\n9. 获取美股期权数据...")
    # us_options = await client.call_tool_mcp(
    #     "us_stock_tools_get_us_stock_options", {"symbol": "AAPL"}
    # )
    # print("苹果期权数据:", us_options.model_dump_json(indent=2))

    # # ==================== 新增市场数据工具测试 ====================
    # print("\n--- 新增市场数据工具 ---")

    # # 获取市场指数
    # print("\n10. 获取市场指数...")
    # market_indices = await client.call_tool_mcp(
    #     "us_stock_tools_get_market_indices",
    #     {"indices": ["^GSPC", "^DJI", "^IXIC"], "period": "1mo", "interval": "1d"}
    # )
    # print("市场指数数据:", market_indices.model_dump_json(indent=2))

    # # 获取实时股票数据
    # print("\n11. 获取实时股票数据...")
    # realtime_data = await client.call_tool_mcp(
    #     "us_stock_tools_get_realtime_stock_data",
    #     {"symbols": ["AAPL", "MSFT", "GOOGL"]}
    # )
    # print("实时股票数据:", realtime_data.model_dump_json(indent=2))

    # # 获取分析师推荐
    # print("\n12. 获取分析师推荐...")
    # analyst_recs = await client.call_tool_mcp(
    #     "us_stock_tools_get_analyst_recommendations",
    #     {"symbol": "AAPL"}
    # )
    # print("苹果分析师推荐:", analyst_recs.model_dump_json(indent=2))

    # # 获取分红历史
    # print("\n13. 获取分红历史...")
    # dividend_history = await client.call_tool_mcp(
    #     "us_stock_tools_get_dividend_history",
    #     {"symbol": "AAPL", "period": "2y"}
    # )
    # print("苹果分红历史:", dividend_history.model_dump_json(indent=2))

    # # ==================== 美股技术分析测试 ====================
    # print("\n--- 美股技术分析 ---")

    # # 美股SMA
    # print("\n14. 计算美股SMA...")
    # us_sma = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_sma",
    #     {"inputs": {"symbol": "AAPL", "timeframe": "1d", "period": 20, "history_len": 10}},
    # )
    # print("苹果SMA:", us_sma.model_dump_json(indent=2))

    # # 美股EMA (新增)
    # print("\n15. 计算美股EMA...")
    # us_ema = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_ema",
    #     {"symbol": "AAPL", "period": 20, "timeframe": "1d", "history_len": 10}
    # )
    # print("苹果EMA:", us_ema.model_dump_json(indent=2))

    # # 美股RSI
    # print("\n16. 计算美股RSI...")
    # us_rsi = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_rsi",
    #     {"inputs": {"symbol": "AAPL", "timeframe": "1d", "period": 14, "history_len": 10}},
    # )
    # print("苹果RSI:", us_rsi.model_dump_json(indent=2))

    # # 美股MACD
    # print("\n17. 计算美股MACD...")
    # us_macd = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_macd",
    #     {"inputs": {"symbol": "AAPL", "timeframe": "1d", "history_len": 10}},
    # )
    # print("苹果MACD:", us_macd.model_dump_json(indent=2))

    # # 美股布林带
    # print("\n18. 计算美股布林带...")
    # us_bbands = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_bbands",
    #     {"inputs": {"symbol": "AAPL", "timeframe": "1d", "period": 20, "history_len": 10}},
    # )
    # print("苹果布林带:", us_bbands.model_dump_json(indent=2))

    # # 美股ATR
    # print("\n19. 计算美股ATR...")
    # us_atr = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_atr",
    #     {"inputs": {"symbol": "AAPL", "timeframe": "1d", "period": 14, "history_len": 10}},
    # )
    # print("苹果ATR:", us_atr.model_dump_json(indent=2))

    # # ==================== 新增技术指标测试 ====================
    # print("\n--- 新增技术指标 ---")

    # 随机指标
    # print("\n20. 计算随机指标...")
    # us_stoch = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_stochastic",
    #     {"symbol": "AAPL", "k_period": 14, "d_period": 3, "timeframe": "1d", "history_len": 10}
    # )
    # print("苹果随机指标:", us_stoch.model_dump_json(indent=2))

    # # 威廉指标
    # print("\n21. 计算威廉指标...")
    # us_willr = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_williams_r",
    #     {"symbol": "AAPL", "period": 14, "timeframe": "1d", "history_len": 10}
    # )
    # print("苹果威廉指标:", us_willr.model_dump_json(indent=2))

    # 成交量指标
    # print("\n22. 计算成交量指标...")
    # us_volume = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_us_stock_volume_indicators",
    #     {"symbol": "AAPL", "timeframe": "1d", "history_len": 10}
    # )
    # print("苹果成交量指标:", us_volume.model_dump_json(indent=2))

    # # ==================== 风险管理工具测试 ====================
    # print("\n--- 风险管理工具 ---")

    # # 计算风险指标
    # print("\n23. 计算风险指标...")
    # risk_metrics = await client.call_tool_mcp(
    #     "us_stock_tools_calculate_risk_metrics",
    #     {"symbol": "AAPL", "period": "1y", "timeframe": "1d", "benchmark_symbol": "^GSPC"}
    # )
    # print("苹果风险指标:", risk_metrics.model_dump_json(indent=2))

    # # ==================== 股票筛选工具测试 ====================
    # print("\n--- 股票筛选工具 ---")

    # # 股票筛选
    # print("\n24. 股票筛选...")
    # screening_result = await client.call_tool_mcp(
    #     "us_stock_tools_screen_stocks_by_criteria",
    #     {
    #         "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    #         "min_price": 100.0,
    #         "max_price": 500.0,
    #         "min_volume": 1000000,
    #         "max_pe_ratio": 50.0
    #     }
    # )
    # print("股票筛选结果:", screening_result.model_dump_json(indent=2))

    # # ==================== 美股综合分析测试 ====================
    # print("\n--- 美股综合分析 ---")

    # # 综合分析报告
    print("\n25. 生成综合分析报告...")
    us_report = await client.call_tool_mcp(
        "us_stock_tools_generate_us_stock_comprehensive_report",
        {"inputs": {"symbol": "AAPL", "timeframe": "1d", "history_len": 10}},
    )
    print("苹果综合分析报告:", us_report.model_dump_json(indent=2))

    # print("\n=== 美股工具测试完成 ===")


async def test_etf_tools(client):
    """测试ETF工具集 - 完整版本"""
    print("\n=== 测试ETF工具集 ===")

    # ==================== ETF基础数据获取测试 ====================
    print("\n--- ETF基础数据获取 ---")

    # 1. 搜索ETF
    # print("1. 搜索ETF...")
    # etf_search_result = await client.call_tool_mcp(
    #     "etf_tools_search_etf_symbols",
    #     {
    #         "inputs": {
    #             "keyword": "龙头",
    #             "market_type": "A",
    #             "limit": 5
    #         }
    #     }
    # )
    # print("科技类ETF搜索结果:", etf_search_result.model_dump_json(indent=2))

    # 搜索特定ETF
    # print("\n1.2. 搜索特定ETF代码...")
    # specific_etf_search = await client.call_tool_mcp(
    #     "etf_tools_search_etf_symbols",
    #     {
    #         "inputs": {
    #             "keyword": "510050",
    #             "market_type": "A",
    #             "limit": 3
    #         }
    #     }
    # )
    # print("50ETF搜索结果:", specific_etf_search.model_dump_json(indent=2))

    # # 2. 获取ETF分类列表
    # print("\n2. 获取ETF分类列表...")
    # etf_category = await client.call_tool_mcp(
    #     "etf_tools_get_etf_category_list",
    #     {
    #         "inputs": {
    #             "category": "ETF基金",
    #             "limit": 10
    #         }
    #     }
    # )
    # print("ETF基金分类:", etf_category.model_dump_json(indent=2))

    # # 3. 获取同花顺ETF排行
    # print("\n3. 获取同花顺ETF排行...")
    # etf_ths_ranking = await client.call_tool_mcp(
    #     "etf_tools_get_etf_ths_ranking",
    #     {
    #         "date": None,  # 使用当前日期
    #         "limit": 10
    #     }
    # )
    # print("同花顺ETF排行:", etf_ths_ranking.model_dump_json(indent=2))

    # # 4. 获取ETF当前价格
    # print("\n4. 获取ETF当前价格...")
    # etf_price = await client.call_tool_mcp(
    #     "etf_tools_get_etf_price",
    #     {
    #         "inputs": {
    #             "symbol": "510050",  # 50ETF
    #             "market_type": "A"
    #         }
    #     }
    # )
    # print("50ETF当前价格:", etf_price.model_dump_json(indent=2))

    # # 5. 获取ETF K线数据
    # print("\n5. 获取ETF K线数据...")
    # etf_candles = await client.call_tool_mcp(
    #     "etf_tools_get_etf_candles",
    #     {
    #         "inputs": {
    #             "symbol": "510050",
    #             "timeframe": "1d",
    #             "limit": 30,
    #             "market_type": "A",
    #             "adjust": "qfq"
    #         }
    #     }
    # )
    # print("50ETF K线数据:", etf_candles.model_dump_json(indent=2))

    # # 6. 获取ETF详细行情
    # print("\n6. 获取ETF详细行情...")
    # etf_ticker = await client.call_tool_mcp(
    #     "etf_tools_get_etf_ticker",
    #     {
    #         "inputs": {
    #             "symbol": "510050",
    #             "market_type": "A"
    #         }
    #     }
    # )
    # print("50ETF详细行情:", etf_ticker.model_dump_json(indent=2))

    # # 7. 获取ETF分时数据 (仅A股)
    # print("\n7. 获取ETF分时数据...")
    # etf_intraday = await client.call_tool_mcp(
    #     "etf_tools_get_etf_intraday_data",
    #     {
    #         "symbol": "510050",
    #         "period": "1",  # 1分钟
    #         "market_type": "A"
    #     }
    # )
    # print("50ETF分时数据:", etf_intraday.model_dump_json(indent=2))

    # # ==================== ETF技术分析测试 ====================
    # print("\n--- ETF技术分析 ---")

    # # 8. ETF SMA
    # print("\n8. 计算ETF SMA...")
    # etf_sma = await client.call_tool_mcp(
    #     "etf_tools_calculate_etf_sma",
    #     {
    #         "inputs": {
    #             "symbol": "510050",
    #             "timeframe": "1d",
    #             "period": 20,
    #             "history_len": 10
    #         },
    #         "market_type": "A"
    #     }
    # )
    # print("50ETF SMA:", etf_sma.model_dump_json(indent=2))

    # # 9. ETF RSI
    # print("\n9. 计算ETF RSI...")
    # etf_rsi = await client.call_tool_mcp(
    #     "etf_tools_calculate_etf_rsi",
    #     {
    #         "inputs": {
    #             "symbol": "510050",
    #             "timeframe": "1d",
    #             "period": 14,
    #             "history_len": 10
    #         },
    #         "market_type": "A"
    #     }
    # )
    # print("50ETF RSI:", etf_rsi.model_dump_json(indent=2))

    # # 10. ETF MACD
    # print("\n10. 计算ETF MACD...")
    # etf_macd = await client.call_tool_mcp(
    #     "etf_tools_calculate_etf_macd",
    #     {
    #         "inputs": {
    #             "symbol": "510050",
    #             "timeframe": "1d",
    #             "fast_period": 12,
    #             "slow_period": 26,
    #             "signal_period": 9,
    #             "history_len": 10
    #         },
    #         "market_type": "A"
    #     }
    # )
    # print("50ETF MACD:", etf_macd.model_dump_json(indent=2))

    # # 11. ETF 布林带
    # print("\n11. 计算ETF 布林带...")
    # etf_bbands = await client.call_tool_mcp(
    #     "etf_tools_calculate_etf_bbands",
    #     {
    #         "inputs": {
    #             "symbol": "510050",
    #             "timeframe": "1d",
    #             "period": 20,
    #             "nbdevup": 2.0,
    #             "nbdevdn": 2.0,
    #             "matype": 0,
    #             "history_len": 10
    #         },
    #         "market_type": "A"
    #     }
    # )
    # print("50ETF 布林带:", etf_bbands.model_dump_json(indent=2))

    # # 12. ETF ATR
    # print("\n12. 计算ETF ATR...")
    # etf_atr = await client.call_tool_mcp(
    #     "etf_tools_calculate_etf_atr",
    #     {
    #         "inputs": {
    #             "symbol": "510050",
    #             "timeframe": "1d",
    #             "period": 14,
    #             "history_len": 10
    #         },
    #         "market_type": "A"
    #     }
    # )
    # print("50ETF ATR:", etf_atr.model_dump_json(indent=2))

    # # ==================== ETF特有分析测试 ====================
    # print("\n--- ETF特有分析 ---")

    # 13. ETF折价率分析 (仅A股)
    # print("\n13. ETF折价率分析...")
    # etf_discount = await client.call_tool_mcp(
    #     "etf_tools_get_etf_discount_analysis",
    #     {
    #         "inputs": {
    #             "symbol": "510050",
    #             "market_type": "A",
    #             "days": 30
    #         }
    #     }
    # )
    # print("50ETF折价率分析:", etf_discount.model_dump_json(indent=2))

    # 14. ETF对比分析
    # print("\n14. ETF对比分析...")
    # etf_comparison = await client.call_tool_mcp(
    #     "etf_tools_compare_etf_performance",
    #     {
    #         "inputs": {
    #             "symbols": ["510050", "510300", "159919"],  # 50ETF, 沪深300ETF, 科技ETF
    #             "market_type": "A",
    #             "period_days": 30
    #         }
    #     }
    # )
    # print("ETF对比分析:", etf_comparison.model_dump_json(indent=2))

    # 15. ETF筛选
    # print("\n15. ETF筛选...")
    # etf_screening = await client.call_tool_mcp(
    #     "etf_tools_screen_etf_by_criteria",
    #     {
    #         "inputs": {
    #             "market_type": "A",
    #             "min_volume": 1000000,
    #             "max_discount_rate": 1.0,
    #             "min_nav": 1.0,
    #             "category": None,
    #             "limit": 10
    #         }
    #     }
    # )
    # print("ETF筛选结果:", etf_screening.model_dump_json(indent=2))

    # # ==================== 港股ETF测试 ====================
    # print("\n--- 港股ETF测试 ---")

    # 16. 港股ETF价格
    # print("\n16. 获取港股ETF价格...")
    # hk_etf_price = await client.call_tool_mcp(
    #     "etf_tools_get_etf_price",
    #     {
    #         "inputs": {
    #             "symbol": "02800",  # 盈富基金
    #             "market_type": "HK"
    #         }
    #     }
    # )
    # print("盈富基金当前价格:", hk_etf_price.model_dump_json(indent=2))

    # 17. 港股ETF K线
    # print("\n17. 获取港股ETF K线...")
    # hk_etf_candles = await client.call_tool_mcp(
    #     "etf_tools_get_etf_candles",
    #     {
    #         "inputs": {
    #             "symbol": "02800",
    #             "timeframe": "1d",
    #             "limit": 20,
    #             "market_type": "HK"
    #         }
    #     }
    # )
    # print("盈富基金K线数据:", hk_etf_candles.model_dump_json(indent=2))

    # 18. 港股ETF技术分析
    # print("\n18. 港股ETF SMA...")
    # hk_etf_sma = await client.call_tool_mcp(
    #     "etf_tools_calculate_etf_sma",
    #     {
    #         "inputs": {
    #             "symbol": "02800",
    #             "timeframe": "1d",
    #             "period": 10,
    #             "history_len": 5
    #         },
    #         "market_type": "HK"
    #     }
    # )
    # print("盈富基金SMA:", hk_etf_sma.model_dump_json(indent=2))

    # # ==================== ETF综合分析报告测试 ====================
    # print("\n--- ETF综合分析报告 ---")

    # 19. A股ETF综合报告
    print("\n19. 生成A股ETF综合分析报告...")
    a_etf_report = await client.call_tool_mcp(
        "etf_tools_generate_etf_comprehensive_report",
        {
            "inputs": {
                "symbol": "510050",
                "timeframe": "1d",
                "history_len": 30,
                "indicators_to_include": ["SMA", "RSI", "MACD", "BBANDS"],
                "sma_period": 20,
                "rsi_period": 14,
                "macd_fast_period": 12,
                "macd_slow_period": 26,
                "macd_signal_period": 9,
                "bbands_period": 20,
            },
            "market_type": "A",
        },
    )
    print("50ETF综合分析报告:", a_etf_report.model_dump_json(indent=2))

    # 20. 港股ETF综合报告
    # print("\n20. 生成港股ETF综合分析报告...")
    # hk_etf_report = await client.call_tool_mcp(
    #     "etf_tools_generate_etf_comprehensive_report",
    #     {
    #         "inputs": {
    #             "symbol": "02800",
    #             "timeframe": "1d",
    #             "history_len": 20,
    #             "indicators_to_include": ["SMA", "RSI"],
    #             "sma_period": 10,
    #             "rsi_period": 14
    #         },
    #         "market_type": "HK"
    #     }
    # )
    # print("盈富基金综合分析报告:", hk_etf_report.model_dump_json(indent=2))

    # # ==================== 批量ETF分析测试 ====================
    # print("\n--- 批量ETF分析测试 ---")

    # 21. 批量获取多个ETF数据
    # print("\n21. 批量获取多个ETF价格...")
    # etf_symbols = ["510050", "510300", "159919", "159928", "512690"]  # 代表性ETF

    # for symbol in etf_symbols:
    #     try:
    #         price_result = await client.call_tool_mcp(
    #             "etf_tools_get_etf_price",
    #             {
    #                 "inputs": {
    #                     "symbol": symbol,
    #                     "market_type": "A"
    #                 }
    #             }
    #         )
    #         print(f"{symbol} 价格: {price_result.model_dump_json(indent=2)}")
    #     except Exception as e:
    #         print(f"获取 {symbol} 价格失败: {e}")

    # # 22. 批量技术分析
    # print("\n22. 批量ETF技术分析...")
    # analysis_symbols = ["510050", "510300"]  # 选择两个主要ETF进行分析

    # for symbol in analysis_symbols:
    #     try:
    #         print(f"\n--- {symbol} 技术分析 ---")

    #         # SMA
    #         sma_result = await client.call_tool_mcp(
    #             "etf_tools_calculate_etf_sma",
    #             {
    #                 "inputs": {
    #                     "symbol": symbol,
    #                     "timeframe": "1d",
    #                     "period": 20,
    #                     "history_len": 5
    #                 },
    #                 "market_type": "A"
    #             }
    #         )
    #         print(f"{symbol} SMA: {sma_result.model_dump_json(indent=2)}")

    #         # RSI
    #         rsi_result = await client.call_tool_mcp(
    #             "etf_tools_calculate_etf_rsi",
    #             {
    #                 "inputs": {
    #                     "symbol": symbol,
    #                     "timeframe": "1d",
    #                     "period": 14,
    #                     "history_len": 5
    #                 },
    #                 "market_type": "A"
    #             }
    #         )
    #         print(f"{symbol} RSI: {rsi_result.model_dump_json(indent=2)}")

    #     except Exception as e:
    #         print(f"分析 {symbol} 失败: {e}")

    # # ==================== 错误处理测试 ====================
    # print("\n--- 错误处理测试 ---")

    # 23. 测试无效ETF代码
    # print("\n23. 测试无效ETF代码...")
    # try:
    #     invalid_etf = await client.call_tool_mcp(
    #         "etf_tools_get_etf_price",
    #         {
    #             "inputs": {
    #                 "symbol": "999999",  # 无效代码
    #                 "market_type": "A"
    #             }
    #         }
    #     )
    #     print("无效ETF测试结果:", invalid_etf.model_dump_json(indent=2))
    # except Exception as e:
    #     print(f"无效ETF代码测试 - 预期错误: {e}")

    # 24. 测试不支持的市场类型
    # print("\n24. 测试不支持的市场组合...")
    # try:
    #     # 美股ETF的分时数据 (应该只支持A股)
    #     us_intraday = await client.call_tool_mcp(
    #         "etf_tools_get_etf_intraday_data",
    #         {
    #             "symbol": "510300",
    #             "period": "1",
    #             "market_type": "A"
    #         }
    #     )
    #     print("美股ETF分时数据测试:", us_intraday.model_dump_json(indent=2))
    # except Exception as e:
    #     print(f"不支持的市场组合测试 - 预期错误: {e}")

    print("\n=== ETF工具集测试完成 ===")


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
    client = Client(transport="http://localhost:8000/sse")
    async with client:
        try:
            # 选择要测试的模块
            test_crypto = False  # 测试加密货币工具
            test_a_stock = False  # 测试A股工具
            test_hk_stock = True  # 测试港股工具
            test_us_stock = False  # 测试美股工具
            test_etf_stock = False  # 测试ETF工具
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

            if test_etf_stock:
                await test_etf_tools(client)

        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
