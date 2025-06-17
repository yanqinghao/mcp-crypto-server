import numpy as np
import talib
import json
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta, time

from config import settings
from models.analysis import (
    SmaInput,
    SmaOutput,
    RsiInput,
    RsiOutput,
    MacdInput,
    MacdOutput,
    BbandsInput,
    BbandsOutput,
    AtrInput,
    AtrOutput,
    ComprehensiveAnalysisInput,
    ComprehensiveAnalysisOutput,
)
from models.market_data import (
    CandlesInput,
    CandlesOutput,
    PriceInput,
    PriceOutput,
    TickerInput,
    TickerOutput,
    OHLCVCandle,
)
from fastmcp import FastMCP, Context

# 导入现有的AKShare服务函数
from services.akshare_service import (
    fetch_stock_hist_data,
    fetch_stock_realtime_data,
    fetch_hk_stock_data,
    search_stock_by_name,
)

mcp = FastMCP()

# ==================== 数据获取辅助函数 ====================


async def _fetch_single_series_data(
    ctx: Context,
    symbol: str,
    period: str,
    required_candles: int,
    series_type: str = "close",
    market_type: str = "a_stock",
) -> Optional[np.ndarray]:
    """
    获取单个数据序列

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        period: 周期
        required_candles: 需要的K线数量
        series_type: 数据类型 (open, high, low, close, volume)
        market_type: 市场类型 (a_stock, hk_stock)

    Returns:
        numpy数组或None
    """
    try:
        # 计算需要的天数，加上缓冲区
        buffer_days = max(20, required_candles // 4)
        total_days = required_candles + buffer_days

        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=total_days + 50)).strftime(
            "%Y%m%d"
        )

        # 根据市场类型获取数据
        if market_type == "hk_stock":
            stock_data = await fetch_hk_stock_data(
                ctx, symbol, period, start_date, end_date
            )
        else:  # a_stock
            stock_data = await fetch_stock_hist_data(
                ctx, symbol, period, start_date, end_date
            )

        if not stock_data or len(stock_data) < required_candles:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(stock_data) if stock_data else 0}, need {required_candles}"
            )
            return None

        # 提取指定类型的数据
        if series_type not in ["open", "high", "low", "close", "volume"]:
            await ctx.error(f"Invalid series type: {series_type}")
            return None

        data_series = np.array([item[series_type] for item in stock_data])

        # 确保数据质量
        if np.any(np.isnan(data_series)) or np.any(np.isinf(data_series)):
            await ctx.warning(
                f"Found NaN or Inf values in {series_type} data for {symbol}"
            )
            # 移除NaN和Inf值
            data_series = data_series[~(np.isnan(data_series) | np.isinf(data_series))]

        return data_series

    except Exception as e:
        await ctx.error(f"Error fetching {series_type} data for {symbol}: {e}")
        return None


async def _fetch_multi_series_data(
    ctx: Context,
    symbol: str,
    period: str,
    required_candles: int,
    series_types: List[str],
    market_type: str = "a_stock",
) -> Optional[Dict[str, np.ndarray]]:
    """
    获取多个数据序列

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        period: 周期
        required_candles: 需要的K线数量
        series_types: 需要的数据类型列表
        market_type: 市场类型

    Returns:
        包含各数据序列的字典或None
    """
    try:
        # 计算需要的天数
        buffer_days = max(20, required_candles // 4)
        total_days = required_candles + buffer_days

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=total_days + 50)).strftime(
            "%Y%m%d"
        )

        # 根据市场类型获取数据
        if market_type == "hk_stock":
            stock_data = await fetch_hk_stock_data(ctx, symbol, start_date, end_date)
        else:  # a_stock
            stock_data = await fetch_stock_hist_data(
                ctx, symbol, period, start_date, end_date
            )

        if not stock_data or len(stock_data) < required_candles:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(stock_data) if stock_data else 0}, need {required_candles}"
            )
            return None

        result = {}
        for series_type in series_types:
            if series_type not in ["open", "high", "low", "close", "volume"]:
                await ctx.error(f"Invalid series type: {series_type}")
                return None

            data_series = np.array([item[series_type] for item in stock_data])

            # 确保数据质量
            if np.any(np.isnan(data_series)) or np.any(np.isinf(data_series)):
                await ctx.warning(
                    f"Found NaN or Inf values in {series_type} data for {symbol}"
                )
                data_series = data_series[
                    ~(np.isnan(data_series) | np.isinf(data_series))
                ]

            result[series_type] = data_series

        # 确保所有序列长度一致
        min_length = min(len(series) for series in result.values())
        if min_length < required_candles:
            await ctx.error(
                f"Insufficient clean data after filtering: {min_length} < {required_candles}"
            )
            return None

        # 截取到相同长度
        for key in result:
            result[key] = result[key][-min_length:]

        return result

    except Exception as e:
        await ctx.error(f"Error fetching multi-series data for {symbol}: {e}")
        return None


def _extract_valid_values(values: np.ndarray, history_len: int) -> List[float]:
    """
    从计算结果中提取有效值

    Args:
        values: TA-Lib计算结果
        history_len: 需要的历史数据长度

    Returns:
        有效值列表
    """
    # 移除NaN值
    valid_values = values[~np.isnan(values)]

    # 返回最后history_len个值
    if len(valid_values) >= history_len:
        return [float(x) for x in valid_values[-history_len:]]
    else:
        return [float(x) for x in valid_values]


# ==================== A股数据获取工具 ====================


@mcp.tool()
async def get_a_stock_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    获取A股K线数据

    该工具从AKShare获取A股（沪深交易所）的历史K线数据，支持多种时间周期和复权方式。
    适用于技术分析、量化交易策略开发和市场研究。

    功能特点:
    - 支持日线、周线、月线数据
    - 支持前复权、后复权和不复权
    - 自动处理数据质量问题
    - 返回标准OHLCV格式

    Args:
        inputs.symbol (str): A股代码，6位数字格式 (如: '000001'平安银行, '600519'贵州茅台)
        inputs.timeframe (TimeFrame): 时间框架，支持 '1d'(日线), '1w'(周线), '1M'(月线)
        inputs.limit (int): 返回的K线数量，范围1-1000，默认24
        inputs.adjust (str): 复权类型 - 'qfq'前复权(默认), 'hfq'后复权, ''不复权

    Returns:
        CandlesOutput: 包含K线数据的输出对象
        - symbol: 股票代码
        - timeframe: 时间框架
        - candles: OHLCVCandle对象列表，按时间顺序排列
        - count: 实际返回的K线数量
        - error: 错误信息(如果有)

    Example:
        获取平安银行最近30天的前复权日K线:
        >>> inputs = CandlesInput(
        ...     symbol="000001",
        ...     timeframe=TimeFrame.ONE_DAY,
        ...     limit=30,
        ...     adjust="qfq"
        ... )
        >>> result = await get_a_stock_candles(ctx, inputs)
        >>> print(f"获取到{result.count}根K线")

    Note:
        - 实际返回的数据量可能少于请求量，取决于交易日历和数据可用性
        - 前复权数据已调整历史价格以反映分红配股影响
        - 数据来源可能有15-20分钟延迟
    """
    await ctx.info(
        f"Fetching A-stock candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")

        # 获取复权类型
        adjust = getattr(inputs, "adjust", "qfq") or "qfq"

        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=inputs.limit + 50)).strftime(
            "%Y%m%d"
        )

        # 使用现有的AKShare服务获取A股数据
        stock_data = await fetch_stock_hist_data(
            ctx=ctx,
            symbol=inputs.symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if stock_data:
            candles = []
            for item in stock_data[-inputs.limit :]:  # 只取需要的数量
                # 转换为OHLCVCandle格式
                date_str = item["date"]
                if isinstance(date_str, str):
                    timestamp = int(
                        datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000
                    )
                else:
                    timestamp = int(
                        datetime.combine(date_str, time()).timestamp() * 1000
                    )

                candle = OHLCVCandle(
                    timestamp=timestamp,
                    open=item["open"],
                    high=item["high"],
                    low=item["low"],
                    close=item["close"],
                    volume=item["volume"],
                )
                candles.append(candle)

            await ctx.info(
                f"Successfully fetched {len(candles)} A-stock candles for {inputs.symbol}"
            )

            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe.value,
                candles=candles,
                count=len(candles),
            )
        else:
            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe.value,
                error=f"No A-stock candle data available for {inputs.symbol}",
            )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(f"Error fetching A-stock candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"A-stock data error: {str(e)}",
        )


@mcp.tool()
async def get_a_stock_price(ctx: Context, inputs: PriceInput) -> PriceOutput:
    """
    获取A股当前价格

    该工具获取A股的实时或最新价格信息，优先使用实时数据API，
    如果不可用则回退到最新的历史数据。

    功能特点:
    - 实时价格获取(可能有15-20分钟延迟)
    - 自动回退到历史数据
    - 价格数据验证
    - 统一的错误处理

    Args:
        inputs.symbol (str): A股代码，6位数字格式

    Returns:
        PriceOutput: 包含价格信息的输出对象
        - symbol: 股票代码
        - price: 当前价格(人民币)
        - timestamp: 价格时间戳(毫秒)
        - error: 错误信息(如果有)

    Example:
        >>> inputs = PriceInput(symbol="000001")
        >>> result = await get_a_stock_price(ctx, inputs)
        >>> if result.price:
        ...     print(f"平安银行当前价格: ¥{result.price:.2f}")
    """
    await ctx.info(f"Fetching A-stock current price for {inputs.symbol}")

    try:
        # 使用实时数据API
        realtime_data = await fetch_stock_realtime_data(ctx, [inputs.symbol])

        if realtime_data and len(realtime_data) > 0:
            stock_info = realtime_data[0]  # 取第一个结果
            timestamp = int(datetime.now().timestamp() * 1000)

            await ctx.info(
                f"A-stock current price for {inputs.symbol}: ¥{stock_info['price']}"
            )
            return PriceOutput(
                symbol=inputs.symbol,
                price=stock_info["price"],
                timestamp=timestamp,
            )
        else:
            # 如果实时数据不可用，尝试获取最新的历史数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

            stock_data = await fetch_stock_hist_data(
                ctx, inputs.symbol, "daily", start_date, end_date
            )

            if stock_data and len(stock_data) > 0:
                latest = stock_data[-1]
                timestamp = int(
                    datetime.strptime(latest["date"], "%Y-%m-%d").timestamp() * 1000
                )

                return PriceOutput(
                    symbol=inputs.symbol,
                    price=latest["close"],
                    timestamp=timestamp,
                )

            return PriceOutput(
                symbol=inputs.symbol, error="A-stock price data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching A-stock price for {inputs.symbol}: {e}")
        return PriceOutput(symbol=inputs.symbol, error=f"A-stock price error: {str(e)}")


@mcp.tool()
async def search_a_stock_symbols(ctx: Context, query: str, limit: int = 10) -> dict:
    """
    搜索A股股票代码

    该工具通过公司名称、股票简称或代码搜索A股市场的股票，
    支持模糊匹配和中文搜索。

    功能特点:
    - 支持中文公司名称搜索
    - 支持股票代码精确匹配
    - 支持拼音缩写搜索
    - 返回详细的股票信息

    Args:
        query (str): 搜索关键词，可以是:
            - 公司名称: "平安银行", "贵州茅台"
            - 股票代码: "000001", "600519"
            - 简称: "平安", "茅台"
        limit (int): 返回结果的最大数量，默认10

    Returns:
        Dict包含搜索结果:
        - success (bool): 搜索是否成功
        - query (str): 搜索关键词
        - market_type (str): 市场类型 "a_stock"
        - results (List[Dict]): 匹配的股票列表，每个包含:
            - code: 股票代码
            - name: 股票名称
            - market: 交易所(SZ/SH)
        - count (int): 结果数量
        - error (str): 错误信息(如果有)

    Example:
        搜索平安相关股票:
        >>> result = await search_a_stock_symbols(ctx, "平安", 5)
        >>> if result["success"]:
        ...     for stock in result["results"]:
        ...         print(f"{stock['code']}: {stock['name']}")
    """
    await ctx.info(f"Searching A-stock symbols for '{query}'")

    try:
        results = await search_stock_by_name(ctx, query)

        if results:
            # 限制结果数量
            limited_results = results[:limit]
            return {
                "success": True,
                "query": query,
                "market_type": "a_stock",
                "results": limited_results,
                "count": len(limited_results),
            }
        else:
            return {
                "success": False,
                "query": query,
                "market_type": "a_stock",
                "error": "No matching stocks found",
            }

    except Exception as e:
        await ctx.error(f"Error searching A-stock symbols: {e}")
        return {
            "success": False,
            "query": query,
            "market_type": "a_stock",
            "error": str(e),
        }


@mcp.tool()
async def get_a_stock_ticker(ctx: Context, inputs: TickerInput) -> TickerOutput:
    """
    获取A股详细行情数据

    该工具获取A股的详细实时行情信息，包括开高低收、成交量、
    涨跌幅等关键市场数据。

    Args:
        inputs.symbol (str): A股代码

    Returns:
        TickerOutput: 详细行情数据
        - symbol: 股票代码
        - last: 最新价
        - open/high/low/close: 开高低收价
        - volume: 成交量
        - change: 价格变化
        - percentage: 涨跌幅百分比
        - timestamp: 数据时间戳
        - error: 错误信息(如果有)
    """
    await ctx.info(f"Fetching A-stock ticker for {inputs.symbol}")

    try:
        # 获取实时数据
        realtime_data = await fetch_stock_realtime_data(ctx, [inputs.symbol])

        if realtime_data and len(realtime_data) > 0:
            stock_info = realtime_data[0]
            timestamp = int(datetime.now().timestamp() * 1000)

            await ctx.info(f"Successfully fetched A-stock ticker for {inputs.symbol}")
            return TickerOutput(
                symbol=inputs.symbol,
                last=stock_info["price"],
                open=stock_info["open"],
                high=stock_info["high"],
                low=stock_info["low"],
                close=stock_info["price"],
                volume=stock_info["volume"],
                change=stock_info["change"],
                percentage=stock_info["pct_change"],
                timestamp=timestamp,
            )
        else:
            return TickerOutput(
                symbol=inputs.symbol, error="A-stock ticker data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching A-stock ticker for {inputs.symbol}: {e}")
        return TickerOutput(
            symbol=inputs.symbol, error=f"A-stock ticker error: {str(e)}"
        )


# ==================== 港股数据获取工具 ====================


@mcp.tool()
async def get_hk_stock_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    获取港股K线数据

    该工具从数据源获取香港交易所股票的历史K线数据。
    港股市场交易时间与A股不同，数据格式经过标准化处理。

    Args:
        inputs.symbol (str): 港股代码，5位数字格式 (如: '00700'腾讯, '09988'阿里巴巴)
        inputs.timeframe (TimeFrame): 时间框架
        inputs.limit (int): K线数量

    Returns:
        CandlesOutput: 港股K线数据，价格单位为港币

    Note:
        - 港股代码通常为5位数字，前面补0
        - 价格单位为港币(HK$)
        - 交易时间: 9:30-12:00, 13:00-16:00 (港时)
    """
    await ctx.info(
        f"Fetching HK stock candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")

        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=inputs.limit + 50)).strftime(
            "%Y%m%d"
        )

        # 使用现有的港股服务获取数据
        stock_data = await fetch_hk_stock_data(
            ctx=ctx,
            period=period,
            symbol=inputs.symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if stock_data:
            candles = []
            for item in stock_data[-inputs.limit :]:  # 只取需要的数量
                # 转换为OHLCVCandle格式
                date_str = item["date"]
                if isinstance(date_str, str):
                    timestamp = int(
                        datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000
                    )
                else:
                    timestamp = int(date_str.timestamp() * 1000)

                candle = OHLCVCandle(
                    timestamp=timestamp,
                    open=item["open"],
                    high=item["high"],
                    low=item["low"],
                    close=item["close"],
                    volume=item["volume"],
                )
                candles.append(candle)

            await ctx.info(
                f"Successfully fetched {len(candles)} HK stock candles for {inputs.symbol}"
            )

            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe.value,
                candles=candles,
                count=len(candles),
            )
        else:
            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe.value,
                error=f"No HK stock candle data available for {inputs.symbol}",
            )

    except Exception as e:
        await ctx.error(f"Error fetching HK stock candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"HK stock data error: {str(e)}",
        )


@mcp.tool()
async def get_hk_stock_price(ctx: Context, inputs: PriceInput) -> PriceOutput:
    """
    获取港股当前价格

    获取香港交易所股票的最新价格信息。

    Args:
        inputs.symbol (str): 港股代码

    Returns:
        PriceOutput: 港股价格数据，价格单位为港币
    """
    await ctx.info(f"Fetching HK stock current price for {inputs.symbol}")

    try:
        # 获取最新的历史数据作为当前价格
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

        stock_data = await fetch_hk_stock_data(ctx, inputs.symbol, start_date, end_date)

        if stock_data and len(stock_data) > 0:
            latest = stock_data[-1]
            timestamp = int(
                datetime.strptime(latest["date"], "%Y-%m-%d").timestamp() * 1000
            )

            await ctx.info(
                f"HK stock current price for {inputs.symbol}: HK${latest['close']}"
            )
            return PriceOutput(
                symbol=inputs.symbol,
                price=latest["close"],
                timestamp=timestamp,
            )
        else:
            return PriceOutput(
                symbol=inputs.symbol, error="HK stock price data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching HK stock price for {inputs.symbol}: {e}")
        return PriceOutput(
            symbol=inputs.symbol, error=f"HK stock price error: {str(e)}"
        )


# ==================== A股技术分析工具 ====================


@mcp.tool()
async def calculate_a_stock_sma(ctx: Context, inputs: SmaInput) -> SmaOutput:
    """
    计算A股简单移动平均线 (Simple Moving Average)

    SMA是最基础的技术指标，通过计算指定期间的平均价格来平滑价格波动，
    帮助识别价格趋势。常用于判断支撑阻力位和趋势方向。

    计算公式:
    SMA(n) = (P1 + P2 + ... + Pn) / n

    其中P为收盘价，n为周期数

    Args:
        inputs.symbol (str): A股代码
        inputs.period (int): 计算周期，常用值: 5, 10, 20, 50, 200
        inputs.history_len (int): 返回的历史数据长度
        inputs.timeframe (str): 时间框架

    Returns:
        SmaOutput: SMA计算结果
        - symbol: 股票代码
        - timeframe: 时间框架
        - period: 计算周期
        - sma: SMA值列表，按时间顺序排列
        - error: 错误信息(如果有)

    应用场景:
    - 趋势判断: 价格在SMA上方表示上升趋势
    - 支撑阻力: SMA线常作为动态支撑或阻力位
    - 交易信号: 价格穿越SMA产生买卖信号

    Example:
        计算平安银行20日均线:
        >>> inputs = SmaInput(symbol="000001", period=20, history_len=10)
        >>> result = await calculate_a_stock_sma(ctx, inputs)
        >>> if result.sma:
        ...     print(f"最新20日均线: ¥{result.sma[-1]:.2f}")
    """
    await ctx.info(
        f"Calculating A-stock SMA for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        # SMA需要period + history_len - 1个数据点
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "a_stock"
        )
        if close_prices is None or len(close_prices) < required_candles:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient A-stock data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)
        valid_sma = _extract_valid_values(sma_values, inputs.history_len)

        if not valid_sma:
            return SmaOutput(
                **output_base,
                error="A-stock SMA calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock SMA for {inputs.symbol}: {len(valid_sma)} values, latest: ¥{valid_sma[-1]:.2f}"
        )
        return SmaOutput(**output_base, sma=valid_sma)

    except Exception as e:
        await ctx.error(f"Error in A-stock SMA calculation for {inputs.symbol}: {e}")
        return SmaOutput(**output_base, error="A-stock SMA calculation error.")


@mcp.tool()
async def calculate_a_stock_rsi(ctx: Context, inputs: RsiInput) -> RsiOutput:
    """
    计算A股相对强弱指数 (Relative Strength Index)

    RSI是威尔斯·威尔德开发的动量震荡指标，用于衡量价格变动的速度和变化。
    RSI在0-100之间波动，帮助识别超买超卖状态。

    计算逻辑:
    1. 计算每日价格变化
    2. 分别计算上涨和下跌的平均值
    3. RS = 平均上涨 / 平均下跌
    4. RSI = 100 - (100 / (1 + RS))

    Args:
        inputs.symbol (str): A股代码
        inputs.period (int): 计算周期，默认14，常用值: 6, 9, 14, 21
        inputs.history_len (int): 返回的历史数据长度
        inputs.timeframe (str): 时间框架

    Returns:
        RsiOutput: RSI计算结果
        - rsi: RSI值列表 (0-100)

    交易信号:
    - RSI > 70: 超买状态，可能的卖出信号
    - RSI < 30: 超卖状态，可能的买入信号
    - RSI 50: 中性水平，上下穿越表示趋势变化

    Example:
        >>> inputs = RsiInput(symbol="000001", period=14, history_len=5)
        >>> result = await calculate_a_stock_rsi(ctx, inputs)
        >>> latest_rsi = result.rsi[-1]
        >>> if latest_rsi > 70:
        ...     print("超买状态")
        >>> elif latest_rsi < 30:
        ...     print("超卖状态")
    """
    await ctx.info(
        f"Calculating A-stock RSI for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        # RSI需要period + history_len个数据点
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "a_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient A-stock data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)
        valid_rsi = _extract_valid_values(rsi_values, inputs.history_len)

        if not valid_rsi:
            return RsiOutput(
                **output_base,
                error="A-stock RSI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock RSI for {inputs.symbol}: {len(valid_rsi)} values, latest: {valid_rsi[-1]:.2f}"
        )
        return RsiOutput(**output_base, rsi=valid_rsi)

    except Exception as e:
        await ctx.error(f"Error in A-stock RSI calculation for {inputs.symbol}: {e}")
        return RsiOutput(**output_base, error="A-stock RSI calculation error.")


@mcp.tool()
async def calculate_a_stock_macd(ctx: Context, inputs: MacdInput) -> MacdOutput:
    """
    计算A股MACD指标 (Moving Average Convergence Divergence)

    MACD由Gerald Appel开发，是趋势跟踪动量指标。通过比较两个不同周期的
    指数移动平均线来识别趋势变化和动量。

    组成部分:
    1. MACD线: 快线EMA - 慢线EMA
    2. 信号线: MACD线的EMA
    3. 柱状图: MACD线 - 信号线

    Args:
        inputs.fast_period (int): 快线周期，默认12
        inputs.slow_period (int): 慢线周期，默认26
        inputs.signal_period (int): 信号线周期，默认9
        inputs.symbol (str): A股代码
        inputs.history_len (int): 返回的历史数据长度
        inputs.timeframe (str): 时间框架

    Returns:
        MacdOutput: MACD计算结果
        - macd: MACD线值列表
        - signal: 信号线值列表
        - histogram: 柱状图值列表

    交易信号:
    - MACD上穿信号线: 买入信号
    - MACD下穿信号线: 卖出信号
    - 柱状图由负转正: 上升动量
    - 柱状图由正转负: 下降动量
    """
    await ctx.info(
        f"Calculating A-stock MACD for {inputs.symbol}, Periods: {inputs.fast_period}/{inputs.slow_period}/{inputs.signal_period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "fast_period": inputs.fast_period,
        "slow_period": inputs.slow_period,
        "signal_period": inputs.signal_period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = (
            inputs.slow_period + inputs.signal_period + inputs.history_len + 10
        )
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "a_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return MacdOutput(
                **output_base, error="Failed to fetch sufficient A-stock data for MACD."
            )

        macd, signal, hist = talib.MACD(
            close_prices,
            fastperiod=inputs.fast_period,
            slowperiod=inputs.slow_period,
            signalperiod=inputs.signal_period,
        )

        valid_macd = _extract_valid_values(macd, inputs.history_len)
        valid_signal = _extract_valid_values(signal, inputs.history_len)
        valid_hist = _extract_valid_values(hist, inputs.history_len)

        if not valid_macd or not valid_signal or not valid_hist:
            return MacdOutput(
                **output_base,
                error="A-stock MACD calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock MACD for {inputs.symbol}: latest MACD: {valid_macd[-1]:.4f}"
        )
        return MacdOutput(
            **output_base, macd=valid_macd, signal=valid_signal, histogram=valid_hist
        )

    except Exception as e:
        await ctx.error(f"Error in A-stock MACD calculation for {inputs.symbol}: {e}")
        return MacdOutput(**output_base, error="A-stock MACD calculation error.")


@mcp.tool()
async def calculate_a_stock_bbands(ctx: Context, inputs: BbandsInput) -> BbandsOutput:
    """
    计算A股布林带 (Bollinger Bands)

    布林带由约翰·博林格开发，是基于统计学的技术指标。
    由中轨(移动平均线)和上下轨(标准差)组成，用于判断价格的相对高低。

    构成:
    - 中轨: n期简单移动平均线
    - 上轨: 中轨 + k × n期标准差
    - 下轨: 中轨 - k × n期标准差

    其中k通常为2，n通常为20

    Args:
        inputs.period (int): 移动平均周期，默认20
        inputs.nbdevup (float): 上轨标准差倍数，默认2.0
        inputs.nbdevdn (float): 下轨标准差倍数，默认2.0
        inputs.symbol (str): A股代码
        inputs.history_len (int): 返回的历史数据长度
        inputs.timeframe (str): 时间框架

    Returns:
        BbandsOutput: 布林带计算结果
        - upper_band: 上轨值列表
        - middle_band: 中轨值列表
        - lower_band: 下轨值列表

    应用:
    - 价格接近上轨: 可能超买
    - 价格接近下轨: 可能超卖
    - 带宽收窄: 波动率降低，可能突破
    - 带宽扩张: 波动率增加，趋势加强
    """
    await ctx.info(
        f"Calculating A-stock Bollinger Bands for {inputs.symbol}, Period: {inputs.period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
        "nbdevup": inputs.nbdevup,
        "nbdevdn": inputs.nbdevdn,
        "matype": inputs.matype,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "a_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return BbandsOutput(
                **output_base,
                error="Failed to fetch sufficient A-stock data for Bollinger Bands.",
            )

        upper, middle, lower = talib.BBANDS(
            close_prices,
            timeperiod=inputs.period,
            nbdevup=inputs.nbdevup,
            nbdevdn=inputs.nbdevdn,
            matype=inputs.matype,
        )

        valid_upper = _extract_valid_values(upper, inputs.history_len)
        valid_middle = _extract_valid_values(middle, inputs.history_len)
        valid_lower = _extract_valid_values(lower, inputs.history_len)

        if not valid_upper or not valid_middle or not valid_lower:
            return BbandsOutput(
                **output_base,
                error="A-stock Bollinger Bands calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock Bollinger Bands for {inputs.symbol}: latest upper: ¥{valid_upper[-1]:.2f}"
        )
        return BbandsOutput(
            **output_base,
            upper_band=valid_upper,
            middle_band=valid_middle,
            lower_band=valid_lower,
        )

    except Exception as e:
        await ctx.error(
            f"Error in A-stock Bollinger Bands calculation for {inputs.symbol}: {e}",
        )
        return BbandsOutput(
            **output_base, error="A-stock Bollinger Bands calculation error."
        )


@mcp.tool()
async def calculate_a_stock_atr(ctx: Context, inputs: AtrInput) -> AtrOutput:
    """
    计算A股平均真实波幅 (Average True Range)

    ATR由威尔斯·威尔德开发，用于衡量市场波动性。
    真实波幅是以下三者中的最大值：
    1. 当日最高价 - 当日最低价
    2. |当日最高价 - 前日收盘价|
    3. |当日最低价 - 前日收盘价|

    ATR是真实波幅的移动平均值。

    Args:
        inputs.period (int): 计算周期，默认14
        inputs.symbol (str): A股代码
        inputs.history_len (int): 返回的历史数据长度
        inputs.timeframe (str): 时间框架

    Returns:
        AtrOutput: ATR计算结果
        - atr: ATR值列表

    应用:
    - 止损设置: ATR × 倍数作为止损距离
    - 仓位管理: 根据ATR调整仓位大小
    - 突破确认: 价格变动 > ATR表示有效突破
    - 市场状态: ATR增大表示波动加剧
    """
    await ctx.info(
        f"Calculating A-stock ATR for {inputs.symbol}, Period: {inputs.period}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len - 1

        price_data = await _fetch_multi_series_data(
            ctx,
            inputs.symbol,
            timeframe,
            required_candles,
            ["high", "low", "close"],
            "a_stock",
        )

        if not price_data:
            return AtrOutput(**output_base, error="Failed to fetch HLC data for ATR.")

        high_prices = price_data["high"]
        low_prices = price_data["low"]
        close_prices = price_data["close"]

        if len(high_prices) < required_candles:
            return AtrOutput(
                **output_base,
                error=f"Insufficient HLC data points for ATR. Need at least {required_candles}.",
            )

        atr_values = talib.ATR(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )

        valid_atr = _extract_valid_values(atr_values, inputs.history_len)

        if not valid_atr:
            return AtrOutput(
                **output_base,
                error="A-stock ATR calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated A-stock ATR for {inputs.symbol}: {len(valid_atr)} values, latest: ¥{valid_atr[-1]:.2f}"
        )
        return AtrOutput(**output_base, atr=valid_atr)

    except Exception as e:
        await ctx.error(f"Error in A-stock ATR calculation for {inputs.symbol}: {e}")
        return AtrOutput(**output_base, error="A-stock ATR calculation error.")


# ==================== 港股技术分析工具 ====================


@mcp.tool()
async def calculate_hk_stock_sma(ctx: Context, inputs: SmaInput) -> SmaOutput:
    """
    计算港股简单移动平均线 (SMA)

    功能与A股SMA相同，但数据来源为香港交易所，价格单位为港币。

    Args:
        inputs.symbol (str): 港股代码
        inputs.period (int): 计算周期
        inputs.history_len (int): 历史数据长度
        inputs.timeframe (str): 时间框架

    Returns:
        SmaOutput: 港股SMA结果，价格单位为港币
    """
    await ctx.info(
        f"Calculating HK stock SMA for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "hk_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient HK stock data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)
        valid_sma = _extract_valid_values(sma_values, inputs.history_len)

        if not valid_sma:
            return SmaOutput(
                **output_base,
                error="HK stock SMA calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated HK stock SMA for {inputs.symbol}: {len(valid_sma)} values, latest: HK${valid_sma[-1]:.2f}"
        )
        return SmaOutput(**output_base, sma=valid_sma)

    except Exception as e:
        await ctx.error(f"Error in HK stock SMA calculation for {inputs.symbol}: {e}")
        return SmaOutput(**output_base, error="HK stock SMA calculation error.")


@mcp.tool()
async def calculate_hk_stock_rsi(ctx: Context, inputs: RsiInput) -> RsiOutput:
    """
    计算港股相对强弱指数 (RSI)

    功能与A股RSI相同，适用于港股市场分析。

    Args:
        inputs.symbol (str): 港股代码
        inputs.period (int): 计算周期
        inputs.history_len (int): 历史数据长度
        inputs.timeframe (str): 时间框架

    Returns:
        RsiOutput: 港股RSI结果
    """
    await ctx.info(
        f"Calculating HK stock RSI for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        timeframe = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, timeframe, required_candles, "close", "hk_stock"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient HK stock data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)
        valid_rsi = _extract_valid_values(rsi_values, inputs.history_len)

        if not valid_rsi:
            return RsiOutput(
                **output_base,
                error="HK stock RSI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated HK stock RSI for {inputs.symbol}: {len(valid_rsi)} values, latest: {valid_rsi[-1]:.2f}"
        )
        return RsiOutput(**output_base, rsi=valid_rsi)

    except Exception as e:
        await ctx.error(f"Error in HK stock RSI calculation for {inputs.symbol}: {e}")
        return RsiOutput(**output_base, error="HK stock RSI calculation error.")


# ==================== 综合分析工具 ====================


@mcp.tool()
async def generate_a_stock_comprehensive_report(
    ctx: Context, inputs: ComprehensiveAnalysisInput
) -> ComprehensiveAnalysisOutput:
    """
    生成A股综合技术分析报告

    该工具整合多个技术指标，生成全面的A股技术分析报告。
    报告包含数值结果、趋势分析和交易信号提示。

    功能特点:
    - 多指标综合分析
    - 智能趋势判断
    - 超买超卖提示
    - 结构化数据输出
    - 中文分析报告

    Args:
        inputs.symbol (str): A股代码
        inputs.timeframe (str): 时间框架，默认"1d"
        inputs.history_len (int): 历史数据长度，默认10
        inputs.indicators_to_include (List[str]): 要包含的指标列表，默认["SMA", "RSI", "MACD", "BBANDS"]
        inputs.sma_period (int): SMA周期，默认20
        inputs.rsi_period (int): RSI周期，默认14
        inputs.macd_fast_period (int): MACD快线周期，默认12
        inputs.macd_slow_period (int): MACD慢线周期，默认26
        inputs.macd_signal_period (int): MACD信号线周期，默认9
        inputs.bbands_period (int): 布林带周期，默认20

    Returns:
        ComprehensiveAnalysisOutput: 综合分析结果
        - symbol: 股票代码
        - timeframe: 时间框架
        - report_text: 中文分析报告
        - structured_data: 结构化指标数据
        - error: 错误信息(如果有)

    报告内容:
    - 各技术指标的最新值
    - 趋势方向指示
    - 超买超卖状态
    - 交易信号提示
    - 指标数据统计

    Example:
        >>> inputs = ComprehensiveAnalysisInput(
        ...     symbol="000001",
        ...     indicators_to_include=["SMA", "RSI", "MACD"],
        ...     history_len=5
        ... )
        >>> result = await generate_a_stock_comprehensive_report(ctx, inputs)
        >>> print(result.report_text)
    """
    await ctx.info(
        f"Generating A-stock comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
    )
    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # 确定要运行的指标
    default_indicators = ["SMA", "RSI", "MACD", "BBANDS"]  # A股常用指标
    indicators_to_run = (
        inputs.indicators_to_include
        if inputs.indicators_to_include is not None
        else default_indicators
    )

    try:
        # SMA分析
        if "SMA" in indicators_to_run:
            sma_period = inputs.sma_period or settings.DEFAULT_SMA_PERIOD
            sma_input = SmaInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=sma_period,
            )
            sma_output = await calculate_a_stock_sma.run(
                {"ctx": ctx, "inputs": sma_input}
            )
            indicator_results_structured["sma"] = json.loads(
                sma_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["sma"]["sma"] is not None
                and len(indicator_results_structured["sma"]["sma"]) > 0
            ):
                latest_sma = indicator_results_structured["sma"]["sma"][-1]
                report_sections.append(
                    f"- A股SMA({indicator_results_structured['sma']['period']}): ¥{latest_sma:.2f} (最新值)"
                )
                if len(indicator_results_structured["sma"]["sma"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["sma"]["sma"][-1]
                        > indicator_results_structured["sma"]["sma"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(indicator_results_structured['sma']['sma'])} 个数据点)"
                    )
            elif indicator_results_structured["sma"]["error"]:
                report_sections.append(
                    f"- A股SMA: 错误 - {indicator_results_structured['sma']['error']}"
                )

        # RSI分析
        if "RSI" in indicators_to_run:
            rsi_period = inputs.rsi_period or settings.DEFAULT_RSI_PERIOD
            rsi_input = RsiInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=rsi_period,
            )
            rsi_output = await calculate_a_stock_rsi.run(
                {"ctx": ctx, "inputs": rsi_input}
            )
            indicator_results_structured["rsi"] = json.loads(
                rsi_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["rsi"]["rsi"] is not None
                and len(indicator_results_structured["rsi"]["rsi"]) > 0
            ):
                latest_rsi = indicator_results_structured["rsi"]["rsi"][-1]
                report_sections.append(
                    f"- A股RSI({indicator_results_structured['rsi']['period']}): {latest_rsi:.2f}"
                )
                if latest_rsi > 70:
                    report_sections.append("  - 注意: RSI表明超买状态 (>70)")
                elif latest_rsi < 30:
                    report_sections.append("  - 注意: RSI表明超卖状态 (<30)")

                if len(indicator_results_structured["rsi"]["rsi"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["rsi"]["rsi"][-1]
                        > indicator_results_structured["rsi"]["rsi"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(indicator_results_structured['rsi']['rsi'])} 个数据点)"
                    )
            elif indicator_results_structured["rsi"]["error"]:
                report_sections.append(
                    f"- A股RSI: 错误 - {indicator_results_structured['rsi']['error']}"
                )

        # MACD分析
        if "MACD" in indicators_to_run:
            macd_input = MacdInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                fast_period=inputs.macd_fast_period or settings.DEFAULT_MACD_FAST,
                slow_period=inputs.macd_slow_period or settings.DEFAULT_MACD_SLOW,
                signal_period=inputs.macd_signal_period or settings.DEFAULT_MACD_SIGNAL,
            )
            macd_output = await calculate_a_stock_macd.run(
                {"ctx": ctx, "inputs": macd_input}
            )
            indicator_results_structured["macd"] = json.loads(
                macd_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["macd"]["macd"] is not None
                and len(indicator_results_structured["macd"]["macd"]) > 0
                and indicator_results_structured["macd"]["signal"] is not None
                and len(indicator_results_structured["macd"]["signal"]) > 0
                and indicator_results_structured["macd"]["histogram"] is not None
                and len(indicator_results_structured["macd"]["histogram"]) > 0
            ):
                latest_macd = indicator_results_structured["macd"]["macd"][-1]
                latest_signal = indicator_results_structured["macd"]["signal"][-1]
                latest_hist = indicator_results_structured["macd"]["histogram"][-1]

                report_sections.append(
                    f"- A股MACD({indicator_results_structured['macd']['fast_period']},{indicator_results_structured['macd']['slow_period']},{indicator_results_structured['macd']['signal_period']}): "
                    f"MACD: {latest_macd:.4f}, 信号线: {latest_signal:.4f}, 柱状图: {latest_hist:.4f}"
                )

                if latest_hist > 0 and latest_macd > latest_signal:
                    report_sections.append("  - 注意: MACD柱状图为正，可能有看涨动量")
                elif latest_hist < 0 and latest_macd < latest_signal:
                    report_sections.append("  - 注意: MACD柱状图为负，可能有看跌动量")

            elif indicator_results_structured["macd"]["error"]:
                report_sections.append(
                    f"- A股MACD: 错误 - {indicator_results_structured['macd']['error']}"
                )

        # Bollinger Bands分析
        if "BBANDS" in indicators_to_run:
            bbands_input = BbandsInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.bbands_period or settings.DEFAULT_BBANDS_PERIOD,
            )
            bbands_output = await calculate_a_stock_bbands.run(
                {"ctx": ctx, "inputs": bbands_input}
            )
            indicator_results_structured["bbands"] = json.loads(
                bbands_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["bbands"]["upper_band"] is not None
                and len(indicator_results_structured["bbands"]["upper_band"]) > 0
                and indicator_results_structured["bbands"]["middle_band"] is not None
                and len(indicator_results_structured["bbands"]["middle_band"]) > 0
                and indicator_results_structured["bbands"]["lower_band"] is not None
                and len(indicator_results_structured["bbands"]["lower_band"]) > 0
            ):
                latest_upper = indicator_results_structured["bbands"]["upper_band"][-1]
                latest_middle = indicator_results_structured["bbands"]["middle_band"][
                    -1
                ]
                latest_lower = indicator_results_structured["bbands"]["lower_band"][-1]

                report_sections.append(
                    f"- A股布林带({indicator_results_structured['bbands']['period']}): "
                    f"上轨: ¥{latest_upper:.2f}, 中轨: ¥{latest_middle:.2f}, 下轨: ¥{latest_lower:.2f}"
                )
                report_sections.append(f"  - 带宽: ¥{latest_upper - latest_lower:.2f}")

            elif indicator_results_structured["bbands"]["error"]:
                report_sections.append(
                    f"- A股布林带: 错误 - {indicator_results_structured['bbands']['error']}"
                )

        # 合成报告
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base,
                error="无法计算任何指标数据",
            )

        report_title = f"A股技术分析报告 - {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} 个数据点:\n"
        report_text = report_title + "\n".join(report_sections)

        # 添加趋势总结
        summary_sections = []
        trend_indicators = []

        for indicator_name, indicator_data in indicator_results_structured.items():
            if isinstance(indicator_data, dict) and not indicator_data.get("error"):
                if indicator_name == "sma" and indicator_data.get("sma"):
                    sma_data = indicator_data["sma"]
                    if len(sma_data) > 1:
                        trend_indicators.append(
                            f"SMA: {'↗' if sma_data[-1] > sma_data[-2] else '↘'}"
                        )

                elif indicator_name == "rsi" and indicator_data.get("rsi"):
                    rsi_data = indicator_data["rsi"]
                    if len(rsi_data) > 1:
                        trend_indicators.append(
                            f"RSI: {'↗' if rsi_data[-1] > rsi_data[-2] else '↘'}"
                        )

        if trend_indicators:
            summary_sections.append("\n趋势总结:")
            summary_sections.extend([f"  {trend}" for trend in trend_indicators])
            report_text += "\n" + "\n".join(summary_sections)

        await ctx.info(
            f"Successfully generated A-stock comprehensive report for {inputs.symbol}"
        )
        return ComprehensiveAnalysisOutput(
            **output_base,
            report_text=report_text,
            structured_data=indicator_results_structured,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(
            f"Error in A-stock comprehensive report for {inputs.symbol}: {e}",
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"A股综合分析报告生成错误: {str(e)}"
        )


@mcp.tool()
async def generate_hk_stock_comprehensive_report(
    ctx: Context, inputs: ComprehensiveAnalysisInput
) -> ComprehensiveAnalysisOutput:
    """
    生成港股综合技术分析报告

    专门针对香港交易所股票的综合技术分析，考虑港股市场特点。

    功能特点:
    - 适配港股交易时间和特点
    - 港币计价显示
    - 港股常用技术指标
    - 中英文股票名称支持

    Args:
        与A股综合分析相同，但默认指标为["SMA", "RSI"]

    Returns:
        ComprehensiveAnalysisOutput: 港股综合分析结果

    Note:
        - 价格单位为港币(HK$)
        - 考虑港股市场流动性特点
        - 适用于港股通和直接港股投资
    """
    await ctx.info(
        f"Generating HK stock comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
    )
    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # 确定要运行的指标
    default_indicators = ["SMA", "RSI"]  # 港股常用指标
    indicators_to_run = (
        inputs.indicators_to_include
        if inputs.indicators_to_include is not None
        else default_indicators
    )

    try:
        # SMA分析
        if "SMA" in indicators_to_run:
            sma_period = inputs.sma_period or settings.DEFAULT_SMA_PERIOD
            sma_input = SmaInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=sma_period,
            )
            sma_output = await calculate_hk_stock_sma.run(
                {"ctx": ctx, "inputs": sma_input}
            )
            indicator_results_structured["sma"] = json.loads(
                sma_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["sma"]["sma"] is not None
                and len(indicator_results_structured["sma"]["sma"]) > 0
            ):
                latest_sma = indicator_results_structured["sma"]["sma"][-1]
                report_sections.append(
                    f"- 港股SMA({indicator_results_structured['sma']['period']}): HK${latest_sma:.2f} (最新值)"
                )
                if len(indicator_results_structured["sma"]["sma"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["sma"]["sma"][-1]
                        > indicator_results_structured["sma"]["sma"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(indicator_results_structured['sma']['sma'])} 个数据点)"
                    )
            elif indicator_results_structured["sma"]["error"]:
                report_sections.append(
                    f"- 港股SMA: 错误 - {indicator_results_structured['sma']['error']}"
                )

        # RSI分析
        if "RSI" in indicators_to_run:
            rsi_period = inputs.rsi_period or settings.DEFAULT_RSI_PERIOD
            rsi_input = RsiInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=rsi_period,
            )
            rsi_output = await calculate_hk_stock_rsi.run(
                {"ctx": ctx, "inputs": rsi_input}
            )
            indicator_results_structured["rsi"] = json.loads(
                rsi_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["rsi"]["rsi"] is not None
                and len(indicator_results_structured["rsi"]["rsi"]) > 0
            ):
                latest_rsi = indicator_results_structured["rsi"]["rsi"][-1]
                report_sections.append(
                    f"- 港股RSI({indicator_results_structured['rsi']['period']}): {latest_rsi:.2f}"
                )
                if latest_rsi > 70:
                    report_sections.append("  - 注意: RSI表明超买状态 (>70)")
                elif latest_rsi < 30:
                    report_sections.append("  - 注意: RSI表明超卖状态 (<30)")

                if len(indicator_results_structured["rsi"]["rsi"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["rsi"]["rsi"][-1]
                        > indicator_results_structured["rsi"]["rsi"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - 趋势: {trend} ({len(indicator_results_structured['rsi']['rsi'])} 个数据点)"
                    )
            elif indicator_results_structured["rsi"]["error"]:
                report_sections.append(
                    f"- 港股RSI: 错误 - {indicator_results_structured['rsi']['error']}"
                )

        # 合成报告
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base,
                error="无法计算任何指标数据",
            )

        report_title = f"港股技术分析报告 - {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} 个数据点:\n"
        report_text = report_title + "\n".join(report_sections)

        await ctx.info(
            f"Successfully generated HK stock comprehensive report for {inputs.symbol}"
        )
        return ComprehensiveAnalysisOutput(
            **output_base,
            report_text=report_text,
            structured_data=indicator_results_structured,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(
            f"Error in HK stock comprehensive report for {inputs.symbol}: {e}",
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"港股综合分析报告生成错误: {str(e)}"
        )
