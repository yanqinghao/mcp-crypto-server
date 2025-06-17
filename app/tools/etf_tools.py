import numpy as np
import talib
import json
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta, time
from enum import Enum
from pydantic import BaseModel

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

# 导入ETF数据服务
from services.etf_service import (
    fetch_etf_spot_data,
    fetch_etf_hist_data,
    fetch_etf_intraday_data,
    fetch_etf_category_data,
    fetch_etf_ths_data,
    search_etf_by_keyword,
)

mcp = FastMCP()

# ==================== ETF专用枚举和模型 ====================


class ETFMarketType(str, Enum):
    A_STOCK = "A"  # A股ETF
    HK_STOCK = "HK"  # 港股ETF
    US_STOCK = "US"  # 美股ETF


class ETFCategoryType(str, Enum):
    ETF_FUND = "ETF基金"
    BOND_ETF = "债券ETF"
    COMMODITY_ETF = "商品ETF"
    CURRENCY_ETF = "货币ETF"
    CLOSED_FUND = "封闭式基金"
    REITS = "REITS"


# ETF专用输入模型
class ETFCandlesInput(CandlesInput):
    market_type: ETFMarketType = ETFMarketType.A_STOCK


class ETFPriceInput(PriceInput):
    market_type: ETFMarketType = ETFMarketType.A_STOCK


class ETFTickerInput(TickerInput):
    market_type: ETFMarketType = ETFMarketType.A_STOCK


class ETFSearchInput(BaseModel):
    keyword: str
    market_type: ETFMarketType = ETFMarketType.A_STOCK
    limit: int = 10


class ETFCategoryInput(BaseModel):
    category: ETFCategoryType = ETFCategoryType.ETF_FUND
    limit: int = 50


class ETFDiscountAnalysisInput(BaseModel):
    symbol: str
    market_type: ETFMarketType = ETFMarketType.A_STOCK
    days: int = 30


class ETFComparisonInput(BaseModel):
    symbols: List[str]
    market_type: ETFMarketType = ETFMarketType.A_STOCK
    period_days: int = 30


class ETFScreenInput(BaseModel):
    market_type: ETFMarketType = ETFMarketType.A_STOCK
    min_volume: Optional[float] = None
    max_discount_rate: Optional[float] = None
    min_nav: Optional[float] = None
    category: Optional[str] = None
    limit: int = 20


# ==================== ETF数据获取辅助函数 ====================


async def _fetch_etf_single_series_data(
    ctx: Context,
    symbol: str,
    period: str,
    market_type: str,
    required_candles: int,
    series_type: str = "close",
) -> Optional[np.ndarray]:
    """
    获取ETF单个数据序列
    """
    try:
        buffer_days = max(20, required_candles // 4)
        total_days = required_candles + buffer_days

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=total_days + 50)).strftime(
            "%Y%m%d"
        )

        etf_data = await fetch_etf_hist_data(
            ctx, symbol, market_type, period, start_date, end_date
        )

        if not etf_data or len(etf_data) < required_candles:
            await ctx.error(
                f"Insufficient ETF data for {symbol}: got {len(etf_data) if etf_data else 0}, need {required_candles}"
            )
            return None

        if series_type not in ["open", "high", "low", "close", "volume"]:
            await ctx.error(f"Invalid series type: {series_type}")
            return None

        data_series = np.array([item[series_type] for item in etf_data])

        # 数据质量检查
        if np.any(np.isnan(data_series)) or np.any(np.isinf(data_series)):
            await ctx.warning(
                f"Found NaN or Inf values in {series_type} data for ETF {symbol}"
            )
            data_series = data_series[~(np.isnan(data_series) | np.isinf(data_series))]

        return data_series

    except Exception as e:
        await ctx.error(f"Error fetching ETF {series_type} data for {symbol}: {e}")
        return None


async def _fetch_etf_multi_series_data(
    ctx: Context,
    symbol: str,
    period: str,
    market_type: str,
    required_candles: int,
    series_types: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    """
    获取ETF多个数据序列
    """
    try:
        buffer_days = max(20, required_candles // 4)
        total_days = required_candles + buffer_days

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=total_days + 50)).strftime(
            "%Y%m%d"
        )

        etf_data = await fetch_etf_hist_data(
            ctx, symbol, market_type, period, start_date, end_date
        )

        if not etf_data or len(etf_data) < required_candles:
            await ctx.error(
                f"Insufficient ETF data for {symbol}: got {len(etf_data) if etf_data else 0}, need {required_candles}"
            )
            return None

        result = {}
        for series_type in series_types:
            if series_type not in ["open", "high", "low", "close", "volume"]:
                await ctx.error(f"Invalid series type: {series_type}")
                return None

            data_series = np.array([item[series_type] for item in etf_data])

            # 数据质量检查
            if np.any(np.isnan(data_series)) or np.any(np.isinf(data_series)):
                await ctx.warning(
                    f"Found NaN or Inf values in {series_type} data for ETF {symbol}"
                )
                data_series = data_series[
                    ~(np.isnan(data_series) | np.isinf(data_series))
                ]

            result[series_type] = data_series

        # 确保所有序列长度一致
        min_length = min(len(series) for series in result.values())
        if min_length < required_candles:
            await ctx.error(
                f"Insufficient clean ETF data after filtering: {min_length} < {required_candles}"
            )
            return None

        # 截取到相同长度
        for key in result:
            result[key] = result[key][-min_length:]

        return result

    except Exception as e:
        await ctx.error(f"Error fetching ETF multi-series data for {symbol}: {e}")
        return None


def _extract_valid_values(values: np.ndarray, history_len: int) -> List[float]:
    """从TA-Lib计算结果中提取有效值"""
    valid_values = values[~np.isnan(values)]
    if len(valid_values) >= history_len:
        return [float(x) for x in valid_values[-history_len:]]
    else:
        return [float(x) for x in valid_values]


# ==================== ETF基础数据获取工具 ====================


@mcp.tool()
async def get_etf_candles(ctx: Context, inputs: ETFCandlesInput) -> CandlesOutput:
    """
    获取ETF K线数据

    该工具专门用于获取ETF的历史价格数据，支持多种时间框架和复权方式。
    ETF作为被动投资工具，其价格走势通常反映标的指数的表现，K线分析有助于
    识别趋势、支撑阻力位以及入场时机。

    ETF K线特点:
    - 价格相对稳定，波动性低于个股
    - 流动性好，价差小
    - 受标的指数影响大
    - 存在申购赎回机制的影响
    - 可能出现折价溢价现象

    功能特点:
    - 支持A股、港股、美股ETF
    - 多种复权方式处理
    - 自动处理节假日和停牌
    - 数据质量验证
    - 统一格式输出

    Args:
        inputs.symbol (str): ETF代码
            - A股: 6位数字代码，如510050(上证50ETF)、159919(沪深300ETF)
            - 港股: 4-5位数字代码，如2800(盈富基金)、3188(CAM中证港股通高息ETF)
            - 美股: 字母代码，如SPY(SPDR S&P500)、QQQ(纳斯达克100)
        inputs.timeframe (TimeFrame): 时间框架
            - 1d: 日线（最常用）
            - 1w: 周线（中期分析）
            - 1M: 月线（长期分析）
        inputs.limit (int): 获取的K线数量，1-1000
        inputs.market_type (ETFMarketType): 市场类型
        inputs.adjust (str): 复权类型
            - "qfq": 前复权（推荐，适合技术分析）
            - "hfq": 后复权（适合查看历史真实价格）
            - "": 不复权（原始价格数据）

    Returns:
        CandlesOutput: K线数据输出对象
        - symbol: ETF代码
        - timeframe: 时间框架
        - candles: K线数据列表，每个包含：
            - timestamp: 时间戳（毫秒）
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量（份额）
        - count: 实际返回的K线数量
        - error: 错误信息(如果有)

    复权类型说明:
    1. 前复权(qfq): 保持最新价格不变，向前调整历史价格
        - 优点: 价格连续性好，适合技术分析
        - 缺点: 历史价格非真实交易价格

    2. 后复权(hfq): 保持历史价格不变，向后调整最新价格
        - 优点: 历史价格真实
        - 缺点: 最新价格可能很高，不便于分析

    3. 不复权(""): 使用原始交易价格
        - 优点: 真实交易价格
        - 缺点: 分红除权会造成价格跳空

    Example:
        获取上证50ETF的日线数据:
        >>> from datetime import datetime
        >>> inputs = ETFCandlesInput(
        ...     symbol="510050",           # 上证50ETF
        ...     timeframe=TimeFrame.ONE_DAY,
        ...     limit=100,                 # 最近100个交易日
        ...     market_type=ETFMarketType.A_STOCK,
        ...     adjust="qfq"              # 前复权
        ... )
        >>> result = await get_etf_candles(ctx, inputs)
        >>> if result.candles:
        ...     print(f"获取到{result.count}根K线")
        ...     latest = result.candles[-1]
        ...     print(f"最新收盘价: ¥{latest.close:.3f}")
        ...     print(f"最新成交量: {latest.volume:,.0f}份")
        ...
        ...     # 计算涨跌幅
        ...     if len(result.candles) > 1:
        ...         prev_close = result.candles[-2].close
        ...         change_pct = (latest.close - prev_close) / prev_close * 100
        ...         print(f"涨跌幅: {change_pct:+.2f}%")

        获取纳斯达克100ETF的周线数据:
        >>> inputs = ETFCandlesInput(
        ...     symbol="QQQ",
        ...     timeframe=TimeFrame.ONE_WEEK,
        ...     limit=52,                  # 一年的周线
        ...     market_type=ETFMarketType.US_STOCK,
        ...     adjust="qfq"
        ... )
        >>> result = await get_etf_candles(ctx, inputs)
        >>> if result.candles:
        ...     # 计算年化收益率
        ...     start_price = result.candles[0].close
        ...     end_price = result.candles[-1].close
        ...     annual_return = (end_price - start_price) / start_price * 100
        ...     print(f"QQQ年化收益率: {annual_return:.2f}%")

    应用场景:
    - 趋势分析: 识别ETF的长期趋势方向
    - 支撑阻力: 寻找关键价格位
    - 技术指标: 为技术分析提供数据源
    - 择时交易: 确定买入卖出时机
    - 波动率分析: 评估ETF的价格波动特征
    - 相关性分析: 对比不同ETF的走势

    ETF分析要点:
    - 关注标的指数表现
    - 注意申购赎回对价格的影响
    - 考虑折价溢价因素
    - 重视成交量变化
    - 关注分红除权日期

    注意事项:
    - ETF价格受标的资产影响，基本面分析也很重要
    - 不同市场的交易时间和规则不同
    - 部分ETF可能存在跟踪误差
    - 新上市ETF历史数据可能不足
    - 停牌或节假日无交易数据

    Note:
        - 数据按交易日提供，不包括节假日
        - 建议使用前复权数据进行技术分析
        - 成交量单位为ETF份额，不是金额
        - 不同市场的价格精度可能不同
    """
    await ctx.info(
        f"Fetching ETF candles for {inputs.symbol} ({inputs.timeframe.value}) in {inputs.market_type.value} market"
    )

    try:
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe.value, "daily")

        # 获取复权类型
        adjust = getattr(inputs, "adjust", "qfq") or "qfq"

        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=inputs.limit + 50)).strftime(
            "%Y%m%d"
        )

        # 获取ETF数据
        etf_data = await fetch_etf_hist_data(
            ctx=ctx,
            symbol=inputs.symbol,
            market=inputs.market_type.value,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if etf_data:
            candles = []
            for item in etf_data[-inputs.limit :]:  # 只取需要的数量
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
                f"Successfully fetched {len(candles)} ETF candles for {inputs.symbol}"
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
                error=f"No ETF candle data available for {inputs.symbol}",
            )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(f"Error fetching ETF candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"ETF data error: {str(e)}",
        )


@mcp.tool()
async def get_etf_price(ctx: Context, inputs: ETFPriceInput) -> PriceOutput:
    """
    获取ETF当前价格

    获取指定ETF的实时或最新价格信息。ETF价格反映了其标的资产的实时价值，
    通过做市商制度和申购赎回机制，ETF价格通常与其净值保持较小差距。

    ETF价格特征:
    - 实时价格: 交易时间内实时变动
    - 净值关联: 与基金净值密切相关
    - 流动性好: 大部分ETF买卖价差较小
    - 套利机制: 存在一二级市场套利机制
    - 透明度高: 价格发现机制完善

    功能特点:
    - 多市场支持
    - 实时价格获取
    - 自动货币识别
    - 异常处理
    - 数据验证

    Args:
        inputs.symbol (str): ETF代码
        inputs.market_type (ETFMarketType): 市场类型, A、US、HK

    Returns:
        PriceOutput: 价格输出对象
        - symbol: ETF代码
        - price: 当前价格（最新成交价或收盘价）
        - timestamp: 价格时间戳（毫秒）
        - error: 错误信息(如果有)

    价格数据说明:
    1. 交易时间内: 返回最新成交价
    2. 交易时间外: 返回最新收盘价
    3. 停牌期间: 返回停牌前最后价格
    4. 新上市ETF: 可能没有历史价格

    Example:
        获取上证50ETF当前价格:
        >>> inputs = ETFPriceInput(
        ...     symbol="510050",
        ...     market_type=ETFMarketType.A_STOCK
        ... )
        >>> result = await get_etf_price(ctx, inputs)
        >>> if result.price:
        ...     print(f"上证50ETF当前价格: ¥{result.price:.3f}")
        ...
        ...     # 时间戳转换
        ...     from datetime import datetime
        ...     price_time = datetime.fromtimestamp(result.timestamp / 1000)
        ...     print(f"价格时间: {price_time.strftime('%Y-%m-%d %H:%M:%S')}")

        批量获取多个ETF价格:
        >>> etf_codes = ["510050", "510300", "159919"]  # 50ETF, 300ETF, 创业板ETF
        >>> prices = {}
        >>> for code in etf_codes:
        ...     inputs = ETFPriceInput(symbol=code, market_type=ETFMarketType.A_STOCK)
        ...     result = await get_etf_price(ctx, inputs)
        ...     if result.price:
        ...         prices[code] = result.price
        >>> print("ETF价格对比:", prices)

        美股ETF价格监控:
        >>> spy_input = ETFPriceInput(
        ...     symbol="SPY",
        ...     market_type=ETFMarketType.US_STOCK
        ... )
        >>> spy_result = await get_etf_price(ctx, spy_input)
        >>> if spy_result.price:
        ...     print(f"标普500ETF价格: ${spy_result.price:.2f}")

    应用场景:
    - 实时监控: 跟踪ETF价格变化
    - 交易决策: 确定买卖时机
    - 套利分析: 比较价格与净值差异
    - 投资组合: 实时估值计算
    - 风险管理: 止损止盈设置
    - 市场分析: 板块轮动观察

    价格解读:
    - 价格上涨: 可能反映标的资产强势或资金流入
    - 价格下跌: 可能反映标的资产弱势或资金流出
    - 价格稳定: 标的资产波动小或市场观望情绪浓
    - 异常波动: 可能存在重大消息或技术性因素

    注意事项:
    - 价格可能存在几秒钟延迟
    - 停牌ETF显示停牌前价格
    - 新上市ETF可能价格波动较大
    - 跨境ETF受汇率影响
    - 部分小众ETF流动性较差

    Note:
        - 实时性取决于数据源更新频率
        - 建议结合成交量分析价格有效性
        - 注意区分交易价格和净值价格
        - 考虑市场开闭市时间的影响
    """
    await ctx.info(
        f"Fetching ETF current price for {inputs.symbol} in {inputs.market_type.value} market"
    )

    try:
        # 获取ETF实时数据
        realtime_data = await fetch_etf_spot_data(
            ctx, [inputs.symbol], inputs.market_type.value
        )

        if realtime_data and len(realtime_data) > 0:
            etf_info = realtime_data[0]
            timestamp = int(datetime.now().timestamp() * 1000)

            currency_symbol = (
                "¥"
                if inputs.market_type == ETFMarketType.A_STOCK
                else ("HK$" if inputs.market_type == ETFMarketType.HK_STOCK else "$")
            )

            await ctx.info(
                f"ETF current price for {inputs.symbol}: {currency_symbol}{etf_info['price']}"
            )
            return PriceOutput(
                symbol=inputs.symbol,
                price=etf_info["price"],
                timestamp=timestamp,
            )
        else:
            # 如果实时数据不可用，尝试获取最新的历史数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

            etf_data = await fetch_etf_hist_data(
                ctx,
                inputs.symbol,
                inputs.market_type.value,
                "daily",
                start_date,
                end_date,
            )

            if etf_data and len(etf_data) > 0:
                latest = etf_data[-1]
                timestamp = int(
                    datetime.strptime(latest["date"], "%Y-%m-%d").timestamp() * 1000
                )

                return PriceOutput(
                    symbol=inputs.symbol,
                    price=latest["close"],
                    timestamp=timestamp,
                )

            return PriceOutput(
                symbol=inputs.symbol, error="ETF price data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching ETF price for {inputs.symbol}: {e}")
        return PriceOutput(symbol=inputs.symbol, error=f"ETF price error: {str(e)}")


@mcp.tool()
async def get_etf_ticker(ctx: Context, inputs: ETFTickerInput) -> TickerOutput:
    """
    获取ETF详细行情数据

    获取指定ETF的全面行情信息，包括价格、成交量、涨跌幅等关键市场数据。
    相比简单的价格查询，详细行情提供了更丰富的市场信息，有助于全面分析ETF的表现。

    ETF行情特点:
    - 价格稳定性相对较高
    - 成交量反映资金关注度
    - 涨跌幅通常与标的指数同步
    - 换手率相对较低
    - 存在申购赎回的影响

    功能特点:
    - 全面行情数据
    - 实时更新
    - 多市场支持
    - 数据验证
    - 统一格式

    Args:
        inputs.symbol (str): ETF代码
        inputs.market_type (ETFMarketType): 市场类型, A、US、HK

    Returns:
        TickerOutput: 详细行情输出对象
        - symbol: ETF代码
        - last: 最新成交价
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价（盘中时等于last）
        - volume: 成交量（份额）
        - change: 价格变化（绝对值）
        - percentage: 涨跌幅百分比
        - timestamp: 数据时间戳
        - error: 错误信息(如果有)

    行情数据解读:
    1. 价格指标:
        - last: 最重要的价格参考
        - open: 当日开盘价，反映市场预期
        - high/low: 当日价格区间，显示波动范围

    2. 成交量指标:
        - volume: 交易活跃度，ETF通常以"万份"为单位
        - 高成交量: 资金关注度高，流动性好
        - 低成交量: 可能影响交易成本

    3. 变化指标:
        - percentage: 标准化涨跌幅，便于比较
        - change: 绝对价格变化

    Example:
        获取沪深300ETF详细行情:
        >>> inputs = ETFTickerInput(
        ...     symbol="510300",
        ...     market_type=ETFMarketType.A_STOCK
        ... )
        >>> result = await get_etf_ticker(ctx, inputs)
        >>> if not result.error:
        ...     print(f"沪深300ETF (510300) 详细行情:")
        ...     print(f"最新价: ¥{result.last:.3f}")
        ...     print(f"开盘价: ¥{result.open:.3f}")
        ...     print(f"最高价: ¥{result.high:.3f}")
        ...     print(f"最低价: ¥{result.low:.3f}")
        ...     print(f"成交量: {result.volume:,.0f}万份")
        ...     print(f"涨跌幅: {result.percentage:+.2f}%")
        ...
        ...     # 当日波动率分析
        ...     if result.high and result.low and result.last:
        ...         daily_range = result.high - result.low
        ...         volatility = daily_range / result.last * 100
        ...         print(f"当日波动率: {volatility:.2f}%")
        ...
        ...     # 价格位置分析
        ...     if result.high and result.low and result.last:
        ...         price_position = (result.last - result.low) / (result.high - result.low)
        ...         position_desc = "高位" if price_position > 0.7 else "低位" if price_position < 0.3 else "中位"
        ...         print(f"价格位置: {position_desc} ({price_position:.1%})")

        港股ETF行情分析:
        >>> inputs = ETFTickerInput(
        ...     symbol="2800",  # 盈富基金
        ...     market_type=ETFMarketType.HK_STOCK
        ... )
        >>> result = await get_etf_ticker(ctx, inputs)
        >>> if not result.error:
        ...     print(f"盈富基金行情: HK${result.last:.2f}")
        ...     print(f"今日涨跌: {result.percentage:+.2f}%")

        美股ETF行情监控:
        >>> spy_inputs = ETFTickerInput(
        ...     symbol="SPY",
        ...     market_type=ETFMarketType.US_STOCK
        ... )
        >>> spy_result = await get_etf_ticker(ctx, spy_inputs)
        >>> if not spy_result.error:
        ...     print(f"标普500ETF: ${spy_result.last:.2f} ({spy_result.percentage:+.2f}%)")

    应用场景:
    - 交易监控: 实时跟踪ETF表现
    - 市场分析: 分析板块资金流向
    - 比较研究: 对比同类ETF表现
    - 择时决策: 根据技术指标判断买卖时机
    - 风险评估: 评估投资组合波动
    - 流动性分析: 判断ETF交易活跃度

    行情分析技巧:
    - 成交量放大通常伴随重要价格变动
    - 涨跌幅与标的指数对比分析
    - 价格位置判断当日强弱
    - 开盘价反映隔夜消息面影响
    - 高低价区间显示支撑阻力

    投资提示:
    - ETF涨跌主要跟随标的指数
    - 异常涨跌可能存在套利机会
    - 成交量萎缩时谨慎交易
    - 关注折价溢价情况
    - 考虑分红除权的影响

    注意事项:
    - 数据可能存在延迟
    - 停牌ETF无当日行情
    - 新上市ETF数据可能不完整
    - 跨境ETF受汇率影响
    - 分级ETF有特殊交易规则

    Note:
        - 建议结合基本面分析
        - 关注标的指数表现
        - 注意市场整体环境
        - 考虑宏观经济因素影响
    """
    await ctx.info(
        f"Fetching ETF ticker for {inputs.symbol} in {inputs.market_type.value} market"
    )

    try:
        # 获取ETF实时数据
        realtime_data = await fetch_etf_spot_data(
            ctx, [inputs.symbol], inputs.market_type.value
        )

        if realtime_data and len(realtime_data) > 0:
            etf_info = realtime_data[0]
            timestamp = int(datetime.now().timestamp() * 1000)

            await ctx.info(f"Successfully fetched ETF ticker for {inputs.symbol}")
            return TickerOutput(
                symbol=inputs.symbol,
                last=etf_info.get("price", 0),
                open=etf_info.get("open", 0),
                high=etf_info.get("high", 0),
                low=etf_info.get("low", 0),
                close=etf_info.get("price", 0),
                volume=etf_info.get("volume", 0),
                change=etf_info.get("change", 0),
                percentage=etf_info.get("pct_change", 0),
                timestamp=timestamp,
            )
        else:
            return TickerOutput(
                symbol=inputs.symbol, error="ETF ticker data not available"
            )

    except Exception as e:
        await ctx.error(f"Error fetching ETF ticker for {inputs.symbol}: {e}")
        return TickerOutput(symbol=inputs.symbol, error=f"ETF ticker error: {str(e)}")


@mcp.tool()
async def get_etf_intraday_data(
    ctx: Context,
    symbol: str,
    period: str = "1",
    market_type: ETFMarketType = ETFMarketType.A_STOCK,
) -> dict:
    """
    获取ETF分时行情数据（仅支持A股ETF）

    分时图显示了ETF在交易日内的价格变化轨迹，是短线交易和日内分析的重要工具。
    通过分时数据可以观察资金流入流出的节奏，识别关键的买卖时点。

    分时数据特点:
    - 高频数据，反映日内交易细节
    - 显示价格和成交量的时间序列
    - 可识别盘中关键转折点
    - 反映市场情绪变化
    - 适用于短线交易策略

    功能特点:
    - 多种时间周期支持
    - 当日实时数据
    - 价格成交量同步
    - 数据质量验证
    - 结构化输出

    Args:
        symbol (str): A股ETF代码（6位数字）
        period (str): 分时周期
            - "1": 1分钟K线
            - "5": 5分钟K线
            - "15": 15分钟K线
            - "30": 30分钟K线
            - "60": 60分钟K线
        market_type (ETFMarketType): 市场类型，仅支持A（A股）

    Returns:
        dict: 分时数据结果
        - success: 是否成功获取数据
        - symbol: ETF代码
        - period: 时间周期
        - data: 分时数据列表，每个数据点包含：
            - time: 时间（HH:MM格式）
            - price: 价格
            - volume: 成交量
            - amount: 成交额
        - count: 数据点数量
        - error: 错误信息(如果有)

    时间周期选择:
    - 1分钟: 超短线交易，捕捉瞬间机会
    - 5分钟: 短线交易，观察短期趋势
    - 15分钟: 中短线结合，确认价格方向
    - 30分钟: 日内波段，寻找较大机会
    - 60分钟: 日内趋势，适合稳健操作

    Example:
        获取上证50ETF的5分钟分时数据:
        >>> result = await get_etf_intraday_data(
        ...     ctx=ctx,
        ...     symbol="510050",
        ...     period="5",
        ...     market_type=ETFMarketType.A_STOCK
        ... )
        >>> if result["success"]:
        ...     print(f"获取到{result['count']}个5分钟数据点")
        ...     data = result["data"]
        ...
        ...     # 分析开盘表现
        ...     if len(data) > 0:
        ...         first_point = data[0]
        ...         print(f"开盘时间: {first_point['time']}")
        ...         print(f"开盘价格: ¥{first_point['price']:.3f}")
        ...
        ...     # 分析最新表现
        ...     if len(data) > 0:
        ...         latest_point = data[-1]
        ...         print(f"最新时间: {latest_point['time']}")
        ...         print(f"最新价格: ¥{latest_point['price']:.3f}")
        ...         print(f"最新成交量: {latest_point['volume']:,.0f}")
        ...
        ...     # 计算日内涨跌幅
        ...     if len(data) > 1:
        ...         start_price = data[0]["price"]
        ...         current_price = data[-1]["price"]
        ...         intraday_change = (current_price - start_price) / start_price * 100
        ...         print(f"日内涨跌幅: {intraday_change:+.2f}%")
        ...
        ...     # 寻找成交量峰值时点
        ...     if len(data) > 10:
        ...         max_volume_point = max(data, key=lambda x: x["volume"])
        ...         print(f"成交量峰值: {max_volume_point['time']} - {max_volume_point['volume']:,.0f}")

        分析创业板ETF的1分钟数据:
        >>> result = await get_etf_intraday_data(
        ...     ctx=ctx,
        ...     symbol="159915",  # 创业板ETF
        ...     period="1"
        ... )
        >>> if result["success"]:
        ...     data = result["data"]
        ...
        ...     # 计算分时波动率
        ...     if len(data) > 30:
        ...         prices = [point["price"] for point in data[-30:]]  # 最近30分钟
        ...         price_changes = [
        ...             (prices[i] - prices[i-1]) / prices[i-1]
        ...             for i in range(1, len(prices))
        ...         ]
        ...         volatility = np.std(price_changes) * np.sqrt(240) * 100  # 年化波动率
        ...         print(f"最近30分钟年化波动率: {volatility:.2f}%")

    应用场景:
    - 日内交易: 寻找最佳买卖时机
    - 量价分析: 观察价格与成交量关系
    - 趋势确认: 验证日线级别的趋势
    - 情绪分析: 判断市场参与者情绪
    - 套利交易: 识别短期价格偏差
    - 风险控制: 实时监控持仓风险

    分时分析技巧:
    - 开盘30分钟: 观察市场方向
    - 上午收盘前: 注意资金流向
    - 下午开盘后: 确认趋势延续
    - 尾盘30分钟: 关注收盘力度
    - 成交量配合: 量价齐升/齐跌更可信

    关键时间节点:
    - 09:30-10:00: 开盘试探阶段
    - 10:00-11:30: 上午主要交易时段
    - 13:00-14:00: 下午开盘确认阶段
    - 14:00-15:00: 下午主要交易时段
    - 14:30-15:00: 尾盘阶段，常有异动

    注意事项:
    - 仅支持A股ETF，港股美股不支持
    - 数据仅限当日交易时间
    - 高频数据可能存在噪音
    - 分时图受市场整体环境影响大
    - 需要结合日线等更大周期确认

    Note:
        - 数据更新频率取决于交易所
        - 建议结合成交量分析价格变动
        - 分时数据适合短线操作参考
        - 长线投资者可关注日内关键时点
    """
    await ctx.info(f"Fetching ETF intraday data for {symbol} ({period}min)")

    try:
        if market_type != ETFMarketType.A_STOCK:
            return {
                "success": False,
                "error": "Intraday data only available for A-stock ETFs",
            }

        # 设置当日时间范围
        today = datetime.now().strftime("%Y-%m-%d")
        start_date = f"{today} 09:30:00"
        end_date = f"{today} 15:00:00"

        intraday_data = await fetch_etf_intraday_data(
            ctx, symbol, period, start_date, end_date
        )

        if intraday_data:
            return {
                "success": True,
                "symbol": symbol,
                "period": f"{period}min",
                "data": intraday_data,
                "count": len(intraday_data),
            }
        else:
            return {
                "success": False,
                "symbol": symbol,
                "error": "No intraday data available",
            }

    except Exception as e:
        await ctx.error(f"Error fetching ETF intraday data: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


# ==================== ETF搜索和分类工具 ====================


@mcp.tool()
async def search_etf_symbols(ctx: Context, inputs: ETFSearchInput) -> dict:
    """
    搜索ETF代码和基本信息

    通过关键词搜索ETF，支持按名称、代码、投资主题等多种方式查找。
    这是ETF投资的第一步，帮助投资者快速找到符合投资需求的ETF产品。

    搜索功能特点:
    - 智能匹配算法
    - 多维度搜索支持
    - 结果相关性排序
    - 基本信息展示
    - 多市场覆盖

    Args:
        inputs.keyword (str): 搜索关键词
            - ETF名称: 如"上证50"、"沪深300"、"科技"
            - ETF代码: 如"510050"、"159919"
            - 投资主题: 如"医疗"、"新能源"、"消费"
            - 英文名称: 如"Technology"、"Healthcare"（美股ETF）
        inputs.market_type (ETFMarketType): 搜索的市场类型, A、US、HK
        inputs.limit (int): 返回结果数量限制，1-100

    Returns:
        dict: 搜索结果
        - success: 搜索是否成功
        - query: 搜索关键词
        - market_type: 市场类型
        - asset_type: 资产类型（固定为"etf"）
        - results: 搜索结果列表，每个结果包含：
            - symbol: ETF代码
            - name: ETF名称
            - category: ETF分类
            - size: 规模（如果有）
            - nav: 净值（如果有）
            - description: 描述信息
        - count: 实际返回结果数量
        - error: 错误信息(如果有)

    搜索技巧:
    1. 精确搜索: 使用完整的ETF代码或名称
    2. 模糊搜索: 使用关键词如"科技"、"医疗"
    3. 主题搜索: 使用投资主题如"新能源"、"5G"
    4. 分类搜索: 使用"债券"、"商品"等分类词
    5. 英文搜索: 美股ETF支持英文关键词

    Example:
        搜索科技类ETF:
        >>> inputs = ETFSearchInput(
        ...     keyword="科技",
        ...     market_type=ETFMarketType.A_STOCK,
        ...     limit=10
        ... )
        >>> result = await search_etf_symbols(ctx, inputs)
        >>> if result["success"]:
        ...     print(f"找到{result['count']}只科技类ETF:")
        ...     for etf in result["results"]:
        ...         print(f"- {etf['symbol']}: {etf['name']}")
        ...         if "size" in etf:
        ...             print(f"  规模: {etf['size']}亿元")

        按代码精确查找:
        >>> inputs = ETFSearchInput(
        ...     keyword="510050",
        ...     market_type=ETFMarketType.A_STOCK,
        ...     limit=5
        ... )
        >>> result = await search_etf_symbols(ctx, inputs)
        >>> if result["success"] and result["results"]:
        ...     etf = result["results"][0]
        ...     print(f"ETF信息: {etf['symbol']} - {etf['name']}")
        ...     print(f"分类: {etf.get('category', '未知')}")

        搜索美股科技ETF:
        >>> inputs = ETFSearchInput(
        ...     keyword="Technology",
        ...     market_type=ETFMarketType.US_STOCK,
        ...     limit=15
        ... )
        >>> result = await search_etf_symbols(ctx, inputs)
        >>> if result["success"]:
        ...     tech_etfs = result["results"]
        ...     print("美股科技ETF列表:")
        ...     for etf in tech_etfs[:5]:  # 显示前5只
        ...         print(f"- {etf['symbol']}: {etf['name']}")

        搜索债券ETF:
        >>> inputs = ETFSearchInput(
        ...     keyword="债券",
        ...     market_type=ETFMarketType.A_STOCK,
        ...     limit=20
        ... )
        >>> result = await search_etf_symbols(ctx, inputs)
        >>> if result["success"]:
        ...     bond_etfs = [etf for etf in result["results"]
        ...                  if "债" in etf["name"] or "债券" in etf.get("category", "")]
        ...     print(f"找到{len(bond_etfs)}只债券ETF")

    应用场景:
    - 产品筛选: 根据投资主题寻找合适ETF
    - 市场调研: 了解某个行业的ETF产品
    - 代码查询: 根据名称查找ETF代码
    - 投资规划: 构建ETF投资组合
    - 竞品分析: 比较同类ETF产品
    - 学习研究: 了解ETF市场结构

    搜索策略:
    - 宽泛搜索: 先用大类关键词了解全貌
    - 精确搜索: 再用具体词汇定位目标
    - 多维验证: 结合代码、名称、分类确认
    - 交叉比较: 对比不同搜索结果
    - 深入研究: 选定目标后深入分析

    投资提示:
    - 搜索只是第一步，还需深入分析
    - 关注ETF的跟踪指数和费率
    - 比较同类ETF的规模和流动性
    - 了解ETF的风险收益特征
    - 考虑ETF在投资组合中的作用

    注意事项:
    - 搜索结果可能不完整
    - 新上市ETF可能搜索不到
    - 停牌或清盘ETF可能仍显示
    - 不同数据源结果可能有差异
    - 建议多关键词交叉验证

    Note:
        - 支持中英文关键词搜索
        - 结果按相关性和规模排序
        - 建议使用具体而非泛泛的关键词
        - 搜索结果仅供参考，投资需谨慎
    """
    await ctx.info(
        f"Searching ETF symbols for '{inputs.keyword}' in {inputs.market_type.value} market"
    )

    try:
        results = await search_etf_by_keyword(
            ctx, inputs.keyword, inputs.market_type.value
        )

        if results:
            limited_results = results[: inputs.limit]
            return {
                "success": True,
                "query": inputs.keyword,
                "market_type": inputs.market_type.value,
                "asset_type": "etf",
                "results": limited_results,
                "count": len(limited_results),
            }
        else:
            return {
                "success": False,
                "query": inputs.keyword,
                "market_type": inputs.market_type.value,
                "asset_type": "etf",
                "error": "No matching ETFs found",
            }

    except Exception as e:
        await ctx.error(f"Error searching ETF symbols: {e}")
        return {
            "success": False,
            "query": inputs.keyword,
            "market_type": inputs.market_type.value,
            "asset_type": "etf",
            "error": str(e),
        }


@mcp.tool()
async def get_etf_category_list(ctx: Context, inputs: ETFCategoryInput) -> dict:
    """
    获取ETF分类列表

    按照资产类型、投资策略、行业主题等维度获取ETF分类列表。
    帮助投资者系统性地了解ETF市场结构，便于进行资产配置和投资决策。

    ETF分类体系:
    1. 按资产类型: 股票ETF、债券ETF、商品ETF、货币ETF等
    2. 按投资范围: 宽基指数、行业主题、风格因子等
    3. 按地域分布: 境内、境外、跨境等
    4. 按投资策略: 被动跟踪、Smart Beta、增强指数等

    功能特点:
    - 全面分类覆盖
    - 结构化展示
    - 多维度筛选
    - 实时数据更新
    - 便于比较分析

    Args:
        inputs.category (ETFCategoryType): ETF分类类型
        inputs.limit (int): 返回结果数量限制，1-200

    Returns:
        dict: 分类列表结果
        - success: 获取是否成功
        - category: 分类名称
        - etfs: ETF列表，每个ETF包含：
            - symbol: ETF代码
            - name: ETF名称
            - nav: 净值
            - size: 规模
            - expense_ratio: 费率
            - tracking_index: 跟踪指数
            - listing_date: 上市日期
        - count: 实际返回数量
        - total_available: 该分类总ETF数量
        - error: 错误信息(如果有)

    分类详解:
    1. ETF基金: 主要投资股票的ETF产品
        - 宽基指数: 上证50、沪深300、中证500等
        - 行业主题: 科技、医疗、消费、金融等
        - 风格因子: 价值、成长、红利、低波等

    2. 债券ETF: 主要投资债券的ETF产品
        - 国债ETF: 国债指数、长期国债等
        - 信用债ETF: 企业债、可转债等
        - 利率债ETF: 政策性金融债等

    3. 商品ETF: 投资大宗商品的ETF产品
        - 贵金属: 黄金ETF、白银ETF
        - 能源: 原油ETF、天然气ETF
        - 农产品: 农业ETF、大豆ETF

    4. 货币ETF: 投资货币市场的ETF产品
        - 货币基金ETF
        - 短期理财ETF

    Example:
        获取股票ETF列表:
        >>> inputs = ETFCategoryInput(
        ...     category=ETFCategoryType.ETF_FUND,
        ...     limit=50
        ... )
        >>> result = await get_etf_category_list(ctx, inputs)
        >>> if result["success"]:
        ...     print(f"股票ETF分类共有{result['total_available']}只产品")
        ...     print(f"返回前{result['count']}只:")
        ...
        ...     for etf in result["etfs"][:10]:  # 显示前10只
        ...         name = etf["name"]
        ...         symbol = etf["symbol"]
        ...         size = etf.get("size", 0)
        ...         print(f"- {symbol}: {name} (规模: {size:.1f}亿)")
        ...
        ...     # 按规模排序
        ...     etfs_by_size = sorted(result["etfs"],
        ...                          key=lambda x: x.get("size", 0),
        ...                          reverse=True)
        ...     print("\n规模最大的5只股票ETF:")
        ...     for etf in etfs_by_size[:5]:
        ...         print(f"- {etf['symbol']}: {etf['name']} ({etf.get('size', 0):.1f}亿)")

        获取债券ETF列表:
        >>> inputs = ETFCategoryInput(
        ...     category=ETFCategoryType.BOND_ETF,
        ...     limit=30
        ... )
        >>> result = await get_etf_category_list(ctx, inputs)
        >>> if result["success"]:
        ...     bond_etfs = result["etfs"]
        ...     print(f"债券ETF共{len(bond_etfs)}只:")
        ...
        ...     # 分析费率分布
        ...     expense_ratios = [etf.get("expense_ratio", 0)
        ...                      for etf in bond_etfs if etf.get("expense_ratio")]
        ...     if expense_ratios:
        ...         avg_fee = sum(expense_ratios) / len(expense_ratios)
        ...         print(f"平均费率: {avg_fee:.3f}%")

        获取商品ETF列表:
        >>> inputs = ETFCategoryInput(
        ...     category=ETFCategoryType.COMMODITY_ETF,
        ...     limit=20
        ... )
        >>> result = await get_etf_category_list(ctx, inputs)
        >>> if result["success"]:
        ...     commodity_etfs = result["etfs"]
        ...     print("商品ETF产品:")
        ...
        ...     for etf in commodity_etfs:
        ...         name = etf["name"]
        ...         symbol = etf["symbol"]
        ...         tracking = etf.get("tracking_index", "未知")
        ...         print(f"- {symbol}: {name}")
        ...         print(f"  跟踪标的: {tracking}")

    应用场景:
    - 资产配置: 了解各类资产的ETF选择
    - 行业分析: 研究特定行业的ETF产品
    - 产品比较: 对比同类ETF的优劣
    - 投资规划: 制定系统性投资策略
    - 市场研究: 分析ETF市场发展状况
    - 教育学习: 了解ETF产品体系

    分析维度:
    - 规模分析: 大规模ETF通常流动性更好
    - 费率比较: 低费率ETF长期收益更优
    - 跟踪误差: 评估ETF跟踪效果
    - 成立时间: 老牌ETF通常更稳定
    - 流动性: 日均成交量反映流动性

    投资建议:
    - 优先选择规模大、流动性好的ETF
    - 关注费率水平，选择低费率产品
    - 了解跟踪的指数特征
    - 考虑ETF在组合中的作用
    - 注意分散化投资原则

    注意事项:
    - 分类可能存在重叠
    - 新产品可能分类未及时更新
    - 不同数据源分类标准可能不同
    - 分类信息仅供参考
    - 投资前需详细了解产品特征

    Note:
        - 数据实时更新，反映最新市场情况
        - 建议结合投资目标选择合适分类
        - 可配合搜索功能精确定位产品
        - 投资需要考虑风险承受能力
    """
    await ctx.info(f"Fetching ETF category list for '{inputs.category.value}'")

    try:
        category_data = await fetch_etf_category_data(ctx, inputs.category.value)

        if category_data:
            limited_data = category_data[: inputs.limit]
            return {
                "success": True,
                "category": inputs.category.value,
                "etfs": limited_data,
                "count": len(limited_data),
                "total_available": len(category_data),
            }
        else:
            return {
                "success": False,
                "category": inputs.category.value,
                "error": "No ETFs found in this category",
            }

    except Exception as e:
        await ctx.error(f"Error fetching ETF category list: {e}")
        return {
            "success": False,
            "category": inputs.category.value,
            "error": str(e),
        }


@mcp.tool()
async def get_etf_ths_ranking(
    ctx: Context, date: Optional[str] = None, limit: int = 50
) -> dict:
    """
    获取同花顺ETF排行数据

    同花顺作为权威的金融数据服务商，其ETF排行榜反映了市场的实时表现和热点。
    通过排行数据可以快速了解市场焦点、资金流向和投资机会。

    排行榜特点:
    - 权威数据源
    - 多维度排序
    - 实时更新
    - 全面覆盖
    - 便于比较

    功能特点:
    - 历史数据查询
    - 多指标排序
    - 结果数量可控
    - 数据完整性好
    - 便于追踪分析

    Args:
        date (Optional[str]): 查询日期，YYYYMMDD格式
            - None: 使用当前日期
            - 历史日期: 查询指定日期的排行
        limit (int): 返回结果数量限制，1-200

    Returns:
        dict: 排行数据结果
        - success: 获取是否成功
        - date: 排行日期
        - source: 数据源（"同花顺"）
        - etfs: ETF排行列表，每个ETF包含：
            - rank: 排名
            - symbol: ETF代码
            - name: ETF名称
            - price: 最新价格
            - change: 价格变化
            - pct_change: 涨跌幅
            - volume: 成交量
            - amount: 成交额
            - turnover_rate: 换手率
            - pe_ratio: 市盈率（如果适用）
            - pb_ratio: 市净率（如果适用）
        - count: 实际返回数量
        - total_available: 总排行数量
        - error: 错误信息(如果有)

    排行指标说明:
    1. 涨跌幅排行: 反映当日表现最佳/最差的ETF
    2. 成交额排行: 反映资金关注度和流动性
    3. 换手率排行: 反映交易活跃程度
    4. 规模排行: 反映ETF市场地位

    Example:
        获取当日ETF排行:
        >>> result = await get_etf_ths_ranking(ctx, limit=20)
        >>> if result["success"]:
        ...     print(f"{result['date']} ETF排行榜 (数据源: {result['source']}):")
        ...     print(f"共{result['total_available']}只ETF参与排行")
        ...
        ...     # 显示前10名
        ...     for i, etf in enumerate(result["etfs"][:10], 1):
        ...         name = etf["name"]
        ...         pct_change = etf.get("pct_change", 0)
        ...         volume = etf.get("volume", 0)
        ...         print(f"{i:2d}. {name}: {pct_change:+.2f}% (成交量: {volume:,.0f})")
        ...
        ...     # 分析涨跌分布
        ...     up_count = sum(1 for etf in result["etfs"] if etf.get("pct_change", 0) > 0)
        ...     down_count = sum(1 for etf in result["etfs"] if etf.get("pct_change", 0) < 0)
        ...     flat_count = len(result["etfs"]) - up_count - down_count
        ...
        ...     print(f"\n市场概况:")
        ...     print(f"上涨: {up_count}只, 下跌: {down_count}只, 平盘: {flat_count}只")

        查询历史排行:
        >>> historical_date = "20241215"  # 查询2024年12月15日
        >>> result = await get_etf_ths_ranking(ctx, date=historical_date, limit=30)
        >>> if result["success"]:
        ...     print(f"{historical_date} 历史ETF排行:")
        ...
        ...     # 寻找当日涨幅最大的ETF
        ...     best_performer = max(result["etfs"],
        ...                         key=lambda x: x.get("pct_change", -999))
        ...     worst_performer = min(result["etfs"],
        ...                          key=lambda x: x.get("pct_change", 999))
        ...
        ...     print(f"当日最佳: {best_performer['name']} (+{best_performer.get('pct_change', 0):.2f}%)")
        ...     print(f"当日最差: {worst_performer['name']} ({worst_performer.get('pct_change', 0):.2f}%)")

        分析成交活跃度:
        >>> result = await get_etf_ths_ranking(ctx, limit=100)
        >>> if result["success"]:
        ...     etfs = result["etfs"]
        ...
        ...     # 按成交额排序
        ...     by_amount = sorted(etfs,
        ...                       key=lambda x: x.get("amount", 0),
        ...                       reverse=True)
        ...
        ...     print("成交额排行TOP10:")
        ...     for i, etf in enumerate(by_amount[:10], 1):
        ...         amount = etf.get("amount", 0)
        ...         print(f"{i:2d}. {etf['name']}: {amount:,.0f}万元")
        ...
        ...     # 按换手率排序
        ...     by_turnover = sorted(etfs,
        ...                         key=lambda x: x.get("turnover_rate", 0),
        ...                         reverse=True)
        ...
        ...     print("\n换手率排行TOP5:")
        ...     for i, etf in enumerate(by_turnover[:5], 1):
        ...         turnover = etf.get("turnover_rate", 0)
        ...         print(f"{i}. {etf['name']}: {turnover:.2f}%")

    应用场景:
    - 市场监控: 实时了解ETF市场表现
    - 热点发现: 识别当前市场热点板块
    - 资金流向: 分析资金偏好和流向
    - 投资机会: 发现表现异常的ETF
    - 风险预警: 识别大幅下跌的ETF
    - 趋势分析: 追踪市场趋势变化

    分析技巧:
    - 涨幅榜: 关注连续上榜的ETF
    - 成交榜: 重视放量上涨的ETF
    - 换手榜: 注意异常活跃的ETF
    - 跌幅榜: 寻找超跌反弹机会
    - 横向比较: 对比同类ETF表现

    投资启示:
    - 排行靠前不代表投资价值
    - 关注排行背后的逻辑
    - 结合基本面分析
    - 注意风险控制
    - 避免盲目跟风

    注意事项:
    - 排行数据仅供参考
    - 短期表现不代表长期趋势
    - 需要结合多方面信息分析
    - 投资决策需要谨慎
    - 注意数据的时效性

    Note:
        - 数据来源于同花顺官方
        - 更新频率为交易日
        - 建议结合其他分析工具使用
        - 投资有风险，入市需谨慎
    """
    if not date:
        date = datetime.now().strftime("%Y%m%d")

    await ctx.info(f"Fetching THS ETF ranking for {date}")

    try:
        ths_data = await fetch_etf_ths_data(ctx, date)

        if ths_data:
            limited_data = ths_data[:limit]
            return {
                "success": True,
                "date": date,
                "source": "同花顺",
                "etfs": limited_data,
                "count": len(limited_data),
                "total_available": len(ths_data),
            }
        else:
            return {
                "success": False,
                "date": date,
                "error": "No THS ETF data available",
            }

    except Exception as e:
        await ctx.error(f"Error fetching THS ETF ranking: {e}")
        return {"success": False, "date": date, "error": str(e)}


# ==================== ETF技术分析工具 ====================


@mcp.tool()
async def calculate_etf_sma(
    ctx: Context, inputs: SmaInput, market_type: ETFMarketType = ETFMarketType.A_STOCK
) -> SmaOutput:
    """
    计算ETF简单移动平均线 (Simple Moving Average)

    SMA是ETF技术分析中最基础和重要的指标，通过平滑价格波动来识别趋势方向。
    由于ETF价格相对稳定，SMA在ETF分析中通常比个股更为有效和稳定。

    ETF SMA特点:
    - 趋势稳定性好: ETF跟踪指数，减少个股噪音
    - 支撑阻力明显: 机构参与度高，技术位效果好
    - 适合中长期: ETF投资者偏向长期持有
    - 与指数关联: SMA走势反映标的指数趋势
    - 流动性保障: 大部分ETF流动性好，价格连续

    计算公式:
    SMA(n) = (P1 + P2 + ... + Pn) / n
    其中P为ETF收盘价，n为计算周期

    Args:
        inputs.symbol (str): ETF代码
        inputs.timeframe (str): 时间框架，通常使用"1d"
        inputs.period (int): SMA计算周期，常用值:
            - 5日线: 超短期趋势，适合日内波段
            - 10日线: 短期趋势，反映近期走势
            - 20日线: 中短期趋势，月度参考线
            - 60日线: 中期趋势，季度参考线
            - 120日线: 中长期趋势，半年参考线
            - 250日线: 长期趋势，年线参考
        inputs.history_len (int): 返回的历史数据长度
        market_type (ETFMarketType): ETF市场类型, A、US、HK

    Returns:
        SmaOutput: SMA计算结果
        - symbol: ETF代码
        - timeframe: 时间框架
        - period: 计算周期
        - sma: SMA值列表，按时间顺序排列
        - error: 错误信息(如果有)

    ETF SMA应用:
    1. 趋势判断:
        - 价格在SMA上方: 多头趋势，可考虑持有或买入
        - 价格在SMA下方: 空头趋势，可考虑减仓或观望
        - 价格与SMA纠缠: 震荡整理，等待方向选择

    2. 支撑阻力:
        - SMA作为动态支撑: 回调不破SMA可加仓
        - SMA作为动态阻力: 反弹受阻SMA应减仓
        - SMA斜率变化: 反映趋势强度变化

    3. 买卖信号:
        - 价格上穿SMA: 潜在买入信号
        - 价格下穿SMA: 潜在卖出信号
        - 多条SMA金叉: 强烈买入信号
        - 多条SMA死叉: 强烈卖出信号

    Example:
        计算上证50ETF的20日均线:
        >>> inputs = SmaInput(
        ...     symbol="510050",
        ...     timeframe="1d",
        ...     period=20,
        ...     history_len=30
        ... )
        >>> result = await calculate_etf_sma(ctx, inputs, ETFMarketType.A_STOCK)
        >>> if result.sma:
        ...     latest_sma = result.sma[-1]
        ...     print(f"上证50ETF 20日均线: ¥{latest_sma:.3f}")
        ...
        ...     # 趋势分析
        ...     if len(result.sma) > 1:
        ...         if result.sma[-1] > result.sma[-2]:
        ...             print("📈 均线上升，趋势向好")
        ...         else:
        ...             print("📉 均线下降，趋势偏弱")
        ...
        ...     # 计算均线斜率（反映趋势强度）
        ...     if len(result.sma) > 5:
        ...         recent_sma = result.sma[-5:]
        ...         slope = (recent_sma[-1] - recent_sma[0]) / recent_sma[0] * 100
        ...         print(f"近5日均线斜率: {slope:+.3f}%")

        多周期SMA分析:
        >>> periods = [5, 20, 60]  # 短中长期均线
        >>> sma_results = {}
        >>>
        >>> for period in periods:
        ...     inputs = SmaInput(
        ...         symbol="159919",  # 沪深300ETF
        ...         period=period,
        ...         history_len=10
        ...     )
        ...     result = await calculate_etf_sma(ctx, inputs)
        ...     if result.sma:
        ...         sma_results[period] = result.sma[-1]
        >>>
        >>> # 均线多头排列判断
        >>> if len(sma_results) == 3:
        ...     sma5, sma20, sma60 = sma_results[5], sma_results[20], sma_results[60]
        ...     if sma5 > sma20 > sma60:
        ...         print("🚀 均线多头排列，趋势强劲")
        ...     elif sma5 < sma20 < sma60:
        ...         print("📉 均线空头排列，趋势偏弱")
        ...     else:
        ...         print("📊 均线混乱，趋势不明")

        与ETF价格对比分析:
        >>> # 假设当前ETF价格
        >>> current_price = 2.850  # 示例价格
        >>> sma20 = result.sma[-1]  # 20日均线
        >>>
        >>> deviation = (current_price - sma20) / sma20 * 100
        >>> print(f"价格偏离20日均线: {deviation:+.2f}%")
        >>>
        >>> if deviation > 5:
        ...     print("💰 价格显著高于均线，可能超涨")
        >>> elif deviation < -5:
        ...     print("💎 价格显著低于均线，可能超跌")
        >>> else:
        ...     print("⚖️ 价格贴近均线，相对合理")

    ETF SMA投资策略:
    1. 均线支撑策略:
        - 在关键均线附近买入
        - 跌破重要均线止损
        - 适用于震荡上涨市场

    2. 均线突破策略:
        - 突破长期均线买入
        - 跌破长期均线卖出
        - 适用于趋势性市场

    3. 均线交叉策略:
        - 短期均线上穿长期均线买入
        - 短期均线下穿长期均线卖出
        - 适合中长期投资

    4. 多重均线系统:
        - 结合多条均线判断
        - 均线排列确认趋势
        - 提高信号可靠性

    ETF特殊考虑:
    - 申购赎回机制影响: 大额申赎可能影响价格
    - 分红除权影响: 使用复权价格计算
    - 跟踪误差考虑: SMA应与标的指数对比
    - 流动性差异: 小众ETF可能出现价格跳跃
    - 市场情绪影响: ETF受整体市场影响较大

    注意事项:
    - SMA是滞后指标，确认趋势而非预测
    - 单一均线信号可能产生假突破
    - 需要结合成交量等其他指标确认
    - 不同市场ETF的均线有效性可能不同
    - 建议使用复权价格进行计算

    Note:
        - ETF均线分析应结合标的指数走势
        - 大盘ETF均线效果通常优于主题ETF
        - 建议多周期均线组合使用
        - 投资决策需要综合多方面因素
    """
    await ctx.info(
        f"Calculating ETF SMA for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )

    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }

    try:
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_etf_single_series_data(
            ctx, inputs.symbol, market_type.value, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient ETF data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)
        valid_sma = _extract_valid_values(sma_values, inputs.history_len)

        if not valid_sma:
            return SmaOutput(
                **output_base,
                error="ETF SMA calculation resulted in insufficient valid data.",
            )

        currency_symbol = (
            "¥"
            if market_type == ETFMarketType.A_STOCK
            else ("HK$" if market_type == ETFMarketType.HK_STOCK else "$")
        )
        await ctx.info(
            f"Calculated ETF SMA for {inputs.symbol}: {len(valid_sma)} values, latest: {currency_symbol}{valid_sma[-1]:.2f}"
        )
        return SmaOutput(**output_base, sma=valid_sma)

    except Exception as e:
        await ctx.error(f"Error in ETF SMA calculation for {inputs.symbol}: {e}")
        return SmaOutput(**output_base, error="ETF SMA calculation error.")


@mcp.tool()
async def calculate_etf_rsi(
    ctx: Context, inputs: RsiInput, market_type: ETFMarketType = ETFMarketType.A_STOCK
) -> RsiOutput:
    """
    计算ETF相对强弱指数 (Relative Strength Index, RSI)

    RSI是衡量ETF价格动量的重要技术指标，通过分析一定周期内价格上涨和下跌的幅度
    来判断ETF是否处于超买或超卖状态。由于ETF具有分散化特征，RSI信号通常比个股更为稳定。

    ETF RSI特点:
    - 波动相对平稳: ETF的分散化特征使RSI波动更加平滑
    - 趋势确认性强: ETF RSI能有效确认标的指数的强弱趋势
    - 超买超卖明显: 机构参与使得RSI的50、70、30关键位更有效
    - 背离信号可靠: ETF价格与RSI的背离往往预示趋势转折
    - 适合波段操作: RSI结合ETF的稳定性，适合中短期波段策略

    计算原理:
    RSI = 100 - 100/(1 + RS)
    其中 RS = 平均上涨幅度 / 平均下跌幅度

    Args:
        inputs.symbol (str): ETF代码
        inputs.timeframe (str): 时间框架，推荐"1d"
        inputs.period (int): RSI计算周期，常用值:
            - 6日RSI: 短期超买超卖判断，灵敏度高
            - 14日RSI: 标准周期，最常用的RSI参数
            - 21日RSI: 中期动量分析，信号相对稳定
            - 28日RSI: 长期动量趋势，适合趋势确认
        inputs.history_len (int): 返回的历史数据长度
        market_type (ETFMarketType): ETF市场类型, A、US、HK

    Returns:
        RsiOutput: RSI计算结果
        - symbol: ETF代码
        - timeframe: 时间框架
        - period: 计算周期
        - rsi: RSI值列表，范围0-100，按时间顺序排列
        - error: 错误信息(如果有)

    RSI区间解读:
    1. 超买区域 (RSI > 70):
        - 强烈超买 (RSI > 80): 短期回调概率大，可考虑减仓
        - 温和超买 (70-80): 关注回调信号，谨慎追涨

    2. 正常区域 (30-70):
        - 强势区间 (50-70): 多头趋势，可持有或逢低加仓
        - 弱势区间 (30-50): 空头趋势，谨慎操作或减仓

    3. 超卖区域 (RSI < 30):
        - 温和超卖 (20-30): 关注反弹信号，可分批建仓
        - 强烈超卖 (RSI < 20): 反弹概率大，积极关注买点

    Example:
        计算沪深300ETF的14日RSI:
        >>> inputs = RsiInput(
        ...     symbol="510300",
        ...     timeframe="1d",
        ...     period=14,
        ...     history_len=20
        ... )
        >>> result = await calculate_etf_rsi(ctx, inputs, ETFMarketType.A_STOCK)
        >>> if result.rsi:
        ...     latest_rsi = result.rsi[-1]
        ...     print(f"沪深300ETF 14日RSI: {latest_rsi:.2f}")
        ...
        ...     # RSI状态判断
        ...     if latest_rsi > 70:
        ...         print("⚠️  RSI超买状态，谨慎追涨")
        ...         if latest_rsi > 80:
        ...             print("🔴 强烈超买，建议减仓")
        ...     elif latest_rsi < 30:
        ...         print("💎 RSI超卖状态，关注买点")
        ...         if latest_rsi < 20:
        ...             print("🟢 强烈超卖，积极关注")
        ...     else:
        ...         strength = "强势" if latest_rsi > 50 else "弱势"
        ...         print(f"📊 RSI正常区间，当前{strength}")
        ...
        ...     # RSI趋势分析
        ...     if len(result.rsi) > 3:
        ...         recent_rsi = result.rsi[-3:]
        ...         if all(recent_rsi[i] > recent_rsi[i-1] for i in range(1, len(recent_rsi))):
        ...             print("📈 RSI连续上升，动量增强")
        ...         elif all(recent_rsi[i] < recent_rsi[i-1] for i in range(1, len(recent_rsi))):
        ...             print("📉 RSI连续下降，动量减弱")

        多周期RSI分析:
        >>> periods = [6, 14, 21]  # 短中长期RSI
        >>> rsi_results = {}
        >>>
        >>> for period in periods:
        ...     inputs = RsiInput(
        ...         symbol="159919",  # 创业板ETF
        ...         period=period,
        ...         history_len=10
        ...     )
        ...     result = await calculate_etf_rsi(ctx, inputs)
        ...     if result.rsi:
        ...         rsi_results[period] = result.rsi[-1]
        >>>
        >>> # 多周期RSI共振分析
        >>> if len(rsi_results) == 3:
        ...     rsi6, rsi14, rsi21 = rsi_results[6], rsi_results[14], rsi_results[21]
        ...     print(f"RSI多周期分析:")
        ...     print(f"6日RSI: {rsi6:.1f}, 14日RSI: {rsi14:.1f}, 21日RSI: {rsi21:.1f}")
        ...
        ...     # 共振信号判断
        ...     if all(rsi > 70 for rsi in rsi_results.values()):
        ...         print("🔴 多周期RSI共振超买，高度警惕")
        ...     elif all(rsi < 30 for rsi in rsi_results.values()):
        ...         print("🟢 多周期RSI共振超卖，重点关注")
        ...     elif all(rsi > 50 for rsi in rsi_results.values()):
        ...         print("📈 多周期RSI强势，趋势向好")

        RSI背离分析示例:
        >>> # 假设ETF价格数据
        >>> prices = [2.800, 2.850, 2.900, 2.880, 2.860]  # 价格创新高
        >>> rsi_values = result.rsi[-5:]  # 对应的RSI值
        >>>
        >>> if len(rsi_values) >= 5:
        ...     # 简单的背离检测
        ...     price_trend = prices[-1] > prices[0]  # 价格趋势
        ...     rsi_trend = rsi_values[-1] > rsi_values[0]  # RSI趋势
        ...
        ...     if price_trend and not rsi_trend:
        ...         print("📉 发现顶背离：价格新高但RSI未创新高")
        ...     elif not price_trend and rsi_trend:
        ...         print("📈 发现底背离：价格新低但RSI未创新低")

    ETF RSI交易策略:
    1. 超买超卖策略:
        - RSI超买时逐步减仓
        - RSI超卖时分批建仓
        - 适合震荡市场

    2. 趋势确认策略:
        - RSI持续在50以上确认多头趋势
        - RSI持续在50以下确认空头趋势
        - 适合趋势性市场

    3. 背离交易策略:
        - 顶背离时准备减仓
        - 底背离时准备加仓
        - 适合趋势转折点

    4. 多周期共振策略:
        - 多个周期RSI同向信号
        - 提高交易胜率
        - 降低假信号概率

    ETF特殊考虑:
    - 指数特征: ETF RSI应与标的指数RSI对比分析
    - 行业轮动: 行业ETF的RSI需结合行业景气度
    - 资金流向: ETF申购赎回可能影响RSI信号
    - 分红影响: 分红除权日前后RSI可能出现跳跃
    - 市场情绪: ETF RSI更多反映市场整体情绪

    应用场景:
    - 买卖时机: 结合RSI超买超卖判断进出场
    - 仓位管理: 根据RSI水平调整仓位大小
    - 风险控制: RSI异常值作为风险预警信号
    - 趋势确认: 利用RSI确认价格趋势有效性
    - 背离捕捉: 发现价格趋势可能的转折点

    注意事项:
    - RSI是滞后指标，确认信号而非预测
    - 强趋势中RSI可能长时间保持极值
    - 需要结合价格形态和成交量确认
    - 不同市场和周期的RSI有效性不同
    - 单一RSI信号可能产生假突破

    Note:
        - ETF的RSI通常比个股更稳定可靠
        - 建议结合移动平均线等趋势指标使用
        - 大盘ETF的RSI效果通常优于主题ETF
        - 投资决策应综合考虑多个技术指标
    """
    await ctx.info(
        f"Calculating ETF RSI for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )

    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }

    try:
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_etf_single_series_data(
            ctx, inputs.symbol, market_type.value, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient ETF data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)
        valid_rsi = _extract_valid_values(rsi_values, inputs.history_len)

        if not valid_rsi:
            return RsiOutput(
                **output_base,
                error="ETF RSI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated ETF RSI for {inputs.symbol}: {len(valid_rsi)} values, latest: {valid_rsi[-1]:.2f}"
        )
        return RsiOutput(**output_base, rsi=valid_rsi)

    except Exception as e:
        await ctx.error(f"Error in ETF RSI calculation for {inputs.symbol}: {e}")
        return RsiOutput(**output_base, error="ETF RSI calculation error.")


@mcp.tool()
async def calculate_etf_macd(
    ctx: Context, inputs: MacdInput, market_type: ETFMarketType = ETFMarketType.A_STOCK
) -> MacdOutput:
    """
    计算ETF MACD指标 (Moving Average Convergence Divergence)

    MACD是ETF技术分析中最重要的趋势跟踪指标之一，通过快慢均线的收敛发散
    来识别趋势变化和交易信号。ETF的稳定特性使得MACD信号通常更加可靠和清晰。

    ETF MACD特点:
    - 趋势信号稳定: ETF价格相对平稳，MACD信号噪音较少
    - 金叉死叉有效: 机构参与度高，使得MACD交叉信号更可靠
    - 背离现象明显: ETF与MACD的背离往往预示重要转折
    - 适合中长期: ETF投资者偏向中长期，MACD契合投资周期
    - 量价配合好: ETF成交量与MACD柱状图配合分析效果佳

    MACD构成:
    1. MACD线: 快速EMA - 慢速EMA，反映趋势方向
    2. 信号线: MACD线的EMA平滑值，产生交易信号
    3. 柱状图: MACD线 - 信号线，反映动量强弱

    Args:
        inputs.symbol (str): ETF代码
        inputs.timeframe (str): 时间框架，建议"1d"或"1w"
        inputs.fast_period (int): 快速EMA周期，常用值:
            - 12: 标准快线周期，适合日线分析
            - 8: 较灵敏设置，适合短期交易
            - 16: 较稳定设置，减少假信号
        inputs.slow_period (int): 慢速EMA周期，常用值:
            - 26: 标准慢线周期，最常用参数
            - 21: 较灵敏设置
            - 30: 较稳定设置
        inputs.signal_period (int): 信号线EMA周期，常用值:
            - 9: 标准信号线周期
            - 7: 较灵敏的信号线
            - 12: 较稳定的信号线
        inputs.history_len (int): 返回的历史数据长度
        market_type (ETFMarketType): ETF市场类型, A、US、HK

    Returns:
        MacdOutput: MACD计算结果
        - symbol: ETF代码
        - timeframe: 时间框架
        - fast_period, slow_period, signal_period: 计算参数
        - macd: MACD线值列表
        - signal: 信号线值列表
        - histogram: 柱状图值列表
        - error: 错误信息(如果有)

    MACD信号解读:
    1. 金叉死叉信号:
        - 金叉: MACD线上穿信号线，买入信号
        - 死叉: MACD线下穿信号线，卖出信号
        - 零轴附近的交叉信号更可靠

    2. 零轴突破:
        - 上穿零轴: 短期趋势转为多头
        - 下穿零轴: 短期趋势转为空头
        - 零轴是多空分界线

    3. 柱状图分析:
        - 柱状图放大: 趋势加速
        - 柱状图缩小: 趋势放缓
        - 柱状图转向: 动量转换先兆

    4. 背离信号:
        - 顶背离: 价格新高但MACD未创新高
        - 底背离: 价格新低但MACD未创新低

    Example:
        计算上证50ETF的MACD:
        >>> inputs = MacdInput(
        ...     symbol="510050",
        ...     timeframe="1d",
        ...     fast_period=12,
        ...     slow_period=26,
        ...     signal_period=9,
        ...     history_len=30
        ... )
        >>> result = await calculate_etf_macd(ctx, inputs, ETFMarketType.A_STOCK)
        >>> if result.macd:
        ...     latest_macd = result.macd[-1]
        ...     latest_signal = result.signal[-1]
        ...     latest_hist = result.histogram[-1]
        ...
        ...     print(f"上证50ETF MACD(12,26,9):")
        ...     print(f"MACD线: {latest_macd:.4f}")
        ...     print(f"信号线: {latest_signal:.4f}")
        ...     print(f"柱状图: {latest_hist:.4f}")
        ...
        ...     # 金叉死叉判断
        ...     if len(result.macd) > 1 and len(result.signal) > 1:
        ...         prev_macd = result.macd[-2]
        ...         prev_signal = result.signal[-2]
        ...
        ...         # 检测金叉
        ...         if prev_macd <= prev_signal and latest_macd > latest_signal:
        ...             print("🟢 MACD金叉信号！可能的买入机会")
        ...         # 检测死叉
        ...         elif prev_macd >= prev_signal and latest_macd < latest_signal:
        ...             print("🔴 MACD死叉信号！可能的卖出机会")
        ...
        ...     # 零轴位置分析
        ...     if latest_macd > 0:
        ...         print("📈 MACD在零轴上方，多头趋势")
        ...     else:
        ...         print("📉 MACD在零轴下方，空头趋势")
        ...
        ...     # 动量分析
        ...     if latest_hist > 0:
        ...         print("⚡ 柱状图为正，上涨动量")
        ...     else:
        ...         print("⚡ 柱状图为负，下跌动量")

        MACD趋势强度分析:
        >>> if len(result.histogram) > 5:
        ...     recent_hist = result.histogram[-5:]
        ...
        ...     # 动量变化分析
        ...     if all(recent_hist[i] > recent_hist[i-1] for i in range(1, len(recent_hist))):
        ...         print("🚀 柱状图连续放大，趋势加速")
        ...     elif all(recent_hist[i] < recent_hist[i-1] for i in range(1, len(recent_hist))):
        ...         print("⚠️  柱状图连续缩小，趋势放缓")
        ...
        ...     # 动量转换预警
        ...     if recent_hist[-1] * recent_hist[-2] < 0:
        ...         print("🔄 柱状图变号，动量可能转换")

        多ETF MACD对比:
        >>> etf_symbols = ["510050", "510300", "159919"]  # 50、300、创业板
        >>> macd_comparison = {}
        >>>
        >>> for symbol in etf_symbols:
        ...     inputs = MacdInput(symbol=symbol, history_len=10)
        ...     result = await calculate_etf_macd(ctx, inputs)
        ...     if result.macd:
        ...         macd_comparison[symbol] = {
        ...             "macd": result.macd[-1],
        ...             "signal": result.signal[-1],
        ...             "histogram": result.histogram[-1]
        ...         }
        >>>
        >>> print("ETF MACD对比分析:")
        >>> for symbol, data in macd_comparison.items():
        ...     status = "多头" if data["macd"] > 0 else "空头"
        ...     momentum = "正向" if data["histogram"] > 0 else "负向"
        ...     print(f"{symbol}: 趋势{status}, 动量{momentum}")

        MACD背离检测示例:
        >>> # 假设价格和MACD历史数据
        >>> if len(result.macd) >= 10:
        ...     # 简化的背离检测逻辑
        ...     recent_macd = result.macd[-5:]
        ...     # 配合价格数据进行背离分析
        ...     print("💡 建议结合价格数据进行背离分析")

    ETF MACD交易策略:
    1. 经典金叉死叉策略:
        - 金叉买入，死叉卖出
        - 零轴附近信号更可靠
        - 适合趋势跟踪

    2. 零轴策略:
        - 上穿零轴买入
        - 下穿零轴卖出
        - 适合趋势转换捕捉

    3. 柱状图策略:
        - 柱状图放大加仓
        - 柱状图缩小减仓
        - 适合动量交易

    4. 背离策略:
        - 顶背离减仓
        - 底背离加仓
        - 适合反转交易

    5. 多周期策略:
        - 日线MACD确定方向
        - 小时线MACD寻找时机
        - 提高操作精确度

    ETF特殊应用:
    - 板块轮动: 行业ETF的MACD比较分析
    - 资金流向: 大盘ETF MACD反映整体资金偏好
    - 风险偏好: 成长vs价值ETF的MACD对比
    - 跨市比较: A股、港股、美股ETF的MACD分析
    - 量化信号: MACD作为量化交易信号源

    参数优化建议:
    - 短线交易: (8, 21, 7) 更灵敏
    - 中线交易: (12, 26, 9) 标准参数
    - 长线交易: (19, 39, 9) 更稳定
    - 周线分析: (5, 13, 5) 适合周线

    注意事项:
    - MACD是趋势跟踪指标，在震荡市效果较差
    - 需要结合价格形态确认信号有效性
    - 假信号可能出现在市场转换期
    - 不同参数设置会影响信号敏感度
    - 建议与其他指标组合使用

    Note:
        - ETF的MACD信号通常比个股更可靠
        - 大盘ETF的MACD可作为市场风向标
        - 建议关注MACD与成交量的配合
        - 跨境ETF需要考虑汇率因素影响
    """
    await ctx.info(
        f"Calculating ETF MACD for {inputs.symbol}, Periods: {inputs.fast_period}/{inputs.slow_period}/{inputs.signal_period}"
    )

    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "fast_period": inputs.fast_period,
        "slow_period": inputs.slow_period,
        "signal_period": inputs.signal_period,
    }

    try:
        required_candles = (
            inputs.slow_period + inputs.signal_period + inputs.history_len + 10
        )
        close_prices = await _fetch_etf_single_series_data(
            ctx, inputs.symbol, market_type.value, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return MacdOutput(
                **output_base, error="Failed to fetch sufficient ETF data for MACD."
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
                error="ETF MACD calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated ETF MACD for {inputs.symbol}: latest MACD: {valid_macd[-1]:.4f}"
        )
        return MacdOutput(
            **output_base, macd=valid_macd, signal=valid_signal, histogram=valid_hist
        )

    except Exception as e:
        await ctx.error(f"Error in ETF MACD calculation for {inputs.symbol}: {e}")
        return MacdOutput(**output_base, error="ETF MACD calculation error.")


@mcp.tool()
async def calculate_etf_bbands(
    ctx: Context,
    inputs: BbandsInput,
    market_type: ETFMarketType = ETFMarketType.A_STOCK,
) -> BbandsOutput:
    """
    计算ETF布林带 (Bollinger Bands)

    布林带是衡量ETF价格波动范围和超买超卖状态的重要技术指标。通过统计学原理
    构建动态的价格通道，为ETF的买卖时机提供客观的参考标准。

    ETF布林带特点:
    - 波动区间明确: ETF价格通常在布林带内运行，突破意义重大
    - 支撑阻力动态: 上下轨作为动态支撑阻力位，指导性较强
    - 收缩扩张有序: 布林带收缩扩张反映ETF波动率变化
    - 回归特性明显: ETF价格偏离布林带后容易回归
    - 趋势确认有效: 布林带方向与ETF趋势高度一致

    布林带构成:
    1. 中轨: n期简单移动平均线，代表价格趋势
    2. 上轨: 中轨 + k倍标准差，代表压力位
    3. 下轨: 中轨 - k倍标准差，代表支撑位
    4. 带宽: 上轨-下轨，反映波动性大小

    Args:
        inputs.symbol (str): ETF代码
        inputs.timeframe (str): 时间框架，建议"1d"
        inputs.period (int): 计算周期，常用值:
            - 10: 短期布林带，敏感度高，适合短线
            - 20: 标准布林带，最常用参数，适合中线
            - 26: 长期布林带，稳定性好，适合长线
        inputs.nbdevup (float): 上轨标准差倍数，常用值:
            - 1.5: 较窄通道，信号频繁
            - 2.0: 标准设置，最常用参数
            - 2.5: 较宽通道，信号稳定
        inputs.nbdevdn (float): 下轨标准差倍数，通常与上轨相同
        inputs.matype (int): 移动平均类型，0为简单移动平均
        inputs.history_len (int): 返回的历史数据长度
        market_type (ETFMarketType): ETF市场类型, A、US、HK

    Returns:
        BbandsOutput: 布林带计算结果
        - symbol: ETF代码
        - timeframe: 时间框架
        - period, nbdevup, nbdevdn, matype: 计算参数
        - upper_band: 上轨值列表
        - middle_band: 中轨值列表
        - lower_band: 下轨值列表
        - error: 错误信息(如果有)

    布林带信号解读:
    1. 价格位置信号:
        - 价格接近上轨: 可能超买，关注卖出时机
        - 价格接近下轨: 可能超卖，关注买入时机
        - 价格在中轨附近: 震荡整理，等待方向选择

    2. 突破信号:
        - 突破上轨: 强势上涨，可能持续
        - 跌破下轨: 弱势下跌，需要警惕
        - 回归中轨: 价格修复，趋于正常

    3. 收缩扩张信号:
        - 布林带收缩: 波动率降低，酝酿突破
        - 布林带扩张: 波动率增加，趋势确立
        - 收缩后扩张: 往往伴随重要行情

    Example:
        计算沪深300ETF的布林带:
        >>> inputs = BbandsInput(
        ...     symbol="510300",
        ...     timeframe="1d",
        ...     period=20,
        ...     nbdevup=2.0,
        ...     nbdevdn=2.0,
        ...     history_len=30
        ... )
        >>> result = await calculate_etf_bbands(ctx, inputs, ETFMarketType.A_STOCK)
        >>> if result.upper_band:
        ...     latest_upper = result.upper_band[-1]
        ...     latest_middle = result.middle_band[-1]
        ...     latest_lower = result.lower_band[-1]
        ...
        ...     print(f"沪深300ETF 布林带(20,2):")
        ...     print(f"上轨: ¥{latest_upper:.3f}")
        ...     print(f"中轨: ¥{latest_middle:.3f}")
        ...     print(f"下轨: ¥{latest_lower:.3f}")
        ...
        ...     # 布林带宽度分析
        ...     band_width = latest_upper - latest_lower
        ...     band_width_ratio = band_width / latest_middle * 100
        ...     print(f"带宽: ¥{band_width:.3f} ({band_width_ratio:.2f}%)")
        ...
        ...     # 假设当前ETF价格
        ...     current_price = 4.250  # 示例价格
        ...
        ...     # 价格位置分析
        ...     if current_price > latest_upper:
        ...         print("🔴 价格突破上轨，强势但警惕回调")
        ...     elif current_price < latest_lower:
        ...         print("🟢 价格跌破下轨，弱势但关注反弹")
        ...     else:
        ...         # 计算价格在布林带中的位置
        ...         position = (current_price - latest_lower) / (latest_upper - latest_lower)
        ...         if position > 0.8:
        ...             print("⚠️  价格靠近上轨，可能超买")
        ...         elif position < 0.2:
        ...             print("💎 价格靠近下轨，可能超卖")
        ...         else:
        ...             print(f"📊 价格在布林带中部 (位置: {position:.1%})")

        布林带收缩扩张分析:
        >>> if len(result.upper_band) > 10:
        ...     # 计算最近10期的平均带宽
        ...     recent_widths = []
        ...     for i in range(-10, 0):
        ...         width = result.upper_band[i] - result.lower_band[i]
        ...         recent_widths.append(width)
        ...
        ...     current_width = recent_widths[-1]
        ...     avg_width = sum(recent_widths[:-1]) / len(recent_widths[:-1])
        ...
        ...     width_change = (current_width - avg_width) / avg_width * 100
        ...
        ...     if width_change > 20:
        ...         print("📈 布林带快速扩张，波动率上升")
        ...     elif width_change < -20:
        ...         print("📉 布林带快速收缩，波动率下降")
        ...     else:
        ...         print("⚖️  布林带宽度稳定")

        布林带趋势分析:
        >>> if len(result.middle_band) > 5:
        ...     recent_middle = result.middle_band[-5:]
        ...
        ...     # 中轨趋势判断
        ...     if all(recent_middle[i] > recent_middle[i-1] for i in range(1, len(recent_middle))):
        ...         print("📈 布林带中轨上升，趋势向好")
        ...     elif all(recent_middle[i] < recent_middle[i-1] for i in range(1, len(recent_middle))):
        ...         print("📉 布林带中轨下降，趋势偏弱")
        ...     else:
        ...         print("📊 布林带中轨震荡，无明确趋势")

        多周期布林带分析:
        >>> periods = [10, 20, 30]  # 短中长期布林带
        >>> for period in periods:
        ...     inputs = BbandsInput(symbol="510300", period=period, history_len=5)
        ...     result = await calculate_etf_bbands(ctx, inputs)
        ...     if result.upper_band:
        ...         width = result.upper_band[-1] - result.lower_band[-1]
        ...         print(f"{period}期布林带宽度: ¥{width:.3f}")

    ETF布林带交易策略:
    1. 布林带边界策略:
        - 触及上轨时减仓
        - 触及下轨时加仓
        - 适合震荡市场

    2. 布林带突破策略:
        - 突破上轨后持有
        - 跌破下轨后观望
        - 适合趋势市场

    3. 布林带压缩策略:
        - 带宽收缩时准备
        - 突破方向时跟进
        - 适合突破交易

    4. 布林带回归策略:
        - 价格偏离后回归
        - 利用均值回归特性
        - 适合稳健投资

    5. 多周期验证策略:
        - 多个周期布林带确认
        - 提高信号可靠性
        - 降低假突破风险

    ETF特殊应用:
    - 波动率交易: 利用布林带宽度变化
    - 套利机会: ETF价格偏离带来套利空间
    - 风险管理: 布林带作为止损止盈参考
    - 仓位配置: 根据价格在布林带位置调整仓位
    - 择时决策: 结合布林带判断买卖时机

    参数调整建议:
    - 高波动ETF: 增大标准差倍数(2.5-3.0)
    - 低波动ETF: 减小标准差倍数(1.5-2.0)
    - 短线交易: 缩短周期(10-15期)
    - 长线投资: 延长周期(25-30期)

    注意事项:
    - 布林带在强趋势中可能持续贴边运行
    - 需要结合成交量确认信号有效性
    - 不同市场环境下布林带效果不同
    - 参数设置需要根据ETF特性调整
    - 建议与其他技术指标组合使用

    Note:
        - ETF的布林带信号通常比个股更稳定
        - 大盘ETF的布林带可反映市场整体波动
        - 行业ETF需要考虑行业特有的波动特征
        - 跨境ETF的布林带会受到汇率影响
    """
    await ctx.info(
        f"Calculating ETF Bollinger Bands for {inputs.symbol}, Period: {inputs.period}"
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
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_etf_single_series_data(
            ctx, inputs.symbol, market_type.value, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return BbandsOutput(
                **output_base,
                error="Failed to fetch sufficient ETF data for Bollinger Bands.",
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
                error="ETF Bollinger Bands calculation resulted in insufficient valid data.",
            )

        currency_symbol = (
            "¥"
            if market_type == ETFMarketType.A_STOCK
            else ("HK$" if market_type == ETFMarketType.HK_STOCK else "$")
        )
        await ctx.info(
            f"Calculated ETF Bollinger Bands for {inputs.symbol}: latest upper: {currency_symbol}{valid_upper[-1]:.2f}"
        )
        return BbandsOutput(
            **output_base,
            upper_band=valid_upper,
            middle_band=valid_middle,
            lower_band=valid_lower,
        )

    except Exception as e:
        await ctx.error(
            f"Error in ETF Bollinger Bands calculation for {inputs.symbol}: {e}"
        )
        return BbandsOutput(
            **output_base, error="ETF Bollinger Bands calculation error."
        )


@mcp.tool()
async def calculate_etf_atr(
    ctx: Context, inputs: AtrInput, market_type: ETFMarketType = ETFMarketType.A_STOCK
) -> AtrOutput:
    """
    计算ETF平均真实波幅 (Average True Range, ATR)

    ATR是衡量ETF价格波动性的重要技术指标，通过计算一定周期内的真实波幅平均值
    来量化ETF的价格波动程度。ETF的分散化特征使得ATR能更好地反映市场整体波动。

    ETF ATR特点:
    - 波动性量化: 客观衡量ETF价格波动强度
    - 趋势独立: ATR不判断方向，只测量波动幅度
    - 相对稳定: ETF的ATR通常比个股更稳定
    - 风险度量: 有效反映ETF的投资风险水平
    - 策略指导: 为止损位和仓位管理提供依据

    真实波幅计算:
    TR = max(|高-低|, |高-前收|, |低-前收|)
    ATR = TR的n期简单移动平均

    Args:
        inputs.symbol (str): ETF代码
        inputs.timeframe (str): 时间框架，建议"1d"
        inputs.period (int): ATR计算周期，常用值:
            - 7: 超短期波动，适合日内交易
            - 14: 标准周期，最常用参数
            - 21: 中期波动，适合波段交易
            - 28: 长期波动，适合趋势跟踪
        inputs.history_len (int): 返回的历史数据长度
        market_type (ETFMarketType): ETF市场类型, A、US、HK

    Returns:
        AtrOutput: ATR计算结果
        - symbol: ETF代码
        - timeframe: 时间框架
        - period: 计算周期
        - atr: ATR值列表，表示平均波动幅度
        - error: 错误信息(如果有)

    ATR应用解读:
    1. 波动性分析:
        - ATR上升: 波动性增加，市场不确定性加大
        - ATR下降: 波动性减少，市场趋于稳定
        - ATR突然放大: 可能有重要事件影响

    2. 风险度量:
        - 高ATR: 高风险高收益，适合激进投资者
        - 低ATR: 低风险低收益，适合稳健投资者
        - ATR比较: 不同ETF风险水平对比

    3. 交易策略:
        - 止损设置: 以ATR倍数设置止损距离
        - 仓位控制: 根据ATR调整仓位大小
        - 入场时机: ATR低位时介入风险较小

    Example:
        计算创业板ETF的ATR:
        >>> inputs = AtrInput(
        ...     symbol="159915",
        ...     timeframe="1d",
        ...     period=14,
        ...     history_len=30
        ... )
        >>> result = await calculate_etf_atr(ctx, inputs, ETFMarketType.A_STOCK)
        >>> if result.atr:
        ...     latest_atr = result.atr[-1]
        ...     print(f"创业板ETF 14日ATR: ¥{latest_atr:.4f}")
        ...
        ...     # ATR趋势分析
        ...     if len(result.atr) > 5:
        ...         recent_atr = result.atr[-5:]
        ...         atr_trend = (recent_atr[-1] - recent_atr[0]) / recent_atr[0] * 100
        ...
        ...         if atr_trend > 20:
        ...             print("📈 ATR显著上升，波动性增加")
        ...         elif atr_trend < -20:
        ...             print("📉 ATR显著下降，波动性减少")
        ...         else:
        ...             print("⚖️  ATR相对稳定")
        ...
        ...     # 假设当前ETF价格
        ...     current_price = 2.350  # 示例价格
        ...
        ...     # 止损位建议 (以2倍ATR为例)
        ...     stop_loss_long = current_price - 2 * latest_atr
        ...     stop_loss_short = current_price + 2 * latest_atr
        ...
        ...     print(f"基于ATR的止损位建议:")
        ...     print(f"做多止损: ¥{stop_loss_long:.3f}")
        ...     print(f"做空止损: ¥{stop_loss_short:.3f}")
        ...
        ...     # ATR相对价格比例
        ...     atr_percentage = latest_atr / current_price * 100
        ...     print(f"ATR占价格比例: {atr_percentage:.2f}%")
        ...
        ...     # 波动性等级判断
        ...     if atr_percentage > 3:
        ...         print("🔴 高波动性ETF，风险较大")
        ...     elif atr_percentage < 1:
        ...         print("🟢 低波动性ETF，相对稳健")
        ...     else:
        ...         print("🟡 中等波动性ETF")

        多ETF波动性对比:
        >>> etf_list = [
        ...     ("510050", "上证50ETF"),
        ...     ("510300", "沪深300ETF"),
        ...     ("159915", "创业板ETF")
        ... ]
        >>>
        >>> volatility_comparison = []
        >>> for symbol, name in etf_list:
        ...     inputs = AtrInput(symbol=symbol, period=14, history_len=10)
        ...     result = await calculate_etf_atr(ctx, inputs)
        ...     if result.atr:
        ...         atr_value = result.atr[-1]
        ...         volatility_comparison.append({
        ...             "name": name,
        ...             "symbol": symbol,
        ...             "atr": atr_value
        ...         })
        >>>
        >>> # 按波动性排序
        >>> volatility_comparison.sort(key=lambda x: x["atr"], reverse=True)
        >>> print("ETF波动性排行 (ATR从高到低):")
        >>> for i, etf in enumerate(volatility_comparison, 1):
        ...     print(f"{i}. {etf['name']}: ATR {etf['atr']:.4f}")

        仓位控制策略:
        >>> # 假设总资金和风险承受度
        >>> total_capital = 100000  # 10万元
        >>> risk_per_trade = 0.02   # 每笔交易风险2%
        >>>
        >>> if latest_atr > 0:
        ...     # 基于ATR的仓位计算
        ...     risk_amount = total_capital * risk_per_trade
        ...     stop_distance = 2 * latest_atr  # 2倍ATR止损
        ...
        ...     if stop_distance > 0:
        ...         position_size = risk_amount / stop_distance
        ...         position_ratio = (position_size * current_price) / total_capital
        ...
        ...         print(f"基于ATR的仓位管理:")
        ...         print(f"建议仓位: {position_size:.0f}份")
        ...         print(f"资金占用比例: {position_ratio:.1%}")

        ATR突破检测:
        >>> if len(result.atr) > 20:
        ...     # 计算ATR的20期均值
        ...     atr_history = result.atr[-20:]
        ...     atr_average = sum(atr_history[:-1]) / len(atr_history[:-1])
        ...     current_atr = atr_history[-1]
        ...
        ...     atr_deviation = (current_atr - atr_average) / atr_average * 100
        ...
        ...     if atr_deviation > 50:
        ...         print("⚠️  ATR异常放大，市场可能有重大变化")
        ...     elif atr_deviation < -30:
        ...         print("💤 ATR异常缩小，市场可能过于平静")

    ETF ATR交易策略:
    1. 动态止损策略:
        - 以1.5-3倍ATR设置止损
        - ATR大时止损宽，ATR小时止损紧
        - 根据市场波动调整止损距离

    2. 仓位管理策略:
        - 高ATR时减少仓位
        - 低ATR时可适当增加仓位
        - 基于ATR计算合理仓位大小

    3. 入场时机策略:
        - ATR低位时入场成本较低
        - ATR高位时谨慎入场
        - 等待ATR回归正常水平

    4. 波动性交易策略:
        - ATR扩张时做趋势
        - ATR收缩时做震荡
        - 利用波动性周期性变化

    5. 风险配对策略:
        - 高ATR ETF配低ATR ETF
        - 平衡投资组合整体风险
        - 优化风险收益比

    ETF特殊应用:
    - 风险预算: 根据ATR分配资金到不同ETF
    - 对冲策略: 利用ATR差异进行风险对冲
    - 择时交易: ATR低点作为入场时机参考
    - 组合优化: ATR作为投资组合风险度量
    - 压力测试: 极端ATR情况下的组合表现

    行业差异分析:
    - 科技ETF: 通常ATR较高，波动性大
    - 消费ETF: ATR相对稳定，波动适中
    - 金融ETF: ATR与市场环境关联度高
    - 债券ETF: ATR较低，波动性小

    市场环境影响:
    - 牛市: ATR通常较低且稳定
    - 熊市: ATR往往较高且波动
    - 震荡市: ATR呈现周期性变化
    - 危机期: ATR急剧放大

    注意事项:
    - ATR只反映波动幅度，不预测方向
    - 极端市场情况下ATR可能失效
    - 需要结合其他指标综合判断
    - 不同周期的ATR含义不同
    - 历史ATR不能完全预测未来波动

    Note:
        - ETF的ATR通常比个股更稳定可靠
        - 建议定期检查和调整ATR参数
        - ATR应作为风险管理工具而非交易信号
        - 不同市场的ETF需要分别分析ATR特征
    """
    await ctx.info(f"Calculating ETF ATR for {inputs.symbol}, Period: {inputs.period}")

    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }

    try:
        required_candles = inputs.period + inputs.history_len - 1

        price_data = await _fetch_etf_multi_series_data(
            ctx,
            inputs.symbol,
            market_type.value,
            required_candles,
            ["high", "low", "close"],
        )

        if not price_data:
            return AtrOutput(
                **output_base, error="Failed to fetch ETF HLC data for ATR."
            )

        high_prices = price_data["high"]
        low_prices = price_data["low"]
        close_prices = price_data["close"]

        if len(high_prices) < required_candles:
            return AtrOutput(
                **output_base,
                error=f"Insufficient ETF HLC data points for ATR. Need at least {required_candles}.",
            )

        atr_values = talib.ATR(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )
        valid_atr = _extract_valid_values(atr_values, inputs.history_len)

        if not valid_atr:
            return AtrOutput(
                **output_base,
                error="ETF ATR calculation resulted in insufficient valid data.",
            )

        currency_symbol = (
            "¥"
            if market_type == ETFMarketType.A_STOCK
            else ("HK$" if market_type == ETFMarketType.HK_STOCK else "$")
        )
        await ctx.info(
            f"Calculated ETF ATR for {inputs.symbol}: {len(valid_atr)} values, latest: {currency_symbol}{valid_atr[-1]:.2f}"
        )
        return AtrOutput(**output_base, atr=valid_atr)

    except Exception as e:
        await ctx.error(f"Error in ETF ATR calculation for {inputs.symbol}: {e}")
        return AtrOutput(**output_base, error="ETF ATR calculation error.")


# ==================== ETF特有分析工具 ====================


@mcp.tool()
async def get_etf_discount_analysis(
    ctx: Context, inputs: ETFDiscountAnalysisInput
) -> dict:
    """
    获取ETF折价率分析 (仅支持A股ETF)

    Args:
        inputs.symbol: ETF代码
        inputs.market_type: 市场类型
        inputs.days: 分析天数
    """
    await ctx.info(
        f"Analyzing ETF discount rate for {inputs.symbol} over {inputs.days} days"
    )

    try:
        if inputs.market_type != ETFMarketType.A_STOCK:
            return {
                "success": False,
                "error": "Discount analysis only available for A-stock ETFs",
            }

        # 获取ETF实时数据
        spot_data = await fetch_etf_spot_data(
            ctx, [inputs.symbol], inputs.market_type.value
        )

        if not spot_data or len(spot_data) == 0:
            return {"success": False, "error": "No ETF data available"}

        etf_info = spot_data[0]

        # 获取历史数据
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=inputs.days + 10)).strftime(
            "%Y%m%d"
        )

        hist_data = await fetch_etf_hist_data(
            ctx, inputs.symbol, inputs.market_type.value, "daily", start_date, end_date
        )

        analysis_result = {
            "symbol": inputs.symbol,
            "analysis_date": datetime.now().isoformat(),
            "period_days": inputs.days,
        }

        # 当前折价率分析
        if "nav" in etf_info and "price" in etf_info:
            nav = etf_info["nav"]
            price = etf_info["price"]
            if nav > 0:
                discount_rate = ((price - nav) / nav) * 100
                analysis_result["current_discount"] = {
                    "nav": nav,
                    "market_price": price,
                    "discount_rate": discount_rate,
                    "status": "折价" if discount_rate < 0 else "溢价",
                    "analysis": f"当前{'折价' if discount_rate < 0 else '溢价'} {abs(discount_rate):.2f}%",
                }

        # 历史表现分析
        if hist_data and len(hist_data) > 1:
            prices = [item["close"] for item in hist_data[-inputs.days :]]
            volumes = [item["volume"] for item in hist_data[-inputs.days :]]

            analysis_result["historical_performance"] = {
                "start_price": prices[0],
                "end_price": prices[-1],
                "total_return": ((prices[-1] - prices[0]) / prices[0]) * 100,
                "max_price": max(prices),
                "min_price": min(prices),
                "avg_volume": sum(volumes) / len(volumes),
                "volatility": np.std(prices) / np.mean(prices) * 100,
            }

        return {"success": True, "analysis": analysis_result}

    except Exception as e:
        await ctx.error(f"Error in ETF discount analysis: {e}")
        return {"success": False, "symbol": inputs.symbol, "error": str(e)}


@mcp.tool()
async def compare_etf_performance(ctx: Context, inputs: ETFComparisonInput) -> dict:
    """
    比较多个ETF的表现

    Args:
        inputs.symbols: ETF代码列表
        inputs.market_type: 市场类型
        inputs.period_days: 比较周期天数
    """
    await ctx.info(
        f"Comparing ETF performance for {len(inputs.symbols)} ETFs over {inputs.period_days} days"
    )

    try:
        comparison_results = []

        for symbol in inputs.symbols:
            try:
                # 获取历史数据
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (
                    datetime.now() - timedelta(days=inputs.period_days + 10)
                ).strftime("%Y%m%d")

                hist_data = await fetch_etf_hist_data(
                    ctx, symbol, inputs.market_type.value, "daily", start_date, end_date
                )

                if hist_data and len(hist_data) > 1:
                    prices = [
                        item["close"] for item in hist_data[-inputs.period_days :]
                    ]
                    volumes = [
                        item["volume"] for item in hist_data[-inputs.period_days :]
                    ]

                    if len(prices) > 1:
                        total_return = ((prices[-1] - prices[0]) / prices[0]) * 100
                        returns = [
                            (prices[i] - prices[i - 1]) / prices[i - 1]
                            for i in range(1, len(prices))
                        ]
                        volatility = np.std(returns) * np.sqrt(252) * 100  # 年化波动率

                        etf_result = {
                            "symbol": symbol,
                            "start_price": prices[0],
                            "end_price": prices[-1],
                            "total_return": total_return,
                            "volatility_annualized": volatility,
                            "max_price": max(prices),
                            "min_price": min(prices),
                            "avg_volume": sum(volumes) / len(volumes),
                            "sharpe_ratio": total_return / volatility
                            if volatility > 0
                            else 0,
                        }
                        comparison_results.append(etf_result)

            except Exception as e:
                await ctx.warning(f"Error processing ETF {symbol}: {e}")
                continue

        if comparison_results:
            # 排序结果
            comparison_results.sort(key=lambda x: x["total_return"], reverse=True)

            return {
                "success": True,
                "comparison_period": inputs.period_days,
                "market_type": inputs.market_type.value,
                "etf_count": len(comparison_results),
                "results": comparison_results,
                "analysis_date": datetime.now().isoformat(),
            }
        else:
            return {"success": False, "error": "No valid ETF data for comparison"}

    except Exception as e:
        await ctx.error(f"Error in ETF performance comparison: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def screen_etf_by_criteria(ctx: Context, inputs: ETFScreenInput) -> dict:
    """
    根据条件筛选ETF

    Args:
        inputs.market_type: 市场类型
        inputs.min_volume: 最小成交量
        inputs.max_discount_rate: 最大折价率 (仅A股)
        inputs.min_nav: 最小净值
        inputs.category: ETF分类
        inputs.limit: 结果数量限制
    """
    await ctx.info(f"Screening ETFs with criteria in {inputs.market_type.value} market")

    try:
        # 获取ETF列表
        if inputs.category:
            etf_list = await fetch_etf_category_data(ctx, inputs.category)
        else:
            etf_list = await fetch_etf_spot_data(ctx, market=inputs.market_type.value)

        if not etf_list:
            return {"success": False, "error": "No ETF data available for screening"}

        filtered_etfs = []

        for etf in etf_list:
            # 应用筛选条件
            if inputs.min_volume and etf.get("volume", 0) < inputs.min_volume:
                continue

            if inputs.min_nav and etf.get("nav", 0) < inputs.min_nav:
                continue

            # A股ETF折价率筛选
            if (
                inputs.max_discount_rate is not None
                and inputs.market_type == ETFMarketType.A_STOCK
                and "nav" in etf
                and "price" in etf
            ):
                nav = etf["nav"]
                price = etf["price"]
                if nav > 0:
                    discount_rate = ((price - nav) / nav) * 100
                    if discount_rate > inputs.max_discount_rate:
                        continue

            filtered_etfs.append(etf)

        # 限制结果数量
        filtered_etfs = filtered_etfs[: inputs.limit]

        return {
            "success": True,
            "market_type": inputs.market_type.value,
            "criteria": {
                "min_volume": inputs.min_volume,
                "max_discount_rate": inputs.max_discount_rate,
                "min_nav": inputs.min_nav,
                "category": inputs.category,
            },
            "results": filtered_etfs,
            "count": len(filtered_etfs),
            "screening_date": datetime.now().isoformat(),
        }

    except Exception as e:
        await ctx.error(f"Error in ETF screening: {e}")
        return {"success": False, "error": str(e)}


# ==================== ETF综合分析报告 ====================


@mcp.tool()
async def generate_etf_comprehensive_report(
    ctx: Context,
    inputs: ComprehensiveAnalysisInput,
    market_type: ETFMarketType = ETFMarketType.A_STOCK,
) -> ComprehensiveAnalysisOutput:
    """
    生成ETF综合技术分析报告

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
    """
    await ctx.info(
        f"Generating ETF comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
    )

    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # 确定要运行的指标
    default_indicators = ["SMA", "RSI", "MACD", "BBANDS"]
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
            sma_output = await calculate_etf_sma.run(
                {"ctx": ctx, "inputs": sma_input, "market_type": market_type}
            )
            indicator_results_structured["sma"] = json.loads(
                sma_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["sma"]["sma"] is not None
                and len(indicator_results_structured["sma"]["sma"]) > 0
            ):
                latest_sma = indicator_results_structured["sma"]["sma"][-1]
                currency_symbol = (
                    "¥"
                    if market_type == ETFMarketType.A_STOCK
                    else ("HK$" if market_type == ETFMarketType.HK_STOCK else "$")
                )
                report_sections.append(
                    f"- ETF SMA({indicator_results_structured['sma']['period']}): {currency_symbol}{latest_sma:.2f} (最新值)"
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
                    f"- ETF SMA: 错误 - {indicator_results_structured['sma']['error']}"
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
            rsi_output = await calculate_etf_rsi.run(
                {"ctx": ctx, "inputs": rsi_input, "market_type": market_type}
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
                    f"- ETF RSI({indicator_results_structured['rsi']['period']}): {latest_rsi:.2f}"
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
                    f"- ETF RSI: 错误 - {indicator_results_structured['rsi']['error']}"
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
            macd_output = await calculate_etf_macd.run(
                {"ctx": ctx, "inputs": macd_input, "market_type": market_type}
            )
            indicator_results_structured["macd"] = json.loads(
                macd_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["macd"]["macd"] is not None
                and len(indicator_results_structured["macd"]["macd"]) > 0
                and indicator_results_structured["macd"]["signal"] is not None
                and len(indicator_results_structured["macd"]["signal"]) > 0
            ):
                latest_macd = indicator_results_structured["macd"]["macd"][-1]
                latest_signal = indicator_results_structured["macd"]["signal"][-1]
                latest_hist = indicator_results_structured["macd"]["histogram"][-1]

                report_sections.append(
                    f"- ETF MACD({indicator_results_structured['macd']['fast_period']},{indicator_results_structured['macd']['slow_period']},{indicator_results_structured['macd']['signal_period']}): "
                    f"MACD: {latest_macd:.4f}, 信号线: {latest_signal:.4f}, 柱状图: {latest_hist:.4f}"
                )

                if latest_hist > 0 and latest_macd > latest_signal:
                    report_sections.append("  - 注意: MACD柱状图为正，可能有看涨动量")
                elif latest_hist < 0 and latest_macd < latest_signal:
                    report_sections.append("  - 注意: MACD柱状图为负，可能有看跌动量")
            elif indicator_results_structured["macd"]["error"]:
                report_sections.append(
                    f"- ETF MACD: 错误 - {indicator_results_structured['macd']['error']}"
                )

        # Bollinger Bands分析
        if "BBANDS" in indicators_to_run:
            bbands_input = BbandsInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.bbands_period or settings.DEFAULT_BBANDS_PERIOD,
            )
            bbands_output = await calculate_etf_bbands.run(
                {"ctx": ctx, "inputs": bbands_input, "market_type": market_type}
            )
            indicator_results_structured["bbands"] = json.loads(
                bbands_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["bbands"]["upper_band"] is not None
                and len(indicator_results_structured["bbands"]["upper_band"]) > 0
            ):
                latest_upper = indicator_results_structured["bbands"]["upper_band"][-1]
                latest_middle = indicator_results_structured["bbands"]["middle_band"][
                    -1
                ]
                latest_lower = indicator_results_structured["bbands"]["lower_band"][-1]

                currency_symbol = (
                    "¥"
                    if market_type == ETFMarketType.A_STOCK
                    else ("HK$" if market_type == ETFMarketType.HK_STOCK else "$")
                )
                report_sections.append(
                    f"- ETF布林带({indicator_results_structured['bbands']['period']}): "
                    f"上轨: {currency_symbol}{latest_upper:.2f}, 中轨: {currency_symbol}{latest_middle:.2f}, 下轨: {currency_symbol}{latest_lower:.2f}"
                )
                report_sections.append(
                    f"  - 带宽: {currency_symbol}{latest_upper - latest_lower:.2f}"
                )
            elif indicator_results_structured["bbands"]["error"]:
                report_sections.append(
                    f"- ETF布林带: 错误 - {indicator_results_structured['bbands']['error']}"
                )

        # ETF特有分析 (仅A股)
        if market_type == ETFMarketType.A_STOCK:
            try:
                discount_input = ETFDiscountAnalysisInput(
                    symbol=inputs.symbol, market_type=market_type, days=30
                )
                discount_result = await get_etf_discount_analysis(ctx, discount_input)

                if discount_result.get("success") and "analysis" in discount_result:
                    analysis = discount_result["analysis"]
                    if "current_discount" in analysis:
                        discount_info = analysis["current_discount"]
                        report_sections.append(
                            f"- ETF折价率分析: {discount_info['analysis']}"
                        )
                        report_sections.append(
                            f"  - 净值: ¥{discount_info['nav']:.3f}, 市价: ¥{discount_info['market_price']:.3f}"
                        )

            except Exception as e:
                await ctx.warning(f"ETF discount analysis failed: {e}")

        # 合成报告
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base, error="无法计算任何指标数据"
            )

        market_name = {"A": "A股", "HK": "港股", "US": "美股"}[market_type.value]
        report_title = f"{market_name}ETF技术分析报告 - {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} 个数据点:\n"
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
            f"Successfully generated ETF comprehensive report for {inputs.symbol}"
        )
        return ComprehensiveAnalysisOutput(
            **output_base,
            report_text=report_text,
            structured_data=indicator_results_structured,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(f"Error in ETF comprehensive report for {inputs.symbol}: {e}")
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"ETF综合分析报告生成错误: {str(e)}"
        )
