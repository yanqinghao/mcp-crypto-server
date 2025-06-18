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

    Args:
        ctx: FastMCP上下文对象
        inputs: ETF K线输入参数，包含symbol（ETF代码）、timeframe（时间框架）、limit（数据条数）、market_type（市场类型，默认A）、adjust（复权类型，默认qfq）

    Returns:
        CandlesOutput: K线数据输出对象，包含symbol、timeframe、candles列表、count和可能的error信息
    """
    await ctx.info(
        f"Fetching ETF candles for {inputs.symbol} ({inputs.timeframe}) in {inputs.market_type.value} market"
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
                timeframe=inputs.timeframe,
                candles=candles,
                count=len(candles),
            )
        else:
            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                error=f"No ETF candle data available for {inputs.symbol}",
            )

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(f"Error fetching ETF candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe,
            error=f"ETF data error: {str(e)}",
        )


@mcp.tool()
async def get_etf_price(ctx: Context, inputs: ETFPriceInput) -> PriceOutput:
    """
    获取ETF当前价格

    Args:
        ctx: FastMCP上下文对象
        inputs: ETF价格输入参数，包含symbol（ETF代码）、market_type（市场类型，默认A）

    Returns:
        PriceOutput: 价格输出对象，包含symbol、price、timestamp和可能的error信息
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

    Args:
        ctx: FastMCP上下文对象
        inputs: ETF行情输入参数，包含symbol（ETF代码）、market_type（市场类型，默认A）

    Returns:
        TickerOutput: 详细行情输出对象，包含symbol、last、open、high、low、close、volume、change、percentage、timestamp和可能的error信息
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

    Args:
        ctx: FastMCP上下文对象
        symbol: A股ETF代码（6位数字）
        period: 分时周期，默认1分钟
        market_type: 市场类型，默认A（仅支持A股）

    Returns:
        dict: 包含success状态、symbol、period、data分时数据列表、count数量或error错误信息的字典
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

    Args:
        ctx: FastMCP上下文对象
        inputs: ETF搜索输入参数，包含keyword（搜索关键词）、market_type（市场类型，默认A）、limit（结果数量，默认10）

    Returns:
        dict: 包含success状态、query查询词、market_type市场类型、asset_type资产类型、results结果列表、count结果数量或error错误信息的字典
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

    Args:
        ctx: FastMCP上下文对象
        inputs: ETF分类输入参数，包含category（分类类型，默认ETF基金）、limit（结果数量，默认50）

    Returns:
        dict: 包含success状态、category分类名称、etfs ETF列表、count数量、total_available总数量或error错误信息的字典
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

    Args:
        ctx: FastMCP上下文对象
        date: 查询日期（YYYYMMDD格式），默认None（当前日期）
        limit: 结果数量限制，默认50

    Returns:
        dict: 包含success状态、date排行日期、source数据源、etfs ETF排行列表、count数量、total_available总数量或error错误信息的字典
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
    计算ETF简单移动平均线（SMA）

    Args:
        ctx: FastMCP上下文对象
        inputs: SMA输入参数，包含symbol（ETF代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为20）、history_len（历史数据长度，默认5）
        market_type: ETF市场类型，默认A

    Returns:
        SmaOutput: SMA输出对象，包含symbol、timeframe、period、sma指标值或error错误信息
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
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")

        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_etf_single_series_data(
            ctx, inputs.symbol, period, market_type.value, required_candles, "close"
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
    计算ETF相对强弱指数（RSI）

    Args:
        ctx: FastMCP上下文对象
        inputs: RSI输入参数，包含symbol（ETF代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为14）、history_len（历史数据长度，默认5）
        market_type: ETF市场类型，默认A

    Returns:
        RsiOutput: RSI输出对象，包含symbol、timeframe、period、rsi指标值或error错误信息
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
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_etf_single_series_data(
            ctx, inputs.symbol, period, market_type.value, required_candles, "close"
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
    计算ETF MACD指标

    Args:
        ctx: FastMCP上下文对象
        inputs: MACD输入参数，包含symbol（ETF代码）、timeframe（时间框架，默认1h）、fast_period（快线周期，默认为12）、slow_period（慢线周期，默认为26）、signal_period（信号线周期，默认为9）、history_len（历史数据长度，默认5）
        market_type: ETF市场类型，默认A

    Returns:
        MacdOutput: MACD输出对象，包含symbol、timeframe、fast_period、slow_period、signal_period、macd主线、signal信号线、histogram柱状图或error错误信息
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
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")
        required_candles = (
            inputs.slow_period + inputs.signal_period + inputs.history_len + 10
        )
        close_prices = await _fetch_etf_single_series_data(
            ctx, inputs.symbol, period, market_type.value, required_candles, "close"
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
    计算ETF布林带（Bollinger Bands）

    Args:
        ctx: FastMCP上下文对象
        inputs: 布林带输入参数，包含symbol（ETF代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为20）、nbdevup（上轨标准差倍数，默认为2.0）、nbdevdn（下轨标准差倍数，默认为2.0）、matype（移动平均类型，默认为0）、history_len（历史数据长度，默认5）
        market_type: ETF市场类型，默认A

    Returns:
        BbandsOutput: 布林带输出对象，包含symbol、timeframe、period、nbdevup、nbdevdn、matype、upper_band上轨、middle_band中轨、lower_band下轨或error错误信息
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
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_etf_single_series_data(
            ctx, inputs.symbol, period, market_type.value, required_candles, "close"
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
    计算ETF平均真实波幅（ATR）

    Args:
        ctx: FastMCP上下文对象
        inputs: ATR输入参数，包含symbol（ETF代码）、timeframe（时间框架，默认1h）、period（计算周期，默认为14）、history_len（历史数据长度，默认5）
        market_type: ETF市场类型，默认A

    Returns:
        AtrOutput: ATR输出对象，包含symbol、timeframe、period、atr指标值或error错误信息
    """
    await ctx.info(f"Calculating ETF ATR for {inputs.symbol}, Period: {inputs.period}")

    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }

    try:
        # 转换时间框架
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(inputs.timeframe, "daily")
        required_candles = inputs.period + inputs.history_len - 1

        price_data = await _fetch_etf_multi_series_data(
            ctx,
            inputs.symbol,
            period,
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
    获取ETF折价率分析（仅支持A股ETF）

    Args:
        ctx: FastMCP上下文对象
        inputs: ETF折价分析输入参数，包含symbol（ETF代码）、market_type（市场类型，默认A）、days（分析天数，默认30）

    Returns:
        dict: 包含success状态、analysis折价分析数据或error错误信息的字典
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
        ctx: FastMCP上下文对象
        inputs: ETF比较输入参数，包含symbols（ETF代码列表）、market_type（市场类型，默认A）、period_days（比较周期天数，默认30）

    Returns:
        dict: 包含success状态、comparison_period比较周期、market_type市场类型、etf_count ETF数量、results比较结果、analysis_date分析日期或error错误信息的字典
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
        ctx: FastMCP上下文对象
        inputs: ETF筛选输入参数，包含market_type（市场类型，默认A）、min_volume（最小成交量）、max_discount_rate（最大折价率）、min_nav（最小净值）、category（ETF分类）、limit（结果数量，默认20）

    Returns:
        dict: 包含success状态、market_type市场类型、criteria筛选条件、results筛选结果、count数量、screening_date筛选日期或error错误信息的字典
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

    Args:
        ctx: FastMCP上下文对象
        inputs: 综合分析输入参数，包含symbol（ETF代码）、timeframe（时间框架，默认1h）、history_len（历史数据长度，默认5）、indicators_to_include（要包含的指标列表，默认全部）以及各指标的可选周期参数（sma_period默认为20、rsi_period默认为14、macd_fast_period默认为12等）
        market_type: ETF市场类型，默认A

    Returns:
        ComprehensiveAnalysisOutput: 综合分析输出对象，包含symbol、timeframe、report_text报告文本、structured_data结构化数据或error错误信息
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
