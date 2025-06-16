import akshare as ak
import asyncio
from typing import List, Optional, Dict, Any
from fastmcp import Context
from datetime import datetime, timedelta

# 使用现有的AKShare管理器
from .akshare_service import akshare_manager


async def fetch_etf_spot_data(
    ctx: Context,
    symbols: Optional[List[str]] = None,
    market: str = "A",  # A: A股, HK: 港股, US: 美股
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取ETF实时行情数据

    Args:
        ctx: FastMCP上下文
        symbols: ETF代码列表，None表示获取所有
        market: 市场类型 ('A', 'HK', 'US')
        max_retries: 最大重试次数
        retry_delay: 重试延迟

    Returns:
        ETF实时数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(
                f"Fetching ETF spot data for {market} market (attempt {attempt + 1})"
            )

            if market.upper() == "A":
                # A股ETF实时数据
                df = await akshare_manager._run_in_executor(ak.fund_etf_fund_daily_em)
            elif market.upper() == "HK":
                # 港股ETF实时数据 - 使用港股相关接口
                df = await akshare_manager._run_in_executor(ak.stock_hk_spot_em)
            elif market.upper() == "US":
                # 美股ETF实时数据 - 使用美股相关接口
                df = await akshare_manager._run_in_executor(ak.stock_us_spot_em)
            else:
                await ctx.error(f"Unsupported market: {market}")
                return None

            if df is None or df.empty:
                await ctx.warning(f"No ETF data available for {market} market")
                return None

            # 筛选指定ETF
            if symbols and market.upper() == "A":
                conditions = None
                for symbol in symbols:
                    if conditions is None:
                        conditions = df["基金代码"].str.contains(symbol) | df[
                            "基金简称"
                        ].str.contains(symbol)
                    else:
                        conditions = (
                            conditions
                            | df["基金代码"].str.contains(symbol)
                            | df["基金简称"].str.contains(symbol)
                        )
                df = df[conditions]
            elif symbols and market.upper() in ["HK", "US"]:
                conditions = None
                for symbol in symbols:
                    if conditions is None:
                        conditions = df["代码"].str.contains(symbol) | df[
                            "名称"
                        ].str.contains(symbol)
                    else:
                        conditions = (
                            conditions
                            | df["代码"].str.contains(symbol)
                            | df["名称"].str.contains(symbol)
                        )
                df = df[conditions]

            # 数据处理
            result = []
            for _, row in df.iterrows():
                try:
                    if market.upper() == "A":
                        data_dict = {
                            "symbol": row.get("基金代码", ""),
                            "name": row.get("基金简称", ""),
                            "type": row.get("类型", ""),
                            "nav": row.get("单位净值", 0),
                            "accumulated_nav": row.get("累计净值", 0),
                            "price": row.get("市价", 0),
                            "change": row.get("涨跌额", 0),
                            "pct_change": row.get("涨跌幅", 0),
                            "volume": row.get("成交量", 0),
                            "amount": row.get("成交额", 0),
                            "discount_rate": row.get("折价率", 0),
                            "market": "A",
                        }
                    elif market.upper() == "HK":
                        data_dict = {
                            "symbol": row.get("代码", ""),
                            "name": row.get("名称", ""),
                            "price": row.get("最新价", 0),
                            "change": row.get("涨跌额", 0),
                            "pct_change": row.get("涨跌幅", 0),
                            "volume": row.get("成交量", 0),
                            "amount": row.get("成交额", 0),
                            "market": "HK",
                        }
                    elif market.upper() == "US":
                        data_dict = {
                            "symbol": row.get("代码", ""),
                            "name": row.get("名称", ""),
                            "price": row.get("最新价", 0),
                            "change": row.get("涨跌额", 0),
                            "pct_change": row.get("涨跌幅", 0),
                            "volume": row.get("成交量", 0),
                            "amount": row.get("成交额", 0),
                            "market": "US",
                        }

                    result.append(data_dict)
                except (ValueError, KeyError) as e:
                    await ctx.warning(f"Invalid ETF data row: {e}")
                    continue
            await ctx.info(
                f"Successfully fetched {len(result)} ETF records for {market} market"
            )
            return result

        except Exception as e:
            await ctx.warning(f"Error fetching ETF data (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2**attempt))
            else:
                await ctx.error("Max retries exceeded for ETF data")
                return None

    return None


async def fetch_etf_hist_data(
    ctx: Context,
    symbol: str,
    market: str = "A",  # A: A股, HK: 港股, US: 美股
    period: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = "qfq",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取ETF历史数据

    Args:
        ctx: FastMCP上下文
        symbol: ETF代码
        market: 市场类型
        period: 数据周期
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权类型
        max_retries: 最大重试次数
        retry_delay: 重试延迟

    Returns:
        ETF历史数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(
                f"Fetching ETF history for {symbol} in {market} market (attempt {attempt + 1})"
            )

            # 构建请求参数
            kwargs = {"symbol": symbol}
            if start_date:
                kwargs["start_date"] = start_date
            if end_date:
                kwargs["end_date"] = end_date

            # 根据市场选择对应的接口
            if market.upper() == "A":
                kwargs["period"] = period
                kwargs["adjust"] = adjust
                df = await akshare_manager._run_in_executor(
                    ak.fund_etf_hist_em, **kwargs
                )
            elif market.upper() == "HK":
                # 港股ETF历史数据
                df = await akshare_manager._run_in_executor(ak.stock_hk_hist, **kwargs)
            elif market.upper() == "US":
                # 美股ETF历史数据
                kwargs["adjust"] = adjust
                df = await akshare_manager._run_in_executor(ak.stock_us_daily, **kwargs)
            else:
                await ctx.error(f"Unsupported market: {market}")
                return None

            if df is None or df.empty:
                await ctx.warning(f"No historical data for {symbol} in {market} market")
                return None

            # 数据处理
            result = []
            for _, row in df.iterrows():
                try:
                    if market.upper() == "A":
                        data_dict = {
                            "date": str(row.get("日期", "")),
                            "open": float(row.get("开盘", 0)),
                            "high": float(row.get("最高", 0)),
                            "low": float(row.get("最低", 0)),
                            "close": float(row.get("收盘", 0)),
                            "volume": float(row.get("成交量", 0)),
                            "amount": float(row.get("成交额", 0)),
                            "pct_change": float(row.get("涨跌幅", 0)),
                            "change": float(row.get("涨跌额", 0)),
                            "amplitude": float(row.get("振幅", 0)),
                            "turnover_rate": float(row.get("换手率", 0)),
                            "market": "A",
                        }
                    elif market.upper() == "HK":
                        data_dict = {
                            "date": str(row.get("日期", "")),
                            "open": float(row.get("开盘", 0)),
                            "high": float(row.get("最高", 0)),
                            "low": float(row.get("最低", 0)),
                            "close": float(row.get("收盘", 0)),
                            "volume": float(row.get("成交量", 0)),
                            "amount": float(row.get("成交额", 0)),
                            "pct_change": float(row.get("涨跌幅", 0)),
                            "market": "HK",
                        }
                    elif market.upper() == "US":
                        data_dict = {
                            "date": str(row.get("date", "")),
                            "open": float(row.get("open", 0)),
                            "high": float(row.get("high", 0)),
                            "low": float(row.get("low", 0)),
                            "close": float(row.get("close", 0)),
                            "volume": float(row.get("volume", 0)),
                            "market": "US",
                        }

                    # 基础数据验证
                    if (
                        data_dict["high"] >= max(data_dict["open"], data_dict["close"])
                        and data_dict["low"]
                        <= min(data_dict["open"], data_dict["close"])
                        and all(
                            x >= 0
                            for x in [
                                data_dict["open"],
                                data_dict["high"],
                                data_dict["low"],
                                data_dict["close"],
                            ]
                        )
                    ):
                        result.append(data_dict)

                except (ValueError, KeyError) as e:
                    await ctx.warning(f"Invalid ETF history data row for {symbol}: {e}")
                    continue

            await ctx.info(
                f"Successfully fetched {len(result)} historical records for {symbol}"
            )
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching ETF history for {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2**attempt))
            else:
                await ctx.error(f"Max retries exceeded for ETF {symbol}")
                return None

    return None


async def fetch_etf_intraday_data(
    ctx: Context,
    symbol: str,
    period: str = "1",  # 1: 1分钟, 5: 5分钟, 15: 15分钟, 30: 30分钟, 60: 60分钟
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = "",
    max_retries: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取A股ETF分时行情数据

    Args:
        ctx: FastMCP上下文
        symbol: ETF代码
        period: 时间周期
        start_date: 开始时间 (格式: 'YYYY-MM-DD HH:MM:SS')
        end_date: 结束时间 (格式: 'YYYY-MM-DD HH:MM:SS')
        adjust: 复权类型
        max_retries: 最大重试次数

    Returns:
        ETF分时数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    # 设置默认时间范围（当日）
    if not start_date or not end_date:
        today = datetime.now().strftime("%Y-%m-%d")
        start_date = f"{today} 09:30:00"
        end_date = f"{today} 15:00:00"

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(
                f"Fetching ETF intraday data for {symbol} (attempt {attempt + 1})"
            )

            df = await akshare_manager._run_in_executor(
                ak.fund_etf_hist_min_em,
                symbol=symbol,
                period=period,
                adjust=adjust,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                await ctx.warning(f"No intraday data for ETF {symbol}")
                return None

            # 数据处理
            result = []
            for _, row in df.iterrows():
                try:
                    data_dict = {
                        "datetime": str(row.get("时间", "")),
                        "open": float(row.get("开盘", 0)),
                        "high": float(row.get("最高", 0)),
                        "low": float(row.get("最低", 0)),
                        "close": float(row.get("收盘", 0)),
                        "volume": float(row.get("成交量", 0)),
                        "amount": float(row.get("成交额", 0)),
                        "average_price": float(row.get("均价", 0)),
                    }
                    result.append(data_dict)
                except (ValueError, KeyError) as e:
                    await ctx.warning(f"Invalid intraday data row for {symbol}: {e}")
                    continue

            await ctx.info(
                f"Successfully fetched {len(result)} intraday records for {symbol}"
            )
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching intraday data for {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error(f"Max retries exceeded for intraday data {symbol}")
                return None

    return None


async def fetch_etf_category_data(
    ctx: Context,
    category: str = "ETF基金",  # ETF基金, 封闭式基金, LOF基金等
    max_retries: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取ETF分类数据

    Args:
        ctx: FastMCP上下文
        category: ETF分类
        max_retries: 最大重试次数

    Returns:
        ETF分类数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(
                f"Fetching ETF category data for {category} (attempt {attempt + 1})"
            )

            df = await akshare_manager._run_in_executor(
                ak.fund_etf_category_sina, symbol=category
            )

            if df is None or df.empty:
                await ctx.warning(f"No category data for {category}")
                return None

            # 数据处理
            result = []
            for _, row in df.iterrows():
                try:
                    data_dict = {
                        "symbol": row.get("代码", ""),
                        "name": row.get("名称", ""),
                        "price": float(row.get("最新价", 0)),
                        "change": float(row.get("涨跌额", 0)),
                        "pct_change": float(row.get("涨跌幅", 0)),
                        "open": float(row.get("今开", 0)),
                        "high": float(row.get("最高", 0)),
                        "low": float(row.get("最低", 0)),
                        "volume": float(row.get("成交量", 0)),
                        "amount": float(row.get("成交额", 0)),
                        "category": category,
                    }
                    result.append(data_dict)
                except (ValueError, KeyError) as e:
                    await ctx.warning(f"Invalid category data row: {e}")
                    continue

            await ctx.info(
                f"Successfully fetched {len(result)} category records for {category}"
            )
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching category data (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error("Max retries exceeded for category data")
                return None

    return None


async def fetch_etf_ths_data(
    ctx: Context,
    date: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取同花顺ETF数据

    Args:
        ctx: FastMCP上下文
        date: 查询日期 (格式: 'YYYYMMDD')
        max_retries: 最大重试次数

    Returns:
        同花顺ETF数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    # 设置默认日期
    if not date:
        date = datetime.now().strftime("%Y%m%d")

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(f"Fetching THS ETF data for {date} (attempt {attempt + 1})")

            df = await akshare_manager._run_in_executor(ak.fund_etf_spot_ths, date=date)

            if df is None or df.empty:
                await ctx.warning(f"No THS ETF data for {date}")
                return None

            # 数据处理
            result = []
            for _, row in df.iterrows():
                try:
                    data_dict = {
                        "ranking": int(row.get("序号", 0)),
                        "symbol": row.get("基金代码", ""),
                        "name": row.get("基金名称", ""),
                        "nav": float(row.get("最新-单位净值", 0)),
                        "accumulated_nav": float(row.get("最新-累计净值", 0)),
                        "pct_change": float(row.get("日增长率", 0)),
                        "fund_type": row.get("基金类型", ""),
                        "query_date": row.get("查询日期", ""),
                    }
                    result.append(data_dict)
                except (ValueError, KeyError) as e:
                    await ctx.warning(f"Invalid THS ETF data row: {e}")
                    continue

            await ctx.info(
                f"Successfully fetched {len(result)} THS ETF records for {date}"
            )
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching THS ETF data (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error("Max retries exceeded for THS ETF data")
                return None

    return None


async def search_etf_by_keyword(
    ctx: Context,
    keyword: str,
    market: str = "A",
) -> Optional[List[Dict[str, str]]]:
    """
    根据关键词搜索ETF

    Args:
        ctx: FastMCP上下文
        keyword: 搜索关键词
        market: 市场类型

    Returns:
        匹配的ETF列表
    """
    try:
        # 先获取ETF列表数据
        etf_data = await fetch_etf_spot_data(ctx, [keyword], market=market)
        if not etf_data:
            return None

        # 搜索匹配的ETF
        results = []
        for etf in etf_data:
            if keyword.lower() in etf.get("name", "").lower() or keyword in etf.get(
                "symbol", ""
            ):
                results.append(
                    {
                        "symbol": etf.get("symbol", ""),
                        "name": etf.get("name", ""),
                        "market": market,
                    }
                )

        return results

    except Exception as e:
        await ctx.error(f"Error searching ETF by keyword {keyword}: {e}")
        return None


# 向后兼容的函数别名
async def _fetch_and_prepare_etf_data(
    ctx: Context,
    symbol: str,
    market: str = "A",
    period: str = "daily",
    required_days: int = 100,
    data_field: str = "close",
) -> Optional[List[float]]:
    """
    向后兼容的ETF数据获取函数
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=required_days + 50)).strftime(
        "%Y%m%d"
    )

    etf_data = await fetch_etf_hist_data(
        ctx, symbol, market, period, start_date, end_date
    )

    if not etf_data:
        return None

    field_map = {
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }

    if data_field not in field_map:
        return None

    return [item[field_map[data_field]] for item in etf_data[-required_days:]]
