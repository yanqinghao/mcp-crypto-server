import akshare as ak
import asyncio
from typing import List, Optional, Dict, Any
from fastmcp import Context
from datetime import datetime, timedelta
import concurrent.futures
from functools import partial


class DataFetchError(Exception):
    """数据获取异常"""

    pass


class AKShareManager:
    """AKShare数据管理器，用于管理A股和H股数据获取"""

    def __init__(self):
        self._initialized = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def initialize(self):
        """初始化AKShare配置"""
        try:
            # 设置AKShare的一些全局配置
            self._initialized = True
        except Exception as e:
            raise DataFetchError(f"Failed to initialize AKShare: {e}")

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    async def _run_in_executor(self, func, *args, **kwargs):
        """在线程池中运行同步函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, partial(func, *args, **kwargs)
        )


# 全局AKShare管理器
akshare_manager = AKShareManager()


def init_akshare():
    """初始化AKShare"""
    try:
        akshare_manager.initialize()
    except Exception as e:
        print(f"Warning: Failed to initialize AKShare: {e}")


# 初始化AKShare
init_akshare()


async def fetch_stock_hist_data(
    ctx: Context,
    symbol: str,
    period: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = "qfq",  # qfq: 前复权, hfq: 后复权, none: 不复权
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取A股历史数据

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码 (例如: '000001', '600519')
        period: 数据周期 ('daily', 'weekly', 'monthly')
        start_date: 开始日期 (格式: 'YYYYMMDD')
        end_date: 结束日期 (格式: 'YYYYMMDD')
        adjust: 复权类型
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）

    Returns:
        历史数据列表，格式为字典列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(
                f"Fetching stock history for {symbol} (attempt {attempt + 1})"
            )

            # 构建请求参数
            kwargs = {"symbol": symbol, "period": period, "adjust": adjust}

            if start_date:
                kwargs["start_date"] = start_date
            if end_date:
                kwargs["end_date"] = end_date

            # 异步调用AKShare函数
            df = await akshare_manager._run_in_executor(ak.stock_zh_a_hist, **kwargs)

            if df is None or df.empty:
                await ctx.warning(f"No data returned for {symbol}")
                return None

            # 数据处理和验证
            df = df.reset_index()
            valid_data = []

            for _, row in df.iterrows():
                try:
                    data_dict = {
                        "date": row.get("日期", ""),
                        "open": float(row.get("开盘", 0)),
                        "high": float(row.get("最高", 0)),
                        "low": float(row.get("最低", 0)),
                        "close": float(row.get("收盘", 0)),
                        "volume": float(row.get("成交量", 0)),
                        "amount": float(row.get("成交额", 0)),
                        "amplitude": float(row.get("振幅", 0)),
                        "pct_change": float(row.get("涨跌幅", 0)),
                        "change": float(row.get("涨跌额", 0)),
                        "turnover_rate": float(row.get("换手率", 0)),
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
                        valid_data.append(data_dict)

                except (ValueError, KeyError) as e:
                    await ctx.warning(f"Invalid data row for {symbol}: {e}")
                    continue

            await ctx.info(
                f"Successfully fetched {len(valid_data)} records for {symbol}"
            )
            return valid_data

        except Exception as e:
            await ctx.warning(
                f"Error fetching {symbol} data (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2**attempt))
            else:
                await ctx.error(f"Max retries exceeded for {symbol}")
                return None

    return None


async def fetch_stock_realtime_data(
    ctx: Context, symbols: Optional[List[str]] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    获取实时股票数据

    Args:
        ctx: FastMCP上下文
        symbols: 股票代码列表，None表示获取所有

    Returns:
        实时数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    try:
        await ctx.info("Fetching real-time stock data")

        # 获取实时数据
        df = await akshare_manager._run_in_executor(ak.stock_zh_a_spot_em)

        if df is None or df.empty:
            await ctx.warning("No real-time data available")
            return None

        # 筛选指定股票
        if symbols:
            df = df[df["代码"].isin(symbols)]

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
                    "turnover_rate": float(row.get("换手率", 0)),
                    "pe_ratio": float(row.get("市盈率-动态", 0)),
                    "pb_ratio": float(row.get("市净率", 0)),
                    "market_cap": float(row.get("总市值", 0)),
                }
                result.append(data_dict)
            except (ValueError, KeyError) as e:
                await ctx.warning(f"Invalid real-time data row: {e}")
                continue

        await ctx.info(f"Successfully fetched real-time data for {len(result)} stocks")
        return result

    except Exception as e:
        await ctx.error(f"Error fetching real-time stock data: {e}")
        return None


async def fetch_hk_stock_data(
    ctx: Context,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取港股历史数据

    Args:
        ctx: FastMCP上下文
        symbol: 港股代码 (例如: '00700', '00941')
        start_date: 开始日期
        end_date: 结束日期
        max_retries: 最大重试次数
        retry_delay: 重试延迟

    Returns:
        港股历史数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(
                f"Fetching HK stock data for {symbol} (attempt {attempt + 1})"
            )

            # 构建请求参数
            kwargs = {"symbol": symbol}
            if start_date:
                kwargs["start_date"] = start_date
            if end_date:
                kwargs["end_date"] = end_date

            # 获取港股数据
            df = await akshare_manager._run_in_executor(ak.stock_hk_hist, **kwargs)

            if df is None or df.empty:
                await ctx.warning(f"No HK stock data returned for {symbol}")
                return None

            # 数据处理
            result = []
            for _, row in df.iterrows():
                try:
                    data_dict = {
                        "date": str(row.get("日期", "")),
                        "open": float(row.get("开盘", 0)),
                        "high": float(row.get("最高", 0)),
                        "low": float(row.get("最低", 0)),
                        "close": float(row.get("收盘", 0)),
                        "volume": float(row.get("成交量", 0)),
                        "amount": float(row.get("成交额", 0)),
                        "pct_change": float(row.get("涨跌幅", 0)),
                    }
                    result.append(data_dict)
                except (ValueError, KeyError) as e:
                    await ctx.warning(f"Invalid HK stock data row: {e}")
                    continue

            await ctx.info(
                f"Successfully fetched {len(result)} HK stock records for {symbol}"
            )
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching HK stock {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2**attempt))
            else:
                await ctx.error(f"Max retries exceeded for HK stock {symbol}")
                return None

    return None


async def fetch_stock_financial_data(
    ctx: Context,
    symbol: str,
    indicator: str = "main",  # main, profit, balance, cash_flow
    max_retries: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取股票财务数据

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        indicator: 财务指标类型
        max_retries: 最大重试次数

    Returns:
        财务数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(f"Fetching financial data for {symbol}")

            # 根据指标类型选择对应的函数
            func_map = {
                "main": ak.stock_financial_em,
                "profit": ak.stock_profit_em,
                "balance": ak.stock_balance_em,
                "cash_flow": ak.stock_cash_flow_em,
            }

            if indicator not in func_map:
                await ctx.error(f"Unsupported financial indicator: {indicator}")
                return None

            df = await akshare_manager._run_in_executor(
                func_map[indicator], symbol=symbol
            )

            if df is None or df.empty:
                await ctx.warning(f"No financial data for {symbol}")
                return None

            # 转换为字典列表
            result = df.to_dict("records")

            await ctx.info(
                f"Successfully fetched {len(result)} financial records for {symbol}"
            )
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching financial data for {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error(f"Max retries exceeded for financial data {symbol}")
                return None

    return None


async def fetch_market_index_data(
    ctx: Context,
    index_code: str = "000001",  # 上证指数
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取市场指数数据

    Args:
        ctx: FastMCP上下文
        index_code: 指数代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        指数数据列表
    """
    if not akshare_manager.is_initialized():
        await ctx.error("AKShare not initialized")
        return None

    try:
        await ctx.info(f"Fetching index data for {index_code}")

        kwargs = {"symbol": index_code}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        df = await akshare_manager._run_in_executor(ak.stock_zh_index_daily, **kwargs)

        if df is None or df.empty:
            await ctx.warning(f"No index data for {index_code}")
            return None

        result = []
        for _, row in df.iterrows():
            try:
                data_dict = {
                    "date": str(row.get("date", "")),
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                }
                result.append(data_dict)
            except (ValueError, KeyError):
                continue

        await ctx.info(f"Successfully fetched {len(result)} index records")
        return result

    except Exception as e:
        await ctx.error(f"Error fetching index data: {e}")
        return None


# 向后兼容的函数别名
async def _fetch_and_prepare_stock_data(
    ctx: Context,
    symbol: str,
    period: str = "daily",
    required_days: int = 100,
    data_field: str = "close",
) -> Optional[List[float]]:
    """
    向后兼容的股票数据获取函数
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=required_days + 50)).strftime(
        "%Y%m%d"
    )

    stock_data = await fetch_stock_hist_data(ctx, symbol, period, start_date, end_date)

    if not stock_data:
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

    return [item[field_map[data_field]] for item in stock_data[-required_days:]]


# A股股票代码名称映射
async def fetch_stock_symbol_mapping(ctx: Context) -> Optional[Dict[str, str]]:
    """获取A股股票代码名称映射表"""
    try:
        # 获取上海交易所股票列表
        sh_df = await akshare_manager._run_in_executor(ak.stock_info_sh_name_code)
        # 获取深圳交易所股票列表
        sz_df = await akshare_manager._run_in_executor(ak.stock_info_sz_name_code)

        mapping = {}
        # 处理上海交易所数据
        for _, row in sh_df.iterrows():
            name = row.get("公司简称", "")
            code = row.get("公司代码", "")
            if name and code:
                mapping[name] = code

        # 处理深圳交易所数据
        for _, row in sz_df.iterrows():
            name = row.get("公司简称", "")
            code = row.get("公司代码", "")
            if name and code:
                mapping[name] = code

        return mapping
    except Exception as e:
        await ctx.error(f"Error fetching stock mapping: {e}")
        return None


# 根据名称查找代码
async def search_stock_by_name(
    ctx: Context, company_name: str
) -> Optional[List[Dict[str, str]]]:
    """根据公司名称搜索股票代码"""
    mapping = await fetch_stock_symbol_mapping(ctx)
    if not mapping:
        return None

    results = []
    for name, code in mapping.items():
        if company_name.lower() in name.lower():
            results.append({"name": name, "symbol": code})

    return results
