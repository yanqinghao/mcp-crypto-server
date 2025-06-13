import yfinance as yf
import json
import asyncio
import requests
import pandas as pd
from typing import List, Optional, Dict, Any
from fastmcp import Context
from datetime import datetime
import concurrent.futures
from functools import partial


class DataFetchError(Exception):
    """数据获取异常"""

    pass


class YFinanceManager:
    """YFinance数据管理器，用于管理美股数据获取"""

    def __init__(self):
        self._initialized = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._session = None

    def initialize(self):
        """初始化YFinance配置"""
        try:
            # 创建会话以提高性能
            import requests

            self._session = requests.Session()
            self._initialized = True
        except Exception as e:
            raise DataFetchError(f"Failed to initialize YFinance: {e}")

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    async def _run_in_executor(self, func, *args, **kwargs):
        """在线程池中运行同步函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, partial(func, *args, **kwargs)
        )

    def _create_ticker(self, symbol: str):
        """创建ticker对象"""
        return yf.Ticker(symbol, session=self._session)


# 全局YFinance管理器
yfinance_manager = YFinanceManager()


def init_yfinance():
    """初始化YFinance"""
    try:
        yfinance_manager.initialize()
    except Exception as e:
        print(f"Warning: Failed to initialize YFinance: {e}")


# 初始化YFinance
init_yfinance()


async def fetch_us_stock_hist_data(
    ctx: Context,
    symbol: str,
    period: str = "1y",  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    interval: str = "1d",  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取美股历史数据

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码 (例如: 'AAPL', 'GOOGL', 'TSLA')
        period: 数据周期
        interval: 数据间隔
        start_date: 开始日期 (格式: 'YYYY-MM-DD')
        end_date: 结束日期 (格式: 'YYYY-MM-DD')
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）

    Returns:
        历史数据列表，格式为字典列表
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(
                f"Fetching US stock history for {symbol} (attempt {attempt + 1})"
            )

            # 创建ticker对象
            ticker = yfinance_manager._create_ticker(symbol)

            # 构建请求参数
            kwargs = {"interval": interval}

            if start_date and end_date:
                kwargs["start"] = start_date
                kwargs["end"] = end_date
            else:
                kwargs["period"] = period

            # 异步获取历史数据
            df = await yfinance_manager._run_in_executor(ticker.history, **kwargs)

            if df is None or df.empty:
                await ctx.warning(f"No data returned for {symbol}")
                return None

            # 数据处理和验证
            df = df.reset_index()
            valid_data = []

            for _, row in df.iterrows():
                try:
                    # 处理日期格式
                    date_val = row.get("Date", row.get("Datetime", ""))
                    if hasattr(date_val, "strftime"):
                        date_str = date_val.strftime("%Y-%m-%d")
                    else:
                        date_str = str(date_val)

                    data_dict = {
                        "date": date_str,
                        "open": float(row.get("Open", 0)),
                        "high": float(row.get("High", 0)),
                        "low": float(row.get("Low", 0)),
                        "close": float(row.get("Close", 0)),
                        "adj_close": float(row.get("Adj Close", 0)),
                        "volume": int(row.get("Volume", 0)),
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
                                data_dict["adj_close"],
                            ]
                        )
                    ):
                        valid_data.append(data_dict)

                except (ValueError, KeyError, TypeError) as e:
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


async def fetch_us_stock_info(
    ctx: Context,
    symbol: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    获取美股基础信息

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        max_retries: 最大重试次数

    Returns:
        股票基础信息字典
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(f"Fetching stock info for {symbol}")

            ticker = yfinance_manager._create_ticker(symbol)
            info = await yfinance_manager._run_in_executor(lambda: ticker.info)

            if not info:
                await ctx.warning(f"No info available for {symbol}")
                return None

            # 提取关键信息
            result = {
                "symbol": info.get("symbol", symbol),
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "trailing_pe": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "float_shares": info.get("floatShares", 0),
                "business_summary": info.get("businessSummary", ""),
            }

            await ctx.info(f"Successfully fetched info for {symbol}")
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching info for {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error(f"Max retries exceeded for {symbol} info")
                return None

    return None


async def fetch_us_stock_realtime_data(
    ctx: Context,
    symbols: List[str],
    max_retries: int = 3,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    获取美股实时数据

    Args:
        ctx: FastMCP上下文
        symbols: 股票代码列表
        max_retries: 最大重试次数

    Returns:
        实时数据字典，key为股票代码
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(f"Fetching real-time data for {len(symbols)} symbols")

            # 批量获取实时数据
            symbols_str = " ".join(symbols)
            tickers = yf.Tickers(symbols_str, session=yfinance_manager._session)

            result = {}
            for symbol in symbols:
                try:
                    ticker = getattr(tickers.tickers, symbol)
                    hist = await yfinance_manager._run_in_executor(
                        ticker.history, period="1d", interval="1m"
                    )

                    if hist.empty:
                        continue

                    latest = hist.iloc[-1]
                    info = await yfinance_manager._run_in_executor(lambda: ticker.info)

                    result[symbol] = {
                        "symbol": symbol,
                        "price": float(latest.get("Close", 0)),
                        "open": float(latest.get("Open", 0)),
                        "high": float(latest.get("High", 0)),
                        "low": float(latest.get("Low", 0)),
                        "volume": int(latest.get("Volume", 0)),
                        "previous_close": float(info.get("previousClose", 0)),
                        "change": float(info.get("regularMarketChange", 0)),
                        "change_percent": float(
                            info.get("regularMarketChangePercent", 0)
                        )
                        * 100,
                        "market_cap": info.get("marketCap", 0),
                        "pe_ratio": info.get("trailingPE", 0),
                        "timestamp": datetime.now().isoformat(),
                    }
                except Exception as e:
                    await ctx.warning(f"Error processing {symbol}: {e}")
                    continue

            await ctx.info(
                f"Successfully fetched real-time data for {len(result)} stocks"
            )
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching real-time data (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error("Max retries exceeded for real-time data")
                return None

    return None


async def fetch_us_stock_financials(
    ctx: Context,
    symbol: str,
    statement_type: str = "income",  # income, balance, cashflow
    quarterly: bool = False,
    max_retries: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取美股财务数据

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        statement_type: 财务报表类型
        quarterly: 是否获取季度数据
        max_retries: 最大重试次数

    Returns:
        财务数据列表
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(f"Fetching {statement_type} statement for {symbol}")

            ticker = yfinance_manager._create_ticker(symbol)

            # 根据报表类型选择对应的属性
            if statement_type == "income":
                if quarterly:
                    df = await yfinance_manager._run_in_executor(
                        lambda: ticker.quarterly_income_stmt
                    )
                else:
                    df = await yfinance_manager._run_in_executor(
                        lambda: ticker.income_stmt
                    )
            elif statement_type == "balance":
                if quarterly:
                    df = await yfinance_manager._run_in_executor(
                        lambda: ticker.quarterly_balance_sheet
                    )
                else:
                    df = await yfinance_manager._run_in_executor(
                        lambda: ticker.balance_sheet
                    )
            elif statement_type == "cashflow":
                if quarterly:
                    df = await yfinance_manager._run_in_executor(
                        lambda: ticker.quarterly_cashflow
                    )
                else:
                    df = await yfinance_manager._run_in_executor(
                        lambda: ticker.cashflow
                    )
            else:
                await ctx.error(f"Unsupported statement type: {statement_type}")
                return None

            if df is None or df.empty:
                await ctx.warning(f"No {statement_type} data for {symbol}")
                return None

            # 转换数据格式
            result = []
            for col in df.columns:
                period_data = {"period": str(col)}
                for idx in df.index:
                    try:
                        value = df.loc[idx, col]
                        if pd.notna(value):
                            period_data[str(idx)] = str(float(value))
                    except (ValueError, TypeError):
                        period_data[str(idx)] = ""
                result.append(period_data)

            await ctx.info(f"Successfully fetched {statement_type} data for {symbol}")
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching {statement_type} for {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error(f"Max retries exceeded for {symbol} {statement_type}")
                return None

    return None


async def fetch_market_indices_data(
    ctx: Context,
    indices: List[str] = ["^GSPC", "^DJI", "^IXIC"],  # S&P500, Dow Jones, NASDAQ
    period: str = "1mo",
    interval: str = "1d",
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    获取美股市场指数数据

    Args:
        ctx: FastMCP上下文
        indices: 指数代码列表
        period: 数据周期
        interval: 数据间隔

    Returns:
        指数数据字典
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    try:
        await ctx.info(f"Fetching market indices data for {len(indices)} indices")

        result = {}
        for index in indices:
            try:
                ticker = yfinance_manager._create_ticker(index)
                df = await yfinance_manager._run_in_executor(
                    ticker.history, period=period, interval=interval
                )

                if df.empty:
                    continue

                index_data = []
                df = df.reset_index()
                for _, row in df.iterrows():
                    try:
                        date_val = row.get("Date", row.get("Datetime", ""))
                        if hasattr(date_val, "strftime"):
                            date_str = date_val.strftime("%Y-%m-%d")
                        else:
                            date_str = str(date_val)

                        data_dict = {
                            "date": date_str,
                            "open": float(row.get("Open", 0)),
                            "high": float(row.get("High", 0)),
                            "low": float(row.get("Low", 0)),
                            "close": float(row.get("Close", 0)),
                            "volume": int(row.get("Volume", 0)),
                        }
                        index_data.append(data_dict)
                    except (ValueError, KeyError, TypeError):
                        continue

                result[index] = index_data

            except Exception as e:
                await ctx.warning(f"Error fetching data for index {index}: {e}")
                continue

        await ctx.info(f"Successfully fetched data for {len(result)} indices")
        return result

    except Exception as e:
        await ctx.error(f"Error fetching market indices data: {e}")
        return None


async def fetch_options_data(
    ctx: Context,
    symbol: str,
    expiration_date: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    获取期权数据

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        expiration_date: 到期日期 (格式: 'YYYY-MM-DD')
        max_retries: 最大重试次数

    Returns:
        期权数据字典
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(f"Fetching options data for {symbol}")

            ticker = yfinance_manager._create_ticker(symbol)

            # 获取期权到期日期
            expiry_dates = await yfinance_manager._run_in_executor(
                lambda: ticker.options
            )

            if not expiry_dates:
                await ctx.warning(f"No options available for {symbol}")
                return None

            # 选择到期日期
            if expiration_date and expiration_date in expiry_dates:
                target_date = expiration_date
            else:
                target_date = expiry_dates[0]  # 使用最近的到期日

            # 获取期权链
            option_chain = await yfinance_manager._run_in_executor(
                ticker.option_chain, target_date
            )

            result = {
                "symbol": symbol,
                "expiration_date": target_date,
                "calls": option_chain.calls.to_dict("records")
                if not option_chain.calls.empty
                else [],
                "puts": option_chain.puts.to_dict("records")
                if not option_chain.puts.empty
                else [],
                "available_dates": expiry_dates,
            }

            await ctx.info(f"Successfully fetched options data for {symbol}")
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching options for {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error(f"Max retries exceeded for {symbol} options")
                return None

    return None


# 向后兼容的函数别名
async def _fetch_and_prepare_us_stock_data(
    ctx: Context,
    symbol: str,
    period: str = "1y",
    required_days: int = 100,
    data_field: str = "close",
) -> Optional[List[float]]:
    """
    向后兼容的美股数据获取函数
    """
    stock_data = await fetch_us_stock_hist_data(ctx, symbol, period=period)

    if not stock_data:
        return None

    field_map = {
        "close": "close",
        "adj_close": "adj_close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }

    if data_field not in field_map:
        return None

    return [item[field_map[data_field]] for item in stock_data[-required_days:]]


async def search_us_stock_by_name(
    ctx: Context, company_name: str
) -> Optional[List[Dict[str, str]]]:
    """根据公司名称搜索美股代码"""
    try:
        # 使用Yahoo Finance搜索API
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc"
        params = {"query": company_name, "region": "1", "lang": "en"}

        response = await yfinance_manager._run_in_executor(
            requests.get, url, params=params
        )

        if response.status_code != 200:
            return None

        # 解析JSONP响应
        content = response.text
        json_start = content.find("(") + 1
        json_end = content.rfind(")")
        json_data = json.loads(content[json_start:json_end])

        results = []
        for item in json_data.get("ResultSet", {}).get("Result", []):
            if item.get("type") == "S":  # 只要股票类型
                results.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "name": item.get("name", ""),
                        "exchange": item.get("exchDisp", ""),
                    }
                )

        return results

    except Exception as e:
        await ctx.error(f"Error searching US stock: {e}")
        return None


# 或者使用yfinance的ticker信息验证
async def validate_us_stock_symbol(
    ctx: Context, symbol: str
) -> Optional[Dict[str, str]]:
    """验证并获取美股符号信息"""
    try:
        ticker = yfinance_manager._create_ticker(symbol)
        info = await yfinance_manager._run_in_executor(lambda: ticker.info)

        if info:
            return {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
            }
        return None
    except Exception as e:
        await ctx.error(f"Error validating symbol {symbol}: {e}")
        return None
