import yfinance as yf
import asyncio
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
        return yf.Ticker(symbol)

    def _create_search(
        self,
        query: str,
        max_results: int = 10,
        news_count: int = 5,
        include_research: bool = False,
    ):
        """创建搜索对象"""
        return yf.Search(
            query,
            max_results=max_results,
            news_count=news_count,
            include_research=include_research,
        )

    def _create_lookup(self, query: str):
        """创建lookup对象"""
        return yf.Lookup(query)


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


# 更新后的搜索功能，使用最新的yfinance Search API
async def search_us_stock_by_name(
    ctx: Context,
    company_name: str,
    max_results: int = 10,
    include_news: bool = False,
    news_count: int = 5,
    include_research: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    根据公司名称搜索美股代码

    Args:
        ctx: FastMCP上下文
        company_name: 公司名称或关键词
        max_results: 最大搜索结果数量
        include_news: 是否包含新闻
        news_count: 新闻数量
        include_research: 是否包含研究报告

    Returns:
        搜索结果字典，包含quotes、news等
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    try:
        await ctx.info(f"Searching for US stocks: {company_name}")

        # 使用yfinance的Search API
        search = yfinance_manager._create_search(
            query=company_name,
            max_results=max_results,
            news_count=news_count if include_news else 0,
            include_research=include_research,
        )

        result = {}

        # 获取股票搜索结果
        try:
            quotes = await yfinance_manager._run_in_executor(lambda: search.quotes)
            if quotes:
                # 格式化搜索结果
                formatted_quotes = []
                for quote in quotes:
                    formatted_quotes.append(
                        {
                            "symbol": quote.get("symbol", ""),
                            "name": quote.get("longname", quote.get("shortname", "")),
                            "type": quote.get("quoteType", ""),
                            "exchange": quote.get("exchange", ""),
                            "sector": quote.get("sector", ""),
                            "industry": quote.get("industry", ""),
                            "market_cap": quote.get("marketCap", 0),
                            "currency": quote.get("currency", ""),
                        }
                    )
                result["quotes"] = formatted_quotes
            else:
                result["quotes"] = []
        except Exception as e:
            await ctx.warning(f"Error fetching quotes: {e}")
            result["quotes"] = []

        # 获取新闻（如果请求）
        if include_news:
            try:
                news = await yfinance_manager._run_in_executor(lambda: search.news)
                result["news"] = news if news else []
            except Exception as e:
                await ctx.warning(f"Error fetching news: {e}")
                result["news"] = []

        # 获取研究报告（如果请求）
        if include_research:
            try:
                research = await yfinance_manager._run_in_executor(
                    lambda: search.research
                )
                result["research"] = research if research else []
            except Exception as e:
                await ctx.warning(f"Error fetching research: {e}")
                result["research"] = []

        await ctx.info(
            f"Search completed. Found {len(result.get('quotes', []))} quotes"
        )
        return result

    except Exception as e:
        await ctx.error(f"Error searching US stock: {e}")
        return None


async def lookup_us_stock_symbol(
    ctx: Context,
    query: str,
    security_type: str = "all",  # all, stock, etf, mutualfund, index, future, currency, cryptocurrency
    count: int = 20,
) -> Optional[List[Dict[str, Any]]]:
    """
    使用Lookup API查找特定类型的证券

    Args:
        ctx: FastMCP上下文
        query: 查询关键词
        security_type: 证券类型
        count: 返回结果数量

    Returns:
        查找结果列表
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    try:
        await ctx.info(f"Looking up {security_type} for: {query}")

        lookup = yfinance_manager._create_lookup(query)

        result = []

        if security_type == "all":
            data = await yfinance_manager._run_in_executor(lookup.get_all, count=count)
        elif security_type == "stock":
            data = await yfinance_manager._run_in_executor(
                lookup.get_stock, count=count
            )
        elif security_type == "etf":
            data = await yfinance_manager._run_in_executor(lookup.get_etf, count=count)
        elif security_type == "mutualfund":
            data = await yfinance_manager._run_in_executor(
                lookup.get_mutualfund, count=count
            )
        elif security_type == "index":
            data = await yfinance_manager._run_in_executor(
                lookup.get_index, count=count
            )
        elif security_type == "future":
            data = await yfinance_manager._run_in_executor(
                lookup.get_future, count=count
            )
        elif security_type == "currency":
            data = await yfinance_manager._run_in_executor(
                lookup.get_currency, count=count
            )
        elif security_type == "cryptocurrency":
            data = await yfinance_manager._run_in_executor(
                lookup.get_cryptocurrency, count=count
            )
        else:
            await ctx.error(f"Unsupported security type: {security_type}")
            return None

        if data:
            for item in data:
                result.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "name": item.get(
                            "name", item.get("longname", item.get("shortname", ""))
                        ),
                        "type": item.get("type", item.get("typeDisp", "")),
                        "exchange": item.get("exchange", item.get("exchDisp", "")),
                        "market": item.get("market", ""),
                    }
                )

        await ctx.info(f"Lookup completed. Found {len(result)} results")
        return result

    except Exception as e:
        await ctx.error(f"Error in lookup: {e}")
        return None


# 验证美股符号的改进版本
async def validate_us_stock_symbol(
    ctx: Context, symbol: str
) -> Optional[Dict[str, Any]]:
    """
    验证并获取美股符号信息

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码

    Returns:
        股票信息字典或None
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    try:
        await ctx.info(f"Validating symbol: {symbol}")

        ticker = yfinance_manager._create_ticker(symbol)

        info = await yfinance_manager._run_in_executor(lambda: ticker.info)

        if info and info.get("regularMarketPrice") is not None:
            return {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "exchange": info.get("exchange", ""),
                "currency": info.get("currency", ""),
                "country": info.get("country", ""),
                "valid": True,
            }
        else:
            await ctx.warning(
                f"Symbol {symbol} appears to be invalid or no market data available"
            )
            return {"symbol": symbol, "valid": False}

    except Exception as e:
        await ctx.error(f"Error validating symbol {symbol}: {e}")
        return {"symbol": symbol, "valid": False, "error": str(e)}


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


# 新增：获取股票分析师评级数据
async def fetch_analyst_recommendations(
    ctx: Context,
    symbol: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    获取分析师推荐评级

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        max_retries: 最大重试次数

    Returns:
        分析师评级数据
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(f"Fetching analyst recommendations for {symbol}")

            ticker = yfinance_manager._create_ticker(symbol)

            # 获取分析师推荐
            recommendations = await yfinance_manager._run_in_executor(
                lambda: ticker.recommendations
            )

            # 获取分析师价格目标
            price_targets = await yfinance_manager._run_in_executor(
                lambda: ticker.analyst_price_targets
            )

            result = {}

            if recommendations is not None and not recommendations.empty:
                result["recommendations"] = recommendations.to_dict("records")
            else:
                result["recommendations"] = []

            if price_targets is not None:
                result["price_targets"] = price_targets
            else:
                result["price_targets"] = {}

            await ctx.info(f"Successfully fetched analyst data for {symbol}")
            return result

        except Exception as e:
            await ctx.warning(
                f"Error fetching analyst recommendations for {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error(f"Max retries exceeded for {symbol} analyst data")
                return None

    return None


# 新增：获取股票分红历史
async def fetch_dividend_history(
    ctx: Context,
    symbol: str,
    period: str = "5y",
    max_retries: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """
    获取股票分红历史

    Args:
        ctx: FastMCP上下文
        symbol: 股票代码
        period: 数据周期
        max_retries: 最大重试次数

    Returns:
        分红历史数据列表
    """
    if not yfinance_manager.is_initialized():
        await ctx.error("YFinance not initialized")
        return None

    for attempt in range(max_retries + 1):
        try:
            await ctx.info(f"Fetching dividend history for {symbol}")

            ticker = yfinance_manager._create_ticker(symbol)

            # 获取股票行为数据（包含分红和股票分割）
            actions = await yfinance_manager._run_in_executor(
                ticker.history, period=period, actions=True
            )

            if actions is None or actions.empty:
                await ctx.warning(f"No dividend data for {symbol}")
                return []

            # 提取分红数据
            dividend_data = []
            actions = actions.reset_index()

            for _, row in actions.iterrows():
                dividend = row.get("Dividends", 0)
                if dividend > 0:
                    date_val = row.get("Date", row.get("Datetime", ""))
                    if hasattr(date_val, "strftime"):
                        date_str = date_val.strftime("%Y-%m-%d")
                    else:
                        date_str = str(date_val)

                    dividend_data.append(
                        {
                            "date": date_str,
                            "dividend": float(dividend),
                            "type": "dividend",
                        }
                    )

            await ctx.info(
                f"Successfully fetched {len(dividend_data)} dividend records for {symbol}"
            )
            return dividend_data

        except Exception as e:
            await ctx.warning(
                f"Error fetching dividend history for {symbol} (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                await ctx.error(f"Max retries exceeded for {symbol} dividend data")
                return None

    return None
