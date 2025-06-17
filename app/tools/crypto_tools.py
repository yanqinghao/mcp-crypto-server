import numpy as np
import talib
import json
from typing import Dict, Optional, List, Any

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
    AdxInput,
    AdxOutput,
    ObvInput,
    ObvOutput,
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
    OrderBookInput,
    OrderBookOutput,
    TradesInput,
    TradesOutput,
    OHLCVCandle,
    OrderBookLevel,
    TradeData,
)

# Import the refined data fetching helpers
from services.crypto_service import (
    fetch_ohlcv_data,
    fetch_ticker_data,
    fetch_order_book,
    fetch_trades,
)
import ccxt  # For exception types
from fastmcp import FastMCP, Context

mcp = FastMCP()


async def _fetch_single_series_data(
    ctx: Context,
    symbol: str,
    timeframe: str,
    required_candles: int,
    series_type: str = "close",
) -> Optional[np.ndarray]:
    """
    获取单个数据序列的通用函数

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号
        timeframe: 时间框架
        required_candles: 需要的K线数量
        series_type: 数据类型 (open, high, low, close, volume)

    Returns:
        numpy数组或None
    """
    try:
        # 获取OHLCV数据，加上缓冲区以确保有足够的数据
        buffer_size = max(50, required_candles // 2)
        fetch_size = required_candles + buffer_size

        ohlcv_data = await fetch_ohlcv_data(ctx, symbol, timeframe, fetch_size)

        if not ohlcv_data or len(ohlcv_data) < required_candles:
            import traceback

            traceback.print_exc()
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(ohlcv_data) if ohlcv_data else 0}, need {required_candles}"
            )
            return None

        # 选择对应的数据列
        series_index_map = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}

        if series_type not in series_index_map:
            await ctx.error(f"Invalid series type: {series_type}")
            import traceback

            traceback.print_exc()
            return None

        series_index = series_index_map[series_type]
        data_series = np.array([candle[series_index] for candle in ohlcv_data])

        # 确保数据质量
        if np.any(np.isnan(data_series)) or np.any(np.isinf(data_series)):
            await ctx.warning(
                f"Found NaN or Inf values in {series_type} data for {symbol}"
            )
            # 移除NaN和Inf值
            data_series = data_series[~(np.isnan(data_series) | np.isinf(data_series))]

        return data_series

    except Exception as e:
        import traceback

        traceback.print_exc()
        await ctx.error(f"Error fetching {series_type} data for {symbol}: {e}")
        return None


async def _fetch_multi_series_data(
    ctx: Context,
    symbol: str,
    timeframe: str,
    required_candles: int,
    series_types: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    """
    获取多个数据序列的通用函数

    Args:
        ctx: FastMCP上下文
        symbol: 交易对符号
        timeframe: 时间框架
        required_candles: 需要的K线数量
        series_types: 需要的数据类型列表

    Returns:
        包含各数据序列的字典或None
    """
    try:
        # 获取OHLCV数据
        buffer_size = max(50, required_candles // 2)
        fetch_size = required_candles + buffer_size

        ohlcv_data = await fetch_ohlcv_data(ctx, symbol, timeframe, fetch_size)

        if not ohlcv_data or len(ohlcv_data) < required_candles:
            await ctx.error(
                f"Insufficient data for {symbol}: got {len(ohlcv_data) if ohlcv_data else 0}, need {required_candles}"
            )
            return None

        series_index_map = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}

        result = {}
        for series_type in series_types:
            if series_type not in series_index_map:
                await ctx.error(f"Invalid series type: {series_type}")
                return None

            series_index = series_index_map[series_type]
            data_series = np.array([candle[series_index] for candle in ohlcv_data])

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


# --- SMA Tool ---
@mcp.tool()
async def calculate_sma(ctx: Context, inputs: SmaInput) -> SmaOutput:
    """
    计算简单移动平均线 (Simple Moving Average, SMA)

    SMA是最基础且最重要的技术指标，通过计算指定期间的平均价格来平滑价格波动，
    帮助识别价格趋势方向。在加密货币交易中广泛用于趋势判断和支撑阻力位确定。

    计算公式:
    SMA(n) = (P1 + P2 + ... + Pn) / n

    其中P为收盘价，n为周期数

    技术特点:
    - 滞后指标，平滑价格波动
    - 周期越长，平滑效果越强，滞后性越大
    - 价格突破SMA常被视为趋势变化信号
    - 多条不同周期SMA的排列反映趋势强度

    Args:
        inputs.symbol (str): 加密货币交易对，如'BTC/USDT', 'ETH/USD', 'DOGE/BTC'
        inputs.timeframe (str): 时间框架 '1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M'
        inputs.period (int): SMA计算周期，常用值:
            - 短期: 5, 10, 20 (适合日内交易)
            - 中期: 50, 100 (适合波段交易)
            - 长期: 200, 365 (适合趋势跟踪)
        inputs.history_len (int): 返回的历史数据长度，用于趋势分析

    Returns:
        SmaOutput: SMA计算结果
        - symbol: 交易对符号
        - timeframe: 时间框架
        - period: 计算周期
        - sma: SMA值列表，按时间顺序排列，最新值在末尾
        - error: 错误信息(如果有)

    应用场景:
    - 趋势判断: 价格在SMA上方=看涨，下方=看跌
    - 支撑阻力: SMA线常作为动态支撑或阻力位
    - 交易信号: 价格穿越SMA产生买卖信号
    - 多周期分析: 短期SMA上穿长期SMA形成金叉

    Example:
        计算BTC/USDT的20日均线:
        >>> inputs = SmaInput(
        ...     symbol="BTC/USDT",
        ...     timeframe="1d",
        ...     period=20,
        ...     history_len=10
        ... )
        >>> result = await calculate_sma(ctx, inputs)
        >>> if result.sma:
        ...     latest_sma = result.sma[-1]
        ...     print(f"BTC 20日均线: ${latest_sma:.2f}")
        ...
        ...     # 判断趋势
        ...     if len(result.sma) > 1:
        ...         if result.sma[-1] > result.sma[-2]:
        ...             print("SMA呈上升趋势")
        ...         else:
        ...             print("SMA呈下降趋势")

    常用策略:
    - 单SMA策略: 价格突破SMA做多/做空
    - 双SMA策略: 短期SMA穿越长期SMA的金叉死叉
    - 三SMA系统: 5-20-60日线的多空排列判断

    Note:
        - SMA对最新价格反应较慢，适合过滤市场噪音
        - 在震荡市场中容易产生虚假信号
        - 建议结合其他指标使用，如RSI、MACD等
        - 周期选择要根据交易风格和市场特点调整
    """
    await ctx.info(
        f"Executing calculate_sma for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # SMA需要period + history_len - 1个数据点来计算history_len个SMA值
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return SmaOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for SMA."
            )

        sma_values = talib.SMA(close_prices, timeperiod=inputs.period)

        # 提取有效的历史值
        valid_sma = _extract_valid_values(sma_values, inputs.history_len)

        if not valid_sma:
            return SmaOutput(
                **output_base,
                error="SMA calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated SMA for {inputs.symbol}: {len(valid_sma)} values, latest: {valid_sma[-1]}"
        )
        return SmaOutput(**output_base, sma=valid_sma)

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_sma for {inputs.symbol}: {e}")
        return SmaOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(f"Unexpected error in calculate_sma for {inputs.symbol}: {e}")
        return SmaOutput(**output_base, error="An unexpected server error occurred.")


# --- RSI Tool ---
@mcp.tool()
async def calculate_rsi(ctx: Context, inputs: RsiInput) -> RsiOutput:
    """
    计算相对强弱指数 (Relative Strength Index, RSI)

    RSI是由威尔斯·威尔德(J. Welles Wilder)开发的动量震荡指标，用于衡量价格变动的
    速度和变化幅度。RSI在0-100之间波动，是识别超买超卖状态的最重要指标之一。

    计算逻辑:
    1. 计算每日价格变化：上涨日的涨幅、下跌日的跌幅
    2. 计算n期内平均上涨幅度(AU)和平均下跌幅度(AD)
    3. 计算相对强度: RS = AU / AD
    4. 计算RSI: RSI = 100 - (100 / (1 + RS))

    技术特点:
    - 震荡指标，在0-100之间波动
    - 反映价格变动的内在强度
    - 提前于价格发出信号（领先指标特性）
    - 在趋势市场和震荡市场都有良好表现

    Args:
        inputs.symbol (str): 加密货币交易对
        inputs.timeframe (str): 时间框架
        inputs.period (int): RSI计算周期，常用值:
            - 14: 经典设置，平衡敏感性和稳定性
            - 9: 更敏感，适合短线交易
            - 21: 更平滑，适合中长线分析
            - 6: 极敏感，适合超短线
        inputs.history_len (int): 返回的历史数据长度

    Returns:
        RsiOutput: RSI计算结果
        - symbol: 交易对符号
        - timeframe: 时间框架
        - period: 计算周期
        - rsi: RSI值列表 (0-100)，按时间顺序排列
        - error: 错误信息(如果有)

    交易信号解读:
    - RSI > 70: 超买状态，价格可能回调，考虑减仓或做空
    - RSI < 30: 超卖状态，价格可能反弹，考虑加仓或做多
    - RSI = 50: 中性位置，多空力量平衡
    - RSI穿越50: 趋势变化的重要信号

    高级应用:
    - 背离分析: 价格创新高但RSI不创新高=顶背离(看跌)
    - 背离分析: 价格创新低但RSI不创新低=底背离(看涨)
    - 区间调整: 牛市中30-70，熊市中20-80
    - 多时框架: 结合不同周期RSI确认信号

    Example:
        计算ETH/USDT的14期RSI:
        >>> inputs = RsiInput(
        ...     symbol="ETH/USDT",
        ...     timeframe="4h",
        ...     period=14,
        ...     history_len=5
        ... )
        >>> result = await calculate_rsi(ctx, inputs)
        >>> if result.rsi:
        ...     latest_rsi = result.rsi[-1]
        ...     print(f"ETH RSI(14): {latest_rsi:.2f}")
        ...
        ...     # 信号判断
        ...     if latest_rsi > 70:
        ...         print("⚠️ 超买警告：价格可能回调")
        ...     elif latest_rsi < 30:
        ...         print("💡 超卖机会：价格可能反弹")
        ...     else:
        ...         print("✅ RSI处于正常区间")
        ...
        ...     # 趋势分析
        ...     if len(result.rsi) > 1:
        ...         if result.rsi[-1] > result.rsi[-2]:
        ...             print("📈 RSI上升，买方力量增强")
        ...         else:
        ...             print("📉 RSI下降，卖方力量增强")

    常用策略:
    - RSI突破策略: RSI突破30做多，跌破70做空
    - RSI背离策略: 价格与RSI背离时逆势交易
    - RSI区间策略: 在超买超卖区间进行均值回归交易
    - 多重确认: RSI配合价格行为和成交量确认

    Note:
        - 在强势趋势中，RSI可能长时间保持在极值区域
        - 单独使用RSI容易产生假信号，建议组合使用
        - 不同周期的RSI可能给出相反信号，需要层次分析
        - 加密货币市场波动大，可考虑调整超买超卖阈值
    """
    await ctx.info(
        f"Executing calculate_rsi for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # RSI需要period + history_len个数据点
        required_candles = inputs.period + inputs.history_len
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return RsiOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for RSI."
            )

        rsi_values = talib.RSI(close_prices, timeperiod=inputs.period)

        # 提取有效的历史值
        valid_rsi = _extract_valid_values(rsi_values, inputs.history_len)

        if not valid_rsi:
            return RsiOutput(
                **output_base,
                error="RSI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated RSI for {inputs.symbol}: {len(valid_rsi)} values, latest: {valid_rsi[-1]}"
        )
        return RsiOutput(**output_base, rsi=valid_rsi)

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_rsi for {inputs.symbol}: {e}")
        return RsiOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(f"Unexpected error in calculate_rsi for {inputs.symbol}: {e}")
        return RsiOutput(**output_base, error="An unexpected server error occurred.")


# --- MACD Tool ---
@mcp.tool()
async def calculate_macd(ctx: Context, inputs: MacdInput) -> MacdOutput:
    """
    计算MACD指标 (Moving Average Convergence Divergence)

    MACD由Gerald Appel开发，是最流行的趋势跟踪动量指标。通过比较两个不同周期的
    指数移动平均线(EMA)来识别趋势变化和动量强弱，被称为"指标之王"。

    组成部分:
    1. MACD线(快线): 12期EMA - 26期EMA
    2. 信号线(慢线): MACD线的9期EMA
    3. MACD柱状图: MACD线 - 信号线

    计算步骤:
    1. 计算12期EMA和26期EMA
    2. MACD = EMA(12) - EMA(26)
    3. Signal = EMA(MACD, 9)
    4. Histogram = MACD - Signal

    技术特点:
    - 趋势跟踪指标，擅长捕捉趋势变化
    - 既有趋势信息又有动量信息
    - 在趋势市场中表现优异
    - 滞后性相对较小

    Args:
        inputs.symbol (str): 加密货币交易对
        inputs.timeframe (str): 时间框架
        inputs.fast_period (int): 快线EMA周期，默认12
        inputs.slow_period (int): 慢线EMA周期，默认26
        inputs.signal_period (int): 信号线EMA周期，默认9
        inputs.history_len (int): 返回的历史数据长度

    Returns:
        MacdOutput: MACD计算结果
        - symbol: 交易对符号
        - timeframe: 时间框架
        - fast_period/slow_period/signal_period: 计算参数
        - macd: MACD线值列表
        - signal: 信号线值列表
        - histogram: 柱状图值列表
        - error: 错误信息(如果有)

    交易信号:
    1. 金叉死叉信号:
        - MACD上穿信号线(金叉): 买入信号
        - MACD下穿信号线(死叉): 卖出信号

    2. 零轴信号:
        - MACD上穿零轴: 多头市场确认
        - MACD下穿零轴: 空头市场确认

    3. 柱状图信号:
        - 柱状图由负转正: 上升动量增强
        - 柱状图由正转负: 下降动量增强
        - 柱状图背离: 价格与动量背离

    4. 背离信号:
        - 价格创新高但MACD不创新高: 顶背离(看跌)
        - 价格创新低但MACD不创新低: 底背离(看涨)

    Example:
        计算BTC/USDT的MACD指标:
        >>> inputs = MacdInput(
        ...     symbol="BTC/USDT",
        ...     timeframe="1h",
        ...     fast_period=12,
        ...     slow_period=26,
        ...     signal_period=9,
        ...     history_len=20
        ... )
        >>> result = await calculate_macd(ctx, inputs)
        >>> if result.macd:
        ...     macd = result.macd[-1]
        ...     signal = result.signal[-1]
        ...     hist = result.histogram[-1]
        ...
        ...     print(f"BTC MACD: {macd:.4f}")
        ...     print(f"信号线: {signal:.4f}")
        ...     print(f"柱状图: {hist:.4f}")
        ...
        ...     # 金叉死叉判断
        ...     if len(result.macd) > 1:
        ...         prev_macd = result.macd[-2]
        ...         prev_signal = result.signal[-2]
        ...
        ...         if prev_macd <= prev_signal and macd > signal:
        ...             print("🚀 金叉信号: MACD上穿信号线")
        ...         elif prev_macd >= prev_signal and macd < signal:
        ...             print("⚡ 死叉信号: MACD下穿信号线")
        ...
        ...     # 零轴位置
        ...     if macd > 0:
        ...         print("📈 MACD在零轴上方，偏向多头")
        ...     else:
        ...         print("📉 MACD在零轴下方，偏向空头")
        ...
        ...     # 动量分析
        ...     if hist > 0:
        ...         print("💪 柱状图为正，上升动量")
        ...     else:
        ...         print("📉 柱状图为负，下降动量")

    参数调优:
    - 快速市场: 5-13-5 (更敏感)
    - 标准设置: 12-26-9 (经典)
    - 稳定市场: 19-39-9 (更平滑)
    - 长线投资: 12-26-1 (减少信号线影响)

    常用策略:
    - 金叉死叉策略: 基础的MACD交易策略
    - 零轴策略: 结合零轴位置判断趋势强弱
    - 背离策略: 寻找价格与MACD的背离机会
    - 柱状图策略: 利用柱状图变化预测转折点

    Note:
        - MACD在震荡市场中容易产生假信号
        - 强烈建议结合价格行为和成交量分析
        - 不同时间框架的MACD可能给出不同信号
        - 在加密货币市场中，可考虑调整参数以适应高波动性
    """
    await ctx.info(
        f"Executing calculate_macd for {inputs.symbol}, Periods: {inputs.fast_period}/{inputs.slow_period}/{inputs.signal_period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "fast_period": inputs.fast_period,
        "slow_period": inputs.slow_period,
        "signal_period": inputs.signal_period,
    }
    try:
        # MACD需要足够的数据用于慢速EMA和信号线计算，加上历史长度
        required_candles = (
            inputs.slow_period + inputs.signal_period + inputs.history_len + 10
        )  # 增加缓冲

        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return MacdOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for MACD."
            )

        macd, signal, hist = talib.MACD(
            close_prices,
            fastperiod=inputs.fast_period,
            slowperiod=inputs.slow_period,
            signalperiod=inputs.signal_period,
        )

        # 提取有效的历史值
        valid_macd = _extract_valid_values(macd, inputs.history_len)
        valid_signal = _extract_valid_values(signal, inputs.history_len)
        valid_hist = _extract_valid_values(hist, inputs.history_len)

        if not valid_macd or not valid_signal or not valid_hist:
            return MacdOutput(
                **output_base,
                error="MACD calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated MACD for {inputs.symbol}: {len(valid_macd)} values, latest MACD: {valid_macd[-1]}"
        )
        return MacdOutput(
            **output_base, macd=valid_macd, signal=valid_signal, histogram=valid_hist
        )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_macd for {inputs.symbol}: {e}")
        return MacdOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_macd for {inputs.symbol}: {e}",
        )
        return MacdOutput(**output_base, error="An unexpected server error occurred.")


# --- Bollinger Bands (BBANDS) Tool ---
@mcp.tool()
async def calculate_bbands(ctx: Context, inputs: BbandsInput) -> BbandsOutput:
    """
    计算布林带 (Bollinger Bands)

    布林带由约翰·博林格(John Bollinger)在1980年代开发，是基于统计学的技术指标。
    它由一条中轨(移动平均线)和上下两条轨道(基于标准差)组成，用于判断价格的相对高低
    和波动率变化，是识别超买超卖和波动率的优秀工具。

    构成要素:
    - 中轨(Middle Band): n期简单移动平均线(通常20期)
    - 上轨(Upper Band): 中轨 + k × n期标准差(通常k=2)
    - 下轨(Lower Band): 中轨 - k × n期标准差(通常k=2)

    统计学原理:
    - 约95%的价格波动应在上下轨之间(2倍标准差)
    - 约68%的价格波动应在1倍标准差内
    - 基于正态分布理论，但市场并非总是正态分布

    技术特点:
    - 自适应性强，能根据市场波动率调整宽度
    - 提供动态支撑阻力位
    - 结合趋势和波动率信息
    - 在不同市场状态下都有应用价值

    Args:
        inputs.symbol (str): 加密货币交易对
        inputs.timeframe (str): 时间框架
        inputs.period (int): 移动平均和标准差计算周期，默认20
        inputs.nbdevup (float): 上轨标准差倍数，默认2.0
        inputs.nbdevdn (float): 下轨标准差倍数，默认2.0
        inputs.matype (int): 移动平均类型，默认0(SMA)
        inputs.history_len (int): 返回的历史数据长度

    Returns:
        BbandsOutput: 布林带计算结果
        - symbol: 交易对符号
        - timeframe: 时间框架
        - period: 计算周期
        - nbdevup/nbdevdn: 标准差倍数
        - matype: 移动平均类型
        - upper_band: 上轨值列表
        - middle_band: 中轨值列表
        - lower_band: 下轨值列表
        - error: 错误信息(如果有)

    交易信号和应用:
    1. 超买超卖信号:
        - 价格触及上轨: 可能超买，考虑减仓
        - 价格触及下轨: 可能超卖，考虑加仓
        - 价格在轨道内: 正常波动范围

    2. 突破信号:
        - 价格突破上轨: 强势突破，可能继续上涨
        - 价格跌破下轨: 弱势突破，可能继续下跌
        - 突破后回归轨道内: 假突破信号

    3. 波动率分析:
        - 布林带收窄: 波动率降低，可能酝酿大行情
        - 布林带扩张: 波动率增加，趋势加强
        - 带宽指标: (上轨-下轨)/中轨 × 100

    4. 走势模式:
        - 布林带走平: 横盘整理
        - 布林带上倾: 上升趋势
        - 布林带下倾: 下降趋势

    Example:
        计算ETH/USDT的布林带:
        >>> inputs = BbandsInput(
        ...     symbol="ETH/USDT",
        ...     timeframe="4h",
        ...     period=20,
        ...     nbdevup=2.0,
        ...     nbdevdn=2.0,
        ...     history_len=10
        ... )
        >>> result = await calculate_bbands(ctx, inputs)
        >>> if result.upper_band:
        ...     upper = result.upper_band[-1]
        ...     middle = result.middle_band[-1]
        ...     lower = result.lower_band[-1]
        ...     band_width = upper - lower
        ...
        ...     print(f"ETH 布林带:")
        ...     print(f"上轨: ${upper:.2f}")
        ...     print(f"中轨: ${middle:.2f}")
        ...     print(f"下轨: ${lower:.2f}")
        ...     print(f"带宽: ${band_width:.2f}")
        ...
        ...     # 波动率分析
        ...     width_ratio = band_width / middle * 100
        ...     print(f"带宽比例: {width_ratio:.2f}%")
        ...
        ...     if len(result.upper_band) > 1:
        ...         prev_width = result.upper_band[-2] - result.lower_band[-2]
        ...         if band_width > prev_width * 1.1:
        ...             print("📈 布林带扩张：波动率增加")
        ...         elif band_width < prev_width * 0.9:
        ...             print("📊 布林带收窄：波动率降低，可能有大行情")
        ...
        ...     # 假设当前价格
        ...     current_price = middle  # 简化示例
        ...     position_ratio = (current_price - lower) / (upper - lower)
        ...     print(f"价格位置: {position_ratio*100:.1f}% (0%=下轨, 100%=上轨)")

    参数优化:
    - 短线交易: 10期, 1.5倍标准差
    - 经典设置: 20期, 2倍标准差
    - 长线分析: 50期, 2.5倍标准差
    - 高波动市场: 增加标准差倍数
    - 低波动市场: 减少标准差倍数

    常用策略:
    - 均值回归策略: 价格触及极值轨道时反向交易
    - 突破策略: 价格突破轨道后顺势交易
    - 挤压策略: 布林带收窄时等待突破方向
    - 趋势跟踪: 价格沿着轨道运行时的趋势跟随

    高级技巧:
    - %B指标: (价格-下轨)/(上轨-下轨) 标准化价格位置
    - 带宽指标: 衡量波动率的标准化指标
    - 多时框架: 结合不同周期布林带确认信号
    - 布林带回归: 价格偏离中轨时的回归交易

    Note:
        - 布林带不是绝对的支撑阻力，只是概率指导
        - 在强势趋势中，价格可能沿着某一轨道运行
        - 建议结合其他指标确认信号
        - 加密货币市场波动大，可考虑调整标准差倍数
    """
    await ctx.info(
        f"Executing calculate_bbands for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
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
        # BBANDS需要period + history_len - 1个数据点
        required_candles = inputs.period + inputs.history_len - 1
        close_prices = await _fetch_single_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, "close"
        )

        if close_prices is None or len(close_prices) < required_candles:
            return BbandsOutput(
                **output_base, error="Failed to fetch sufficient OHLCV data for BBANDS."
            )

        upper, middle, lower = talib.BBANDS(
            close_prices,
            timeperiod=inputs.period,
            nbdevup=inputs.nbdevup,
            nbdevdn=inputs.nbdevdn,
            matype=inputs.matype,
        )

        # 提取有效的历史值
        valid_upper = _extract_valid_values(upper, inputs.history_len)
        valid_middle = _extract_valid_values(middle, inputs.history_len)
        valid_lower = _extract_valid_values(lower, inputs.history_len)

        if not valid_upper or not valid_middle or not valid_lower:
            return BbandsOutput(
                **output_base,
                error="BBANDS calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated BBANDS for {inputs.symbol}: {len(valid_upper)} values, latest upper: {valid_upper[-1]}"
        )
        return BbandsOutput(
            **output_base,
            upper_band=valid_upper,
            middle_band=valid_middle,
            lower_band=valid_lower,
        )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_bbands for {inputs.symbol}: {e}")
        return BbandsOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error in calculate_bbands for {inputs.symbol}: {e}",
        )
        return BbandsOutput(**output_base, error="An unexpected server error occurred.")


# --- Average True Range (ATR) Tool ---
@mcp.tool()
async def calculate_atr(ctx: Context, inputs: AtrInput) -> AtrOutput:
    """
    计算平均真实波幅 (Average True Range, ATR)

    ATR由威尔斯·威尔德(J. Welles Wilder)开发，是衡量市场波动性的经典指标。
    它测量价格在给定时间内的波动幅度，帮助交易者了解市场的波动性水平，
    是风险管理和仓位计算的重要工具。

    真实波幅(True Range)定义:
    每日真实波幅是以下三者中的最大值:
    1. 当日最高价 - 当日最低价
    2. |当日最高价 - 前日收盘价|
    3. |当日最低价 - 前日收盘价|

    ATR计算:
    ATR = n期真实波幅的移动平均值(通常使用威尔德移动平均)

    技术特点:
    - 纯波动性指标，不指示方向
    - 数值越大表示波动性越高
    - 具有趋势性，波动性变化相对平缓
    - 不受价格方向影响，只反映波动幅度

    Args:
        inputs.symbol (str): 加密货币交易对
        inputs.timeframe (str): 时间框架
        inputs.period (int): ATR计算周期，常用值:
            - 14: 经典设置，威尔德原始建议
            - 7: 短期波动分析
            - 21: 中期波动分析
            - 50: 长期波动分析
        inputs.history_len (int): 返回的历史数据长度

    Returns:
        AtrOutput: ATR计算结果
        - symbol: 交易对符号
        - timeframe: 时间框架
        - period: 计算周期
        - atr: ATR值列表，按时间顺序排列
        - error: 错误信息(如果有)

    主要应用:
    1. 止损设置:
        - 动态止损: 入场价 ± n × ATR
        - 常用倍数: 1.5-3倍ATR
        - 根据风险承受能力调整倍数

    2. 仓位管理:
        - 固定风险仓位 = 风险资金 / (ATR × 倍数)
        - ATR越大，仓位越小(控制风险)
        - ATR越小，仓位可适当增大

    3. 利润目标:
        - 利润目标 = 入场价 + n × ATR
        - 风险回报比 = 利润目标距离 / 止损距离

    4. 突破确认:
        - 价格变动 > 1.5×ATR: 可能是有效突破
        - 价格变动 < 0.5×ATR: 可能是虚假突破

    5. 市场状态判断:
        - ATR上升: 波动性增加，趋势可能加强
        - ATR下降: 波动性降低，可能进入整理
        - ATR极值: 波动性异常，注意风险

    Example:
        计算BTC/USDT的ATR并用于风险管理:
        >>> inputs = AtrInput(
        ...     symbol="BTC/USDT",
        ...     timeframe="1d",
        ...     period=14,
        ...     history_len=10
        ... )
        >>> result = await calculate_atr(ctx, inputs)
        >>> if result.atr:
        ...     current_atr = result.atr[-1]
        ...     print(f"BTC ATR(14): ${current_atr:.2f}")
        ...
        ...     # 假设当前价格和入场价格
        ...     current_price = 45000  # 示例价格
        ...     entry_price = 44000   # 示例入场价
        ...
        ...     # 止损计算
        ...     stop_loss_distance = 2 * current_atr
        ...     long_stop = entry_price - stop_loss_distance
        ...     short_stop = entry_price + stop_loss_distance
        ...
        ...     print(f"做多止损位: ${long_stop:.2f}")
        ...     print(f"做空止损位: ${short_stop:.2f}")
        ...
        ...     # 利润目标
        ...     profit_target_long = entry_price + 3 * current_atr
        ...     profit_target_short = entry_price - 3 * current_atr
        ...
        ...     print(f"做多目标位: ${profit_target_long:.2f}")
        ...     print(f"做空目标位: ${profit_target_short:.2f}")
        ...
        ...     # 仓位计算(假设风险资金1000美元)
        ...     risk_capital = 1000
        ...     position_size = risk_capital / stop_loss_distance
        ...     print(f"建议仓位: {position_size:.4f} BTC")
        ...
        ...     # 波动性分析
        ...     if len(result.atr) > 5:
        ...         recent_avg = sum(result.atr[-5:]) / 5
        ...         if current_atr > recent_avg * 1.2:
        ...             print("⚠️ 当前波动性较高，建议降低仓位")
        ...         elif current_atr < recent_avg * 0.8:
        ...             print("✅ 当前波动性较低，适合增加仓位")

    ATR在不同市场的应用:
    - 加密货币: 波动性大，建议使用较大ATR倍数
    - 外汇市场: 波动性中等，标准ATR倍数适用
    - 股票市场: 波动性相对较小，可使用较小倍数

    常用策略:
    - ATR突破策略: 价格突破n×ATR时入场
    - ATR反转策略: 价格偏离MA超过n×ATR时反向交易
    - ATR趋势跟踪: 结合ATR设置动态止损的趋势跟踪
    - ATR波动率交易: 根据ATR水平调整交易频率

    高级应用:
    - 归一化ATR: ATR/价格，消除价格水平影响
    - ATR%: ATR与价格的百分比关系
    - 多时框架ATR: 结合不同周期ATR制定策略
    - ATR突破过滤器: 用ATR过滤假突破信号

    Note:
        - ATR是滞后指标，反映历史波动性
        - 新闻事件可能导致ATR快速变化
        - 不同品种的ATR水平差异很大，需要单独分析
        - 建议结合其他指标确认交易信号
        - 在极端市场条件下，ATR可能失效
    """
    await ctx.info(
        f"Executing calculate_atr for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # ATR需要period + history_len - 1个数据点
        required_candles = inputs.period + inputs.history_len - 1

        price_data = await _fetch_multi_series_data(
            ctx,
            inputs.symbol,
            inputs.timeframe,
            required_candles,
            ["high", "low", "close"],
        )

        if not price_data:
            return AtrOutput(**output_base, error="Failed to fetch HLC data for ATR.")

        high_prices = price_data["high"]
        low_prices = price_data["low"]
        close_prices = price_data["close"]

        # 确保数据长度足够
        if len(high_prices) < required_candles:
            return AtrOutput(
                **output_base,
                error=f"Insufficient HLC data points for ATR. Need at least {required_candles}.",
            )

        atr_values = talib.ATR(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )

        # 提取有效的历史值
        valid_atr = _extract_valid_values(atr_values, inputs.history_len)

        if not valid_atr:
            return AtrOutput(
                **output_base,
                error="ATR calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated ATR for {inputs.symbol}: {len(valid_atr)} values, latest: {valid_atr[-1]}"
        )
        return AtrOutput(**output_base, atr=valid_atr)

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_atr for {inputs.symbol}: {e}")
        return AtrOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(f"Unexpected error in calculate_atr for {inputs.symbol}: {e}")
        return AtrOutput(**output_base, error="An unexpected server error occurred.")


# --- Average Directional Index (ADX) Tool ---
@mcp.tool()
async def calculate_adx(ctx: Context, inputs: AdxInput) -> AdxOutput:
    """
    计算平均方向指数 (Average Directional Index, ADX)

    ADX同样由威尔斯·威尔德开发，是衡量趋势强度的核心指标。与其他技术指标不同，
    ADX不显示趋势方向，而是量化趋势的强度。它与+DI和-DI指标结合使用，
    提供趋势强度和方向的完整分析。

    组成部分:
    1. ADX: 平均方向指数，衡量趋势强度(0-100)
    2. +DI: 正方向指标，反映向上趋势强度
    3. -DI: 负方向指标，反映向下趋势强度

    计算逻辑:
    1. 计算方向移动: +DM, -DM
    2. 计算真实波幅: TR
    3. 计算方向指标: +DI, -DI
    4. 计算方向指数: DX = |+DI - -DI| / (+DI + -DI) × 100
    5. 计算ADX: DX的移动平均

    技术特点:
    - 纯趋势强度指标，不指示方向
    - 数值范围0-100，越高表示趋势越强
    - 滞后性相对较小
    - 在不同市场状态下都有指导意义

    Args:
        inputs.symbol (str): 加密货币交易对
        inputs.timeframe (str): 时间框架
        inputs.period (int): ADX计算周期，常用值:
            - 14: 经典设置，威尔德原始建议
            - 7: 更敏感，适合短线交易
            - 21: 更平滑，适合中长线分析
            - 28: 长期趋势分析
        inputs.history_len (int): 返回的历史数据长度

    Returns:
        AdxOutput: ADX计算结果
        - symbol: 交易对符号
        - timeframe: 时间框架
        - period: 计算周期
        - adx: ADX值列表 (0-100)
        - plus_di: +DI值列表
        - minus_di: -DI值列表
        - error: 错误信息(如果有)

    信号解读:
    1. 趋势强度判断:
        - ADX > 25: 强趋势，适合趋势跟踪策略
        - ADX 20-25: 中等趋势，谨慎跟踪
        - ADX < 20: 弱趋势或横盘，避免趋势策略
        - ADX > 40: 极强趋势，但可能过度延伸

    2. 趋势方向判断:
        - +DI > -DI: 上升趋势占主导
        - -DI > +DI: 下降趋势占主导
        - +DI与-DI交叉: 趋势方向可能改变

    3. 综合信号:
        - ADX上升 + +DI > -DI: 强烈看涨
        - ADX上升 + -DI > +DI: 强烈看跌
        - ADX下降: 趋势强度减弱
        - ADX极低: 趋势即将开始或市场横盘

    4. 入场时机:
        - ADX从低位(< 20)开始上升: 新趋势开始
        - +DI上穿-DI且ADX > 25: 看涨信号
        - -DI上穿+DI且ADX > 25: 看跌信号

    5. 出场信号:
        - ADX从高位开始下降: 趋势强度减弱
        - ADX跌破25: 趋势结束信号
        - +DI和-DI开始收敛: 趋势可能转向

    Example:
        计算ETH/USDT的ADX指标:
        >>> inputs = AdxInput(
        ...     symbol="ETH/USDT",
        ...     timeframe="4h",
        ...     period=14,
        ...     history_len=20
        ... )
        >>> result = await calculate_adx(ctx, inputs)
        >>> if result.adx:
        ...     adx = result.adx[-1]
        ...     plus_di = result.plus_di[-1]
        ...     minus_di = result.minus_di[-1]
        ...
        ...     print(f"ETH ADX指标:")
        ...     print(f"ADX: {adx:.2f}")
        ...     print(f"+DI: {plus_di:.2f}")
        ...     print(f"-DI: {minus_di:.2f}")
        ...
        ...     # 趋势强度分析
        ...     if adx > 40:
        ...         print("🔥 极强趋势，但需警惕过度延伸")
        ...     elif adx > 25:
        ...         print("💪 强趋势，适合趋势跟踪")
        ...     elif adx > 20:
        ...         print("📊 中等趋势，谨慎操作")
        ...     else:
        ...         print("😴 弱趋势或横盘，避免趋势策略")
        ...
        ...     # 方向分析
        ...     if plus_di > minus_di:
        ...         direction = "看涨"
        ...         strength = plus_di - minus_di
        ...     else:
        ...         direction = "看跌"
        ...         strength = minus_di - plus_di
        ...
        ...     print(f"趋势方向: {direction}")
        ...     print(f"方向强度: {strength:.2f}")
        ...
        ...     # 交叉信号分析
        ...     if len(result.plus_di) > 1:
        ...         prev_plus = result.plus_di[-2]
        ...         prev_minus = result.minus_di[-2]
        ...
        ...         if prev_plus <= prev_minus and plus_di > minus_di and adx > 20:
        ...             print("🚀 +DI上穿-DI：看涨信号")
        ...         elif prev_plus >= prev_minus and plus_di < minus_di and adx > 20:
        ...             print("📉 -DI上穿+DI：看跌信号")
        ...
        ...     # ADX趋势分析
        ...     if len(result.adx) > 1:
        ...         if result.adx[-1] > result.adx[-2]:
        ...             print("📈 ADX上升：趋势强度增加")
        ...         else:
        ...             print("📉 ADX下降：趋势强度减弱")

    交易策略应用:
    - ADX突破策略: ADX突破25时入场趋势方向
    - DI交叉策略: +DI和-DI交叉时的方向交易
    - ADX过滤器: 用ADX过滤其他指标的信号
    - 趋势强度分级: 根据ADX水平调整仓位大小

    参数优化:
    - 敏感设置: ADX(7) - 适合短线交易
    - 标准设置: ADX(14) - 经典平衡设置
    - 稳定设置: ADX(21) - 适合中长线
    - 长期设置: ADX(28) - 长期趋势分析

    常见组合:
    - ADX + 移动平均: 趋势确认系统
    - ADX + MACD: 趋势强度和动量结合
    - ADX + 价格行为: 突破信号确认
    - ADX + 成交量: 趋势质量分析

    注意事项:
    - ADX在震荡市场中数值会较低
    - 新趋势开始时ADX可能滞后
    - 极端市场条件下ADX可能失真
    - 建议结合价格行为分析使用

    Note:
        - ADX只告诉你趋势有多强，不告诉你价格会涨还是跌
        - +DI和-DI的相对关系比绝对数值更重要
        - ADX的转折点往往是重要的交易信号
        - 在加密货币市场中，可以适当降低ADX阈值
    """
    await ctx.info(
        f"Executing calculate_adx for {inputs.symbol}, Period: {inputs.period}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "period": inputs.period,
    }
    try:
        # ADX需要更多数据来稳定计算，加上历史长度
        required_candles = inputs.period * 3 + inputs.history_len

        price_data = await _fetch_multi_series_data(
            ctx,
            inputs.symbol,
            inputs.timeframe,
            required_candles,
            ["high", "low", "close"],
        )

        if not price_data:
            return AdxOutput(**output_base, error="Failed to fetch HLC data for ADX.")

        high_prices = price_data["high"]
        low_prices = price_data["low"]
        close_prices = price_data["close"]

        if len(high_prices) < required_candles:
            return AdxOutput(
                **output_base,
                error=f"Insufficient HLC data points for ADX. Need at least {required_candles}.",
            )

        adx_values = talib.ADX(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )
        plus_di_values = talib.PLUS_DI(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )
        minus_di_values = talib.MINUS_DI(
            high_prices, low_prices, close_prices, timeperiod=inputs.period
        )

        # 提取有效的历史值
        valid_adx = _extract_valid_values(adx_values, inputs.history_len)
        valid_plus_di = _extract_valid_values(plus_di_values, inputs.history_len)
        valid_minus_di = _extract_valid_values(minus_di_values, inputs.history_len)

        if not valid_adx or not valid_plus_di or not valid_minus_di:
            return AdxOutput(
                **output_base,
                error="ADX/DI calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated ADX for {inputs.symbol}: {len(valid_adx)} values, latest ADX: {valid_adx[-1]}"
        )
        return AdxOutput(
            **output_base, adx=valid_adx, plus_di=valid_plus_di, minus_di=valid_minus_di
        )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_adx for {inputs.symbol}: {e}")
        return AdxOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(f"Unexpected error in calculate_adx for {inputs.symbol}: {e}")
        return AdxOutput(**output_base, error="An unexpected server error occurred.")


# --- On-Balance Volume (OBV) Tool ---
@mcp.tool()
async def calculate_obv(ctx: Context, inputs: ObvInput) -> ObvOutput:
    """
    计算成交量平衡指标 (On-Balance Volume, OBV)

    OBV由Joseph Granville在1963年开发，是最重要的成交量技术指标之一。
    它基于成交量跟随价格的理念，将成交量与价格变化方向结合，
    用于预测价格趋势的强度和可持续性。

    计算逻辑:
    - 如果今日收盘价 > 昨日收盘价：OBV = 前日OBV + 今日成交量
    - 如果今日收盘价 < 昨日收盘价：OBV = 前日OBV - 今日成交量
    - 如果今日收盘价 = 昨日收盘价：OBV = 前日OBV

    核心理念:
    - 成交量是价格变化的先行指标
    - 聪明资金的流入流出反映在成交量上
    - 价格上涨时的大成交量比价格下跌时的大成交量更重要
    - OBV的趋势变化往往先于价格趋势变化

    技术特点:
    - 累积性指标，具有趋势性
    - 绝对数值不重要，趋势方向更关键
    - 对价格和成交量都很敏感
    - 具有一定的预测性

    Args:
        inputs.symbol (str): 加密货币交易对
        inputs.timeframe (str): 时间框架
        inputs.data_points (int): 用于计算的数据点数量
        inputs.history_len (int): 返回的历史数据长度

    Returns:
        ObvOutput: OBV计算结果
        - symbol: 交易对符号
        - timeframe: 时间框架
        - data_points: 使用的数据点数量
        - obv: OBV值列表，按时间顺序排列
        - error: 错误信息(如果有)

    信号分析:
    1. 趋势确认:
        - 价格上涨 + OBV上涨: 上涨趋势确认，成交量支持
        - 价格下跌 + OBV下跌: 下跌趋势确认，抛压充足
        - 趋势一致性越强，信号越可靠

    2. 背离信号:
        - 价格新高 + OBV未新高: 顶背离，上涨动能不足
        - 价格新低 + OBV未新低: 底背离，下跌动能不足
        - 背离是趋势反转的重要预警信号

    3. 突破信号:
        - OBV突破重要阻力: 价格可能跟随突破
        - OBV跌破重要支撑: 价格可能跟随下跌
        - OBV突破往往先于价格突破

    4. 成交量质量:
        - OBV快速上升: 有主力资金流入
        - OBV快速下降: 有主力资金流出
        - OBV横盘: 成交量平衡，等待方向选择

    5. 趋势强度:
        - OBV与价格同步创新高/低: 趋势健康强劲
        - OBV滞后于价格: 趋势强度减弱
        - OBV领先于价格: 成交量先行确认

    Example:
        计算BTC/USDT的OBV指标:
        >>> inputs = ObvInput(
        ...     symbol="BTC/USDT",
        ...     timeframe="1h",
        ...     data_points=100,
        ...     history_len=20
        ... )
        >>> result = await calculate_obv(ctx, inputs)
        >>> if result.obv:
        ...     current_obv = result.obv[-1]
        ...     print(f"BTC OBV: {current_obv:.2f}")
        ...
        ...     # 趋势分析
        ...     if len(result.obv) >= 5:
        ...         recent_obv = result.obv[-5:]
        ...         obv_trend = "上升" if recent_obv[-1] > recent_obv[0] else "下降"
        ...         print(f"OBV短期趋势: {obv_trend}")
        ...
        ...         # 计算OBV变化率
        ...         obv_change = (recent_obv[-1] - recent_obv[0]) / abs(recent_obv[0]) * 100
        ...         print(f"OBV变化率: {obv_change:.2f}%")
        ...
        ...     # 背离分析示例
        ...     if len(result.obv) >= 10:
        ...         # 假设我们有价格数据
        ...         print("背离分析:")
        ...         print("- 观察OBV与价格的趋势一致性")
        ...         print("- 寻找价格新高/低但OBV未确认的情况")
        ...
        ...         # OBV趋势强度
        ...         obv_momentum = result.obv[-1] - result.obv[-10]
        ...         if obv_momentum > 0:
        ...             print(f"💪 OBV动量: +{obv_momentum:.2f} (资金流入)")
        ...         else:
        ...             print(f"📉 OBV动量: {obv_momentum:.2f} (资金流出)")
        ...
        ...     # 成交量质量评估
        ...     if len(result.obv) > 1:
        ...         obv_change_rate = result.obv[-1] - result.obv[-2]
        ...         if abs(obv_change_rate) > 1000000:  # 假设阈值
        ...             print("⚡ 检测到大额资金流动")
        ...
        ...         if obv_change_rate > 0:
        ...             print("📈 今日净流入")
        ...         elif obv_change_rate < 0:
        ...             print("📉 今日净流出")
        ...         else:
        ...             print("➡️ 今日资金平衡")

    实战应用策略:
    1. OBV趋势跟踪:
        - OBV上升时做多，下降时做空
        - 结合价格趋势确认信号

    2. OBV背离交易:
        - 发现背离时准备反向交易
        - 等待价格确认信号再入场

    3. OBV突破策略:
        - OBV突破关键位时跟随交易
        - 设置合理的止损和止盈

    4. 成交量确认系统:
        - 用OBV确认其他技术指标信号
        - 只在OBV支持时进行交易

    分析技巧:
    - 关注OBV的相对位置而非绝对数值
    - 结合价格行为进行综合分析
    - 注意OBV的趋势线支撑阻力
    - 观察OBV在关键价格位的表现

    局限性:
    - 在低成交量时期信号可能不可靠
    - 不能单独用于交易决策
    - 对假突破的过滤能力有限
    - 需要结合其他指标使用

    Note:
        - OBV的绝对数值不重要，重要的是趋势变化
        - 在加密货币市场中，注意异常成交量的影响
        - OBV适合中长期分析，短期噪音较大
        - 建议结合价格形态和其他技术指标使用
    """
    await ctx.info(
        f"Executing calculate_obv for {inputs.symbol}, Data points: {inputs.data_points}, History: {inputs.history_len}"
    )
    output_base = {
        "symbol": inputs.symbol,
        "timeframe": inputs.timeframe,
        "data_points": inputs.data_points,
    }
    try:
        # OBV需要data_points个数据点来计算，确保至少有history_len个有效结果
        required_candles = max(inputs.data_points, inputs.history_len)

        price_data = await _fetch_multi_series_data(
            ctx, inputs.symbol, inputs.timeframe, required_candles, ["close", "volume"]
        )

        if not price_data:
            return ObvOutput(
                **output_base, error="Failed to fetch close and volume data for OBV."
            )

        close_prices = price_data["close"]
        volume_data = price_data["volume"]

        if len(close_prices) < required_candles or len(volume_data) < required_candles:
            return ObvOutput(
                **output_base, error="Insufficient close/volume data points for OBV."
            )

        obv_values = talib.OBV(close_prices, volume_data)

        # 提取有效的历史值
        valid_obv = _extract_valid_values(obv_values, inputs.history_len)

        if not valid_obv:
            return ObvOutput(
                **output_base,
                error="OBV calculation resulted in insufficient valid data.",
            )

        await ctx.info(
            f"Calculated OBV for {inputs.symbol}: {len(valid_obv)} values, latest: {valid_obv[-1]}"
        )
        return ObvOutput(**output_base, obv=valid_obv)

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error in calculate_obv for {inputs.symbol}: {e}")
        return ObvOutput(
            **output_base, error=f"Exchange/Network Error: {type(e).__name__}"
        )
    except Exception as e:
        await ctx.error(f"Unexpected error in calculate_obv for {inputs.symbol}: {e}")
        return ObvOutput(**output_base, error="An unexpected server error occurred.")


@mcp.tool()
async def get_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    获取加密货币K线数据

    该工具从加密货币交易所获取指定交易对的历史K线数据。K线图是技术分析的基础，
    提供了价格变动的完整信息，包括开盘价、最高价、最低价、收盘价和成交量。

    功能特点:
    - 支持100+主流交易所 (通过CCXT)
    - 多种时间框架支持
    - 实时数据获取
    - 自动数据验证和格式化
    - 统一的错误处理

    Args:
        inputs.symbol (str): 交易对符号，格式为 'BASE/QUOTE'
            - 主流币: 'BTC/USDT', 'ETH/USD', 'BNB/BTC'
            - 山寨币: 'DOGE/USDT', 'ADA/USD', 'DOT/BTC'
            - 注意大小写和分隔符格式
        inputs.timeframe (TimeFrame): 时间框架枚举值
            - 分钟级: '1m', '5m', '15m', '30m'
            - 小时级: '1h', '4h', '12h'
            - 日级: '1d', '3d'
            - 周月级: '1w', '1M'
        inputs.limit (int): 获取的K线数量，范围1-1000
            - 默认值: 100
            - 建议值: 根据分析需求调整
            - 注意: 过多数据可能影响性能
        inputs.since (Optional[int]): 起始时间戳(毫秒)，可选
            - 如果指定，从该时间开始获取数据
            - 如果不指定，获取最新数据

    Returns:
        CandlesOutput: K线数据输出对象
        - symbol: 交易对符号
        - timeframe: 时间框架字符串
        - candles: OHLCVCandle对象列表，按时间顺序排列
        - count: 实际返回的K线数量
        - error: 错误信息(如果有)

    每个K线包含:
    - timestamp: Unix时间戳(毫秒)
    - open: 开盘价
    - high: 最高价
    - low: 最低价
    - close: 收盘价
    - volume: 成交量

    Example:
        获取BTC/USDT的4小时K线:
        >>> inputs = CandlesInput(
        ...     symbol="BTC/USDT",
        ...     timeframe=TimeFrame.FOUR_HOURS,
        ...     limit=50
        ... )
        >>> result = await get_candles(ctx, inputs)
        >>> if result.candles:
        ...     print(f"获取到{result.count}根K线")
        ...
        ...     # 分析最新K线
        ...     latest = result.candles[-1]
        ...     print(f"最新价格: ${latest.close}")
        ...     print(f"24h最高: ${latest.high}")
        ...     print(f"24h最低: ${latest.low}")
        ...     print(f"成交量: {latest.volume}")
        ...
        ...     # 计算价格变化
        ...     if len(result.candles) > 1:
        ...         prev_close = result.candles[-2].close
        ...         price_change = latest.close - prev_close
        ...         change_pct = price_change / prev_close * 100
        ...         print(f"价格变化: ${price_change:.2f} ({change_pct:.2f}%)")
        ...
        ...     # 分析成交量
        ...     volumes = [candle.volume for candle in result.candles]
        ...     avg_volume = sum(volumes) / len(volumes)
        ...     if latest.volume > avg_volume * 1.5:
        ...         print("📈 当前成交量异常放大")

        获取特定时间段的数据:
        >>> from datetime import datetime
        >>> start_time = int(datetime(2024, 1, 1).timestamp() * 1000)
        >>> inputs = CandlesInput(
        ...     symbol="ETH/USD",
        ...     timeframe=TimeFrame.ONE_DAY,
        ...     limit=30,
        ...     since=start_time
        ... )
        >>> result = await get_candles(ctx, inputs)

    时间框架选择指南:
    - 超短线(秒杀): 1m, 5m
    - 短线交易: 15m, 30m, 1h
    - 波段交易: 4h, 1d
    - 长线投资: 1d, 1w, 1M

    数据质量保证:
    - 自动验证数据完整性
    - 过滤异常价格数据
    - 处理时间戳格式统一
    - 成交量数据验证

    常见用途:
    - 技术指标计算的数据源
    - 价格趋势分析
    - 支撑阻力位识别
    - 交易信号验证
    - 回测策略开发

    注意事项:
    - 不同交易所的数据可能略有差异
    - 新币种可能历史数据不足
    - 网络问题可能导致数据获取失败
    - 某些时间框架在部分交易所不支持

    错误处理:
    - 网络错误: 自动重试机制
    - 交易所错误: 详细错误信息
    - 数据格式错误: 自动格式转换
    - 参数错误: 输入验证和提示

    Note:
        - 实际获取的数据量可能少于请求量(取决于可用性)
        - 数据按时间顺序排列，最新数据在末尾
        - 建议批量获取数据以提高效率
        - 频繁请求可能触发交易所限制
    """
    await ctx.info(
        f"Fetching {inputs.limit} candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        # Fetch OHLCV data from exchange
        ohlcv_data = await fetch_ohlcv_data(
            ctx=ctx,
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            limit=inputs.limit,
            since=inputs.since,
        )

        if ohlcv_data:
            candles = [OHLCVCandle.from_list(candle) for candle in ohlcv_data]
            await ctx.info(
                f"Successfully fetched {len(candles)} candles for {inputs.symbol}"
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
                error=f"No candle data available for {inputs.symbol}",
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"Exchange/Network Error: {str(e)}",
        )
    except Exception as e:
        await ctx.error(f"Unexpected error fetching candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error="An unexpected server error occurred.",
        )


@mcp.tool()
async def get_current_price(inputs: PriceInput, ctx: Context) -> PriceOutput:
    """
    获取加密货币当前价格

    该工具从加密货币交易所获取指定交易对的实时价格信息。
    价格数据通常延迟不超过几秒，是短线交易和价格监控的重要工具。

    功能特点:
    - 实时价格获取(延迟<1秒)
    - 支持所有主流交易对
    - 包含时间戳信息
    - 自动错误处理和重试
    - 统一的数据格式

    Args:
        inputs.symbol (str): 交易对符号，格式为 'BASE/QUOTE'
            - 示例: 'BTC/USDT', 'ETH/USD', 'DOGE/BTC'
            - 支持所有在交易所上市的交易对

    Returns:
        PriceOutput: 价格数据输出对象
        - symbol: 交易对符号
        - price: 当前价格(最新成交价)
        - timestamp: 价格时间戳(毫秒)
        - error: 错误信息(如果有)

    Example:
        获取BTC当前价格:
        >>> inputs = PriceInput(symbol="BTC/USDT")
        >>> result = await get_current_price(ctx, inputs)
        >>> if result.price:
        ...     print(f"BTC当前价格: ${result.price:,.2f}")
        ...
        ...     # 时间戳转换
        ...     from datetime import datetime
        ...     price_time = datetime.fromtimestamp(result.timestamp / 1000)
        ...     print(f"价格时间: {price_time}")

        批量价格监控:
        >>> symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        >>> prices = {}
        >>> for symbol in symbols:
        ...     inputs = PriceInput(symbol=symbol)
        ...     result = await get_current_price(ctx, inputs)
        ...     if result.price:
        ...         prices[symbol] = result.price
        >>> print("当前价格:", prices)

    应用场景:
    - 实时价格监控
    - 交易信号触发
    - 投资组合估值
    - 价格提醒系统
    - 套利机会发现

    数据来源:
    - 交易所最新成交价
    - 通常是bid/ask的中间价
    - 数据来自真实交易
    - 24小时不间断更新

    Note:
        - 价格可能存在秒级延迟
        - 极端市场条件下可能获取失败
        - 建议添加价格合理性验证
        - 频繁请求可能触发限制
    """
    await ctx.info(f"Fetching current price for {inputs.symbol}")

    try:
        ticker = await fetch_ticker_data(ctx, inputs.symbol)
        if ticker and "last" in ticker and ticker["last"] is not None:
            await ctx.info(f"Current price for {inputs.symbol}: {ticker['last']}")
            return PriceOutput(
                symbol=inputs.symbol,
                price=ticker["last"],
                timestamp=ticker.get("timestamp"),
            )
        else:
            return PriceOutput(symbol=inputs.symbol, error="Price data not available")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching price for {inputs.symbol}: {e}")
        return PriceOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(f"Unexpected error fetching price for {inputs.symbol}: {e}")
        return PriceOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_ticker(ctx: Context, inputs: TickerInput) -> TickerOutput:
    """
    获取加密货币详细行情数据

    该工具获取指定交易对的全面行情信息，包括买卖价、开高低收、成交量、
    涨跌幅等关键市场数据。这是进行深度市场分析的重要数据源。

    功能特点:
    - 全面的市场数据
    - 实时更新(延迟<1秒)
    - 标准化数据格式
    - 包含统计信息
    - 支持所有主流交易对

    Args:
        inputs.symbol (str): 交易对符号

    Returns:
        TickerOutput: 详细行情数据输出对象
        - symbol: 交易对符号
        - bid: 买一价(最高买入价)
        - ask: 卖一价(最低卖出价)
        - last: 最新成交价
        - open: 24h开盘价
        - high: 24h最高价
        - low: 24h最低价
        - close: 当前价格(通常等于last)
        - volume: 24h成交量(基础货币)
        - percentage: 24h涨跌幅百分比
        - change: 24h价格变化(绝对值)
        - timestamp: 数据时间戳
        - error: 错误信息(如果有)

    Example:
        获取ETH详细行情:
        >>> inputs = TickerInput(symbol="ETH/USDT")
        >>> result = await get_ticker(ctx, inputs)
        >>> if not result.error:
        ...     print(f"ETH/USDT 详细行情:")
        ...     print(f"最新价: ${result.last:.2f}")
        ...     print(f"买一价: ${result.bid:.2f}")
        ...     print(f"卖一价: ${result.ask:.2f}")
        ...     print(f"24h最高: ${result.high:.2f}")
        ...     print(f"24h最低: ${result.low:.2f}")
        ...     print(f"24h成交量: {result.volume:,.2f} ETH")
        ...     print(f"24h涨跌幅: {result.percentage:.2f}%")
        ...
        ...     # 买卖价差分析
        ...     if result.bid and result.ask:
        ...         spread = result.ask - result.bid
        ...         spread_pct = spread / result.last * 100
        ...         print(f"买卖价差: ${spread:.4f} ({spread_pct:.3f}%)")
        ...
        ...     # 24h波动率
        ...     if result.high and result.low:
        ...         volatility = (result.high - result.low) / result.last * 100
        ...         print(f"24h波动率: {volatility:.2f}%")

    数据解读:
    1. 价格信息:
        - last: 最新成交价，最重要的价格参考
        - bid/ask: 买卖价差，反映流动性
        - open: 24h前的价格，用于计算涨跌

    2. 极值信息:
        - high/low: 24h价格区间，显示波动范围
        - 接近high: 可能存在阻力
        - 接近low: 可能存在支撑

    3. 成交量信息:
        - volume: 交易活跃度指标
        - 高成交量 + 价格突破: 信号更可靠
        - 低成交量: 价格变动可能不可持续

    4. 变化信息:
        - percentage: 标准化的涨跌幅
        - change: 绝对价格变化
        - 用于排序和筛选

    应用场景:
    - 市场概览和监控
    - 交易前的市场分析
    - 流动性评估
    - 价格提醒和报警
    - 套利机会识别
    - 风险管理决策

    分析技巧:
    - 买卖价差: 反映市场流动性
    - 成交量: 验证价格变动的可靠性
    - 24h极值: 确定支撑阻力位
    - 涨跌幅: 横向比较不同币种表现

    Note:
        - 数据来源于真实交易所
        - 24h统计数据基于UTC时间
        - 某些字段可能在部分交易所不可用
        - 建议结合K线数据进行分析
    """
    await ctx.info(f"Fetching ticker data for {inputs.symbol}")

    try:
        ticker = await fetch_ticker_data(ctx, inputs.symbol)
        if ticker:
            await ctx.info(f"Successfully fetched ticker for {inputs.symbol}")
            return TickerOutput(
                symbol=inputs.symbol,
                bid=ticker.get("bid"),
                ask=ticker.get("ask"),
                last=ticker.get("last"),
                open=ticker.get("open"),
                high=ticker.get("high"),
                low=ticker.get("low"),
                close=ticker.get("close"),
                volume=ticker.get("baseVolume"),
                percentage=ticker.get("percentage"),
                change=ticker.get("change"),
                timestamp=ticker.get("timestamp"),
            )
        else:
            return TickerOutput(symbol=inputs.symbol, error="Ticker data not available")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching ticker for {inputs.symbol}: {e}")
        return TickerOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(f"Unexpected error fetching ticker for {inputs.symbol}: {e}")
        return TickerOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_order_book(ctx: Context, inputs: OrderBookInput) -> OrderBookOutput:
    """
    获取加密货币订单簿数据

    订单簿(Order Book)显示了市场中所有未成交的买卖订单，是理解市场深度、
    流动性和短期价格走向的重要工具。通过分析订单簿，可以识别支撑阻力位、
    评估市场流动性，以及预测短期价格变动。

    订单簿构成:
    - 买盘(Bids): 所有待成交的买入订单，按价格从高到低排列
    - 卖盘(Asks): 所有待成交的卖出订单，按价格从低到高排列
    - 价格档位: 每个价格级别的订单数量
    - 市场深度: 各价格水平的流动性分布

    功能特点:
    - 实时订单簿数据
    - 可配置深度级别
    - 标准化数据格式
    - 自动排序和验证
    - 流动性分析支持

    Args:
        inputs.symbol (str): 交易对符号
        inputs.limit (int): 获取的买卖盘档位数量，范围1-100
            - 5-10档: 快速分析最佳买卖价
            - 20-50档: 标准市场深度分析
            - 50-100档: 深度流动性分析

    Returns:
        OrderBookOutput: 订单簿数据输出对象
        - symbol: 交易对符号
        - bids: 买盘档位列表，按价格从高到低排序
        - asks: 卖盘档位列表，按价格从低到高排序
        - timestamp: 数据时间戳
        - error: 错误信息(如果有)

    每个档位包含:
    - price: 价格水平
    - amount: 该价格的订单数量

    Example:
        分析BTC订单簿:
        >>> inputs = OrderBookInput(symbol="BTC/USDT", limit=20)
        >>> result = await get_order_book(ctx, inputs)
        >>> if not result.error:
        ...     print(f"BTC/USDT 订单簿分析:")
        ...
        ...     # 最佳买卖价
        ...     if result.bids and result.asks:
        ...         best_bid = result.bids[0].price
        ...         best_ask = result.asks[0].price
        ...         spread = best_ask - best_bid
        ...         spread_pct = spread / best_bid * 100
        ...
        ...         print(f"最佳买价: ${best_bid:.2f}")
        ...         print(f"最佳卖价: ${best_ask:.2f}")
        ...         print(f"买卖价差: ${spread:.2f} ({spread_pct:.3f}%)")
        ...
        ...     # 流动性分析
        ...     total_bid_volume = sum(bid.amount for bid in result.bids)
        ...     total_ask_volume = sum(ask.amount for ask in result.asks)
        ...
        ...     print(f"买盘总量: {total_bid_volume:.4f} BTC")
        ...     print(f"卖盘总量: {total_ask_volume:.4f} BTC")
        ...
        ...     # 买卖盘平衡
        ...     if total_bid_volume > 0 and total_ask_volume > 0:
        ...         balance_ratio = total_bid_volume / total_ask_volume
        ...         if balance_ratio > 1.2:
        ...             print("📈 买盘力量强于卖盘")
        ...         elif balance_ratio < 0.8:
        ...             print("📉 卖盘力量强于买盘")
        ...         else:
        ...             print("⚖️ 买卖盘相对平衡")
        ...
        ...     # 价格冲击分析
        ...     def calculate_impact(orders, target_volume):
        ...         volume_filled = 0
        ...         total_cost = 0
        ...         for order in orders:
        ...             if volume_filled >= target_volume:
        ...                 break
        ...             fill_volume = min(order.amount, target_volume - volume_filled)
        ...             total_cost += fill_volume * order.price
        ...             volume_filled += fill_volume
        ...         return total_cost / volume_filled if volume_filled > 0 else 0
        ...
        ...     # 假设买入1个BTC的价格冲击
        ...     if result.asks:
        ...         impact_price = calculate_impact(result.asks, 1.0)
        ...         market_price = result.asks[0].price
        ...         impact_pct = (impact_price - market_price) / market_price * 100
        ...         print(f"买入1 BTC的平均价格: ${impact_price:.2f}")
        ...         print(f"价格冲击: {impact_pct:.3f}%")

    分析指标:
    1. 买卖价差(Spread):
        - 价差小: 流动性好，交易成本低
        - 价差大: 流动性差，需谨慎交易
        - 价差突然扩大: 可能有重大消息

    2. 市场深度:
        - 深度好: 大单交易不会显著影响价格
        - 深度差: 容易出现价格滑点
        - 单边深度: 可能预示价格方向

    3. 订单分布:
        - 买卖盘平衡: 市场相对稳定
        - 买盘堆积: 下方支撑强，可能上涨
        - 卖盘堆积: 上方阻力大，可能下跌

    4. 大单分析:
        - 大买单: 可能的支撑位
        - 大卖单: 可能的阻力位
        - 订单墙: 心理价位，需要关注

    交易应用:
    1. 入场时机:
        - 买卖价差小时入场，降低成本
        - 观察订单簿变化，寻找最佳时机

    2. 价格预测:
        - 大单支撑/阻力位预测转折点
        - 订单簿失衡预示价格方向

    3. 风险管理:
        - 评估流动性，确定合适的仓位
        - 避免在流动性差时大额交易

    4. 市场操作:
        - 识别虚假订单和操纵行为
        - 判断真实的买卖意图

    高级分析:
    - 订单簿热力图: 可视化价格分布
    - 订单流分析: 追踪大单进出
    - 微观结构: 研究高频交易行为
    - 流动性挖掘: 寻找最佳执行策略

    注意事项:
    - 订单簿数据变化很快，需要实时更新
    - 虚假订单可能误导分析
    - 市场深度在不同时段差异很大
    - 机器人交易可能影响订单簿形状

    Note:
        - 数据实时性要求高，建议频繁更新
        - 大额交易前务必分析订单簿
        - 结合成交记录验证订单簿信号
        - 不同交易所的订单簿可能差异较大
    """
    await ctx.info(f"Fetching order book for {inputs.symbol} (limit: {inputs.limit})")

    try:
        order_book = await fetch_order_book(ctx, inputs.symbol, inputs.limit)
        if order_book:
            bids = [
                OrderBookLevel(price=bid[0], amount=bid[1])
                for bid in order_book.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=ask[0], amount=ask[1])
                for ask in order_book.get("asks", [])
            ]

            await ctx.info(f"Successfully fetched order book for {inputs.symbol}")
            return OrderBookOutput(
                symbol=inputs.symbol,
                bids=bids,
                asks=asks,
                timestamp=order_book.get("timestamp"),
            )
        else:
            return OrderBookOutput(
                symbol=inputs.symbol, error="Order book data not available"
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching order book for {inputs.symbol}: {e}")
        return OrderBookOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching order book for {inputs.symbol}: {e}",
        )
        return OrderBookOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_recent_trades(ctx: Context, inputs: TradesInput) -> TradesOutput:
    """
    获取加密货币最近成交记录

    成交记录(Trade History)显示了市场中最近发生的真实交易，包括成交价格、
    数量、时间和买卖方向。这是分析市场活跃度、价格发现机制和交易行为的
    重要数据源，对短线交易和市场微观结构分析特别有价值。

    成交记录信息:
    - 成交价格: 实际交易发生的价格
    - 成交数量: 交易的数量(基础货币)
    - 成交时间: 精确到毫秒的时间戳
    - 交易方向: 买入(taker买)或卖出(taker卖)
    - 交易ID: 唯一标识符

    功能特点:
    - 实时成交数据
    - 精确时间戳
    - 买卖方向识别
    - 可配置数据量
    - 标准化格式输出

    Args:
        inputs.symbol (str): 交易对符号
        inputs.limit (int): 获取的成交记录数量，范围1-500
            - 10-20条: 快速了解最新交易
            - 50-100条: 标准交易分析
            - 200-500条: 深度交易模式分析

    Returns:
        TradesOutput: 成交记录输出对象
        - symbol: 交易对符号
        - trades: 成交记录列表，按时间倒序排列(最新在前)
        - count: 实际返回的记录数量
        - error: 错误信息(如果有)

    每条成交记录包含:
    - id: 成交ID(可选)
    - timestamp: 成交时间戳(毫秒)
    - price: 成交价格
    - amount: 成交数量
    - side: 交易方向('buy'或'sell')

    Example:
        分析ETH最近成交:
        >>> inputs = TradesInput(symbol="ETH/USDT", limit=100)
        >>> result = await get_recent_trades(ctx, inputs)
        >>> if result.trades:
        ...     print(f"ETH/USDT 最近{result.count}笔成交分析:")
        ...
        ...     # 基础统计
        ...     prices = [trade.price for trade in result.trades]
        ...     volumes = [trade.amount for trade in result.trades]
        ...
        ...     avg_price = sum(prices) / len(prices)
        ...     total_volume = sum(volumes)
        ...     max_price = max(prices)
        ...     min_price = min(prices)
        ...
        ...     print(f"平均成交价: ${avg_price:.2f}")
        ...     print(f"价格区间: ${min_price:.2f} - ${max_price:.2f}")
        ...     print(f"总成交量: {total_volume:.4f} ETH")
        ...
        ...     # 买卖力量分析
        ...     buy_trades = [t for t in result.trades if t.side == 'buy']
        ...     sell_trades = [t for t in result.trades if t.side == 'sell']
        ...
        ...     buy_volume = sum(t.amount for t in buy_trades)
        ...     sell_volume = sum(t.amount for t in sell_trades)
        ...
        ...     print(f"买入笔数: {len(buy_trades)}")
        ...     print(f"卖出笔数: {len(sell_trades)}")
        ...     print(f"买入量: {buy_volume:.4f} ETH")
        ...     print(f"卖出量: {sell_volume:.4f} ETH")
        ...
        ...     # 市场情绪分析
        ...     if total_volume > 0:
        ...         buy_ratio = buy_volume / total_volume * 100
        ...         print(f"买盘占比: {buy_ratio:.2f}%")
        ...
        ...         if buy_ratio > 60:
        ...             print("💪 买盘占优，市场偏向乐观")
        ...         elif buy_ratio < 40:
        ...             print("📉 卖盘占优，市场偏向悲观")
        ...         else:
        ...             print("⚖️ 买卖相对平衡")
        ...
        ...     # 大单分析
        ...     avg_trade_size = total_volume / len(result.trades)
        ...     large_trades = [t for t in result.trades if t.amount > avg_trade_size * 3]
        ...
        ...     if large_trades:
        ...         print(f"检测到{len(large_trades)}笔大单交易")
        ...         for trade in large_trades[:5]:  # 显示前5笔
        ...             direction = "买入" if trade.side == 'buy' else "卖出"
        ...             print(f"  - {direction}: {trade.amount:.4f} ETH @ ${trade.price:.2f}")
        ...
        ...     # 价格趋势分析
        ...     if len(result.trades) >= 10:
        ...         recent_trades = result.trades[:10]  # 最近10笔
        ...         older_trades = result.trades[-10:]   # 较早10笔
        ...
        ...         recent_avg = sum(t.price for t in recent_trades) / len(recent_trades)
        ...         older_avg = sum(t.price for t in older_trades) / len(older_trades)
        ...
        ...         price_change = (recent_avg - older_avg) / older_avg * 100
        ...         if price_change > 0.1:
        ...             print(f"📈 短期价格上升趋势: +{price_change:.2f}%")
        ...         elif price_change < -0.1:
        ...             print(f"📉 短期价格下降趋势: {price_change:.2f}%")
        ...         else:
        ...             print("➡️ 短期价格相对稳定")

    分析维度:
    1. 成交量分析:
        - 总成交量: 市场活跃度指标
        - 单笔平均: 交易者类型推断
        - 大单占比: 机构活动程度

    2. 价格分析:
        - 成交价分布: 价格发现效率
        - 价格趋势: 短期方向判断
        - 价格波动: 市场稳定性

    3. 时间分析:
        - 交易频率: 市场流动性
        - 时间间隔: 交易密集度
        - 交易节奏: 市场状态判断

    4. 方向分析:
        - 买卖比例: 市场情绪指标
        - 主动买卖: taker行为分析
        - 方向变化: 情绪转换信号

    交易应用:
    1. 入场时机:
        - 观察成交密集度，选择活跃时段
        - 分析买卖力量，判断入场方向

    2. 价格预测:
        - 大单方向预示短期走势
        - 成交量异常可能预示变盘

    3. 风险控制:
        - 成交稀少时避免大额交易
        - 观察市场深度变化

    4. 策略优化:
        - 分析最佳成交时机
        - 优化订单执行策略

    高级分析技巧:
    - 成交量加权平均价格(VWAP)
    - 买卖压力指数计算
    - 交易者行为模式识别
    - 市场微观结构分析

    数据质量:
    - 数据来源于真实交易
    - 时间戳精确到毫秒
    - 买卖方向基于taker识别
    - 过滤异常和错误交易

    注意事项:
    - 成交记录更新频率很高
    - 大量数据可能影响性能
    - 不同交易所的side定义可能不同
    - 需要结合订单簿数据综合分析

    Note:
        - 数据按时间倒序排列，最新交易在前
        - side字段表示taker的方向
        - 建议配合实时数据使用
        - 可用于验证技术分析信号
    """
    await ctx.info(f"Fetching {inputs.limit} recent trades for {inputs.symbol}")

    try:
        trades_data = await fetch_trades(ctx, inputs.symbol, inputs.limit)
        if trades_data:
            trades = [
                TradeData(
                    id=trade.get("id"),
                    timestamp=trade["timestamp"],
                    price=trade["price"],
                    amount=trade["amount"],
                    side=trade["side"],
                )
                for trade in trades_data
            ]

            await ctx.info(
                f"Successfully fetched {len(trades)} trades for {inputs.symbol}"
            )
            return TradesOutput(symbol=inputs.symbol, trades=trades, count=len(trades))
        else:
            return TradesOutput(
                symbol=inputs.symbol, error="Recent trades data not available"
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching trades for {inputs.symbol}: {e}")
        return TradesOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(f"Unexpected error fetching trades for {inputs.symbol}: {e}")
        return TradesOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def generate_comprehensive_market_report(
    ctx: Context, inputs: ComprehensiveAnalysisInput
) -> ComprehensiveAnalysisOutput:
    """
    生成加密货币综合市场分析报告

    该工具是本模块的核心功能，它整合多个技术指标和市场数据，
    生成全面的加密货币技术分析报告。报告包含数值结果、趋势分析、
    交易信号和投资建议，是专业交易者和投资者的重要决策工具。

    功能特点:
    - 多指标综合分析
    - 智能信号识别
    - 趋势强度评估
    - 风险评级系统
    - 个性化参数配置
    - 结构化数据输出
    - 专业分析报告

    支持的技术指标:
    - SMA: 简单移动平均线(趋势跟踪)
    - RSI: 相对强弱指数(超买超卖)
    - MACD: 异同移动平均线(动量分析)
    - BBANDS: 布林带(波动率分析)
    - ATR: 平均真实波幅(风险测量)
    - ADX: 平均方向指数(趋势强度)
    - OBV: 成交量平衡指标(资金流向)

    Args:
        inputs.symbol (str): 加密货币交易对
        inputs.timeframe (str): 分析时间框架，默认"1h"
        inputs.history_len (int): 历史数据长度，默认20
        inputs.indicators_to_include (List[str]): 要分析的指标列表
            - None: 使用默认全套指标
            - 自定义: ["SMA", "RSI", "MACD"] 等
        inputs.sma_period (int): SMA周期，默认20
        inputs.rsi_period (int): RSI周期，默认14
        inputs.macd_fast_period (int): MACD快线，默认12
        inputs.macd_slow_period (int): MACD慢线，默认26
        inputs.macd_signal_period (int): MACD信号线，默认9
        inputs.bbands_period (int): 布林带周期，默认20
        inputs.atr_period (int): ATR周期，默认14
        inputs.adx_period (int): ADX周期，默认14
        inputs.obv_data_points (int): OBV数据点，默认50

    Returns:
        ComprehensiveAnalysisOutput: 综合分析结果
        - symbol: 交易对符号
        - timeframe: 时间框架
        - report_text: 完整的文字分析报告
        - structured_data: 结构化的指标数据
        - error: 错误信息(如果有)

    报告内容结构:
    1. 报告头部:
        - 分析对象和时间
        - 数据来源说明
        - 分析参数概览

    2. 技术指标分析:
        - 各指标最新值
        - 信号强度评估
        - 买卖建议
        - 风险提示

    3. 趋势分析:
        - 主要趋势方向
        - 趋势强度评级
        - 支撑阻力位
        - 关键价格位

    4. 交易建议:
        - 入场时机建议
        - 止损止盈设置
        - 仓位管理建议
        - 风险评级

    5. 市场情绪:
        - 超买超卖状态
        - 市场情绪指数
        - 资金流向分析
        - 波动率评估

    Example:
        生成BTC完整分析报告:
        >>> inputs = ComprehensiveAnalysisInput(
        ...     symbol="BTC/USDT",
        ...     timeframe="4h",
        ...     history_len=30,
        ...     indicators_to_include=["SMA", "RSI", "MACD", "BBANDS", "ATR"]
        ... )
        >>> result = await generate_comprehensive_market_report(ctx, inputs)
        >>> if not result.error:
        ...     print(result.report_text)
        ...
        ...     # 提取关键信号
        ...     signals = []
        ...     if result.structured_data:
        ...         if "rsi" in result.structured_data:
        ...             rsi_data = result.structured_data["rsi"]
        ...             if rsi_data.get("rsi"):
        ...                 latest_rsi = rsi_data["rsi"][-1]
        ...                 if latest_rsi > 70:
        ...                     signals.append("RSI超买")
        ...                 elif latest_rsi < 30:
        ...                     signals.append("RSI超卖")
        ...
        ...         if "macd" in result.structured_data:
        ...             macd_data = result.structured_data["macd"]
        ...             if (macd_data.get("macd") and macd_data.get("signal") and
        ...                 len(macd_data["macd"]) > 1):
        ...                 if (macd_data["macd"][-1] > macd_data["signal"][-1] and
        ...                     macd_data["macd"][-2] <= macd_data["signal"][-2]):
        ...                     signals.append("MACD金叉")
        ...
        ...     if signals:
        ...         print(f"检测到信号: {', '.join(signals)}")

        快速市场扫描:
        >>> symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        >>> for symbol in symbols:
        ...     inputs = ComprehensiveAnalysisInput(
        ...         symbol=symbol,
        ...         indicators_to_include=["RSI", "MACD"],
        ...         history_len=10
        ...     )
        ...     result = await generate_comprehensive_market_report(ctx, inputs)
        ...     print(f"{symbol}: {result.report_text[:200]}...")

    报告解读指南:
    1. 技术指标权重:
        - 趋势指标(SMA, MACD): 中长期方向
        - 震荡指标(RSI): 短期超买超卖
        - 波动率指标(ATR, BBANDS): 风险评估
        - 成交量指标(OBV): 资金确认

    2. 信号可靠性:
        - 多指标一致: 信号强度高
        - 指标背离: 需要谨慎
        - 极端读数: 关注反转
        - 趋势确认: 等待多重验证

    3. 时间框架影响:
        - 短周期: 信号频繁但噪音多
        - 长周期: 信号稳定但滞后
        - 多周期: 综合确认最可靠

    使用建议:
    1. 交易前分析:
        - 生成完整报告
        - 重点关注信号一致性
        - 评估风险回报比

    2. 持仓管理:
        - 定期更新分析
        - 关注趋势变化
        - 及时调整策略

    3. 风险控制:
        - 参考ATR设置止损
        - 观察RSI避免追高杀跌
        - 结合成交量确认信号

    高级应用:
    - 多币种比较分析
    - 策略回测验证
    - 风险评估系统
    - 自动化交易信号

    局限性说明:
    - 技术分析不能预测未来
    - 突发事件可能使分析失效
    - 需要结合基本面分析
    - 不构成投资建议

    Note:
        - 报告内容仅供参考，不构成投资建议
        - 加密货币投资存在高风险
        - 建议在模拟环境中验证策略
        - 请根据自身风险承受能力投资
    """
    await ctx.info(
        f"Generating comprehensive report for {inputs.symbol} ({inputs.timeframe}) with {inputs.history_len} data points."
    )
    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # Determine which indicators to run
    default_indicators = ["SMA", "RSI", "MACD", "BBANDS", "ATR", "ADX", "OBV"]
    indicators_to_run = (
        inputs.indicators_to_include
        if inputs.indicators_to_include is not None
        else default_indicators
    )

    try:
        # --- SMA ---
        if "SMA" in indicators_to_run:
            sma_period = inputs.sma_period or settings.DEFAULT_SMA_PERIOD
            sma_input = SmaInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=sma_period,
            )
            sma_output = await calculate_sma.run({"ctx": ctx, "inputs": sma_input})
            indicator_results_structured["sma"] = json.loads(
                sma_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["sma"]["sma"] is not None
                and len(indicator_results_structured["sma"]["sma"]) > 0
            ):
                latest_sma = indicator_results_structured["sma"]["sma"][-1]
                report_sections.append(
                    f"- SMA({indicator_results_structured['sma']['period']}): {latest_sma:.4f} (Latest)"
                )
                if len(indicator_results_structured["sma"]["sma"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["sma"]["sma"][-1]
                        > indicator_results_structured["sma"]["sma"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Trend: {trend} ({len(indicator_results_structured['sma']['sma'])} data points)"
                    )
            elif indicator_results_structured["sma"].get("error"):
                report_sections.append(
                    f"- SMA: Error - {indicator_results_structured['sma']['error']}"
                )

        # --- RSI ---
        if "RSI" in indicators_to_run:
            rsi_period = inputs.rsi_period or settings.DEFAULT_RSI_PERIOD
            rsi_input = RsiInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=rsi_period,
            )
            rsi_output = await calculate_rsi.run({"ctx": ctx, "inputs": rsi_input})
            indicator_results_structured["rsi"] = json.loads(
                rsi_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["rsi"]["rsi"] is not None
                and len(indicator_results_structured["rsi"]["rsi"]) > 0
            ):
                latest_rsi = indicator_results_structured["rsi"]["rsi"][-1]
                report_sections.append(
                    f"- RSI({indicator_results_structured['rsi']['period']}): {latest_rsi:.2f}"
                )
                if latest_rsi > 70:
                    report_sections.append(
                        "  - Note: RSI suggests overbought conditions (>70)."
                    )
                elif latest_rsi < 30:
                    report_sections.append(
                        "  - Note: RSI suggests oversold conditions (<30)."
                    )

                # Add trend analysis if we have multiple data points
                if len(indicator_results_structured["rsi"]["rsi"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["rsi"]["rsi"][-1]
                        > indicator_results_structured["rsi"]["rsi"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Trend: {trend} ({len(indicator_results_structured['rsi']['rsi'])} data points)"
                    )
            elif indicator_results_structured["rsi"].get("error"):
                report_sections.append(
                    f"- RSI: Error - {indicator_results_structured['rsi']['error']}"
                )

        # --- MACD ---
        if "MACD" in indicators_to_run:
            macd_input = MacdInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                fast_period=inputs.macd_fast_period or settings.DEFAULT_MACD_FAST,
                slow_period=inputs.macd_slow_period or settings.DEFAULT_MACD_SLOW,
                signal_period=inputs.macd_signal_period or settings.DEFAULT_MACD_SIGNAL,
            )
            macd_output = await calculate_macd.run({"ctx": ctx, "inputs": macd_input})
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
                    f"- MACD({indicator_results_structured['macd']['fast_period']},{indicator_results_structured['macd']['slow_period']},{indicator_results_structured['macd']['signal_period']}): "
                    f"MACD: {latest_macd:.4f}, Signal: {latest_signal:.4f}, Hist: {latest_hist:.4f}"
                )

                if latest_hist > 0 and latest_macd > latest_signal:
                    report_sections.append(
                        "  - Note: MACD histogram positive, potentially bullish momentum."
                    )
                elif latest_hist < 0 and latest_macd < latest_signal:
                    report_sections.append(
                        "  - Note: MACD histogram negative, potentially bearish momentum."
                    )

                # Add trend analysis
                if len(indicator_results_structured["macd"]["histogram"]) > 1:
                    hist_trend = (
                        "↗"
                        if indicator_results_structured["macd"]["histogram"][-1]
                        > indicator_results_structured["macd"]["histogram"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Histogram Trend: {hist_trend} ({len(indicator_results_structured['macd']['histogram'])} data points)"
                    )

            elif indicator_results_structured["macd"].get("error"):
                report_sections.append(
                    f"- MACD: Error - {indicator_results_structured['macd']['error']}"
                )

        # --- Bollinger Bands (BBANDS) ---
        if "BBANDS" in indicators_to_run:
            bbands_input = BbandsInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.bbands_period or settings.DEFAULT_BBANDS_PERIOD,
            )
            bbands_output = await calculate_bbands.run(
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
                    f"- Bollinger Bands({indicator_results_structured['bbands']['period']}, {indicator_results_structured['bbands']['nbdevup']}dev): "
                    f"Upper: {latest_upper:.4f}, Middle: {latest_middle:.4f}, Lower: {latest_lower:.4f}"
                )
                report_sections.append(
                    f"  - Band Width: {latest_upper - latest_lower:.4f}"
                )

                # Add trend analysis
                if len(indicator_results_structured["bbands"]["middle_band"]) > 1:
                    trend = (
                        "↗"
                        if indicator_results_structured["bbands"]["middle_band"][-1]
                        > indicator_results_structured["bbands"]["middle_band"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Middle Band Trend: {trend} ({len(indicator_results_structured['bbands']['middle_band'])} data points)"
                    )

            elif indicator_results_structured["bbands"].get("error"):
                report_sections.append(
                    f"- BBANDS: Error - {indicator_results_structured['bbands']['error']}"
                )

        # --- Average True Range (ATR) ---
        if "ATR" in indicators_to_run:
            atr_input = AtrInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.atr_period or settings.DEFAULT_ATR_PERIOD,
            )
            atr_output = await calculate_atr.run({"ctx": ctx, "inputs": atr_input})
            indicator_results_structured["atr"] = json.loads(
                atr_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["atr"]["atr"] is not None
                and len(indicator_results_structured["atr"]["atr"]) > 0
            ):
                latest_atr = indicator_results_structured["atr"]["atr"][-1]
                report_sections.append(
                    f"- ATR({indicator_results_structured['atr']['period']}): {latest_atr:.4f} (Volatility Measure)"
                )

                # Add trend analysis for volatility
                if len(indicator_results_structured["atr"]["atr"]) > 1:
                    vol_trend = (
                        "↗"
                        if indicator_results_structured["atr"]["atr"][-1]
                        > indicator_results_structured["atr"]["atr"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Volatility Trend: {vol_trend} ({len(indicator_results_structured['atr']['atr'])} data points)"
                    )

            elif indicator_results_structured["atr"].get("error"):
                report_sections.append(
                    f"- ATR: Error - {indicator_results_structured['atr']['error']}"
                )

        # --- Average Directional Index (ADX) ---
        if "ADX" in indicators_to_run:
            adx_input = AdxInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.adx_period or settings.DEFAULT_ADX_PERIOD,
            )
            adx_output = await calculate_adx.run({"ctx": ctx, "inputs": adx_input})
            indicator_results_structured["adx"] = json.loads(
                adx_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["adx"]["adx"] is not None
                and len(indicator_results_structured["adx"]["adx"]) > 0
                and indicator_results_structured["adx"]["plus_di"] is not None
                and len(indicator_results_structured["adx"]["plus_di"]) > 0
                and indicator_results_structured["adx"]["minus_di"] is not None
                and len(indicator_results_structured["adx"]["minus_di"]) > 0
            ):
                latest_adx = indicator_results_structured["adx"]["adx"][-1]
                latest_plus_di = indicator_results_structured["adx"]["plus_di"][-1]
                latest_minus_di = indicator_results_structured["adx"]["minus_di"][-1]

                report_sections.append(
                    f"- ADX({indicator_results_structured['adx']['period']}): {latest_adx:.2f}, +DI: {latest_plus_di:.2f}, -DI: {latest_minus_di:.2f}"
                )

                if latest_adx > 25:
                    report_sections.append(
                        f"  - Note: ADX ({latest_adx:.2f}) suggests a trending market."
                    )
                else:
                    report_sections.append(
                        f"  - Note: ADX ({latest_adx:.2f}) suggests a weak or non-trending market."
                    )

                # Directional analysis
                if latest_plus_di > latest_minus_di:
                    report_sections.append("  - Direction: Bullish (+DI > -DI)")
                else:
                    report_sections.append("  - Direction: Bearish (-DI > +DI)")

                # Add trend analysis
                if len(indicator_results_structured["adx"]["adx"]) > 1:
                    trend_strength = (
                        "↗"
                        if indicator_results_structured["adx"]["adx"][-1]
                        > indicator_results_structured["adx"]["adx"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Trend Strength: {trend_strength} ({len(indicator_results_structured['adx']['adx'])} data points)"
                    )

            elif indicator_results_structured["adx"].get("error"):
                report_sections.append(
                    f"- ADX: Error - {indicator_results_structured['adx']['error']}"
                )

        # --- On-Balance Volume (OBV) ---
        if "OBV" in indicators_to_run:
            obv_input = ObvInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                data_points=inputs.obv_data_points or settings.DEFAULT_OBV_DATA_POINTS,
            )
            obv_output = await calculate_obv.run({"ctx": ctx, "inputs": obv_input})
            indicator_results_structured["obv"] = json.loads(
                obv_output[0].model_dump()["text"]
            )

            if (
                indicator_results_structured["obv"]["obv"] is not None
                and len(indicator_results_structured["obv"]["obv"]) > 0
            ):
                latest_obv = indicator_results_structured["obv"]["obv"][-1]
                report_sections.append(
                    f"- OBV (using {indicator_results_structured['obv']['data_points']} points): {latest_obv:.2f}"
                )

                # Add trend analysis for volume flow
                if len(indicator_results_structured["obv"]["obv"]) > 1:
                    volume_trend = (
                        "↗"
                        if indicator_results_structured["obv"]["obv"][-1]
                        > indicator_results_structured["obv"]["obv"][-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Volume Flow: {volume_trend} ({len(indicator_results_structured['obv']['obv'])} data points)"
                    )

            elif indicator_results_structured["obv"].get("error"):
                report_sections.append(
                    f"- OBV: Error - {indicator_results_structured['obv']['error']}"
                )

        # --- Synthesize the report ---
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base,
                error="No indicator data could be calculated or selected.",
            )

        report_title = f"Comprehensive Technical Analysis for {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} Data Points:\n"
        report_text = report_title + "\n".join(report_sections)

        # Add summary section with overall trend analysis
        summary_sections = []
        trend_indicators = []

        # Collect trend information from indicators that have it
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
            summary_sections.append("\nTrend Summary:")
            summary_sections.extend([f"  {trend}" for trend in trend_indicators])
            report_text += "\n" + "\n".join(summary_sections)

        await ctx.info(
            f"Successfully generated comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
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
            f"Unexpected error in generate_comprehensive_market_report for {inputs.symbol}: {e}",
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"An unexpected server error occurred: {str(e)}"
        )
