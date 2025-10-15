# filename: futures_usdtm_1h_refresh_5m_breakout_breakdown_safeEntry_safeDist_dynTPSL_heartbeat.py
import os
import time
import math
import ccxt
import pandas as pd
import requests
from datetime import datetime, timezone

# ======================== Config ========================
DEBUG = True
EXCHANGE_ID = "binance"
MARKET_TYPE = "future"  # Binance USDT-M 永续
QUOTE = "USDT"
TIMEFRAME_HOURLY = "1h"
TIMEFRAME_FAST = "5m"

TOP_N = 80  # 24h 涨幅榜初选数量
RANK_BY_1H = 40  # 最终进入 5m 扫描的符号数量上限

LOOKBACK_VOL = 50
BREAKOUT_WINDOW = 20
MIN_BASE_VOLUME_USD = 500_000
SLEEP_MS = 10
LOOP_INTERVAL_SEC = 300
HOURLY_REFRESH_JITTER = 30

# —— 多空 5m 即时阈值 ——
M5_PCT_RISE = 0.6
M5_VOL_RATIO_STRONG = 2.0
M5_PCT_DROP = 0.6
M5_VOL_RATIO_STRONG_BEAR = 2.0

# —— 辅助 flags（不触发，仅显示用）——
M5_STAGNATION_VOL = 0.9
M5_DUMP_PCT = -1.5
M5_DUMP_VOL_MAX = 1.0

# —— 预备期（整体关闭）——
PREP_ENABLED = True
PREP_N = 5
PREP_MIN_CUM_PCT = 0.4
PREP_MAX_CUM_PCT = 3.0
PREP_MIN_POS_RATIO = 0.8
PREP_MIN_CLOSE_SPEARMAN = 0.85
PREP_MIN_VOL_SPEARMAN = 0.20
PREP_MAX_SINGLE_BAR_PCT = 1.2
PREP_MIN_BODY_TO_RANGE = 0.35
PREP_LAST_VOL_TO_MA_MIN = 1.0
PREP_LAST_VOL_TO_MA_MAX = 2.5

# —— 近1小时趋势过滤 ——
TREND_LOOKBACK_BARS = 12  # 12×5m = 60m
TREND_MIN_SPEARMAN = 0.25
TREND_MIN_NET_PCT_UP = 0.20
TREND_MIN_NET_PCT_DN = 0.20
TREND_MA_SHORT = 4
TREND_MA_LONG = 12

# —— 1h 强势（多头必需；空头可选）——
ONE_H_STRONG_PCT = 0.8
ONE_H_STRONG_VOL = 1.5
REQUIRE_1H_DOWN_FOR_BEAR = False

# Tick 过滤
MAX_TICK_TO_PRICE = 0.002
BREAKOUT_TICKS = 1
BREAKDOWN_TICKS = 1

# —— 保守入场开关与参数 ——
SAFE_MODE_ALWAYS = False  # True = 只推保守入场命令
SAFE_FIB_RATIO = 0.382
SAFE_ATR_MULT = 0.5
SAFE_RETEST_TICKS = 1
SAFE_SL_PCT = 5.0
SAFE_MIN_SL_PCT = 1.5
SAFE_MAX_SL_PCT = 6.0
SAFE_TP_PCTS = (1.5, 3.0, 6.0)

# —— 与现价的“最小安全距离” ——
SAFE_MIN_DIST_ATR = 0.25  # 保守入场与现价至少相距 0.25×ATR
SAFE_MIN_DIST_TICKS = 3  # 或至少 3 个 tick（两者取较大）

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

# 冷却 & 心跳
ALERT_COOLDOWN_SEC = 30 * 60
HEARTBEAT_ENABLED = True
HEARTBEAT_JITTER = 30

# ======== 新增：候选“补充来源”参数 ========
# （A）基于24h成交额（quoteVolume）的补充 —— 只用 tickers 排序，不拉K线
ADD_BY_VOLUME_SKIP_TOPK = 5  # 跳过成交额前K（通常是BTC/ETH等超级大盘，避免占坑）
LIQ_RETURN_TOP = 30  # 最终返回的流动性候选上限

# （B）基于结构的 Near-Breakout / Near-Breakdown（两段式）
NEAR_HHV_MAX_DIST_PCT = 0.8  # ≤ 该距离（%）视为“临近前高”
NEAR_LLV_MAX_DIST_PCT = 0.8  # ≤ 该距离（%）视为“临近前低”
NEAR_VOLR_MIN = 0.7  # 1h量能/MA 下限（不过冷）
NEAR_VOLR_MAX = 1.5  # 1h量能/MA 上限（不过热）
STRUCT_PREFILTER_TOP = 120  # 轻筛后进入精筛的最大数量（控制 OHLCV 调用量）
STRUCT_RETURN_TOP = 30  # 最终返回的结构候选上限（每个方向）

# ======== 全局候选上限与分路配额（可调建议：80~100）=======
MAX_MERGED_CANDIDATES = 100  # 合并后用于小时分析的最大候选数
# 三路配额（合并前先裁剪；合并时再应用全局上限）
MOMENTUM_QUOTA = 40  # 动量（24h涨幅→1h快筛）
LIQUIDITY_QUOTA = 30  # 24h成交额补充
STRUCT_UP_QUOTA = 20  # 结构：临近前高
STRUCT_DN_QUOTA = 20  # 结构：临近期低
# ========================================================


def ts_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def dbg(msg):
    if DEBUG:
        print(f"[DEBUG {ts_now()}] {msg}")


def build_exchange():
    dbg(f"Connecting {EXCHANGE_ID} (type={MARKET_TYPE})")
    ex = getattr(ccxt, EXCHANGE_ID)(
        {
            "enableRateLimit": True,
            "options": {"defaultType": MARKET_TYPE},
        }
    )
    ex.load_markets()
    dbg(f"Loaded {len(ex.markets)} markets")
    return ex


def market_is_usdt_swap(ex, symbol: str) -> bool:
    m = ex.markets.get(symbol)
    return bool(
        m
        and m.get("swap", False)
        and m.get("linear", False)
        and m.get("quote") == QUOTE
    )


def is_good_symbol(ex, symbol: str) -> bool:
    return symbol.endswith(":USDT") and market_is_usdt_swap(ex, symbol)


def get_tick_size(ex, symbol: str) -> float:
    m = ex.markets.get(symbol, {})
    info = m.get("info") or {}
    for f in info.get("filters") or []:
        if f.get("filterType") == "PRICE_FILTER":
            ts = float(f.get("tickSize", 0) or 0)
            if ts > 0:
                return ts
    prec = (m.get("precision") or {}).get("price")
    if isinstance(prec, (int, float)) and prec >= 0:
        return 10 ** (-int(prec))
    return 1e-8


def round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return x
    decimals = int(abs(math.log10(tick))) if tick < 1 else 0
    return round(round(x / tick) * tick, decimals)


def percent_change(a, b):
    try:
        return (a - b) / b * 100.0 if b else 0.0
    except Exception:
        return 0.0


def fetch_ohlcv_df(ex, symbol, timeframe, limit):
    dbg(f"fetch_ohlcv_df({symbol}, {timeframe}, limit={limit})")
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def telegram_send(text: str, parse_mode="HTML"):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        dbg("TG disabled -> printing")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        dbg("TG message sent")
    except Exception as e:
        print("[TG ERROR]", e)


# ------------------- Rank/Trend helpers -------------------
def rank_corr(x, y):
    xr = pd.Series(x).rank(method="average")
    yr = pd.Series(y).rank(method="average")
    if xr.std() == 0 or yr.std() == 0:
        return 0.0
    return float(((xr - xr.mean()) * (yr - yr.mean())).mean() / (xr.std() * yr.std()))


def compute_atr(df_closed, period=14):
    h = df_closed["high"].astype(float)
    l = df_closed["low"].astype(float)
    c = df_closed["close"].astype(float)
    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    val = atr.iloc[-1]
    return float(val) if (pd.notna(val)) else None


def trend_scores(df_closed, bars=TREND_LOOKBACK_BARS):
    if len(df_closed) < (bars + 1):
        return {"spearman": 0.0, "net_pct": 0.0, "ma_up": False, "ma_down": False}
    w = df_closed.tail(bars).copy()
    closes = w["close"].astype(float).tolist()
    t = list(range(bars))
    spearman = rank_corr(t, closes)
    net_pct = percent_change(closes[-1], closes[0])
    s = pd.Series(closes)
    sma_short = s.rolling(TREND_MA_SHORT, min_periods=TREND_MA_SHORT).mean().iloc[-1]
    sma_long = s.rolling(TREND_MA_LONG, min_periods=TREND_MA_LONG).mean().iloc[-1]
    ma_up = pd.notna(sma_short) and pd.notna(sma_long) and (sma_short >= sma_long)
    ma_down = pd.notna(sma_short) and pd.notna(sma_long) and (sma_short <= sma_long)
    return {
        "spearman": float(spearman),
        "net_pct": float(net_pct),
        "ma_up": bool(ma_up),
        "ma_down": bool(ma_down),
    }


def trend_up_1h(df_closed):
    sc = trend_scores(df_closed, TREND_LOOKBACK_BARS)
    ok = (
        (sc["spearman"] >= TREND_MIN_SPEARMAN)
        and (sc["net_pct"] >= TREND_MIN_NET_PCT_UP)
        and sc["ma_up"]
    )
    if DEBUG:
        dbg(
            f"[TREND-UP] ρ={sc['spearman']:.2f} net%={sc['net_pct']:.2f} ma_up={sc['ma_up']} -> {ok}"
        )
    return ok


def trend_down_1h(df_closed):
    sc = trend_scores(df_closed, TREND_LOOKBACK_BARS)
    ok = (
        (sc["spearman"] <= -TREND_MIN_SPEARMAN)
        and (-sc["net_pct"] >= TREND_MIN_NET_PCT_DN)
        and sc["ma_down"]
    )
    if DEBUG:
        dbg(
            f"[TREND-DN] ρ={sc['spearman']:.2f} net%={sc['net_pct']:.2f} ma_down={sc['ma_down']} -> {ok}"
        )
    return ok


# ------------------- Hourly analysis -------------------
def analyze_symbol_hourly(ex, symbol):
    try:
        df = fetch_ohlcv_df(
            ex, symbol, TIMEFRAME_HOURLY, limit=max(LOOKBACK_VOL, BREAKOUT_WINDOW) + 6
        )
    except Exception as e:
        dbg(f"{symbol} fetch 1h error: {e}")
        return None

    if len(df) < (LOOKBACK_VOL + 3):
        dbg(f"{symbol} insufficient bars for 1h")
        return None
    df_closed = df.iloc[:-1]
    if len(df_closed) < (LOOKBACK_VOL + 2):
        dbg(f"{symbol} insufficient CLOSED bars for 1h")
        return None

    last_closed = df_closed.iloc[-1]
    prev_closed = df_closed.iloc[-2]

    close_last = float(last_closed["close"])
    close_prev = float(prev_closed["close"])

    vol_last = float(last_closed["volume"])
    vol_ma = df_closed["volume"].iloc[:-1].tail(LOOKBACK_VOL).mean()
    vol_ratio = (vol_last / vol_ma) if vol_ma else 0.0

    pct_1h = percent_change(close_last, close_prev)
    tick = get_tick_size(ex, symbol)

    hhv_prev = df_closed["high"].iloc[:-1].tail(BREAKOUT_WINDOW).max()
    llv_prev = df_closed["low"].iloc[:-1].tail(BREAKOUT_WINDOW).min()

    breakout_up = (close_last >= hhv_prev + BREAKOUT_TICKS * tick) and (vol_ratio > 2.0)
    breakdown_dn = (close_last <= llv_prev - BREAKDOWN_TICKS * tick) and (
        vol_ratio > 2.0
    )

    price_ref = close_prev
    base_liq_usd = (vol_ma or 0) * price_ref
    tick_ratio = (tick / price_ref) if price_ref > 0 else 1.0

    strong_1h_up = breakout_up or (
        (pct_1h >= ONE_H_STRONG_PCT) and (vol_ratio >= ONE_H_STRONG_VOL)
    )
    strong_1h_down = breakdown_dn or (
        (pct_1h <= -ONE_H_STRONG_PCT) and (vol_ratio >= ONE_H_STRONG_VOL)
    )

    dbg(
        f"[1H] {symbol}: pct={pct_1h:.2f}% vol_ratio={vol_ratio:.2f} "
        f"HHV={hhv_prev:.6g} LLV={llv_prev:.6g} up_brk={breakout_up} dn_brk={breakdown_dn} "
        f"base_liq={base_liq_usd:.0f} tick_ratio={tick_ratio:.4f}"
    )

    return {
        "symbol": symbol,
        "pct_1h": round(pct_1h, 2),
        "vol_ratio_1h": round(vol_ratio, 2),
        "breakout_1h_up": breakout_up,
        "breakdown_1h_dn": breakdown_dn,
        "base_liq_usd": round(base_liq_usd, 2),
        "tick_size": tick,
        "tick_ratio": tick_ratio,
        "close_ref": price_ref,
        "strong_1h_up": strong_1h_up,
        "strong_1h_down": strong_1h_down,
    }


# ------------------- Light (tickers-only) candidate builders -------------------
def fetch_top24h_candidates_from_tickers(ex, tickers):
    """用已有 tickers 生成 24h 涨幅榜（不重复 fetch）。"""
    rows = []
    for sym, t in tickers.items():
        if not t:
            continue
        if not is_good_symbol(ex, sym):
            continue
        pct24 = t.get("percentage")
        if pct24 is None:
            continue
        try:
            rows.append({"symbol": sym, "pct24": float(pct24)})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["symbol", "pct24"])
    df = (
        pd.DataFrame(rows)
        .sort_values("pct24", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    return df


def liquidity_candidates_from_tickers(ex, tickers):
    """
    仅用 tickers 的 24h 成交额做排序，不拉 K。
    - 跳过前 K 个超大盘，避免 BTC/ETH 占坑
    - 直接截断 LIQ_RETURN_TOP
    """
    rows = []
    for sym, t in tickers.items():
        if not t:
            continue
        if not is_good_symbol(ex, sym):
            continue
        qv = t.get("quoteVolume")
        if qv is None:
            # 兜底：用 last * baseVolume 近似
            last = t.get("last") or t.get("close")
            bv = t.get("baseVolume")
            if last and bv:
                try:
                    qv = float(last) * float(bv)
                except Exception:
                    qv = None
        if qv is None:
            continue
        try:
            rows.append({"symbol": sym, "qv": float(qv)})
        except Exception:
            continue

    if not rows:
        return []

    df = pd.DataFrame(rows).sort_values("qv", ascending=False).reset_index(drop=True)
    # 跳过成交额前 K 的超大盘
    skip = set(df.head(ADD_BY_VOLUME_SKIP_TOPK)["symbol"].tolist())
    pool = [s for s in df["symbol"].tolist() if s not in skip]
    return pool[:LIQ_RETURN_TOP]


def body_to_range_ratio(o, h, l, c):
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    return body / rng


def prep_window_checks(df_closed, n=PREP_N, direction="up"):
    """
    返回 (ok, stats)
    stats 字段：
      cum_pct, pos_ratio, close_spearman, vol_spearman, max_bar_abs_pct, body2range_avg, last_vol_ma_ratio
      hhv_prev / llv_prev （用于结构参考）
    """
    if len(df_closed) < (n + 1):
        return False, {}

    w = df_closed.tail(n + 1).copy()  # n 根 + 1 根用于基准
    w = w.iloc[1:]  # 仅取最近的 n 根完全收盘K
    closes = w["close"].astype(float).tolist()
    opens = w["open"].astype(float).tolist()
    highs = w["high"].astype(float).tolist()
    lows = w["low"].astype(float).tolist()
    vols = w["volume"].astype(float).tolist()

    cum_pct = percent_change(closes[-1], closes[0])
    pos_cnt = sum(1 for o, c in zip(opens, closes) if c >= o)
    pos_ratio = pos_cnt / max(1, len(closes))

    t = list(range(len(closes)))
    close_spearman = rank_corr(t, closes)
    vol_spearman = rank_corr(t, vols)

    bar_pcts = [abs(percent_change(c, o)) for o, c in zip(opens, closes)]
    max_bar_abs_pct = max(bar_pcts) if bar_pcts else 0.0

    body2ranges = [
        body_to_range_ratio(o, h, l, c)
        for o, h, l, c in zip(opens, highs, lows, closes)
    ]
    body2range_avg = float(sum(body2ranges) / len(body2ranges)) if body2ranges else 0.0

    vol_ma = (
        pd.Series(vols)
        .rolling(
            LOOKBACK_VOL if LOOKBACK_VOL <= len(vols) else len(vols), min_periods=1
        )
        .mean()
        .iloc[-1]
    )
    last_vol_ma_ratio = (vols[-1] / vol_ma) if vol_ma else 0.0

    hhv_prev = df_closed["high"].iloc[:-1].tail(BREAKOUT_WINDOW).max()
    llv_prev = df_closed["low"].iloc[:-1].tail(BREAKOUT_WINDOW).min()

    if direction == "up":
        dir_ok = (
            close_spearman >= PREP_MIN_CLOSE_SPEARMAN
            and PREP_MIN_CUM_PCT <= cum_pct <= PREP_MAX_CUM_PCT
        )
    else:
        dir_ok = (
            close_spearman <= -PREP_MIN_CLOSE_SPEARMAN
            and -PREP_MAX_CUM_PCT <= cum_pct <= -PREP_MIN_CUM_PCT
        )

    conds = [
        dir_ok,
        (pos_ratio >= PREP_MIN_POS_RATIO)
        if direction == "up"
        else ((1.0 - pos_ratio) >= PREP_MIN_POS_RATIO),
        (vol_spearman >= PREP_MIN_VOL_SPEARMAN),
        (max_bar_abs_pct <= PREP_MAX_SINGLE_BAR_PCT),
        (body2range_avg >= PREP_MIN_BODY_TO_RANGE),
        (PREP_LAST_VOL_TO_MA_MIN <= last_vol_ma_ratio <= PREP_LAST_VOL_TO_MA_MAX),
    ]
    ok = all(conds)

    stats = {
        "cum_pct": float(cum_pct),
        "pos_ratio": float(pos_ratio),
        "close_spearman": float(close_spearman),
        "vol_spearman": float(vol_spearman),
        "max_bar_abs_pct": float(max_bar_abs_pct),
        "body2range_avg": float(body2range_avg),
        "last_vol_ma_ratio": float(last_vol_ma_ratio),
        "hhv_prev": float(hhv_prev) if pd.notna(hhv_prev) else None,
        "llv_prev": float(llv_prev) if pd.notna(llv_prev) else None,
        "n": int(n),
    }
    if DEBUG:
        dbg(
            f"[PREP-{direction.upper()}] n={n} cum%={cum_pct:.2f} pos%={pos_ratio * 100:.0f}% "
            f"ρ_close={close_spearman:.2f} ρ_vol={vol_spearman:.2f} "
            f"max_bar%={max_bar_abs_pct:.2f} body2rng={body2range_avg:.2f} "
            f"lastVol/MA={last_vol_ma_ratio:.2f} -> {ok}"
        )
    return ok, stats


def dynamic_targets(symbol, side, entry_price, df_closed, tick):
    atr = compute_atr(df_closed)
    atr_pct = (atr / entry_price * 100.0) if (atr and entry_price) else None

    if (atr_pct is None) or (atr_pct < 0.6):
        d1, d2, d3 = entry_price * 0.02, entry_price * 0.05, entry_price * 0.10
    else:
        d1, d2, d3 = atr * 1.0, atr * 2.5, atr * 5.0

    if atr is None:
        base = entry_price * 0.003
    else:
        base = atr * (0.8 if atr_pct < 0.6 else 1.2)
    sl_dist = max(base, entry_price * 0.003)
    if atr_pct is not None and atr_pct < 0.3:
        sl_dist = min(sl_dist, 2.0)
    sl_dist = min(sl_dist, 5.0)

    if side == "long":
        tp1, tp2, tp3 = entry_price + d1, entry_price + d2, entry_price + d3
        sl = entry_price - sl_dist
    else:
        tp1, tp2, tp3 = entry_price - d1, entry_price - d2, entry_price - d3
        sl = entry_price + sl_dist

    tp1 = round_to_tick(tp1, tick)
    tp2 = round_to_tick(tp2, tick)
    tp3 = round_to_tick(tp3, tick)
    sl = round_to_tick(sl, tick)
    dbg(
        f"[TPSL] {symbol} side={side} entry={entry_price:.6g} atr={atr if atr else 'NA'} -> tp1={tp1} tp2={tp2} tp3={tp3} sl={sl}"
    )
    return tp1, tp2, tp3, sl


def structure_candidates_twostep(ex, tickers, direction="up"):
    """
    结构候选（两段式，严控个数）：
    1) 轻筛：只用 tickers 的当日 high/low/last 粗估“接近 24h 高/低”的程度，排序后取前 STRUCT_PREFILTER_TOP 个。
       —— 不额外拉 K —— 轻且快。
    2) 精筛：仅对上面入围的少量币，拉 1h K 校验真实 HHV/LLV(BREAKOUT_WINDOW)、1h 趋势、量能区间，最后取前 STRUCT_RETURN_TOP。
    """
    # ---------- 轻筛：用 24h high/low 粗估接近度 ----------
    items = []
    for sym, t in tickers.items():
        if not t:
            continue
        if not is_good_symbol(ex, sym):
            continue
        last = t.get("last") or t.get("close")
        high = t.get("high")
        low = t.get("low")
        qv = t.get("quoteVolume") or 0
        try:
            last = float(last)
            high = float(high)
            low = float(low)
            qv = float(qv)
        except Exception:
            continue
        if last <= 0:
            continue

        if direction == "up":
            if high is None or high <= 0:
                continue
            dist_pct = (high - last) / last * 100.0
            if dist_pct < 0:
                dist_pct = 0.0
            items.append((sym, dist_pct, qv))
        else:
            if low is None or low <= 0:
                continue
            dist_pct = (last - low) / last * 100.0
            if dist_pct < 0:
                dist_pct = 0.0
            items.append((sym, dist_pct, qv))

    if not items:
        return []

    # 先按接近度升序，再按 24h 成交额降序，取前 N 做“精筛”
    items.sort(key=lambda x: (x[1], -x[2]))
    prelist = [sym for sym, _, _ in items[:STRUCT_PREFILTER_TOP]]

    # ---------- 精筛：只对少量币拉 1h K ----------
    out_rows = []
    for sym in prelist:
        try:
            df = fetch_ohlcv_df(
                ex, sym, TIMEFRAME_HOURLY, limit=max(LOOKBACK_VOL, BREAKOUT_WINDOW) + 6
            )
            dfc = df.iloc[:-1]
            if len(dfc) < (LOOKBACK_VOL + 2):
                continue

            last_close = float(dfc.iloc[-1]["close"])
            vol_last = float(dfc.iloc[-1]["volume"])
            vol_ma = dfc["volume"].iloc[:-1].tail(LOOKBACK_VOL).mean()
            volr = (vol_last / vol_ma) if vol_ma else 0.0

            hhv_prev = dfc["high"].iloc[:-1].tail(BREAKOUT_WINDOW).max()
            llv_prev = dfc["low"].iloc[:-1].tail(BREAKOUT_WINDOW).min()

            if direction == "up":
                dist_pct_true = (
                    (hhv_prev - last_close) / last_close * 100.0 if last_close else 999
                )
                trend_ok = trend_up_1h(dfc)
                vol_ok = NEAR_VOLR_MIN <= volr <= NEAR_VOLR_MAX
                if (0 < dist_pct_true <= NEAR_HHV_MAX_DIST_PCT) and trend_ok and vol_ok:
                    out_rows.append((sym, dist_pct_true, volr))
            else:
                dist_pct_true = (
                    (last_close - llv_prev) / last_close * 100.0 if last_close else 999
                )
                trend_ok = trend_down_1h(dfc)
                vol_ok = NEAR_VOLR_MIN <= volr <= NEAR_VOLR_MAX
                if (0 < dist_pct_true <= NEAR_LLV_MAX_DIST_PCT) and trend_ok and vol_ok:
                    out_rows.append((sym, dist_pct_true, volr))

            ex.sleep(SLEEP_MS)
        except Exception as e:
            dbg(f"struct-2nd fail {sym}: {e}")

    # 最终排序：接近度更小更靠前，量能比更接近 1.0（不过热/不过冷）可适度加权
    out_rows.sort(key=lambda x: (x[1], abs(x[2] - 1.0)))
    return [sym for sym, _, _ in out_rows[:STRUCT_RETURN_TOP]]


# ------------------- Hourly candidate refresh (merged) -------------------
def hourly_refresh_candidates(ex):
    print(f"[{ts_now()}] Hourly refresh…")

    # ✅ 统一一次性获取 tickers，后续函数复用，避免重复 IO
    tickers = ex.fetch_tickers()

    # 原动量候选：先按24h涨幅TOP_N（基于 tickers） → 1h快筛取前 RANK_BY_1H
    top24 = fetch_top24h_candidates_from_tickers(ex, tickers)
    momentum_syms = []
    if not top24.empty:
        rows = []
        for _, r in top24.iterrows():
            sym = r["symbol"]
            try:
                df = fetch_ohlcv_df(ex, sym, TIMEFRAME_HOURLY, limit=4)
                df_closed = df.iloc[:-1]
                if len(df_closed) < 2:
                    continue
                pct_1h_quick = percent_change(
                    float(df_closed.iloc[-1]["close"]),
                    float(df_closed.iloc[-2]["close"]),
                )
                rows.append({"symbol": sym, "pct_1h_quick": pct_1h_quick})
                ex.sleep(SLEEP_MS)
            except Exception as e:
                dbg(f"1h quick fail {sym}: {e}")

        if rows:
            quick_df = (
                pd.DataFrame(rows)
                .sort_values("pct_1h_quick", ascending=False)
                .head(RANK_BY_1H)
            )
            momentum_syms = list(quick_df["symbol"])

    # 新：基于 tickers 的轻量函数（不会为每个币拉 K）
    liq_syms = liquidity_candidates_from_tickers(ex, tickers)
    stru_up = structure_candidates_twostep(ex, tickers, "up")
    stru_dn = structure_candidates_twostep(ex, tickers, "down")

    # ---- 先按配额裁剪各自来源 ----
    momentum_syms = momentum_syms[:MOMENTUM_QUOTA]
    liq_syms = liq_syms[:LIQUIDITY_QUOTA]
    stru_up = stru_up[:STRUCT_UP_QUOTA]
    stru_dn = stru_dn[:STRUCT_DN_QUOTA]

    # ---- 合并去重 + 全局上限（避免 analyze_symbol_hourly 过多调用）----
    merged, seen = [], set()
    for s in momentum_syms + liq_syms + stru_up + stru_dn:
        if s not in seen:
            merged.append(s)
            seen.add(s)
            if len(merged) >= MAX_MERGED_CANDIDATES:
                break

    dbg(
        f"Merged candidates pre-analyze: {len(merged)} "
        f"(mom={len(momentum_syms)}, liq={len(liq_syms)}, up={len(stru_up)}, dn={len(stru_dn)}) "
        f"cap={MAX_MERGED_CANDIDATES}"
    )

    results, strong_up_map, strong_dn_map = [], {}, {}
    dropped_liq = dropped_tick = 0

    # 跑小时分析 + 硬过滤（流动性/最小tick）
    for sym in merged:
        try:
            info = analyze_symbol_hourly(ex, sym)
            if not info:
                continue
            if info["base_liq_usd"] < MIN_BASE_VOLUME_USD:
                dropped_liq += 1
                dbg(
                    f"{sym} drop: liq {info['base_liq_usd']:.0f} < {MIN_BASE_VOLUME_USD}"
                )
                continue
            if info["tick_ratio"] > MAX_TICK_TO_PRICE:
                dropped_tick += 1
                dbg(
                    f"{sym} drop: tick_ratio {info['tick_ratio']:.4f} > {MAX_TICK_TO_PRICE:.4f}"
                )
                continue
            strong_up_map[sym] = info["strong_1h_up"]
            strong_dn_map[sym] = info["strong_1h_down"]
            results.append(info)
            ex.sleep(SLEEP_MS)
        except Exception as e:
            dbg(f"hourly analyze fail {sym}: {e}")

    # —— 综合打分排序（可调权重，易解释）——
    def source_score(sym):
        sc = 0.0
        if sym in momentum_syms:
            sc += 2.0  # 动量更高权
        if sym in liq_syms:
            sc += 1.0  # 流动性加分
        if sym in stru_up or sym in stru_dn:
            sc += 1.5  # 结构临近加分
        if strong_up_map.get(sym) or strong_dn_map.get(sym):
            sc += 0.5  # 小幅偏好1h强势
        return sc

    results.sort(
        key=lambda x: (
            source_score(x["symbol"]),
            x["breakout_1h_up"],  # 已经突破者略优先
            x["pct_1h"],  # 1h涨跌幅
            x["vol_ratio_1h"],  # 1h量能
        ),
        reverse=True,
    )
    final = [r["symbol"] for r in results][:RANK_BY_1H]

    # map 只保留最终集合
    strong_up_map = {s: strong_up_map.get(s, False) for s in final}
    strong_dn_map = {s: strong_dn_map.get(s, False) for s in final}

    dbg(
        f"Hourly kept={len(final)} from merged={len(merged)} (liq_drop={dropped_liq}, tick_drop={dropped_tick})"
    )
    print(f"[{ts_now()}] Hourly candidates ({len(final)}): {final}")
    return final, strong_up_map, strong_dn_map


# ------------------- 5m checker -------------------
def format_alert(symbol, title, lines_extra):
    lines = [title]
    lines.extend(lines_extra)
    lines.append(f"Time: {ts_now()}")
    return "\n".join(lines)


# ------------------- 保守入场（带最小安全距离） -------------------
def conservative_entry(symbol, side, close_last, df_closed, tick, hhv_prev, llv_prev):
    """
    多：max( 前HHV+buffer,  当前价 - max(波段*fib, ATR*SAFE_ATR_MULT) )，但 ≤ 现价 - min_dist
    空：min( 前LLV-buffer,  当前价 + max(波段*fib, ATR*SAFE_ATR_MULT) )，但 ≥ 现价 + min_dist
    其中 min_dist = max(ATR * SAFE_MIN_DIST_ATR, SAFE_MIN_DIST_TICKS * tick)
    """
    bars_needed = max(14, TREND_LOOKBACK_BARS)
    if len(df_closed) < bars_needed:
        return round_to_tick(close_last, tick)

    w = df_closed.tail(bars_needed).copy()
    atr = compute_atr(df_closed) or 0.0
    swing_high = float(w["high"].max())
    swing_low = float(w["low"].min())
    swing_range = max(swing_high - swing_low, 0.0)

    retrace = max(swing_range * SAFE_FIB_RATIO, atr * SAFE_ATR_MULT)
    min_dist = max(atr * SAFE_MIN_DIST_ATR, SAFE_MIN_DIST_TICKS * tick)

    if side == "long":
        retest = (
            (hhv_prev + SAFE_RETEST_TICKS * tick)
            if pd.notna(hhv_prev)
            else (close_last - retrace)
        )
        pullback = close_last - retrace
        entry = min(max(retest, pullback), close_last - min_dist)
    else:
        retest = (
            (llv_prev - SAFE_RETEST_TICKS * tick)
            if pd.notna(llv_prev)
            else (close_last + retrace)
        )
        rebound = close_last + retrace
        entry = max(min(retest, rebound), close_last + min_dist)

    entry = round_to_tick(entry, tick)
    dbg(
        f"[SAFE-ENTRY] {symbol} side={side} close={close_last:.6g} atr={atr:.6g} "
        f"min_dist={min_dist:.6g} -> entry_safe={entry}"
    )
    return entry


def tpsl_for_safe_entry(side, entry_safe, tick):
    sl_pct = max(SAFE_MIN_SL_PCT, min(SAFE_SL_PCT, SAFE_MAX_SL_PCT)) / 100.0
    tp1_pct, tp2_pct, tp3_pct = [p / 100.0 for p in SAFE_TP_PCTS]

    if side == "long":
        tp1 = entry_safe * (1.0 + tp1_pct)
        tp2 = entry_safe * (1.0 + tp2_pct)
        tp3 = entry_safe * (1.0 + tp3_pct)
        sl = entry_safe * (1.0 - sl_pct)
    else:
        tp1 = entry_safe * (1.0 - tp1_pct)
        tp2 = entry_safe * (1.0 - tp2_pct)
        tp3 = entry_safe * (1.0 - tp3_pct)
        sl = entry_safe * (1.0 + sl_pct)

    tp1 = round_to_tick(tp1, tick)
    tp2 = round_to_tick(tp2, tick)
    tp3 = round_to_tick(tp3, tick)
    sl = round_to_tick(sl, tick)
    return tp1, tp2, tp3, sl


def five_min_check_one(ex, symbol, strong_up_map, strong_dn_map):
    df = fetch_ohlcv_df(
        ex, symbol, TIMEFRAME_FAST, limit=max(LOOKBACK_VOL, BREAKOUT_WINDOW) + 6
    )
    if len(df) < (LOOKBACK_VOL + 3):
        return False, None, None
    df_closed = df.iloc[:-1]
    if len(df_closed) < (LOOKBACK_VOL + 2):
        return False, None, None

    last_closed = df_closed.iloc[-1]
    prev_closed = df_closed.iloc[-2]
    close_last = float(last_closed["close"])
    close_prev = float(prev_closed["close"])
    vol_last = float(last_closed["volume"])
    vol_ma = df_closed["volume"].iloc[:-1].tail(LOOKBACK_VOL).mean()
    vol_ratio = (vol_last / vol_ma) if vol_ma else 0.0
    pct_5m = percent_change(close_last, close_prev)
    tick = get_tick_size(ex, symbol)

    hhv_prev = df_closed["high"].iloc[:-1].tail(BREAKOUT_WINDOW).max()
    llv_prev = df_closed["low"].iloc[:-1].tail(BREAKOUT_WINDOW).min()

    bullish_breakout = (close_last >= hhv_prev + BREAKOUT_TICKS * tick) and (
        vol_ratio > M5_VOL_RATIO_STRONG
    )
    bullish_pump = (pct_5m > M5_PCT_RISE) and (vol_ratio > M5_VOL_RATIO_STRONG)

    bearish_breakdown = (close_last <= llv_prev - BREAKDOWN_TICKS * tick) and (
        vol_ratio > M5_VOL_RATIO_STRONG_BEAR
    )
    bearish_dump = ((-pct_5m) > M5_PCT_DROP) and (vol_ratio > M5_VOL_RATIO_STRONG_BEAR)

    dbg(
        f"[5M] {symbol} pct={pct_5m:.2f}% vol_ratio={vol_ratio:.2f} "
        f"BO={bullish_breakout} BD={bearish_breakdown} hhv={hhv_prev:.6g} llv={llv_prev:.6g}"
    )

    # 多头：需 1h 强势(上) + 近1小时上行
    if bullish_breakout or bullish_pump:
        if not strong_up_map.get(symbol, False):
            dbg(f"{symbol} long blocked: not 1h-strong-up")
        elif not trend_up_1h(df_closed):
            dbg(f"{symbol} long blocked by up-trend filter")
        else:
            tp1, tp2, tp3, sl = dynamic_targets(
                symbol, "long", close_last, df_closed, tick
            )
            text = format_alert(
                symbol,
                "🔥 <b>5m 量价信号（多）</b>",
                [
                    f"Symbol: <b>{symbol}</b>",
                    f"Price: <code>{close_last:.6g}</code>",
                    f"5m %: <b>{pct_5m:.2f}%</b> | Vol/MA: <b>{vol_ratio:.2f}x</b>",
                    f"Breakout(5m HHV{BREAKOUT_WINDOW}): <b>{'Yes' if bullish_breakout else 'No'}</b>",
                    f"TPs: <code>{tp1}</code> / <code>{tp2}</code> / <code>{tp3}</code> | SL: <code>{sl}</code>",
                ],
            )

            if not SAFE_MODE_ALWAYS:
                cmd = (
                    f"/forcelong {symbol} 10 10 {tp1} {tp2} {tp3} {sl} {close_last:.6g}"
                )
                text += f"\n\n<code>{cmd}</code>"
            else:
                text += "\n<b>SAFE MODE:</b> 仅推送保守入场命令"

            entry_safe = conservative_entry(
                symbol, "long", close_last, df_closed, tick, hhv_prev, None
            )
            tp1s, tp2s, tp3s, sls = tpsl_for_safe_entry("long", entry_safe, tick)
            cmd_safe = (
                f"/forcelong {symbol} 10 10 {tp1s} {tp2s} {tp3s} {sls} {entry_safe}"
            )
            text += f"\n<b>Conservative Entry:</b> <code>{entry_safe}</code> | SL(≈{SAFE_SL_PCT:.1f}%): <code>{sls}</code>"
            text += f"\n<code>{cmd_safe}</code>"

            kind = "breakout_long" if bullish_breakout else "pump_long"
            return True, kind, text

    # 空头：需 近1小时下行（可选再要 1h 强势(下)）
    if bearish_breakdown or bearish_dump:
        if REQUIRE_1H_DOWN_FOR_BEAR and (not strong_dn_map.get(symbol, False)):
            dbg(f"{symbol} short blocked: not 1h-strong-down")
        elif not trend_down_1h(df_closed):
            dbg(f"{symbol} short blocked by down-trend filter")
        else:
            tp1, tp2, tp3, sl = dynamic_targets(
                symbol, "short", close_last, df_closed, tick
            )
            text = format_alert(
                symbol,
                "🩸 <b>5m 量价信号（空）</b>",
                [
                    f"Symbol: <b>{symbol}</b>",
                    f"Price: <code>{close_last:.6g}</code>",
                    f"5m %: <b>{pct_5m:.2f}%</b> | Vol/MA: <b>{vol_ratio:.2f}x</b>",
                    f"Breakdown(5m LLV{BREAKOUT_WINDOW}): <b>{'Yes' if bearish_breakdown else 'No'}</b>",
                    f"TPs: <code>{tp1}</code> / <code>{tp2}</code> / <code>{tp3}</code> | SL: <code>{sl}</code>",
                ],
            )

            if not SAFE_MODE_ALWAYS:
                cmd = f"/forceshort {symbol} 10 10 {tp1} {tp2} {tp3} {sl} {close_last:.6g}"
                text += f"\n\n<code>{cmd}</code>"
            else:
                text += "\n<b>SAFE MODE:</b> 仅推送保守入场命令"

            entry_safe = conservative_entry(
                symbol, "short", close_last, df_closed, tick, None, llv_prev
            )
            tp1s, tp2s, tp3s, sls = tpsl_for_safe_entry("short", entry_safe, tick)
            cmd_safe = (
                f"/forceshort {symbol} 10 10 {tp1s} {tp2s} {tp3s} {sls} {entry_safe}"
            )
            text += f"\n<b>Conservative Entry:</b> <code>{entry_safe}</code> | SL(≈{SAFE_SL_PCT:.1f}%): <code>{sls}</code>"
            text += f"\n<code>{cmd_safe}</code>"

            kind = "breakdown_short" if bearish_breakdown else "dump_short"
            return True, kind, text

    # ---- PREP 信号：当常规信号未触发时才考虑 ----
    if PREP_ENABLED:
        # 多向预备（上行筹备）
        ok_up, stats_up = prep_window_checks(df_closed, n=PREP_N, direction="up")
        if ok_up and strong_up_map.get(symbol, False) and trend_up_1h(df_closed):
            entry_safe = conservative_entry(
                symbol,
                "long",
                close_last,
                df_closed,
                tick,
                stats_up.get("hhv_prev"),
                None,
            )
            tp1s, tp2s, tp3s, sls = tpsl_for_safe_entry("long", entry_safe, tick)
            text = format_alert(
                symbol,
                "⏳ <b>PREP 信号（上）</b>",
                [
                    f"Symbol: <b>{symbol}</b>",
                    f"Price: <code>{close_last:.6g}</code>",
                    f"Window(n={stats_up.get('n', PREP_N)}): cum%=<b>{stats_up['cum_pct']:.2f}%</b>, pos%=<b>{stats_up['pos_ratio'] * 100:.0f}%</b>",
                    f"ρ(close,t)=<b>{stats_up['close_spearman']:.2f}</b>, ρ(vol,t)=<b>{stats_up['vol_spearman']:.2f}</b>",
                    f"max|bar%|=<b>{stats_up['max_bar_abs_pct']:.2f}%</b>, body2rng=<b>{stats_up['body2range_avg']:.2f}</b>, lastVol/MA=<b>{stats_up['last_vol_ma_ratio']:.2f}x</b>",
                    f"Structure(HHV{BREAKOUT_WINDOW}): <code>{stats_up['hhv_prev']:.6g}</code>",
                    f"Conservative Entry: <code>{entry_safe}</code> | SL(≈{SAFE_SL_PCT:.1f}%): <code>{sls}</code>",
                    f"TPs: <code>{tp1s}</code> / <code>{tp2s}</code> / <code>{tp3s}</code>",
                    f"<code>/forcelong {symbol} 10 10 {tp1s} {tp2s} {tp3s} {sls} {entry_safe}</code>",
                ],
            )
            return True, "prep_up", text

        # 空向预备（下行筹备）
        ok_dn, stats_dn = prep_window_checks(df_closed, n=PREP_N, direction="down")
        allow_dn = trend_down_1h(df_closed) and (
            strong_dn_map.get(symbol, False) or not REQUIRE_1H_DOWN_FOR_BEAR
        )
        if ok_dn and allow_dn:
            entry_safe = conservative_entry(
                symbol,
                "short",
                close_last,
                df_closed,
                tick,
                None,
                stats_dn.get("llv_prev"),
            )
            tp1s, tp2s, tp3s, sls = tpsl_for_safe_entry("short", entry_safe, tick)
            text = format_alert(
                symbol,
                "⏳ <b>PREP 信号（下）</b>",
                [
                    f"Symbol: <b>{symbol}</b>",
                    f"Price: <code>{close_last:.6g}</code>",
                    f"Window(n={stats_dn.get('n', PREP_N)}): cum%=<b>{stats_dn['cum_pct']:.2f}%</b>, neg%=<b>{(1.0 - stats_dn['pos_ratio']) * 100:.0f}%</b>",
                    f"ρ(close,t)=<b>{stats_dn['close_spearman']:.2f}</b>, ρ(vol,t)=<b>{stats_dn['vol_spearman']:.2f}</b>",
                    f"max|bar%|=<b>{stats_dn['max_bar_abs_pct']:.2f}%</b>, body2rng=<b>{stats_dn['body2range_avg']:.2f}</b>, lastVol/MA=<b>{stats_dn['last_vol_ma_ratio']:.2f}x</b>",
                    f"Structure(LLV{BREAKOUT_WINDOW}): <code>{stats_dn['llv_prev']:.6g}</code>",
                    f"Conservative Entry: <code>{entry_safe}</code> | SL(≈{SAFE_SL_PCT:.1f}%): <code>{sls}</code>",
                    f"TPs: <code>{tp1s}</code> / <code>{tp2s}</code> / <code>{tp3s}</code>",
                    f"<code>/forceshort {symbol} 10 10 {tp1s} {tp2s} {tp3s} {sls} {entry_safe}</code>",
                ],
            )
            return True, "prep_down", text

    return False, None, None


# ------------------- Timing helpers -------------------
def align_sleep_to_next_5m():
    now = int(time.time())
    sec_to_next = (300 - (now % 300)) + 5
    dbg(f"Align to next 5m close: sleep {sec_to_next}s")
    time.sleep(sec_to_next)


# ------------------- Main loop -------------------
def main():
    ex = build_exchange()
    candidates, strong_up_map, strong_dn_map = hourly_refresh_candidates(ex)
    last_alert_at = {}
    last_heartbeat_key = None
    align_sleep_to_next_5m()
    loop_durations = []

    while True:
        loop_start = time.time()
        try:
            now_ts = int(time.time())
            sec_into_hour = now_ts % 3600

            # 小时候选重算窗口
            if (
                sec_into_hour <= HOURLY_REFRESH_JITTER
                or sec_into_hour >= 3600 - HOURLY_REFRESH_JITTER
            ):
                dbg("Hourly candidate refresh window")
                new_list, up_map, dn_map = hourly_refresh_candidates(ex)
                if new_list:
                    candidates = new_list[:RANK_BY_1H]
                    strong_up_map = up_map
                    strong_dn_map = dn_map
                    telegram_send(
                        f"🕐 <b>Hourly refreshed</b>\nCount: <b>{len(candidates)}</b>"
                    )

            # 心跳
            if HEARTBEAT_ENABLED:
                utc_now = datetime.now(timezone.utc)
                heartbeat_key = utc_now.strftime("%Y-%m-%d %H:00")
                in_window = (sec_into_hour <= HEARTBEAT_JITTER) or (
                    sec_into_hour >= 3600 - HEARTBEAT_JITTER
                )
                if in_window and heartbeat_key != last_heartbeat_key:
                    cnt = len(candidates) if candidates else 0
                    avg_dur = sum(loop_durations[-6:]) / max(
                        1, len(loop_durations[-6:])
                    )
                    telegram_send(
                        f"⏱️ <b>Heartbeat</b>\nTime: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}\nCandidates: <b>{cnt}</b>\nAvg Loop: <b>{avg_dur:.1f}s</b>"
                    )
                    last_heartbeat_key = heartbeat_key

            # 5m 扫描
            if not candidates:
                dbg("No candidates; skip.")
            else:
                dbg(f"5m scanning {len(candidates)} symbols…")
                for sym in candidates:
                    try:
                        should, kind, text = five_min_check_one(
                            ex, sym, strong_up_map, strong_dn_map
                        )
                        ex.sleep(SLEEP_MS)
                        if not should:
                            continue
                        now = time.time()
                        key = (sym, kind)
                        if now - last_alert_at.get(key, 0) >= ALERT_COOLDOWN_SEC:
                            telegram_send(text)
                            last_alert_at[key] = now
                    except Exception as e:
                        print(f"[ERROR] 5m check {sym}: {e}")

        except Exception as e:
            print("[LOOP ERROR]", e)

        elapsed = time.time() - loop_start
        loop_durations.append(elapsed)
        dbg(f"Loop {elapsed:.1f}s")
        time.sleep(max(5, LOOP_INTERVAL_SEC - int(elapsed)))


if __name__ == "__main__":
    main()
