import pandas as pd
from .config import (
    MIN_QV24_USD,
    MIN_QV5M_USD,
    MAX_TICK_TO_PRICE,
    PCT24_ABS_MIN,
    W_PCT24,
    W_QV24,
    W_NEAR_EXTREME,
    MAX_CANDIDATES,
    COIN_METRICS,
    MARKETCAP_MAP,
    SYMBOL_CLASS,
    KNOWN_MAJORS,
    TIMEFRAME_HOURLY,
    PARAM_PRESETS,
)
from .exchange import is_good_symbol, get_tick_size
from .loggingx import ts_now, dbg
from .strategies import Strategy


def hourly_refresh_candidates(ex, strategy: Strategy):
    print(f"[{ts_now()}] Hourly refresh (tickers-only)…")
    bars_per_day = strategy.bars_per_day
    try:
        tickers = ex.fetch_tickers()
    except Exception as e:
        dbg(f"fetch_tickers error: {e}")
        return [], {}, {}
    rows = []
    for sym, t in (tickers or {}).items():
        if not t or not is_good_symbol(ex, sym):
            continue
        last = t.get("last") or t.get("close")
        high = t.get("high")
        low = t.get("low")
        pct = t.get("percentage")
        qv = t.get("quoteVolume")
        if qv is None:
            bv = t.get("baseVolume")
            if last and bv:
                try:
                    qv = float(last) * float(bv)
                except Exception:
                    qv = None
        try:
            last = float(last) if last is not None else None
            high = float(high) if high is not None else None
            low = float(low) if low is not None else None
            pct = float(pct) if pct is not None else None
            qv = float(qv) if qv is not None else None
        except Exception:
            continue
        if (last is None) or (pct is None) or (qv is None):
            continue
        if qv < MIN_QV24_USD:
            continue
        qv_bar = qv / bars_per_day
        if qv_bar < MIN_QV5M_USD:
            continue
        tick = get_tick_size(ex, sym)
        if last <= 0 or tick <= 0:
            continue
        if (tick / last) > MAX_TICK_TO_PRICE:
            continue
        dist_hi = ((high - last) / last * 100.0) if (high and last > 0) else None
        dist_lo = ((last - low) / last * 100.0) if (low and last > 0) else None
        if abs(pct) < PCT24_ABS_MIN:
            continue
        dist_ext = None
        if (dist_hi is not None) and (dist_lo is not None):
            dist_ext = min(max(dist_hi, 0.0), max(dist_lo, 0.0))
        elif dist_hi is not None:
            dist_ext = max(dist_hi, 0.0)
        elif dist_lo is not None:
            dist_ext = max(dist_lo, 0.0)
        rows.append(
            {
                "symbol": sym,
                "abs_pct24": abs(pct),
                "pct24": pct,
                "qv": qv,
                "qv_bar": qv_bar,
                "dist_ext": dist_ext,
            }
        )
    if not rows:
        dbg("tickers-only filter produced empty set.")
        print(f"[{ts_now()}] Hourly candidates (0): []")
        return [], {}, {}
    df = pd.DataFrame(rows)

    def rank01(s, ascending=False):
        r = s.rank(method="average", ascending=ascending)
        return (r - r.min()) / max(1e-12, (r.max() - r.min()))

    r_pct = rank01(df["abs_pct24"], False)
    r_qv = rank01(df["qv"], False)
    if (
        ("dist_ext" in df.columns)
        and (df["dist_ext"].notna().any())
        and W_NEAR_EXTREME > 0
    ):
        dist_filled = df["dist_ext"].fillna(
            df["dist_ext"].max() if df["dist_ext"].notna().any() else 999.0
        )
        r_ext = 1.0 - rank01(dist_filled, True)
    else:
        from pandas import Series

        r_ext = Series([0.0] * len(df), index=df.index)

    df["score"] = (W_PCT24 * r_pct) + (W_QV24 * r_qv) + (W_NEAR_EXTREME * r_ext)
    df = (
        df.sort_values("score", ascending=False)
        .head(MAX_CANDIDATES)
        .reset_index(drop=True)
    )
    symbols = df["symbol"].tolist()
    strong_up_map = {row["symbol"]: (row["pct24"] >= 1.0) for _, row in df.iterrows()}
    strong_dn_map = {row["symbol"]: (row["pct24"] <= -1.0) for _, row in df.iterrows()}

    for _, row in df.iterrows():
        sym = row["symbol"]
        qv = float(row["qv"])
        pct = float(row["pct24"]) if "pct24" in row else None
        COIN_METRICS[sym] = {"qv24": qv, "pct24": pct, "mcap": MARKETCAP_MAP.get(sym)}

    for sym in symbols:
        m = COIN_METRICS.get(sym, {})
        qv = m.get("qv24", 0.0)
        pct = abs(m.get("pct24", 0.0) or 0.0)
        SYMBOL_CLASS[sym] = classify_symbol(
            ex, sym, qv24_usd=qv, pct24_abs=pct, mcap_usd=m.get("mcap")
        )

    dbg(
        "symbol classes: "
        + ", ".join([f"{s}:{SYMBOL_CLASS.get(s, '?')}" for s in symbols])
    )
    print(f"[{ts_now()}] Hourly candidates ({len(symbols)}): {symbols}")
    return symbols, strong_up_map, strong_dn_map


def classify_symbol(ex, symbol, qv24_usd, pct24_abs, mcap_usd=None) -> str:
    base_hint = (
        "MAJOR"
        if ((symbol in KNOWN_MAJORS) or (mcap_usd and mcap_usd >= 2.0e10))
        else None
    )
    if qv24_usd >= 1.0e9:
        vol_hint = "MAJOR"
    elif qv24_usd >= 2.0e8:
        vol_hint = "MID"
    else:
        vol_hint = "ALT"
    # 用 1h ATR% 作为波动提示
    from .ta_utils import compute_atr
    from .exchange import fetch_ohlcv_df

    try:
        df1h = fetch_ohlcv_df(ex, symbol, TIMEFRAME_HOURLY, limit=60)
        last_close = float(df1h["close"].iloc[-1]) if len(df1h) else 0.0
        atr = compute_atr(df1h, period=14) if len(df1h) else 0.0
        atrp_1h = float(atr / last_close * 100.0) if (atr and last_close > 0) else 0.0
    except Exception:
        atrp_1h = 0.0
    atr_hint = "MAJOR" if atrp_1h <= 1.5 else ("MID" if atrp_1h <= 3.0 else "ALT")
    pct_hint = "MAJOR" if pct24_abs <= 5.0 else ("MID" if pct24_abs <= 10.0 else "ALT")
    votes = [hint for hint in [base_hint, vol_hint, atr_hint, pct_hint] if hint]
    if base_hint:
        votes.append(base_hint)
    cls = max(("MAJOR", "MID", "ALT"), key=lambda k: votes.count(k))
    return cls


def resolve_params_for_symbol(symbol: str):
    from .config import (
        SYMBOL_CLASS,
        EXPLODE_VOLR,
        PRICE_UP_TH,
        PRICE_DN_TH,
        CAP_VOLR,
        CAP_WICK_RATIO,
        PB_LOOKBACK_HI_PCT,
        MIN_QV5M_USD,
    )

    preset = SYMBOL_CLASS.get(symbol)
    if not preset:
        return {
            "EXPLODE_VOLR": EXPLODE_VOLR,
            "PRICE_UP_TH": PRICE_UP_TH,
            "PRICE_DN_TH": PRICE_DN_TH,
            "CAP_VOLR": CAP_VOLR,
            "CAP_WICK_RATIO": CAP_WICK_RATIO,
            "PB_LOOKBACK_HI_PCT": PB_LOOKBACK_HI_PCT,
            "MIN_QV5M_USD": MIN_QV5M_USD,
        }
    override = PARAM_PRESETS.get(preset, {})
    return {
        "EXPLODE_VOLR": override.get("EXPLODE_VOLR", EXPLODE_VOLR),
        "PRICE_UP_TH": override.get("PRICE_UP_TH", PRICE_UP_TH),
        "PRICE_DN_TH": override.get("PRICE_DN_TH", PRICE_DN_TH),
        "CAP_VOLR": override.get("CAP_VOLR", CAP_VOLR),
        "CAP_WICK_RATIO": override.get("CAP_WICK_RATIO", CAP_WICK_RATIO),
        "PB_LOOKBACK_HI_PCT": override.get("PB_LOOKBACK_HI_PCT", PB_LOOKBACK_HI_PCT),
        "MIN_QV5M_USD": override.get("MIN_QV5M_USD", MIN_QV5M_USD),
    }
