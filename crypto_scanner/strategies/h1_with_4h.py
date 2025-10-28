from . import Strategy

H1_4H = Strategy(
    name="H1_with_4h_gate",
    timeframe_fast="1h",
    timeframe_htf="4h",
    bars_per_day=24,
    bb_width_th=0.018,
    explode_contraction_width=0.022,
    explode_min_dist_dayext_pct=0.8,
    scale_eq_bar_usd=1_200_000.0,
    w_volr_now=0.35,
    w_eq_now_usd=0.45,
    w_abs_pct=0.10,
    w_trend_align=0.10,
    pb_lookback_hi=60,
    overrides={
        "ALERT_COOLDOWN_SEC": 60 * 60,
        "MIN_QV5M_USD": 2_400_000,
        "SCALE_ABS_PCT": 1.2,
        "PB_MIN_BOUNCE_PCT": 0.12,
        "PB_LOOKBACK_HI_PCT": 2.2,
        "EXPLODE_QUIET_EXTRA_SCORE": 0.15,
    },
)
