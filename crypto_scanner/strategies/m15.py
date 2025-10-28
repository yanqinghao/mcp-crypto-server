from . import Strategy

M15 = Strategy(
    name="M15_with_1h_gate",
    timeframe_fast="15m",
    timeframe_htf="1h",
    bars_per_day=96,
    bb_width_th=0.015,
    explode_contraction_width=0.025,
    explode_min_dist_dayext_pct=0.6,
    scale_eq_bar_usd=600_000.0,
    w_volr_now=0.35,
    w_eq_now_usd=0.40,
    w_abs_pct=0.15,
    w_trend_align=0.10,
    pb_lookback_hi=48,
    overrides={
        "ALERT_COOLDOWN_SEC": 45 * 60,
        "MIN_QV5M_USD": 900_000,
        "SCALE_ABS_PCT": 1.0,
        "PB_MIN_BOUNCE_PCT": 0.10,
        "PB_LOOKBACK_HI_PCT": 2.5,
        "EXPLODE_QUIET_EXTRA_SCORE": 0.12,
    },
)
