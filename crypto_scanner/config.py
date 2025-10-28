import os
from copy import deepcopy
from typing import Dict, Any

# ========= Sensitivity Mode =========
MODE = os.getenv("MODE", "QUIET").strip().upper()

# ======================== Config (Baseline) ========================
DEBUG = True
EXCHANGE_ID = "binance"
MARKET_TYPE = "future"
QUOTE = "USDT"

TIMEFRAME_HOURLY = "1h"
TIMEFRAME_FAST = "15m"

FRAME_SEC = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
BAR_SEC = FRAME_SEC[TIMEFRAME_FAST]

RANK_BY_1H = 60
LOOKBACK_VOL = 50
BREAKOUT_WINDOW = 20
SLEEP_MS = 10
HOURLY_REFRESH_JITTER = 30

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

# Auto delete
AUTO_DELETE_ENABLED = True
AUTO_DELETE_HOURS = 12
AUTO_DELETE_GRACE = 30

# Cadence settings
SCAN_FAST_ENABLED = False
SCAN_SLOW_ENABLED = True
CANDIDATE_REFRESH_SEC = 4 * 3600
FAST_SCAN_KINDS = {"explode", "ema_rebound"}  # fast=off 时不触发
POLL_SEC = 30

# 批量/冷却
MAX_ALERTS_PER_ROUND = 2
BATCH_COOLDOWN_SEC = 180
ALERT_COOLDOWN_SEC = 45 * 60
ONLY_PUSH_EXPLODE = True
MIN_SIGNAL_SCORE = 0.55
BATCH_PUSH_ENABLED = True

# 分批中文推送
PER_MESSAGE_LIMIT = 3
MESSAGE_DELAY_SEC = 3
SEND_ALL_COLLECTED = True
TITLE_PREFIX = "📣 扫描信号"
SEPARATOR_LINE = "——————————————"

# 打印原因
PRINT_FULL_REASONS = False
MAX_REASONS_IN_MSG = 5

# 开关
ENABLE_PULLBACK = True
ENABLE_CAPITULATION = True
ENABLE_EMA_REBOUND = True
ENABLE_BB_SQUEEZE = True

# 中文映射
KINDS_CN = {
    "explode_up": "爆发行情 · 上破",
    "explode_down": "爆发行情 · 下破",
    "pullback_long": "顺势回撤 · 反弹",
    "pullback_short": "顺势回撤 · 转弱",
    "ema_rebound_long": "EMA 回踩 · 反弹",
    "ema_rebound_short": "EMA 回踩 · 下压",
    "bb_squeeze_long": "布林挤压 · 上越",
    "bb_squeeze_short": "布林挤压 · 下破",
    "cap_long": "疑似止跌 · 长下影",
    "cap_short": "疑似见顶 · 长上影",
}

# 流动性&粗糙度（15m）
MIN_QV24_USD = 40_000_000
MIN_QV5M_USD = 900_000  # 语义：本 bar 等效成交额（15m 量级）
MAX_TICK_TO_PRICE = 0.001

# 爆发判定（baseline）
BASELINE_BARS = 12
EXPLODE_VOLR = 3.0
PRICE_UP_TH = 0.35
PRICE_DN_TH = -0.35
NO_FOLLOW_UP_TH = 0.05

# 趋势
REQUIRE_TREND_ALIGN = True
ADX_MIN_TREND = 18
TREND_LOOKBACK_BARS = 12
TREND_MIN_SPEARMAN = 0.25
TREND_MIN_NET_UP = 0.20
TREND_MIN_NET_DN = 0.20

# Pullback
PB_RSI_TH = 38
PB_WR_TH = -75
PB_MIN_BOUNCE_PCT = 0.10
PB_LOOKBACK_HI = 48
PB_LOOKBACK_HI_PCT = 2.5

# Capitulation
CAP_VOLR = 4.0
CAP_WICK_RATIO = 0.55
CAP_ALLOW_BOUNCE = 0.20

# 结构确认
BO_REQUIRE_HHV = True
BO_WINDOW = 20

# 评分
W_VOLR_NOW = 0.35
W_EQ_NOW_USD = 0.40
W_ABS_PCT = 0.15
W_TREND_ALIGN = 0.10
SCALE_EQ5M_USD = 600_000.0
SCALE_ABS_PCT = 1.0

# 小时候选打分
PCT24_ABS_MIN = 0.5
W_PCT24 = 0.65
W_QV24 = 0.35
W_NEAR_EXTREME = 0.05
MAX_CANDIDATES = RANK_BY_1H

# 入场命令（保留）
SAFE_MODE_ALWAYS = False
SAFE_FIB_RATIO = 0.382
SAFE_ATR_MULT = 0.5
SAFE_RETEST_TICKS = 1
SAFE_SL_PCT = 5.0
SAFE_MIN_SL_PCT = 1.5
SAFE_MAX_SL_PCT = 6.0
SAFE_TP_PCTS = (1.5, 3.0, 6.0)
SAFE_MIN_DIST_ATR = 0.25
SAFE_MIN_DIST_TICKS = 3

# 附加权重
W_PATTERN_BO = 0.10
W_PATTERN_PB = 0.10
W_PATTERN_CAP = 0.15

# 15m 量级参数预设
KNOWN_MAJORS = {
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
}
MARKETCAP_MAP: Dict[str, float] = {}
PARAM_PRESETS = {
    "MAJOR": {
        "EXPLODE_VOLR": 2.6,
        "PRICE_UP_TH": 0.25,
        "PRICE_DN_TH": -0.25,
        "CAP_VOLR": 4.5,
        "CAP_WICK_RATIO": 0.60,
        "PB_LOOKBACK_HI_PCT": 1.5,
        "MIN_QV5M_USD": 1_500_000,
    },
    "MID": {
        "EXPLODE_VOLR": 3.0,
        "PRICE_UP_TH": 0.30,
        "PRICE_DN_TH": -0.30,
        "CAP_VOLR": 4.5,
        "CAP_WICK_RATIO": 0.58,
        "PB_LOOKBACK_HI_PCT": 2.0,
        "MIN_QV5M_USD": 1_050_000,
    },
    "ALT": {
        "EXPLODE_VOLR": 3.6,
        "PRICE_UP_TH": 0.40,
        "PRICE_DN_TH": -0.40,
        "CAP_VOLR": 5.5,
        "CAP_WICK_RATIO": 0.62,
        "PB_LOOKBACK_HI_PCT": 3.0,
        "MIN_QV5M_USD": 540_000,
    },
}
COIN_METRICS: Dict[str, Any] = {}
SYMBOL_CLASS: Dict[str, str] = {}

GLOBAL_TWEAKS = {
    "QUIET": {
        "MIN_SIGNAL_SCORE": +0.05,
        "ONLY_PUSH_EXPLODE": False,
        "EXPLODE_VOLR": +0.0,
        "PRICE_UP_TH": +0.00,
        "PRICE_DN_TH": +0.00,
        "NO_FOLLOW_UP_TH": +0.00,
        "ADX_MIN_TREND": +0,
        "TREND_MIN_SPEARMAN": +0.00,
        "TREND_MIN_NET_UP": +0.00,
        "TREND_MIN_NET_DN": +0.00,
        "PB_MIN_BOUNCE_PCT": +0.00,
        "PB_LOOKBACK_HI_PCT": +0.0,
        "CAP_VOLR": +0.0,
        "CAP_WICK_RATIO": +0.00,
    },
    "SENSITIVE": {
        "MIN_SIGNAL_SCORE": -0.05,
        "ONLY_PUSH_EXPLODE": False,
        "EXPLODE_VOLR": -0.4,
        "PRICE_UP_TH": -0.10,
        "PRICE_DN_TH": +0.10,
        "NO_FOLLOW_UP_TH": -0.03,
        "ADX_MIN_TREND": -3,
        "TREND_MIN_SPEARMAN": -0.05,
        "TREND_MIN_NET_UP": -0.05,
        "TREND_MIN_NET_DN": -0.05,
        "PB_MIN_BOUNCE_PCT": -0.04,
        "PB_LOOKBACK_HI_PCT": -0.5,
        "CAP_VOLR": -0.5,
        "CAP_WICK_RATIO": -0.05,
    },
    "AGGRESSIVE": {
        "MIN_SIGNAL_SCORE": -0.10,
        "ONLY_PUSH_EXPLODE": False,
        "EXPLODE_VOLR": -0.7,
        "PRICE_UP_TH": -0.15,
        "PRICE_DN_TH": +0.15,
        "NO_FOLLOW_UP_TH": -0.04,
        "ADX_MIN_TREND": -4,
        "TREND_MIN_SPEARMAN": -0.07,
        "TREND_MIN_NET_UP": -0.07,
        "TREND_MIN_NET_DN": -0.07,
        "PB_MIN_BOUNCE_PCT": -0.06,
        "PB_LOOKBACK_HI_PCT": -0.9,
        "CAP_VOLR": -0.7,
        "CAP_WICK_RATIO": -0.07,
    },
}
PRESET_TWEAKS = {
    "QUIET": {
        "MAJOR": {
            "EXPLODE_VOLR": 2.6,
            "PRICE_UP_TH": 0.25,
            "PRICE_DN_TH": -0.25,
            "CAP_VOLR": 4.5,
            "CAP_WICK_RATIO": 0.60,
            "PB_LOOKBACK_HI_PCT": 1.5,
            "MIN_QV5M_USD": 1_500_000,
        },
        "MID": {
            "EXPLODE_VOLR": 3.0,
            "PRICE_UP_TH": 0.30,
            "PRICE_DN_TH": -0.30,
            "CAP_VOLR": 4.5,
            "CAP_WICK_RATIO": 0.58,
            "PB_LOOKBACK_HI_PCT": 2.0,
            "MIN_QV5M_USD": 1_050_000,
        },
        "ALT": {
            "EXPLODE_VOLR": 3.6,
            "PRICE_UP_TH": 0.40,
            "PRICE_DN_TH": -0.40,
            "CAP_VOLR": 5.5,
            "CAP_WICK_RATIO": 0.62,
            "PB_LOOKBACK_HI_PCT": 3.0,
            "MIN_QV5M_USD": 540_000,
        },
    },
    "SENSITIVE": {
        "MAJOR": {
            "EXPLODE_VOLR": 2.2,
            "PRICE_UP_TH": 0.20,
            "PRICE_DN_TH": -0.20,
            "CAP_VOLR": 4.0,
            "CAP_WICK_RATIO": 0.55,
            "PB_LOOKBACK_HI_PCT": 1.2,
            "MIN_QV5M_USD": 1_350_000,
        },
        "MID": {
            "EXPLODE_VOLR": 2.6,
            "PRICE_UP_TH": 0.25,
            "PRICE_DN_TH": -0.25,
            "CAP_VOLR": 4.0,
            "CAP_WICK_RATIO": 0.54,
            "PB_LOOKBACK_HI_PCT": 1.6,
            "MIN_QV5M_USD": 900_000,
        },
        "ALT": {
            "EXPLODE_VOLR": 3.0,
            "PRICE_UP_TH": 0.30,
            "PRICE_DN_TH": -0.30,
            "CAP_VOLR": 5.0,
            "CAP_WICK_RATIO": 0.58,
            "PB_LOOKBACK_HI_PCT": 2.4,
            "MIN_QV5M_USD": 540_000,
        },
    },
    "AGGRESSIVE": {
        "MAJOR": {
            "EXPLODE_VOLR": 2.0,
            "PRICE_UP_TH": 0.18,
            "PRICE_DN_TH": -0.18,
            "CAP_VOLR": 3.6,
            "CAP_WICK_RATIO": 0.52,
            "PB_LOOKBACK_HI_PCT": 1.0,
            "MIN_QV5M_USD": 1_200_000,
        },
        "MID": {
            "EXPLODE_VOLR": 2.3,
            "PRICE_UP_TH": 0.22,
            "PRICE_DN_TH": -0.22,
            "CAP_VOLR": 3.6,
            "CAP_WICK_RATIO": 0.51,
            "PB_LOOKBACK_HI_PCT": 1.4,
            "MIN_QV5M_USD": 780_000,
        },
        "ALT": {
            "EXPLODE_VOLR": 2.8,
            "PRICE_UP_TH": 0.28,
            "PRICE_DN_TH": -0.28,
            "CAP_VOLR": 4.6,
            "CAP_WICK_RATIO": 0.55,
            "PB_LOOKBACK_HI_PCT": 2.0,
            "MIN_QV5M_USD": 540_000,
        },
    },
}

# QUIET 模式：爆发类额外门槛
EXPLODE_QUIET_EXTRA_SCORE = 0.12
EXPLODE_MAX_BAR_AGE_FRAC = 0.33
EXPLODE_MIN_DIST_DAYEXT_PCT = 0.6
EXPLODE_REQUIRE_CONTRACTION = True
EXPLODE_CONTRACTION_BB_WIDTH = 0.025


def _apply_overrides(obj, overrides):
    for k, v in overrides.items():
        obj[k] = v


def apply_mode():
    global MIN_SIGNAL_SCORE, ONLY_PUSH_EXPLODE
    global EXPLODE_VOLR, PRICE_UP_TH, PRICE_DN_TH, NO_FOLLOW_UP_TH
    global ADX_MIN_TREND, TREND_MIN_SPEARMAN, TREND_MIN_NET_UP, TREND_MIN_NET_DN
    global PB_MIN_BOUNCE_PCT, PB_LOOKBACK_HI_PCT
    global CAP_VOLR, CAP_WICK_RATIO
    global PARAM_PRESETS

    mode = MODE if MODE in GLOBAL_TWEAKS else "SENSITIVE"
    g = GLOBAL_TWEAKS[mode]
    MIN_SIGNAL_SCORE = max(0.0, MIN_SIGNAL_SCORE + g["MIN_SIGNAL_SCORE"])
    ONLY_PUSH_EXPLODE = g["ONLY_PUSH_EXPLODE"]
    EXPLODE_VOLR = max(1.0, EXPLODE_VOLR + g["EXPLODE_VOLR"])
    PRICE_UP_TH = max(0.0, PRICE_UP_TH + g["PRICE_UP_TH"])
    PRICE_DN_TH = PRICE_DN_TH + g["PRICE_DN_TH"]
    NO_FOLLOW_UP_TH = max(0.0, NO_FOLLOW_UP_TH + g["NO_FOLLOW_UP_TH"])
    ADX_MIN_TREND = max(0, ADX_MIN_TREND + g["ADX_MIN_TREND"])
    TREND_MIN_SPEARMAN = TREND_MIN_SPEARMAN + g["TREND_MIN_SPEARMAN"]
    TREND_MIN_NET_UP = TREND_MIN_NET_UP + g["TREND_MIN_NET_UP"]
    TREND_MIN_NET_DN = TREND_MIN_NET_DN + g["TREND_MIN_NET_DN"]
    PB_MIN_BOUNCE_PCT = max(0.0, PB_MIN_BOUNCE_PCT + g["PB_MIN_BOUNCE_PCT"])
    PB_LOOKBACK_HI_PCT = max(0.0, PB_LOOKBACK_HI_PCT + g["PB_LOOKBACK_HI_PCT"])
    CAP_VOLR = max(1.0, CAP_VOLR + g["CAP_VOLR"])
    CAP_WICK_RATIO = min(0.95, max(0.0, CAP_WICK_RATIO + g["CAP_WICK_RATIO"]))
    base = PRESET_TWEAKS[mode]
    merged = deepcopy(PARAM_PRESETS)
    for cls in ("MAJOR", "MID", "ALT"):
        if cls in base:
            _apply_overrides(merged[cls], base[cls])
    PARAM_PRESETS = merged


apply_mode()

# 供 detect 模块动态调整
PB_MIN_BOUNCE_PCT_BASE = PB_MIN_BOUNCE_PCT
CAP_VOLR_BASE = CAP_VOLR
CAP_WICK_RATIO_BASE = CAP_WICK_RATIO
