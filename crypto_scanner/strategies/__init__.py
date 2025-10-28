# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Strategy:
    # —— 周期相关 —— #
    timeframe_fast: str  # 主检测周期，如 "15m" / "1h"
    timeframe_htf: Optional[str]  # 高周期闸门，如 "1h" / "4h"；没有就设 None
    bars_per_day: int  # 一天多少根 fast K（15m=96, 1h=24）

    # —— 关键阈值/权重（仅放“与周期耦合”的差异） —— #
    bb_width_th: float  # BB 挤压阈值（width<=x 视为挤压）
    explode_contraction_width: float
    explode_min_dist_dayext_pct: float
    scale_eq_bar_usd: float  # 评分标尺
    w_volr_now: float
    w_eq_now_usd: float
    w_abs_pct: float
    w_trend_align: float

    # —— 其它周期化偏好 —— #
    pb_lookback_hi: int

    # —— 继承 config.py 的全局常量（不用改的就不放） —— #
    # 例如：MIN_QV5M_USD, EXPLODE_VOLR 等，仍使用 config.py 中的值；
    # 如果策略希望覆盖它们，可在 detect.py 中按策略进行 override。

    # —— 策略名（日志/调试用） —— #
    name: str

    # 覆盖项：允许策略对全局 config 做“按需覆盖”
    overrides: Optional[Dict[str, Any]] = None
