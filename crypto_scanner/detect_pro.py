# -*- coding: utf-8 -*-
"""
detect_signal.py
最终版：H1 + H4，结构信号4 + 价格行为信号3 = 共7个职业级信号
不使用15m，不产生噪音，不过度计算。
返回格式与 multi_loop 兼容。
"""

import numpy as np
import pandas as pd
import talib as ta

from .exchange import fetch_ohlcv_df, get_tick_size
from .liquidity import get_day_stats
from .config import (
    KINDS_CN,
    SAFE_SL_PCT,
    SAFE_ATR_MULT,
)
from .loggingx import dbg  # ⭐ 新增：引入日志

# =========================
# 工具函数
# =========================

# =========================
# SR 辅助函数（融合版：v3 核心 + 安全填充）
# =========================

from typing import List, Tuple, Dict, Optional

# SR 间最小相对间距（相对于现价，用在稀疏化阶段）
SR_MIN_GAP_PCT = 0.015  # 1.5%
# 最小 S/R 级别数（仅在 mode="safe" 下触发几何填充）
SR_MIN_LEVEL_COUNT = 3


def _sr_swing_points(
    df: pd.DataFrame, window: int = 2
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    从 df 里找摆动高/低点:
    - 返回: (R_points, S_points)，每个都是 [(idx, price), ...]
    """
    highs, lows = df["high"].values, df["low"].values
    R, S = [], []
    n = len(df)
    for i in range(window, n - window):
        if highs[i] >= np.max(highs[i - window : i + window + 1]):
            R.append((i, float(highs[i])))
        if lows[i] <= np.min(lows[i - window : i + window + 1]):
            S.append((i, float(lows[i])))
    return R, S


def _sr_cluster_levels(
    points: List[Tuple[int, float]], band_abs: float, n_total: int
) -> List[Dict]:
    """
    对摆动点按 band_abs 聚类，输出带 score 的 SR 水平
    kind = 'swing'
    """
    if not points:
        return []
    pts = sorted(points, key=lambda x: x[1])
    clusters, cur = [], [pts[0]]
    for p in pts[1:]:
        if abs(p[1] - np.mean([x[1] for x in cur])) <= band_abs:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)

    lvls = []
    for c in clusters:
        prices = [x[1] for x in c]
        idxs = [x[0] for x in c]
        price = float(np.mean(prices))
        touches = len(c)
        recency = 0.5 * (max(idxs) / (n_total - 1)) if n_total > 1 else 0.0
        lvls.append(
            {
                "price": price,
                "touches": touches,
                "last_idx": max(idxs),
                "score": touches + recency,
                "kind": "swing",
            }
        )
    lvls.sort(key=lambda x: (x["score"], x["touches"]), reverse=True)
    return lvls


def _sr_densify_support_by_hist(
    lows: np.ndarray, current: float, band_abs: float, max_levels: int
) -> List[Dict]:
    """
    用 band_abs 为桶宽，对 <= current 的所有 low 做直方密度，挑出峰值作为补充支撑
    kind = 'hist'
    """
    arr = np.array([x for x in lows if x <= current], dtype=float)
    if len(arr) == 0 or max_levels <= 0:
        return []
    lo, hi = arr.min(), current
    bins = max(8, int((hi - lo) / max(1e-12, band_abs)))
    bins = min(bins, 200)
    if bins <= 0:
        return []

    hist, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0

    idxs = np.argsort(-hist)
    out = []
    for k in idxs[:max_levels]:
        if hist[k] <= 0:
            continue
        out.append(
            {
                "price": float(centers[k]),
                "touches": int(hist[k]),
                "last_idx": 0,
                "score": float(hist[k]),
                "kind": "hist",
            }
        )

    # 相近中心合并去重（<= band_abs）
    out_sorted = sorted(out, key=lambda x: x["price"])
    merged: List[Dict] = []
    for node in out_sorted:
        if not merged or abs(node["price"] - merged[-1]["price"]) > band_abs:
            merged.append(node)
        else:
            prev = merged[-1]
            merged[-1] = {
                "price": (prev["price"] + node["price"]) / 2.0,
                "touches": prev["touches"] + node["touches"],
                "last_idx": max(prev["last_idx"], node["last_idx"]),
                "score": prev["score"] + node["score"],
                "kind": "hist",
            }

    merged.sort(key=lambda x: (x["score"], x["touches"]), reverse=True)
    return merged


def _sr_compute_band_and_atr(df: pd.DataFrame) -> Tuple[float, float]:
    """
    自适应带宽:
    band_abs = max(0.25*ATR, 0.2%*price)
    """
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    px = float(close[-1])
    if len(close) < 2:
        return max(px * 0.002, 1e-12), px

    hl = high - low
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr = np.maximum(hl[1:], np.maximum(hc, lc))
    atr = float(pd.Series(tr).rolling(14, min_periods=1).mean().iloc[-1])
    band_abs = max(atr * 0.25, px * 0.002)
    return band_abs, px


# ========= Fib：ABC 三点扩展 + 摆动 leg 回撤 + Rmin 兜底 =========


def _sr_find_abc_points(
    R_pts: List[Tuple[int, float]],
    S_pts: List[Tuple[int, float]],
    df: pd.DataFrame,
    current: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    找 ABC 三点用于 Fib Extension
    trend_type: 'up' / 'down' / ''
    """
    combined = []
    for idx, price in R_pts:
        combined.append(("H", idx, float(price)))
    for idx, price in S_pts:
        combined.append(("L", idx, float(price)))

    if len(combined) < 2:
        return None, None, None, ""

    combined.sort(key=lambda x: x[1])

    if len(combined) >= 3:
        p0, p1, p2 = combined[-3], combined[-2], combined[-1]
        # 上升：低-高-回调低
        if p0[0] == "L" and p1[0] == "H" and p2[0] == "L":
            return p0[2], p1[2], p2[2], "up"
        # 下降：高-低-反弹高
        if p0[0] == "H" and p1[0] == "L" and p2[0] == "H":
            return p0[2], p1[2], p2[2], "down"

        # 不标准，就用最后两个点 + 当前价
        p0, p1 = combined[-2], combined[-1]
        if p0[0] == "L" and p1[0] == "H":
            return p0[2], p1[2], current, "up"
        if p0[0] == "H" and p1[0] == "L":
            return p0[2], p1[2], current, "down"

    # 再退化：最后两点 + 当前价
    p0, p1 = combined[-2], combined[-1]
    if p0[0] != p1[0]:
        trend = "up" if p0[0] == "L" else "down"
        return p0[2], p1[2], current, trend

    return None, None, None, ""


def _sr_fib_extension_abc_candidates(
    df: pd.DataFrame,
    current: float,
    R_pts: List[Tuple[int, float]],
    S_pts: List[Tuple[int, float]],
) -> List[Dict]:
    """
    v3: 改进的 Fibonacci Extension（标准 ABC 三点法）
    不区分 side，这里统一生成，再按 price>current / <current 划分到 R/S。
    kind = 'fib_ext_R' / 'fib_ext_S'
    """
    A, B, C, trend = _sr_find_abc_points(R_pts, S_pts, df, current)
    if A is None or B is None or C is None:
        return []

    ab_range = abs(B - A)
    if ab_range < current * 0.005:  # AB 波段 < 0.5% 时不做 Fib
        return []

    ext_ratios = [0.618, 1.0, 1.272, 1.618, 2.0]
    out: List[Dict] = []

    for r in ext_ratios:
        p: Optional[float] = None
        kind: str = ""

        if trend == "up":
            # C + (B-A) * ratio -> 阻力
            p = C + ab_range * r
            kind = "fib_ext_R"
            if p <= current:
                continue
        elif trend == "down":
            # C - (A-B) * ratio -> 支撑
            p = C - ab_range * r
            kind = "fib_ext_S"
            if p >= current:
                continue
        else:
            continue

        dist_pct = abs(p - current) / current
        if dist_pct < 0.003:  # <0.3% 视为噪音
            continue
        if dist_pct > 0.4:  # >40% 太远
            continue

        out.append(
            {
                "price": float(p),
                "touches": 0,
                "last_idx": len(df) - 1,
                "score": 0.3,  # 比 swing 低一点
                "kind": kind,
                "ratio": r,
            }
        )

    return out


def _sr_find_last_swing_leg(
    R_pts: List[Tuple[int, float]],
    S_pts: List[Tuple[int, float]],
    df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    v3: 从摆动高低点中找到最近一段 swing 区间 (leg_low, leg_high)
    """
    combined = []
    for idx, price in R_pts:
        combined.append(("H", idx, float(price)))
    for idx, price in S_pts:
        combined.append(("L", idx, float(price)))

    if not combined:
        hi = float(df["high"].max())
        lo = float(df["low"].min())
        return lo, hi

    combined.sort(key=lambda x: x[1])

    leg_low = None
    leg_high = None
    for i in range(len(combined) - 1, 0, -1):
        typ1, idx1, p1 = combined[i]
        typ0, idx0, p0 = combined[i - 1]
        if typ1 != typ0:
            leg_low = min(p0, p1)
            leg_high = max(p0, p1)
            break

    if leg_low is None or leg_high is None or leg_high - leg_low < 1e-8:
        hi = float(df["high"].max())
        lo = float(df["low"].min())
        return lo, hi

    return leg_low, leg_high


def _sr_fib_retracement_candidates(
    df: pd.DataFrame,
    current: float,
    R_pts: List[Tuple[int, float]],
    S_pts: List[Tuple[int, float]],
) -> List[Dict]:
    """
    v3: 基于最近一个 H/L 摆动区间 (leg)
    生成内部回撤 (Retracement) 和外部扩展 (Extension)
    kind = 'fib_retr'
    """
    leg_low, leg_high = _sr_find_last_swing_leg(R_pts, S_pts, df)
    rng = leg_high - leg_low

    if rng < current * 0.005:  # 0.5%
        return []

    retr_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    ext_ratios = [0.236, 0.382, 0.618, 1.0, 1.272, 1.618]

    out: List[Dict] = []

    def add_level(p, r):
        dist_pct = abs(p - current) / current
        if dist_pct < 0.003 or dist_pct > 0.4:  # 0.3% ~ 40%
            return
        out.append(
            {
                "price": float(p),
                "touches": 0,
                "last_idx": len(df) - 1,
                "score": 0.2,  # 权重低于 ABC 扩展
                "kind": "fib_retr",
                "ratio": r,
            }
        )

    # 1. 内部回撤
    for r in retr_ratios:
        p = leg_high - rng * r
        add_level(p, r)

    # 2. 向上扩展
    for r in ext_ratios:
        p = leg_high + rng * r
        add_level(p, r)

    # 3. 向下扩展
    for r in ext_ratios:
        p = leg_low - rng * r
        if p > 0:
            add_level(p, r)

    return out


def _sr_fib_from_rmin_candidates(
    df: pd.DataFrame,
    current: float,
    R_lvls_all: List[Dict],
    band_abs: float,
    max_levels: int,
) -> List[Dict]:
    """
    v3: 极端兜底：如果 S 这边特别少，用 R 里的最小价作为 anchor 向下扩展几级支撑
    kind = 'fib_rmin'
    """
    if max_levels <= 0 or not R_lvls_all:
        return []

    anchor = min(r["price"] for r in R_lvls_all)
    rng = max(band_abs * 4, anchor * 0.01)
    ratios = [1.0, 1.618, 2.0]

    out: List[Dict] = []
    for r in ratios:
        p = anchor - rng * r
        if p <= 0 or p >= current:
            continue
        if abs(p - current) / current < 0.01:  # 至少 1% 间距
            continue
        out.append(
            {
                "price": float(p),
                "touches": 0,
                "last_idx": len(df) - 1,
                "score": 0.1,
                "kind": "fib_rmin",
                "ratio": r,
            }
        )

    out.sort(key=lambda x: abs(x["price"] - current))
    return out[:max_levels]


def _sr_select_levels(
    candidates: List[Dict],
    current: float,
    min_gap_abs: float,
    side: str,
    topn: int,
) -> List[Dict]:
    """
    v3: 候选池 -> 稀疏 SR
    从候选池中，按 kind 优先级 + 距离现价远近，做贪心选择
    """
    if not candidates or topn <= 0:
        return []

    if side == "R":
        pool = [x for x in candidates if x["price"] > current]
    else:
        pool = [x for x in candidates if x["price"] < current]

    if not pool:
        return []

    kind_priority = {
        "swing": 5,
        "fib_ext_R": 4,
        "fib_ext_S": 4,
        "hist": 3,
        "fib_retr": 2,
        "fib_rmin": 1,
    }

    pool_sorted = sorted(
        pool,
        key=lambda x: (
            -kind_priority.get(x.get("kind", ""), 0),
            abs(x["price"] - current),
        ),
    )

    selected: List[Dict] = []
    for lvl in pool_sorted:
        if len(selected) >= topn:
            break
        ok = True
        for s in selected:
            if abs(lvl["price"] - s["price"]) < min_gap_abs:
                ok = False
                break
        if ok:
            selected.append(lvl)

    if side == "R":
        selected.sort(key=lambda x: x["price"])
    else:
        selected.sort(key=lambda x: x["price"], reverse=True)

    return selected


def _sr_fill_missing_levels(
    existing_levels: List[Dict],
    current: float,
    side: str,
    min_count: int,
    min_gap_abs: float,
    n_total: int,
) -> List[Dict]:
    """
    detect_signal 特有：如果 S/R 级别太少，用几何级数扩展补充，确保至少有 min_count 个。
    kind = 'geometric_fill'
    仅在 compute_sr_levels_from_df(..., mode="safe") 下使用
    """
    n_missing = min_count - len(existing_levels)
    if n_missing <= 0:
        return existing_levels

    filled = list(existing_levels)
    base_price = filled[-1]["price"] if filled else current

    step_ratio = 1.025 if side == "R" else 0.975
    next_price = base_price * step_ratio

    # 先保证第一个扩展点和 base 有足够距离
    while abs(next_price - base_price) < min_gap_abs:
        next_price *= step_ratio
        if next_price <= 0 or next_price > current * 5 or next_price < current * 0.05:
            return filled

    for _ in range(n_missing):
        if filled and abs(next_price - filled[-1]["price"]) < min_gap_abs:
            next_price *= step_ratio

        if side == "R" and next_price <= base_price:
            next_price *= step_ratio
            continue
        if side == "S" and next_price >= base_price:
            next_price *= step_ratio
            continue

        if next_price <= 0:
            break

        filled.append(
            {
                "price": float(next_price),
                "touches": 0,
                "last_idx": n_total - 1,
                "score": 0.05,
                "kind": "geometric_fill",
            }
        )

        base_price = next_price
        next_price *= step_ratio

    if side == "R":
        filled.sort(key=lambda x: x["price"])
    else:
        filled.sort(key=lambda x: x["price"], reverse=True)

    return filled


def compute_sr_levels_from_df(
    df: pd.DataFrame,
    current_price: float,
    topn: int = 6,
    mode: str = "safe",  # "pure"：纯 v3 逻辑；"safe"：强制补齐最少级别
):
    """
    通用：给一个 OHLCV df（1h / 4h 都行）+ 当前价，算 SR。
    返回:
        near_R_dict, near_S_dict, R_out(list[dict]), S_out(list[dict]), band_abs
    """
    df = df.copy().reset_index(drop=True)
    if len(df) < 50:
        return None, None, [], [], None

    band_abs, _ = _sr_compute_band_and_atr(df)
    R_pts, S_pts = _sr_swing_points(df, window=2)

    # 1. 基础 swing 聚类
    R_lvls_all = _sr_cluster_levels(R_pts, band_abs, len(df))
    S_lvls_all = _sr_cluster_levels(S_pts, band_abs, len(df))

    # 2. Fib 候选（ABC 扩展 + 摆动区间回撤）
    abc_ext_levels = _sr_fib_extension_abc_candidates(
        df=df,
        current=current_price,
        R_pts=R_pts,
        S_pts=S_pts,
    )
    retr_levels = _sr_fib_retracement_candidates(
        df=df,
        current=current_price,
        R_pts=R_pts,
        S_pts=S_pts,
    )
    all_fib_levels = abc_ext_levels + retr_levels

    # 3. 极端兜底（从最小阻力向下扩展支撑）
    fallback_s = _sr_fib_from_rmin_candidates(
        df=df,
        current=current_price,
        R_lvls_all=R_lvls_all,
        band_abs=band_abs,
        max_levels=4,
    )

    # 4. 构建候选池
    R_candidates: List[Dict] = []
    S_candidates: List[Dict] = []

    # 阻力候选池：swing + Fib 中 price > current 的
    R_candidates.extend(R_lvls_all)
    R_candidates.extend([lvl for lvl in all_fib_levels if lvl["price"] > current_price])

    # 支撑候选池：swing + Fib 中 price < current 的 + 直方图 + 兜底
    S_candidates.extend(S_lvls_all)
    S_candidates.extend([lvl for lvl in all_fib_levels if lvl["price"] < current_price])

    hist_s = _sr_densify_support_by_hist(
        df["low"].values.astype(float),
        current=current_price,
        band_abs=band_abs,
        max_levels=10,
    )
    S_candidates.extend(hist_s)
    S_candidates.extend(fallback_s)

    # 5. 贪心选择 + 最小间距（用 SR_MIN_GAP_PCT 覆盖 v3 原来的 MIN_GAP_PCT）
    min_gap_abs = max(band_abs, current_price * SR_MIN_GAP_PCT)

    R_raw = _sr_select_levels(
        candidates=R_candidates,
        current=current_price,
        min_gap_abs=min_gap_abs,
        side="R",
        topn=topn,
    )
    S_raw = _sr_select_levels(
        candidates=S_candidates,
        current=current_price,
        min_gap_abs=min_gap_abs,
        side="S",
        topn=topn,
    )

    # 6. 强制补齐至少 SR_MIN_LEVEL_COUNT 个（仅 safe 模式）
    if mode == "safe":
        R_out = _sr_fill_missing_levels(
            existing_levels=R_raw,
            current=current_price,
            side="R",
            min_count=SR_MIN_LEVEL_COUNT,
            min_gap_abs=min_gap_abs,
            n_total=len(df),
        )
        S_out = _sr_fill_missing_levels(
            existing_levels=S_raw,
            current=current_price,
            side="S",
            min_count=SR_MIN_LEVEL_COUNT,
            min_gap_abs=min_gap_abs,
            n_total=len(df),
        )
        R_out = R_out[:topn]
        S_out = S_out[:topn]
    else:
        # mode="pure"：完全按 v3 输出，不做几何补线
        R_out = R_raw[:topn]
        S_out = S_raw[:topn]

    # 最近一上/一下
    near_R = next((x for x in R_out if x["price"] >= current_price), None)
    near_S = next((x for x in S_out if x["price"] <= current_price), None)

    return near_R, near_S, R_out, S_out, band_abs


def ema(arr, n):
    arr = np.array(arr, dtype=float)
    if len(arr) < n:
        return None
    k = 2 / (n + 1)
    ema_arr = []
    prev = arr[0]
    for price in arr:
        prev = price * k + prev * (1 - k)
        ema_arr.append(prev)
    return ema_arr[-1]


def swing_high(arr, idx, left=2, right=2):
    if idx < left or idx + right >= len(arr):
        return False
    p = arr[idx]
    for i in range(1, left + 1):
        if arr[idx - i] >= p:
            return False
    for i in range(1, right + 1):
        if arr[idx + i] >= p:
            return False
    return True


def swing_low(arr, idx, left=2, right=2):
    if idx < left or idx + right >= len(arr):
        return False
    p = arr[idx]
    for i in range(1, left + 1):
        if arr[idx - i] <= p:
            return False
    for i in range(1, right + 1):
        if arr[idx + i] <= p:
            return False
    for i in range(1, right + 1):
        if arr[idx + i] <= p:
            return False
    return True


# =========================
# SR（支撑/阻力）简单检测：关键点极值
# =========================


def detect_sr_levels(h1_closes, window=30):
    """
    基于 swing 高低点的简化 SR 检测
    """
    n = len(h1_closes)
    highs = []
    lows = []
    for i in range(n):
        if swing_high(h1_closes, i, 2, 2):
            highs.append(h1_closes[i])
        if swing_low(h1_closes, i, 2, 2):
            lows.append(h1_closes[i])

    highs = sorted(list(set(highs)))
    lows = sorted(list(set(lows)))

    return highs[-5:], lows[:5]  # 上方阻力5，下方支撑5


# =========================
# H4 趋势方向（EMA50 vs EMA200）
# =========================


def h4_direction(h4_closes):
    ema50 = ema(h4_closes, 50)
    ema200 = ema(h4_closes, 200)
    if ema50 is None or ema200 is None:
        return None

    if ema50 > ema200:
        return "long"
    elif ema50 < ema200:
        return "short"
    else:
        return None


# =========================
# 七大核心信号（H1）
# =========================


def signal_trend_break(h1_closes):
    """
    趋势突破：突破最近结构点
    """
    if len(h1_closes) < 5:
        return None, None

    last = h1_closes[-1]
    prev = h1_closes[-2]

    # 找最近 swing high/low
    highs = []
    lows = []
    for i in range(len(h1_closes)):
        if swing_high(h1_closes, i):
            highs.append(h1_closes[i])
        if swing_low(h1_closes, i):
            lows.append(h1_closes[i])

    if not highs or not lows:
        return None, None

    last_high = highs[-1]
    last_low = lows[0]

    if last > last_high and prev <= last_high:
        return "trend_break_up", last_high
    if last < last_low and prev >= last_low:
        return "trend_break_down", last_low

    return None, None


def signal_pullback(h1_closes):
    """
    EMA20 回踩确认
    """
    if len(h1_closes) < 25:
        return None, None

    ema20 = ema(h1_closes, 20)
    last = h1_closes[-1]
    prev = h1_closes[-2]

    # 回踩上来
    if prev < ema20 and last > ema20:
        return "pullback_long", ema20
    # 回踩下去
    if prev > ema20 and last < ema20:
        return "pullback_short", ema20

    return None, None


def signal_sr_flip(h1_closes, sr_highs, sr_lows):
    """
    SR flip（最强 SR 反转）
    """
    price = h1_closes[-1]

    # 上方阻力翻转 → 做空
    for r in sr_highs:
        if abs(price - r) / r < 0.004:  # 0.4%
            if price < r:  # 反转
                return "sr_flip_short", r

    # 下方支撑翻转 → 做多
    for s in sr_lows:
        if abs(price - s) / s < 0.004:
            if price > s:
                return "sr_flip_long", s

    return None, None


def signal_range_break(h1_closes):
    """
    区间突破：最后5根区间 + 突破
    """
    window = h1_closes[-6:-1]
    if not window:
        return None, None

    last = h1_closes[-1]
    high_max = max(window)
    low_min = min(window)

    if last > high_max:
        return "range_break_up", high_max
    if last < low_min:
        return "range_break_down", low_min

    return None, None


# ===== 价格行为 PA =====


def signal_pinbar(h1_opens, h1_closes, h1_highs, h1_lows):
    """
    长影线反转（Pin Bar）
    """
    O = h1_opens[-1]
    C = h1_closes[-1]
    H = h1_highs[-1]
    L = h1_lows[-1]

    body = abs(C - O)
    upper = H - max(C, O)
    lower = min(C, O) - L

    full = H - L
    if full == 0:
        return None, None

    # 下影线长 → 多
    if lower > body * 2 and lower > full * 0.5:
        return "pinbar_long", L

    # 上影线长 → 空
    if upper > body * 2 and upper > full * 0.5:
        return "pinbar_short", H

    return None, None


def signal_engulfing(h1_opens, h1_closes):
    """
    吞没形态
    """
    if len(h1_opens) < 3:
        return None, None

    O1, C1 = h1_opens[-2], h1_closes[-2]
    O2, C2 = h1_opens[-1], h1_closes[-1]

    # 看涨吞没
    if C2 > O2 and O1 > C1 and C2 >= O1 and O2 <= C1:
        return "engulfing_long", (O1 + C1) / 2

    # 看跌吞没
    if C2 < O2 and O1 < C1 and C2 <= O1 and O2 >= C1:
        return "engulfing_short", (O1 + C1) / 2

    return None, None


def signal_fakey(h1_highs, h1_lows):
    """
    假突破 Fakey
    必须扫过 swing 高/低，然后快速反向
    """
    if len(h1_highs) < 4:
        return None, None

    last = h1_highs[-1]
    prev = h1_highs[-2]

    # upward fake
    if last < prev and h1_highs[-2] == max(h1_highs[-5:]):
        return "fakey_short", prev

    # downward fake
    if last > prev and h1_lows[-2] == min(h1_lows[-5:]):
        return "fakey_long", prev

    return None, None


def _recent_trend_dir(
    closes: np.ndarray, atr_arr: np.ndarray, N: int = 12
) -> Tuple[Optional[str], float]:
    """
    最近一段是否存在相对干净的单边趋势：
    - 使用两个窗口：
        · window_ext: 从最近局部高/低点算整体涨跌幅
        · window_dir: 统计最近 K 根里，收阴/收阳占比
    返回:
        ("down"/"up"/None, trend_pct)
        trend_pct: 对于 down 用整体跌幅 drop_pct；对于 up 用整体涨幅 rise_pct
    """
    window_ext = 24
    window_dir = N

    n = len(closes)
    if n <= max(window_ext, window_dir) + 3:
        return None, 0.0

    closes = np.asarray(closes, dtype=float)

    # 1) 从最近 window_ext 段里找局部高/低点，与当前价比较
    seg_ext = closes[-window_ext - 1 : -1]  # 留一根给“当前”
    local_max = float(seg_ext.max())
    local_min = float(seg_ext.min())
    last = float(closes[-1])

    drop_pct = (last / local_max - 1.0) * 100.0  # 从最近高点跌了多少
    rise_pct = (last / local_min - 1.0) * 100.0  # 从最近低点涨了多少

    # 2) 最近 window_dir 段方向统计（收阴/收阳）
    seg_dir = closes[-window_dir - 1 :]
    down_cnt = 0
    up_cnt = 0
    for i in range(1, len(seg_dir)):
        if seg_dir[i] < seg_dir[i - 1]:
            down_cnt += 1
        elif seg_dir[i] > seg_dir[i - 1]:
            up_cnt += 1
    steps = max(len(seg_dir) - 1, 1)
    down_ratio = down_cnt / float(steps)
    up_ratio = up_cnt / float(steps)

    # 阈值：整体跌/涨 ≥ 3%，且方向占比 ≥ 60%
    if drop_pct <= -3.0 and down_ratio >= 0.6:
        return "down", drop_pct
    if rise_pct >= 3.0 and up_ratio >= 0.6:
        return "up", rise_pct

    # 没形成清晰单边
    return None, 0.0


def signal_exhaustion_reversal(
    closes_h1: np.ndarray,
    rsi_series: np.ndarray,
    atr_arr: np.ndarray,
    pct_now: float,
    volr_now: float,
    o: float,
    h: float,
    l: float,
    c: float,
):
    """
    趋势衰竭 + V 型反转检测：
    1) 先用 _recent_trend_dir 判断是否存在一段相对干净的单边（down / up）
    2) 再看动能是否衰竭（ATR 连续变小 或 RSI 有轻微背离）
    3) 当前 K 需是反向带量 K 线（多：阳 + pct≥0.7% + VolR≥1.8；空：反之）
    返回:
        ("exhaustion_reversal_long/short", level) or (None, None)
    """
    closes_h1 = np.asarray(closes_h1, dtype=float)
    atr_arr = np.asarray(atr_arr, dtype=float)
    rsi_series = np.asarray(rsi_series, dtype=float)

    # 给 RSI / ATR 留出足够历史
    if len(closes_h1) < 50 or len(atr_arr) < 20:
        return None, None

    trend_dir, trend_pct = _recent_trend_dir(closes_h1, atr_arr, N=12)
    if trend_dir is None:
        return None, None

    # --- 动能衰竭：ATR 连续缩窄 ---
    atr1, atr2, atr3 = atr_arr[-1], atr_arr[-2], atr_arr[-3]
    atr_exhaust = bool(atr1 < atr2 < atr3)

    # --- RSI 轻微背离 ---
    r_now = float(rsi_series[-1])
    # 往前取一个小 offset 的 RSI 做对比（避免刚好是当前这几根的噪音）
    r_prev_raw = rsi_series[-4] if len(rsi_series) >= 4 else rsi_series[-1]
    r_prev = float(r_prev_raw) if not np.isnan(r_prev_raw) else r_now

    if trend_dir == "down":
        # 底部 RSI 稍微抬高一点即可（不要求很极端）
        rsi_exhaust = r_now > r_prev + 1.0

        if not (atr_exhaust or rsi_exhaust):
            return None, None

        # 反转确认：阳线 + 有一定涨幅 + 放量
        if c > o and pct_now >= 0.7 and volr_now >= 1.8:
            return "exhaustion_reversal_long", c
        return None, None

    elif trend_dir == "up":
        # 顶部 RSI 稍微回落
        rsi_exhaust = r_now < r_prev - 1.0

        if not (atr_exhaust or rsi_exhaust):
            return None, None

        # 反转确认：阴线 + 有一定跌幅 + 放量
        if c < o and pct_now <= -0.7 and volr_now >= 1.8:
            return "exhaustion_reversal_short", c
        return None, None

    return None, None


# =========================
# 主 detect_signal 函数
# =========================


def detect_signal(ex, symbol, strong_up_map, strong_dn_map, strategy):
    """
    极简 H1+H4 版信号检测（只用于合约短线）
    信号类型（核心）：
        - breakout_up / breakout_down：1h 硬突破近 20 根区间
        - wick_bottom / wick_top：极端长影线反转
        - htf_trend_pullback_long/short：4h 趋势回踩 + 1h 拒绝
        - breakout_retest_long/short：突破后的回踩确认
        - range_reject_long/short：区间边缘拒绝
        - double_top / double_bottom：H1 强双顶/双底（RSI 背离 + 颈线确认）
    返回: (ok: bool, payload: dict|None)
    """

    TIMEFRAME_FAST = "1h"
    BAR_SEC = 3600

    dbg(f"[DETECT] {symbol}: start H1+H4 detect")

    # ===== 拉 K 线 =====
    try:
        h1_full = fetch_ohlcv_df(ex, symbol, "1h", limit=288)
        h4 = fetch_ohlcv_df(ex, symbol, "4h", limit=160)
    except Exception as e:
        dbg(f"[DETECT] {symbol}: fetch_ohlcv failed: {e}")
        return False, None

    if not isinstance(h1_full, pd.DataFrame) or h1_full.empty or len(h1_full) < 60:
        dbg(f"[DETECT] {symbol}: H1 data insufficient.")
        return False, None

    if not isinstance(h4, pd.DataFrame) or h4.empty:
        h4 = h1.copy()

    h1 = h1_full.tail(160).copy()

    cur = h1.iloc[-1]
    hist = h1.iloc[:-1]
    if hist.empty:
        dbg(f"[DETECT] {symbol}: H1 hist empty after tail.")
        return False, None

    o = float(cur["open"])
    h = float(cur["high"])
    l = float(cur["low"])
    c = float(cur["close"])
    vol = float(cur["volume"])

    if o <= 0 or c <= 0:
        dbg(f"[DETECT] {symbol}: invalid OHLC (o={o}, c={c}).")
        return False, None

    # ===== 基本统计 =====
    pct_now = (c / o - 1.0) * 100.0
    vps_now = vol / BAR_SEC

    vps_hist = (hist["volume"].astype(float) / BAR_SEC).tail(60)
    vps_base = float(vps_hist.mean()) if len(vps_hist) else vps_now
    if vps_base <= 0:
        vps_base = vps_now or 1e-12

    volr_now = vps_now / max(vps_base, 1e-12)

    eq_now_bar_usd = c * vol
    eq_hist = (hist["close"].astype(float) * hist["volume"].astype(float)).tail(60)
    eq_base_bar_usd = float(eq_hist.mean()) if len(eq_hist) else eq_now_bar_usd

    # day 高低 & 24h 涨跌
    try:
        tick = get_tick_size(ex, symbol)
    except Exception:
        tick = 0.0001

    try:
        day_high, day_low, pct24, last_price = get_day_stats(ex, symbol, tick)
    except Exception:
        day_high = float(h1["high"].tail(24).max())
        day_low = float(h1["low"].tail(24).min())
        pct24 = pct_now
        last_price = c

    dh = float(day_high) if isinstance(day_high, (int, float)) else None
    dl = float(day_low) if isinstance(day_low, (int, float)) else None
    pct24_val = float(pct24) if isinstance(pct24, (int, float)) else None

    dist_day_high_pct = ((dh - c) / c * 100.0) if (dh and c > 0) else None
    dist_day_low_pct = ((c - dl) / c * 100.0) if (dl and c > 0) else None

    # ===== H1 指标（RSI / ADX / ATR）=====
    closes_h1 = hist["close"].astype(float).values
    highs_h1 = hist["high"].astype(float).values
    lows_h1 = hist["low"].astype(float).values

    if len(closes_h1) < 30:
        dbg(f"[DETECT] {symbol}: H1 len<30.")
        return False, None

    rsi_series = ta.RSI(closes_h1, timeperiod=14)
    rsi_val = rsi_series[-1]
    rsi = float(rsi_val) if not np.isnan(rsi_val) else None

    adx_series = ta.ADX(highs_h1, lows_h1, closes_h1, timeperiod=14)
    adx_val = adx_series[-1]
    adx = float(adx_val) if not np.isnan(adx_val) else 0.0

    atr_arr = ta.ATR(highs_h1, lows_h1, closes_h1, timeperiod=14)
    atr_val = atr_arr[-1]
    atr_abs = float(atr_val) if not np.isnan(atr_val) else c * 0.01

    # ===== H4 高周期方向（EMA20 / EMA50）=====
    closes_h4 = h4["close"].astype(float).values
    if len(closes_h4) >= 50:
        ema20_h4 = ta.EMA(closes_h4, timeperiod=20)[-1]
        ema50_h4 = ta.EMA(closes_h4, timeperiod=50)[-1]
    else:
        ema20_h4 = ema50_h4 = np.nan

    HTF_BULL = (
        isinstance(ema20_h4, (int, float))
        and isinstance(ema50_h4, (int, float))
        and not np.isnan(ema20_h4)
        and not np.isnan(ema50_h4)
        and ema20_h4 >= ema50_h4
        and c >= ema20_h4
    )
    HTF_BEAR = (
        isinstance(ema20_h4, (int, float))
        and isinstance(ema50_h4, (int, float))
        and not np.isnan(ema20_h4)
        and not np.isnan(ema50_h4)
        and ema20_h4 <= ema50_h4
        and c <= ema20_h4
    )
    htf_gate = "BULL" if HTF_BULL else ("BEAR" if HTF_BEAR else "")

    atr_pct = (atr_abs / c * 100.0) if c > 0 else 0.0
    dbg(
        f"[DETECT] {symbol}: pct_now={pct_now:.2f}%, volr_now={volr_now:.2f}x, "
        f"eq_bar_now=${eq_now_bar_usd:,.0f}, eq_bar_base=${eq_base_bar_usd:,.0f}"
    )
    rsi_str = f"{rsi:.2f}" if rsi is not None else "NA"
    adx_str = f"{adx:.1f}" if adx is not None else "NA"
    atrpct_str = f"{(atr_abs / c * 100):.2f}" if atr_abs is not None else "NA"

    dbg(
        f"[DETECT] {symbol}: RSI={rsi_str}, ADX={adx_str}, ATR%≈{atrpct_str}%, HTF={htf_gate}"
    )

    trend_text = (
        f"ADX≈{adx:.1f}, ATR≈{atr_pct:.2f}% | HTF:"
        f"{'BULL' if HTF_BULL else ('BEAR' if HTF_BEAR else '—')}"
    )

    # ===== 简单阻力/支撑（H4 上下 30 根极值）=====
    # ===== SR：基于 H4 的 swing + 聚类 + 直方密度 =====
    near_R = None
    near_S = None
    dist_R = None
    dist_S = None
    sr_levels_res = []
    sr_levels_sup = []
    sr_band_abs = None

    try:
        near_R_dict, near_S_dict, R_list, S_list, band_abs = compute_sr_levels_from_df(
            h1_full, c, topn=6
        )
        sr_band_abs = band_abs

        if near_R_dict is not None:
            near_R = float(near_R_dict["price"])
            if c > 0:
                dist_R = (near_R - c) / c * 100.0

        if near_S_dict is not None:
            near_S = float(near_S_dict["price"])
            if c > 0:
                dist_S = (c - near_S) / c * 100.0

        # 只保留价格列表给 payload，用于未来展示/调试
        sr_levels_res = [float(x["price"]) for x in R_list]
        sr_levels_sup = [float(x["price"]) for x in S_list]

    except Exception:
        # SR 失败不影响主逻辑，直接忽略
        near_R = near_R or None
        near_S = near_S or None
        dist_R = dist_R or None
        dist_S = dist_S or None
        sr_levels_res = []
        sr_levels_sup = []
        sr_band_abs = None

    # ===== 信号判定（扩展版）=====
    reasons = []

    # 1) 1h 区间高低
    HH_WIN = 20
    LL_WIN = 20
    if len(hist) >= max(HH_WIN, LL_WIN):
        hh = float(hist["high"].tail(HH_WIN).max())
        ll = float(hist["low"].tail(LL_WIN).min())
    else:
        hh = float(hist["high"].max())
        ll = float(hist["low"].min())

    BREAK_BUF = 0.002  # 0.2%

    # 2) 本根 K 的影线比例
    rng = max(h - l, 1e-12)
    lower_w = max(min(o, c) - l, 0.0) / rng  # 下影线占比
    upper_w = max(h - max(o, c), 0.0) / rng  # 上影线占比

    # 3) 基础四类信号：区间突破 + 极端长影线
    breakout_up = c >= hh * (1.0 + BREAK_BUF)
    breakout_down = c <= ll * (1.0 - BREAK_BUF)

    wick_bottom = (
        (lower_w >= 0.6)
        and (c > l + 0.3 * rng)
        and volr_now >= 2.0
        and (rsi is None or rsi <= 45)
    )

    wick_top = (
        (upper_w >= 0.6)
        and (c < h - 0.3 * rng)
        and volr_now >= 2.0
        and (rsi is None or rsi >= 55)
    )

    dbg(
        f"[DETECT] {symbol}: hh={hh:.6g}, ll={ll:.6g}, c={c:.6g}, "
        f"breakout_up={breakout_up}, breakout_down={breakout_down}, "
        f"lower_w={lower_w:.2f}, upper_w={upper_w:.2f}, "
        f"wick_bottom={wick_bottom}, wick_top={wick_top}"
    )

    # ===== 新信号 1：HTF 趋势回踩 + 1h 拒绝 =====
    near_ema20_h4 = (
        isinstance(ema20_h4, (int, float))
        and not np.isnan(ema20_h4)
        and c > 0
        and abs(c - ema20_h4) / c <= 0.01  # 距 4h EMA20 在 1% 内
    )

    trend_pullback_long = (
        HTF_BULL
        and near_ema20_h4
        and pct_now > 0
        and lower_w >= 0.3
        and volr_now >= 0.8
    )

    trend_pullback_short = (
        HTF_BEAR
        and near_ema20_h4
        and pct_now < 0
        and upper_w >= 0.3
        and volr_now >= 0.8
    )

    # ===== 新信号 2：突破后的回踩确认（breakout + retest）=====
    breakout_retest_long = False
    breakout_retest_short = False

    if len(hist) >= HH_WIN + 2:
        hist_ex_prev = hist.iloc[:-1]
        prev_bar = hist.iloc[-1]
        c_prev = float(prev_bar["close"])

        hh_prev = float(hist_ex_prev["high"].tail(HH_WIN).max())
        ll_prev = float(hist_ex_prev["low"].tail(LL_WIN).min())

        breakout_up_prev = c_prev >= hh_prev * (1.0 + BREAK_BUF)
        breakout_down_prev = c_prev <= ll_prev * (1.0 - BREAK_BUF)

        if breakout_up_prev and hh > 0:
            touch_support = abs(l - hh) / hh <= 0.003  # 0.3%
            if touch_support and c > hh:
                breakout_retest_long = True

        if breakout_down_prev and ll > 0:
            touch_resist = abs(h - ll) / ll <= 0.003
            if touch_resist and c < ll:
                breakout_retest_short = True

    # ===== 新信号 3：区间边缘拒绝（range_reject）=====
    range_reject_long = False
    range_reject_short = False
    range_top = hh if hh > near_R else near_R
    range_bottom = ll if ll < near_S else near_S
    range_width = range_top - range_bottom
    if range_width > 0:
        pos_in_range = (c - range_bottom) / range_width  # 0~1
        near_top = pos_in_range >= 0.8
        near_bottom = pos_in_range <= 0.2

        if near_top and upper_w >= 0.4 and not breakout_up:
            range_reject_short = True
        if near_bottom and lower_w >= 0.4 and not breakout_down:
            range_reject_long = True

    # ===== 新信号 4：H1 强双顶 / 双底（结构 + RSI 背离 + 颈线确认）=====
    double_top = False
    double_bottom = False

    try:
        prices = closes_h1
        highs_arr = highs_h1
        lows_arr = lows_h1
        n_hist = len(prices)

        if n_hist >= 30:
            # ---- 双顶 ----
            swing_high_idx = []
            for i in range(2, n_hist - 2):
                if swing_high(prices, i, 2, 2):
                    swing_high_idx.append(i)

            if len(swing_high_idx) >= 2:
                i1, i2 = swing_high_idx[-2], swing_high_idx[-1]
                p1, p2 = prices[i1], prices[i2]
                if abs(p2 - p1) / max(p1, 1e-12) <= 0.003 and 3 <= i2 - i1 <= 30:
                    neckline = float(np.min(lows_arr[i1 : i2 + 1]))
                    r1 = r2 = None
                    v1 = rsi_series[i1]
                    v2 = rsi_series[i2]
                    if not np.isnan(v1):
                        r1 = float(v1)
                    if not np.isnan(v2):
                        r2 = float(v2)
                    rsi_div = r1 is not None and r2 is not None and r2 <= r1
                    if rsi_div and neckline > 0 and c < neckline * 0.999:
                        double_top = True

            # ---- 双底 ----
            swing_low_idx = []
            for i in range(2, n_hist - 2):
                if swing_low(prices, i, 2, 2):
                    swing_low_idx.append(i)

            if len(swing_low_idx) >= 2:
                j1, j2 = swing_low_idx[-2], swing_low_idx[-1]
                p1, p2 = prices[j1], prices[j2]
                if abs(p2 - p1) / max(p1, 1e-12) <= 0.003 and 3 <= j2 - j1 <= 30:
                    neckline = float(np.max(highs_arr[j1 : j2 + 1]))
                    r1 = r2 = None
                    v1 = rsi_series[j1]
                    v2 = rsi_series[j2]
                    if not np.isnan(v1):
                        r1 = float(v1)
                    if not np.isnan(v2):
                        r2 = float(v2)
                    rsi_div = r1 is not None and r2 is not None and r2 >= r1
                    if neckline > 0 and rsi_div and c > neckline * 1.001:
                        double_bottom = True
    except Exception as e:
        dbg(f"[DETECT] {symbol}: double_top/bottom calc error: {e}")
        double_top = False
        double_bottom = False

    # ===== 新增：趋势衰竭 + V 型反转 =====
    kind_exh, lvl_exh = signal_exhaustion_reversal(
        closes_h1=closes_h1,
        rsi_series=rsi_series,
        atr_arr=atr_arr,
        pct_now=pct_now,
        volr_now=volr_now,
        o=o,
        h=h,
        l=l,
        c=c,
    )

    # ===== 汇总所有候选信号 =====
    candidates = []

    # 原基础 4 类
    if breakout_up:
        candidates.append(("breakout_up", "1h 突破区间上沿"))
    if breakout_down:
        candidates.append(("breakout_down", "1h 跌破区间下沿"))
    if wick_bottom and not breakout_up:
        candidates.append(("wick_bottom", "1h 长下影线反转"))
    if wick_top and not breakout_down:
        candidates.append(("wick_top", "1h 长上影线反转"))

    # 新增：HTF 趋势回踩
    if trend_pullback_long:
        candidates.append(("htf_trend_pullback_long", "4h 多头趋势回踩 + 1h 拒绝"))
    if trend_pullback_short:
        candidates.append(("htf_trend_pullback_short", "4h 空头趋势回踩 + 1h 拒绝"))

    # 新增：突破后的回踩确认
    if breakout_retest_long:
        candidates.append(("breakout_retest_long", "突破后回踩区间上沿得到支撑"))
    if breakout_retest_short:
        candidates.append(("breakout_retest_short", "跌破后回踩区间下沿遇阻下行"))

    # 新增：区间边缘拒绝
    if range_reject_long:
        candidates.append(("range_reject_long", "区间下沿附近长下影线承接"))
    if range_reject_short:
        candidates.append(("range_reject_short", "区间上沿附近长上影线受阻"))

    # 新增：强双顶 / 双底
    if double_top:
        candidates.append(("double_top", "H1 双顶确认（RSI 背离 + 跌破颈线）"))
    if double_bottom:
        candidates.append(("double_bottom", "H1 双底确认（RSI 背离 + 突破颈线）"))

    if kind_exh == "exhaustion_reversal_long":
        candidates.append(("exhaustion_reversal_long", "趋势衰竭 · V 型反转 · 多"))
    elif kind_exh == "exhaustion_reversal_short":
        candidates.append(("exhaustion_reversal_short", "趋势衰竭 · V 型反转 · 空"))

    if not candidates:
        dbg(f"[DETECT] {symbol}: no signal candidates, return False")
        return False, None

    # 按优先级挑一个（数值越小优先级越高）
    priority = {
        "double_top": 0,
        "double_bottom": 0,
        "breakout_retest_long": 1,
        "breakout_retest_short": 1,
        "breakout_up": 2,
        "breakout_down": 2,
        "htf_trend_pullback_long": 3,
        "htf_trend_pullback_short": 3,
        "exhaustion_reversal_long": 3,
        "exhaustion_reversal_short": 3,
        "wick_bottom": 4,
        "wick_top": 4,
        "range_reject_long": 5,
        "range_reject_short": 5,
    }
    candidates.sort(key=lambda x: priority.get(x[0], 99))
    kind, label = candidates[0]

    # 方向
    side = (
        "long"
        if kind
        in (
            "breakout_up",
            "wick_bottom",
            "htf_trend_pullback_long",
            "breakout_retest_long",
            "range_reject_long",
            "exhaustion_reversal_long",
            "double_bottom",
        )
        else "short"
    )

    dbg(
        f"[DETECT] {symbol}: choose kind={kind}, label={label}, "
        f"side={side}, HTF={'BULL' if HTF_BULL else ('BEAR' if HTF_BEAR else '—')}, "
        f"trend_align={(HTF_BULL and side == 'long') or (HTF_BEAR and side == 'short')}"
    )

    # ===== 入场 & 止损（简单 ATR+百分比）=====
    pct_sl_abs = c * (SAFE_SL_PCT / 100.0)
    atr_sl_abs = atr_abs * SAFE_ATR_MULT if atr_abs > 0 else 0.0
    slip = max(pct_sl_abs, atr_sl_abs)

    if side == "long":
        sl_price = max(1e-12, c - slip)
    else:
        sl_price = c + slip

    cmd_safe = (
        f"<code>{symbol} {('做多' if side == 'long' else '做空')}："
        f"市价进场，止损约 {SAFE_SL_PCT:.1f}% 或最近波段"
        f"{'低' if side == 'long' else '高'}位（≈{sl_price:.6g}）</code>"
    )

    # ===== 文案 reasons =====
    reasons.append(label)
    reasons.append(
        f"1h 涨跌 {pct_now:.2f}%  VolR≈{volr_now:.2f}x  Bar≈${eq_now_bar_usd:,.0f}"
    )
    if HTF_BULL:
        reasons.append("HTF=BULL（4h EMA20≥50，价格在 EMA 上方）")
    elif HTF_BEAR:
        reasons.append("HTF=BEAR（4h EMA20≤50，价格在 EMA 下方）")
    if near_R is not None and dist_R is not None:
        reasons.append(f"上方阻力≈{near_R:.6g}（ΔR≈{dist_R:.2f}%）")
    if near_S is not None and dist_S is not None:
        reasons.append(f"下方支撑≈{near_S:.6g}（ΔS≈{dist_S:.2f}%）")

    # ===== 组装 payload =====
    fallback_kind_cn = {
        "breakout_up": "区间突破 · 向上",
        "breakout_down": "区间突破 · 向下",
        "wick_bottom": "长下影线反转",
        "wick_top": "长上影线反转",
        "htf_trend_pullback_long": "趋势回踩 · 多",
        "htf_trend_pullback_short": "趋势回踩 · 空",
        "breakout_retest_long": "突破回踩确认 · 多",
        "breakout_retest_short": "突破回踩确认 · 空",
        "range_reject_long": "区间边缘拒绝 · 多",
        "range_reject_short": "区间边缘拒绝 · 空",
        "double_top": "双顶反转 · 空",
        "double_bottom": "双底反转 · 多",
        "exhaustion_reversal_long": "趋势衰竭反转 · 多",
        "exhaustion_reversal_short": "趋势衰竭反转 · 空",
    }
    kind_cn = KINDS_CN.get(kind, fallback_kind_cn.get(kind, kind))

    payload = {
        "symbol": symbol,
        "kind": kind,
        "kind_cn": kind_cn,
        "title": kind_cn,
        "timeframe_fast": TIMEFRAME_FAST,
        "pct_now": float(pct_now),
        "volr_now": float(volr_now),
        "vps_now": float(vps_now),
        "vps_base": float(vps_base),
        "eq_base_bar_usd": float(eq_base_bar_usd),
        "eq_now_bar_usd": float(eq_now_bar_usd),
        "htf_gate": htf_gate,
        "htf_bull": bool(HTF_BULL),
        "htf_bear": bool(HTF_BEAR),
        "trend_text": trend_text,
        "reasons": reasons,
        "trend_align": bool(
            (HTF_BULL and side == "long") or (HTF_BEAR and side == "short")
        ),
        "score": 1.0,  # 先统一给 1.0，如果以后要按信号类型区分强弱再调
        "day_high": dh,
        "day_low": dl,
        "pct24": pct24_val,
        "dist_day_high_pct": dist_day_high_pct,
        "dist_day_low_pct": dist_day_low_pct,
        "last_price": last_price or c,
        "cmd_safe": cmd_safe,
        # SR 信息（给 formatter 用）
        "sr_near_resistance": near_R,
        "sr_near_support": near_S,
        "sr_dist_to_resistance_pct": dist_R,
        "sr_dist_to_support_pct": dist_S,
        "sr_levels_resistance": sr_levels_res,
        "sr_levels_support": sr_levels_sup,
        "sr_band_abs": sr_band_abs,
    }

    dbg(
        f"[DETECT] {symbol}: DONE kind={kind}, side={side}, "
        f"HTF={'BULL' if HTF_BULL else ('BEAR' if HTF_BEAR else '—')}, "
        f"trend_align={payload['trend_align']}"
    )

    return True, payload
