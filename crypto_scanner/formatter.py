# -*- coding: utf-8 -*-
"""
formatter.py (下单指令版 + 风险提示增强)
通知格式类似：

TRUMP/USDT:USDT｜长下影反转｜方向：多
现价：7.119
日内最高/最低：7.767 / 6.903
阻力：7.18（0.74%）
支撑：7.103（0.34%）

市价建议：
  · /forcelong TRUMP/USDT:USDT 10 10

限价建议：
  · /forcelong TRUMP/USDT:USDT 10 10 7.103

止损参考：6.987

提示：
  · ⚠️ 高周期偏空，当前做多属于逆势，建议轻仓或放弃。
  · ✅ 止损距离当前价约 2.10%，风险区间较合理。
"""

from typing import Optional

from .config import (
    KINDS_CN,
    SAFE_MODE_ALWAYS,  # 目前只用于“是否一定展示 SL”，保留
)

# 你可以按自己的习惯改这两个默认值
DEFAULT_LEVERAGE = 10
DEFAULT_SIZE = 10


def _fmt_price(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def _fmt_pct(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)


def _infer_side(kind: str, side_hint: Optional[str] = None) -> str:
    """
    从 kind 推方向；payload 里如果有 side，就优先用。
    """
    if side_hint in ("long", "short"):
        return side_hint

    k = (kind or "").lower()

    long_kinds = {
        "breakout_up",
        "wick_bottom",
        "range_rebound_long",
        "eqb_rebound_long",
        "double_bottom",
        "htf_trend_pullback_long",
        "breakout_retest_long",
        "range_reject_long",
        "exhaustion_reversal_long",
    }
    short_kinds = {
        "breakout_down",
        "wick_top",
        "range_rebound_short",
        "eqb_rebound_short",
        "double_top",
        "htf_trend_pullback_short",
        "breakout_retest_short",
        "range_reject_short",
        "exhaustion_reversal_short",
    }

    if k in long_kinds:
        return "long"
    if k in short_kinds:
        return "short"

    # 默认当多处理
    return "long"


def _risk_hints(p: dict, side: str, last_price, sl_price) -> list[str]:
    """
    风险提示区块：
    - HTF 顺/逆势
    - 日内高低位置
    - 阻力/支撑空间
    - 止损距离是否合理
    """
    hints: list[str] = []

    gate = (p.get("htf_gate") or "").strip().upper()
    htf_bull = bool(p.get("htf_bull"))
    htf_bear = bool(p.get("htf_bear"))

    # ===== HTF 大趋势顺/逆势 =====
    if (gate == "BULL" or htf_bull) and side == "short":
        hints.append("⚠️ 高周期偏多，当前做空属于逆势，建议减仓或放弃。")
    if (gate == "BEAR" or htf_bear) and side == "long":
        hints.append("⚠️ 高周期偏空，当前做多属于逆势，建议轻仓。")

    # ===== 日内位置（接近高/低点）=====
    dist_day_high = p.get("dist_day_high_pct")
    dist_day_low = p.get("dist_day_low_pct")

    # 对多单：靠近日内高点 → 小心追多；靠近日内低点 → 性价比高
    if side == "long":
        if dist_day_high is not None and dist_day_high < 0.5:
            hints.append("⚠️ 价格接近日内高点，上行空间有限，追多需谨慎。")
        if (
            dist_day_low is not None
            and dist_day_low < 0.5
            and (dist_day_high is None or dist_day_high > 1.5)
        ):
            hints.append("📉 价格接近日内低位，性价比较好，可关注潜在反弹。")

    # 对空单：靠近日内低点 → 小心追空；靠近日内高点 → 做空性价比高
    if side == "short":
        if dist_day_low is not None and dist_day_low < 0.5:
            hints.append("⚠️ 价格接近日内低点，下行空间有限，追空需谨慎。")
        if (
            dist_day_high is not None
            and dist_day_high < 0.5
            and (dist_day_low is None or dist_day_low > 1.5)
        ):
            hints.append("📈 价格接近日内高点，做空性价比相对更好。")

    # ===== SR 空间提示 =====
    dist_R = p.get("sr_dist_to_resistance_pct")
    dist_S = p.get("sr_dist_to_support_pct")

    if side == "long" and dist_R is not None:
        try:
            d = float(dist_R)
            if d < 1.0:
                hints.append(f"⚠️ 上方最近阻力仅约 {_fmt_pct(d)}，目标空间较窄。")
            elif d > 3.0:
                hints.append(f"📈 上方阻力尚有约 {_fmt_pct(d)} 空间，可关注。")
        except Exception:
            pass

    if side == "short" and dist_S is not None:
        try:
            d = float(dist_S)
            if d < 1.0:
                hints.append(f"⚠️ 下方最近支撑仅约 {_fmt_pct(d)}，下跌空间有限。")
            elif d > 3.0:
                hints.append(f"📉 距离下方主要支撑尚有约 {_fmt_pct(d)} 空间。")
        except Exception:
            pass

    # ===== 止损距离合理性 =====
    loss_pct = None
    try:
        lp = float(last_price) if last_price is not None else None
        sp = float(sl_price) if sl_price is not None else None
        if lp and sp:
            if side == "long" and sp < lp:
                loss_pct = (lp - sp) / lp * 100.0
            elif side == "short" and sp > lp:
                loss_pct = (sp - lp) / lp * 100.0
    except Exception:
        loss_pct = None

    if loss_pct is not None:
        if loss_pct < 0.5:
            hints.append(
                "⚠️ 止损距离非常近，容易被来回扫，考虑适当放宽或寻找更好结构点。"
            )
        elif loss_pct > 4.0:
            hints.append("⚠️ 止损距离较远，注意控制仓位，避免单笔风险过大。")
        else:
            hints.append(
                f"✅ 止损距离当前价约 {_fmt_pct(loss_pct)}，风险区间相对合理。"
            )

    return hints


def format_signal_cn(p: dict) -> str:
    """
    detect_signal → 文本格式（无 HTML）
    主体结构：
    1) 头部 + 价格 + SR
    2) 市价/限价建议
    3) 止损参考
    4) 提示（风险&位置&SL 合理性 + 结构 SR 列表）
    """
    symbol = p.get("symbol", "?")
    kind = str(p.get("kind", "") or "")
    kind_cn = p.get("kind_cn") or KINDS_CN.get(kind, kind) or kind

    # === 方向 ===
    side = _infer_side(kind, p.get("side"))
    side_cn = "多" if side == "long" else "空"
    side_emoji = "📈" if side == "long" else "📉"

    # === 价格数据 ===
    last_price = p.get("last_price") or p.get("close") or p.get("c")
    day_high = p.get("day_high")
    day_low = p.get("day_low")

    near_R = p.get("sr_near_resistance")
    near_S = p.get("sr_near_support")
    dist_R = p.get("sr_dist_to_resistance_pct")
    dist_S = p.get("sr_dist_to_support_pct")

    sl_price = p.get("sl_price")

    # 结构 SR 列表（来自 detect_signal payload）
    sr_res_list = p.get("sr_levels_resistance") or []
    sr_sup_list = p.get("sr_levels_support") or []

    # ===== 头部 =====
    # 例：📈 TRUMP/USDT:USDT｜长下影反转｜方向：多
    lines: list[str] = [
        f"{side_emoji} {symbol}｜{kind_cn}｜方向：{side_cn}",
    ]

    if last_price is not None:
        lines.append(f"现价：{_fmt_price(last_price)}")

    if day_high is not None or day_low is not None:
        lines.append(f"日内最高/最低：{_fmt_price(day_high)} / {_fmt_price(day_low)}")

    # SR 信息（最近一上/一下）
    if near_R is not None and dist_R is not None:
        lines.append(f"阻力：{_fmt_price(near_R)}（{_fmt_pct(dist_R)}）")
    if near_S is not None and dist_S is not None:
        lines.append(f"支撑：{_fmt_price(near_S)}（{_fmt_pct(dist_S)}）")

    lines.append("")

    # ===== 市价建议 =====
    if side == "long":
        cmd_mkt = f"<code>/forcelong {symbol} {DEFAULT_LEVERAGE} {DEFAULT_SIZE}</code>"
    else:
        cmd_mkt = f"<code>/forceshort {symbol} {DEFAULT_LEVERAGE} {DEFAULT_SIZE}</code>"

    lines.append("市价建议：")
    lines.append(f"  · {cmd_mkt}")
    lines.append("")

    # ===== 限价建议 =====
    if side == "long":
        entry_price = near_S or last_price
        cmd_lmt = f"<code>/forcelong {symbol} {DEFAULT_LEVERAGE} {DEFAULT_SIZE} {_fmt_price(entry_price)}</code>"
    else:
        entry_price = near_R or last_price
        cmd_lmt = f"<code>/forceshort {symbol} {DEFAULT_LEVERAGE} {DEFAULT_SIZE} {_fmt_price(entry_price)}</code>"

    lines.append("限价建议：")
    lines.append(f"  · {cmd_lmt}")

    # ===== 止损参考（受 SAFE_MODE_ALWAYS 控制）=====
    if sl_price is not None and SAFE_MODE_ALWAYS:
        lines.append("")
        lines.append(f"止损参考：{_fmt_price(sl_price)}")

    # ===== 风险提示 & 结构 SR =====
    hints = _risk_hints(p, side, last_price, sl_price)

    # 只要有风险提示 or 有 SR 列表，就展示“提示：”区块
    if hints or sr_res_list or sr_sup_list:
        lines.append("")
        lines.append("提示：")

        # 先展示结构 SR（最多 3 个阻力 + 3 个支撑）
        if (sr_res_list or sr_sup_list) and last_price is not None:
            lines.append("  · 结构 SR：")

            # 上方阻力
            for i, price in enumerate(sr_res_list[:3], 1):
                try:
                    lp = float(last_price)
                    p_val = float(price)
                    gap = (p_val - lp) / lp * 100.0
                    lines.append(f"    R{i}: {_fmt_price(p_val)}（{_fmt_pct(gap)}）")
                except Exception:
                    lines.append(f"    R{i}: {_fmt_price(price)}")

            # 下方支撑
            for i, price in enumerate(sr_sup_list[:3], 1):
                try:
                    lp = float(last_price)
                    p_val = float(price)
                    gap = (p_val - lp) / lp * 100.0
                    lines.append(f"    S{i}: {_fmt_price(p_val)}（{_fmt_pct(gap)}）")
                except Exception:
                    lines.append(f"    S{i}: {_fmt_price(price)}")

        # 再展示风险提示
        for h in hints:
            lines.append(f"  · {h}")

    return "\n".join(lines)
