# -*- coding: utf-8 -*-
"""
formatter.py (融合版 + SR参考·精简 Top3)
在你原有版式基础上，保持原字段与格式，SR区块改为：
- 仅展示“最近阻力/支撑”以及“最多3个阻力 / 3个支撑”的紧凑行
- 最近一档加粗并打 ⭐ 标
"""

from .config import (
    KINDS_CN,
    PRINT_FULL_REASONS,
    MAX_REASONS_IN_MSG,
    SEPARATOR_LINE,
)


def _fmt_price(x):
    if x is None:
        return "—"
    try:
        return f"{x:.6g}"
    except Exception:
        return str(x)


def _fmt_pct(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)


def _fmt_gate(p):
    gate = (p.get("htf_gate") or "").strip().upper()
    if gate in ("BULL", "BEAR"):
        return gate
    if p.get("htf_bull"):
        return "BULL"
    if p.get("htf_bear"):
        return "BEAR"
    return "—"


def _badges(p):
    tags = []
    if p.get("_confirm_tag"):
        tags.append(p["_confirm_tag"])
    if p.get("_risk_tag"):
        tags.append(p["_risk_tag"])
    return " ".join(tags)


def _pull_reasons(p):
    # 1) 优先结构化 reasons
    reasons = p.get("reasons")
    if isinstance(reasons, list) and reasons:
        out = [str(r).strip() for r in reasons if str(r).strip()]
        if not PRINT_FULL_REASONS:
            out = out[:MAX_REASONS_IN_MSG]
        return out
    # 2) 兜底：从 text_core 中抽取 "Why: " 行
    tc = p.get("text_core") or []
    for line in tc:
        if isinstance(line, str) and line.startswith("Why: "):
            content = line[5:].strip()
            if content:
                parts = [x.strip() for x in content.split(";") if x.strip()]
                if not PRINT_FULL_REASONS:
                    parts = parts[:MAX_REASONS_IN_MSG]
                return parts
    return []


# ===== SR（精简 Top3） =====
def _fmt_sr_compact(levels, current, nearest_price, is_resistance=True, topn=3):
    """
    levels: detect 注入的 sr_levels_resistance / sr_levels_support
    current: last_price
    nearest_price: sr_near_resistance / sr_near_support
    is_resistance: True=阻力；False=支撑
    输出示例：R: ⭐ 8.064(+0.35%), 8.157(+1.57%), 8.221(+2.37%)
    """
    if not isinstance(levels, list) or current is None or current == 0:
        prefix = "R" if is_resistance else "S"
        return f"{prefix}：—"

    rows = []
    # 只取前 topn 个（detect 里已经按“上方升序/下方降序”贴近现价排序过）
    for x in levels[:topn]:
        price = x.get("price")
        try:
            gap = (float(price) - float(current)) / float(current) * 100.0
            gap_str = f"{gap:+.2f}%"
        except Exception:
            gap_str = "—"
        # 最近位加 ⭐ 与加粗
        if nearest_price is not None and price == nearest_price:
            rows.append(f"⭐ <b>{_fmt_price(price)}({_fmt_pct(gap)})</b>")
        else:
            rows.append(f"{_fmt_price(price)}({_fmt_pct(gap)})")

    prefix = "R" if is_resistance else "S"
    return f"{prefix}：{', '.join(rows) if rows else '—'}"


def format_signal_cn(p):
    symbol = p["symbol"]
    kind = p["kind"]
    kind_cn = p.get("kind_cn") or KINDS_CN.get(kind, p.get("title", kind))

    now_pct = f"{p['pct_now']:.2f}%"
    volr = f"{p['volr_now']:.2f}x"
    eqbar_now = f"${p['eq_now_bar_usd']:,.0f}"
    vps_pair = f"{p['vps_now']:.4f} / {p['vps_base']:.4f}"
    trend_txt = p.get("trend_text", "")

    dh = p.get("day_high")
    dl = p.get("day_low")
    pct24 = p.get("pct24")
    dh_str = f"{dh:.6g}" if isinstance(dh, (int, float)) else "—"
    dl_str = f"{dl:.6g}" if isinstance(dl, (int, float)) else "—"
    pct24_str = f"{pct24:.2f}%" if isinstance(pct24, (int, float)) else "—"

    dhi = p.get("dist_day_high_pct")
    dli = p.get("dist_day_low_pct")
    dhi_str = f"{dhi:+.2f}%" if isinstance(dhi, (int, float)) else "—"
    dli_str = f"{dli:+.2f}%" if isinstance(dli, (int, float)) else "—"

    # HTF 闸门 & 徽标
    htf_gate = _fmt_gate(p)
    badges = _badges(p)
    title_line = f"<b>{symbol}</b>｜<b>{kind_cn}</b>｜HTF：<b>{htf_gate}</b>"
    if badges:
        title_line = f"{badges} {title_line}"

    # 触发原因（尽量短）
    reasons_out = _pull_reasons(p)
    if not reasons_out and p.get("reasons"):
        reasons_out = p["reasons"]
        if not PRINT_FULL_REASONS:
            reasons_out = reasons_out[:MAX_REASONS_IN_MSG]
    why = "；".join(reasons_out) if reasons_out else "—"

    # 指令
    cmd_lines = []
    # if (not SAFE_MODE_ALWAYS) and p.get("cmd_immd"):
    #     cmd_lines.append(p["cmd_immd"])
    if p.get("cmd_safe"):
        cmd_lines.append(p["cmd_safe"])
    cmds = "\n".join([f"  · {x}" for x in cmd_lines]) if cmd_lines else "  · —"

    # ===== SR（精简显示）=====
    last_px = p.get("last_price") or p.get("price_now")

    sr_nr = p.get("sr_near_resistance")
    sr_ns = p.get("sr_near_support")
    sr_dr = p.get("sr_dist_to_resistance_pct")
    sr_ds = p.get("sr_dist_to_support_pct")

    sr_R = p.get("sr_levels_resistance") or []
    sr_S = p.get("sr_levels_support") or []

    # 第一行：最近阻力/支撑（只给出数值和Δ）
    sr_near_line = (
        "SR参考："
        f"最近阻力 <b>{_fmt_price(sr_nr)}</b>（ΔR={_fmt_pct(sr_dr) if isinstance(sr_dr, (int, float)) else '—'}）｜"
        f"最近支撑 <b>{_fmt_price(sr_ns)}</b>（ΔS={_fmt_pct(sr_ds) if isinstance(sr_ds, (int, float)) else '—'}）"
    )

    # 第二行&第三行：Top3 紧凑行（最近位加 ⭐）
    sr_R_line = _fmt_sr_compact(sr_R, last_px, sr_nr, is_resistance=True, topn=3)
    sr_S_line = _fmt_sr_compact(sr_S, last_px, sr_ns, is_resistance=False, topn=3)

    block = [
        SEPARATOR_LINE,
        title_line,
        f"现价{p['timeframe_fast']}涨跌：<b>{now_pct}</b>   量速倍数：<b>{volr}</b>   等效本bar成交额：<b>{eqbar_now}</b>",
        f"日内最高/最低：<b>{dh_str}</b> / <b>{dl_str}</b>   24h涨跌：<b>{pct24_str}</b>",
        f"距日高/距日低：<b>{dhi_str}</b> / <b>{dli_str}</b>",
        f"现价：<b>{_fmt_price(last_px)}</b>",
        f"成交速率(vps)：{vps_pair}   趋势：{trend_txt}",
        f"触发原因：{why}",
        # —— SR（三行搞定）——
        sr_near_line,
        sr_R_line,
        sr_S_line,
        "入场命令：",
        cmds,
    ]
    return "\n".join(block)
