# -*- coding: utf-8 -*-
"""
formatter.py (融合版)
在你原有版式基础上，补充：
- HTF 闸门显示（htf_gate / htf_bull / htf_bear / fallback '—'）
- 互证/风险徽标（_confirm_tag / _risk_tag）
- 触发原因优先用 payload['reasons']，否则从 text_core 的 "Why: ..." 兜底
"""

from .config import (
    KINDS_CN,
    PRINT_FULL_REASONS,
    MAX_REASONS_IN_MSG,
    SEPARATOR_LINE,
    SAFE_MODE_ALWAYS,
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
    # 优先结构化字段
    gate = (p.get("htf_gate") or "").strip().upper()
    if gate in ("BULL", "BEAR"):
        return gate
    # 次选布尔
    if p.get("htf_bull"):
        return "BULL"
    if p.get("htf_bear"):
        return "BEAR"
    # 兜底：不从 trend_text 解析，保持简洁
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

    # 新增：HTF 闸门 & 徽标
    htf_gate = _fmt_gate(p)
    badges = _badges(p)
    title_line = f"<b>{symbol}</b>｜<b>{kind_cn}</b>｜HTF：<b>{htf_gate}</b>"
    if badges:
        title_line = f"{badges} {title_line}"

    # 新增：触发原因优先结构化
    reasons_out = _pull_reasons(p)
    if not reasons_out and p.get("reasons"):  # 容错
        reasons_out = p["reasons"]
        if not PRINT_FULL_REASONS:
            reasons_out = reasons_out[:MAX_REASONS_IN_MSG]
    why = "；".join(reasons_out) if reasons_out else "—"

    # 指令（与你原逻辑一致）
    cmd_lines = []
    if (not SAFE_MODE_ALWAYS) and p.get("cmd_immd"):
        cmd_lines.append(p["cmd_immd"])
    if p.get("cmd_safe"):
        cmd_lines.append(p["cmd_safe"])
    cmds = "\n".join([f"  · {x}" for x in cmd_lines]) if cmd_lines else "  · —"

    block = [
        SEPARATOR_LINE,
        title_line,
        f"现价{p['timeframe_fast']}涨跌：<b>{now_pct}</b>   量速倍数：<b>{volr}</b>   等效本bar成交额：<b>{eqbar_now}</b>",
        f"日内最高/最低：<b>{dh_str}</b> / <b>{dl_str}</b>   24h涨跌：<b>{pct24_str}</b>",
        f"距日高/距日低：<b>{dhi_str}</b> / <b>{dli_str}</b>",
        f"现价：<b>{_fmt_price(p.get('last_price') or p.get('price_now'))}</b>",
        f"成交速率(vps)：{vps_pair}   趋势：{trend_txt}",
        f"触发原因：{why}",
        "入场命令：",
        cmds,
    ]
    return "\n".join(block)
