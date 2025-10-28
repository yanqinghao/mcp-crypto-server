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
        # 尽量少小数，保留你原格式风格即可
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

    reasons_out = p.get("reasons", [])
    if not PRINT_FULL_REASONS and reasons_out:
        reasons_out = reasons_out[:MAX_REASONS_IN_MSG]
    why = "；".join(reasons_out) if reasons_out else "—"

    cmd_lines = []
    if (not SAFE_MODE_ALWAYS) and p.get("cmd_immd"):
        cmd_lines.append(p["cmd_immd"])
    if p.get("cmd_safe"):
        cmd_lines.append(p["cmd_safe"])
    cmds = "\n".join([f"  · {x}" for x in cmd_lines]) if cmd_lines else "  · —"

    block = [
        SEPARATOR_LINE,
        f"<b>{symbol}</b>｜<b>{kind_cn}</b>",
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
