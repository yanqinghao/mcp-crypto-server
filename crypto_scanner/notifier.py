import time
import requests
from .config import (
    TG_BOT_TOKEN,
    TG_CHAT_ID,
    AUTO_DELETE_ENABLED,
    AUTO_DELETE_HOURS,
    AUTO_DELETE_GRACE,
    PER_MESSAGE_LIMIT,
    MESSAGE_DELAY_SEC,
    TITLE_PREFIX,
    SEPARATOR_LINE,
)
from .loggingx import dbg, ts_now
from .formatter import format_signal_cn

PENDING_DELETES = []


def telegram_send(text: str, parse_mode="HTML"):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        dbg("TG disabled -> print")
        print(text)
        return None
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={
                "chat_id": TG_CHAT_ID,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("ok") and isinstance(data.get("result"), dict):
            mid = data["result"].get("message_id")
            return (TG_CHAT_ID, mid)
        else:
            dbg(f"TG send non-ok: {data}")
            return None
    except Exception as e:
        print("[TG ERROR]", e)
        return None


def schedule_delete(chat_id, message_id, due_ts: int):
    if not AUTO_DELETE_ENABLED:
        return
    if not (chat_id and message_id):
        return
    PENDING_DELETES.append(
        {
            "chat_id": chat_id,
            "message_id": int(message_id),
            "due": int(due_ts),
            "tries": 0,
        }
    )


def tg_delete_message(chat_id, message_id) -> bool:
    if not TG_BOT_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/deleteMessage"
    try:
        r = requests.post(
            url, data={"chat_id": chat_id, "message_id": message_id}, timeout=10
        )
        if r.status_code == 200 and r.json().get("ok"):
            return True
        else:
            dbg(f"TG delete non-ok: {r.text}")
            return False
    except Exception as e:
        print("[TG DELETE ERROR]", e)
        return False


def cleanup_pending_deletes(now_ts: int):
    if not AUTO_DELETE_ENABLED or not PENDING_DELETES:
        return
    keep = []
    for item in PENDING_DELETES:
        if now_ts + AUTO_DELETE_GRACE < item["due"]:
            keep.append(item)
            continue
        ok = tg_delete_message(item["chat_id"], item["message_id"])
        if not ok:
            item["tries"] += 1
            if item["tries"] <= 3:
                item["due"] = now_ts + 60
                keep.append(item)
    PENDING_DELETES.clear()
    PENDING_DELETES.extend(keep)


def send_batches(collected):
    """
    collected: List[payload]
    分批中文推送 + 自动删除登记；返回所有设置了冷却的key
    """
    if not collected:
        return []

    # kind 优先级：把爆发放到最后，其它更有含义的在前
    priority = {
        "ema_rebound_long": 0,
        "ema_rebound_short": 0,
        "pullback_long": 1,
        "pullback_short": 1,
        "bb_squeeze_long": 2,
        "bb_squeeze_short": 2,
        "cap_long": 3,
        "cap_short": 3,
        "explode_up": 4,
        "explode_down": 4,
    }
    # 二阶段排序：先按分数，再按类别优先级
    collected.sort(key=lambda x: x["score"], reverse=True)
    collected.sort(key=lambda p: priority.get(p["kind"], 5))

    from .config import SEND_ALL_COLLECTED, MAX_ALERTS_PER_ROUND

    to_send = collected if SEND_ALL_COLLECTED else collected[:MAX_ALERTS_PER_ROUND]
    total_batches = (len(to_send) + PER_MESSAGE_LIMIT - 1) // PER_MESSAGE_LIMIT

    cooled_keys = []
    for b_idx in range(total_batches):
        group = to_send[b_idx * PER_MESSAGE_LIMIT : (b_idx + 1) * PER_MESSAGE_LIMIT]
        header = f"{TITLE_PREFIX}｜第 {b_idx + 1}/{total_batches} 批｜{ts_now()}"
        types_line = "，".join(
            f"{p['symbol']}·{p.get('kind_cn', p['kind'])}" for p in group
        )
        msg_lines = [header, f"本批：{types_line}"]
        for p in group:
            msg_lines.append(format_signal_cn(p))
        msg_lines.append(SEPARATOR_LINE)

        res = telegram_send("\n".join(msg_lines))
        if res:
            chat_id, msg_id = res
            schedule_delete(
                chat_id, msg_id, int(time.time()) + AUTO_DELETE_HOURS * 3600
            )

        for p in group:
            cooled_keys.append((p["symbol"], p["kind"]))

        if (b_idx + 1) < total_batches and MESSAGE_DELAY_SEC > 0:
            time.sleep(MESSAGE_DELAY_SEC)

    return cooled_keys
