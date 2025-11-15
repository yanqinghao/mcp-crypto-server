# -*- coding: utf-8 -*-
"""
loop.py
- 每 15 分钟扫描一次候选列表，但信号只基于 H1+H4（策略 H1_4H）
- 使用 detect_signal + format_signal_cn
- 发送内容为“纯信号正文”，批量拼接，无标题 / 无批次标记 / 无横线装饰
"""

import time
from typing import Dict, List, Tuple

from .loggingx import dbg
from .config import (
    SLEEP_MS,
    POLL_SEC,
    PER_MESSAGE_LIMIT,
    MESSAGE_DELAY_SEC,
    ONLY_PUSH_EXPLODE,
    MIN_SIGNAL_SCORE,
    EXPLODE_QUIET_EXTRA_SCORE,
    MODE,
    CANDIDATE_REFRESH_SEC,
    FRAME_SEC,
    AUTO_DELETE_HOURS,
    AUTO_DELETE_GRACE,
)
from .strategies.h1_with_4h import H1_4H
from .exchange import build_exchange
from .candidates import hourly_refresh_candidates
from .detect_pro import detect_signal
from .notifier import telegram_send, schedule_delete, cleanup_pending_deletes
from .formatter import format_signal_cn

TELEGRAM_MAX_CHARS = 3900  # 安全一点，略低于 TG 实际上限


# ===== 发送封装 =====
def _send_text_with_delete(text: str):
    """发送并自动登记删除时间"""
    if len(text) > TELEGRAM_MAX_CHARS:
        text = text[: TELEGRAM_MAX_CHARS - 20] + "\n... (truncated)"
    res = telegram_send(text)
    if res:
        chat_id, msg_id = res
        schedule_delete(
            chat_id,
            msg_id,
            int(time.time()) + AUTO_DELETE_HOURS * 3600 + AUTO_DELETE_GRACE,
        )


def _send_batches(payloads: List[dict]):
    """
    按 Telegram 字数限制 + PER_MESSAGE_LIMIT 做分批发送
    - 不加任何标题/批次/横线
    - 每条消息里面是若干个 format_signal_cn(p)，之间用空行分隔
    """
    if not payloads:
        return

    batches: List[str] = []
    cur_parts: List[str] = []
    cur_len = 0
    cur_count = 0

    for p in payloads:
        text = format_signal_cn(p)  # 只要信号正文
        if not text:
            continue

        sep = "\n\n" if cur_parts else ""
        add_len = len(sep) + len(text)

        # 当前批非空，且加入后超过字符上限或数量上限 → 先封一批
        if cur_parts and (
            cur_len + add_len > TELEGRAM_MAX_CHARS or cur_count >= PER_MESSAGE_LIMIT
        ):
            batches.append("\n\n".join(cur_parts))
            cur_parts = []
            cur_len = 0
            cur_count = 0
            sep = ""
            add_len = len(text)

        cur_parts.append(text)
        cur_len += add_len
        cur_count += 1

    if cur_parts:
        batches.append("\n\n".join(cur_parts))

    for msg in batches:
        _send_text_with_delete(msg)
        time.sleep(MESSAGE_DELAY_SEC)


# ===== 扫描一次（用 H1_4H 策略）=====
def _scan_once(
    ex,
    candidates: List[str],
    strong_up_map: Dict[str, bool],
    strong_dn_map: Dict[str, bool],
    last_alert_at: Dict[Tuple[str, str], float],
):
    if not candidates:
        dbg("[LOOP] No candidates; skip scan.")
        return

    from .config import ALERT_COOLDOWN_SEC as GLOBAL_CD

    payloads: List[dict] = []
    dbg(f"[LOOP] Scanning {len(candidates)} symbols with H1_4H")

    for sym in candidates:
        try:
            ok, payload = detect_signal(ex, sym, strong_up_map, strong_dn_map, H1_4H)
            ex.sleep(SLEEP_MS)
            if not ok or not payload:
                dbg(f"[FILTER] {sym}: detect returned empty or not ok.")
                continue

            kind = str(payload.get("kind", ""))
            score_now = float(payload.get("score", 0.0))
            key = (sym, kind)

            # ====== cooldown 逻辑 ======
            cd_override = (
                (H1_4H.overrides or {}).get("ALERT_COOLDOWN_SEC")
                if getattr(H1_4H, "overrides", None)
                else None
            )

            cooldown = (
                cd_override
                if isinstance(cd_override, (int, float)) and cd_override > 0
                else GLOBAL_CD
            )

            last_ts = last_alert_at.get(key, 0.0)
            now_ts = time.time()

            if now_ts - last_ts < cooldown:
                dbg(
                    f"[FILTER] {sym}/{kind}: cooldown "
                    f"passed={now_ts - last_ts:.1f}s < CD={cooldown}s"
                )
                continue

            # ====== MIN_SIGNAL_SCORE 过滤 ======
            if score_now < MIN_SIGNAL_SCORE:
                dbg(
                    f"[FILTER] {sym}/{kind}: score {score_now:.2f} < MIN_SIGNAL_SCORE={MIN_SIGNAL_SCORE}"
                )
                continue

            # ====== QUIET 模式：explode 额外要求 ======
            if MODE == "QUIET":
                if "explode" in kind and score_now < (
                    MIN_SIGNAL_SCORE + EXPLODE_QUIET_EXTRA_SCORE
                ):
                    dbg(
                        f"[FILTER] {sym}/{kind}: QUIET explode score {score_now:.2f} "
                        f"< required {MIN_SIGNAL_SCORE + EXPLODE_QUIET_EXTRA_SCORE}"
                    )
                    continue
            else:
                # ===== ONLY_PUSH_EXPLODE ======
                if ONLY_PUSH_EXPLODE and ("explode" not in kind):
                    dbg(
                        f"[FILTER] {sym}/{kind}: ONLY_PUSH_EXPLODE enabled, kind not explode → drop"
                    )
                    continue

            # ====== 通过所有过滤 ======
            dbg(
                f"[PASS] {sym}/{kind}: score={score_now:.2f}, strong_up={strong_up_map.get(sym)}, strong_dn={strong_dn_map.get(sym)}"
            )
            payload["_score_routed"] = score_now
            payloads.append(payload)

        except Exception as e:
            import traceback

            traceback.print_exc()
            dbg(f"[ERROR] detect {sym}: {e}")

    if not payloads:
        dbg("[LOOP] No signals after filter.")
        return

    # ====== 发送消息 ======
    _send_batches(payloads)
    now_mark = time.time()
    for p in payloads:
        sym = p.get("symbol")
        kind = p.get("kind")
        if sym and kind:
            last_alert_at[(sym, kind)] = now_mark


# ===== 主循环：每 15m 扫一次 H1/H4 =====
def run_loop():
    ex = build_exchange()
    dbg("[LOOP] Exchange built. Starting main loop...")

    # 初始候选列表（按 H1_4H 的选币逻辑，每 CANDIDATE_REFRESH_SEC 刷新一次）
    candidates, strong_up_map, strong_dn_map = hourly_refresh_candidates(ex, H1_4H)
    dbg(
        f"[LOOP] Initial candidates loaded: {len(candidates)} symbols "
        f"(refresh every {CANDIDATE_REFRESH_SEC // 60} min)"
    )

    last_alert_at: Dict[Tuple[str, str], float] = {}
    last_candidates_refresh_ts = int(time.time())

    last_15m_slot = None
    first_run = True

    while True:
        loop_start = time.time()
        try:
            now_ts = int(time.time())

            # ---- 候选列表周期刷新（默认按小时）----
            if now_ts - last_candidates_refresh_ts >= CANDIDATE_REFRESH_SEC:
                candidates, strong_up_map, strong_dn_map = hourly_refresh_candidates(
                    ex, H1_4H
                )
                dbg(
                    f"[LOOP] candidates refreshed: {len(candidates)} symbols "
                    f"({CANDIDATE_REFRESH_SEC // 60} min interval)"
                )
                last_candidates_refresh_ts = now_ts

            # ---- 15m 槽位判断 ----
            slot_15m = now_ts // FRAME_SEC["15m"]

            # 首次启动 / 槽位变更 → 执行一次扫描（用 H1/H4）
            do_scan = False
            if first_run:
                do_scan = True
            elif slot_15m != last_15m_slot:
                do_scan = True

            if do_scan:
                dbg(
                    f"[LOOP] 15m slot={slot_15m} scan start, candidates={len(candidates)}"
                )
                _scan_once(ex, candidates, strong_up_map, strong_dn_map, last_alert_at)
                last_15m_slot = slot_15m
                first_run = False

            # 清理待删除消息
            cleanup_pending_deletes(int(time.time()))

        except Exception as e:
            import traceback

            traceback.print_exc()
            dbg(f"[LOOP ERROR] {e}")

        elapsed = time.time() - loop_start
        dbg(f"[LOOP] tick elapsed={elapsed:.2f}s")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    run_loop()
