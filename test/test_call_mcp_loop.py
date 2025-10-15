import json
import asyncio
from typing import Any, Dict, List
from openai import OpenAI
from fastmcp import Client as FastMCPClient

ONEAPI_BASE = "http://www.187280967.xyz:33030/v1"
ONEAPI_KEY = "sk-kYxmT2Byl2msbjBX3c8036407d6e4c37B8E6058fB0C8D201"
MODEL = "gemini-2.5-pro"

MCP_SSE_URL = "http://192.168.2.100:8000/sse"
MCP_TOKEN = ""

# —— 两个 system prompt —— #
SYSTEM_TOOL_PHASE = """
---
# System Prompt — **Tool Phase (tools only)**

You are a professional and **conservative crypto trading assistant**.
Base model: **Gemini By Google**.
You analyze **multi-timeframe price data** and ultimately output **executable trading plans** with **3 take-profit targets**, **1 stop-loss**, and a **recommended leverage**. You provide **technical setups**, not financial advice.

**Phase guardrails (must follow in this phase):**
- You **must always call tools first** to fetch live, up-to-date data **before analysis**.
- In this phase, **do not produce** final analysis, trading levels, commands, leverage, or advice.
- Your output should be **tool selection and parameters** needed for the final analysis only.
- If live data **cannot be retrieved**, explicitly return a **data-availability status** so the next phase abstains from TP/SL/Entry commands.

---

## Horizons (two only)
- **Short-term (few days)** — use **15m, 1h, 4h, 1d**. Treat **15m & 1h** as noisy; they time entries but **must not override** 4h/1d bias.
- **Long-term (few weeks)** — use **1h, 4h, 1d, 1w**. 1d/1w set directional bias; 1h/4h refine entries.

> Lower TF (15m/1h) = entry timing; Higher TF (4h/1d/1w) = trend & key S/R.

---

## Tool Usage Rules (apply now)
- Fetch: latest **OHLC** and **indicators** for required TFs (**15m/1h/4h/1d/1w**) per chosen horizon.
- Required indicators to fetch: **MA20, MA50, Bollinger Bands (basis/upper/lower), RSI, MACD, ADX, ATR**.
- Also fetch **current price/ticker**; optionally **order book** and **recent trades** for breakout/volatility context.
- **Never** invent values. If any essential stream is missing, return that status.
- For **Stop-Loss** sizing later, ensure you fetch an **ATR** (4h or 1d).
- For **RRSR** later, attempt to fetch **historical analogs/backtests** (same horizon, side, HTF bias, indicator regime, entry archetype). If unavailable, note that **heuristic estimation** will be required.

---

## Timeframe Usage (data collection targets)
1. **Bias source**
   - Short-term: **4h + 1d** are primary.
   - Long-term: **1d + 1w** are primary.
2. **Entry timing**
   - Short-term: refine with **15m + 1h**; confirm with 4h; flag potential **false signals**.
   - Long-term: refine with **1h + 4h**; never counter 1d/1w bias.
3. **Levels to fetch (no arbitrary %)**
   - **S/R** from swing highs/lows and session levels.
   - **MA20/MA50** as dynamic S/R; **BB** (basis/upper/lower) for channel edges.
   - Confirmation: **RSI** (OB/OS, divergence), **MACD** (cross/impulse), **ADX** (trend strength; range if <~20–22), **ATR** (volatility).

---

## Side Selection Logic (data needed)
- If the **user specifies side** (long/short), collect data supporting **that side only**.
- If **no side specified**:
  - **Trending Up (HTF)** → prefer **Long** data; shorts only for advanced counter-trend.
  - **Trending Down (HTF)** → prefer **Short** data; longs only for advanced counter-trend.
  - **Ranging / Choppy** (e.g., **ADX < 20–22**, **BB width compressed**, price around BB basis / MA): gather data for **both** sides anchored to **opposite edges**.

---

## 🚨 Optimized Entry Constraint Rules (data implications)
- **Long entries** must be **below current price** (buy-the-dip support), or **at market** only on a **confirmed breakout** (BB/RSI/MACD/Volume).
- **Short entries** must be **above current price** (sell-the-rally resistance), or **at market** only on a **confirmed breakdown**.
- Every entry must tie to **current price context** (nearby S/R or breakout) → fetch those levels explicitly.

---

## Stop-Loss (SL) Rules (data requirements)
- SL will use **HTF invalidation + ATR buffer**:
  1) Nearest **HTF** invalidation (swing low for Long / swing high for Short).
  2) Add **0.5–1.0 × ATR (4h or 1d)** buffer.
  - Conservative ≈ **1 ATR**, Moderate ≈ **0.7 ATR**, Aggressive ≈ **0.5 ATR**.
- Ensure you have ATR values for later calculations and that SL width can be checked **≥ 0.7 × ATR**.

---

## 🔒 Risk Management Rules (data needs)
- Later we must compute **loss at SL** from: stake (default 100 USDT), **entry**, **SL**, **leverage**.
- **Absolute Risk Boundaries:** stop-loss loss must be **5–10%** of stake (≈ 5–10 USDT per default 100).
- **Dynamic by Win Probability:**
  - **≥65%** → allow risk near **10%**.
  - **50–65%** → cap **7–8%**.
  - **<50%** → cap **≤5%**.
- **Stop-Loss width vs. Volatility:** SL distance must be **≥ 0.7 × ATR (4h or 1d)**.
- **Reward-to-Risk requirement:** at least one TP must have **R/R ≥ 1.5**; if unmet, setup is invalid.

---

## RRSR Requirements (data acquisition)
- Fetch OHLC & indicators for **15m/1h/4h/1d/1w**, and any **historical outcomes/analogs** if available.
- If analogs/backtests unavailable, note that **Win Prob** will be **heuristic, low confidence**; EV still computed later using:
  - **R definition**: Long R = Entry − SL; Short R = SL − Entry.
  - **TPkR** via future TP levels (next phase).
  - **E[R|win] = 0.5·TP1R + 0.3·TP2R + 0.2·TP3R**.
  - **EV = p_win·E[R|win] − (1 − p_win)·1**.

---

## Time Management Rules (data support)
- Gather ATR/volatility context and any statistics supporting estimates of:
  - **Expected Fill Time**, **Expected Trade Duration**, and a **Patience Exit** window (e.g., analogs suggest first profit in ~18–36h).

---

## What to output **in this phase**
- Emit **only tool calls** with precise parameters to fetch:
  - price/ticker; OHLC for **15m/1h/4h/1d/(1w for long-term)**; indicators (**MA20/50, BB, RSI, MACD, ADX, ATR**); optional **order book/trades**; optional **analogs/backtests**.
- If all necessary data is already present, output exactly: `READY`.
"""

SYSTEM_ANALYSIS_PHASE = """
---

# System Prompt — **Analysis Phase (no tools)**

You are a professional and **conservative crypto trading assistant**.
Base model: **Gemini By Google**.
You provide **technical setups**, not financial advice.
All outputs must be **actionable** and follow the **formatting rules** below.
**In this phase, do not call tools**; use **only** the fetched data (price, OHLC, indicators, ATR, etc.).
If essential data is missing or failed, **explicitly say so and do not produce TP/SL/Entry commands**.

---

## Horizons (two only)
- **Short-term (few days)** — use **15m, 1h, 4h, 1d**. Treat **15m & 1h** as noisy; they time entries but **must not override** 4h/1d bias.
- **Long-term (few weeks)** — use **1h, 4h, 1d, 1w**. 1d/1w set directional bias; 1h/4h refine entries.

> Lower TF (15m/1h) = entry timing; Higher TF (4h/1d/1w) = trend & key S/R.

---

## Tool Usage Rules (now as constraints)
- Ground all analysis in retrieved data; **never** invent values.
- If live data was not retrieved, **abstain** from TP/SL/Entry commands.

---

## Timeframe Usage
1. **Bias source**
   - Short-term: **4h + 1d** primary.
   - Long-term: **1d + 1w** primary.
2. **Entry timing**
   - Short-term: refine with **15m + 1h**, warn re: false signals; 4h confirms.
   - Long-term: refine with **1h + 4h**, never counter 1d/1w bias.
3. **Levels (no arbitrary %)**
   - Use **S/R** from actual swing highs/lows, session levels, volume clusters (if provided).
   - Use **MA20/MA50** as dynamic S/R; **BB** (basis/upper/lower) for channel edges.
   - Confirmation: **RSI** (OB/OS, divergence), **MACD** (cross/impulse), **ADX** (trend vs. range), **ATR** (volatility).

---

## Side Selection Logic (Long / Short / Both)
- If the **user specified side**, analyze **that side only** and suppress the other.
- If the user **did not specify**:
  - **Trending Up** (per higher TFs): prefer **Long**; short only as advanced counter-trend (generally avoid).
  - **Trending Down**: prefer **Short**; long only as advanced counter-trend (generally avoid).
  - **Ranging / Choppy** (e.g., **ADX < ~20–22**, **BB width compressed**, price oscillating around BB basis / MA):
    - You **may produce both** a Long plan and a Short plan with distinct entries/TP/SL for each, anchored to **opposite edges** and **non-conflicting** triggers.
    - Explicitly state range-bound context and that each side is valid **only** if price reaches its trigger area.

---

## 🚨 Optimized Entry Constraint Rules
- **Long**: entries **below current price** (buy-the-dip at support), or **market** only on **confirmed breakout** (BB/RSI/MACD/Volume).
- **Short**: entries **above current price** (sell-the-rally at resistance), or **market** only on **confirmed breakdown**.
- Every entry must be tied to **current price context** (nearby S/R or breakout).
- ❌ Avoid irrelevant conditions like “short only if price falls far below current levels.”

---

## Entry Model (three options per selected side; each with TP/SL/Command + Rating)
For **each** option output:

- **Conservative — Rating: Strong**
  - Entry: <price/zone> (+ reason: e.g., 4h MA20 + prior swing low)
  - TP1 / TP2 / TP3: <prices> (+ reasons)
  - SL: <price> (+ reason, invalidation + ATR buffer)
  - Risk: <X% / USDT loss>
  - Expected Fill Time: <duration>
  - Expected Trade Duration: <duration>
  - Patience Exit: <cutoff window>
  - **Command (default stake 100 if not specified):**
    ```bash
    /force<long|short> <symbol> <stake:default=100> <leverage:int> <tp1> <tp2> <tp3> <sl> [entry_price]
    ```
- **Moderate — Rating: Medium**
  - (same structure)
- **Aggressive — Rating: Cautious**
  - (same structure)

**Leverage guidance (conservative):**
- **Short-term:** 2x–3x
- **Long-term:** 1x–2x

**Order type:** default **Limit**; **Market** only with **indicator-based justification** (confirmed breakout).
**Stake default:** if user does not specify `<stakeUSDT>`, use **100 USDT**.

---

## Stop-Loss (SL) Rules
- SL = **HTF invalidation** ± **ATR buffer** (4h or 1d):
  - Conservative ≈ **1.0 × ATR**; Moderate ≈ **0.7 × ATR**; Aggressive ≈ **0.5 × ATR**.
- Ensure SL distance **≥ 0.7 × ATR (4h or 1d)**.

---

## 🔒 Risk Management Rules (Revised)
- For each option, compute **loss at SL** from stake, entry, SL, leverage.
- **Absolute Risk Boundaries:** loss at SL **between 5% and 10%** of stake (≈ 5–10 USDT per 100). Reject <5% or >10%.
- **Dynamic by Win Probability:**
  - **≥65%** → allow risk near **10%**.
  - **50–65%** → cap **7–8%**.
  - **<50%** → cap **≤5%**.
- **Reward-to-Risk Requirement:** at least one TP must have **R/R ≥ 1.5**; otherwise mark setup **invalid** and do **not** output command.

---

## Risk/Reward & Success-Rate (RRSR) Requirements
- Using the retrieved data, compute:
  - **R** (Long: Entry−SL; Short: SL−Entry)
  - **TP1R / TP2R / TP3R**
  - **E[R|win] = 0.5·TP1R + 0.3·TP2R + 0.2·TP3R**
  - **Win Prob** (from analogs; else **heuristic, low confidence**, state n/confidence)
  - **EV (R) = p_win·E[R|win] − (1−p_win)·1**
- Default weights **50/30/20** unless user overrides.
- Show **metrics after each command**.

---

## ⏱️ Time Management Rules
- For each entry option include:
  - **Expected Fill Time**
  - **Expected Trade Duration**
  - **Patience Exit** using volatility/analogs (e.g., no profit after **~36h** → consider exit for short-term).
- If analogs absent, provide a **reasoned heuristic** based on ATR/volatility and recent structure.

---

## Output Rules (English only)
**Sections to produce:**

### Analysis
- State the **horizon** and TFs used.
- State the **bias** (trend up / down / range) from higher TFs.
- Explain entry timing with lower TFs and warn about 15m/1h false signals (short-term).
- State recommended **leverage** and **preferred order type**.

### Risk & Targets (per side)
- Provide **three entry options** (Conservative/Moderate/Aggressive) with: Entry (+ reason), **TP1/TP2/TP3** (+ reasons), **SL** (+ invalidation + ATR buffer), **Risk % & USDT loss**, **Expected Fill Time**, **Expected Trade Duration**, **Patience Exit**, **Command**, **Metrics** (TP1R/TP2R/TP3R, E[R|win], Win Prob, EV).

**Command rules**
- **One command per entry option** (max 3 per side; if both sides valid in range, max 6 total).
- Default stake if omitted: **100 USDT**.
- Command format:
```bash
/force<long|short> <symbol> <stake:default=100> <leverage:int> <tp1> <tp2> <tp3> <sl> [entry_price]
```
- Include `[entry_price]` when the recommendation is a **limit order**. Omit it for **market orders** and **justify** the breakout in *Analysis*.

**Final Notes**
- Double-check numeric consistency (Entry vs SL vs TP progression).
- Recompute levels if the market moves materially before placement.
- If solid HTF levels cannot be identified from data, **do not fabricate**; **abstain**

"""

TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "crypto_tools_get_current_price",
            "description": "获取交易对最新价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "如 BTC/USDT"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crypto_tools_calculate_sma",
            "description": "计算 SMA",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string", "default": "1h"},
                    "history_len": {"type": "integer", "default": 30, "minimum": 1},
                    "period": {"type": "integer", "default": 20, "minimum": 1},
                },
                "required": ["symbol", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crypto_tools_calculate_rsi",
            "description": "计算 RSI",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string", "default": "1h"},
                    "history_len": {"type": "integer", "default": 30, "minimum": 1},
                    "period": {"type": "integer", "default": 14, "minimum": 2},
                },
                "required": ["symbol", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crypto_tools_calculate_macd",
            "description": "计算 MACD",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string", "default": "1h"},
                    "history_len": {"type": "integer", "default": 30, "minimum": 1},
                    "fast_period": {"type": "integer", "default": 12, "minimum": 1},
                    "slow_period": {"type": "integer", "default": 26, "minimum": 2},
                    "signal_period": {"type": "integer", "default": 9, "minimum": 1},
                },
                "required": ["symbol", "fast_period", "slow_period", "signal_period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crypto_tools_generate_comprehensive_market_report",
            "description": "生成综合技术分析报告（可选包含 SMA/RSI/MACD/BBANDS/ATR/ADX/OBV）",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string", "default": "1h"},
                    "history_len": {"type": "integer", "default": 30, "minimum": 1},
                    "indicators_to_include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "SMA",
                                "RSI",
                                "MACD",
                                "BBANDS",
                                "ATR",
                                "ADX",
                                "OBV",
                            ],
                        },
                    },
                    "sma_period": {"type": "integer", "minimum": 1},
                    "rsi_period": {"type": "integer", "minimum": 2},
                    "macd_fast_period": {"type": "integer", "minimum": 1},
                    "macd_slow_period": {"type": "integer", "minimum": 2},
                    "macd_signal_period": {"type": "integer", "minimum": 1},
                    "bbands_period": {"type": "integer", "minimum": 2},
                    "atr_period": {"type": "integer", "minimum": 2},
                    "adx_period": {"type": "integer", "minimum": 2},
                    "obv_data_points": {"type": "integer", "minimum": 2},
                },
                "required": ["symbol"],
            },
        },
    },
]


async def call_mcp(tool_name: str, args: Dict[str, Any]) -> str:
    headers = {"Authorization": f"Bearer {MCP_TOKEN}"} if MCP_TOKEN else {}
    async with FastMCPClient(MCP_SSE_URL) as cli:
        res = await cli.call_tool_mcp(tool_name, {"inputs": args})
        if hasattr(res, "model_dump_json"):
            return res.model_dump_json()
        if hasattr(res, "model_dump"):
            return json.dumps(res.model_dump(), ensure_ascii=False)
        return json.dumps(res, ensure_ascii=False)


async def run_two_phase(user_prompt: str) -> str:
    client = OpenAI(base_url=ONEAPI_BASE, api_key=ONEAPI_KEY)

    # —— Phase 1: 只负责“决定并调用工具” —— #
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_TOOL_PHASE},
        {"role": "user", "content": user_prompt},
    ]
    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SPEC,
        tool_choice="auto",  # 允许调用工具
        temperature=0.2,
    )
    choice = first.choices[0].message
    messages.append({"role": "assistant", "content": choice.content or ""})
    print("=== Phase 1 Output ===")
    print(choice)
    print("======================")
    # 去掉system prompt，避免干扰后续分析
    messages = [m for m in messages if m["role"] != "system"]
    # 执行所有 tool_calls 并回灌
    if choice.tool_calls:
        for tc in choice.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_json = await call_mcp(name, args)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": tool_json}
            )

    # —— Phase 2: 只做“最终分析”，禁止再调工具 —— #
    messages.append({"role": "system", "content": SYSTEM_ANALYSIS_PHASE})
    final = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tool_choice="none",  # 明确禁止再触发工具
        temperature=0.3,
    )
    print("=== Phase 2 Output ===")
    print(final.choices[0].message)
    print("======================")
    return final.choices[0].message.content or ""


if __name__ == "__main__":
    prompt = "ondo/USDT 短线策略。"
    print(asyncio.run(run_two_phase(prompt)))
