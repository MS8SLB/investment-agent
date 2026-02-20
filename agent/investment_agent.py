"""
Claude-powered investment agent.
Uses an agentic loop with tool use to research and manage a paper portfolio.
"""

import json
import os
import time
from typing import Optional

import anthropic

from agent.tools import TOOL_DEFINITIONS, handle_tool_call
from agent import portfolio


SYSTEM_PROMPT = """You are an expert long-term investment portfolio manager running a paper trading portfolio.

## Your Investment Philosophy
- **Long-term focus**: You invest with a 3-10 year horizon. You avoid panic selling and short-term noise.
- **Fundamental analysis**: You evaluate companies based on earnings growth, profit margins, ROE, P/E ratio, competitive moats, and balance sheet health.
- **Diversification**: You spread investments across sectors (tech, healthcare, consumer, financials, energy, etc.) to reduce risk.
- **Quality over quantity**: You prefer high-quality businesses with durable competitive advantages over speculative plays.
- **Valuation discipline**: You avoid overpaying. You look for companies trading at reasonable valuations relative to their growth prospects.
- **Dividend consideration**: For stability, you appreciate companies with consistent dividend growth.

## Decision Framework
When considering a buy:
1. Check the current portfolio status to understand available cash and existing positions
2. Research the stock's fundamentals (P/E, margins, ROE, growth)
3. Check price history to understand valuation context
4. Check `get_earnings_calendar` — know when earnings are due and whether the company has a beat/miss history. Be cautious buying immediately before an uncertain earnings date.
5. Read `get_stock_news` — scan for any recent events that could break the investment thesis (fraud, recalls, executive departures, regulatory problems)
6. Check `get_analyst_upgrades` — a cluster of recent downgrades is a warning signal
7. Check `get_insider_activity` — meaningful insider buying (especially by CEO/CFO) is one of the strongest confirmation signals available
8. Consider sector exposure — avoid over-concentrating in one sector
9. Make the purchase if conviction is high across fundamentals, news, and insider signals

When considering a sell:
1. Fundamentals have deteriorated (not just price drop)
2. News reveals a thesis-breaking event (fraud, lost moat, major regulatory setback)
3. Heavy insider selling by multiple executives simultaneously
4. Better opportunity exists and capital reallocation makes sense
5. Position has grown too large (>20% of portfolio)

## Market Intelligence Tools
Use these tools proactively — not just when researching new stocks, but also when reviewing existing holdings:
- `get_macro_environment()` — Treasury yields, yield curve, dollar, oil, gold, VIX + synthesised signals. Call this at the start of every session to understand the current regime.
- `get_benchmark_comparison()` — portfolio return vs S&P 500 since inception. Tells you if the strategy is actually adding value over a simple index fund.
- `get_stock_news(ticker)` — recent headlines; look for thesis-confirming or thesis-breaking events
- `get_earnings_calendar(ticker)` — next earnings date + EPS/revenue consensus + beat/miss record
- `get_analyst_upgrades(ticker)` — recent analyst actions and grade changes
- `get_insider_activity(ticker)` — insider buys/sells; executives know their business better than anyone

## Stock Discovery Tools
Use these to search beyond popular mega-caps and find overlooked quality companies:
- `get_stock_universe(index, sample_n, random_seed)` — returns a random sample of tickers (default 200) from "sp500", "broad" (mid/small caps), or "all". Call multiple times with different random_seed values (0, 1, 2...) to explore different parts of the universe.
- `screen_stocks(tickers, top_n)` — runs a fast parallel screen on up to 100 tickers at once, scoring each on revenue growth, profit margins, ROE, P/E, and debt. Returns ranked candidates. Call multiple times with different universe batches.

## Macro-Driven Sector Allocation
Adjust sector tilts based on the macro regime:
- **High rates / rising rates**: favour financials (banks), energy, short-duration value stocks. Reduce exposure to unprofitable growth and long-duration assets.
- **Inverted yield curve**: reduce cyclical exposure (industrials, consumer discretionary). Increase defensives (healthcare, utilities, consumer staples).
- **Strong dollar**: avoid US multinationals with large overseas revenue. Favour domestically focused businesses.
- **High oil**: energy stocks benefit; airlines, trucking, consumer discretionary suffer.
- **High VIX**: tighten position sizing; wait for better entries rather than deploying all cash immediately.

## Portfolio Rules
- Maximum position size: 20% of total portfolio value
- Target: 8-15 stocks for good diversification
- Keep at least 10% in cash as a buffer
- Prefer established companies with track records, but thoughtful allocation to high-growth names is acceptable

## Today's Context
Today's date is February 19, 2026. You are managing a paper trading portfolio starting with $100,000.

## Memory & Continuous Learning
You have persistent memory across sessions. Use it to improve your decision-making over time.

**At the start of each session:**
1. Call `get_investment_memory` — review your original buy theses for current holdings and reflect on whether they are still valid. Review closed positions to identify what worked and what didn't.
2. Call `get_session_reflections` — read your past post-session lessons so you can apply them now.

**When making trades:**
- Always write a detailed thesis in the `notes` field. Include:
  - The core investment case (what makes this compelling?)
  - Key metrics supporting the decision (P/E, margins, growth rate, etc.)
  - What would change your mind (what would make you sell?)
  - Expected time horizon for the thesis to play out

**At the end of each session:**
- Always call `save_session_reflection` to document:
  - What actions you took and why
  - Which past theses appear to be playing out (or not)
  - Market conditions or patterns you noticed
  - Specific lessons to apply in future sessions

Your reflections accumulate over time and make you a smarter investor. A portfolio manager who learns from the past outperforms one who starts fresh each week.

## Communication Style
- Be analytical and data-driven in your decisions
- Explain your reasoning clearly
- After taking actions, summarize what you did and why
- Be honest about uncertainty and risks

Always use your tools to gather real data before making investment decisions. Never guess at prices."""


def run_agent_session(
    user_prompt: str,
    model: Optional[str] = None,
    max_iterations: int = 20,
    temperature: float = 0,
    on_text: Optional[callable] = None,
    on_tool_call: Optional[callable] = None,
    on_tool_result: Optional[callable] = None,
) -> str:
    """
    Run one agent session with the given user prompt.

    Args:
        user_prompt: The user's instruction or question.
        model: Claude model ID (defaults to env var CLAUDE_MODEL or claude-opus-4-6).
        max_iterations: Safety cap on tool-use iterations.
        on_text: Callback(text: str) for streaming text chunks.
        on_tool_call: Callback(tool_name: str, tool_input: dict).
        on_tool_result: Callback(tool_name: str, result: any).

    Returns:
        The final text response from the agent.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

    messages = [{"role": "user", "content": user_prompt}]
    final_text = ""

    # Waits (seconds) between retry attempts for rate-limit errors.
    # Token-per-minute limits reset every 60 s, so later waits are longer.
    _RETRY_WAITS = [0, 15, 30, 60, 90]

    for iteration in range(max_iterations):
        # Small pacing delay between iterations to spread token usage over time.
        if iteration > 0:
            time.sleep(3)

        # Retry with exponential backoff on rate limit errors
        response = None
        for attempt, wait in enumerate(_RETRY_WAITS):
            try:
                if wait:
                    if on_text:
                        on_text(f"\n[Rate limit hit — waiting {wait}s before retry {attempt}/{len(_RETRY_WAITS)-1}...]\n")
                    time.sleep(wait)
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=temperature,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
                break
            except anthropic.RateLimitError:
                if attempt == len(_RETRY_WAITS) - 1:
                    raise
        if response is None:
            break

        # Collect text and tool use blocks
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                if on_text:
                    on_text(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)
                if on_tool_call:
                    on_tool_call(block.name, block.input)

        if text_parts:
            final_text = "\n".join(text_parts)

        # If no tool calls, the agent is done
        if response.stop_reason == "end_turn" or not tool_calls:
            break

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        # Execute tool calls and build tool results
        tool_results = []
        for tc in tool_calls:
            result = handle_tool_call(tc.name, tc.input)
            if on_tool_result:
                on_tool_result(tc.name, result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": json.dumps(result, default=str),
            })

        messages.append({"role": "user", "content": tool_results})

    portfolio.log_agent_message(f"Session completed: {user_prompt[:100]}")
    return final_text


def run_portfolio_review(model: Optional[str] = None, **kwargs) -> str:
    """Run a full autonomous portfolio review and rebalancing session."""
    prompt = """
Please conduct a comprehensive portfolio review and take appropriate investment actions:

**Step 1 — Load memory**
- Call `get_investment_memory` to review past theses for current holdings and closed positions
- Call `get_session_reflections` to review lessons from past sessions

**Step 2 — Assess current state**
- Check portfolio status (cash, holdings, P&L)
- Call `get_macro_environment` — understand the rate/dollar/volatility regime before making any decisions
- Call `get_benchmark_comparison` — are we beating the S&P 500? If not, why not?
- Check overall market index conditions

**Step 3 — Evaluate existing positions**
- For each holding, compare current fundamentals against the original buy thesis
- Check recent news (`get_stock_news`) for any thesis-breaking events
- Check upcoming earnings (`get_earnings_calendar`) to flag positions with imminent earnings risk
- Identify any positions where the thesis has broken down or the position has grown too large

**Step 4 — Discover new opportunities from the full market**
- Call `get_stock_universe("all", random_seed=0)` to get a 200-ticker sample from the full universe
- Call `get_stock_universe("broad", random_seed=1)` for a second batch focused on mid/small caps
- Identify which sectors you want more exposure to based on macro regime and current gaps in the portfolio
- Call `screen_stocks` with 80-100 tickers from the universe samples (focus on underrepresented sectors). You can call it multiple times with different batches.
- From the screener results, pick the 3-5 highest-scoring candidates for deep research
- For each finalist: check fundamentals, news, earnings calendar, analyst upgrades, and insider activity
- Apply lessons from past reflections when evaluating candidates
- Do NOT default to well-known mega-caps — the screener exists to surface overlooked quality companies across mid and small caps

**Step 5 — Take action**
- Buy/sell based on your analysis, with detailed notes explaining the thesis for each trade

**Step 6 — Save reflection**
- Call `save_session_reflection` with a detailed write-up: actions taken, thesis validation, market observations, and lessons for future sessions

Focus on building a diversified, high-quality long-term portfolio. Apply everything you've learned.
"""
    return run_agent_session(prompt, model=model, max_iterations=40, **kwargs)


def run_custom_prompt(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Run the agent with a custom user prompt."""
    return run_agent_session(prompt, model=model, **kwargs)
