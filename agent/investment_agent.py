"""
Claude-powered investment agent.
Uses an agentic loop with tool use to research and manage a paper portfolio.
"""

import datetime
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
2. Call `get_sector_exposure` — see current sector weights so you don't double-down on an already heavy sector
3. Research the stock's fundamentals (P/E, PEG, FCF yield, margins, ROE, growth)
4. Check price history to understand valuation context
5. Check `get_earnings_calendar` — know when earnings are due. Be cautious buying immediately before an uncertain earnings date; if timing is wrong, add to watchlist instead.
6. Read `get_stock_news` — scan for any recent events that could break the investment thesis
7. Check `get_analyst_upgrades` — a cluster of recent downgrades is a warning signal
8. Check `get_insider_activity` — meaningful insider buying (especially by CEO/CFO) is one of the strongest confirmation signals available
9. Make the purchase if conviction is high. Pass the `screener_snapshot` dict from screen_stocks so the signal state is recorded for future performance attribution.
10. If you like the stock but timing is wrong (earnings too close, slightly overvalued, sector already heavy), call `add_to_watchlist` with a target entry price instead of buying.

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
- `get_stock_universe(index, sample_n, random_seed, sector)` — returns a sample of tickers from "sp500", "broad" (mid/small caps), or "all". Use the `sector` parameter (e.g. "Health Care", "Financials") to target sectors that fit the current macro regime. Sector filtering works for S&P 500 tickers. Call multiple times with different seeds or sectors to broaden coverage.
- `screen_stocks(tickers, top_n)` — fast parallel screen on up to 100 tickers. Scores each on revenue growth, margins, ROE, PEG ratio, FCF yield, debt, and 52-week momentum relative to the S&P 500. Returns ranked candidates with `peg_ratio`, `fcf_yield_pct`, and `relative_momentum_pct`. The ideal pick has a low PEG (cheap relative to growth), positive FCF yield (real cash generation), AND positive relative momentum (already working). Stocks with strongly negative relative momentum require extra conviction.

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
3. Call `get_watchlist` — check if any watchlist candidates have reached their target entry price or had a meaningful pullback since you added them.
4. Call `get_trade_outcomes` — raw signal snapshots for all past trades.
5. Call `get_signal_performance` — statistical breakdown of which signal thresholds have predicted positive returns. Use this to weight signals in screening: if PEG < 1.5 shows 70% win rate vs 40% without, make it a near-requirement.
6. Call `get_shadow_performance` — check how stocks you previously passed on have moved. Validate or challenge your past reasoning.

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


def _serialize_content(content) -> list:
    """Convert Anthropic SDK content blocks to plain JSON-serializable dicts."""
    out = []
    for block in content:
        t = getattr(block, "type", None)
        if t == "text":
            out.append({"type": "text", "text": block.text})
        elif t == "tool_use":
            out.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        else:
            out.append(block if isinstance(block, dict) else {"type": str(t)})
    return out


def _save_checkpoint(path: str, messages: list, iteration: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"iteration": iteration, "messages": messages}, f)


def _delete_checkpoint(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def run_agent_session(
    user_prompt: str,
    model: Optional[str] = None,
    max_iterations: int = 20,
    temperature: float = 0,
    on_text: Optional[callable] = None,
    on_tool_call: Optional[callable] = None,
    on_tool_result: Optional[callable] = None,
    checkpoint_path: Optional[str] = None,
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
        checkpoint_path: File path for saving/resuming session state. When set,
            the message history is written after each tool round-trip so the
            session can be resumed after a crash or credit error.

    Returns:
        The final text response from the agent.
    """
    # Resolve API key: prefer explicit env var, then fall back to session token file.
    _token_file = os.environ.get(
        "CLAUDE_SESSION_INGRESS_TOKEN_FILE",
        "/home/claude/.claude/remote/.session_ingress_token",
    )
    _api_key = os.environ.get("ANTHROPIC_API_KEY") or (
        open(_token_file).read().strip() if os.path.exists(_token_file) else None
    )
    if not _api_key:
        raise RuntimeError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY in .env or ensure the session token file exists."
        )
    client = anthropic.Anthropic(api_key=_api_key)
    model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

    # Resume from checkpoint if one exists, otherwise start fresh.
    messages = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            messages = ckpt["messages"]
            if on_text:
                on_text(
                    f"\n[Resuming from checkpoint — {len(messages)} messages, "
                    f"iteration {ckpt.get('iteration', '?')} — skipping already-completed work]\n\n"
                )
        except Exception:
            messages = None  # corrupt checkpoint → start fresh

    if messages is None:
        messages = [{"role": "user", "content": user_prompt}]

    final_text = ""

    # Waits (seconds) between retry attempts for rate-limit errors.
    # Token-per-minute limits reset every 60 s, so later waits are longer.
    _RETRY_WAITS = [0, 15, 30, 60, 90]

    for iteration in range(max_iterations):
        # Small pacing delay between iterations to spread token usage over time.
        if iteration > 0:
            time.sleep(3)

        # Retry with exponential backoff on rate-limit (429) and overloaded (529) errors.
        response = None
        last_err_label = "API limit"
        for attempt, wait in enumerate(_RETRY_WAITS):
            try:
                if wait:
                    if on_text:
                        on_text(f"\n[{last_err_label} — waiting {wait}s before retry {attempt}/{len(_RETRY_WAITS)-1}...]\n")
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
                last_err_label = "Rate limit hit"
                if attempt == len(_RETRY_WAITS) - 1:
                    raise
            except anthropic.InternalServerError as e:
                if getattr(e, "status_code", None) != 529:
                    raise
                last_err_label = "API overloaded (529)"
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

        # If no tool calls, the agent is done — clean up checkpoint and exit.
        if response.stop_reason == "end_turn" or not tool_calls:
            if checkpoint_path:
                _delete_checkpoint(checkpoint_path)
            break

        # Append assistant message (serialized so messages stay JSON-safe).
        messages.append({"role": "assistant", "content": _serialize_content(response.content)})

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

        # Persist state after every completed round-trip so we can resume on failure.
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, messages, iteration)

    portfolio.log_agent_message(f"Session completed: {user_prompt[:100]}")
    return final_text


def run_portfolio_review(model: Optional[str] = None, **kwargs) -> str:
    """Run a full autonomous portfolio review and rebalancing session."""
    # Rotate seeds daily so each review explores a fresh slice of the universe.
    # Seeds are deterministic within a day (temperature=0 handles within-session
    # consistency); over weeks the full ~2,700-ticker universe gets covered.
    day_seed = datetime.date.today().toordinal() % 1000
    seed_a, seed_b, seed_c = day_seed, day_seed + 1, day_seed + 2

    prompt = f"""
Please conduct a comprehensive portfolio review and take appropriate investment actions:

**Step 1 — Load memory**
- Call `get_investment_memory` to review past theses for current holdings and closed positions
- Call `get_session_reflections` to review lessons from past sessions
- Call `get_watchlist` — check if any watchlist candidates have hit their target price or had a meaningful pullback
- Call `get_trade_outcomes` — review raw signal snapshots for all past trades
- Call `get_signal_performance` — get statistical analysis of which signals (PEG, FCF yield, momentum, revenue growth) have actually predicted positive returns; use this to weight signals in Step 4
- Call `get_shadow_performance` — review stocks you previously passed on; note which passes were validated (stock fell) and which were mistakes (stock rose); apply lessons to this session's screening

**Step 2 — Assess current state**
- Check portfolio status (cash, holdings, P&L)
- Call `get_sector_exposure` — see current sector weights before making any new allocation decisions
- Call `get_macro_environment` — understand the rate/dollar/volatility regime before making any decisions
- Call `get_benchmark_comparison` — are we beating the S&P 500? If not, why not?
- Check overall market index conditions

**Step 3 — Evaluate existing positions**
- For each holding, compare current fundamentals against the original buy thesis
- Check recent news (`get_stock_news`) for any thesis-breaking events
- Check upcoming earnings (`get_earnings_calendar`) to flag positions with imminent earnings risk
- Identify any positions where the thesis has broken down or the position has grown too large

**Step 4 — Discover new opportunities from the full market**
- Based on the macro regime (Step 2), decide the 2 sectors best positioned right now.
  Use the sector allocation rules from your investment philosophy:
  high rates → Financials / Energy; inverted curve → Health Care / Consumer Staples;
  strong dollar → domestically focused sectors; high VIX → wait or size small.
- Call `get_stock_universe("sp500", sector="<your top sector>")` — targeted S&P 500 screen of your #1 macro-favoured sector
- Call `get_stock_universe("sp500", sector="<your second sector>")` — same for #2 sector
- Call `get_stock_universe("broad", random_seed={seed_b})` — broad mid/small-cap batch to surface overlooked names
- Call `get_stock_universe("broad", random_seed={seed_c})` — second broad batch for wider coverage
- Screen each batch with `screen_stocks` (50-100 tickers per call). The screener returns:
  - `peg_ratio`: prefer < 1.5 — paying a fair price for the growth rate
  - `fcf_yield_pct`: prefer > 3% — company generating real cash, not just accounting profits
  - `relative_momentum_pct`: prefer positive — stock already outperforming the S&P 500
  The ideal candidate is strong on ALL three: cheap relative to growth, cash-generative, and trending well.
  Be cautious of stocks with strongly negative relative momentum even if fundamentals look attractive.
- From all screener results, pick the 3-5 highest-scoring candidates for deep research
- For each finalist: check fundamentals, news, earnings calendar, analyst upgrades, and insider activity
- Apply lessons from `get_signal_performance` — if PEG < 1.5 has a 70% positive-return rate vs 40% when not met, require it; if momentum shows no edge, treat it as a tiebreaker only
- Do NOT default to well-known mega-caps — the screener exists to surface overlooked quality companies

**Step 5 — Take action**
- For each buy: pass `screener_snapshot` (the screen_stocks result dict for that ticker) to `buy_stock` so signals are recorded
- For stocks you like but won't buy yet (earnings soon, slightly overvalued, sector already heavy): call `add_to_watchlist` with target entry price
- For stocks you researched deeply but decided against AND won't watchlist: call `add_to_shadow_portfolio` with the current price and reason (e.g. "overvalued", "weak FCF", "thesis unclear"); this creates a record to audit next session
- For sells: clear reasoning in notes; if relevant, remove from watchlist

**Step 6 — Save reflection**
- Call `save_session_reflection` using the structured template defined in the tool description
- Be specific in "Lessons for Next Session" — write rules, not vague intentions

Focus on building a diversified, high-quality long-term portfolio. Apply everything you've learned.
"""
    kwargs.setdefault("checkpoint_path", "data/session_checkpoint.json")
    return run_agent_session(prompt, model=model, max_iterations=40, **kwargs)


def run_custom_prompt(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Run the agent with a custom user prompt."""
    return run_agent_session(prompt, model=model, **kwargs)
