"""
Research subagent for parallel stock deep-dives.

Each call to run_research_subagent() spins up an independent Claude agentic loop
focused on one ticker. research_stocks_parallel() fans N of these out concurrently
using a thread pool and collects structured JSON reports.

The coordinator agent (investment_agent.py) calls research_stocks_parallel via
the research_stocks_parallel tool, receives all reports at once, and makes
trade decisions from the synthesised results — without having to research
stocks one-by-one in its own context.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import anthropic

from agent.tools import TOOL_DEFINITIONS, handle_tool_call


# ── Tool subset ───────────────────────────────────────────────────────────────
# Only include tools relevant to researching a single stock.
# Keeps the subagent context lean (~15 tools instead of 41).

_RESEARCH_TOOL_NAMES = {
    "get_stock_quote",
    "get_stock_fundamentals",
    "get_price_history",
    "get_stock_news",
    "get_earnings_calendar",
    "get_analyst_upgrades",
    "get_insider_activity",
    "get_google_trends",
    "get_retail_sentiment",
    "get_rss_news",
    "analyze_earnings_call",
    "analyze_sec_filing",
    "get_material_events",
    "get_competitor_analysis",
    "get_superinvestor_positions",
}

RESEARCH_TOOL_DEFINITIONS = [t for t in TOOL_DEFINITIONS if t["name"] in _RESEARCH_TOOL_NAMES]


# ── System prompt ─────────────────────────────────────────────────────────────

_RESEARCH_SYSTEM_PROMPT = """You are a senior fundamental equity analyst conducting a deep-dive on a single stock.

Your job is to research the assigned ticker thoroughly and return a structured JSON report.
You are NOT making the final trade decision — the portfolio manager will do that.
Your job is to surface all relevant facts, highlight genuine positives and risks, and give
a clear recommendation with supporting evidence.

## Research Checklist
Work through these in order, using available tools:

1. **Fundamentals** — `get_stock_fundamentals`: P/E, PEG, FCF yield, profit margin, ROE, revenue growth, debt
2. **Price context** — `get_price_history` (1y): understand valuation vs. recent range
3. **Earnings risk** — `get_earnings_calendar`: next date, consensus, beat/miss history
4. **News** — `get_stock_news` + `get_rss_news`: scan for thesis-breaking or confirming events
5. **Analyst sentiment** — `get_analyst_upgrades`: cluster of downgrades = warning
6. **Insider signal** — `get_insider_activity`: meaningful CEO/CFO buying = strong bullish signal
7. **Material events** — `get_material_events` (90 days): CFO exits, impairments, restatements
8. **Peer valuation** — `get_competitor_analysis`: is the premium/discount justified?
9. **Management quality** — `analyze_earnings_call`: confidence, guidance quality, Q&A tone
10. **Smart money** — `get_superinvestor_positions`: do Buffett/Ackman/Druckenmiller agree?
11. **SEC filings** — `analyze_sec_filing` (10-K): moat language, new risk factors, MD&A tone
12. **Alternative signals** — `get_google_trends`, `get_retail_sentiment`: demand trends, contrarian thermometer

## Output Format
After completing your research, output ONLY a JSON object with this exact structure (no markdown, no extra text):

{
  "ticker": "XXXX",
  "recommendation": "buy" | "watchlist" | "pass",
  "conviction_score": <integer 1-10>,
  "target_entry_price": <number or null>,
  "one_line_thesis": "<25 words max>",
  "key_positives": ["<point 1>", "<point 2>", "<point 3>"],
  "key_risks": ["<risk 1>", "<risk 2>"],
  "full_thesis": "<2-3 sentences for the buy notes if purchased>",
  "earnings_risk": "low" | "medium" | "high",
  "insider_signal": "bullish" | "neutral" | "bearish",
  "superinvestor_backing": true | false,
  "metrics": {
    "current_price": <number or null>,
    "pe_ratio": <number or null>,
    "peg_ratio": <number or null>,
    "fcf_yield_pct": <number or null>,
    "revenue_growth_pct": <number or null>,
    "profit_margin_pct": <number or null>,
    "roe_pct": <number or null>,
    "next_earnings_date": "<date string or null>"
  }
}

Recommendation guide:
- "buy": high conviction (score ≥ 7), fundamentals are strong, no major red flags, good entry timing
- "watchlist": like the business but timing is off (earnings too close, slightly overvalued, sector risk)
- "pass": weak fundamentals, thesis unclear, or serious red flags found

Be honest. Do not inflate conviction to please anyone. A well-reasoned "pass" is more valuable than a weak "buy"."""


# ── Core subagent runner ──────────────────────────────────────────────────────

def run_research_subagent(
    ticker: str,
    screener_data: Optional[dict] = None,
    context: str = "",
    model: Optional[str] = None,
    max_iterations: int = 18,
) -> dict:
    """
    Run a focused research subagent on a single ticker.

    Args:
        ticker: Stock symbol to research.
        screener_data: Screener metrics for this ticker (from screen_stocks), passed as context.
        context: Additional context from the coordinator (macro regime, sector exposure, etc.).
        model: Claude model ID. Defaults to CLAUDE_MODEL env var or claude-sonnet-4-6.
        max_iterations: Safety cap on tool-use iterations.

    Returns:
        Structured research report dict. Always includes 'ticker' key.
        On failure, includes an 'error' key with the error message.
    """
    _token_file = os.environ.get(
        "CLAUDE_SESSION_INGRESS_TOKEN_FILE",
        "/home/claude/.claude/remote/.session_ingress_token",
    )
    _api_key = os.environ.get("ANTHROPIC_API_KEY") or (
        open(_token_file).read().strip() if os.path.exists(_token_file) else None
    )
    if not _api_key:
        return {"ticker": ticker, "error": "No API key available"}

    client = anthropic.Anthropic(api_key=_api_key)
    model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

    # Build the user prompt with all available context
    screener_summary = ""
    if screener_data:
        screener_summary = f"\nScreener snapshot: {json.dumps(screener_data, default=str)}"

    user_prompt = f"""Research {ticker} thoroughly and return a structured JSON report.{screener_summary}

Portfolio context:
{context if context else "No additional context provided."}

Complete all research steps from your checklist, then output the JSON report."""

    messages = [{"role": "user", "content": user_prompt}]
    final_text = ""

    _RETRY_WAITS = [0, 15, 30, 60]

    for iteration in range(max_iterations):
        if iteration > 0:
            time.sleep(2)

        response = None
        for attempt, wait in enumerate(_RETRY_WAITS):
            try:
                if wait:
                    time.sleep(wait)
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=0,
                    system=_RESEARCH_SYSTEM_PROMPT,
                    tools=RESEARCH_TOOL_DEFINITIONS,
                    messages=messages,
                )
                break
            except anthropic.RateLimitError:
                if attempt == len(_RETRY_WAITS) - 1:
                    return {"ticker": ticker, "error": "Rate limit exceeded during research"}
            except anthropic.InternalServerError as e:
                if getattr(e, "status_code", None) != 529:
                    return {"ticker": ticker, "error": str(e)}
                if attempt == len(_RETRY_WAITS) - 1:
                    return {"ticker": ticker, "error": "API overloaded (529) during research"}

        if response is None:
            break

        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        if text_parts:
            final_text = "\n".join(text_parts)

        if response.stop_reason == "end_turn" or not tool_calls:
            break

        # Serialize assistant turn
        serialized = []
        for block in response.content:
            t = getattr(block, "type", None)
            if t == "text":
                serialized.append({"type": "text", "text": block.text})
            elif t == "tool_use":
                serialized.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        messages.append({"role": "assistant", "content": serialized})

        # Execute tools
        tool_results = []
        for tc in tool_calls:
            result = handle_tool_call(tc.name, tc.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": tool_results})

    # Parse the JSON report from the final text
    return _parse_research_report(ticker, final_text, screener_data)


def _parse_research_report(ticker: str, text: str, screener_data: Optional[dict]) -> dict:
    """Extract the JSON report from the subagent's final response."""
    text = text.strip()

    # Try to extract JSON from the response
    for attempt_text in [text, text.split("```json")[-1].split("```")[0] if "```" in text else text]:
        attempt_text = attempt_text.strip()
        # Find the outermost JSON object
        start = attempt_text.find("{")
        end = attempt_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                report = json.loads(attempt_text[start:end])
                report["ticker"] = ticker  # ensure ticker is always set
                if screener_data:
                    report["screener_snapshot"] = screener_data
                return report
            except json.JSONDecodeError:
                continue

    # Fallback: return a pass with error context
    return {
        "ticker": ticker,
        "recommendation": "pass",
        "conviction_score": 0,
        "error": "Could not parse structured report from subagent",
        "raw_response": text[:500] if text else "empty response",
        "screener_snapshot": screener_data,
    }


# ── Parallel orchestrator ─────────────────────────────────────────────────────

def research_stocks_parallel(
    tickers_with_data: list[dict],
    context: str = "",
    model: Optional[str] = None,
    max_workers: int = 4,
) -> list[dict]:
    """
    Fan out research subagents across multiple tickers concurrently.

    Args:
        tickers_with_data: List of dicts, each with 'ticker' and optional 'screener_data'.
            Example: [{"ticker": "AAPL", "screener_data": {...}}, {"ticker": "MSFT"}]
        context: Shared portfolio context for all subagents (macro regime, sector exposure, etc.).
        model: Claude model to use for each subagent.
        max_workers: Maximum concurrent subagents. Keep ≤ 5 to avoid rate limits.

    Returns:
        List of research report dicts, sorted by conviction_score descending.
    """
    if not tickers_with_data:
        return []

    reports = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_research_subagent,
                item["ticker"],
                item.get("screener_data"),
                context,
                model,
            ): item["ticker"]
            for item in tickers_with_data
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                report = future.result()
            except Exception as exc:
                report = {
                    "ticker": ticker,
                    "recommendation": "pass",
                    "conviction_score": 0,
                    "error": str(exc),
                }
            reports.append(report)

    # Sort by conviction score, highest first
    reports.sort(key=lambda r: r.get("conviction_score", 0), reverse=True)
    return reports
