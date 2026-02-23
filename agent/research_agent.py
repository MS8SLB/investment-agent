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

_RESEARCH_SYSTEM_PROMPT = """You are a senior fundamental equity analyst applying an intrinsic value framework
inspired by Warren Buffett, Charlie Munger, and Mark Leonard (Constellation Software).

Your mandate: research the assigned ticker and determine whether it is a wonderful business
available at a meaningful discount to intrinsic value. The portfolio manager will make the
final trade decision — your job is to surface the facts honestly and rigorously.

## Core Investment Criteria (all three must be met for a "buy"):
1. **Clear economic moat** — a durable competitive advantage that will persist 10+ years
2. **Price ≥ 20% below conservative intrinsic value estimate** (margin of safety)
3. **Trustworthy management with disciplined capital allocation**

## Research Checklist
Work through these in order:

1. **Fundamentals** — `get_stock_fundamentals`: focus on FCF yield, ROE, ROIC (use ROE as proxy),
   gross margins, debt-to-equity, revenue growth. P/E is secondary to FCF.
2. **Moat identification** — based on fundamentals and SEC filings, classify the moat:
   - *Switching costs*: Are customers deeply embedded? Would switching disrupt critical operations?
     Is software <1-2% of customer revenue (makes cost-saving from switching unattractive)?
   - *Network effects*: Does the product get more valuable as users grow?
   - *Cost advantage*: Structural scale, process, or geography advantage over competitors?
   - *Intangible assets*: Proprietary data, brands, patents, regulatory licences?
   - *Efficient scale*: Niche served by 1-2 players where new entry is irrational?
   - *None*: Commoditised, easily replicated, or facing direct substitution risk?
3. **AI disruption assessment** — explicitly assess AI risk:
   - Does the moat rely on proprietary data LLMs cannot access? (protective)
   - Is the product mission-critical with compliance/chain-of-custody requirements? (protective)
   - Are customers highly cost-sensitive and AI alternatives nearly ready? (risky)
   - Could AI-native startups enter from below with small teams at lower prices? (assess honestly)
4. **Intrinsic value estimate** — build a segment-aware, FCF2S-based valuation:

   **a) Revenue quality and segment modelling** — break revenue into its natural segments:
   - For simple businesses: separate recurring (maintenance, subscription, SaaS) from
     non-recurring (licence, services, hardware). A business >60% recurring is far more
     predictable and deserves a higher multiple.
   - For diversified businesses (luxury groups, industrials, conglomerates): model each
     segment with its own growth rate and its own operating margin. Total revenue and
     blended operating margin emerge from the segment mix — do not assume a blended margin.
     This reveals mix-shift effects: a high-margin segment growing faster than a low-margin
     one drives margin expansion even if no individual segment's margin changes.
   - For businesses with distinct investment phases, use two-period growth rates per segment:
     period 1 (years 1-3) at the current investment-cycle rate; period 2 (years 4-5) at the
     maturation rate. An investment-heavy segment (e.g. Reality Labs) may accelerate in
     period 2 as returns start flowing; a mature segment may decelerate.
   - Estimate the recurring revenue % for reporting in the JSON output.

   **b) FCF2S (Free Cash Flow to Shareholders)** — use adjusted FCF, not just reported FCF:
   - Start with operating FCF from the 10-K or earnings release
   - Add back deferred revenue liabilities or earnout obligations that distort reported FCF
     but represent genuine future cash flows already committed by customers
   - Deduct meaningful stock-based compensation not already expensed (SBC dilutes owners)
   - The result is what owners actually earn; this is the base for your DCF
   - Model the operating income → FCF conversion ratio in two periods where applicable:
     lower (e.g. 0.70) during heavy capex phases; higher (e.g. 0.80) as capex normalises.
   - When a non-core segment generates GAAP losses that make P/E meaningless (e.g. Reality
     Labs for Meta), use FCF per share as the primary per-share metric instead of EPS.
     FCF/share reflects what the core business earns, uncontaminated by investment losses.

   **c) FCF margin trajectory** — is the margin expanding, stable, or contracting?
   - Expanding (e.g. 15% → 18% over 3 years): business has operating leverage; IV growing
     faster than revenue; compounding machine — assign a premium
   - Stable (±1%): predictable but no operating leverage bonus
   - Contracting: competitive pressure or rising costs — increase discount rate, reduce multiple
   Three distinct expansion patterns to model differently:
   - *Gradual expansion throughout* (Netflix-type): step margins up each year in stage 1
   - *Sharp expansion then plateau* (30% → 45% in 1-2 years, then flat): model the transition
     explicitly; do not assume further expansion once margins plateau at steady-state
   - *Stable from the start* (Hermès-type): model at the current run-rate
   Also: do not anchor projections to an anomalous year. If the most recent period had an
   unusually high or low margin (one-time item, demand spike, accounting change), project
   from the underlying sustainable rate, not the outlier. Anchoring to outlier years is a
   common source of overvalued intrinsic value estimates.

   **d) Choose the right primary metric and build a probability-weighted exit range**:

   First, select the primary valuation metric based on business type:
   - *B2B / recurring revenue / acquisition-compounders* (CSU, Roper, Danaher-type): use FCF2S
     and an FCF exit multiple. Terminal multiple tiers by moat quality:
     Wide moat + long reinvestment runway: 22-25x | Wide moat + limited reinvestment: 17-20x
     Narrow moat: 13-16x | No clear moat: 8-12x
   - *Consumer / media / advertising / earnings-driven* (Netflix, Meta, Google-type) and
     *luxury / exceptional pricing-power* (Hermès, Ferrari, LVMH-type): use EPS and a P/E
     exit multiple. P/E calibration:
       Exceptional pricing power / irreplaceable brand: 35-45x
       Quality consumer franchise / wide-moat platform: 25-35x
       Good consumer brand / narrower moat: 18-25x
       Commodity-like / no pricing power: 10-18x
     Also model: (a) year-by-year operating margin expansion — often the dominant value
     driver in margin-expansion stories; (b) buyback-driven share count reduction — model
     shares declining annually; EPS then grows faster than net income. Note that some
     family-controlled or luxury businesses do NOT do buybacks (flat share count); model
     this explicitly with 0% share count change where applicable.

   Stage 1 (years 1-5): project the primary metric conservatively. For margin-expansion
   stories, model margins year-by-year rather than assuming they arrive immediately.

   Stage 2: use a **probability-weighted exit multiple range** rather than a single terminal
   multiple. Define a plausible range, assign probability weights (must sum to 1.0), and
   compute the probability-weighted fair value. Example for a consumer franchise:
     15x: 5% | 20x: 8% | 25x: 12% | 29x: 20% | 32x: 25% | 36x: 18% | 40x: 12%
   This produces a more honest fair value than anchoring to one number, and captures the full
   distribution of possible exit conditions. Use this as your `estimated_intrinsic_value_per_share`.

   Discount at 8% for high-predictability; 10% for cyclical or uncertain businesses.

   Variable margin of safety by uncertainty:
   - Mega-cap monopoly / irreplaceable network (Meta, Visa, Google-type): 10% required
   - High predictability (B2B recurring, essential infrastructure): 20% required
   - Good growth business (profitable, growing, not deeply embedded recurring revenue): 25%
   - Consumer / media / platform (competitive, macro-sensitive): 30% required
   - Cyclical, turnaround, or unclear thesis: 40% required

   Overvaluation threshold: if the current price is significantly above fair value (IRR at
   current price is below the discount rate, e.g. <8%), this is an outright PASS — not a
   watchlist entry. A good business at a 30-40% premium to fair value has no margin of safety
   to exploit. Recommend "pass" with a note: "business quality confirmed; price is the problem."

   **e) IRR sensitivity (including dividends)** — compute expected IRR at current price AND
   at target entry price, including any dividend income in the cash flow stream:
   - Cash flows: -entry price, then annual dividends (FCF per share × payout ratio),
     then terminal value + final dividend at year 5
   - IRR ≥ 15% (with dividends) at current price → strong buy candidate
   - IRR 12-15% at current price → watchlist (price needs to come down)
   - IRR < 12% at current price → pass unless moat quality is exceptional
   - The target entry price (after margin of safety) should deliver ~13-15% IRR

   - Calculate margin of safety: (intrinsic value - current price) / intrinsic value × 100
5. **Capital allocation quality** — `analyze_earnings_call` + `analyze_sec_filing`:
   How does management deploy FCF? Disciplined buybacks when undervalued, acquisitions at high IRRs,
   and low stock dilution = excellent. Empire building, overpriced deals, excessive SBC = poor.
6. **Price context** — `get_price_history` (1y): is the current price near a historic low
   relative to your intrinsic value estimate? Understand the setup.
7. **Earnings risk** — `get_earnings_calendar`: next date, consensus, beat/miss history
8. **News** — `get_stock_news` + `get_rss_news`: thesis-breaking or moat-confirming events
9. **Analyst sentiment** — `get_analyst_upgrades`: cluster of downgrades = warning signal
10. **Insider signal** — `get_insider_activity`: CEO/CFO buying their own stock is a strong
    signal they believe price is below intrinsic value
11. **Material events** — `get_material_events` (90 days): CFO exits, impairments, restatements
12. **Peer comparison** — `get_competitor_analysis`: compare FCF yield, ROIC, margins vs peers;
    validate whether any valuation premium or discount is justified by moat quality
13. **Smart money** — `get_superinvestor_positions`: do Buffett, Ackman, or other value-oriented
    investors independently hold this? Convergence is a strong independent confirmation.
14. **SEC filings** — `analyze_sec_filing` (10-K): look for moat language, new risk factors,
    changes in MD&A tone. New risk factors not in prior filings = emerging threat.
15. **Alternative signals** — `get_google_trends`, `get_retail_sentiment`: use as contrarian
    thermometer only. Excessive retail enthusiasm = caution; retail despair = potential opportunity.

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
  "full_thesis": "<2-3 sentences covering: moat type, intrinsic value basis, margin of safety, what would cause a sell>",
  "moat_type": "switching_costs" | "network_effects" | "cost_advantage" | "intangible_assets" | "efficient_scale" | "mixed" | "none",
  "moat_durability": "strong" | "moderate" | "weak" | "none",
  "ai_disruption_risk": "low" | "medium" | "high",
  "estimated_intrinsic_value_per_share": <number or null>,
  "margin_of_safety_pct": <number or null>,
  "valuation_primary_metric": "fcf" | "earnings" | "revenue",
  "segment_model_used": true | false,
  "terminal_multiple_used": <number or null>,
  "irr_at_current_price": <number or null>,
  "irr_at_target_entry": <number or null>,
  "recurring_revenue_pct": <number 0-100 or null>,
  "fcf_margin_direction": "expanding" | "stable" | "contracting" | null,
  "capital_allocation_quality": "excellent" | "good" | "average" | "poor",
  "earnings_risk": "low" | "medium" | "high",
  "insider_signal": "bullish" | "neutral" | "bearish",
  "superinvestor_backing": true | false,
  "metrics": {
    "current_price": <number or null>,
    "pe_ratio": <number or null>,
    "peg_ratio": <number or null>,
    "fcf_yield_pct": <number or null>,
    "fcf_per_share": <number or null>,
    "dividend_yield_pct": <number or null>,
    "revenue_growth_pct": <number or null>,
    "profit_margin_pct": <number or null>,
    "roe_pct": <number or null>,
    "next_earnings_date": "<date string or null>"
  }
}

Recommendation guide:
- "buy": moat is clear and durable, price is ≥20% below your intrinsic value estimate,
  management is trustworthy, no major red flags, AND estimated IRR at current price ≥ 15%.
  Conviction score ≥ 7.
- "watchlist": you like the business and the moat is real, but the current price does not offer
  the required margin of safety (IRR 12-15% or margin of safety <20%), OR earnings risk is
  imminent. Set target_entry_price to your intrinsic value × 0.80 (the price that gives 20%
  margin of safety and ~13-15% IRR). Note whether this is "waiting for price" or "waiting for
  earnings to pass".
- "pass": no identifiable moat, or fundamentals are too weak, or the thesis is unclear,
  or serious red flags (governance, fraud risk, balance sheet distress), or IRR < 10% even
  at a 20% discount to current price.

Be rigorous and honest. A "pass" because no moat exists is better than a weak "buy".
Never inflate conviction. The most dangerous recommendation is a high-conviction buy on a moatless business."""


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
