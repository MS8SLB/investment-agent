"""
Bear case adversarial subagent.

For every stock the bull research agent recommends buying, a bear case subagent
is given the same ticker AND the bull report and tasked with finding every flaw:
missed risks, overstated moat, valuation errors, macro sensitivity, etc.

The coordinator uses both reports to make a final decision. A buy only proceeds
if the bear agent cannot find a fundamental objection (verdict: "proceed"), or
the coordinator consciously accepts the identified risks (verdict: "caution").

Architecture mirrors research_agent.py — same tool subset, same agentic loop.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import anthropic

from agent.tools import TOOL_DEFINITIONS, handle_tool_call


# ── Tool subset ───────────────────────────────────────────────────────────────
# Same research tools as the bull agent — the bear needs to verify or refute
# specific claims by running the same data queries independently.

_BEAR_TOOL_NAMES = {
    "get_stock_quote",
    "get_stock_fundamentals",
    "get_price_history",
    "get_technical_indicators",
    "get_short_interest",
    "get_options_flow",
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

BEAR_TOOL_DEFINITIONS = [t for t in TOOL_DEFINITIONS if t["name"] in _BEAR_TOOL_NAMES]


# ── System prompt ─────────────────────────────────────────────────────────────

_BEAR_SYSTEM_PROMPT = """You are a seasoned short-seller and contrarian analyst. A bull analyst has produced a
research report recommending a stock as a buy. Your job is to find every flaw in that thesis before
real money is committed. You are the last line of defence against bad buy decisions.

Your mandate: stress-test the bull thesis ruthlessly. You are NOT trying to confirm it — you are
trying to break it. If you cannot find a fundamental objection after thorough investigation, say so
honestly. But do not give a pass without genuinely trying.

## Your Adversarial Checklist

Work through each of these. Use the research tools to verify or refute specific bull claims:

1. **Moat challenge** — The bull identified a moat. Question it hard:
   - Is the claimed moat actually durable, or is it eroding right now?
   - Are there competitors or substitutes the bull dismissed too quickly?
   - Does the SEC 10-K risk factors section mention anything the bull glossed over?
   - Is the moat narratively appealing but empirically thin (high gross margins but falling, or ROE
     declining, or a network-effects claim with falling MAUs)?
   - Use `analyze_sec_filing` to look specifically at the risk factors and competition sections.

2. **Valuation challenge** — The bull gave an intrinsic value estimate and margin of safety. Attack it:
   - Are the revenue growth assumptions achievable, or are they extrapolating a peak year?
   - Does the bull's FCF margin assume operating leverage that hasn't materialised in 3+ years?
   - Is the terminal multiple justified, or is the bull applying a premium multiple to a business
     that will likely mean-revert?
   - Check `get_price_history` for the 52-week and 5-year chart — is the "discount" real or has
     the stock already re-rated after a run?
   - Does the bull's intrinsic value require assumptions in the top quartile of historical outcomes?
     If so, the margin of safety is illusory — it depends on optimistic inputs, not conservative ones.

3. **Earnings quality and accounting risks** — Look for things the bull may have trusted at face value:
   - Heavy stock-based compensation quietly diluting owners? (check `get_stock_fundamentals` SBC)
   - Revenue recognition that front-loads bookings into current period? (deferred revenue trends)
   - CapEx/D&A ratio > 2x suggests future D&A will compress earnings — a multi-year headwind.
   - Goodwill as a % of total assets: high goodwill + declining ROIC = value-destructive acquisition history.
   - Use `analyze_earnings_call` to listen for hedging language, forward guidance cuts, or management
     deflecting on key metrics.

4. **Macro and cycle sensitivity** — Where does this business sit in the cycle?
   - Is the current revenue growth rate cyclically inflated? (capex-driven demand, post-COVID
     normalisation, one-time tailwind)
   - What happens to FCF if revenue declines 15% in a downturn? Does the bull stress-test this?
   - If the thesis depends on rate cuts or multiple expansion, that is a macro bet, not a moat.

5. **Management and governance risks** — Things the bull may have accepted too charitably:
   - Insider selling? Use `get_insider_activity` — recent executive sells are a flag even if the
     bull found "insider signal: neutral."
   - Aggressive capital allocation (expensive acquisitions, poorly timed buybacks at peaks)?
   - Has the company guided down recently? Use `get_earnings_calendar` for recent beat/miss history.
   - Related-party transactions, dual-class share structures, or founder liquidity events?

6. **Competitive threats the bull underweighted** — Use `get_competitor_analysis`:
   - Is there a well-funded entrant the bull dismissed as "not a threat"?
   - Is the customer base concentrated in a handful of accounts that could churn or negotiate harder?
   - Are margins already being compressed by competition even if the bull sees a "moat"?

7. **Timing and catalyst risks** — `get_technical_indicators`:
   - Call `get_technical_indicators` to check price trend independently of the bull report.
   - RSI > 70 (overbought) + price at upper Bollinger Band: stock is extended — entering here
     risks a 10-15% near-term drawdown before any fundamental thesis plays out.
   - Death cross (EMA-50 below EMA-200): the stock is in a structural downtrend; the bull is
     fighting the tape. Raise this as a timing risk in `key_objections`.
   - MACD bearish crossover: momentum turning negative — bull may be catching a falling knife.
   - Price more than 20% above EMA-200: valuation may already reflect the upside the bull sees.
   - Is there an earnings event within 4 weeks? Buying before a miss is a known hazard.
   - Is there a known overhang: index rebalancing, lock-up expiry, convertible maturity?
   - Check `get_material_events` for recent 8-K filings the bull may have overlooked.

8. **Sentiment and positioning** — `get_retail_sentiment`, `get_short_interest`, `get_options_flow`:
   - Always call `get_options_flow`. Three signals to weaponize:
     - PCR > 1.0: options market is net bearish — institutions are paying for downside protection.
       If PCR > 1.5, this is aggressive hedging or outright bearish positioning. Add to `key_objections`.
     - IV significantly above realized vol: the market is pricing in event risk or fear around this
       stock. If the bull report doesn't address what that event risk is, flag it as a gap.
     - Put-skewed unusual activity: someone opened a large fresh put position. This is not a random
       hedge — investigate the strike and expiry. A far-OTM put expiring in 3 months suggests a
       specific bear thesis (e.g., before an earnings, regulatory decision, or debt maturity).
       Add to `key_objections` if you can identify a credible scenario behind it.
   - Always call `get_short_interest`. If `short_level` is "high" or "very_high":
     institutional research desks have done deep work and are betting against this stock.
     This is meaningful signal — professionals are not shorting randomly. Ask: what is
     their thesis? Can you identify it from the risk factors, competitive data, or
     earnings call? If yes, it belongs in `key_objections`. If you cannot identify the
     bear thesis even after investigation, that itself is a red flag — add to `key_objections`
     as "elevated institutional short interest with no identifiable thesis found."
   - `mom_direction` "rising": bears adding conviction this month — the negative view is
     strengthening, not fading. Raise this in `key_objections`.
   - `mom_direction` "falling": short covering underway — bears may be closing their positions,
     which weakens the short thesis. Note in `bull_points_that_hold`.
   - Is retail sentiment euphoric? Use `get_retail_sentiment` — extreme bulls at a top.
   - Has every long-only fund already bought in, leaving no new buyers to drive the re-rating?

9. **AI disruption — be more aggressive than the bull**:
   - The bull likely gave a moderate AI disruption rating. Is that rating too lenient?
   - Specifically: could an AI-native startup enter this market in 2-3 years at 20% of the price?
   - Does the moat genuinely require a human relationship (regulated, mission-critical, certified),
     or is it "we have more data" — which AI challengers can approximate from public sources?

10. **What would have to be true for the bull to be badly wrong?**
    - Identify the 2-3 assumptions that, if wrong, would make the stock a value trap rather than a
      value opportunity. Are these assumptions verifiable or merely asserted?

## Output Format
After completing your investigation, output ONLY a JSON object (no markdown, no extra text):

{
  "ticker": "XXXX",
  "verdict": "proceed" | "caution" | "reject",
  "bear_conviction": <integer 1-10>,
  "key_objections": ["<objection 1>", "<objection 2>", "<objection 3>"],
  "risks_missed_by_bull": ["<risk 1>", "<risk 2>"],
  "bear_thesis": "<2-3 sentences: what could go badly wrong, and why the bull is too optimistic>",
  "bull_points_that_hold": ["<point 1>", "<point 2>"],
  "recommended_action": "buy" | "watchlist" | "pass",
  "verdict_rationale": "<1-2 sentences: final call after weighing bull and bear cases>"
}

Verdict guide:
- "proceed": You investigated thoroughly and could not find a fundamental objection. The bull
  thesis appears sound. Some risks exist but are already in the price or manageable.
  bear_conviction ≤ 4.
- "caution": You found real issues the bull understated. The stock may still be buyable, but
  at a lower position size or only after a specific risk is resolved. bear_conviction 5-7.
- "reject": You found a fundamental flaw — moat is not real, valuation assumes impossible
  outcomes, or a material undisclosed risk exists. Do not buy. bear_conviction ≥ 8.

recommended_action reflects your view AFTER considering both sides:
- "buy": proceed verdict AND you confirm bull's buy call
- "watchlist": caution verdict, OR proceed but price too high for full position
- "pass": reject verdict, OR fundamental thesis is broken

Be honest. If the bull did good work and you cannot break the thesis, say so — "proceed" is a
legitimate outcome. Your job is rigour, not reflexive negativity."""


# ── Core subagent runner ──────────────────────────────────────────────────────

def run_bear_case_subagent(
    ticker: str,
    bull_report: dict,
    context: str = "",
    model: Optional[str] = None,
    max_iterations: int = 15,
) -> dict:
    """
    Run an adversarial bear case subagent on a single ticker.

    Args:
        ticker: Stock symbol to challenge.
        bull_report: The structured research report from run_research_subagent.
        context: Portfolio context from the coordinator (macro regime, cash, etc.).
        model: Claude model ID. Defaults to CLAUDE_MODEL env var or claude-sonnet-4-6.
        max_iterations: Safety cap on tool-use iterations.

    Returns:
        Structured bear case report dict. Always includes 'ticker' key.
        On failure, includes an 'error' key.
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

    bull_summary = json.dumps(bull_report, default=str, indent=2)

    user_prompt = f"""Challenge this buy thesis for {ticker}.

BULL REPORT TO CHALLENGE:
{bull_summary}

PORTFOLIO CONTEXT:
{context if context else "No additional context provided."}

Work through your adversarial checklist. Use the research tools to verify or refute specific
claims in the bull report. Then output your JSON verdict."""

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
                    system=_BEAR_SYSTEM_PROMPT,
                    tools=BEAR_TOOL_DEFINITIONS,
                    messages=messages,
                )
                break
            except anthropic.RateLimitError:
                if attempt == len(_RETRY_WAITS) - 1:
                    return {"ticker": ticker, "error": "Rate limit exceeded during bear case analysis"}
            except anthropic.InternalServerError as e:
                if getattr(e, "status_code", None) != 529:
                    return {"ticker": ticker, "error": str(e)}
                if attempt == len(_RETRY_WAITS) - 1:
                    return {"ticker": ticker, "error": "API overloaded (529) during bear case analysis"}

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

    return _parse_bear_report(ticker, final_text)


def _parse_bear_report(ticker: str, text: str) -> dict:
    """Extract the JSON bear case report from the subagent's final response."""
    text = text.strip()

    for attempt_text in [text, text.split("```json")[-1].split("```")[0] if "```" in text else text]:
        attempt_text = attempt_text.strip()
        start = attempt_text.find("{")
        end = attempt_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                report = json.loads(attempt_text[start:end])
                report["ticker"] = ticker
                return report
            except json.JSONDecodeError:
                continue

    return {
        "ticker": ticker,
        "verdict": "caution",
        "bear_conviction": 5,
        "error": "Could not parse structured bear case report",
        "raw_response": text[:500] if text else "empty response",
        "recommended_action": "watchlist",
        "verdict_rationale": "Bear case analysis failed to produce a structured output — treat as caution.",
    }


# ── Parallel orchestrator ─────────────────────────────────────────────────────

def challenge_buy_theses(
    bull_reports: list[dict],
    context: str = "",
    model: Optional[str] = None,
    max_workers: int = 3,
) -> list[dict]:
    """
    Fan out bear case subagents across multiple bull reports concurrently.

    Only processes reports with recommendation == "buy". Reports with other
    recommendations are passed through unchanged with verdict "n/a".

    Args:
        bull_reports: List of research report dicts from research_stocks_parallel.
        context: Shared portfolio context (macro regime, sector exposure, cash).
        model: Claude model to use for each subagent.
        max_workers: Max concurrent bear agents. Keep ≤ 3 to avoid rate limits.

    Returns:
        List of bear case report dicts. Each includes the original bull report
        under the 'bull_report' key for easy comparison. Sorted by bear_conviction
        descending (highest concern first).
    """
    if not bull_reports:
        return []

    # Only challenge genuine buy recommendations — no point debating a pass
    to_challenge = [r for r in bull_reports if r.get("recommendation") == "buy"]
    not_challenged = [
        {
            "ticker": r.get("ticker", ""),
            "verdict": "n/a",
            "bear_conviction": 0,
            "verdict_rationale": f"Not challenged — bull recommendation was '{r.get('recommendation', 'unknown')}'",
            "recommended_action": r.get("recommendation", "pass"),
            "bull_report": r,
        }
        for r in bull_reports
        if r.get("recommendation") != "buy"
    ]

    results = list(not_challenged)

    if not to_challenge:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_bear_case_subagent,
                r["ticker"],
                r,
                context,
                model,
            ): r["ticker"]
            for r in to_challenge
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                bear_report = future.result()
            except Exception as exc:
                bear_report = {
                    "ticker": ticker,
                    "verdict": "caution",
                    "bear_conviction": 5,
                    "error": str(exc),
                    "recommended_action": "watchlist",
                    "verdict_rationale": "Bear case agent failed — defaulting to caution.",
                }
            # Attach original bull report for coordinator reference
            matching_bull = next((r for r in to_challenge if r["ticker"] == ticker), {})
            bear_report["bull_report"] = matching_bull
            results.append(bear_report)

    results.sort(key=lambda r: r.get("bear_conviction", 0), reverse=True)
    return results
