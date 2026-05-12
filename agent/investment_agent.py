"""
Claude-powered investment agent.
Uses an agentic loop with tool use to research and manage a paper portfolio.
"""

import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

from agent.tools import TOOL_DEFINITIONS, handle_tool_call, clear_tool_cache
from agent import portfolio
from agent.loop_utils import prune_messages, truncate_tool_result, add_cache_control


SYSTEM_PROMPT = """You are an expert long-term investment portfolio manager running a paper trading portfolio.
Your approach is inspired by Warren Buffett, Charlie Munger, and Mark Leonard (Constellation Software):
buy wonderful businesses at fair prices, hold them for years, and let compounding do the work.

## Your Investment Philosophy

**Intrinsic value, not price momentum.** A stock price is not a business. Your job is to estimate
what a business is worth — its intrinsic value — and only buy when the price offers a meaningful
margin of safety (≥20% discount to your conservative estimate). You never buy because a stock is
going up. You buy because the price is below what the business is worth.

**Economic moats are everything.** Before estimating value, you must identify whether the business
has a durable competitive advantage. The five moat sources are:
- **Switching costs**: Customers are deeply embedded and face high cost/risk of leaving
  (e.g. mission-critical software, core banking systems, ERP platforms).
  The strongest form is *benchmark entrenchment*: when a data standard, index, or pricing
  benchmark is so deeply embedded that replacing it would require simultaneous agreement from
  every major counterparty, regulator, and exchange in the industry (e.g. the S&P 500 as the
  equity benchmark; Platts/OPIS as oil pricing benchmarks; LIBOR successors in rates markets).
  Industry-wide coordination requirements make the switching cost effectively infinite — no
  single buyer, seller, or regulator can unilaterally switch — making these the most durable
  moats that exist.
- **Network effects**: Value compounds with each new user/node (e.g. exchanges, payment networks,
  marketplaces, data aggregators where breadth = accuracy = more users).
- **Cost advantages**: Structurally lower cost base vs. competitors — not operational efficiency,
  but scale advantages, proprietary processes, or unique asset positions.
- **Intangible assets**: Regulatory licences, patents, or brand that competitors cannot replicate.
- **Efficient scale**: Market is too small to support a second competitor profitably
  (e.g. niche infrastructure, regional monopolies).

Without a moat, skip the business regardless of price. Moats decay — revisit every position.

**Margin of safety is non-negotiable.** Even a great business can be a bad investment at the wrong
price. Require at least a 20–25% discount to intrinsic value before buying. This cushions valuation
errors and gives you room to be wrong. In uncertain regimes (see macro section below), tighten
to 25–28%.

**Concentrated, high-conviction portfolio.** 8–15 positions max. Each position must clear a higher
bar than the one before it — adding position #9 requires believing it is better than everything
already held. Never diversify to reduce discomfort; diversify only when you genuinely cannot rank
your ideas.

**Hold unless the thesis breaks.** Volatility is not a reason to sell. The questions that matter:
(1) Is the moat still intact? (2) Is management still allocating capital well? (3) Is the original
thesis still valid? If yes to all three, hold. If any answer is no, sell — regardless of price.

**Temperament over analysis.** The biggest edge in long-term investing is not smarter models; it is
the discipline to buy when others are fearful, hold through volatility, and sell only when the
thesis breaks — not when the price falls.

## Decision Matrix

Use this matrix at the end of every research session before committing capital:

| Criterion                        | Threshold        | Weight |
|----------------------------------|------------------|--------|
| Moat quality (1–5)               | ≥ 3              | 30%    |
| Margin of safety                 | ≥ 20%            | 25%    |
| FCF yield                        | ≥ 4%             | 15%    |
| ROIC                             | ≥ 15%            | 15%    |
| Management quality (1–5)         | ≥ 3              | 15%    |

A stock clears the matrix if it meets ALL hard thresholds AND scores ≥ 70/100 on the weighted
composite. Any criterion below threshold is a hard veto regardless of composite score.

**Regime-adjusted thresholds**: In a risk-off regime (inverted yield curve + VIX > 25), tighten
margin of safety to 25–28% and require FCF yield ≥ 5%. In a risk-on regime, the base thresholds
apply. Check `get_decision_thresholds` each session — it returns the current regime-adjusted thresholds
so you never have to guess.

## Position Sizing

Use Kelly Criterion as a guide, not a mandate:

  f = (p × b − q) / b

where p = probability of thesis being right, b = upside/downside ratio, q = 1 − p.

- **High conviction** (moat score 4–5, MoS ≥ 30%, FCF yield ≥ 6%): 8–12% of portfolio
- **Medium conviction** (moat score 3, MoS 20–30%, FCF yield 4–6%): 4–8% of portfolio
- **Starter / monitoring position**: 2–4% of portfolio

Never exceed 15% in a single position. Size down in risk-off regimes (max 8% high conviction,
max 5% medium conviction). Call `get_conviction_position_size` for regime-adjusted Kelly outputs.

## Macro Overlay

Macro does not change what you buy — it changes how much you pay and how large a position you take.

**Yield curve**: Inverted (10Y < 2Y) = recession risk elevated; tighten MoS, reduce position sizes
  by 20–30%, build cash reserve. Flat/normal = baseline thresholds apply.
**VIX**: > 30 = fear spike — often the best buying opportunity for high-conviction ideas already
  on the watchlist. < 15 = complacency — be more selective, require wider MoS.
**Dollar (DXY)**: Rising dollar pressures multinationals and EM holdings. Favour domestic
  revenue businesses in a strong-dollar environment.
**Oil**: Rising oil raises input costs for non-energy businesses; headwind for consumer/transport.
  Benefit for energy holdings but avoid chasing cyclical peaks.
**Regime synthesis**: Combine yield curve + VIX to determine the current regime:
  - Risk-off (inverted curve AND VIX > 25): Tighten MoS to 25–28%, max position 8%, build cash.
  - Neutral (mixed signals): Apply base thresholds; hold existing positions.
  - Risk-on (normal curve AND VIX < 20): Base thresholds; deploy opportunistically.
  Always call `get_macro_environment` at session start to get the current regime assessment.

## Capital Allocation and Cash Management

Cash is a residual, not a target. You do not hold cash because you are uncertain about markets.
You hold cash because you cannot find businesses that clear the decision matrix at current prices.

**Deployment rules**:
1. Only deploy capital when a stock clears ALL decision matrix thresholds.
2. Do not force deployment to reduce cash — idle cash is better than a mediocre business.
3. Build cash by trimming or selling positions whose thesis has weakened, not by avoiding buys.
4. In a risk-off regime, hold more cash (target 15–25%) as a tactical buffer and as dry powder
   for the dislocations that follow.

**Cash reserve**: Always retain at least $500 in cash for transaction costs and unexpected
opportunities. Never be 100% invested.

## Behavioural Guardrails

These are rules, not guidelines. They exist because the biggest losses in long-term investing
come from behavioural errors, not analytical mistakes.

1. **Never chase a stock that has already re-rated.** If a stock has risen 30%+ since you first
   identified it and no longer offers margin of safety, it is off the buy list. Add a trigger
   to revisit if it pulls back.
2. **Never sell on price decline alone.** Price down ≠ thesis broken. Re-read the thesis.
   Ask: has the moat changed? Has management changed? If no, hold or add.
3. **No FOMO buys.** If you missed the entry, you missed it. Do not buy at full valuation
   because you are afraid of missing further gains.
4. **One research pass per ticker per session.** If you have already researched a ticker this
   session and decided to watchlist or pass, do not re-research it. Log it and move on.
5. **Watchlist before screener.** Before screening for new stocks, check the watchlist.
   You already researched those companies. If one is now at or near its target entry price,
   buy it before deploying capital in new, unresearched names.
6. **Sell discipline.** Sell when: (a) moat is impaired, (b) management has destroyed trust,
   (c) a materially better opportunity exists and you are at position limit, or (d) price has
   risen so far above intrinsic value that expected returns over 5 years are below 8% p.a.
   Do not sell just because a position is down.

## Moat Decay Checklist

Run this before adding to or holding any position > 6 months:
- [ ] Have switching costs weakened? (e.g. new open-source alternative, API commoditisation)
- [ ] Have network effects reversed? (e.g. user churn, competing network growing faster)
- [ ] Is regulatory protection at risk? (e.g. antitrust, licence review, new entrant approved)
- [ ] Has pricing power declined? (e.g. margin compression, lost contract renewals, discounting)
- [ ] Is management reinvesting in moat or harvesting it? (R&D/revenue trend, capex mix)
If two or more boxes are checked, the moat is at risk — treat as a watchlist item, not a hold.

---

## Sector Caps

**There are no sector caps.** Do not apply concentration limits based on sector. If the best
businesses at the best prices all happen to be in the same sector, buy them. Concentrate in
quality. Diversification by sector is not a virtue — it is a concession to uncertainty.

---
"""

def run_agent_session(
    initial_prompt: str,
    model: Optional[str] = None,
    initial_content: Optional[list] = None,
    **kwargs
) -> str:
    """Core agentic loop shared by all entry points."""

    clear_tool_cache()
    client = anthropic.Anthropic()
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")

    messages = []
    # Cache the main workflow prompt — it's static and the same every session,
    # so marking it ephemeral lets the API reuse the KV cache across calls.
    cached_prompt = [{"type": "text", "text": initial_prompt, "cache_control": {"type": "ephemeral"}}]
    # If initial_content is provided (e.g. user-flagged tickers), prepend it to the main prompt
    if initial_content:
        # initial_content is a list of content blocks; wrap in a "user" message
        messages.append({"role": "user", "content": initial_content})
        messages.append({"role": "user", "content": cached_prompt})
    else:
        messages.append({"role": "user", "content": cached_prompt})

    tools = TOOL_DEFINITIONS

    max_iterations = int(os.environ.get("MAX_ITERATIONS", "150"))
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # prune and cache
        messages = prune_messages(messages)

        # Stream the response so text is visible token-by-token in the terminal.
        # We accumulate tool_use blocks fully (they're not streamed as text) and
        # reconstruct the same assistant_content / tool_uses lists as before.
        assistant_content = []
        tool_uses = []
        stop_reason = None

        with client.messages.stream(
            model=model,
            max_tokens=16000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=add_cache_control(tools),
            messages=messages,
        ) as stream:
            current_text = ""
            for event in stream:
                event_type = type(event).__name__
                if event_type == "RawContentBlockStartEvent":
                    block = event.content_block
                    if getattr(block, "type", None) == "text":
                        current_text = ""
                        print()  # newline before each text block
                elif event_type == "RawContentBlockDeltaEvent":
                    delta = event.delta
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        print(delta.text, end="", flush=True)
                        current_text += delta.text
                elif event_type == "RawContentBlockStopEvent":
                    pass  # text accumulation done for this block

            # After streaming completes, get the final message for full block data
            final = stream.get_final_message()
            stop_reason = final.stop_reason

            for block in final.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_uses.append(block)
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

        messages.append({"role": "assistant", "content": assistant_content})

        if stop_reason == "end_turn":
            break

        if not tool_uses:
            break

        # execute tools — parallel when multiple tools are called in one turn
        tool_results = [None] * len(tool_uses)

        def _run_tool(idx: int, tool_use) -> None:
            print(f"  ⚙ {tool_use.name}")
            result = handle_tool_call(tool_use.name, tool_use.input)
            result_str = truncate_tool_result(tool_use.name, json.dumps(result, default=str))
            tool_results[idx] = {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result_str,
            }

        if len(tool_uses) == 1:
            _run_tool(0, tool_uses[0])
        else:
            with ThreadPoolExecutor(max_workers=min(len(tool_uses), 8)) as pool:
                futures = {pool.submit(_run_tool, i, tu): i for i, tu in enumerate(tool_uses)}
                for f in as_completed(futures):
                    f.result()  # propagate exceptions

        messages.append({"role": "user", "content": tool_results})

    return "Session complete"


def run_portfolio_review(model: Optional[str] = None, **kwargs) -> str:
    """Run a full autonomous portfolio review and rebalancing session."""
    prompt = f"""
Please conduct a comprehensive portfolio review and take appropriate investment actions:

**Step 1 — Load memory**
- Call `get_triaged_alerts` to check for thesis-breaking news. Any `thesis_breaking` alert on a held
  position must be investigated before new capital is deployed. `watch` alerts go on the review agenda.
  `noise` alerts can be noted and ignored.
- Call `get_investment_memory` to review past theses for current holdings and closed positions
- Call `get_session_reflections` to review lessons from past sessions. If the result contains
  `master_lessons`, read it carefully — it supersedes the individual reflections as the distilled
  wisdom of all past sessions. Note whether `sessions_covered` matches the current total — if not,
  update it at end of session.
- Call `get_active_triggers` — check for trade triggers set in previous sessions. For each:
  - `trigger_type=price_below/above`: the tool auto-fires matched triggers; surfaced ones in the result
    are already confirmed. Treat `action=research` as mandatory addition to Step 4 research list;
    `action=buy/add` as priority capital deployment; `action=sell` as an immediate position review.
  - `trigger_type=date/earnings_after`: date-based triggers also auto-fire; investigate the event now.
  - Cancel any triggered items that are no longer relevant (thesis changed, position already closed).
- Call `get_trade_outcomes` — review raw signal snapshots for all past trades
- Call `get_signal_performance` — binary threshold win-rate analysis of signals
- Call `get_ml_factor_weights` — continuous ML-derived factor weights that go beyond binary thresholds; use blended_weights and actionable_guidance when scoring screener candidates in Step 4
- Call `prioritize_watchlist_ml` — replaces plain get_watchlist; returns watchlist ranked by ML score with current fundamentals already fetched; start with rank-1 items when doing deep research
- Call `get_shadow_performance` — review stocks you previously passed on; note which passes were validated (stock fell) and which were mistakes (stock rose); apply lessons to this session's screening
- Call `connect_shadow_to_ml_training()` — this aggregates ALL shadow outcomes and generates per-factor learning signals. If validated_decisions > missed_opportunities for a factor, that factor is predictive and should be weighted higher in ML training. Use this to inform the next factor retraining cycle.
- Call `get_behaviour_summary` — load your own behaviour patterns from past sessions. Before
  doing anything else, read the averages and flags. If re_researched_watchlist > 0 in recent
  sessions, you have been wasting research budget — actively avoid repeating this. If
  deviated_from_matrix > 0, your prior decisions were not rules-based — correct this session.
  If workflow suggestions exist, apply them immediately without being asked.

**Step 2 — Assess current state**
- Check portfolio status (cash, holdings, P&L)
- Call `get_sector_exposure` — see current sector weights for situational awareness (no caps apply)
- Call `get_macro_environment` — market-price macro signals: yield curve, dollar, VIX, oil
- Call `get_economic_indicators` — real-economy macro signals: GDP growth, CPI, unemployment, consumer sentiment; synthesise with get_macro_environment for complete regime picture
- Call `get_benchmark_comparison` — are we beating the S&P 500? If not, why not?
- Call `get_portfolio_metrics` — review Sharpe ratio, max drawdown, volatility, and rolling 1/3/6-month returns vs S&P 500; if max drawdown > 15% or Sharpe < 0, tighten position sizing this session
- Check overall market index conditions

**Step 3 — Evaluate existing positions**

Quality-of-earnings check (CRITICAL — prevents holding deteriorating quality businesses):
- For each holding, call `score_earnings_quality(ticker)` — check the Sloan accrual ratio
  (red flag if >0.05). High accruals = earnings quality concern; upcoming disappointment likely.
- Call `score_piotroski_fscore(ticker)` — 9-point financial health check. Score <5 is weak;
  if score declined from last check, investigate cause. Declining F-Score predicts underperformance.
- Call `get_historical_valuation_range(ticker)` — is the stock now at the 90th+ percentile of
  its 5-year valuation? If yes, it's expensively priced; raises the bar for holding (thesis must
  be stronger, moat more durable).

Thesis integrity check:
- For each holding, ask the most important question first: **Is the moat still intact?** Has anything
  changed that erodes switching costs, network effects, or competitive barriers? If yes, that is a sell signal.
  Price decline alone is NOT a moat impairment — it may be a buying opportunity.
- Compare current fundamentals (FCF yield, ROIC, margins, retention) against the original buy thesis
- Check recent news (`get_stock_news`) for any thesis-breaking events
- Check upcoming earnings (`get_earnings_calendar`) to flag positions with imminent earnings risk
- Call `get_material_events(ticker, days=90)` for each holding — catch any 8-K filings (exec departures,
  impairments, auditor changes) that may have slipped past news feeds
- For each holding, **generate a fresh bear case**: What would a short-seller argue? What 3-4 key risks
  have you underestimated? What would break the thesis? Use `detect_financial_anomalies` to surface
  statistical red flags. If the moat or fundamentals have deteriorated, the bear case wins → sell.

**Step 3b — Work the watchlist BEFORE screening for new stocks**

This step is mandatory. Do it before Step 4. Watchlist items are already researched — buying one
is more capital-efficient than discovering and researching a new name.

Using the `prioritize_watchlist_ml` result from Step 1:
1. For any TRIGGERED or APPROACHING items (at or within 10% of target entry): re-read the original
   thesis from `get_investment_memory`, confirm the moat is intact, check for material events, and
   make a buy decision. If buying, execute immediately.
2. For rank-1 and rank-2 watchlist items (top ML scores): if not already bought, verify current
   price vs target entry — if within 20%, treat as a near-buy candidate and do a quick fundamentals
   refresh before deciding.
3. **Only after completing steps 1-2** proceed to Step 4. Step 4 is **always mandatory** — even if
   the watchlist consumed all available cash, you still screen the full universe to find new watchlist
   candidates. The screener builds the pipeline; it does not require cash to run.

**Step 4 — Discover new opportunities: screen the full S&P 500 AND the international universe**

This step is **always mandatory**, every session, regardless of cash available. It builds the
watchlist pipeline. Follow these steps exactly:

1. Call `get_stock_universe("sp500")` — returns all ~500 S&P 500 tickers.
2. Call `get_international_universe()` — returns ~200 major non-US tickers (ADRs, foreign-listed).
   **Call every session** — this is the only way to surface international opportunities.
3. Merge both lists and pass the combined tickers to `screen_stocks` **once**.
   `screen_stocks` automatically removes already-held, watchlisted, and shadow tickers internally —
   you do not need to call `filter_already_analyzed` first.
4. Call `auto_pass_screener_rejects` — ML-scores every screened ticker and auto-passes
   anything below 4.5/10 to shadow so it is never re-screened. Do this every session.

Apply ML-informed pre-filters from `get_ml_factor_weights` **before** sending tickers to
`research_stocks_parallel`. Only forward tickers that meet the ML-derived thresholds:
- Gross margin ≥ learned threshold (check `blended_weights`)
- FCF yield ≥ learned threshold
- Revenue growth positive
- Eliminate any ticker where `relative_momentum_pct` > 40 (already re-rated; no margin of safety)

After screening:
1. From the screener results, apply the ML pre-filters above. Drop any ticker below threshold.
2. From survivors, take the top 3–6 by score for deep research.
3. For each surviving ticker, verify it is NOT already in the portfolio, watchlist, or shadow
   portfolio (screen_stocks filters at call time, but state can change mid-session).
4. Only send the final survivors to `research_stocks_parallel`. Typical result: 25 → 18 → 14 survivors.

- Call `research_stocks_parallel` with the final pre-filtered tickers and their screener rows in `tickers_with_data`.

After research, for every stock researched this session:
- Call `log_prediction(ticker, action, conviction, ...)` — log every buy/watchlist/pass decision with full signal snapshot.
- `log_prediction` already records the thesis and conviction — no separate save needed.

**Step 4b — Challenge every buy recommendation before committing capital**

Before executing any buy, call `challenge_buy_theses(reports)` with all buy-recommended research
reports from `research_stocks_parallel`. This stress-tests the recommendations:
- Generates bear cases for each recommendation
- Identifies shared macro/sector risks across multiple buys
- Flags which recommendations are weakest (lowest conviction, most assumptions)

Only proceed with buys that survive the challenge. If a thesis cannot withstand a stress-test,
it is not ready — watchlist it instead.

**Step 5 — Take action**
- Execute buy/sell/watchlist decisions based on research findings
  - Before any buy: call `get_decision_thresholds` to confirm current regime-adjusted thresholds
  - Before any buy: call `get_conviction_position_size` to confirm correct Kelly-adjusted size
  - **Before any buy: call `check_portfolio_correlation(ticker)` — ensure new position isn't highly correlated
    with existing holdings.** If avg correlation > 0.6 or any pairwise > 0.7, size down or reject. This prevents
    building a hidden large concentration in a single theme (e.g., all high-multiple compounders that sell off together).
  - Only buy if the stock clears ALL decision matrix thresholds (hard veto on any below-threshold criterion)
  - Buy in one tranche unless size > 8% of portfolio, in which case split into 2–3 tranches over 2–3 weeks
- Set appropriate stop-losses and trade triggers for new positions
- For each trade trigger set, record the rationale in notes so future sessions understand why

**Step 5b — Cross-asset hedge review**
- If VIX > 25: consider protective puts on top 2 holdings (size: 1–2% of portfolio per hedge)
- If portfolio beta > 1.2: trim the highest-beta position by 20–30%
- If max drawdown from `get_portfolio_metrics` > 15%: reduce overall exposure by 10–15% (sell weakest conviction position first)

**Step 6 — Reflect and save**
- Call `save_session_reflection` with: lessons learned, what worked, what to improve, all tickers researched this session
- If there are ≥5 full reflections and no master document (or `sessions_covered` is stale): call `save_master_lessons`
- Call `save_session_audit` with a complete accounting of this session:
  - `tickers_screened`: total count passed to screen_stocks
  - `tickers_researched`: total count passed to research_stocks_parallel
  - `watchlist_added`: count of new watchlist additions
  - `trades_executed`: count of buy or sell orders placed
  - `cash_deployed`: total $ amount invested this session
  - `session_type`: "portfolio_review"
  - `re_researched_watchlist`: number of tickers you sent to research_stocks_parallel that were already on the watchlist (target: 0)
  - `reviewed_watchlist_before_screening`: 1 if you completed Step 3b before running the screener, 0 if not
  - `deviated_from_matrix`: 1 if you made any buy that did not clear ALL decision matrix thresholds, 0 if not
  - `workflow_suggestions`: any process improvements you noticed this session
  - `stocks_watchlisted`: list of tickers added to watchlist this session

**Accounting rules for save_session_audit (read carefully):**

1. `re_researched_watchlist` — how many tickers you sent to research_stocks_parallel were already
   on the watchlist BEFORE this session started? Count only tickers that were watchlisted at session
   start, not ones you added to the watchlist this session (those are new decisions, not re-research).
   Target is 0. Any value > 0 means you wasted research budget — note it in workflow_suggestions.

2. `reviewed_watchlist_before_screening` — did you complete Step 3b (evaluate every TRIGGERED and
   APPROACHING watchlist item in Step 3b before running the screener)? 1 = yes (correct), 0 = no (error to fix).

3. `deviated_from_matrix` — did you execute any buy order where one or more decision matrix criteria
   were below threshold? 1 = yes (a rules violation to correct next session), 0 = no (correct).
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_new_session(model: Optional[str] = None, **kwargs) -> str:
    """Alias kept for backwards compatibility."""
    return run_portfolio_review(model=model, **kwargs)


def run_quick_check(model: Optional[str] = None, **kwargs) -> str:
    """Run a focused position monitoring check without full portfolio research."""
    prompt = """
Please do a focused portfolio check:
1. Review current holdings and their recent performance
2. Check for any breaking news on held positions
3. Review any pending trade triggers
4. Make any urgent buy/sell decisions if warranted
5. Save a brief session reflection

Focus on monitoring and maintenance, not new stock discovery.
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_single_stock_research(ticker: str, model: Optional[str] = None, **kwargs) -> str:
    """Research a single stock in depth."""
    prompt = f"""
Please conduct deep research on {ticker}:
1. Get comprehensive fundamentals
2. Review recent earnings calls and SEC filings
3. Analyze insider activity and superinvestor positions
4. Assess competitive position and moat quality
5. Estimate intrinsic value
6. Make a buy/watchlist/pass recommendation with conviction score
7. Call `log_prediction` to record your investment thesis and decision
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_market_scan(tickers: list, model: Optional[str] = None, **kwargs) -> str:
    """Scan a list of stocks and identify best opportunities."""
    ticker_str = ", ".join(tickers)
    prompt = f"""
Please scan these stocks and identify the best opportunities: {ticker_str}

1. Screen all tickers for key metrics
2. Deep research the top 3-5 candidates
3. Make buy/watchlist/pass decisions
4. Execute any buys that meet criteria
5. Save decisions to memory
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_earnings_review(model: Optional[str] = None, **kwargs) -> str:
    """Review recent earnings for held positions and watchlist items."""
    prompt = """
Please conduct an earnings review session:
1. Get investment memory for context on held positions
2. Check recent earnings calls for all held positions
3. Check earnings calls for top watchlist items
4. Assess whether original theses still hold
5. Make any necessary portfolio adjustments
6. Save updated theses to memory
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_thesis_check(model: Optional[str] = None, **kwargs) -> str:
    """Check whether existing investment theses still hold."""
    prompt = """
Please validate all existing investment theses:
1. Review investment memory for all current holdings
2. For each holding, check: current fundamentals, recent news, SEC filings
3. Assess: Is the moat still intact? Is the thesis still valid?
4. Make sell decisions for any broken theses
5. Update investment memory with refreshed theses
6. Save session reflection
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_portfolio_review_with_ideas(ideas: list, model: Optional[str] = None, **kwargs) -> str:
    """Run portfolio review but also research specific user-provided ideas."""
    ideas_str = ", ".join(ideas)
    initial_content = [
        {
            "type": "text",
            "text": f"User has flagged these stocks for investigation this session: {ideas_str}. Prioritise them in Step 4 research alongside any screener finalists."
        }
    ]
    prompt = f"""
Please conduct a comprehensive portfolio review and take appropriate investment actions.
The user has flagged these stocks for prioritised research this session: {ideas_str}

Follow the standard portfolio review workflow but ensure these tickers are included in the
research phase (Step 4) regardless of screener score — the user has a specific interest in them.

**Step 1 — Load memory**
- Call `get_triaged_alerts` to check for thesis-breaking news
- Call `get_investment_memory` to review past theses
- Call `get_session_reflections` to review lessons from past sessions
- Call `get_active_triggers` — check for trade triggers
- Call `get_trade_outcomes` — review raw signal snapshots for all past trades
- Call `get_signal_performance` — binary threshold win-rate analysis of signals
- Call `get_ml_factor_weights` — continuous ML-derived factor weights
- Call `prioritize_watchlist_ml` — ML-ranked watchlist with current fundamentals
- Call `get_shadow_performance` — review stocks previously passed on
- Call `get_behaviour_summary` — load behaviour patterns from past sessions

**Step 2 — Assess current state**
- Check portfolio status (cash, holdings, P&L)
- Call `get_sector_exposure` — situational awareness only, no caps
- Call `get_macro_environment` — yield curve, dollar, VIX, oil
- Call `get_economic_indicators` — GDP, CPI, unemployment, consumer sentiment
- Call `get_benchmark_comparison` — vs S&P 500
- Call `get_portfolio_metrics` — Sharpe, max drawdown, rolling returns
- Check overall market index conditions

**Step 3 — Evaluate existing positions**
- Review each holding for moat integrity and thesis validity
- Check news, filings, and material events for each holding

**Step 3b — Work the watchlist BEFORE screening for new stocks**
- Using `prioritize_watchlist_ml` from Step 1, evaluate TRIGGERED/APPROACHING items first
- For rank-1 and rank-2 items, do a quick fundamentals refresh and buy if within 20% of target
- Only proceed to Step 4 after completing watchlist review

**Step 4 — Research user ideas + screen for new opportunities**
- Research the user's flagged tickers: {ideas_str}
- Also screen the full universe for additional opportunities
- Apply ML pre-filters, challenge buy theses, then execute

**Step 5 — Take action**
- Execute decisions, set stop-losses and triggers

**Step 6 — Reflect and save**
- Save session reflection and log session stats
"""
    return run_agent_session(prompt, model=model, initial_content=initial_content, **kwargs)


def run_shadow_review(model: Optional[str] = None, **kwargs) -> str:
    """Review shadow portfolio performance and extract lessons."""
    prompt = """
Please review the shadow portfolio and extract investment lessons:

1. Call get_shadow_performance to see how stocks we passed on have moved
2. For stocks that rose significantly after we passed: re-examine why we passed. Was the thesis wrong? Did we miss something?
3. For stocks that fell after we passed: validate the original reasoning
4. Update investment memory with any lessons learned
5. Consider whether any shadow portfolio stocks now offer a better entry point
6. Save a session reflection with specific lessons from the shadow portfolio analysis
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_learning_session(model: Optional[str] = None, **kwargs) -> str:
    """Dedicated session for reviewing ML signals and improving factor weights."""
    prompt = """
Please run a focused learning and calibration session:

1. Call get_trade_outcomes to review all past trade signal snapshots
2. Call get_signal_performance to see which factor thresholds have historically separated winners from losers
3. Call get_ml_factor_weights to review the current ML-derived factor importances
4. Call get_behaviour_summary to review your own decision patterns
5. Call get_shadow_performance to validate past pass decisions
6. Synthesise findings: which factors are most predictive in THIS portfolio's history?
7. Document specific threshold adjustments you will apply in the next portfolio review session
8. Save a session reflection capturing the key calibration insights

Focus entirely on learning and calibration — no new stock research or trades this session.
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_watchlist_review(model: Optional[str] = None, **kwargs) -> str:
    """Dedicated session for reviewing and acting on the watchlist."""
    prompt = """
Please run a focused watchlist review session:

1. Call get_investment_memory to review original theses for watchlist items
2. Call get_session_reflections to apply any relevant lessons
3. Call prioritize_watchlist_ml to get ML-ranked watchlist with current fundamentals
4. Call prune_watchlist to archive any entries >40% above target entry
5. For TRIGGERED and APPROACHING items: do a full thesis check (moat intact? fundamentals held?)
   and make buy/continue-watching/remove decisions
6. For remaining items: note whether thesis still holds or needs updating
7. Set new trade triggers for items approaching target entry
8. Save investment memory updates for any revised theses
9. Save session reflection

Focus entirely on watchlist management — no new stock discovery this session.
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_staleness_review(model: Optional[str] = None, **kwargs) -> str:
    """Refresh stale watchlist entries that haven't been evaluated in 60+ days."""
    prompt = """
Please run a focused staleness review session:

1. Call check_watchlist_staleness to identify entries not re-evaluated in 60+ days
2. Call get_investment_memory to load original theses for stale entries
3. For each stale entry, run a quick research refresh:
   - get_stock_fundamentals (has moat/thesis held?)
   - get_stock_news (any thesis-breaking events?)
   - get_material_events (any 8-K red flags?)
4. Make a keep / update-thesis / remove decision for each stale entry
5. For keeps: update target entry price if fundamentals have changed; save refreshed thesis
6. For removes: call remove_from_watchlist with reason
7. Log decisions with log_prediction
8. Save session reflection capturing what you learned

Focus entirely on refreshing stale entries — no new stock discovery this session.
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_custom_prompt(prompt: str, model: Optional[str] = None, initial_content: Optional[list] = None, **kwargs) -> str:
    """Run a custom prompt."""
    return run_agent_session(prompt, model=model, initial_content=initial_content, **kwargs)
