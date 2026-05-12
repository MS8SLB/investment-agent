"""
Structured thesis assumption tracking and shadow portfolio learning.

When a thesis is logged, assumptions are tagged for later verification.
Shadow outcomes (what happened to stocks we passed) feed back into ML factor weights.
"""

import json
from typing import Optional
from agent import portfolio


def structure_thesis_assumptions(ticker: str, thesis_text: str) -> dict:
    """
    Parse a thesis string and extract key assumptions into structured form.
    Used when logging predictions to enable per-assumption verification later.

    Returns: structured dict of assumptions with type, assumption, threshold, catalysts.
    """
    assumptions = {
        "ticker": ticker.upper(),
        "raw_thesis": thesis_text,
        "structured_assumptions": [],
    }

    # Common assumption patterns to extract
    assumption_patterns = [
        {
            "pattern": "growth",
            "keywords": ["revenue growth", "cagr", "growth rate", "expand", "growth to"],
            "type": "growth_rate",
        },
        {
            "pattern": "margin",
            "keywords": ["margin", "operating margin", "gross margin", "margin expansion", "margin compression"],
            "type": "margin_assumption",
        },
        {
            "pattern": "multiple",
            "keywords": ["multiple", "pe ratio", "ev/ebitda", "fcf multiple", "re-rate"],
            "type": "valuation_multiple",
        },
        {
            "pattern": "moat",
            "keywords": ["moat", "switching cost", "network effect", "competitive advantage", "defensible"],
            "type": "moat_durability",
        },
        {
            "pattern": "management",
            "keywords": ["management", "capital allocation", "ceo", "leadership", "execution"],
            "type": "management_quality",
        },
        {
            "pattern": "catalyst",
            "keywords": ["catalyst", "trigger", "event", "acquisition", "product launch", "announcement"],
            "type": "catalyst_timing",
        },
        {
            "pattern": "market",
            "keywords": ["market size", "tam", "addressable market", "penetration", "market share"],
            "type": "market_opportunity",
        },
    ]

    thesis_lower = thesis_text.lower()

    for pattern in assumption_patterns:
        for keyword in pattern["keywords"]:
            if keyword in thesis_lower:
                # Found a mention — extract approximate statement
                # (Simple heuristic: find sentences containing the keyword)
                sentences = thesis_text.split(".")
                relevant = [s.strip() for s in sentences if keyword in s.lower()]
                if relevant:
                    assumptions["structured_assumptions"].append({
                        "type": pattern["type"],
                        "keyword_triggered_by": keyword,
                        "extracted_statement": relevant[0][:150],  # first 150 chars
                        "verification_needed": True,
                        "notes": f"Verify this assumption in next research session. {pattern['pattern'].title()} assumptions are critical to thesis.",
                    })
                    break  # Only one assumption per pattern to avoid duplication

    # Persist structured assumptions to DB for future verification sessions
    if assumptions["structured_assumptions"]:
        try:
            portfolio.save_thesis_assumptions(ticker, assumptions["structured_assumptions"])
            assumptions["saved_to_db"] = len(assumptions["structured_assumptions"])
        except Exception as e:
            assumptions["save_error"] = str(e)

    return assumptions


def score_shadow_portfolio_learning(ticker: str, original_action: str, actual_outcome: dict) -> dict:
    """
    Analyze a shadow portfolio position (stock we passed on or watchlisted but didn't buy).
    Determine whether our decision was validated (stock fell, thesis was right to be cautious)
    or a miss (stock rose significantly, we were too conservative).

    Returns: learning signal to adjust future ML factor weights.
    """
    if original_action not in ["pass", "watchlist"]:
        return {"error": "Only pass and watchlist decisions generate shadow learning"}

    outcome_price = actual_outcome.get("current_price")
    entry_price = actual_outcome.get("entry_price")  # price when we passed/watchlisted
    original_thesis = actual_outcome.get("original_thesis", "")

    if not entry_price or not outcome_price:
        return {"error": "Need both entry and outcome prices"}

    pct_change = (outcome_price - entry_price) / entry_price * 100

    # Classification
    if original_action == "pass":
        if pct_change < -10:  # Fell >10%: we were right to pass
            learning = "VALIDATED_PASS"
            signal_strength = "strong"
            insight = "Stock fell as we predicted. Pass decision was correct. Reinforce the factors that flagged this risk."
        elif pct_change > 30:  # Rose >30%: we missed it
            learning = "MISSED_OPPORTUNITY"
            signal_strength = "strong"
            insight = "Stock rose sharply after we passed. Analyze what we missed in the thesis. Adjust factor weights."
        else:
            learning = "NEUTRAL_PASS"
            signal_strength = "weak"
            insight = "Stock moved in a normal range. Pass decision neither validated nor contradicted."
    else:  # watchlist
        if pct_change < -15:  # Fell >15%: watchlist was correct (we didn't buy)
            learning = "VALIDATED_WATCHLIST"
            signal_strength = "strong"
            insight = "Stock fell as expected. Watchlist decision to wait for better entry was correct."
        elif pct_change > 40:  # Rose >40%: we were too cautious
            learning = "MISSED_APPRECIATION"
            signal_strength = "strong"
            insight = "Stock appreciated significantly after we watchlisted it. We were too cautious on valuation."
        else:
            learning = "NEUTRAL_WATCHLIST"
            signal_strength = "weak"
            insight = "Stock moved moderately. Watchlist decision neither clearly right nor wrong."

    return {
        "ticker": ticker.upper(),
        "original_action": original_action,
        "entry_price": entry_price,
        "outcome_price": outcome_price,
        "pct_change": round(pct_change, 1),
        "learning_signal": learning,
        "signal_strength": signal_strength,
        "insight": insight,
        "ml_implication": (
            f"For ML factor weights: {insight} "
            f"Review the factors that were below-threshold for this ticker. "
            f"If {'low' if pct_change < 0 else 'high'} momentum is a pattern in {learning}, "
            f"adjust momentum weight {'down' if pct_change < 0 else 'up'} in next training cycle."
        ),
    }


def connect_shadow_to_ml_training() -> dict:
    """
    Aggregate all shadow portfolio outcomes and compute learning signals to feed into
    ML factor retraining. This bridges the gap between closed trades (which train factors)
    and pass/watchlist decisions (which test factors).

    Returns: per-factor learning signals and recommended weight adjustments.
    """
    try:
        # Get all shadow portfolio positions
        shadow_positions = portfolio.get_shadow_positions()

        if not shadow_positions:
            return {"note": "No shadow portfolio history yet. Learning will begin after first watchlist/pass outcomes."}

        # For each shadow position, score the outcome
        factor_signals = {
            "gross_margin": [],
            "fcf_yield": [],
            "relative_momentum": [],
            "revenue_growth": [],
            "profit_margin": [],
            "roe": [],
        }

        learning_summary = {
            "total_shadow_positions_analyzed": 0,
            "validated_decisions": 0,
            "missed_opportunities": 0,
            "factor_signals": factor_signals,
            "recommended_weight_adjustments": {},
        }

        for shadow in shadow_positions:
            ticker = shadow.get("ticker")
            action = shadow.get("action")
            entry_price = shadow.get("price_at_entry")
            current_price = shadow.get("current_price")
            screener_data = shadow.get("screener_data", {})

            if not all([ticker, action, entry_price, current_price]):
                continue

            pct_move = (current_price - entry_price) / entry_price

            learning_summary["total_shadow_positions_analyzed"] += 1

            # Map outcome to factor signals
            # Example: if we passed on a stock and it fell (good pass), signal that the factors
            # that flagged it (e.g., low FCF yield) were predictive.

            if action == "pass" and pct_move < -0.10:
                learning_summary["validated_decisions"] += 1
                # The factors that made us pass were correct
                if screener_data.get("fcf_yield_pct", 0) < 4:
                    factor_signals["fcf_yield"].append({"ticker": ticker, "outcome": "validated_low_yield_pass"})
            elif action == "watchlist" and pct_move > 0.40:
                learning_summary["missed_opportunities"] += 1
                # We were too conservative
                if screener_data.get("revenue_growth_pct", 0) > 20:
                    factor_signals["revenue_growth"].append({"ticker": ticker, "outcome": "missed_high_growth"})

        # Compute recommended weight adjustments
        for factor, signals in factor_signals.items():
            if not signals:
                continue
            validated = sum(1 for s in signals if "validated" in s.get("outcome", ""))
            missed = sum(1 for s in signals if "missed" in s.get("outcome", ""))

            if validated > missed:
                learning_summary["recommended_weight_adjustments"][factor] = "increase_weight"
            elif missed > validated:
                learning_summary["recommended_weight_adjustments"][factor] = "decrease_weight"

        return learning_summary

    except Exception as e:
        return {"error": f"Shadow portfolio learning failed: {e}"}
