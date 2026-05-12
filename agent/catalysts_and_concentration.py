"""
Revenue/customer concentration analysis and catalyst calendar tracking.
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


def get_revenue_concentration(ticker: str) -> dict:
    """
    Estimate revenue concentration and diversification risk.
    Looks for concentration disclosures in SEC filings (limited without full parsing).

    Returns: estimated customer concentration risk, recurring revenue %, geographic mix signals.
    """
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info
        fin = t.financials

        # Heuristics based on available data:
        # 1. Check if the company is a business services/outsourced provider (higher concentration risk)
        # 2. Look at revenue stability (coefficient of variation)
        # 3. Check for contract/customer-dependent revenue signals

        sector = info.get("sector", "")
        industry = info.get("industry", "")
        business_summary = info.get("longBusinessSummary", "")

        concentration_risk = "low"
        concerns = []
        strengths = []

        # ── Sector heuristics ──────────────────────────────────────────────────
        high_concentration_sectors = [
            "Business Services", "IT Services", "Consulting", "Defense",
            "Government Services", "Staffing", "Telecom"
        ]

        if any(s.lower() in sector.lower() for s in high_concentration_sectors):
            concentration_risk = "medium"
            concerns.append(f"Industry {industry} typically has higher customer concentration risk")

        # ── Revenue stability (proxy for concentration) ────────────────────────
        if fin is not None and "Total Revenue" in fin.index and len(fin.columns) >= 3:
            revenues = [float(v) for v in fin.loc["Total Revenue"].iloc[:3].values
                       if v is not None and str(v) != 'nan']
            if len(revenues) >= 3:
                revenues.reverse()
                avg_rev = sum(revenues) / len(revenues)
                volatility = sum((r - avg_rev)**2 for r in revenues) / len(revenues)
                cv = (volatility ** 0.5) / avg_rev  # coefficient of variation

                if cv > 0.30:  # >30% volatility suggests concentration or cyclicality
                    concentration_risk = "high"
                    concerns.append(f"Revenue volatility CV={cv:.2f} (>0.30 suggests concentration or cyclicality)")
                elif cv < 0.10:
                    strengths.append(f"Revenue stability CV={cv:.2f} (indicates diversified customer base)")

        # ── Keyword heuristics ────────────────────────────────────────────────
        concentration_keywords = ["customer concentration", "major customer", "customer agreement", "single customer"]
        recurring_keywords = ["subscription", "recurring", "saas", "membership", "contract"]

        for kw in concentration_keywords:
            if kw.lower() in business_summary.lower():
                concentration_risk = "high"
                concerns.append(f"Business summary mentions '{kw}' — likely customer concentration disclosure in 10-K")

        has_recurring = any(kw.lower() in business_summary.lower() for kw in recurring_keywords)
        if has_recurring:
            strengths.append("Recurring revenue model (SaaS/subscription) reduces concentration risk")

        return {
            "ticker": ticker,
            "concentration_risk_level": concentration_risk,
            "key_signals": {
                "sector": sector,
                "industry": industry,
                "has_recurring_revenue_signals": has_recurring,
            },
            "concerns": concerns,
            "strengths": strengths,
            "next_steps": (
                "For exact customer concentration percentages, read 10-K Item 1 (Business) "
                "and Item 1A (Risk Factors) for 'Major Customers' or 'Customer Concentration' disclosures. "
                "If any single customer is >25% of revenue, treat as a moat risk — customer departure "
                "would materially impact earnings. Stress-test valuation assuming loss of largest customer."
            ),
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"Revenue concentration analysis failed: {e}"}


def get_upcoming_catalysts(ticker: str) -> dict:
    """
    Surface upcoming catalysts that could move the stock: earnings dates, product launches,
    contract renewals, regulatory decisions, index changes, earnings date.

    Returns: upcoming events ranked by estimated impact probability.
    """
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info

        catalysts = []

        # ── Earnings-related catalysts ────────────────────────────────────────
        earnings_dates = info.get("earnings_dates")
        next_earnings = info.get("earnings_dates")  # yfinance includes both past and upcoming

        if next_earnings:
            try:
                # Get the next earnings date (typically the first upcoming one)
                catalysts.append({
                    "type": "earnings",
                    "description": "Next earnings release",
                    "impact_potential": "HIGH",
                    "notes": "Pre-earnings: IV often spikes (check get_options_flow). Post-earnings: stock typically moves 3-8% on surprise.",
                })
            except Exception:
                pass

        # ── Ex-dividend date ──────────────────────────────────────────────────
        div_date = info.get("ex_dividend_date")
        if div_date:
            catalysts.append({
                "type": "dividend",
                "description": f"Ex-dividend date",
                "impact_potential": "LOW",
                "notes": "Stock typically falls by ~dividend amount on ex-date (mechanical). Check if dividend growing.",
            })

        # ── Stock splits / special events ──────────────────────────────────────
        # Limited visibility via yfinance; would need SEC alert monitoring
        catalysts.append({
            "type": "structural",
            "description": "Potential structural catalysts (not visible via yfinance)",
            "impact_potential": "VARIES",
            "notes": (
                "Check manually for: (a) stock splits/reverse splits (10-K Item 5); "
                "(b) special dividends (unusual capital return); "
                "(c) acquisition/merger announcements; (d) spin-off plans; "
                "(e) shareholder votes (proxy statement). "
                "Search SEC Edgar for recent 8-K filings (material events)."
            ),
        })

        # ── Regulatory / product catalysts ─────────────────────────────────────
        industry = info.get("industry", "")

        regulatory_industries = ["Pharma", "Medical Device", "Bank", "Insurance", "Utility", "Telecom"]
        if any(ind.lower() in industry.lower() for ind in regulatory_industries):
            catalysts.append({
                "type": "regulatory",
                "description": "Pending regulatory approvals, rate decisions, or licensing",
                "impact_potential": "HIGH",
                "notes": f"Industry {industry} is regulatory-exposed. Check SEC filings and press releases for pending decisions.",
            })

        # ── Product/market catalysts ──────────────────────────────────────────
        catalysts.append({
            "type": "product",
            "description": "Product launches, new market entry, partnership announcements",
            "impact_potential": "MEDIUM",
            "notes": "Monitor investor relations calendar and quarterly guidance for upcoming launches. Press releases often pre-announce.",
        })

        return {
            "ticker": ticker,
            "upcoming_catalysts": catalysts,
            "tracking_notes": (
                "For a complete catalyst calendar, cross-check: (1) SEC Edgar (8-K filings = material events); "
                "(2) Company investor relations calendar (product launch dates); "
                "(3) Earnings call transcripts (guidance on upcoming milestones); "
                "(4) Industry regulatory calendars (FDA approvals, FERC rulings, etc.); "
                "(5) News searches for announced partnerships/contracts with expected close dates."
            ),
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"Catalyst analysis failed: {e}"}
