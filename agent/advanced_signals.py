"""
Advanced investment signals: management quality, momentum analysis, short thesis monitoring.
"""

import yfinance as yf
from typing import Optional


def analyze_management_compensation(ticker: str) -> dict:
    """
    Analyze CEO/management ownership, compensation structure, and capital allocation discipline.
    Uses publicly available info from yfinance (limited scope without full SEC parsing).

    Returns: ownership %, compensation signals, dilution rate, and quality assessment.
    """
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # Available from yfinance (limited):
        # - percentInsiders: insider ownership %
        # - insiderHoldingsPercent: similar
        insider_ownership = info.get("insiderHoldingsPercent") or info.get("percentInsiders") or 0

        # Dilution proxy: YoY change in shares outstanding
        bs = t.balance_sheet
        shares_curr = info.get("sharesOutstanding")

        # Estimate dilution from financials if available
        dilution_pct = None
        if bs is not None and len(bs.columns) > 1:
            try:
                col_curr = bs.columns[0]
                col_prev = bs.columns[1]
                if "Common Stock Shares Outstanding" in bs.index:
                    shares_prev = bs.loc["Common Stock Shares Outstanding"].iloc[1]
                    if shares_prev and shares_prev > 0:
                        dilution_pct = (shares_curr - shares_prev) / shares_prev * 100 if shares_curr else None
            except Exception:
                pass

        # Dividends + buybacks signal capital allocation
        div_per_share = info.get("dividends") or info.get("trailingAnnualDividendRate")
        div_yield = info.get("dividendYield")

        # Assessment
        quality_flag = "green"
        concerns = []
        strengths = []

        if insider_ownership is not None:
            if insider_ownership > 0.10:
                strengths.append(f"+Insider ownership {insider_ownership*100:.1f}% — management has skin in the game")
                quality_flag = "green"
            elif insider_ownership > 0.01:
                pass  # neutral
            else:
                concerns.append(f"-Minimal insider ownership {insider_ownership*100:.2f}% — management may lack conviction")
                quality_flag = "yellow"

        if dilution_pct is not None and dilution_pct > 0.05:
            quality_flag = "yellow"
            concerns.append(f"-Dilution {dilution_pct:.1f}% YoY — aggressive stock-based comp or capital raises")
        elif dilution_pct is not None and dilution_pct < 0:
            strengths.append(f"+Net share buyback {abs(dilution_pct):.1f}% — management returning capital")

        if div_yield is not None and div_yield > 0.02:
            strengths.append(f"+Dividend yield {div_yield*100:.1f}% — visible capital return")

        return {
            "ticker": ticker,
            "insider_ownership_pct": round(insider_ownership * 100, 2) if insider_ownership else None,
            "yoy_dilution_pct": round(dilution_pct, 2) if dilution_pct is not None else None,
            "dividend_yield_pct": round(div_yield * 100, 2) if div_yield else None,
            "quality_flag": quality_flag,
            "strengths": strengths,
            "concerns": concerns,
            "note": (
                "Green: good management alignment and discipline. Yellow: potential misalignment or concerns. "
                "For full compensation analysis (options vs restricted stock, golden parachutes, claw-backs), "
                "read the DEF 14A proxy statement from the SEC Edgar database."
            ),
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"Management analysis failed: {e}"}


def analyze_momentum_acceleration(ticker: str) -> dict:
    """
    Detect second-derivative shifts in revenue/margin growth (acceleration vs deceleration).

    A company with slowing growth (20% → 15% → 10%) often re-rates downward even if 10%
    growth is respectable. Conversely, accelerating growth compounds value creation.

    Returns: growth acceleration direction, trend confidence, and valuation implications.
    """
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        fin = t.financials

        if fin is None or fin.empty or len(fin.columns) < 4:
            return {"ticker": ticker, "error": "Insufficient quarterly/annual history for acceleration analysis"}

        # Get revenue trend (most recent 4 years)
        rev_data = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else fin.loc["Revenue"]
        rev_4yr = [float(v) for v in rev_data.iloc[:4].values if v is not None and str(v) != 'nan']

        if len(rev_4yr) < 4:
            return {"ticker": ticker, "error": "Insufficient revenue data"}

        rev_4yr.reverse()  # oldest to newest

        # Calculate YoY growth rates
        growth_rates = []
        for i in range(len(rev_4yr) - 1):
            if rev_4yr[i] > 0:
                growth = (rev_4yr[i+1] - rev_4yr[i]) / rev_4yr[i]
                growth_rates.append(growth)

        if len(growth_rates) < 3:
            return {"ticker": ticker, "error": "Not enough data for acceleration analysis"}

        # Second derivative: change in growth rate
        acceleration = []
        for i in range(len(growth_rates) - 1):
            accel = growth_rates[i+1] - growth_rates[i]
            acceleration.append(accel)

        avg_accel = sum(acceleration) / len(acceleration)

        # Trend classification
        if avg_accel > 0.05:  # growth accelerating by 5%+ per year
            trend = "ACCELERATING"
            interpretation = "Growth is inflecting upward. Often re-rates higher."
        elif avg_accel < -0.05:  # growth decelerating by 5%+ per year
            trend = "DECELERATING"
            interpretation = "Growth is slowing. Often re-rates lower, even if absolute growth is positive."
        else:
            trend = "STABLE"
            interpretation = "Growth rate relatively stable."

        return {
            "ticker": ticker,
            "growth_rates_pct": [round(g * 100, 1) for g in growth_rates],
            "acceleration_rates": [round(a * 100, 1) for a in acceleration],
            "trend": trend,
            "interpretation": interpretation,
            "valuation_implication": (
                "Accelerating → upside to multiples expansion. Decelerating → downside risk even if "
                "absolute growth remains healthy. Use to calibrate bull/bear case scenarios."
            ),
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"Momentum acceleration analysis failed: {e}"}


def get_short_seller_thesis_risks(ticker: str) -> dict:
    """
    Surface common short-seller narrative themes and check if they apply to this ticker.
    Does NOT fetch actual short reports (would require web scraping/external API).
    Instead, provides a checklist of common short thesis patterns to verify manually.

    Returns: common risk patterns, degree-of-concern flags, and research steps.
    """
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info
        fin = t.financials
        bs = t.balance_sheet

        if not info:
            return {"ticker": ticker, "error": "Cannot fetch company info"}

        concerns = []
        strength_score = 0  # higher = less vulnerable to short thesis

        # ── Common Short Thesis Patterns ──────────────────────────────────────

        # 1. Accounting tricks / accrual buildup
        # (would be detected by score_earnings_quality; note it here)
        concerns.append({
            "pattern": "Accounting red flags (high accruals, divergence between NI and CFO)",
            "check": "Call score_earnings_quality(). If Sloan accrual > 0.05, this is a real short vulnerability.",
        })

        # 2. Debt accumulation without revenue growth
        if bs is not None and len(bs.columns) > 1:
            debt_curr = bs.loc["Total Debt"].iloc[0] if "Total Debt" in bs.index else None
            debt_prev = bs.loc["Total Debt"].iloc[1] if "Total Debt" in bs.index and len(bs.columns) > 1 else None
            rev_curr = fin.loc["Total Revenue"].iloc[0] if "Total Revenue" in fin.index else None
            rev_prev = fin.loc["Total Revenue"].iloc[1] if "Total Revenue" in fin.index and len(fin.columns) > 1 else None

            if all([debt_curr, debt_prev, rev_curr, rev_prev]):
                debt_growth = (debt_curr - debt_prev) / debt_prev if debt_prev > 0 else 0
                rev_growth = (rev_curr - rev_prev) / rev_prev if rev_prev > 0 else 0
                if debt_growth > rev_growth and debt_growth > 0.10:
                    concerns.append({
                        "pattern": "Debt growing faster than revenue (bad capital allocation)",
                        "degree": "HIGH" if debt_growth > 0.20 else "MEDIUM",
                        "check": "Investigate acquisition financing, capex discipline, covenant compliance.",
                    })

        # 3. Revenue concentration (customer concentration)
        # Can't directly detect from yfinance; ask the user to check 10-K risk section
        concerns.append({
            "pattern": "Revenue concentration (>25% from one customer)",
            "check": "Read 10-K Item 1.A risk factors for customer concentration disclosures.",
        })

        # 4. Gross/operating margin compression
        if fin is not None and "Gross Profit" in fin.index:
            gp_curr = fin.loc["Gross Profit"].iloc[0]
            rev_curr = fin.loc["Total Revenue"].iloc[0]
            if len(fin.columns) > 1:
                gp_prev = fin.loc["Gross Profit"].iloc[1]
                rev_prev = fin.loc["Total Revenue"].iloc[1]
                if all([gp_curr, rev_curr, gp_prev, rev_prev]):
                    gm_curr = gp_curr / rev_curr
                    gm_prev = gp_prev / rev_prev
                    if gm_curr < gm_prev * 0.95:
                        concerns.append({
                            "pattern": "Gross margin compressing >5% (loss of pricing power or rising input costs)",
                            "degree": "MEDIUM",
                            "check": "Investigate: product mix shift, competition, supply chain costs.",
                        })

        # 5. Executive departures
        # Limited visibility via yfinance; news would surface this
        concerns.append({
            "pattern": "Recent executive departures (especially CFO or COO)",
            "check": "Call get_material_events() and get_stock_news() for leadership changes.",
        })

        # 6. Regulatory / legal risks
        concerns.append({
            "pattern": "Pending litigation, regulatory investigation, licensing risk",
            "check": "Read 10-K Item 3 legal proceedings; search SEC Edgar for current litigation.",
        })

        return {
            "ticker": ticker,
            "short_thesis_vulnerability_check": concerns,
            "next_steps": (
                "1. For each pattern above, verify if it applies to this ticker. "
                "2. If ANY pattern shows HIGH concern, treat it as a hard sell signal. "
                "3. If MEDIUM concerns outnumber strengths, demand a very wide margin of safety. "
                "4. To review actual short reports, search Hindenburg Research, Muddy Waters, or "
                "use the FactSet short research database."
            ),
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"Short thesis analysis failed: {e}"}
