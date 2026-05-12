"""
Advanced investment signals: management quality, momentum analysis, short thesis monitoring.
"""

import yfinance as yf
from typing import Optional


def _fetch_xbrl_facts(cik: str) -> Optional[dict]:
    """Fetch XBRL company facts from SEC EDGAR (free, no API key required)."""
    import requests
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {"User-Agent": "investment-agent-research sec-data@example.com"}
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def _xbrl_latest(facts: dict, concept: str, namespace: str = "us-gaap") -> Optional[float]:
    """Extract the most recent annual value for an XBRL concept."""
    try:
        units = facts.get("facts", {}).get(namespace, {}).get(concept, {}).get("units", {})
        # Annual 10-K filings use form '10-K'
        for unit_key in ("USD", "shares", "pure"):
            entries = units.get(unit_key, [])
            annual = [e for e in entries if e.get("form") in ("10-K", "20-F") and e.get("val") is not None]
            if annual:
                annual.sort(key=lambda e: e.get("end", ""), reverse=True)
                return float(annual[0]["val"])
        return None
    except Exception:
        return None


def _xbrl_annual_series(facts: dict, concept: str, namespace: str = "us-gaap", n: int = 3) -> list:
    """Extract the last n annual values for an XBRL concept, newest first."""
    try:
        units = facts.get("facts", {}).get(namespace, {}).get("units", {})
        for unit_key in ("USD", "shares", "pure"):
            entries = facts.get("facts", {}).get(namespace, {}).get(concept, {}).get("units", {}).get(unit_key, [])
            annual = [e for e in entries if e.get("form") in ("10-K", "20-F") and e.get("val") is not None]
            if annual:
                annual.sort(key=lambda e: e.get("end", ""), reverse=True)
                # Deduplicate by fiscal year end date (take first per date)
                seen = set()
                result = []
                for e in annual:
                    d = e.get("end", "")[:7]  # YYYY-MM
                    if d not in seen:
                        seen.add(d)
                        result.append(float(e["val"]))
                    if len(result) >= n:
                        break
                return result
        return []
    except Exception:
        return []


def analyze_management_compensation(ticker: str) -> dict:
    """
    Analyze CEO/management ownership, compensation structure, and capital allocation discipline.

    Primary source: SEC EDGAR XBRL company facts (free API, no key required).
    Extracts: CEO total compensation, stock-based compensation as % of revenue,
    share dilution trend, insider ownership from DEF 14A proxy data.
    Falls back to yfinance for ownership and dilution when XBRL lacks proxy data.

    Returns: ownership %, SBC/revenue %, dilution trend, and quality assessment.
    """
    ticker = ticker.upper()
    try:
        # ── Step 1: Get CIK for EDGAR lookup ──────────────────────────────────
        from agent.sec_data import _get_cik
        cik = _get_cik(ticker)
        xbrl_data = None
        if cik:
            xbrl_data = _fetch_xbrl_facts(cik)

        # ── Step 2: EDGAR XBRL signals ────────────────────────────────────────
        sbc_series = []          # stock-based compensation history
        revenue_series = []      # revenue history for SBC/revenue ratio
        shares_series = []       # shares outstanding history for dilution
        exec_comp = None         # executive total compensation (XBRL: ExecutiveCompensation)

        if xbrl_data:
            sbc_series = _xbrl_annual_series(xbrl_data, "ShareBasedCompensation", n=3)
            revenue_series = _xbrl_annual_series(xbrl_data, "Revenues", n=3)
            if not revenue_series:
                revenue_series = _xbrl_annual_series(xbrl_data, "RevenueFromContractWithCustomerExcludingAssessedTax", n=3)
            shares_series = _xbrl_annual_series(xbrl_data, "CommonStockSharesOutstanding", n=3)
            # DEI namespace often has ExecutiveCompensationAmount
            exec_comp = _xbrl_latest(xbrl_data, "ExecutiveCompensationAmount", namespace="us-gaap")
            if exec_comp is None:
                exec_comp = _xbrl_latest(xbrl_data, "DefinedBenefitPlanContributionsByEmployer", namespace="us-gaap")

        # ── Step 3: yfinance fallback data ────────────────────────────────────
        t = yf.Ticker(ticker)
        info = t.info or {}
        insider_ownership = info.get("insiderHoldingsPercent") or info.get("percentInsiders") or 0
        shares_curr = info.get("sharesOutstanding")
        div_yield = info.get("dividendYield")

        # ── Step 4: Compute derived signals ──────────────────────────────────
        # SBC as % of revenue (most recent year)
        sbc_pct = None
        if sbc_series and revenue_series and revenue_series[0] > 0:
            sbc_pct = sbc_series[0] / revenue_series[0] * 100

        # SBC trend (accelerating = concern)
        sbc_trend = None
        if len(sbc_series) >= 2 and sbc_series[1] > 0:
            sbc_change = (sbc_series[0] - sbc_series[1]) / sbc_series[1] * 100
            sbc_trend = round(sbc_change, 1)

        # Share dilution from XBRL (more accurate than yfinance)
        dilution_pct = None
        if len(shares_series) >= 2 and shares_series[1] > 0:
            dilution_pct = (shares_series[0] - shares_series[1]) / shares_series[1] * 100
        elif shares_curr:
            # yfinance fallback: compare to balance sheet
            try:
                bs = t.balance_sheet
                if bs is not None and len(bs.columns) > 1 and "Common Stock Shares Outstanding" in bs.index:
                    shares_prev = float(bs.loc["Common Stock Shares Outstanding"].iloc[1])
                    if shares_prev > 0:
                        dilution_pct = (shares_curr - shares_prev) / shares_prev * 100
            except Exception:
                pass

        # ── Step 5: Quality assessment ────────────────────────────────────────
        quality_flag = "green"
        concerns = []
        strengths = []

        if insider_ownership:
            if insider_ownership > 0.10:
                strengths.append(f"+Insider ownership {insider_ownership*100:.1f}% — management has skin in the game")
            elif insider_ownership < 0.01:
                concerns.append(f"-Minimal insider ownership {insider_ownership*100:.2f}% — weak alignment")
                quality_flag = "yellow"

        if sbc_pct is not None:
            if sbc_pct > 10:
                quality_flag = "red"
                concerns.append(f"-SBC {sbc_pct:.1f}% of revenue — very aggressive stock compensation; shareholders paying")
            elif sbc_pct > 5:
                if quality_flag == "green":
                    quality_flag = "yellow"
                concerns.append(f"-SBC {sbc_pct:.1f}% of revenue — elevated stock-based compensation")
            else:
                strengths.append(f"+SBC {sbc_pct:.1f}% of revenue — disciplined compensation structure")

        if sbc_trend is not None and sbc_trend > 20:
            if quality_flag == "green":
                quality_flag = "yellow"
            concerns.append(f"-SBC growing {sbc_trend:+.1f}% YoY — compensation scaling faster than business")

        if dilution_pct is not None:
            if dilution_pct > 3:
                if quality_flag == "green":
                    quality_flag = "yellow"
                concerns.append(f"-Share count +{dilution_pct:.1f}% YoY — diluting shareholders")
            elif dilution_pct < -1:
                strengths.append(f"+Share buyback reduced count {abs(dilution_pct):.1f}% YoY — returning capital")

        if div_yield and div_yield > 0.02:
            strengths.append(f"+Dividend yield {div_yield*100:.1f}% — visible capital return")

        if exec_comp and revenue_series:
            exec_comp_pct = exec_comp / revenue_series[0] * 100 if revenue_series[0] > 0 else None
            if exec_comp_pct and exec_comp_pct > 1:
                concerns.append(f"-Exec comp ${exec_comp/1e6:.1f}M is {exec_comp_pct:.2f}% of revenue — high relative to size")

        return {
            "ticker": ticker,
            "data_source": "EDGAR XBRL" if xbrl_data else "yfinance",
            "insider_ownership_pct": round(insider_ownership * 100, 2) if insider_ownership else None,
            "sbc_pct_of_revenue": round(sbc_pct, 2) if sbc_pct is not None else None,
            "sbc_yoy_growth_pct": sbc_trend,
            "sbc_history_usd": [int(v) for v in sbc_series[:3]] if sbc_series else None,
            "shares_dilution_yoy_pct": round(dilution_pct, 2) if dilution_pct is not None else None,
            "dividend_yield_pct": round(div_yield * 100, 2) if div_yield else None,
            "quality_flag": quality_flag,
            "strengths": strengths,
            "concerns": concerns,
            "note": (
                "Green: good alignment and discipline. Yellow: concerns worth investigating. Red: significant misalignment. "
                "SBC > 10% of revenue is a major red flag — it means shareholders, not the company, are paying employees. "
                "For full proxy details (CEO pay ratio, option grants, claw-backs, golden parachutes), read the DEF 14A on EDGAR."
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
