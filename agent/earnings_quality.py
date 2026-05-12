"""
Earnings quality & financial strength scoring tools.
- Sloan accrual ratio (predicts earnings disappointment)
- Piotroski F-Score (9-point financial health framework)
- Historical valuation ranges for relative value assessment
"""

import numpy as np
import yfinance as yf
from typing import Optional


def score_earnings_quality(ticker: str) -> dict:
    """
    Score earnings quality using Sloan accrual ratio and FCF conversion efficiency.

    High accruals (earnings >> cash flow) predict future earnings disappointments.
    Returns: accrual_ratio, fcf_conversion_efficiency, quality_flag (red/yellow/green).
    """
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # Fetch annual financials
        fin = t.financials
        bs = t.balance_sheet
        cf = t.cashflow

        if fin is None or fin.empty or bs is None or bs.empty or cf is None or cf.empty:
            return {"ticker": ticker, "error": "Insufficient annual financial data"}

        # Get most recent year (first column)
        col = fin.columns[0]

        # ── Sloan Accrual Ratio ───────────────────────────────────────────────
        # Definition: (Net Income - Operating CF - Investing CF) / Average Total Assets
        # High ratio (>0.05) indicates earnings quality concerns

        net_income = fin.loc["Net Income"].iloc[0] if "Net Income" in fin.index else None
        if net_income is None:
            net_income = fin.loc["Net Income Common Stockholders"].iloc[0] if "Net Income Common Stockholders" in fin.index else None

        opcf = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
        if opcf is None:
            opcf = cf.loc["Total Cash From Operating Activities"].iloc[0] if "Total Cash From Operating Activities" in cf.index else None

        invcf = cf.loc["Investing Cash Flow"].iloc[0] if "Investing Cash Flow" in cf.index else None
        if invcf is None:
            invcf = cf.loc["Total Cash From Investing Activities"].iloc[0] if "Total Cash From Investing Activities" in cf.index else None

        total_assets_curr = bs.loc["Total Assets"].iloc[0] if "Total Assets" in bs.index else None
        total_assets_prev = bs.loc["Total Assets"].iloc[1] if "Total Assets" in bs.index and len(bs.columns) > 1 else None

        if not all([net_income, opcf, invcf, total_assets_curr]):
            return {"ticker": ticker, "error": "Missing required financial statement data"}

        avg_assets = (total_assets_curr + (total_assets_prev or total_assets_curr)) / 2

        if avg_assets == 0:
            return {"ticker": ticker, "error": "Zero average assets"}

        # Accrual = Net Income - Operating CF - Investing CF
        accrual = net_income - opcf - invcf
        accrual_ratio = accrual / avg_assets

        # ── FCF Conversion Efficiency ─────────────────────────────────────────
        # How much of net income converts to actual free cash flow?
        # FCF = Operating CF - CapEx
        capex = cf.loc["Capital Expenditure"].iloc[0] if "Capital Expenditure" in cf.index else 0
        capex = abs(capex) if capex < 0 else capex  # capex usually negative
        fcf = (opcf or 0) - (capex or 0)

        fcf_conversion = fcf / net_income if net_income > 0 else None

        # ── Quality Assessment ────────────────────────────────────────────────
        quality_flag = "green"
        concerns = []

        if abs(accrual_ratio) > 0.10:
            quality_flag = "red"
            concerns.append(f"High accrual ratio {accrual_ratio:.3f} >> 0.05 threshold — earnings quality red flag")
        elif abs(accrual_ratio) > 0.05:
            quality_flag = "yellow"
            concerns.append(f"Elevated accrual ratio {accrual_ratio:.3f} — monitor for earnings trends")

        if fcf_conversion is not None and fcf_conversion < 0.6:
            if quality_flag == "green":
                quality_flag = "yellow"
            concerns.append(f"Poor FCF conversion {fcf_conversion:.1%} — only {fcf_conversion:.1%} of NI converts to FCF")

        return {
            "ticker": ticker,
            "sloan_accrual_ratio": round(accrual_ratio, 4),
            "fcf_conversion_efficiency": round(fcf_conversion, 3) if fcf_conversion is not None else None,
            "quality_flag": quality_flag,
            "concerns": concerns,
            "interpretation": (
                "Green: healthy earnings quality. Yellow: some concerns, investigate further. "
                "Red: high likelihood of earnings disappointment or restatement — strong caution signal."
            ),
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"Earnings quality scoring failed: {e}"}


def score_piotroski_fscore(ticker: str) -> dict:
    """
    Score financial strength using Piotroski's 9-point framework.

    Checks: profitability (ROA, CFO, accruals), leverage (debt trend),
    and operating efficiency (asset/revenue/GMM trends).

    Score 8-9: strong, 5-7: moderate, 0-4: weak.
    Historically, high F-score stocks outperform and low F-score underperform.
    """
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        fin = t.financials
        bs = t.balance_sheet
        cf = t.cashflow

        if fin is None or fin.empty:
            return {"ticker": ticker, "error": "Insufficient data to score F-Score"}

        # We need at least 2 years for comparison
        if len(fin.columns) < 2:
            return {"ticker": ticker, "error": "Need 2+ years of history for F-Score"}

        col_curr = fin.columns[0]
        col_prev = fin.columns[1]

        score = 0
        signals = []

        # ── Profitability Signals (4 points) ──────────────────────────────────
        # 1. ROA (Return on Assets): current year > prior year
        if "Net Income" in fin.index and "Total Assets" in bs.index:
            ni_curr = fin.loc["Net Income"].iloc[0]
            ta_curr = bs.loc["Total Assets"].iloc[0]
            ni_prev = fin.loc["Net Income"].iloc[1]
            ta_prev = bs.loc["Total Assets"].iloc[1]

            if ta_curr and ta_prev and ni_curr and ni_prev:
                roa_curr = ni_curr / ta_curr
                roa_prev = ni_prev / ta_prev
                if roa_curr > roa_prev:
                    score += 1
                    signals.append("+ROA improving")
                else:
                    signals.append("-ROA declining")

        # 2. CFO (Operating Cash Flow) > 0
        if "Operating Cash Flow" in cf.index:
            cfo = cf.loc["Operating Cash Flow"].iloc[0]
            if cfo and cfo > 0:
                score += 1
                signals.append("+Operating CF positive")
            else:
                signals.append("-Operating CF negative/zero")

        # 3. Quality of Earnings: CFO > NI (earnings are backed by cash)
        if "Operating Cash Flow" in cf.index and "Net Income" in fin.index:
            cfo = cf.loc["Operating Cash Flow"].iloc[0]
            ni = fin.loc["Net Income"].iloc[0]
            if cfo and ni and cfo > ni:
                score += 1
                signals.append("+CFO > NI (high quality earnings)")
            else:
                signals.append("-CFO <= NI (earnings less cash-backed)")

        # 4. Accruals declining (or low): ∆Accruals < 0
        if "Operating Cash Flow" in cf.index and "Net Income" in fin.index:
            cfo_curr = cf.loc["Operating Cash Flow"].iloc[0]
            ni_curr = fin.loc["Net Income"].iloc[0]
            cfo_prev = cf.loc["Operating Cash Flow"].iloc[1]
            ni_prev = fin.loc["Net Income"].iloc[1]

            if all([cfo_curr, ni_curr, cfo_prev, ni_prev]):
                accrual_curr = ni_curr - cfo_curr
                accrual_prev = ni_prev - cfo_prev
                if accrual_curr < accrual_prev:
                    score += 1
                    signals.append("+Accruals declining")
                else:
                    signals.append("-Accruals rising (earnings quality concern)")

        # ── Leverage & Liquidity Signals (2 points) ──────────────────────────
        # 5. Debt declining: total debt curr < prior
        if "Total Debt" in bs.index:
            debt_curr = bs.loc["Total Debt"].iloc[0]
            debt_prev = bs.loc["Total Debt"].iloc[1]
            if debt_curr and debt_prev and debt_curr < debt_prev:
                score += 1
                signals.append("+Debt declining")
            else:
                signals.append("-Debt same/rising")

        # 6. Current Ratio improving: (CA / CL) curr > prior
        if "Current Assets" in bs.index and "Current Liabilities" in bs.index:
            ca_curr = bs.loc["Current Assets"].iloc[0]
            cl_curr = bs.loc["Current Liabilities"].iloc[0]
            ca_prev = bs.loc["Current Assets"].iloc[1]
            cl_prev = bs.loc["Current Liabilities"].iloc[1]

            if all([ca_curr, cl_curr, ca_prev, cl_prev]):
                cr_curr = ca_curr / cl_curr
                cr_prev = ca_prev / cl_prev
                if cr_curr > cr_prev:
                    score += 1
                    signals.append("+Current ratio improving")
                else:
                    signals.append("-Current ratio declining")

        # ── Operating Efficiency Signals (3 points) ──────────────────────────
        # 7. Asset Turnover increasing: (Revenue / Total Assets) curr > prior
        if "Total Revenue" in fin.index and "Total Assets" in bs.index:
            rev_curr = fin.loc["Total Revenue"].iloc[0]
            ta_curr = bs.loc["Total Assets"].iloc[0]
            rev_prev = fin.loc["Total Revenue"].iloc[1]
            ta_prev = bs.loc["Total Assets"].iloc[1]

            if all([rev_curr, ta_curr, rev_prev, ta_prev]):
                at_curr = rev_curr / ta_curr
                at_prev = rev_prev / ta_prev
                if at_curr > at_prev:
                    score += 1
                    signals.append("+Asset turnover improving")
                else:
                    signals.append("-Asset turnover declining")

        # 8. Gross Margin stable or improving
        if "Gross Profit" in fin.index and "Total Revenue" in fin.index:
            gp_curr = fin.loc["Gross Profit"].iloc[0]
            rev_curr = fin.loc["Total Revenue"].iloc[0]
            gp_prev = fin.loc["Gross Profit"].iloc[1]
            rev_prev = fin.loc["Total Revenue"].iloc[1]

            if all([gp_curr, rev_curr, gp_prev, rev_prev]):
                gm_curr = gp_curr / rev_curr
                gm_prev = gp_prev / rev_prev
                if gm_curr >= gm_prev * 0.99:  # allow 1% slip
                    score += 1
                    signals.append("+Gross margin stable/improving")
                else:
                    signals.append("-Gross margin compressing")

        # 9. Shares Outstanding stable or declining (no massive dilution)
        if "Common Stock Shares Outstanding" in bs.index:
            shares_curr = bs.loc["Common Stock Shares Outstanding"].iloc[0]
            shares_prev = bs.loc["Common Stock Shares Outstanding"].iloc[1]

            if shares_curr and shares_prev and shares_curr <= shares_prev * 1.05:  # allow 5% increase
                score += 1
                signals.append("+Share count stable/declining")
            else:
                signals.append("-Significant share dilution")

        # ── Assessment ────────────────────────────────────────────────────────
        if score >= 8:
            assessment = "STRONG"
        elif score >= 5:
            assessment = "MODERATE"
        else:
            assessment = "WEAK"

        return {
            "ticker": ticker,
            "piotroski_fscore": score,
            "assessment": assessment,
            "signals": signals,
            "interpretation": (
                "F-Score 8-9: strong financial health; historically outperform. "
                "5-7: moderate; 0-4: weak financial health. "
                "Historically, low F-Score stocks significantly underperform."
            ),
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"F-Score calculation failed: {e}"}


def get_historical_valuation_range(ticker: str) -> dict:
    """
    Get current P/FCF and EV/EBITDA vs their 5-year and 10-year historical ranges.

    Returns percentile of current valuation within historical range.
    100th percentile = most expensive in the history
    0th percentile = cheapest in the history
    50th percentile = median valuation

    Use this to avoid buying at peak valuations even if the absolute DCF says "fair".
    """
    ticker = ticker.upper()
    try:
        import pandas as pd

        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="10y")
        fin = t.financials

        if hist is None or hist.empty:
            return {"ticker": ticker, "error": "Insufficient price history"}

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not current_price:
            return {"ticker": ticker, "error": "Cannot determine current price"}

        shares = info.get("sharesOutstanding")
        if not shares:
            return {"ticker": ticker, "error": "Cannot determine shares outstanding"}

        market_cap = current_price * shares

        # ── FCF ───────────────────────────────────────────────────────────────
        fcf_curr = info.get("freeCashflow")
        pe_ratio = None
        fcf_yield = None
        p_fcf = None

        if fcf_curr and fcf_curr > 0:
            p_fcf = market_cap / fcf_curr
            fcf_yield = fcf_curr / market_cap * 100

        # ── EBITDA ────────────────────────────────────────────────────────────
        ebitda = info.get("ebitda")
        ev = info.get("enterpriseValue")
        ev_ebitda = None

        if ebitda and ebitda > 0 and ev and ev > 0:
            ev_ebitda = ev / ebitda

        # ── Build historical series (12-month rolling) ────────────────────────
        historical_p_fcf = []
        historical_ev_ebitda = []

        # For each month in history, calculate the multiples
        # This is approximate — we'd need quarterly FCF/EBITDA data for perfect accuracy
        # For now, we'll use the most recent available FCF/EBITDA

        # Approximate: assume FCF and EBITDA are stable-ish, so P/FCF and EV/EBITDA
        # move mainly with price. Not perfect, but directionally useful.

        if fcf_curr and fcf_curr > 0:
            for date, row in hist.iterrows():
                price_then = row["Close"]
                market_cap_then = price_then * shares
                p_fcf_then = market_cap_then / fcf_curr  # rough approximation
                historical_p_fcf.append(p_fcf_then)

        if ebitda and ebitda > 0 and ev and ev > 0:
            # Even rougher approximation
            ev_to_market_cap = ev / market_cap
            for date, row in hist.iterrows():
                price_then = row["Close"]
                market_cap_then = price_then * shares
                ev_then = market_cap_then * ev_to_market_cap
                ev_ebitda_then = ev_then / ebitda
                historical_ev_ebitda.append(ev_ebitda_then)

        result = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "current_multiples": {
                "p_fcf": round(p_fcf, 1) if p_fcf else None,
                "fcf_yield_pct": round(fcf_yield, 2) if fcf_yield else None,
                "ev_ebitda": round(ev_ebitda, 1) if ev_ebitda else None,
            },
        }

        # ── P/FCF Range ───────────────────────────────────────────────────────
        if historical_p_fcf:
            p_fcf_5y = historical_p_fcf[-252*5:] if len(historical_p_fcf) > 252*5 else historical_p_fcf
            p_fcf_min = np.nanmin(p_fcf_5y)
            p_fcf_max = np.nanmax(p_fcf_5y)
            p_fcf_median = np.nanmedian(p_fcf_5y)

            if p_fcf and p_fcf > 0:
                p_fcf_percentile = (p_fcf - p_fcf_min) / (p_fcf_max - p_fcf_min) * 100 if p_fcf_max > p_fcf_min else 50
            else:
                p_fcf_percentile = None

            result["p_fcf_5y_range"] = {
                "min": round(p_fcf_min, 1),
                "median": round(p_fcf_median, 1),
                "max": round(p_fcf_max, 1),
                "current_percentile": round(p_fcf_percentile, 0) if p_fcf_percentile is not None else None,
            }

        # ── EV/EBITDA Range ───────────────────────────────────────────────────
        if historical_ev_ebitda:
            ev_ebitda_5y = historical_ev_ebitda[-252*5:] if len(historical_ev_ebitda) > 252*5 else historical_ev_ebitda
            ev_ebitda_min = np.nanmin(ev_ebitda_5y)
            ev_ebitda_max = np.nanmax(ev_ebitda_5y)
            ev_ebitda_median = np.nanmedian(ev_ebitda_5y)

            if ev_ebitda and ev_ebitda > 0:
                ev_ebitda_percentile = (ev_ebitda - ev_ebitda_min) / (ev_ebitda_max - ev_ebitda_min) * 100 if ev_ebitda_max > ev_ebitda_min else 50
            else:
                ev_ebitda_percentile = None

            result["ev_ebitda_5y_range"] = {
                "min": round(ev_ebitda_min, 1),
                "median": round(ev_ebitda_median, 1),
                "max": round(ev_ebitda_max, 1),
                "current_percentile": round(ev_ebitda_percentile, 0) if ev_ebitda_percentile is not None else None,
            }

        result["interpretation"] = (
            "Percentile shows where current valuation sits historically. "
            "90-100: expensive (narrow MoS); 50: median; 0-10: cheap. "
            "Combine with DCF for robust buy decision."
        )

        return result

    except Exception as e:
        return {"ticker": ticker, "error": f"Historical valuation analysis failed: {e}"}
