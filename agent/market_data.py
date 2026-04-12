"""
Market data retrieval using yfinance with Polygon.io as a fallback.
Fundamentals (income statement, balance sheet, ratios) use Financial Modeling Prep (FMP)
as the primary source with yfinance as fallback. Price data stays on yfinance.
Set FMP_API_KEY env var to enable (free tier: 250 req/day at financialmodelingprep.com).
"""

import json
import os
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional


_SCREENER_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "screener_cache.json"
)


def _load_screener_cache() -> dict:
    try:
        with open(_SCREENER_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_screener_cache(date: str, results: list) -> None:
    os.makedirs(os.path.dirname(_SCREENER_CACHE_FILE), exist_ok=True)
    with open(_SCREENER_CACHE_FILE, "w") as f:
        json.dump({"date": date, "results": results}, f)


_SCREENER_STALE_DAYS = 7   # Re-fetch a ticker's fundamentals after this many days


# ── Financial Modeling Prep (primary for fundamentals) ────────────────────────
FMP_API_KEY = os.environ.get("FMP_API_KEY")
_FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _fmp_get(path: str, **params) -> Optional[list]:
    """Make a GET request to the FMP API. Returns parsed JSON or None on error."""
    if not FMP_API_KEY:
        return None
    try:
        url = f"{_FMP_BASE}/{path.lstrip('/')}"
        params["apikey"] = FMP_API_KEY
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # FMP returns [] or {} on bad tickers rather than an error status
        if not data:
            return None
        return data
    except Exception:
        return None


def _fmp_get_fundamentals(ticker: str) -> Optional[dict]:
    """
    Fetch fundamentals from FMP and return them mapped to the same keys
    used by get_stock_fundamentals() so callers need no changes.
    Returns None if FMP is unavailable or the ticker is unknown.
    """
    t = ticker.upper()

    # Fetch in parallel: profile, key metrics TTM, financial growth
    profile_data = _fmp_get(f"profile/{t}")
    metrics_data = _fmp_get(f"key-metrics-ttm/{t}")
    ratios_data  = _fmp_get(f"ratios-ttm/{t}")
    growth_data  = _fmp_get(f"financial-growth/{t}", period="annual", limit=1)
    quote_data   = _fmp_get(f"quote/{t}")

    profile  = profile_data[0]  if isinstance(profile_data,  list) and profile_data  else {}
    metrics  = metrics_data[0]  if isinstance(metrics_data,  list) and metrics_data  else {}
    ratios   = ratios_data[0]   if isinstance(ratios_data,   list) and ratios_data   else {}
    growth   = growth_data[0]   if isinstance(growth_data,   list) and growth_data   else {}
    quote    = quote_data[0]    if isinstance(quote_data,    list) and quote_data    else {}

    if not profile and not metrics:
        return None

    market_cap = quote.get("marketCap") or profile.get("mktCap")
    free_cashflow = metrics.get("freeCashFlowPerShareTTM")
    shares = profile.get("sharesOutstanding")
    # Convert FCF per share → absolute FCF if shares available
    fcf_abs = (free_cashflow * shares) if (free_cashflow and shares) else None

    return {
        "ticker": t,
        "name": profile.get("companyName"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "description": (profile.get("description") or "")[:500],
        # Valuation
        "pe_ratio":        metrics.get("peRatioTTM"),
        "forward_pe":      ratios.get("priceEarningsRatioTTM"),   # best TTM proxy
        "peg_ratio":       metrics.get("pegRatioTTM"),
        "price_to_book":   metrics.get("pbRatioTTM"),
        "price_to_sales":  metrics.get("priceToSalesRatioTTM"),
        "ev_to_ebitda":    metrics.get("enterpriseValueOverEBITDATTM"),
        # Profitability
        "profit_margin":   ratios.get("netProfitMarginTTM"),
        "roe":             metrics.get("roeTTM"),
        "roa":             ratios.get("returnOnAssetsTTM"),
        "revenue_growth":  growth.get("revenueGrowth"),
        "earnings_growth": growth.get("netIncomeGrowth"),
        # Balance sheet
        "total_cash":      None,   # not directly in TTM metrics
        "total_debt":      None,
        "debt_to_equity":  metrics.get("debtToEquityTTM"),
        "current_ratio":   metrics.get("currentRatioTTM"),
        # Dividends
        "dividend_yield":              ratios.get("dividendYieldTTM"),
        "payout_ratio":                ratios.get("payoutRatioTTM"),
        "five_year_avg_dividend_yield": None,
        # Analyst
        "analyst_target_price":         quote.get("priceAvg12m"),
        "recommendation":               None,
        "number_of_analyst_opinions":   None,
        # Share info
        "shares_outstanding":  shares,
        "float_shares":        profile.get("floatShares"),
        "insider_ownership":   None,
        "institutional_ownership": None,
        # Extras used by screener
        "_market_cap":      market_cap,
        "_free_cashflow":   fcf_abs,
        "_volume":          quote.get("volume"),
        "_price":           quote.get("price"),
        "_fcf_yield":       metrics.get("freeCashFlowYieldTTM"),
        "_week52_change":   None,   # price-based — leave to yfinance
        "_sp52_change":     None,
    }


def _fmp_get_quarterly_statements(ticker: str) -> Optional[dict]:
    """
    Fetch quarterly income statement, balance sheet, and cash flow from FMP.
    Returns a dict with keys: income, balance, cashflow — each a list of
    quarterly dicts (newest first, up to 8 quarters).
    Returns None if FMP unavailable.
    """
    if not FMP_API_KEY:
        return None
    t = ticker.upper()
    income  = _fmp_get(f"income-statement/{t}",          period="quarter", limit=8)
    balance = _fmp_get(f"balance-sheet-statement/{t}",   period="quarter", limit=8)
    cashflow = _fmp_get(f"cash-flow-statement/{t}",      period="quarter", limit=8)
    if not income and not balance and not cashflow:
        return None
    return {
        "income":   income   or [],
        "balance":  balance  or [],
        "cashflow": cashflow or [],
    }


# ── Polygon.io fallback ────────────────────────────────────────────────────────
# Sign up at https://polygon.io (free tier) and set POLYGON_API_KEY in .env
# Free tier provides: end-of-day quotes, news, reference data — enough for a
# monthly investor. Activated automatically when yfinance returns no data.

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
_POLYGON_BASE = "https://api.polygon.io"


def _polygon_get_quote(ticker: str) -> dict:
    """Polygon.io fallback for stock quote — returns previous close OHLCV."""
    if not POLYGON_API_KEY:
        return {"error": "POLYGON_API_KEY not set"}
    try:
        url = f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker.upper()}/prev"
        r = requests.get(url, params={"adjusted": "true", "apiKey": POLYGON_API_KEY}, timeout=8)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if not results:
            return {"error": f"Polygon: no data for {ticker}"}
        bar = results[0]
        return {
            "ticker": ticker.upper(),
            "price": bar.get("c"),          # previous close
            "currency": "USD",
            "name": ticker.upper(),
            "day_high": bar.get("h"),
            "day_low": bar.get("l"),
            "volume": bar.get("v"),
            "source": "polygon",
        }
    except Exception as e:
        return {"error": f"Polygon quote error: {e}"}


def _polygon_get_news(ticker: str, limit: int = 5) -> list[dict]:
    """Polygon.io fallback for stock news."""
    if not POLYGON_API_KEY:
        return [{"error": "POLYGON_API_KEY not set"}]
    try:
        url = f"{_POLYGON_BASE}/v2/reference/news"
        r = requests.get(url, params={"ticker": ticker.upper(), "limit": limit, "apiKey": POLYGON_API_KEY}, timeout=8)
        r.raise_for_status()
        articles = []
        for a in r.json().get("results", []):
            pub = a.get("published_utc", "")
            articles.append({
                "title": a.get("title", ""),
                "publisher": (a.get("publisher") or {}).get("name", ""),
                "published_at": pub[:16].replace("T", " ") if pub else "unknown",
                "related_tickers": a.get("tickers", []),
                "link": a.get("article_url", ""),
                "source": "polygon",
            })
        return articles if articles else [{"note": f"No Polygon news for {ticker.upper()}"}]
    except Exception as e:
        return [{"error": f"Polygon news error: {e}"}]


def get_spy_price() -> Optional[float]:
    """Fetch current S&P 500 index level (^GSPC)."""
    try:
        info = yf.Ticker("^GSPC").info
        return info.get("regularMarketPrice") or info.get("previousClose")
    except Exception:
        return None


def get_stock_quote(ticker: str) -> dict:
    """
    Get current price and basic info for a ticker.
    Tries yfinance first; falls back to Polygon.io if price is unavailable.
    """
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if price is not None:
            return {
                "ticker": ticker.upper(),
                "price": price,
                "currency": info.get("currency", "USD"),
                "name": info.get("longName") or info.get("shortName", ticker),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "exchange": info.get("exchange", ""),
                "market_cap": info.get("marketCap"),
                "volume": info.get("regularMarketVolume") or info.get("volume"),
                "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
            }
    except Exception:
        pass

    # yfinance failed — try Polygon.io
    result = _polygon_get_quote(ticker)
    if "error" not in result:
        return result
    return {"error": f"Could not retrieve price for {ticker} from any source"}


def get_stock_fundamentals(ticker: str) -> dict:
    """
    Get fundamental data useful for long-term investing.
    Uses FMP as primary source (more reliable financials); falls back to yfinance.
    """
    # Try FMP first
    fmp = _fmp_get_fundamentals(ticker)
    if fmp:
        return fmp

    # yfinance fallback
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info

        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": (info.get("longBusinessSummary") or "")[:500],
            # Valuation
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            # Profitability
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            # Balance sheet
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            # Dividends
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield"),
            # Analyst
            "analyst_target_price": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey"),
            "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
            # Share info
            "shares_outstanding": info.get("sharesOutstanding"),
            "float_shares": info.get("floatShares"),
            "insider_ownership": info.get("heldPercentInsiders"),
            "institutional_ownership": info.get("heldPercentInstitutions"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_price_history(ticker: str, period: str = "1y") -> dict:
    """
    Get historical price data.
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period=period)
        if hist.empty:
            return {"error": f"No history for {ticker}"}

        prices = hist["Close"].dropna()
        start_price = float(prices.iloc[0])
        end_price = float(prices.iloc[-1])
        pct_change = ((end_price - start_price) / start_price) * 100

        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_date": prices.index[0].strftime("%Y-%m-%d"),
            "end_date": prices.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(start_price, 2),
            "current_price": round(end_price, 2),
            "pct_change": round(pct_change, 2),
            "high": round(float(hist["High"].max()), 2),
            "low": round(float(hist["Low"].min()), 2),
            "avg_volume": round(float(hist["Volume"].mean())),
            "data_points": len(prices),
        }
    except Exception as e:
        return {"error": str(e)}


def get_options_flow(ticker: str) -> dict:
    """
    Retrieve options market data: put/call ratio, ATM implied volatility,
    IV vs. realized volatility, and unusual contract activity.

    Analyzes the nearest 3 expiries (most liquid, most relevant for sentiment).
    Realized volatility is computed from 30-day daily price history.

    Key signals:
    - put/call volume ratio > 1.0: bearish positioning
    - put/call volume ratio < 0.7: bullish positioning
    - IV materially above realized vol: market pricing in event risk or fear
    - Unusual contracts (volume >> open interest): fresh directional bets opening
    """
    try:
        t = yf.Ticker(ticker.upper())
        expiries = t.options
        if not expiries:
            return {"ticker": ticker.upper(), "error": "No options data available for this ticker"}

        # Current price for ATM selection
        info = t.info
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if not current_price:
            return {"ticker": ticker.upper(), "error": "Could not determine current price"}

        # Select nearest 3 expiries (up to ~6 weeks out for liquidity)
        today = datetime.now().date()
        near_expiries = []
        for exp in expiries:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days_out = (exp_date - today).days
            if 5 <= days_out <= 60:
                near_expiries.append(exp)
            if len(near_expiries) == 3:
                break
        # Fall back to first expiry if none in the 5-60 day window
        if not near_expiries:
            near_expiries = expiries[:2]

        # ── Aggregate put/call data across near expiries ──────────────────────
        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        atm_iv_calls = []
        atm_iv_puts = []
        unusual_contracts = []

        for exp in near_expiries:
            try:
                chain = t.option_chain(exp)
            except Exception:
                continue

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # Fill NaN volumes/OI with 0
            for col in ("volume", "openInterest"):
                calls[col] = calls[col].fillna(0)
                puts[col] = puts[col].fillna(0)

            total_call_vol += int(calls["volume"].sum())
            total_put_vol += int(puts["volume"].sum())
            total_call_oi += int(calls["openInterest"].sum())
            total_put_oi += int(puts["openInterest"].sum())

            # ATM IV: find the strike closest to current price in each chain
            if not calls.empty and "impliedVolatility" in calls.columns:
                calls["strike_dist"] = (calls["strike"] - current_price).abs()
                atm_call = calls.loc[calls["strike_dist"].idxmin()]
                iv = atm_call.get("impliedVolatility")
                if iv and iv > 0:
                    atm_iv_calls.append(float(iv))

            if not puts.empty and "impliedVolatility" in puts.columns:
                puts["strike_dist"] = (puts["strike"] - current_price).abs()
                atm_put = puts.loc[puts["strike_dist"].idxmin()]
                iv = atm_put.get("impliedVolatility")
                if iv and iv > 0:
                    atm_iv_puts.append(float(iv))

            # Unusual activity: volume ≥ 3× open interest, minimum 500 contracts
            # This signals fresh directional positioning (not just rolling/hedging)
            for side, df in [("call", calls), ("put", puts)]:
                if df.empty:
                    continue
                unusual = df[
                    (df["volume"] >= 500) &
                    (df["openInterest"] > 0) &
                    (df["volume"] >= df["openInterest"] * 3)
                ].copy()
                for _, row in unusual.iterrows():
                    vol = int(row["volume"])
                    oi = int(row["openInterest"])
                    iv_pct = round(float(row["impliedVolatility"]) * 100, 1) if row.get("impliedVolatility") else None
                    unusual_contracts.append({
                        "type": side,
                        "strike": float(row["strike"]),
                        "expiry": exp,
                        "volume": vol,
                        "open_interest": oi,
                        "vol_oi_ratio": round(vol / oi, 1),
                        "iv_pct": iv_pct,
                        "note": f"Volume {round(vol/oi, 1)}x OI — likely fresh {side} position",
                    })

        # Sort unusual by volume descending, keep top 5
        unusual_contracts.sort(key=lambda x: x["volume"], reverse=True)
        unusual_contracts = unusual_contracts[:5]

        # ── Put/call ratios ───────────────────────────────────────────────────
        pcr_volume = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else None
        pcr_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None

        if pcr_volume is None:
            pcr_signal = "unknown"
        elif pcr_volume < 0.7:
            pcr_signal = "bullish"
        elif pcr_volume < 1.0:
            pcr_signal = "neutral"
        elif pcr_volume < 1.5:
            pcr_signal = "bearish"
        else:
            pcr_signal = "strongly_bearish"

        # ── ATM implied volatility ────────────────────────────────────────────
        all_atm_ivs = atm_iv_calls + atm_iv_puts
        atm_iv = round(sum(all_atm_ivs) / len(all_atm_ivs) * 100, 1) if all_atm_ivs else None

        # ── 30-day realized volatility (annualised) ───────────────────────────
        realized_vol = None
        try:
            hist = t.history(period="3mo")
            if len(hist) >= 21:
                log_returns = hist["Close"].pct_change().dropna()
                realized_vol = round(float(log_returns.tail(21).std() * (252 ** 0.5) * 100), 1)
        except Exception:
            pass

        # ── IV vs realized vol ────────────────────────────────────────────────
        iv_premium = None
        iv_vs_realized = "unknown"
        if atm_iv is not None and realized_vol is not None:
            iv_premium = round(atm_iv - realized_vol, 1)
            if iv_premium > 10:
                iv_vs_realized = "elevated"    # market pricing in fear or upcoming event
            elif iv_premium > 3:
                iv_vs_realized = "slightly_elevated"
            elif iv_premium < -5:
                iv_vs_realized = "depressed"   # unusually cheap options
            else:
                iv_vs_realized = "normal"

        # ── Interpretation ────────────────────────────────────────────────────
        interpretation = []

        if pcr_volume is not None:
            if pcr_signal in ("bullish",):
                interpretation.append(
                    f"Put/call volume ratio {pcr_volume} — options traders net bullish; "
                    "more call than put volume in near-term expiries."
                )
            elif pcr_signal in ("bearish", "strongly_bearish"):
                interpretation.append(
                    f"Put/call volume ratio {pcr_volume} — elevated put buying; "
                    "options market is hedging or positioning for downside."
                )
            else:
                interpretation.append(f"Put/call volume ratio {pcr_volume} — neutral options positioning.")

        if iv_vs_realized == "elevated":
            interpretation.append(
                f"ATM IV {atm_iv}% vs. 30d realized vol {realized_vol}% "
                f"(+{iv_premium}pp premium) — market is pricing in an event or elevated fear. "
                "Options are expensive; buying stock rather than calls may be more efficient."
            )
        elif iv_vs_realized == "depressed":
            interpretation.append(
                f"ATM IV {atm_iv}% vs. 30d realized vol {realized_vol}% "
                f"({iv_premium:+.1f}pp) — unusually cheap options; low implied event risk."
            )
        elif atm_iv is not None and realized_vol is not None:
            interpretation.append(
                f"ATM IV {atm_iv}% vs. 30d realized vol {realized_vol}% — options fairly priced."
            )

        if unusual_contracts:
            sides = [c["type"] for c in unusual_contracts]
            call_unusual = sides.count("call")
            put_unusual = sides.count("put")
            if call_unusual > put_unusual:
                interpretation.append(
                    f"{len(unusual_contracts)} unusual contract(s) detected — "
                    f"skewed toward calls ({call_unusual} call vs {put_unusual} put); "
                    "suggests fresh bullish positioning by large traders."
                )
            elif put_unusual > call_unusual:
                interpretation.append(
                    f"{len(unusual_contracts)} unusual contract(s) detected — "
                    f"skewed toward puts ({put_unusual} put vs {call_unusual} call); "
                    "suggests hedging or directional bearish bet."
                )
            else:
                interpretation.append(
                    f"{len(unusual_contracts)} unusual contracts detected across calls and puts."
                )

        return {
            "ticker": ticker.upper(),
            "as_of": today.strftime("%Y-%m-%d"),
            "current_price": round(current_price, 2),
            "expiries_analyzed": near_expiries,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_volume_ratio": pcr_volume,
            "put_call_oi_ratio": pcr_oi,
            "pcr_signal": pcr_signal,
            "atm_iv_pct": atm_iv,
            "realized_vol_30d_pct": realized_vol,
            "iv_vs_realized": iv_vs_realized,
            "iv_premium_pp": iv_premium,
            "unusual_contracts": unusual_contracts,
            "interpretation": interpretation,
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}


def get_short_interest(ticker: str) -> dict:
    """
    Retrieve short interest data for a stock from yfinance.

    Returns short % of float, days-to-cover, month-over-month change,
    and interpreted signals (institutional bear conviction vs. squeeze risk).

    Short interest data from yfinance is sourced from FINRA and updated
    twice monthly — figures lag by ~2 weeks.
    """
    try:
        info = yf.Ticker(ticker.upper()).info

        shares_short = info.get("sharesShort")
        shares_short_prior = info.get("sharesShortPriorMonth")
        short_pct_float = info.get("shortPercentOfFloat")     # decimal, e.g. 0.05
        short_ratio = info.get("shortRatio")                  # days to cover
        float_shares = info.get("floatShares")
        date_ts = info.get("dateShortInterest")               # UNIX timestamp

        if short_pct_float is None and shares_short is None:
            return {"ticker": ticker.upper(), "error": "No short interest data available"}

        # Convert to percentage
        short_pct = round(short_pct_float * 100, 2) if short_pct_float is not None else None

        # Month-over-month change in shares short
        mom_change_pct = None
        mom_direction = "unknown"
        if shares_short is not None and shares_short_prior and shares_short_prior > 0:
            mom_change_pct = round((shares_short - shares_short_prior) / shares_short_prior * 100, 1)
            if mom_change_pct > 10:
                mom_direction = "rising"       # bears adding conviction
            elif mom_change_pct < -10:
                mom_direction = "falling"      # bears covering — potential bullish signal
            else:
                mom_direction = "stable"

        # Data date
        data_date = None
        if date_ts:
            try:
                from datetime import timezone
                data_date = datetime.fromtimestamp(date_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                pass

        # ── Short % of float signal ───────────────────────────────────────────
        # Thresholds based on typical institutional research:
        #   < 5%   → low     (market neutral; no strong view)
        #   5-15%  → moderate (some institutional bears)
        #   15-25% → high    (significant short conviction)
        #   > 25%  → very_high (crowded short — major red flag OR squeeze fuel)
        if short_pct is None:
            short_level = "unknown"
        elif short_pct < 5:
            short_level = "low"
        elif short_pct < 15:
            short_level = "moderate"
        elif short_pct < 25:
            short_level = "high"
        else:
            short_level = "very_high"

        # ── Days-to-cover / squeeze risk ─────────────────────────────────────
        if short_ratio is None:
            squeeze_risk = "unknown"
        elif short_ratio < 3:
            squeeze_risk = "low"
        elif short_ratio < 7:
            squeeze_risk = "moderate"
        elif short_ratio < 10:
            squeeze_risk = "elevated"
        else:
            squeeze_risk = "high"

        # ── Interpretation ────────────────────────────────────────────────────
        # Short interest has a dual reading:
        # (A) Institutional bear signal: smart money has done deep work and is short
        # (B) Squeeze catalyst: if the long thesis is correct, a positive catalyst
        #     can force rapid short covering, amplifying gains
        #
        # Rule of thumb:
        #   High short % + rising trend = weight (A) heavily — treat as a fundamental red flag
        #   High short % + falling trend = bears covering, potential momentum reversal
        #   High short % + strong fundamental thesis = flag (B) as an upside catalyst, not the thesis itself
        interpretation = []

        if short_level in ("high", "very_high"):
            interpretation.append(
                f"Short interest elevated at {short_pct}% of float — "
                "institutional bears have a view; verify their thesis isn't something the bull case missed."
            )
        elif short_level == "moderate":
            interpretation.append(f"Moderate short interest at {short_pct}% — some institutional skepticism but not alarming.")
        else:
            interpretation.append(f"Low short interest at {short_pct}% — market broadly not positioned against this stock.")

        if mom_direction == "rising":
            interpretation.append(
                f"Short interest grew {mom_change_pct:+.1f}% month-over-month — bears adding conviction; investigate why."
            )
        elif mom_direction == "falling":
            interpretation.append(
                f"Short interest declined {mom_change_pct:+.1f}% month-over-month — short covering in progress; potential near-term tailwind."
            )

        if squeeze_risk in ("elevated", "high"):
            interpretation.append(
                f"Days-to-cover {short_ratio:.1f} days — high squeeze risk if a positive catalyst emerges. "
                "This amplifies upside on a correct long thesis but is NOT a standalone buy reason."
            )

        return {
            "ticker": ticker.upper(),
            "data_date": data_date,
            "shares_short": shares_short,
            "shares_short_prior_month": shares_short_prior,
            "short_percent_of_float": short_pct,
            "short_ratio_days_to_cover": round(float(short_ratio), 1) if short_ratio else None,
            "float_shares": float_shares,
            "mom_change_pct": mom_change_pct,
            "mom_direction": mom_direction,
            "short_level": short_level,
            "squeeze_risk": squeeze_risk,
            "interpretation": interpretation,
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}


def get_technical_indicators(ticker: str) -> dict:
    """
    Compute key technical indicators from 1-year daily price history.

    Returns RSI-14, MACD (12/26/9), Bollinger Bands (20, 2σ), EMA-50/200,
    volume trend, and an overall signal summary for entry timing.

    All calculations use pandas/numpy — no external TA library required.
    """
    try:
        hist = yf.Ticker(ticker.upper()).history(period="1y")
        if len(hist) < 30:
            return {"error": f"Insufficient price history for {ticker} (need ≥30 days)"}

        close = hist["Close"].dropna()
        volume = hist["Volume"].dropna()
        n = len(close)

        # ── RSI-14 ────────────────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Wilder smoothing (EWM with alpha=1/14)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else None

        if rsi is None:
            rsi_signal = "unknown"
        elif rsi >= 70:
            rsi_signal = "overbought"
        elif rsi <= 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"

        # ── MACD 12/26/9 ─────────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_val = round(float(macd_line.iloc[-1]), 4)
        signal_val = round(float(signal_line.iloc[-1]), 4)
        hist_val = round(float(histogram.iloc[-1]), 4)
        hist_prev = float(histogram.iloc[-2]) if n >= 2 else 0.0

        if macd_val > signal_val:
            macd_trend = "bullish"
        else:
            macd_trend = "bearish"

        if hist_val > 0 and hist_prev <= 0:
            macd_crossover = "bullish_crossover"
        elif hist_val < 0 and hist_prev >= 0:
            macd_crossover = "bearish_crossover"
        else:
            macd_crossover = "none"

        # ── Bollinger Bands 20, 2σ ────────────────────────────────────────────
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_mid_val = round(float(sma20.iloc[-1]), 2)
        bb_upper_val = round(float(bb_upper.iloc[-1]), 2)
        bb_lower_val = round(float(bb_lower.iloc[-1]), 2)
        current_price = round(float(close.iloc[-1]), 2)
        band_width = bb_upper_val - bb_lower_val
        bb_position_pct = round(
            ((current_price - bb_lower_val) / band_width * 100) if band_width > 0 else 50.0,
            1,
        )

        if current_price >= bb_upper_val:
            bb_signal = "at_upper_band"
        elif current_price <= bb_lower_val:
            bb_signal = "at_lower_band"
        else:
            bb_signal = "inside_bands"

        # ── EMA 50 / 200 ──────────────────────────────────────────────────────
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        ema50_val = round(float(ema50.iloc[-1]), 2)
        ema200_val = round(float(ema200.iloc[-1]), 2)
        price_vs_ema50_pct = round((current_price - ema50_val) / ema50_val * 100, 1)
        price_vs_ema200_pct = round((current_price - ema200_val) / ema200_val * 100, 1)

        if ema50_val > ema200_val:
            ma_trend = "golden_cross"       # longer-term uptrend
        else:
            ma_trend = "death_cross"        # longer-term downtrend

        # ── Volume trend ──────────────────────────────────────────────────────
        vol_20d_avg = round(float(volume.rolling(20).mean().iloc[-1]))
        last_vol = float(volume.iloc[-1])
        vol_ratio = round(last_vol / vol_20d_avg, 2) if vol_20d_avg > 0 else None

        if vol_ratio is None:
            vol_signal = "unknown"
        elif vol_ratio >= 2.0:
            vol_signal = "high_volume"
        elif vol_ratio >= 1.2:
            vol_signal = "above_average"
        elif vol_ratio <= 0.5:
            vol_signal = "very_low_volume"
        else:
            vol_signal = "normal"

        # ── Overall signal ────────────────────────────────────────────────────
        bull_points = 0
        bear_points = 0

        if rsi_signal == "oversold":
            bull_points += 2
        elif rsi_signal == "overbought":
            bear_points += 2

        if macd_trend == "bullish":
            bull_points += 1
        else:
            bear_points += 1

        if macd_crossover == "bullish_crossover":
            bull_points += 2
        elif macd_crossover == "bearish_crossover":
            bear_points += 2

        if ma_trend == "golden_cross":
            bull_points += 1
        else:
            bear_points += 1

        if current_price > ema50_val:
            bull_points += 1
        else:
            bear_points += 1

        if bb_signal == "at_lower_band":
            bull_points += 1
        elif bb_signal == "at_upper_band":
            bear_points += 1

        if bull_points > bear_points + 1:
            overall_signal = "bullish"
        elif bear_points > bull_points + 1:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # ── Human-readable summary ────────────────────────────────────────────
        signals = []
        if rsi is not None:
            signals.append(f"RSI-14 {rsi:.1f} ({rsi_signal})")
        signals.append(f"MACD {macd_trend}" + (f" — {macd_crossover.replace('_', ' ')}" if macd_crossover != "none" else ""))
        signals.append(f"Price vs EMA50: {price_vs_ema50_pct:+.1f}%, EMA200: {price_vs_ema200_pct:+.1f}% ({ma_trend.replace('_', ' ')})")
        signals.append(f"Bollinger position {bb_position_pct}% ({bb_signal.replace('_', ' ')})")
        signals.append(f"Volume {vol_ratio}x 20-day avg ({vol_signal.replace('_', ' ')})")

        return {
            "ticker": ticker.upper(),
            "as_of": close.index[-1].strftime("%Y-%m-%d"),
            "current_price": current_price,
            "rsi_14": round(rsi, 1) if rsi is not None else None,
            "rsi_signal": rsi_signal,
            "macd": macd_val,
            "macd_signal_line": signal_val,
            "macd_histogram": hist_val,
            "macd_trend": macd_trend,
            "macd_crossover": macd_crossover,
            "bb_upper": bb_upper_val,
            "bb_middle": bb_mid_val,
            "bb_lower": bb_lower_val,
            "bb_position_pct": bb_position_pct,
            "bb_signal": bb_signal,
            "ema_50": ema50_val,
            "ema_200": ema200_val,
            "price_vs_ema50_pct": price_vs_ema50_pct,
            "price_vs_ema200_pct": price_vs_ema200_pct,
            "ma_trend": ma_trend,
            "volume_20d_avg": vol_20d_avg,
            "volume_ratio": vol_ratio,
            "volume_signal": vol_signal,
            "overall_signal": overall_signal,
            "signals_summary": signals,
        }
    except Exception as e:
        return {"error": str(e)}


def search_stocks(query: str) -> list[dict]:
    try:
        results = yf.Search(query, max_results=5)
        quotes = results.quotes if hasattr(results, "quotes") else []
        return [
            {
                "ticker": q.get("symbol", ""),
                "name": q.get("longname") or q.get("shortname", ""),
                "exchange": q.get("exchange", ""),
                "type": q.get("quoteType", ""),
            }
            for q in quotes
            if q.get("quoteType") in ("EQUITY", "ETF")
        ]
    except Exception as e:
        return [{"error": str(e)}]


def get_market_summary() -> dict:
    """Get a quick snapshot of major indices."""
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX": "^VIX",
    }
    summary = {}
    for name, ticker in indices.items():
        try:
            t = yf.Ticker(ticker)
            info = t.info
            price = info.get("regularMarketPrice") or info.get("previousClose")
            prev = info.get("regularMarketPreviousClose") or info.get("previousClose")
            change_pct = ((price - prev) / prev * 100) if price and prev else None
            summary[name] = {
                "price": price,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
            }
        except Exception:
            summary[name] = {"price": None, "change_pct": None}
    return summary


def get_stock_news(ticker: str, limit: int = 8) -> list[dict]:
    """
    Fetch recent news headlines for a stock from Yahoo Finance.
    Returns title, publisher, publish date, and related tickers.
    Handles both yfinance <1.0 (flat dict) and >=1.0 (nested content) formats.
    """
    try:
        t = yf.Ticker(ticker.upper())
        raw = t.news or []
        articles = []
        for a in raw[:limit]:
            # yfinance >= 1.0 wraps everything under a "content" key
            if "content" in a and isinstance(a["content"], dict):
                content = a["content"]
                title = content.get("title", "")
                publisher = (content.get("provider") or {}).get("displayName", "")
                pub_date = content.get("pubDate", "")
                published = pub_date[:16].replace("T", " ") if pub_date else "unknown"
                link = (content.get("canonicalUrl") or {}).get("url", "") or (content.get("clickThroughUrl") or {}).get("url", "")
                related = [r.get("symbol", "") for r in (content.get("relatedTickers") or []) if isinstance(r, dict)]
            else:
                # Legacy flat dict format (yfinance < 1.0)
                title = a.get("title", "")
                publisher = a.get("publisher", "")
                ts = a.get("providerPublishTime")
                published = (
                    datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC")
                    if ts else "unknown"
                )
                link = a.get("link", "")
                related = a.get("relatedTickers", [])

            if title:
                articles.append({
                    "title": title,
                    "publisher": publisher,
                    "published_at": published,
                    "related_tickers": related,
                    "link": link,
                })
        if articles:
            return articles
    except Exception:
        pass

    # yfinance returned nothing — try Polygon.io
    return _polygon_get_news(ticker, limit)


def get_earnings_calendar(ticker: str) -> dict:
    """
    Return upcoming earnings date with consensus EPS/revenue estimates,
    plus the last 4 quarters' beat/miss history.
    Handles yfinance <1.0 and >=1.0 calendar formats.
    """
    try:
        t = yf.Ticker(ticker.upper())
        result: dict = {"ticker": ticker.upper()}

        # ── Upcoming earnings date ────────────────────────────────────────────
        result["next_earnings_date"]    = None
        result["eps_estimate_avg"]      = None
        result["eps_estimate_low"]      = None
        result["eps_estimate_high"]     = None
        result["revenue_estimate_avg"]  = None
        result["revenue_estimate_low"]  = None
        result["revenue_estimate_high"] = None
        try:
            cal = t.calendar
            if isinstance(cal, dict) and cal:
                # yfinance < 1.0 and some 1.x versions return a plain dict
                dates = cal.get("Earnings Date") or cal.get("earningsDate") or []
                if dates:
                    first = dates[0]
                    result["next_earnings_date"] = (
                        str(first.date()) if hasattr(first, "date") else str(first)
                    )

                def _safe(key, alt=None):
                    v = cal.get(key) or (cal.get(alt) if alt else None)
                    return float(v) if v is not None else None

                result["eps_estimate_avg"]      = _safe("Earnings Average", "epsAverage")
                result["eps_estimate_low"]       = _safe("Earnings Low", "epsLow")
                result["eps_estimate_high"]      = _safe("Earnings High", "epsHigh")
                result["revenue_estimate_avg"]   = _safe("Revenue Average", "revenueAverage")
                result["revenue_estimate_low"]   = _safe("Revenue Low", "revenueLow")
                result["revenue_estimate_high"]  = _safe("Revenue High", "revenueHigh")
        except Exception:
            pass  # defaults already set above

        # ── Historical EPS beat/miss (last 4 quarters) ───────────────────────
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                # Normalise column names (they differ across yfinance versions)
                col_map = {}
                for col in ed.columns:
                    lc = col.lower().replace(" ", "").replace("_", "")
                    if "reportedeps" in lc or "reportedearnings" in lc:
                        col_map["reported"] = col
                    elif "epsestimate" in lc or "estimate" in lc:
                        col_map["estimate"] = col
                    elif "surprise" in lc:
                        col_map["surprise"] = col

                rep_col = col_map.get("reported", "Reported EPS")
                est_col = col_map.get("estimate", "EPS Estimate")
                sur_col = col_map.get("surprise", "Surprise(%)")

                if rep_col in ed.columns:
                    past = ed.dropna(subset=[rep_col]).head(4)
                else:
                    past = ed.head(4)

                history = []
                for date, row in past.iterrows():
                    estimated = row.get(est_col)
                    reported  = row.get(rep_col)
                    surprise  = row.get(sur_col)
                    try:
                        estimated = float(estimated) if estimated is not None else None
                        reported  = float(reported)  if reported  is not None else None
                        surprise  = float(surprise)  if surprise  is not None else None
                    except (TypeError, ValueError):
                        estimated = reported = surprise = None

                    history.append({
                        "date":         str(date.date()) if hasattr(date, "date") else str(date),
                        "eps_estimate": round(estimated, 4) if estimated is not None else None,
                        "eps_reported": round(reported, 4)  if reported  is not None else None,
                        "surprise_pct": round(surprise, 2)  if surprise  is not None else None,
                        "beat": (reported > estimated) if (reported is not None and estimated is not None) else None,
                    })
                result["last_4_quarters"] = history
            else:
                result["last_4_quarters"] = []
        except Exception:
            result["last_4_quarters"] = []

        return result
    except Exception as e:
        return {"error": str(e)}


def get_analyst_upgrades(ticker: str, limit: int = 10) -> list[dict]:
    """
    Return recent analyst upgrades and downgrades for a stock.
    Includes firm name, action (upgrade/downgrade/init), and grade change.
    Handles column name variations across yfinance versions.
    """
    try:
        t = yf.Ticker(ticker.upper())
        df = t.upgrades_downgrades
        if df is None or df.empty:
            return [{"note": f"No recent analyst actions found for {ticker.upper()}"}]

        df = df.head(limit)

        # Normalise column names (yfinance 1.x lowercased some)
        col = {c.lower(): c for c in df.columns}
        firm_col       = col.get("firm",       col.get("firm",       None))
        action_col     = col.get("action",     col.get("action",     None))
        from_col       = col.get("fromgrade",  col.get("from grade", None))
        to_col         = col.get("tograde",    col.get("to grade",   None))

        results = []
        for date, row in df.iterrows():
            results.append({
                "date":       str(date.date()) if hasattr(date, "date") else str(date),
                "firm":       row.get(firm_col,   row.get("Firm",      "")),
                "action":     row.get(action_col, row.get("Action",    "")),
                "from_grade": row.get(from_col,   row.get("FromGrade", "")),
                "to_grade":   row.get(to_col,     row.get("ToGrade",   "")),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def get_analyst_consensus(ticker: str) -> dict:
    """
    Return aggregated Wall Street analyst consensus: rating distribution, price targets,
    upside to mean target, and EPS revision momentum (raises vs cuts last 30 days).
    """
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info or {}
        result = {"ticker": ticker.upper()}

        # ── Price targets ────────────────────────────────────────────────────
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        result["current_price"]       = current_price
        result["target_mean_price"]   = info.get("targetMeanPrice")
        result["target_high_price"]   = info.get("targetHighPrice")
        result["target_low_price"]    = info.get("targetLowPrice")
        result["target_median_price"] = info.get("targetMedianPrice")
        result["num_analysts"]        = info.get("numberOfAnalystOpinions")
        result["recommendation_key"]  = info.get("recommendationKey")   # "buy","hold","sell"
        result["recommendation_mean"] = info.get("recommendationMean")  # 1=StrongBuy 5=StrongSell

        if current_price and result.get("target_mean_price"):
            result["upside_to_mean_pct"] = round(
                (result["target_mean_price"] / current_price - 1) * 100, 1
            )

        # ── Rating distribution ──────────────────────────────────────────────
        try:
            recs = t.recommendations_summary
            if recs is not None and not recs.empty:
                # Normalise column names (varies across yfinance versions)
                col = {c.lower().replace(" ", "").replace("_", ""): c for c in recs.columns}
                latest = recs.iloc[0]

                def _rating(keys):
                    for k in keys:
                        c = col.get(k)
                        if c is not None:
                            v = latest.get(c)
                            if v is not None:
                                try:
                                    return int(float(v))
                                except (TypeError, ValueError):
                                    pass
                    return 0

                sb = _rating(["strongbuy",  "strongbuy"])
                b  = _rating(["buy"])
                h  = _rating(["hold"])
                s  = _rating(["sell"])
                ss = _rating(["strongsell", "strongsell"])
                total = sb + b + h + s + ss
                result["rating_distribution"] = {
                    "strong_buy":   sb,
                    "buy":          b,
                    "hold":         h,
                    "sell":         s,
                    "strong_sell":  ss,
                    "total":        total,
                }
                if total > 0:
                    result["pct_bullish"] = round((sb + b) / total * 100, 1)
                    result["pct_bearish"] = round((s + ss) / total * 100, 1)
        except Exception:
            pass

        # ── EPS revision momentum ────────────────────────────────────────────
        try:
            rev = t.eps_revisions
            if rev is not None and not rev.empty:
                # Normalise index and column names
                rev.index = [str(i).lower().replace(" ", "").replace("_", "") for i in rev.index]
                col_map = {c: c for c in rev.columns}

                def _rev_val(row_key, col):
                    try:
                        if row_key in rev.index:
                            v = rev.at[row_key, col]
                            return int(float(v)) if v is not None else None
                    except Exception:
                        pass
                    return None

                revisions = {}
                for c in rev.columns:
                    period = str(c)
                    revisions[period] = {
                        "up_last_7d":  _rev_val("uplast7days",   c),
                        "up_last_30d": _rev_val("uplast30days",  c),
                        "dn_last_7d":  _rev_val("downlast7days", c),
                        "dn_last_30d": _rev_val("downlast30days", c),
                    }
                result["eps_revisions"] = revisions

                # Summarise: net revision direction for next year
                for period_key in ["nextyear", "0y", "1y"]:
                    if period_key in [str(c).lower().replace(" ", "") for c in rev.columns]:
                        col_match = next(
                            (c for c in rev.columns
                             if str(c).lower().replace(" ", "") == period_key), None
                        )
                        if col_match:
                            up   = (_rev_val("uplast30days",  col_match) or 0)
                            down = (_rev_val("downlast30days", col_match) or 0)
                            net  = up - down
                            result["next_year_revision_net"] = net
                            result["next_year_revision_direction"] = (
                                "positive" if net > 0 else "negative" if net < 0 else "neutral"
                            )
                            break
        except Exception:
            pass

        return result
    except Exception as e:
        return {"error": str(e)}


def get_financial_history(ticker: str) -> dict:
    """
    Return 4-5 years of annual financial history: revenue, margins, FCF, debt.
    Use this to assess long-run business quality and earnings power trends.
    """
    try:
        t = yf.Ticker(ticker.upper())

        def _row(df, *keys):
            """Return list of floats (newest first, up to 5 cols) for the first matching key."""
            if df is None or df.empty:
                return []
            for key in keys:
                matches = [i for i in df.index if str(i).lower() == key.lower()]
                if matches:
                    row = df.loc[matches[0]]
                    vals = []
                    for v in row.iloc[:5]:
                        try:
                            vals.append(float(v))
                        except (TypeError, ValueError):
                            vals.append(None)
                    return vals
            return []

        fin = t.financials
        cf  = t.cashflow
        bs  = t.balance_sheet

        years = []
        if fin is not None and not fin.empty:
            years = [str(c)[:4] for c in fin.columns[:5]]
        elif cf is not None and not cf.empty:
            years = [str(c)[:4] for c in cf.columns[:5]]

        revenue      = _row(fin, "Total Revenue")
        gross_profit = _row(fin, "Gross Profit")
        op_income    = _row(fin, "Operating Income", "Ebit")
        net_income   = _row(fin, "Net Income")
        op_cf        = _row(cf,  "Operating Cash Flow", "Total Cash From Operating Activities")
        capex        = _row(cf,  "Capital Expenditure", "Capital Expenditures")
        total_debt   = _row(bs,  "Total Debt", "Long Term Debt")
        cash         = _row(bs,  "Cash And Cash Equivalents",
                                 "Cash Cash Equivalents And Short Term Investments")

        def _pct(num, denom):
            out = []
            for i in range(max(len(num), len(denom))):
                n_val = num[i] if i < len(num) else None
                d_val = denom[i] if i < len(denom) else None
                if n_val is not None and d_val and d_val != 0:
                    out.append(round(n_val / d_val * 100, 1))
                else:
                    out.append(None)
            return out

        gross_margin = _pct(gross_profit, revenue)
        op_margin    = _pct(op_income,    revenue)
        net_margin   = _pct(net_income,   revenue)

        rev_growth = []
        for i in range(len(revenue) - 1):
            curr, prev = revenue[i], revenue[i + 1]
            if curr is not None and prev and prev != 0:
                rev_growth.append(round((curr - prev) / abs(prev) * 100, 1))
            else:
                rev_growth.append(None)
        rev_growth.append(None)

        # capex is typically negative in yfinance; FCF = op_cf + capex
        fcf = []
        for i in range(max(len(op_cf), len(capex))):
            oc = op_cf[i]  if i < len(op_cf)  else None
            cx = capex[i]  if i < len(capex)  else None
            if oc is not None and cx is not None:
                fcf.append(round(oc + cx))
            elif oc is not None:
                fcf.append(round(oc))
            else:
                fcf.append(None)

        records = []
        for i, yr in enumerate(years):
            def _g(lst, idx=i):
                return lst[idx] if idx < len(lst) else None
            records.append({
                "year":                 yr,
                "revenue":              _g(revenue),
                "gross_profit":         _g(gross_profit),
                "operating_income":     _g(op_income),
                "net_income":           _g(net_income),
                "gross_margin_pct":     _g(gross_margin),
                "operating_margin_pct": _g(op_margin),
                "net_margin_pct":       _g(net_margin),
                "revenue_growth_pct":   _g(rev_growth),
                "operating_cf":         _g(op_cf),
                "free_cash_flow":       _g(fcf),
                "total_debt":           _g(total_debt),
                "cash":                 _g(cash),
            })

        return {
            "ticker":          ticker.upper(),
            "years_available": len(years),
            "history":         records,
        }
    except Exception as e:
        return {"error": str(e)}


def get_insider_activity(ticker: str) -> dict:
    """
    Return recent insider transactions (buys/sells by executives and directors).
    High insider buying is often a bullish signal; heavy selling can be a warning.
    Handles column name variations across yfinance versions.
    """
    try:
        t = yf.Ticker(ticker.upper())
        df = t.insider_transactions
        if df is None or df.empty:
            return {
                "ticker": ticker.upper(),
                "transactions": [],
                "note": "No recent insider transactions found.",
            }

        # Build a lowercase → actual column name map
        col = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

        def _col(*candidates):
            """Return the first matching actual column name, or None."""
            for c in candidates:
                if c in col:
                    return col[c]
            return None

        date_col     = _col("startdate", "date", "transactiondate")
        shares_col   = _col("shares", "sharestransacted")
        value_col    = _col("value", "transactionvalue")
        text_col     = _col("text", "transaction", "transactiontext")
        insider_col  = _col("insider", "reportingname", "name")
        position_col = _col("position", "relationship", "title")

        transactions = []
        for _, row in df.head(15).iterrows():
            date_val = row.get(date_col) if date_col else None
            if hasattr(date_val, "date"):
                date_val = str(date_val.date())
            elif date_val is not None:
                date_val = str(date_val)[:10]

            shares = row.get(shares_col) if shares_col else None
            value  = row.get(value_col)  if value_col  else None
            text   = row.get(text_col)   if text_col   else ""

            try:
                shares = int(float(shares)) if shares is not None else None
            except (TypeError, ValueError):
                shares = None
            try:
                value = round(float(value)) if value is not None else None
            except (TypeError, ValueError):
                value = None

            transactions.append({
                "date":        date_val,
                "insider":     str(row.get(insider_col,  "")) if insider_col  else "",
                "position":    str(row.get(position_col, "")) if position_col else "",
                "transaction": str(text)[:120],
                "shares":      shares,
                "value_usd":   value,
            })

        tx_text = " ".join(str(tx.get("transaction", "")).lower() for tx in transactions)
        buy_count  = tx_text.count("purchase") + tx_text.count("buy")
        sell_count = tx_text.count("sale") + tx_text.count("sell")

        return {
            "ticker": ticker.upper(),
            "recent_buy_transactions":  buy_count,
            "recent_sell_transactions": sell_count,
            "transactions": transactions,
        }
    except Exception as e:
        return {"error": str(e)}


def get_hedge_recommendations(equity_pct: Optional[float] = None, cash_pct: Optional[float] = None) -> dict:
    """
    Translate the current macro regime into concrete defensive ETF hedge recommendations.

    Fetches macro data internally. Accepts optional equity_pct / cash_pct
    (from the caller's portfolio state) to personalise the allocation sizing.

    Hedge universe is restricted to plain, non-leveraged, non-inverse ETFs:
      TLT  — iShares 20+ Year Treasury Bond ETF  (RISK_OFF flight-to-safety)
      IEF  — iShares 7-10 Year Treasury Bond ETF  (moderate duration hedge)
      SHV  — iShares Short Treasury Bond ETF      (cash equivalent with yield)
      GLD  — SPDR Gold Trust                       (inflation + crisis hedge)
      TIP  — iShares TIPS Bond ETF                 (inflation-protected bonds)
      GSG  — iShares S&P GSCI Commodity ETF        (broad commodity inflation hedge)

    Never recommends leveraged, inverse, or single-commodity futures ETFs
    — they decay and are inconsistent with a long-term value approach.
    """
    # ── Fetch current macro regime ────────────────────────────────────────────
    macro = get_macro_environment()
    rates = macro.get("rates", {})
    sentiment = macro.get("sentiment", {})
    commodities = macro.get("commodities", {})
    dollar = macro.get("dollar", {})

    vix = sentiment.get("vix")
    yield_spread = rates.get("yield_curve_spread")        # positive = normal, negative = inverted
    ten_yr = rates.get("ten_yr_treasury_yield_pct")
    oil = commodities.get("oil_wti_usd")
    gold_price = commodities.get("gold_usd")
    dxy = dollar.get("dxy")

    # ── Detect regime ─────────────────────────────────────────────────────────
    # Mirrors the logic in ml_insights.py for consistency
    risk_off = bool(vix and vix > 25) or bool(yield_spread is not None and yield_spread < 0)
    high_rates = bool(ten_yr and ten_yr > 4.5)
    inflationary = bool(oil and oil > 85)   # crude >$85 signals inflation pressure
    stagflation = risk_off and inflationary

    if stagflation:
        regime = "STAGFLATION"
    elif risk_off:
        regime = "RISK_OFF"
    elif inflationary:
        regime = "INFLATIONARY"
    elif high_rates:
        regime = "HIGH_RATES"
    else:
        regime = "NORMAL"

    # ── Hedge ETF definitions ─────────────────────────────────────────────────
    # Each entry: ticker, name, category, base alloc %, regimes it applies to,
    # entry / exit trigger text, rationale
    _HEDGE_UNIVERSE = [
        {
            "ticker": "TLT",
            "name": "iShares 20+ Year Treasury Bond ETF",
            "category": "long_bonds",
            "base_alloc_pct": 8,
            "regimes": {"RISK_OFF", "STAGFLATION"},
            "rationale": (
                "Long-duration Treasuries are the classic flight-to-safety asset. "
                "When equities sell off in a RISK_OFF regime, bond yields typically fall "
                "and TLT prices rise — providing an inverse correlation that cushions drawdowns."
            ),
            "entry_signal": "VIX > 25 and/or yield curve inverted",
            "exit_signal": "VIX sustainably below 20 AND yield curve re-normalises (spread > 0.3%)",
        },
        {
            "ticker": "IEF",
            "name": "iShares 7-10 Year Treasury Bond ETF",
            "category": "medium_duration_bonds",
            "base_alloc_pct": 5,
            "regimes": {"RISK_OFF", "HIGH_RATES"},
            "rationale": (
                "Intermediate Treasuries carry less duration risk than TLT but still provide "
                "meaningful defensive exposure. In a HIGH_RATES regime, medium-duration bonds "
                "are preferable to long-duration (TLT) to avoid excessive interest-rate sensitivity."
            ),
            "entry_signal": "RISK_OFF regime or 10yr yield > 4.5% (as a duration-safe bond allocation)",
            "exit_signal": "Regime normalises or 10yr yield falls below 3.5%",
        },
        {
            "ticker": "SHV",
            "name": "iShares Short Treasury Bond ETF",
            "category": "cash_equivalent",
            "base_alloc_pct": 5,
            "regimes": {"RISK_OFF", "HIGH_RATES", "STAGFLATION"},
            "rationale": (
                "Sub-1-year T-bills: near-zero duration risk, earns the risk-free rate. "
                "In a RISK_OFF or HIGH_RATES regime, parking cash in SHV earns the prevailing "
                "short rate (currently elevated) rather than leaving it idle. Not a hedge — "
                "a yield-generating cash alternative."
            ),
            "entry_signal": "Any risk-reducing regime; especially when short-term rates are elevated",
            "exit_signal": "When attractive long equity opportunities appear or rates fall significantly",
        },
        {
            "ticker": "GLD",
            "name": "SPDR Gold Trust",
            "category": "gold",
            "base_alloc_pct": 5,
            "regimes": {"RISK_OFF", "INFLATIONARY", "STAGFLATION"},
            "rationale": (
                "Gold holds real value during inflation (unlike nominal bonds) and also "
                "rallies in RISK_OFF / flight-to-safety episodes. Most useful when facing "
                "STAGFLATION — where bonds lose to inflation but gold typically holds purchasing power."
            ),
            "entry_signal": "RISK_OFF + inflation rising, or STAGFLATION regime",
            "exit_signal": "Inflation falls below 3% and equity risk premium normalises",
        },
        {
            "ticker": "TIP",
            "name": "iShares TIPS Bond ETF",
            "category": "inflation_protected_bonds",
            "base_alloc_pct": 5,
            "regimes": {"INFLATIONARY", "STAGFLATION"},
            "rationale": (
                "Treasury Inflation-Protected Securities adjust principal for CPI, "
                "preserving real bond value during inflationary periods. Preferable to nominal "
                "bonds (TLT) in an INFLATIONARY regime where inflation erodes fixed coupons."
            ),
            "entry_signal": "CPI trending above 3.5% and rising, or oil > $85",
            "exit_signal": "CPI falls below 3% on a sustained basis",
        },
        {
            "ticker": "GSG",
            "name": "iShares S&P GSCI Commodity ETF",
            "category": "commodities",
            "base_alloc_pct": 3,
            "regimes": {"INFLATIONARY"},
            "rationale": (
                "Broad commodity basket (energy, metals, agriculture) — the underlying driver "
                "of inflation itself. Provides a partial natural hedge: when commodity prices "
                "drive CPI higher, GSG rises. Small allocation only — commodities are volatile "
                "and carry no intrinsic yield."
            ),
            "entry_signal": "Oil > $85 and CPI > 4% and rising",
            "exit_signal": "Commodity prices and CPI trend reverse",
        },
    ]

    # ── Build recommendations for the current regime ──────────────────────────
    applicable = [h for h in _HEDGE_UNIVERSE if regime in h["regimes"]]

    # Scale allocations based on equity concentration and regime severity
    # More equity exposure + worse regime = more aggressive hedging
    severity_multiplier = 1.0
    if regime == "STAGFLATION":
        severity_multiplier = 1.5
    elif regime == "RISK_OFF" and vix and vix > 35:
        severity_multiplier = 1.4   # crisis-level VIX
    elif regime == "RISK_OFF":
        severity_multiplier = 1.1

    # If caller provided equity_pct, scale up if heavily invested
    concentration_multiplier = 1.0
    if equity_pct is not None:
        if equity_pct > 90:
            concentration_multiplier = 1.3
        elif equity_pct > 80:
            concentration_multiplier = 1.1
        elif equity_pct < 60:
            concentration_multiplier = 0.7   # already defensive

    recommendations = []
    total_hedge_pct = 0
    for h in applicable:
        alloc = round(h["base_alloc_pct"] * severity_multiplier * concentration_multiplier, 1)
        # Hard cap: no single hedge ETF > 10%, total hedge cap handled below
        alloc = min(alloc, 10.0)
        total_hedge_pct += alloc
        recommendations.append({
            "ticker": h["ticker"],
            "name": h["name"],
            "category": h["category"],
            "suggested_allocation_pct": alloc,
            "rationale": h["rationale"],
            "entry_signal": h["entry_signal"],
            "exit_signal": h["exit_signal"],
        })

    # Hard cap: total hedge ≤ 20% of portfolio (long-term value strategy, not macro trading)
    if total_hedge_pct > 20 and recommendations:
        scale = 20 / total_hedge_pct
        for r in recommendations:
            r["suggested_allocation_pct"] = round(r["suggested_allocation_pct"] * scale, 1)
        total_hedge_pct = 20.0

    # ── Hedge warranted? ──────────────────────────────────────────────────────
    # Hedges are insurance for equity positions. With less than 25% in equities
    # the portfolio is already overwhelmingly cash-defensive; hedging a small
    # equity sliver adds noise without meaningful protection.
    if equity_pct is not None and equity_pct < 25:
        return {
            "regime": regime,
            "hedge_warranted": False,
            "no_hedge_rationale": (
                f"Equity exposure is only {equity_pct:.0f}% — the portfolio is already "
                "predominantly cash, which is the most effective defensive position. "
                "Hedges protect significant equity holdings; build equity positions to ≥25% "
                "before layering on cross-asset insurance."
            ),
            "recommendations": [],
            "total_recommended_hedge_pct": 0,
            "equity_pct_input": equity_pct,
            "cash_pct_input": cash_pct,
            "regime_signals": [],
        }

    hedge_warranted = regime in ("RISK_OFF", "INFLATIONARY", "STAGFLATION", "HIGH_RATES")
    no_hedge_rationale = None
    if not hedge_warranted:
        no_hedge_rationale = (
            f"Current regime is {regime} — no macro trigger for defensive hedging. "
            "Holding cash is sufficient; deploying into hedge ETFs would dilute equity returns "
            "without a clear compensating risk-reduction benefit."
        )
        recommendations = []
        total_hedge_pct = 0

    # ── Cash check ────────────────────────────────────────────────────────────
    # Hedges should ALWAYS be funded from cash, not by selling equity
    funding_note = None
    if cash_pct is not None and hedge_warranted:
        if cash_pct < total_hedge_pct:
            funding_note = (
                f"Warning: recommended hedge allocation ({total_hedge_pct:.0f}%) exceeds "
                f"current cash ({cash_pct:.0f}%). Fund hedges ONLY from cash — do NOT sell "
                "equity positions to finance hedging. Scale back recommendations to fit available cash."
            )
            # Scale down to fit available cash (with 5% buffer for equity buys)
            available = max(0, cash_pct - 5)
            if available < total_hedge_pct and available > 0:
                scale = available / total_hedge_pct
                for r in recommendations:
                    r["suggested_allocation_pct"] = round(r["suggested_allocation_pct"] * scale, 1)
                total_hedge_pct = round(available, 1)
        else:
            funding_note = (
                f"Sufficient cash ({cash_pct:.0f}%) available to fund recommended "
                f"hedge allocation ({total_hedge_pct:.0f}%) without selling any equity."
            )

    # ── Key signals used ─────────────────────────────────────────────────────
    regime_signals = []
    if vix:
        regime_signals.append(f"VIX {vix:.1f} ({'elevated' if vix > 25 else 'normal'})")
    if yield_spread is not None:
        regime_signals.append(
            f"Yield curve {yield_spread:+.2f}% ({'inverted' if yield_spread < 0 else 'normal'})"
        )
    if ten_yr:
        regime_signals.append(f"10yr Treasury {ten_yr:.1f}%")
    if oil:
        regime_signals.append(f"WTI crude ${oil:.0f}")

    return {
        "regime": regime,
        "hedge_warranted": hedge_warranted,
        "equity_pct_input": equity_pct,
        "cash_pct_input": cash_pct,
        "total_recommended_hedge_pct": total_hedge_pct,
        "recommendations": recommendations,
        "no_hedge_rationale": no_hedge_rationale,
        "funding_note": funding_note,
        "regime_signals": regime_signals,
        "philosophy": (
            "Hedges are funded exclusively from cash, not by selling equity. "
            "Maximum hedge allocation is 20% of total portfolio. "
            "These are defensive positions to reduce drawdown in stress regimes — "
            "not macro bets or yield plays. Unwind when the triggering regime resolves."
        ),
    }


def get_macro_environment() -> dict:
    """
    Fetch key macroeconomic indicators: Treasury yields, yield curve,
    dollar index, oil, and gold. Includes a brief text interpretation
    to help the agent understand the current regime.
    """
    def _fetch(ticker: str) -> tuple[Optional[float], Optional[float]]:
        """Returns (price, change_pct) or (None, None) on failure."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            price = info.get("regularMarketPrice") or info.get("previousClose")
            prev = info.get("regularMarketPreviousClose") or info.get("previousClose")
            change = ((price - prev) / prev * 100) if price and prev and prev != 0 else None
            return price, (round(change, 2) if change is not None else None)
        except Exception:
            return None, None

    result: dict = {}

    # Treasury yields
    y10, y10c = _fetch("^TNX")   # 10-year yield (in percent, e.g. 4.5 = 4.5%)
    y2, y2c = _fetch("^IRX")     # 13-week T-bill as short-end proxy
    result["rates"] = {
        "ten_yr_treasury_yield_pct": y10,
        "two_yr_treasury_yield_pct": y2,
        "yield_curve_spread": round(y10 - y2, 3) if y10 and y2 else None,
        "yield_curve_status": (
            "inverted (recession signal)" if y10 and y2 and y10 < y2
            else "normal" if y10 and y2
            else "unknown"
        ),
    }

    # Dollar index
    dxy, dxyc = _fetch("DX-Y.NYB")
    result["dollar"] = {
        "dxy": dxy,
        "dxy_change_pct": dxyc,
        "note": "Strong dollar hurts multinational earnings and emerging markets",
    }

    # Commodities
    oil, oilc = _fetch("CL=F")
    gold, goldc = _fetch("GC=F")
    result["commodities"] = {
        "oil_wti_usd": oil,
        "oil_change_pct": oilc,
        "gold_usd": gold,
        "gold_change_pct": goldc,
    }

    # VIX (market fear gauge)
    vix, vixc = _fetch("^VIX")
    if vix:
        if vix < 15:
            vix_regime = "low fear — complacency risk, markets pricing in low volatility"
        elif vix < 25:
            vix_regime = "moderate — normal market conditions"
        elif vix < 35:
            vix_regime = "elevated fear — increased uncertainty, consider position sizing carefully"
        else:
            vix_regime = "extreme fear — crisis conditions, potential opportunity for patient investors"
    else:
        vix_regime = "unknown"

    result["sentiment"] = {
        "vix": vix,
        "vix_change_pct": vixc,
        "vix_regime": vix_regime,
    }

    # Synthesised interpretation — specific signals worth noting
    signals = []
    if result["rates"]["yield_curve_status"].startswith("inverted"):
        signals.append("yield curve is inverted — historically precedes recessions by 12-18 months")
    if y10 and y10 > 4.5:
        signals.append(f"10yr yield at {y10:.1f}% — high rates compress growth stock valuations, favour value/financials")
    if dxy and dxy > 105:
        signals.append(f"dollar index at {dxy:.1f} — strong dollar is a headwind for US multinationals")
    if oil and oil > 90:
        signals.append(f"oil at ${oil:.0f} — elevated energy costs pressure consumer spending and margins")
    if vix and vix > 25:
        signals.append(f"VIX at {vix:.1f} — consider tighter position sizing until volatility subsides")

    result["key_signals"] = signals if signals else ["No major macro warning signals detected"]

    return result


def get_sector_exposure(holdings: list[dict]) -> dict:
    """
    Compute current sector breakdown of the portfolio by market value.
    holdings: list of {ticker, shares, avg_cost} dicts (from get_holdings()).
    Returns sector weights so the agent can see concentration before new buys.
    """
    if not holdings:
        return {"sector_breakdown": [], "total_invested": 0, "note": "No holdings."}

    def _fetch(h: dict) -> dict:
        ticker = h["ticker"]
        try:
            info = yf.Ticker(ticker).info
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            sector = info.get("sector") or "Unknown"
            market_value = h["shares"] * price if price else h["shares"] * h["avg_cost"]
        except Exception:
            sector = "Unknown"
            market_value = h["shares"] * h["avg_cost"]
        return {"ticker": ticker, "sector": sector, "market_value": market_value}

    holding_data: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch, h): h for h in holdings}
        for future in as_completed(futures):
            holding_data.append(future.result())

    total_value = sum(h["market_value"] for h in holding_data)

    sector_map: dict[str, dict] = {}
    for h in holding_data:
        s = h["sector"]
        if s not in sector_map:
            sector_map[s] = {"tickers": [], "total_value": 0.0}
        sector_map[s]["tickers"].append(h["ticker"])
        sector_map[s]["total_value"] += h["market_value"]

    breakdown = sorted(
        [
            {
                "sector": sector,
                "tickers": sorted(data["tickers"]),
                "market_value": round(data["total_value"], 2),
                "weight_pct": round(data["total_value"] / total_value * 100, 1) if total_value else 0,
            }
            for sector, data in sector_map.items()
        ],
        key=lambda x: -x["market_value"],
    )

    return {
        "sector_breakdown": breakdown,
        "total_invested": round(total_value, 2),
        "note": (
            "Weights are % of invested value (cash excluded). "
            "Use this before new buys to avoid over-concentrating in one sector."
        ),
    }


def get_international_universe(region: Optional[str] = None) -> dict:
    """
    Return a curated list of major international stocks for screening.

    Covers ~200 top companies across Europe, Asia-Pacific, Latin America,
    Canada, and India/Israel. Mix of:
      - US-listed ADRs / direct NYSE/NASDAQ listings (highest yfinance data quality)
      - Foreign-listed tickers with exchange suffixes (.L, .DE, .AS, .PA, .SW,
        .T, .HK, .NS, .KS) for companies with no US listing

    All tickers are accessible via yfinance. Pass the returned tickers list
    directly to screen_stocks for fundamental screening.

    region options (case-insensitive):
      "europe"  — UK, Germany, France, Switzerland, Netherlands, Sweden, Denmark
      "asia"    — Japan, South Korea, Hong Kong/China, Taiwan, Singapore, Australia
      "latam"   — Brazil, Mexico, Chile, Colombia
      "canada"  — Canadian companies
      "india"   — India + Israel tech
      None      — all regions combined
    """
    # ── Europe ────────────────────────────────────────────────────────────────
    # Mix of US ADRs and direct EU listings. EU-suffix tickers may have slightly
    # less complete yfinance fundamental data but price history is reliable.
    europe = [
        # UK (no suffix = US-listed; .L = London)
        "SHEL",   # Shell (NYSE)
        "BP",     # BP (NYSE)
        "AZN",    # AstraZeneca (NASDAQ)
        "GSK",    # GSK (NYSE)
        "LSXMA",  # Liberty Media (skip — use RIO for diversification)
        "RIO",    # Rio Tinto (NYSE)
        "HSBC",   # HSBC (NYSE)
        "BCS",    # Barclays (NYSE)
        "VOD",    # Vodafone (NASDAQ)
        "RR.L",   # Rolls-Royce (London)
        "ULVR.L", # Unilever (London)
        "DGE.L",  # Diageo (London)
        "BATS.L", # BAT (London)
        "REL.L",  # RELX (London)
        "NG.L",   # National Grid (London)
        # Germany
        "SAP",    # SAP (NYSE)
        "SIE.DE", # Siemens (Frankfurt)
        "ALV.DE", # Allianz (Frankfurt)
        "BMW.DE", # BMW (Frankfurt)
        "VOW3.DE",# Volkswagen (Frankfurt)
        "BAYN.DE",# Bayer (Frankfurt)
        "MBG.DE", # Mercedes-Benz (Frankfurt)
        "DTE.DE", # Deutsche Telekom (Frankfurt)
        "BAS.DE", # BASF (Frankfurt)
        "ADS.DE", # Adidas (Frankfurt)
        # France
        "LVMUY",  # LVMH (US ADR)
        "LRLCY",  # L'Oréal (US ADR)
        "SNYNF",  # Sanofi (US ADR)
        "AIR.PA", # Airbus (Paris)
        "TTE",    # TotalEnergies (NYSE)
        "BNP.PA", # BNP Paribas (Paris)
        "MC.PA",  # LVMH (Paris — use if ADR has thin data)
        # Switzerland
        "NESN.SW",# Nestlé (Swiss)
        "ROG.SW", # Roche (Swiss)
        "NOVN.SW",# Novartis (Swiss)
        "ABBN.SW",# ABB (Swiss)
        "ZURN.SW",# Zurich Insurance (Swiss)
        # Netherlands
        "ASML",   # ASML (NASDAQ — already in broad but too important to miss)
        "PHIA.AS",# Philips (Amsterdam)
        "HEIA.AS",# Heineken (Amsterdam)
        "INGA.AS",# ING (Amsterdam)
        # Sweden / Denmark / Norway
        "NVO",    # Novo Nordisk (NYSE)
        "VOLV-B.ST", # Volvo B (Stockholm)
        "ATCO-A.ST", # Atlas Copco (Stockholm)
        "ERICB.ST",  # Ericsson B (Stockholm)
        "NESTE.HE",  # Neste (Helsinki)
        # Spain / Italy
        "SAN",    # Santander (NYSE)
        "BBVA",   # BBVA (NYSE)
        "TEF",    # Telefónica (NYSE)
        "ENEL.MI",# Enel (Milan)
        "ENI.MI", # ENI (Milan)
    ]

    # ── Asia-Pacific ──────────────────────────────────────────────────────────
    asia = [
        # Taiwan
        "TSM",    # TSMC (NYSE ADR — most important semiconductor)
        "UMC",    # United Microelectronics (NYSE)
        # Japan
        "TM",     # Toyota (NYSE ADR)
        "SONY",   # Sony (NYSE ADR)
        "HMC",    # Honda (NYSE ADR)
        "MUFG",   # Mitsubishi UFJ (NYSE ADR)
        "NMR",    # Nomura (NYSE ADR)
        "7203.T", # Toyota (Tokyo)
        "6758.T", # Sony (Tokyo)
        "9984.T", # SoftBank (Tokyo)
        "6861.T", # Keyence (Tokyo)
        "7974.T", # Nintendo (Tokyo)
        "8306.T", # Mitsubishi UFJ (Tokyo)
        "9432.T", # NTT (Tokyo)
        "6098.T", # Recruit Holdings (Tokyo)
        "4063.T", # Shin-Etsu Chemical (Tokyo)
        # South Korea
        "005930.KS", # Samsung Electronics (Seoul)
        "000660.KS", # SK Hynix (Seoul)
        "035420.KS", # NAVER (Seoul)
        "051910.KS", # LG Chem (Seoul)
        # China / Hong Kong (ADRs)
        "BABA",   # Alibaba (NYSE ADR)
        "BIDU",   # Baidu (NASDAQ ADR)
        "JD",     # JD.com (NASDAQ ADR)
        "PDD",    # PDD/Temu (NASDAQ)
        "TCEHY",  # Tencent (US OTC ADR)
        "NTES",   # NetEase (NASDAQ)
        "VIPS",   # Vipshop (NYSE)
        "9988.HK",# Alibaba (HK)
        "0700.HK",# Tencent (HK)
        "3690.HK",# Meituan (HK)
        "9618.HK",# JD.com (HK)
        "1299.HK",# AIA Group (HK)
        "0941.HK",# China Mobile (HK)
        # Australia
        "BHP",    # BHP (NYSE ADR)
        "RIO",    # Rio Tinto (NYSE — appears in Europe too, global miner)
        "BHP.AX", # BHP (ASX)
        "CBA.AX", # Commonwealth Bank (ASX)
        "CSL.AX", # CSL Limited (ASX — biotech)
        "WDS.AX", # Woodside Energy (ASX)
        # Singapore
        "D05.SI", # DBS Bank (Singapore)
        "O39.SI", # OCBC Bank (Singapore)
        "C6L.SI", # Singapore Airlines (Singapore)
    ]

    # ── Latin America ─────────────────────────────────────────────────────────
    latam = [
        "MELI",   # MercadoLibre (NASDAQ)
        "NU",     # Nubank (NYSE)
        "VALE",   # Vale (NYSE ADR)
        "PBR",    # Petrobras (NYSE ADR)
        "ITUB",   # Itaú Unibanco (NYSE ADR)
        "BBD",    # Bradesco (NYSE ADR)
        "ABEV",   # Ambev (NYSE ADR)
        "BAP",    # Credicorp — Peru (NYSE)
        "AMX",    # América Móvil (NYSE ADR)
        "GFNORTEO.MX", # Banorte (Mexico)
        "WALMEX.MX",   # Walmart de Mexico (Mexico)
        "EC",     # Ecopetrol Colombia (NYSE ADR)
    ]

    # ── Canada ────────────────────────────────────────────────────────────────
    canada = [
        "SHOP",   # Shopify (NYSE)
        "RY",     # Royal Bank (NYSE)
        "TD",     # Toronto-Dominion (NYSE)
        "BN",     # Brookfield (NYSE)
        "BAM",    # Brookfield Asset Mgmt (NYSE)
        "CNQ",    # Canadian Natural Resources (NYSE)
        "SU",     # Suncor Energy (NYSE)
        "CP",     # Canadian Pacific Kansas City (NYSE)
        "CNI",    # Canadian National Railway (NYSE)
        "MFC",    # Manulife (NYSE)
        "SLF",    # Sun Life (NYSE)
        "BCE",    # BCE Telecom (NYSE)
        "ABX",    # Barrick Gold (NYSE)
        "WPM",    # Wheaton Precious Metals (NYSE)
        "CCO.TO", # Cameco (Toronto — uranium)
        "ATD.TO", # Alimentation Couche-Tard (Toronto)
        "CSU.TO", # Constellation Software (Toronto)
    ]

    # ── India + Israel ────────────────────────────────────────────────────────
    india = [
        # India (US ADRs)
        "INFY",   # Infosys (NYSE)
        "WIT",    # Wipro (NYSE)
        "HDB",    # HDFC Bank (NYSE ADR)
        "IBN",    # ICICI Bank (NYSE ADR)
        "SIFY",   # Sify Technologies (NASDAQ)
        "RELIANCE.NS", # Reliance Industries (NSE)
        "TCS.NS",      # Tata Consultancy Services (NSE)
        "HCLTECH.NS",  # HCL Technologies (NSE)
        "BAJFINANCE.NS",# Bajaj Finance (NSE)
        "TITAN.NS",    # Titan Company (NSE)
        # Israel tech (NASDAQ-listed)
        "CHKP",   # Check Point Software
        "NICE",   # NICE Systems
        "MNDY",   # Monday.com
        "GLBE",   # Global-E Online
        "WDAY",   # (skip — US company)
        "CYBR",   # CyberArk
    ]

    # ── Assemble ──────────────────────────────────────────────────────────────
    region_map = {
        "europe": europe,
        "asia":   asia,
        "latam":  latam,
        "canada": canada,
        "india":  india,
    }

    if region:
        key = region.lower()
        if key not in region_map:
            return {
                "error": f"Unknown region '{region}'. Valid: europe, asia, latam, canada, india"
            }
        tickers = region_map[key]
        used_regions = [key]
    else:
        # Deduplicate across regions (RIO appears in both europe and asia)
        seen: set[str] = set()
        tickers = []
        for r_tickers in region_map.values():
            for t in r_tickers:
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
        used_regions = list(region_map.keys())

    # Remove any accidental duplicates or blank entries
    tickers = [t for t in tickers if t and t.strip()]

    return {
        "tickers": tickers,
        "total_count": len(tickers),
        "regions_included": used_regions,
        "note": (
            f"{len(tickers)} major international stocks across {', '.join(used_regions)}. "
            "Mix of US-listed ADRs (NYSE/NASDAQ — best data quality) and foreign-listed tickers "
            "with exchange suffixes (.L=London, .DE=Frankfurt, .PA=Paris, .SW=Swiss, "
            ".AS=Amsterdam, .T=Tokyo, .HK=HongKong, .KS=Seoul, .AX=ASX, .NS/.BO=India). "
            "Pass the tickers list to screen_stocks in batches of 50-80 for fundamental screening. "
            "Foreign-suffix tickers may return fewer fundamentals from yfinance — "
            "if a ticker errors in screen_stocks it is silently skipped."
        ),
        "region_counts": {r: len(v) for r, v in region_map.items()},
    }


def get_stock_universe(
    index: str = "all",
    sample_n: int = 200,
    random_seed: Optional[int] = None,
    sector: Optional[str] = None,
) -> dict:
    """
    Return a random sample of tickers for US stocks fetched from public GitHub datasets.
    Returns sample_n tickers (default 200) to keep response size manageable.
    Call multiple times with different random_seed values to cover the full universe.

    index options:
      "sp500"  — ~500 S&P 500 large-cap constituents
      "broad"  — ~2700 US-listed stocks (mid + small caps included)
      "all"    — sp500 + broad combined and deduplicated

    sector: optional GICS sector filter (only applies to S&P 500 tickers which have
      sector data). Examples: "Information Technology", "Health Care", "Financials",
      "Consumer Discretionary", "Communication Services", "Industrials",
      "Consumer Staples", "Energy", "Utilities", "Real Estate", "Materials".
    """
    import urllib.request
    import csv
    import io
    import random

    SP500_URL = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
        "/main/data/constituents.csv"
    )
    BROAD_URL = (
        "https://raw.githubusercontent.com/shilewenuw/get_all_tickers"
        "/master/get_all_tickers/tickers.csv"
    )

    def _fetch_sp500() -> list[dict]:
        """Returns list of {ticker, sector} dicts."""
        with urllib.request.urlopen(SP500_URL, timeout=15) as r:
            content = r.read().decode()
        reader = csv.DictReader(io.StringIO(content))
        return [
            {
                "ticker": row["Symbol"].strip().replace(".", "-"),
                "sector": row.get("Sector", "").strip(),
            }
            for row in reader
            if row.get("Symbol", "").strip()
        ]

    def _fetch_broad() -> list[str]:
        with urllib.request.urlopen(BROAD_URL, timeout=15) as r:
            lines = r.read().decode().splitlines()
        return [line.strip() for line in lines if line.strip()]

    sp500_entries: list[dict] = []
    broad_tickers: list[str] = []
    result: dict = {}

    if index in ("sp500", "all"):
        try:
            sp500_entries = _fetch_sp500()
        except Exception as e:
            result["sp500_error"] = str(e)

    if index in ("broad", "all"):
        try:
            broad_tickers = _fetch_broad()
        except Exception as e:
            result["broad_error"] = str(e)

    # Build sector map from S&P 500 data (broad universe has no sector info)
    sector_map: dict[str, str] = {e["ticker"]: e["sector"] for e in sp500_entries}
    available_sectors = sorted(set(v for v in sector_map.values() if v))

    # Build combined deduplicated list
    all_tickers: list[str] = []
    seen: set[str] = set()
    for e in sp500_entries:
        t = e["ticker"]
        if t not in seen:
            seen.add(t)
            all_tickers.append(t)
    for t in broad_tickers:
        if t not in seen:
            seen.add(t)
            all_tickers.append(t)

    total_count = len(all_tickers)

    # Apply sector filter — matches against the S&P 500 sector map.
    # Tickers not in the S&P 500 (broad-only) are excluded when filtering by sector
    # since their sector is unknown. Use index="sp500" for pure sector-targeted screens.
    if sector:
        sector_lower = sector.lower()
        filtered = [t for t in all_tickers if sector_lower in sector_map.get(t, "").lower()]
        if filtered:
            all_tickers = filtered
            result["sector_filter"] = sector
            result["sector_match_count"] = len(filtered)
        else:
            result["sector_filter_warning"] = (
                f"No S&P 500 tickers matched sector '{sector}'. "
                f"Available sectors: {available_sectors}. "
                "Returning unfiltered sample instead."
            )

    # For sp500-only queries, return the full list (≤500 tickers — manageable).
    # For broad/all queries, random-sample to keep response size reasonable.
    if index == "sp500":
        sample = all_tickers
    else:
        rng = random.Random(random_seed)
        sample = rng.sample(all_tickers, min(sample_n, len(all_tickers)))

    result["total_count"] = total_count
    result["returned_count"] = len(sample)
    result["tickers"] = sample
    sector_note = f" (sector: {sector})" if sector and "sector_filter_warning" not in result else ""
    if index == "sp500":
        result["note"] = (
            f"Returning all {len(sample)} S&P 500 tickers{sector_note}. "
            "Pass to screen_stocks in batches of 100 tickers at a time."
        )
    else:
        result["note"] = (
            f"Returning {len(sample)} tickers{sector_note} out of {len(all_tickers)} available. "
            "Pass these to screen_stocks (50-100 at a time). "
            "Call get_stock_universe again with a different random_seed to get a different batch."
        )
    if available_sectors and not sector:
        result["available_sectors"] = available_sectors
    return result


def screen_stocks(tickers: list) -> list:
    """
    Run a parallel fundamental screen.  Results accumulate in a persistent
    cache across sessions — new tickers are merged in and existing scores are
    only refreshed when stale (>7 days old).  This produces a stable,
    globally-ranked list the agent can work through sequentially across many
    sessions rather than re-ranking a different random sample each time.
    """
    today = datetime.utcnow().date().isoformat()
    stale_cutoff = (
        datetime.utcnow() - timedelta(days=_SCREENER_STALE_DAYS)
    ).date().isoformat()

    cache = _load_screener_cache()
    # Build a lookup: ticker -> result dict (includes a "cached_date" field)
    cache_by_ticker: dict = {}
    for r in cache.get("results", []):
        cache_by_ticker[r["ticker"]] = r

    upper_tickers = [t.upper() for t in tickers]

    if len(tickers) >= 50:
        # Full-universe request: only fetch tickers that are new or stale.
        tickers_to_fetch = [
            t for t in upper_tickers
            if t not in cache_by_ticker
            or cache_by_ticker[t].get("cached_date", "1970-01-01") < stale_cutoff
        ]
        if not tickers_to_fetch:
            # Everything is fresh — return the full accumulated ranked list.
            return sorted(cache_by_ticker.values(), key=lambda x: x["score"], reverse=True)
    else:
        # Subset request (e.g. watchlist re-check): fetch any not in cache or stale.
        tickers_to_fetch = [
            t for t in upper_tickers
            if t not in cache_by_ticker
            or cache_by_ticker[t].get("cached_date", "1970-01-01") < stale_cutoff
        ]
        if not tickers_to_fetch:
            requested = set(upper_tickers)
            return [r for r in sorted(cache_by_ticker.values(), key=lambda x: x["score"], reverse=True)
                    if r["ticker"] in requested]

    def _fetch(ticker: str) -> Optional[dict]:
        try:
            # Shared variables initialised with safe defaults
            name           = ticker
            sector         = ""
            industry       = ""
            recommendation = ""

            # Try FMP first for reliable fundamentals; fall back to yfinance
            fmp = _fmp_get_fundamentals(ticker)
            if fmp:
                name          = fmp.get("name") or ticker
                sector        = fmp.get("sector") or ""
                industry      = fmp.get("industry") or ""
                price         = fmp.get("_price")
                pe            = fmp.get("pe_ratio")
                forward_pe    = fmp.get("forward_pe")
                peg           = fmp.get("peg_ratio")
                revenue_growth  = fmp.get("revenue_growth")
                profit_margin   = fmp.get("profit_margin")
                roe             = fmp.get("roe")
                debt_to_equity  = fmp.get("debt_to_equity")
                market_cap      = fmp.get("_market_cap") or 0
                volume          = fmp.get("_volume") or 0
                week52_change   = fmp.get("_week52_change")
                sp52_change     = fmp.get("_sp52_change")
                free_cashflow   = fmp.get("_free_cashflow")
                fcf_yield       = fmp.get("_fcf_yield")
                if not price:
                    return None
            else:
                info = yf.Ticker(ticker).info
                name           = info.get("longName") or info.get("shortName", ticker)
                sector         = info.get("sector", "")
                industry       = info.get("industry", "")
                recommendation = info.get("recommendationKey", "")
                price = (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("previousClose")
                )
                if not price:
                    # Foreign-suffix ticker (e.g. NESN.SW, 0700.HK) with no data —
                    # try yf.Search to find a US-listed ADR/OTC equivalent.
                    if "." in ticker:
                        try:
                            base = ticker.split(".")[0]
                            _us_exchanges = {"NYQ", "NMS", "NGM", "PCX", "ASE", "OTC", "PNK"}
                            for _q in (yf.Search(base, max_results=5).quotes or []):
                                _sym = _q.get("symbol", "")
                                if _sym and _sym != ticker and "." not in _sym and _q.get("exchange") in _us_exchanges:
                                    _alt = yf.Ticker(_sym).info
                                    _alt_price = (_alt.get("currentPrice") or _alt.get("regularMarketPrice") or _alt.get("previousClose"))
                                    if _alt_price:
                                        info = _alt
                                        ticker = _sym
                                        name           = info.get("longName") or info.get("shortName", _sym)
                                        sector         = info.get("sector", "")
                                        industry       = info.get("industry", "")
                                        recommendation = info.get("recommendationKey", "")
                                        price = _alt_price
                                        break
                        except Exception:
                            pass
                    if not price:
                        return None
                pe             = info.get("trailingPE")
                forward_pe     = info.get("forwardPE")
                peg            = info.get("pegRatio")
                revenue_growth = info.get("revenueGrowth")
                profit_margin  = info.get("profitMargins")
                roe            = info.get("returnOnEquity")
                debt_to_equity = info.get("debtToEquity")
                market_cap     = info.get("marketCap") or 0
                volume         = info.get("regularMarketVolume") or info.get("volume") or 0
                week52_change  = info.get("52WeekChange")
                sp52_change    = info.get("SandP52WeekChange")
                free_cashflow  = info.get("freeCashflow")
                fcf_yield = (free_cashflow / market_cap) if (free_cashflow and market_cap > 0) else None

            score = 0.0

            # ── Quality ────────────────────────────────────────────────────────
            if revenue_growth and revenue_growth > 0.08:
                score += 2
            if revenue_growth and revenue_growth > 0.20:
                score += 1  # bonus for high growth
            if profit_margin and profit_margin > 0.10:
                score += 2
            if roe and roe > 0.15:
                score += 2

            # ── Valuation: PEG-first, PE as fallback ───────────────────────────
            # PEG < 1 = growing faster than you're paying; PEG > 2.5 = expensive
            if peg is not None and peg > 0:
                if peg < 1.0:
                    score += 3
                elif peg < 1.5:
                    score += 2
                elif peg < 2.5:
                    score += 1
            elif pe:
                if 5 < pe < 20:
                    score += 2
                elif pe < 30:
                    score += 1
            # Forward PE meaningfully below trailing → earnings growth expected
            if forward_pe and pe and forward_pe < pe * 0.9:
                score += 1

            # ── Balance sheet ──────────────────────────────────────────────────
            if debt_to_equity is not None and debt_to_equity < 1.0:
                score += 1

            # ── FCF yield: cash generation relative to market cap ──────────────
            # >5% FCF yield = business generating real cash; not just accounting profit
            if fcf_yield is not None and fcf_yield > 0.05:
                score += 2
            elif fcf_yield is not None and fcf_yield > 0.02:
                score += 1

            # ── Liquidity ──────────────────────────────────────────────────────
            if volume > 200_000:
                score += 1

            # ── Momentum: performance relative to S&P 500 ─────────────────────
            # Relative momentum is a strong empirical factor; an outperformer
            # that's also fundamentally cheap is the ideal combination.
            relative_momentum = None
            if week52_change is not None and sp52_change is not None:
                relative_momentum = week52_change - sp52_change
                if relative_momentum > 0.10:
                    score += 2   # significantly beating the market
                elif relative_momentum > 0:
                    score += 1   # modestly outperforming
                elif relative_momentum < -0.20:
                    score -= 1   # meaningful laggard — raises the bar for a buy
            elif week52_change is not None:
                # No S&P baseline available; use absolute momentum
                if week52_change > 0.15:
                    score += 1

            return {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": industry,
                "market_cap_b": round(market_cap / 1e9, 2) if market_cap else None,
                "price": price,
                "pe_ratio": round(pe, 1) if pe else None,
                "forward_pe": round(forward_pe, 1) if forward_pe else None,
                "peg_ratio": round(peg, 2) if peg is not None else None,
                "revenue_growth_pct": round(revenue_growth * 100, 1) if revenue_growth else None,
                "profit_margin_pct": round(profit_margin * 100, 1) if profit_margin else None,
                "roe_pct": round(roe * 100, 1) if roe else None,
                "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
                "fcf_yield_pct": round(fcf_yield * 100, 1) if fcf_yield is not None else None,
                "week52_return_pct": round(week52_change * 100, 1) if week52_change is not None else None,
                "relative_momentum_pct": round(relative_momentum * 100, 1) if relative_momentum is not None else None,
                "recommendation": recommendation,
                "score": score,
            }
        except Exception:
            return None

    # Process in small batches with a pause between batches to avoid
    # Yahoo Finance rate-limiting. Without this, 640 concurrent requests
    # cause Yahoo to silently drop most responses after ~100 requests,
    # meaning only those that "won the race" get scored — not a true ranking.
    # 5 workers ≈ 3–5 req/s which stays under Yahoo's threshold.
    # At ~1 s/ticker the full universe takes ~2 min; the daily cache
    # means this only runs once per day — all subsequent calls are instant.
    import time as _time

    results: list[dict] = []
    BATCH_SIZE = 60
    BATCH_PAUSE = 1.5  # seconds between batches

    for _batch_start in range(0, len(tickers_to_fetch), BATCH_SIZE):
        batch = tickers_to_fetch[_batch_start: _batch_start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_fetch, t): t for t in batch}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)
        if _batch_start + BATCH_SIZE < len(tickers_to_fetch):
            _time.sleep(BATCH_PAUSE)

    results.sort(key=lambda x: x["score"], reverse=True)

    # Stamp each freshly-fetched result with today's date, then merge into the
    # accumulated cache so the global ranking grows and stabilises over time.
    for r in results:
        r["cached_date"] = today
        cache_by_ticker[r["ticker"]] = r

    all_results = sorted(cache_by_ticker.values(), key=lambda x: x["score"], reverse=True)

    if all_results:
        _save_screener_cache(today, all_results)

    if len(tickers) >= 50:
        return all_results
    # Subset: filter to requested tickers only
    requested = set(upper_tickers)
    return [r for r in all_results if r["ticker"] in requested]


# ══════════════════════════════════════════════════════════════════════════════
# News Alert Severity Classification
# ══════════════════════════════════════════════════════════════════════════════

_THESIS_BREAKING_KW = [
    "fraud", "investigation", "sec charges", "ceo resign", "ceo fired",
    "guidance cut", "going concern", "bankruptcy", "restatement",
    "accounting irregularities",
]
_WATCH_KW = [
    "regulatory", "competition", "market share loss", "lawsuit", "class action",
    "short seller", "downgrade",
]
_NOISE_KW = [
    "upgrade", "price target", "analyst day", "buyback", "dividend",
]


def _classify_alert_severity(headline: str) -> tuple[str, str]:
    """Classify a headline into thesis_breaking/watch/noise/unknown with response protocol."""
    title = (headline or "").lower()
    if any(kw in title for kw in _THESIS_BREAKING_KW):
        return "thesis_breaking", "URGENT: Review position sizing immediately"
    if any(kw in title for kw in _WATCH_KW):
        return "watch", "Monitor: Check next research cycle"
    if any(kw in title for kw in _NOISE_KW):
        return "noise", "FYI: No action needed"
    return "unknown", "Review: Assess relevance to thesis"


def check_news_alerts(holdings: list) -> list:
    """
    Fetch recent news headlines for held positions and flag material events.
    Material events: CEO/CFO departure, restatement, acquisition/merger,
    regulatory action, bankruptcy, guidance cut, major product failure.

    Each alert now includes a `severity` field (thesis_breaking/watch/noise/unknown)
    and a `response_protocol` field with recommended action.
    """
    MATERIAL_KEYWORDS = [
        ("ceo", "cfo", "president", "resign", "depart", "step down", "fired", "replace"),
        ("restat", "accounting", "fraud", "investigation", "sec ", "doj ", "ftc "),
        ("acqui", "merger", "takeover", "buyout", "bid"),
        ("bankrupt", "chapter 11", "insolvency", "default", "restructur"),
        ("guidance", "cut", "lower", "withdraw", "miss", "disappoint"),
        ("recall", "fda", "warning", "ban", "suspend"),
    ]
    SEVERITY_MAP = {0: "LEADERSHIP", 1: "LEGAL", 2: "M&A", 3: "DISTRESS", 4: "GUIDANCE", 5: "REGULATORY"}

    def _check(holding):
        ticker = holding["ticker"]
        try:
            news = yf.Ticker(ticker).news
            if not news:
                return None
            alerts = []
            for item in news[:10]:
                title = (item.get("title") or "").lower()
                for idx, keywords in enumerate(MATERIAL_KEYWORDS):
                    if any(kw in title for kw in keywords):
                        severity, protocol = _classify_alert_severity(item.get("title", ""))
                        alerts.append({
                            "headline": item.get("title"),
                            "category": SEVERITY_MAP[idx],
                            "severity": severity,
                            "response_protocol": protocol,
                            "url": item.get("link"),
                            "published": item.get("providerPublishTime"),
                        })
                        break
            if not alerts:
                return None
            return {
                "ticker": ticker,
                "alert_count": len(alerts),
                "alerts": alerts[:3],
                "recommended_action": "Unscheduled re-research recommended",
            }
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_check, holdings):
            if r:
                results.append(r)
    return sorted(results, key=lambda x: x["alert_count"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# Business Trajectory Tracker
# ══════════════════════════════════════════════════════════════════════════════

def get_business_trajectory(ticker: str) -> dict:
    """
    Analyze up to 8 quarters of key business metrics and classify the trend
    for each using a simple linear regression slope (manual least-squares).

    Metrics analyzed:
      - gross_margin_pct
      - fcf_margin_pct
      - revenue_growth_pct_yoy
      - roic_proxy (operating_income / (total_assets - current_liabilities))

    Returns: ticker, metrics dict with values/slope/trend per metric,
             overall_direction, quarters_analyzed, summary_text.
    """
    ticker = ticker.upper()

    def _linreg_slope(values: list) -> float:
        """Manual least-squares slope (pp/quarter). Newest value is index 0."""
        n = len(values)
        if n < 2:
            return 0.0
        # Reverse so index 0 = oldest quarter
        y = [v for v in reversed(values)]
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        den = sum((x[i] - x_mean) ** 2 for i in range(n))
        return round(num / den, 4) if den != 0 else 0.0

    def _classify_trend(slope: float) -> str:
        if slope > 0.5:
            return "improving"
        elif slope < -0.5:
            return "deteriorating"
        return "stable"

    # Try FMP first for quarterly statements (most reliable source)
    stmts = _fmp_get_quarterly_statements(ticker)

    if stmts:
        def _fmp_series(rows: list, *keys) -> list:
            """Extract up to 8 quarterly values (newest first) from FMP statement rows."""
            vals = []
            for row in rows[:8]:
                for k in keys:
                    v = row.get(k)
                    if v is not None:
                        try:
                            vals.append(float(v))
                            break
                        except (TypeError, ValueError):
                            continue
            return vals

        inc  = stmts["income"]
        bal  = stmts["balance"]
        cf   = stmts["cashflow"]

        rev          = _fmp_series(inc, "revenue")
        gross        = _fmp_series(inc, "grossProfit")
        opcf         = _fmp_series(cf,  "operatingCashFlow")
        capex        = _fmp_series(cf,  "capitalExpenditure")
        op_inc       = _fmp_series(inc, "operatingIncome", "ebitda")
        total_assets = _fmp_series(bal, "totalAssets")
        cur_liab     = _fmp_series(bal, "totalCurrentLiabilities")
    else:
        # Fallback: yfinance quarterly data
        try:
            t = yf.Ticker(ticker)
            qfin = t.quarterly_financials
            qbs  = t.quarterly_balance_sheet
            qcf  = t.quarterly_cashflow
        except Exception as e:
            return {
                "ticker": ticker,
                "error": f"Could not fetch financial data: {e}",
                "metrics": {},
                "overall_direction": "unknown",
                "quarters_analyzed": 0,
                "summary_text": f"Data fetch failed: {e}",
            }

        def _qseries(df, *keys):
            if df is None or df.empty:
                return []
            for key in keys:
                try:
                    row = df.loc[key]
                    vals = [float(v) for v in row.values if v is not None and str(v) != "nan"]
                    if vals:
                        return vals[:8]
                except (KeyError, TypeError):
                    continue
            return []

        rev          = _qseries(qfin, "Total Revenue", "Revenue")
        gross        = _qseries(qfin, "Gross Profit")
        opcf         = _qseries(qcf,  "Operating Cash Flow", "Total Cash From Operating Activities")
        capex        = _qseries(qcf,  "Capital Expenditure", "Capital Expenditures")
        op_inc       = _qseries(qfin, "Operating Income", "EBIT")
        total_assets = _qseries(qbs,  "Total Assets")
        cur_liab     = _qseries(qbs,  "Current Liabilities", "Total Current Liabilities")

    metrics = {}

    # gross_margin_pct
    if rev and gross:
        n = min(len(rev), len(gross), 8)
        vals = [round(gross[i] / rev[i] * 100, 2) for i in range(n) if rev[i] and abs(rev[i]) > 1]
        if vals:
            slope = _linreg_slope(vals)
            metrics["gross_margin_pct"] = {
                "values": vals,
                "slope": slope,
                "trend": _classify_trend(slope),
            }

    # fcf_margin_pct
    if rev and opcf and capex:
        n = min(len(rev), len(opcf), len(capex), 8)
        fcf_vals = []
        for i in range(n):
            if rev[i] and abs(rev[i]) > 1:
                fcf = opcf[i] + capex[i] if capex[i] < 0 else opcf[i] - capex[i]
                fcf_vals.append(round(fcf / rev[i] * 100, 2))
        if fcf_vals:
            slope = _linreg_slope(fcf_vals)
            metrics["fcf_margin_pct"] = {
                "values": fcf_vals,
                "slope": slope,
                "trend": _classify_trend(slope),
            }

    # revenue_growth_pct_yoy (requires 9 quarters to compute 8 yoy values, best effort)
    if len(rev) >= 5:
        # Compare quarter i vs quarter i+4 (same quarter prior year)
        yoy_vals = []
        for i in range(min(len(rev) - 4, 8)):
            prev = rev[i + 4]
            if prev and abs(prev) > 1:
                yoy_vals.append(round((rev[i] - prev) / abs(prev) * 100, 2))
        if yoy_vals:
            slope = _linreg_slope(yoy_vals)
            metrics["revenue_growth_pct_yoy"] = {
                "values": yoy_vals,
                "slope": slope,
                "trend": _classify_trend(slope),
            }

    # roic_proxy = operating_income / (total_assets - current_liabilities)
    if op_inc and total_assets and cur_liab:
        n = min(len(op_inc), len(total_assets), len(cur_liab), 8)
        roic_vals = []
        for i in range(n):
            ic = total_assets[i] - cur_liab[i]
            if ic and abs(ic) > 1:
                roic_vals.append(round(op_inc[i] / ic * 100, 2))
        if roic_vals:
            slope = _linreg_slope(roic_vals)
            metrics["roic_proxy"] = {
                "values": roic_vals,
                "slope": slope,
                "trend": _classify_trend(slope),
            }

    quarters_analyzed = max((len(m["values"]) for m in metrics.values()), default=0)

    # Overall direction
    trends = [m["trend"] for m in metrics.values()]
    improving = trends.count("improving")
    deteriorating = trends.count("deteriorating")
    if improving > deteriorating and improving >= len(trends) / 2:
        overall_direction = "improving"
    elif deteriorating > improving and deteriorating >= len(trends) / 2:
        overall_direction = "deteriorating"
    else:
        overall_direction = "mixed"

    trend_summaries = [
        f"{name}: {data['trend']} (slope={data['slope']:+.2f} pp/qtr)"
        for name, data in metrics.items()
    ]
    summary_text = (
        f"{ticker} business trajectory ({quarters_analyzed}q analyzed): {overall_direction.upper()}. "
        + "; ".join(trend_summaries) if trend_summaries else "Insufficient data."
    )

    return {
        "ticker": ticker,
        "metrics": metrics,
        "overall_direction": overall_direction,
        "quarters_analyzed": quarters_analyzed,
        "summary_text": summary_text,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pre-Earnings Preparation Briefing
# ══════════════════════════════════════════════════════════════════════════════

def get_preearnings_briefing(ticker: str) -> dict:
    """
    Build a structured pre-earnings preparation briefing for a stock.

    Pulls: next earnings date, EPS/revenue estimates, historical beat rate,
    stored thesis from the KB, and prior earnings sentiment tone.

    Returns a briefing dict with thesis_confirms, thesis_denies, watch_metrics.
    """
    ticker = ticker.upper()
    from datetime import datetime

    # Earnings calendar
    cal = get_earnings_calendar(ticker)
    next_earnings_date = cal.get("next_earnings_date")

    days_until_earnings = None
    if next_earnings_date:
        try:
            ned = datetime.strptime(next_earnings_date, "%Y-%m-%d")
            days_until_earnings = (ned - datetime.utcnow()).days
        except Exception:
            pass

    # Historical beat rate from last 4 quarters
    history = cal.get("last_4_quarters", [])
    beats = [q for q in history if q.get("beat") is True]
    historical_beat_rate = round(len(beats) / len(history), 2) if history else None

    # Stored thesis from KB
    thesis_content = None
    try:
        from agent.portfolio import _get_connection
        conn = _get_connection()
        row = conn.execute(
            "SELECT content FROM kb_entries WHERE topic='trade_thesis' AND title LIKE ?",
            (f"%{ticker}%",),
        ).fetchone()
        conn.close()
        if row:
            thesis_content = row["content"]
    except Exception:
        pass

    # Fallback: watchlist entry
    if not thesis_content:
        try:
            from agent.portfolio import _get_connection
            conn = _get_connection()
            row = conn.execute(
                "SELECT notes FROM watchlist WHERE ticker = ?", (ticker,)
            ).fetchone()
            conn.close()
            if row and row["notes"]:
                thesis_content = row["notes"]
        except Exception:
            pass

    # Default thesis confirms/denies based on IV investing principles
    thesis_confirms = [
        "Revenue growth acceleration vs. prior quarter",
        "FCF margin expansion or positive FCF surprise",
        "Management raises full-year guidance",
    ]
    thesis_denies = [
        "Revenue miss >3% vs. consensus estimate",
        "Gross margin compression vs. prior year quarter",
        "Guidance cut or withdrawal of full-year outlook",
    ]

    # If we have stored thesis content, customize slightly
    if thesis_content:
        thesis_confirms = [
            "Revenue growth confirms thesis trajectory",
            "FCF margin expansion supports intrinsic value assumptions",
            "Management commentary confirms competitive moat strength",
        ]
        thesis_denies = [
            "Revenue miss >3% contradicts growth thesis",
            "Margin compression challenges profitability assumptions",
            "Any guidance cut undermines intrinsic value model",
        ]

    watch_metrics = [
        "Revenue vs. consensus (beat/miss %)",
        "EPS vs. consensus (beat/miss %)",
        "Gross margin (QoQ and YoY trend)",
        "FCF generation (or burn rate)",
        "Full-year guidance revision",
        "Management tone on competitive environment",
    ]

    # Prior sentiment
    prior_sentiment = None
    try:
        from agent.portfolio import get_earnings_tone_delta
        prior_sentiment = get_earnings_tone_delta(ticker)
    except Exception:
        pass

    return {
        "ticker": ticker,
        "next_earnings_date": next_earnings_date,
        "days_until_earnings": days_until_earnings,
        "eps_estimate": cal.get("eps_estimate_avg"),
        "revenue_estimate": cal.get("revenue_estimate_avg"),
        "historical_beat_rate": historical_beat_rate,
        "last_4_quarters_history": history,
        "thesis_stored": bool(thesis_content),
        "thesis_confirms": thesis_confirms,
        "thesis_denies": thesis_denies,
        "watch_metrics": watch_metrics,
        "prior_sentiment": prior_sentiment,
        "summary": (
            f"{ticker} reports in {days_until_earnings}d "
            f"(EPS est: {cal.get('eps_estimate_avg')}, "
            f"beat rate: {historical_beat_rate*100:.0f}% last 4Q)."
            if days_until_earnings is not None and historical_beat_rate is not None
            else f"{ticker} earnings briefing prepared."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Alert Severity Triage
# ══════════════════════════════════════════════════════════════════════════════

def get_triaged_alerts(tickers: list = None) -> dict:
    """
    Get news alerts triaged by severity: thesis_breaking, watch, or noise.

    If tickers is provided, checks those tickers. Otherwise checks portfolio holdings.
    Returns grouped alerts by severity tier plus top_priority item.
    """
    # Determine holdings to check
    if tickers:
        holdings = [{"ticker": t.upper()} for t in tickers]
    else:
        try:
            from agent.portfolio import get_holdings
            holdings = get_holdings()
        except Exception:
            holdings = []

    if not holdings:
        return {
            "thesis_breaking": [],
            "watch": [],
            "noise": [],
            "summary_counts": {"thesis_breaking": 0, "watch": 0, "noise": 0},
            "top_priority": None,
            "note": "No holdings to check.",
        }

    raw_alerts = check_news_alerts(holdings)

    thesis_breaking = []
    watch = []
    noise = []

    for ticker_alert in raw_alerts:
        for alert in ticker_alert.get("alerts", []):
            severity = alert.get("severity", "unknown")
            entry = {
                "ticker": ticker_alert["ticker"],
                "headline": alert.get("headline"),
                "category": alert.get("category"),
                "severity": severity,
                "response_protocol": alert.get("response_protocol"),
                "url": alert.get("url"),
                "published": alert.get("published"),
            }
            if severity == "thesis_breaking":
                thesis_breaking.append(entry)
            elif severity == "watch":
                watch.append(entry)
            elif severity == "noise":
                noise.append(entry)

    top_priority = thesis_breaking[0] if thesis_breaking else (watch[0] if watch else None)

    return {
        "thesis_breaking": thesis_breaking,
        "watch": watch,
        "noise": noise,
        "summary_counts": {
            "thesis_breaking": len(thesis_breaking),
            "watch": len(watch),
            "noise": len(noise),
        },
        "top_priority": top_priority,
    }



def calculate_intrinsic_value(ticker: str) -> dict:
    """
    Standardized 3-stage DCF model for a given ticker.

    Stage 1 (years 1-5):  conservative_growth = min(analyst_5yr_eps_growth, fcf_cagr_proxy, 0.20) * 0.80
    Stage 2 (years 6-10): linearly fade from conservative_growth -> 0.03
    Stage 3 (terminal):   2.5% perpetuity, capitalised at (discount_rate - 0.025)
    Discount rate:        10.0%

    Returns bear/base/bull scenarios plus margin-of-safety and a simple verdict.
    """
    try:
        info = yf.Ticker(ticker).info
        if not info:
            return {"ticker": ticker, "error": "No data returned from yfinance."}

        # ── Raw data ──────────────────────────────────────────────────────────
        fcf    = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        price  = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        earnings_growth = info.get("earningsGrowth")   # analyst forward EPS growth (decimal)
        revenue_growth  = info.get("revenueGrowth")    # fallback growth signal
        roe             = info.get("returnOnEquity")    # informational

        # Validate minimum viable data
        if price is None and shares is None and earnings_growth is None:
            return {"ticker": ticker, "error": "Insufficient data to run DCF model."}

        # ── FCF / data-quality handling ───────────────────────────────────────
        data_quality = "full"
        currency_warning = None

        market_cap = info.get("marketCap")

        if fcf is None or fcf <= 0:
            # Use earnings_growth * price * shares as a rough FCF proxy
            if price and shares and earnings_growth is not None:
                fcf = earnings_growth * price * shares
                data_quality = "estimated"
            elif price and shares:
                proxy_growth = revenue_growth if revenue_growth else 0.05
                fcf = proxy_growth * price * shares
                data_quality = "estimated"
            else:
                return {"ticker": ticker, "error": "Insufficient data to run DCF model."}
        else:
            if not (price and shares):
                data_quality = "partial"
            # Sanity check: FCF yield > 25% almost always means a currency mismatch.
            # ADRs (TSM, ASML, etc.) sometimes have FCF reported in home currency
            # (TWD, EUR) while market_cap is in USD — making FCF look 30x too large.
            if market_cap and market_cap > 0:
                implied_fcf_yield = fcf / market_cap
                if implied_fcf_yield > 0.25:
                    currency_warning = (
                        f"FCF yield implied by raw data is {implied_fcf_yield*100:.1f}% — "
                        "almost certainly a currency mismatch (ADR FCF in home currency vs "
                        "USD market cap). Switching to earnings-based FCF proxy. "
                        "IV estimate should be treated as approximate only."
                    )
                    data_quality = "suspect_currency"
                    # Override with earnings-based proxy in USD
                    if price and shares and earnings_growth is not None and earnings_growth > 0:
                        fcf = earnings_growth * price * shares
                    elif price and shares:
                        proxy_growth = max(revenue_growth or 0.05, 0.03)
                        fcf = proxy_growth * price * shares

        # Guard against still-missing price or shares (edge case)
        if price is None or shares is None or shares == 0:
            data_quality = "partial"
            # Provide placeholder values so model can still run
            if price is None:
                price = 0.0
            if shares is None or shares == 0:
                shares = 1  # avoid ZeroDivisionError; IV will be meaningless

        # ── Mid-cycle normalization ───────────────────────────────────────────
        norm = _get_normalized_metrics(ticker)
        normalized_fcf    = norm.get("normalized_fcf")
        normalized_margin = norm.get("normalized_margin")
        years_averaged    = norm.get("years_averaged", 0)
        normalization_note = None
        pit_fcf = fcf  # preserve point-in-time value for reporting

        if normalized_fcf is not None and pit_fcf and pit_fcf != 0:
            ratio = normalized_fcf / pit_fcf
            if 1 / 3 <= ratio <= 3:
                # Normalized FCF is within 3x of point-in-time; use it as DCF base
                fcf = normalized_fcf
                normalization_note = (
                    f"FCF averaged over {min(years_averaged, 3)} year(s) "
                    f"(normalized: ${normalized_fcf/1e6:.1f}m vs PIT: ${pit_fcf/1e6:.1f}m)"
                )

        # ── Growth-rate inputs ────────────────────────────────────────────────
        # Analyst 5-yr EPS growth (decimal); clamp to reasonable range
        analyst_growth = earnings_growth if earnings_growth is not None else 0.05
        analyst_growth = max(0.0, analyst_growth)

        # FCF CAGR proxy: use earningsGrowth as single-year approximation
        # (yfinance does not expose multi-year FCF history in .info)
        fcf_cagr = analyst_growth  # best available proxy

        DISCOUNT_RATE   = 0.10
        TERMINAL_GROWTH = 0.025
        MAX_STAGE1      = 0.20

        # Conservative Stage-1 growth with 20% haircut
        conservative_growth = min(analyst_growth, fcf_cagr, MAX_STAGE1) * 0.80

        # ── DCF helper ────────────────────────────────────────────────────────
        def _dcf(stage1_growth: float) -> float:
            """Return total intrinsic value (not per share) for given stage-1 growth."""
            discount = DISCOUNT_RATE
            terminal = TERMINAL_GROWTH

            cumulative_pv = 0.0
            fcf_t = float(fcf)

            # Stage 1: years 1-5
            for t in range(1, 6):
                fcf_t *= (1 + stage1_growth)
                pv = fcf_t / ((1 + discount) ** t)
                cumulative_pv += pv

            fcf_stage1_end = fcf_t  # FCF at end of year 5

            # Stage 2: years 6-10, linearly fading stage1_growth -> terminal
            for t in range(6, 11):
                # Linear interpolation: at t=6 weight=4/5 on stage1, at t=10 weight=0
                step = t - 5          # 1 … 5
                fade_growth = stage1_growth + (terminal - stage1_growth) * (step / 5.0)
                fcf_t *= (1 + fade_growth)
                pv = fcf_t / ((1 + discount) ** t)
                cumulative_pv += pv

            fcf_year10 = fcf_t

            # Stage 3: terminal value at end of year 10
            terminal_value = fcf_year10 * (1 + terminal) / (discount - terminal)
            pv_terminal = terminal_value / ((1 + discount) ** 10)
            cumulative_pv += pv_terminal

            return cumulative_pv

        # ── Scenarios ─────────────────────────────────────────────────────────
        bear_growth = conservative_growth * 0.7
        base_growth = conservative_growth
        bull_growth = min(conservative_growth * 1.3, 0.25)

        bear_total = _dcf(bear_growth)
        base_total = _dcf(base_growth)
        bull_total = _dcf(bull_growth)

        bear_iv = bear_total / shares
        base_iv = base_total / shares
        bull_iv = bull_total / shares

        # ── Margin of safety ──────────────────────────────────────────────────
        def _mos(iv: float) -> Optional[float]:
            if iv <= 0:
                return None
            return round((iv - price) / iv * 100, 1)

        mos_bear = _mos(bear_iv)
        mos_base = _mos(base_iv)
        mos_bull = _mos(bull_iv)

        # ── Verdict ───────────────────────────────────────────────────────────
        if mos_base is not None and mos_base >= 20:
            verdict = "BUY"
        elif mos_base is not None and mos_base >= 10:
            verdict = "WATCHLIST"
        else:
            verdict = "OVERVALUED"

        return {
            "ticker":                   ticker.upper(),
            "current_price":            price,
            "fcf_trailing":             pit_fcf,
            "stage1_growth_rate_pct":   round(conservative_growth * 100, 1),
            "discount_rate_pct":        10.0,
            "terminal_growth_rate_pct": 2.5,
            "intrinsic_value": {
                "bear": round(bear_iv, 2),
                "base": round(base_iv, 2),
                "bull": round(bull_iv, 2),
            },
            "margin_of_safety_pct": {
                "bear": mos_bear,
                "base": mos_base,
                "bull": mos_bull,
            },
            "verdict":      verdict,
            "note":         (
                "Standardized 3-stage DCF. Use base IV as primary reference. "
                "Bear/bull show sensitivity range."
            ),
            "data_quality":       data_quality,
            "currency_warning":   currency_warning,
            "normalized_fcf":     normalized_fcf,
            "normalized_margin":  normalized_margin,
            "years_averaged":     years_averaged,
            "normalization_note": normalization_note,
        }

    except Exception as e:
        return {"ticker": ticker, "error": f"DCF calculation failed: {e}"}


def check_earnings_surprises(holdings: list) -> list:
    """
    Check held positions for recent earnings surprises.
    Flags stocks where actual EPS significantly beat or missed estimates.
    Returns list of alerts with ticker, surprise_pct, direction, and recommended action.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _check(holding):
        ticker = holding["ticker"]
        try:
            tk = yf.Ticker(ticker)
            # earnings_history gives actual vs estimate
            hist = tk.earnings_history
            if hist is None or hist.empty:
                return None
            # Most recent quarter
            latest = hist.iloc[0]
            eps_actual = latest.get("epsActual")
            eps_estimate = latest.get("epsEstimate")
            if eps_actual is None or eps_estimate is None or eps_estimate == 0:
                return None
            surprise_pct = (eps_actual - eps_estimate) / abs(eps_estimate) * 100
            if abs(surprise_pct) < 5:  # ignore small surprises
                return None
            direction = "BEAT" if surprise_pct > 0 else "MISS"
            severity = "SIGNIFICANT" if abs(surprise_pct) > 15 else "MODERATE"
            action = "re-research — strong confirmation" if direction == "BEAT" and abs(surprise_pct) > 15 \
                else "re-research — thesis may be breaking" if direction == "MISS" else "monitor"
            return {
                "ticker": ticker,
                "eps_actual": eps_actual,
                "eps_estimate": eps_estimate,
                "surprise_pct": round(surprise_pct, 1),
                "direction": direction,
                "severity": severity,
                "recommended_action": action,
            }
        except Exception:
            return None

    alerts = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for r in ex.map(_check, holdings):
            if r:
                alerts.append(r)
    return sorted(alerts, key=lambda x: abs(x["surprise_pct"]), reverse=True)


def check_fundamental_deterioration(holdings: list[dict]) -> list[dict]:
    """
    For each held position, check for fundamental red flags that may warrant exit:
    - Revenue growth gone negative (was positive at buy)
    - Gross margin compression > 500bps YoY
    - FCF turned negative
    - Debt/equity spiked > 2x from last known level
    - ROE dropped below 10% (was above at quality screen)

    holdings: list of dicts with at least 'ticker' key.
    Returns list of alert dicts with ticker, flags, severity (WATCH/REVIEW/EXIT).
    """

    def _check_one(holding):
        ticker = holding["ticker"]
        try:
            info = yf.Ticker(ticker).info
            flags = []

            rev_growth = info.get("revenueGrowth")
            if rev_growth is not None and rev_growth < -0.05:
                flags.append(f"Revenue declining {rev_growth*100:.1f}% YoY")

            gross_margins = info.get("grossMargins")
            # Flag if gross margin < 20% (thin margins = vulnerable)
            if gross_margins is not None and gross_margins < 0.20:
                flags.append(f"Gross margin thin at {gross_margins*100:.1f}%")

            fcf = info.get("freeCashflow")
            if fcf is not None and fcf < 0:
                flags.append(f"FCF negative (${fcf/1e6:.0f}M)")

            dte = info.get("debtToEquity")
            if dte is not None and dte > 200:  # >2x D/E
                flags.append(f"High leverage: D/E {dte/100:.1f}x")

            roe = info.get("returnOnEquity")
            if roe is not None and roe < 0.08:
                flags.append(f"ROE deteriorated to {roe*100:.1f}%")

            # Earnings trend: forward PE >> trailing PE = earnings expected to fall
            trailing_pe = info.get("trailingPE")
            forward_pe = info.get("forwardPE")
            if trailing_pe and forward_pe and forward_pe > trailing_pe * 1.25:
                flags.append(f"Earnings declining: fwd PE {forward_pe:.1f} >> trail PE {trailing_pe:.1f}")

            if not flags:
                return None

            severity = "EXIT" if len(flags) >= 3 else "REVIEW" if len(flags) >= 2 else "WATCH"
            return {
                "ticker": ticker,
                "flags": flags,
                "severity": severity,
                "flag_count": len(flags),
            }
        except Exception:
            return None

    alerts = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(_check_one, holdings))
    for r in results:
        if r:
            alerts.append(r)
    alerts.sort(key=lambda x: x["flag_count"], reverse=True)
    return alerts


def check_watchlist_triggers(watchlist: list[dict]) -> dict:
    """
    Fetch live prices for all watchlist items and classify each against its target entry price.

    Status values:
      TRIGGERED   — current price ≤ target (buy criteria met on price)
      APPROACHING — current price within 10% above target (worth watching closely)
      WAITING     — current price > 10% above target
      NO_TARGET   — no target entry price set; reports current price only

    Returns items sorted: TRIGGERED first, then APPROACHING, then WAITING.
    """
    if not watchlist:
        return {"triggered": [], "approaching": [], "waiting": [], "no_target": [],
                "summary": "Watchlist is empty."}

    def _fetch_price(item: dict) -> dict:
        ticker = item.get("ticker", "")
        target = item.get("target_entry_price")
        try:
            info  = yf.Ticker(ticker).info
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
        except Exception:
            price = None

        result = {
            "ticker":             ticker,
            "company_name":       item.get("company_name", ""),
            "current_price":      round(price, 2) if price else None,
            "target_entry_price": target,
            "reason":             item.get("reason", ""),
            "added_at":           item.get("added_at", "")[:10],
        }

        if price is None:
            result["status"] = "PRICE_UNAVAILABLE"
            result["pct_above_target"] = None
        elif target is None:
            result["status"] = "NO_TARGET"
            result["pct_above_target"] = None
        else:
            pct_above = (price - target) / target * 100
            result["pct_above_target"] = round(pct_above, 1)
            if price <= target:
                result["status"] = "TRIGGERED"
                result["action"] = "PRICE AT OR BELOW TARGET — run deep research immediately"
            elif pct_above <= 10:
                result["status"] = "APPROACHING"
                result["action"] = f"Only {pct_above:.1f}% above target — monitor closely"
            else:
                result["status"] = "WAITING"
                result["action"] = f"{pct_above:.1f}% above target — continue waiting"

        return result

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_price, item): item for item in watchlist}
        for future in as_completed(futures):
            results.append(future.result())

    triggered   = [r for r in results if r["status"] == "TRIGGERED"]
    approaching = [r for r in results if r["status"] == "APPROACHING"]
    waiting     = [r for r in results if r["status"] == "WAITING"]
    no_target   = [r for r in results if r["status"] in ("NO_TARGET", "PRICE_UNAVAILABLE")]

    triggered.sort(key=lambda x: (x["pct_above_target"] or 0))
    approaching.sort(key=lambda x: (x["pct_above_target"] or 999))

    summary_parts = []
    if triggered:
        summary_parts.append(f"🔴 {len(triggered)} TRIGGERED (price at/below target)")
    if approaching:
        summary_parts.append(f"🟡 {len(approaching)} APPROACHING (within 10% of target)")
    if waiting:
        summary_parts.append(f"⚪ {len(waiting)} waiting")

    return {
        "triggered":   triggered,
        "approaching": approaching,
        "waiting":     waiting,
        "no_target":   no_target,
        "summary":     " | ".join(summary_parts) if summary_parts else "All items waiting.",
    }


def get_watchlist_earnings(watchlist: list) -> dict:
    """
    Fetch upcoming earnings dates for all watchlist items.
    Returns items bucketed by urgency:
      IMMINENT  — earnings within 7 days
      UPCOMING  — earnings within 30 days
      DISTANT   — earnings more than 30 days away
      UNKNOWN   — no earnings date found
    """
    from datetime import date
    import pandas as pd

    today = date.today()

    def _fetch(item: dict) -> dict:
        ticker = item.get("ticker", "")
        try:
            cal = yf.Ticker(ticker).calendar
            # yfinance returns a dict with 'Earnings Date' key (list of timestamps)
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed and len(ed) > 0:
                    earnings_date = pd.Timestamp(ed[0]).date()
            elif hasattr(cal, 'columns') and 'Earnings Date' in cal.columns:
                val = cal['Earnings Date'].iloc[0] if len(cal) > 0 else None
                if val is not None:
                    earnings_date = pd.Timestamp(val).date()
        except Exception:
            earnings_date = None

        days_away = (earnings_date - today).days if earnings_date else None

        result = {
            "ticker": ticker,
            "company_name": item.get("company_name", ""),
            "target_entry_price": item.get("target_entry_price"),
            "earnings_date": str(earnings_date) if earnings_date else None,
            "days_until_earnings": days_away,
        }

        if days_away is None:
            result["urgency"] = "UNKNOWN"
        elif days_away <= 7:
            result["urgency"] = "IMMINENT"
            result["action"] = f"Earnings in {days_away} days — research NOW before results"
        elif days_away <= 30:
            result["urgency"] = "UPCOMING"
            result["action"] = f"Earnings in {days_away} days — prepare thesis"
        else:
            result["urgency"] = "DISTANT"
            result["action"] = f"Earnings in {days_away} days — no immediate action needed"

        return result

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch, item): item for item in watchlist}
        for future in as_completed(futures):
            results.append(future.result())

    imminent = sorted([r for r in results if r["urgency"] == "IMMINENT"],
                      key=lambda x: x["days_until_earnings"])
    upcoming = sorted([r for r in results if r["urgency"] == "UPCOMING"],
                      key=lambda x: x["days_until_earnings"])
    distant  = [r for r in results if r["urgency"] == "DISTANT"]
    unknown  = [r for r in results if r["urgency"] == "UNKNOWN"]

    summary_parts = []
    if imminent: summary_parts.append(f"⚡ {len(imminent)} IMMINENT (≤7 days)")
    if upcoming: summary_parts.append(f"📅 {len(upcoming)} upcoming (≤30 days)")

    return {
        "imminent":  imminent,
        "upcoming":  upcoming,
        "distant":   distant,
        "unknown":   unknown,
        "summary":   " | ".join(summary_parts) if summary_parts else "No imminent earnings.",
    }


def check_dividend_payments(holdings: list) -> list:
    """
    Check for recent dividend payments for held positions.
    Returns list of dicts with ticker, dividend_rate, ex_date, estimated_annual_yield.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _check(holding):
        ticker = holding["ticker"]
        shares = holding.get("shares", 0)
        try:
            info = yf.Ticker(ticker).info
            div_rate = info.get("dividendRate") or 0
            div_yield = info.get("dividendYield") or 0
            ex_date = info.get("exDividendDate")
            if div_rate and div_rate > 0:
                return {
                    "ticker": ticker,
                    "shares": shares,
                    "annual_dividend_rate": div_rate,
                    "annual_dividend_yield_pct": round(div_yield * 100, 2) if div_yield else 0,
                    "estimated_annual_income": round(div_rate * shares, 2),
                    "ex_dividend_date": str(ex_date) if ex_date else None,
                }
            return None
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for r in ex.map(_check, holdings):
            if r:
                results.append(r)
    return sorted(results, key=lambda x: x["estimated_annual_income"], reverse=True)


def _get_normalized_metrics(ticker: str) -> dict:
    """
    Fetch 3-year average FCF, profit margin, and ROE to reduce point-in-time distortions.
    Returns dict with normalized_fcf, normalized_margin, normalized_roe, years_averaged.
    Falls back to None values if insufficient history.
    """
    try:
        tk = yf.Ticker(ticker)
        cf = tk.cash_flow  # columns = annual periods, rows = line items
        inc = tk.income_stmt

        result = {"normalized_fcf": None, "normalized_margin": None, "normalized_roe": None, "years_averaged": 0}

        # FCF = Operating Cash Flow - CapEx
        fcf_values = []
        if cf is not None and not cf.empty:
            # Look for operating cash flow row
            ocf_rows = [r for r in cf.index if "Operating" in str(r) and "Cash" in str(r)]
            capex_rows = [r for r in cf.index if "Capital" in str(r) and ("Expenditure" in str(r) or "Expenditures" in str(r))]
            if ocf_rows:
                ocf_series = cf.loc[ocf_rows[0]].dropna()
                capex_series = cf.loc[capex_rows[0]].dropna() if capex_rows else None
                for col in list(ocf_series.index)[:4]:  # up to 4 years
                    ocf = ocf_series.get(col)
                    capex = capex_series.get(col, 0) if capex_series is not None else 0
                    if ocf is not None and not (isinstance(ocf, float) and (ocf != ocf)):
                        fcf = float(ocf) - abs(float(capex or 0))
                        if abs(fcf) > 0:
                            fcf_values.append(fcf)

        # Profit margin = Net Income / Revenue
        margin_values = []
        if inc is not None and not inc.empty:
            ni_rows = [r for r in inc.index if "Net Income" in str(r) and "Common" not in str(r)]
            rev_rows = [r for r in inc.index if "Total Revenue" in str(r) or r == "Revenue"]
            if ni_rows and rev_rows:
                ni_series = inc.loc[ni_rows[0]].dropna()
                rev_series = inc.loc[rev_rows[0]].dropna()
                for col in list(ni_series.index)[:4]:
                    ni = ni_series.get(col)
                    rev = rev_series.get(col)
                    if ni and rev and float(rev) > 0:
                        margin_values.append(float(ni) / float(rev))

        years = min(len(fcf_values), 4)
        if fcf_values:
            result["normalized_fcf"] = sum(fcf_values[:3]) / min(len(fcf_values), 3)
        if margin_values:
            result["normalized_margin"] = sum(margin_values[:3]) / min(len(margin_values), 3)
        result["years_averaged"] = years
        return result
    except Exception:
        return {"normalized_fcf": None, "normalized_margin": None, "normalized_roe": None, "years_averaged": 0}


def detect_financial_anomalies(ticker: str, screener_data: dict = None) -> dict:
    """
    Detect statistical anomalies in a company's financial ratios.

    Computes z-scores for key metrics vs:
    (a) The company's own 5-year historical average (temporal z-score)
    (b) Sector peers from the universe_scores table (cross-sectional z-score)

    Returns flagged anomalies with direction and interpretation.
    Positive outliers (z > +2): potential peak earnings trap
    Negative outliers (z < -2): potential temporary dislocation / opportunity
    """
    import numpy as np
    import yfinance as yf
    from agent.portfolio import _get_connection

    ticker = ticker.upper()
    anomalies = []
    sector_anomalies = []

    # ── Fetch yfinance annual financials ──────────────────────────────────────
    try:
        t = yf.Ticker(ticker)
        fin = t.financials          # income statement, columns = years
        bs = t.balance_sheet
        cf = t.cashflow
    except Exception as e:
        return {
            "ticker": ticker,
            "anomalies": [],
            "sector_anomalies": [],
            "overall_flag": "clean",
            "summary": f"Could not fetch financial data: {e}",
        }

    def _series(df, *keys):
        """Try multiple row labels; return the first found as a list (newest first)."""
        if df is None or df.empty:
            return []
        for key in keys:
            try:
                row = df.loc[key]
                vals = [float(v) for v in row.values if v is not None and str(v) != "nan"]
                if vals:
                    return vals
            except (KeyError, TypeError):
                continue
        return []

    # Revenue
    rev = _series(fin, "Total Revenue", "Revenue")
    # Gross profit
    gross = _series(fin, "Gross Profit")
    # Net income
    net = _series(fin, "Net Income", "Net Income Common Stockholders")
    # Free cash flow (operating CF - capex)
    opcf = _series(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = _series(cf, "Capital Expenditure", "Capital Expenditures")

    def _pct_list(numerator, denominator):
        """Compute numerator/denominator pct for each year (aligned)."""
        results = []
        for n, d in zip(numerator, denominator):
            if d and abs(d) > 1:
                results.append(n / d * 100)
        return results

    def _growth_list(vals):
        """Compute YoY growth % from a list (newest first)."""
        results = []
        for i in range(len(vals) - 1):
            if vals[i + 1] and abs(vals[i + 1]) > 1:
                results.append((vals[i] - vals[i + 1]) / abs(vals[i + 1]) * 100)
        return results

    # Compute FCF
    fcf = []
    for o, c in zip(opcf, capex):
        # capex is typically negative in yfinance
        fcf.append(o + c if c < 0 else o - c)

    gross_margin = _pct_list(gross, rev)
    fcf_margin = _pct_list(fcf, rev)
    fcf_conversion = _pct_list(fcf, net) if net else []
    rev_growth = _growth_list(rev)

    metric_series = {
        "gross_margin_pct": gross_margin,
        "fcf_margin_pct": fcf_margin,
        "fcf_conversion": fcf_conversion,
        "revenue_growth_pct": rev_growth,
    }

    interpretations = {
        "gross_margin_pct": {
            "above": "Gross margin at 5-year high — potential peak earnings trap, normalise before DCF",
            "below": "Gross margin at 5-year low — temporary compression or structural deterioration?",
        },
        "fcf_margin_pct": {
            "above": "FCF margin well above historical norm — check for working capital release or one-time items",
            "below": "FCF margin below historical norm — potential structural FCF deterioration",
        },
        "fcf_conversion": {
            "above": "FCF conversion well above historical norm — check for working capital harvest or deferred capex",
            "below": "FCF conversion below historical norm — earnings quality concern; check accruals",
        },
        "revenue_growth_pct": {
            "above": "Revenue growth at multi-year high — verify sustainability; mean-reversion risk in DCF",
            "below": "Revenue growth at multi-year low — cyclical trough or fundamental slowdown?",
        },
    }

    for metric, series in metric_series.items():
        if len(series) < 3:
            continue
        current = series[0]
        historical = series[1:]  # exclude current
        mean = float(np.mean(historical))
        std = float(np.std(historical))
        if std < 0.01:
            continue
        z = (current - mean) / std

        if abs(z) < 1.5:
            continue

        severity = "significant" if abs(z) >= 2.0 else "notable"
        direction = "above_norm" if z > 0 else "below_norm"
        interp_key = "above" if z > 0 else "below"
        interpretation = interpretations.get(metric, {}).get(interp_key, "")

        anomalies.append({
            "metric": metric,
            "current_value": round(current, 2),
            "5yr_mean": round(mean, 2),
            "temporal_z": round(z, 2),
            "severity": severity,
            "direction": direction,
            "interpretation": interpretation,
        })

    # ── Cross-sectional z-score vs sector peers ───────────────────────────────
    try:
        conn = _get_connection()
        sector_row = conn.execute(
            "SELECT sector FROM universe_scores WHERE ticker = ?", (ticker,)
        ).fetchone()
        sector = sector_row["sector"] if sector_row else None

        if sector:
            peer_rows = conn.execute("""
                SELECT ticker, profit_margin, revenue_growth, roe
                FROM universe_scores
                WHERE sector = ? AND ticker != ?
                LIMIT 50
            """, (sector, ticker)).fetchall()
            peers = [dict(r) for r in peer_rows]
        else:
            peers = []
        conn.close()
    except Exception:
        peers = []
        sector = None

    if len(peers) >= 5 and screener_data:
        peer_metrics = {
            "profit_margin": [p["profit_margin"] for p in peers if p.get("profit_margin") is not None],
            "revenue_growth": [p["revenue_growth"] for p in peers if p.get("revenue_growth") is not None],
            "roe": [p["roe"] for p in peers if p.get("roe") is not None],
        }
        stock_metrics = {
            "profit_margin": screener_data.get("profit_margin_pct"),
            "revenue_growth": screener_data.get("revenue_growth_pct"),
            "roe": screener_data.get("roe_pct"),
        }
        for metric, peer_vals in peer_metrics.items():
            if len(peer_vals) < 5:
                continue
            stock_val = stock_metrics.get(metric)
            if stock_val is None:
                continue
            mean = float(np.mean(peer_vals))
            std = float(np.std(peer_vals))
            if std < 0.01:
                continue
            z = (stock_val - mean) / std
            if abs(z) < 1.5:
                continue
            severity = "significant" if abs(z) >= 2.0 else "notable"
            direction = "above_norm" if z > 0 else "below_norm"
            sector_anomalies.append({
                "metric": metric,
                "stock_value": round(stock_val, 2),
                "sector_mean": round(mean, 2),
                "cross_sectional_z": round(z, 2),
                "severity": severity,
                "direction": direction,
                "n_peers": len(peer_vals),
                "interpretation": (
                    f"{metric} is {'well above' if z > 0 else 'well below'} {sector} sector average "
                    f"(z={z:.1f}, n={len(peer_vals)} peers)"
                ),
            })

    # Overall flag
    significant_count = sum(1 for a in anomalies if a["severity"] == "significant")
    significant_count += sum(1 for a in sector_anomalies if a["severity"] == "significant")

    if significant_count >= 2:
        overall_flag = "flagged"
    elif anomalies or sector_anomalies:
        overall_flag = "watch"
    else:
        overall_flag = "clean"

    notable_items = [a["interpretation"] for a in anomalies if a["interpretation"]]
    summary = (
        f"Found {len(anomalies)} temporal and {len(sector_anomalies)} cross-sectional anomalies. "
        + (f"Key flags: {'; '.join(notable_items[:3])}" if notable_items else "No material anomalies detected.")
    )

    return {
        "ticker": ticker,
        "anomalies": anomalies,
        "sector_anomalies": sector_anomalies,
        "overall_flag": overall_flag,
        "summary": summary,
    }


def get_fresh_valuation(candidates: list[dict]) -> list[dict]:
    """
    Enrich a list of quality-scored candidates with fresh price-driven metrics.
    Fetches: FCF yield, PEG ratio, forward PE, and 52-week momentum.
    Returns the same list with valuation_score added and candidates re-ranked
    by combined (quality + valuation) score.

    Valuation factors (max score = 9):
      +3  peg < 1.0
      +2  peg < 1.5  (or pe 5-20)
      +1  peg < 2.5  (or pe < 30)
      +1  forward_pe < trailing_pe * 0.9
      +2  fcf_yield > 5%
      +1  fcf_yield > 2%
      +2  relative_momentum > 10%
      +1  relative_momentum > 0%
      -1  relative_momentum < -20%
    """
    ticker_to_candidate = {c["ticker"]: c for c in candidates}

    def _val(ticker: str) -> Optional[dict]:
        try:
            info = yf.Ticker(ticker).info
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            if not price:
                return None

            pe          = info.get("trailingPE")
            forward_pe  = info.get("forwardPE")
            peg         = info.get("pegRatio")
            market_cap  = info.get("marketCap") or 0
            fcf         = info.get("freeCashflow")
            w52         = info.get("52WeekChange")
            sp52        = info.get("SandP52WeekChange")

            fcf_yield = (fcf / market_cap) if (fcf and market_cap > 0) else None
            rel_mom   = (w52 - sp52) if (w52 is not None and sp52 is not None) else None

            score = 0.0
            if peg is not None and peg > 0:
                if peg < 1.0:   score += 3
                elif peg < 1.5: score += 2
                elif peg < 2.5: score += 1
            elif pe:
                if 5 < pe < 20: score += 2
                elif pe < 30:   score += 1
            if forward_pe and pe and forward_pe < pe * 0.9:
                score += 1
            if fcf_yield is not None:
                if fcf_yield > 0.05: score += 2
                elif fcf_yield > 0.02: score += 1
            if rel_mom is not None:
                if rel_mom > 0.10:   score += 2
                elif rel_mom > 0:    score += 1
                elif rel_mom < -0.20: score -= 1
            elif w52 is not None and w52 > 0.15:
                score += 1

            return {
                "ticker":           ticker,
                "price":            price,
                "peg_ratio":        round(peg, 2)              if peg is not None else None,
                "pe_ratio":         round(pe, 1)               if pe else None,
                "forward_pe":       round(forward_pe, 1)       if forward_pe else None,
                "fcf_yield_pct":    round(fcf_yield * 100, 1)  if fcf_yield is not None else None,
                "week52_return_pct":  round(w52 * 100, 1)      if w52 is not None else None,
                "relative_momentum_pct": round(rel_mom * 100, 1) if rel_mom is not None else None,
                "valuation_score":  score,
            }
        except Exception:
            return None

    val_results = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_val, c["ticker"]): c["ticker"] for c in candidates}
        for future in as_completed(futures):
            res = future.result()
            if res:
                val_results[res["ticker"]] = res

    enriched = []
    for c in candidates:
        ticker = c["ticker"]
        val = val_results.get(ticker, {})
        combined = dict(c)
        combined.update(val)
        combined["valuation_score"] = val.get("valuation_score", 0.0)
        combined["combined_score"]  = c["quality_score"] + combined["valuation_score"]
        enriched.append(combined)

    enriched.sort(key=lambda x: x["combined_score"], reverse=True)
    return enriched


def quick_moat_check(ticker: str, screener_data: dict = None) -> dict:
    """
    Quick moat signal check before committing to full research.
    Uses already-fetched screener_data if available, otherwise fetches minimal info.
    Returns: {has_moat_signal: bool, signals: list[str], confidence: str}
    """
    import yfinance as yf

    signals = []

    # Use screener_data if available (free — already fetched)
    data = screener_data or {}

    profit_margin = data.get("profit_margin_pct")
    roe = data.get("roe_pct")
    revenue_growth = data.get("revenue_growth_pct")
    gross_margin = None

    # Fetch minimal extra data if needed
    if not data:
        try:
            info = yf.Ticker(ticker).info
            profit_margin = (info.get("profitMargins") or 0) * 100
            roe = (info.get("returnOnEquity") or 0) * 100
            revenue_growth = (info.get("revenueGrowth") or 0) * 100
            gross_margin = (info.get("grossMargins") or 0) * 100
        except Exception:
            pass

    # Moat signals
    if profit_margin and profit_margin > 15:
        signals.append(f"High profit margin {profit_margin:.1f}% → pricing power signal")
    if roe and roe > 20:
        signals.append(f"High ROE {roe:.1f}% → capital efficiency / competitive advantage")
    if revenue_growth and revenue_growth > 10:
        signals.append(f"Strong revenue growth {revenue_growth:.1f}% → market position")
    if gross_margin and gross_margin > 60:
        signals.append(f"High gross margin {gross_margin:.1f}% → software/brand moat signal")

    # Score from screener
    score = data.get("score", 0)
    if score and score >= 6:
        signals.append(f"Quality screen score {score}/10")

    has_moat = len(signals) >= 2
    confidence = "HIGH" if len(signals) >= 3 else "MEDIUM" if len(signals) >= 2 else "LOW"

    return {
        "has_moat_signal": has_moat,
        "signals": signals,
        "confidence": confidence,
        "signal_count": len(signals),
    }


def score_quality_universe(us_tickers: list[str], intl_tickers: list[str]) -> list[dict]:
    """
    Score all tickers on stable quality factors only (no price-dependent metrics).
    Designed to be run infrequently (quarterly) and cached.

    Quality factors (max score = 8):
      +2  revenue_growth > 8%   (growing business)
      +1  revenue_growth > 20%  (high-growth bonus)
      +2  profit_margin > 10%   (pricing power / moat)
      +2  roe > 15%             (efficient capital deployment)
      +1  debt_to_equity < 1.0  (financial resilience)
    """
    intl_set = set(intl_tickers)
    all_tickers = us_tickers + intl_tickers

    def _score(ticker: str) -> Optional[dict]:
        try:
            info = yf.Ticker(ticker).info
            # Skip tickers with no meaningful data
            if not (info.get("longName") or info.get("shortName")):
                return None

            revenue_growth  = info.get("revenueGrowth")
            profit_margin   = info.get("profitMargins")
            roe             = info.get("returnOnEquity")
            debt_to_equity  = info.get("debtToEquity")
            market_cap      = info.get("marketCap") or 0

            # Require at least two quality signals to be present
            signals_present = sum([
                revenue_growth is not None,
                profit_margin is not None,
                roe is not None,
            ])
            if signals_present < 2:
                return None

            score = 0.0
            if revenue_growth is not None and revenue_growth > 0.08:
                score += 2
            if revenue_growth is not None and revenue_growth > 0.20:
                score += 1
            if profit_margin is not None and profit_margin > 0.10:
                score += 2
            if roe is not None and roe > 0.15:
                score += 2
            if debt_to_equity is not None and debt_to_equity < 1.0:
                score += 1

            return {
                "ticker":        ticker,
                "universe":      "international" if ticker in intl_set else "us_sp500",
                "name":          info.get("longName") or info.get("shortName", ticker),
                "sector":        info.get("sector", ""),
                "industry":      info.get("industry", ""),
                "quality_score": score,
                "revenue_growth": round(revenue_growth * 100, 1) if revenue_growth is not None else None,
                "profit_margin":  round(profit_margin * 100, 1) if profit_margin is not None else None,
                "roe":            round(roe * 100, 1)            if roe is not None            else None,
                "debt_to_equity": round(debt_to_equity, 2)       if debt_to_equity is not None else None,
                "market_cap_b":   round(market_cap / 1e9, 2)     if market_cap else None,
            }
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_score, t): t for t in all_tickers}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: x["quality_score"], reverse=True)
    return results