"""
Market data retrieval using yfinance with Polygon.io as a fallback.
Provides stock quotes, fundamentals, historical data, news, earnings, and analyst data.
"""

import os
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional


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
    """Get fundamental data useful for long-term investing."""
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


def search_stocks(query: str) -> list[dict]:
    """
    Look up tickers for a company name or keyword using yfinance search.
    Returns a list of matching results.
    """
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


def screen_stocks(tickers: list, top_n: int = 25) -> list:
    """
    Run a fast parallel fundamental screen across a list of tickers.
    Returns the top_n candidates ranked by a composite quality + value score.
    Cap input at 100 tickers per call for speed.
    """
    def _fetch(ticker: str) -> Optional[dict]:
        try:
            info = yf.Ticker(ticker).info
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            if not price:
                return None

            pe = info.get("trailingPE")
            forward_pe = info.get("forwardPE")
            peg = info.get("pegRatio")
            revenue_growth = info.get("revenueGrowth")
            profit_margin = info.get("profitMargins")
            roe = info.get("returnOnEquity")
            debt_to_equity = info.get("debtToEquity")
            market_cap = info.get("marketCap") or 0
            volume = info.get("regularMarketVolume") or info.get("volume") or 0
            week52_change = info.get("52WeekChange")        # stock 52-wk return
            sp52_change = info.get("SandP52WeekChange")     # S&P 500 52-wk return
            free_cashflow = info.get("freeCashflow")
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
                "name": info.get("longName") or info.get("shortName", ticker),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
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
                "recommendation": info.get("recommendationKey", ""),
                "score": score,
            }
        except Exception:
            return None

    batch = tickers[:100]
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_fetch, t): t for t in batch}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]
