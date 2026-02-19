"""
Market data retrieval using yfinance.
Provides stock quotes, fundamentals, historical data, news, earnings, and analyst data.
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


def get_stock_quote(ticker: str) -> dict:
    """Get current price and basic info for a ticker."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if price is None:
            return {"error": f"Could not retrieve price for {ticker}"}

        return {
            "ticker": ticker.upper(),
            "price": price,
            "currency": info.get("currency", "USD"),
            "name": info.get("longName") or info.get("shortName", ticker),
            "exchange": info.get("exchange", ""),
            "market_cap": info.get("marketCap"),
            "volume": info.get("regularMarketVolume") or info.get("volume"),
            "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
        }
    except Exception as e:
        return {"error": str(e)}


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
        if not articles:
            return [{"note": f"No recent news found for {ticker.upper()}"}]
        return articles
    except Exception as e:
        return [{"error": str(e)}]


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
        try:
            cal = t.calendar
            if cal is not None:
                # yfinance >= 1.0 may return a dict keyed by date strings or a
                # DataFrame; yfinance < 1.0 returns a plain dict with list values.
                if isinstance(cal, dict) and cal:
                    # Try the legacy key names first
                    dates = cal.get("Earnings Date") or cal.get("earningsDate") or []
                    if dates:
                        first = dates[0]
                        result["next_earnings_date"] = (
                            str(first.date()) if hasattr(first, "date") else str(first)
                        )
                    else:
                        result["next_earnings_date"] = None

                    def _safe(key, alt=None):
                        v = cal.get(key) or (cal.get(alt) if alt else None)
                        return float(v) if v is not None else None

                    result["eps_estimate_avg"]     = _safe("Earnings Average", "epsAverage")
                    result["eps_estimate_low"]      = _safe("Earnings Low", "epsLow")
                    result["eps_estimate_high"]     = _safe("Earnings High", "epsHigh")
                    result["revenue_estimate_avg"]  = _safe("Revenue Average", "revenueAverage")
                    result["revenue_estimate_low"]  = _safe("Revenue Low", "revenueLow")
                    result["revenue_estimate_high"] = _safe("Revenue High", "revenueHigh")
                else:
                    result["next_earnings_date"] = None
        except Exception:
            result["next_earnings_date"] = None

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
