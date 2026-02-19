"""
Market data retrieval using yfinance.
Provides stock quotes, fundamentals, and historical data.
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
