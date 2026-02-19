"""
Tests for agent/market_data.py
All yfinance calls are mocked so tests run without network access.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, date
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent import market_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ticker(info=None, news=None, calendar=None, earnings_dates=None,
                upgrades=None, insider=None, history=None):
    """Return a mock yf.Ticker with the given attributes."""
    t = MagicMock()
    t.info = info or {}
    t.news = news if news is not None else []
    t.calendar = calendar
    t.earnings_dates = earnings_dates
    t.upgrades_downgrades = upgrades
    t.insider_transactions = insider
    if history is not None:
        t.history.return_value = history
    else:
        t.history.return_value = pd.DataFrame()
    return t


# ══════════════════════════════════════════════════════════════════════════════
# get_stock_quote
# ══════════════════════════════════════════════════════════════════════════════

class TestGetStockQuote:

    def _call(self, info):
        with patch("agent.market_data.yf.Ticker", return_value=make_ticker(info=info)):
            return market_data.get_stock_quote("AAPL")

    def test_returns_price_from_currentPrice(self):
        r = self._call({"currentPrice": 150.0, "currency": "USD",
                        "longName": "Apple Inc.", "exchange": "NASDAQ"})
        assert r["price"] == 150.0
        assert r["ticker"] == "AAPL"
        assert r["name"] == "Apple Inc."

    def test_falls_back_to_regularMarketPrice(self):
        r = self._call({"regularMarketPrice": 149.5})
        assert r["price"] == 149.5

    def test_falls_back_to_previousClose(self):
        r = self._call({"previousClose": 148.0})
        assert r["price"] == 148.0

    def test_returns_error_when_no_price(self):
        r = self._call({})
        assert "error" in r

    def test_returns_error_on_exception(self):
        with patch("agent.market_data.yf.Ticker", side_effect=Exception("network error")):
            r = market_data.get_stock_quote("AAPL")
        assert "error" in r

    def test_includes_market_stats(self):
        r = self._call({
            "currentPrice": 150.0,
            "marketCap": 2_000_000_000_000,
            "fiftyTwoWeekHigh": 200.0,
            "fiftyTwoWeekLow": 120.0,
        })
        assert r["market_cap"] == 2_000_000_000_000
        assert r["52w_high"] == 200.0
        assert r["52w_low"] == 120.0


# ══════════════════════════════════════════════════════════════════════════════
# get_stock_fundamentals
# ══════════════════════════════════════════════════════════════════════════════

class TestGetStockFundamentals:

    def _call(self, info):
        with patch("agent.market_data.yf.Ticker", return_value=make_ticker(info=info)):
            return market_data.get_stock_fundamentals("MSFT")

    def test_returns_core_fields(self):
        r = self._call({
            "longName": "Microsoft Corp",
            "sector": "Technology",
            "industry": "Software",
            "trailingPE": 30.5,
            "profitMargins": 0.35,
            "returnOnEquity": 0.45,
            "dividendYield": 0.008,
        })
        assert r["ticker"] == "MSFT"
        assert r["sector"] == "Technology"
        assert r["pe_ratio"] == 30.5
        assert r["profit_margin"] == 0.35
        assert r["roe"] == 0.45
        assert r["dividend_yield"] == 0.008

    def test_missing_fields_return_none(self):
        r = self._call({"longName": "Microsoft Corp"})
        assert r["pe_ratio"] is None
        assert r["profit_margin"] is None

    def test_error_on_exception(self):
        with patch("agent.market_data.yf.Ticker", side_effect=RuntimeError("fail")):
            r = market_data.get_stock_fundamentals("MSFT")
        assert "error" in r


# ══════════════════════════════════════════════════════════════════════════════
# get_price_history
# ══════════════════════════════════════════════════════════════════════════════

class TestGetPriceHistory:

    def _make_hist(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.DataFrame({
            "Close": [100.0, 102.0, 101.0, 105.0, 110.0],
            "High":  [101.0, 103.0, 102.0, 106.0, 112.0],
            "Low":   [ 99.0, 101.0, 100.0, 104.0, 109.0],
            "Volume":[1_000_000]*5,
        }, index=dates)

    def test_returns_correct_pct_change(self):
        ticker = make_ticker(history=self._make_hist())
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_price_history("AAPL", "1y")
        assert r["start_price"] == 100.0
        assert r["current_price"] == 110.0
        assert r["pct_change"] == pytest.approx(10.0)

    def test_returns_high_low(self):
        ticker = make_ticker(history=self._make_hist())
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_price_history("AAPL", "1y")
        assert r["high"] == 112.0
        assert r["low"] == 99.0

    def test_empty_history_returns_error(self):
        ticker = make_ticker(history=pd.DataFrame())
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_price_history("AAPL", "1y")
        assert "error" in r


# ══════════════════════════════════════════════════════════════════════════════
# get_stock_news — both yfinance formats
# ══════════════════════════════════════════════════════════════════════════════

class TestGetStockNews:

    def test_legacy_flat_dict_format(self):
        """yfinance < 1.0 format: flat dict with providerPublishTime."""
        news = [{
            "title": "Apple hits record high",
            "publisher": "Reuters",
            "providerPublishTime": 1700000000,
            "link": "https://reuters.com/aapl",
            "relatedTickers": ["AAPL"],
        }]
        ticker = make_ticker(news=news)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_stock_news("AAPL")
        assert len(r) == 1
        assert r[0]["title"] == "Apple hits record high"
        assert r[0]["publisher"] == "Reuters"
        assert r[0]["link"] == "https://reuters.com/aapl"

    def test_new_nested_content_format(self):
        """yfinance >= 1.0 format: nested under 'content' key."""
        news = [{
            "id": "abc123",
            "content": {
                "title": "Apple launches new product",
                "pubDate": "2024-11-15T10:30:00Z",
                "provider": {"displayName": "Bloomberg"},
                "canonicalUrl": {"url": "https://bloomberg.com/apple"},
                "relatedTickers": [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
            }
        }]
        ticker = make_ticker(news=news)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_stock_news("AAPL")
        assert len(r) == 1
        assert r[0]["title"] == "Apple launches new product"
        assert r[0]["publisher"] == "Bloomberg"
        assert r[0]["link"] == "https://bloomberg.com/apple"
        assert "AAPL" in r[0]["related_tickers"]

    def test_empty_news_returns_note(self):
        ticker = make_ticker(news=[])
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_stock_news("AAPL")
        assert "note" in r[0]

    def test_respects_limit(self):
        news = [{"title": f"News {i}", "publisher": "X",
                 "providerPublishTime": 1700000000,
                 "link": "", "relatedTickers": []}
                for i in range(10)]
        ticker = make_ticker(news=news)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_stock_news("AAPL", limit=3)
        assert len(r) == 3

    def test_error_on_exception(self):
        with patch("agent.market_data.yf.Ticker", side_effect=Exception("timeout")):
            r = market_data.get_stock_news("AAPL")
        assert "error" in r[0]


# ══════════════════════════════════════════════════════════════════════════════
# get_earnings_calendar
# ══════════════════════════════════════════════════════════════════════════════

class TestGetEarningsCalendar:

    def _make_earnings_df(self):
        """Simulate a 4-quarter earnings history DataFrame."""
        idx = pd.DatetimeIndex([
            "2024-10-31", "2024-07-31", "2024-04-30", "2024-01-31"
        ], name="Earnings Date")
        return pd.DataFrame({
            "EPS Estimate": [1.50, 1.40, 1.35, 1.30],
            "Reported EPS": [1.60, 1.38, 1.40, 1.25],
            "Surprise(%)":  [ 6.7, -1.4,  3.7, -3.8],
        }, index=idx)

    def test_legacy_calendar_format(self):
        cal = {
            "Earnings Date": [pd.Timestamp("2025-01-28")],
            "Earnings Average": 2.10,
            "Earnings Low": 1.90,
            "Earnings High": 2.30,
            "Revenue Average": 124_000_000_000,
            "Revenue Low": 120_000_000_000,
            "Revenue High": 128_000_000_000,
        }
        ticker = make_ticker(calendar=cal, earnings_dates=self._make_earnings_df())
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_earnings_calendar("AAPL")
        assert r["next_earnings_date"] == "2025-01-28"
        assert r["eps_estimate_avg"] == pytest.approx(2.10)
        assert r["revenue_estimate_avg"] == 124_000_000_000

    def test_earnings_history_beat_miss(self):
        ticker = make_ticker(calendar=None, earnings_dates=self._make_earnings_df())
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_earnings_calendar("AAPL")
        hist = r["last_4_quarters"]
        assert len(hist) == 4
        assert hist[0]["beat"] is True   # 1.60 > 1.50
        assert hist[1]["beat"] is False  # 1.38 < 1.40
        assert hist[0]["surprise_pct"] == pytest.approx(6.7, abs=0.1)

    def test_no_calendar_returns_none_date(self):
        ticker = make_ticker(calendar=None, earnings_dates=None)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_earnings_calendar("AAPL")
        assert r["next_earnings_date"] is None
        assert r["last_4_quarters"] == []

    def test_empty_earnings_dates_returns_empty_list(self):
        ticker = make_ticker(calendar=None, earnings_dates=pd.DataFrame())
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_earnings_calendar("AAPL")
        assert r["last_4_quarters"] == []


# ══════════════════════════════════════════════════════════════════════════════
# get_analyst_upgrades
# ══════════════════════════════════════════════════════════════════════════════

class TestGetAnalystUpgrades:

    def _make_df(self, columns):
        idx = pd.DatetimeIndex(["2024-11-01", "2024-10-15"])
        data = {c: ["val_a", "val_b"] for c in columns}
        return pd.DataFrame(data, index=idx)

    def test_legacy_column_names(self):
        df = self._make_df(["Firm", "Action", "FromGrade", "ToGrade"])
        df["Firm"] = ["Goldman Sachs", "Morgan Stanley"]
        df["Action"] = ["upgrade", "downgrade"]
        df["FromGrade"] = ["Neutral", "Buy"]
        df["ToGrade"] = ["Buy", "Hold"]
        ticker = make_ticker(upgrades=df)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_analyst_upgrades("AAPL")
        assert r[0]["firm"] == "Goldman Sachs"
        assert r[0]["action"] == "upgrade"
        assert r[0]["to_grade"] == "Buy"

    def test_lowercase_column_names(self):
        """yfinance 1.x may use lowercase column names."""
        df = self._make_df(["firm", "action", "fromgrade", "tograde"])
        df["firm"] = ["Citi", "BofA"]
        df["action"] = ["init", "upgrade"]
        df["fromgrade"] = ["", "Hold"]
        df["tograde"] = ["Buy", "Buy"]
        ticker = make_ticker(upgrades=df)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_analyst_upgrades("AAPL")
        assert r[0]["firm"] == "Citi"
        assert r[0]["to_grade"] == "Buy"

    def test_empty_returns_note(self):
        ticker = make_ticker(upgrades=pd.DataFrame())
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_analyst_upgrades("AAPL")
        assert "note" in r[0]

    def test_respects_limit(self):
        idx = pd.DatetimeIndex([f"2024-11-{i+1:02d}" for i in range(8)])
        df = pd.DataFrame({
            "Firm": ["Bank"] * 8, "Action": ["upgrade"] * 8,
            "FromGrade": ["Hold"] * 8, "ToGrade": ["Buy"] * 8,
        }, index=idx)
        ticker = make_ticker(upgrades=df)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_analyst_upgrades("AAPL", limit=3)
        assert len(r) == 3


# ══════════════════════════════════════════════════════════════════════════════
# get_insider_activity
# ══════════════════════════════════════════════════════════════════════════════

class TestGetInsiderActivity:

    def _make_df(self, col_names):
        data = {
            col_names["date"]:     [pd.Timestamp("2024-11-01"), pd.Timestamp("2024-10-15")],
            col_names["insider"]:  ["Tim Cook", "Luca Maestri"],
            col_names["position"]: ["CEO", "CFO"],
            col_names["text"]:     ["Purchase", "Sale"],
            col_names["shares"]:   [10_000, 5_000],
            col_names["value"]:    [1_500_000, 750_000],
        }
        return pd.DataFrame(data)

    def test_legacy_column_names(self):
        df = self._make_df({"date": "startDate", "insider": "insider",
                            "position": "position", "text": "text",
                            "shares": "shares", "value": "value"})
        ticker = make_ticker(insider=df)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_insider_activity("AAPL")
        assert len(r["transactions"]) == 2
        assert r["transactions"][0]["insider"] == "Tim Cook"
        assert r["transactions"][0]["shares"] == 10_000
        assert r["transactions"][0]["value_usd"] == 1_500_000

    def test_alternative_column_names(self):
        """yfinance 1.x may use different column names."""
        df = self._make_df({"date": "date", "insider": "reportingName",
                            "position": "relationship", "text": "transactionText",
                            "shares": "sharesTransacted", "value": "transactionValue"})
        ticker = make_ticker(insider=df)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_insider_activity("AAPL")
        assert len(r["transactions"]) == 2
        assert r["transactions"][0]["shares"] == 10_000

    def test_buy_sell_summary(self):
        df = self._make_df({"date": "startDate", "insider": "insider",
                            "position": "position", "text": "text",
                            "shares": "shares", "value": "value"})
        ticker = make_ticker(insider=df)
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_insider_activity("AAPL")
        assert r["recent_buy_transactions"] == 1   # "Purchase"
        assert r["recent_sell_transactions"] == 1  # "Sale"

    def test_empty_df_returns_note(self):
        ticker = make_ticker(insider=pd.DataFrame())
        with patch("agent.market_data.yf.Ticker", return_value=ticker):
            r = market_data.get_insider_activity("AAPL")
        assert r["transactions"] == []
        assert "note" in r


# ══════════════════════════════════════════════════════════════════════════════
# get_market_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestGetMarketSummary:

    def test_returns_all_indices(self):
        def fake_ticker(sym):
            t = MagicMock()
            t.info = {"regularMarketPrice": 5000.0, "regularMarketPreviousClose": 4950.0}
            return t
        with patch("agent.market_data.yf.Ticker", side_effect=fake_ticker):
            r = market_data.get_market_summary()
        assert "S&P 500" in r
        assert "NASDAQ" in r
        assert "VIX" in r

    def test_calculates_change_pct(self):
        def fake_ticker(sym):
            t = MagicMock()
            t.info = {"regularMarketPrice": 110.0, "regularMarketPreviousClose": 100.0}
            return t
        with patch("agent.market_data.yf.Ticker", side_effect=fake_ticker):
            r = market_data.get_market_summary()
        assert r["S&P 500"]["change_pct"] == pytest.approx(10.0)

    def test_handles_individual_ticker_failure(self):
        call_count = [0]
        def fake_ticker(sym):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("timeout")
            t = MagicMock()
            t.info = {"regularMarketPrice": 100.0, "regularMarketPreviousClose": 99.0}
            return t
        with patch("agent.market_data.yf.Ticker", side_effect=fake_ticker):
            r = market_data.get_market_summary()
        # Should not raise; first index has None values, rest are fine
        assert r["S&P 500"]["price"] is None


# ══════════════════════════════════════════════════════════════════════════════
# get_macro_environment
# ══════════════════════════════════════════════════════════════════════════════

class TestGetMacroEnvironment:

    def test_returns_expected_structure(self):
        def fake_ticker(sym):
            t = MagicMock()
            t.info = {"regularMarketPrice": 4.5, "regularMarketPreviousClose": 4.4}
            return t
        with patch("agent.market_data.yf.Ticker", side_effect=fake_ticker):
            r = market_data.get_macro_environment()
        assert "rates" in r
        assert "dollar" in r
        assert "commodities" in r
        assert "sentiment" in r
        assert "key_signals" in r

    def test_inverted_yield_curve_signal(self):
        prices = {"^TNX": 3.5, "^IRX": 5.0}  # inverted: short > long
        def fake_ticker(sym):
            t = MagicMock()
            p = prices.get(sym, 100.0)
            t.info = {"regularMarketPrice": p, "regularMarketPreviousClose": p}
            return t
        with patch("agent.market_data.yf.Ticker", side_effect=fake_ticker):
            r = market_data.get_macro_environment()
        assert "inverted" in r["rates"]["yield_curve_status"]
        signals_text = " ".join(r["key_signals"])
        assert "inverted" in signals_text

    def test_handles_all_failures_gracefully(self):
        with patch("agent.market_data.yf.Ticker", side_effect=Exception("network error")):
            r = market_data.get_macro_environment()
        assert "rates" in r
        assert r["rates"]["ten_yr_treasury_yield_pct"] is None
