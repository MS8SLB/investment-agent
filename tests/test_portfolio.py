"""
Tests for agent/portfolio.py
Uses a temporary SQLite DB so nothing touches the real portfolio.
"""

import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Redirect DB_PATH to a fresh temp file for every test."""
    db = str(tmp_path / "test_portfolio.db")
    monkeypatch.setattr("agent.portfolio.DB_PATH", db)
    from agent.portfolio import initialize_portfolio
    initialize_portfolio(100_000.0)
    return db


from agent import portfolio


# ══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestInitialise:

    def test_starting_cash(self):
        assert portfolio.get_cash() == 100_000.0

    def test_no_holdings(self):
        assert portfolio.get_holdings() == []

    def test_idempotent(self):
        """Calling initialize_portfolio twice should not reset cash."""
        portfolio.buy_stock("AAPL", 10, 150.0)
        portfolio.initialize_portfolio(100_000.0)
        assert portfolio.get_cash() == pytest.approx(100_000.0 - 1_500.0)


# ══════════════════════════════════════════════════════════════════════════════
# buy_stock
# ══════════════════════════════════════════════════════════════════════════════

class TestBuyStock:

    def test_successful_buy_deducts_cash(self):
        r = portfolio.buy_stock("AAPL", 10, 150.0)
        assert r["success"] is True
        assert portfolio.get_cash() == pytest.approx(100_000.0 - 1_500.0)

    def test_successful_buy_creates_holding(self):
        portfolio.buy_stock("MSFT", 5, 300.0)
        h = portfolio.get_holding("MSFT")
        assert h is not None
        assert h["shares"] == pytest.approx(5.0)
        assert h["avg_cost"] == pytest.approx(300.0)

    def test_buy_insufficient_funds(self):
        r = portfolio.buy_stock("AAPL", 10_000, 150.0)
        assert r["success"] is False
        assert "Insufficient" in r["error"]
        assert portfolio.get_cash() == pytest.approx(100_000.0)  # unchanged

    def test_buy_adds_to_existing_position(self):
        portfolio.buy_stock("AAPL", 10, 100.0)
        portfolio.buy_stock("AAPL", 10, 200.0)
        h = portfolio.get_holding("AAPL")
        assert h["shares"] == pytest.approx(20.0)
        assert h["avg_cost"] == pytest.approx(150.0)  # (10*100 + 10*200) / 20

    def test_ticker_uppercased(self):
        portfolio.buy_stock("aapl", 1, 150.0)
        assert portfolio.get_holding("AAPL") is not None

    def test_returns_transaction_id(self):
        r = portfolio.buy_stock("AAPL", 1, 150.0)
        assert "transaction_id" in r
        assert isinstance(r["transaction_id"], int)


# ══════════════════════════════════════════════════════════════════════════════
# sell_stock
# ══════════════════════════════════════════════════════════════════════════════

class TestSellStock:

    def setup_method(self):
        portfolio.buy_stock("AAPL", 20, 100.0)

    def test_successful_sell_adds_cash(self):
        cash_before = portfolio.get_cash()
        r = portfolio.sell_stock("AAPL", 10, 120.0)
        assert r["success"] is True
        assert portfolio.get_cash() == pytest.approx(cash_before + 1_200.0)

    def test_sell_reduces_shares(self):
        portfolio.sell_stock("AAPL", 10, 120.0)
        h = portfolio.get_holding("AAPL")
        assert h["shares"] == pytest.approx(10.0)

    def test_sell_all_removes_holding(self):
        portfolio.sell_stock("AAPL", 20, 120.0)
        assert portfolio.get_holding("AAPL") is None

    def test_sell_calculates_realized_pnl(self):
        r = portfolio.sell_stock("AAPL", 10, 120.0)
        assert r["realized_pnl"] == pytest.approx((120.0 - 100.0) * 10)

    def test_sell_records_negative_pnl(self):
        r = portfolio.sell_stock("AAPL", 10, 80.0)
        assert r["realized_pnl"] == pytest.approx((80.0 - 100.0) * 10)

    def test_sell_more_than_owned(self):
        r = portfolio.sell_stock("AAPL", 100, 120.0)
        assert r["success"] is False
        assert "Cannot sell" in r["error"]

    def test_sell_non_existent_ticker(self):
        r = portfolio.sell_stock("NVDA", 1, 500.0)
        assert r["success"] is False
        assert "No position" in r["error"]


# ══════════════════════════════════════════════════════════════════════════════
# get_holdings / get_holding
# ══════════════════════════════════════════════════════════════════════════════

class TestHoldings:

    def test_get_holdings_sorted_alphabetically(self):
        portfolio.buy_stock("MSFT", 1, 300.0)
        portfolio.buy_stock("AAPL", 1, 150.0)
        portfolio.buy_stock("NVDA", 1, 500.0)
        tickers = [h["ticker"] for h in portfolio.get_holdings()]
        assert tickers == sorted(tickers)

    def test_get_holding_case_insensitive(self):
        portfolio.buy_stock("AAPL", 1, 150.0)
        assert portfolio.get_holding("aapl") is not None
        assert portfolio.get_holding("AAPL") is not None

    def test_get_holding_missing_returns_none(self):
        assert portfolio.get_holding("ZZZZ") is None


# ══════════════════════════════════════════════════════════════════════════════
# Transactions
# ══════════════════════════════════════════════════════════════════════════════

class TestTransactions:

    def test_buy_creates_transaction(self):
        portfolio.buy_stock("AAPL", 5, 150.0)
        txs = portfolio.get_transactions(10)
        assert len(txs) == 1
        assert txs[0]["action"] == "BUY"
        assert txs[0]["ticker"] == "AAPL"

    def test_sell_creates_transaction(self):
        portfolio.buy_stock("AAPL", 10, 150.0)
        portfolio.sell_stock("AAPL", 5, 160.0)
        txs = portfolio.get_transactions(10)
        actions = [t["action"] for t in txs]
        assert "SELL" in actions

    def test_transactions_ordered_newest_first(self):
        portfolio.buy_stock("AAPL", 1, 100.0)
        portfolio.buy_stock("MSFT", 1, 200.0)
        txs = portfolio.get_transactions(10)
        assert txs[0]["ticker"] == "MSFT"

    def test_limit_respected(self):
        for i in range(5):
            portfolio.buy_stock("AAPL", 1, float(100 + i))
        txs = portfolio.get_transactions(3)
        assert len(txs) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Trade thesis
# ══════════════════════════════════════════════════════════════════════════════

class TestTradeThesis:

    def test_thesis_saved_and_retrieved(self):
        r = portfolio.buy_stock("AAPL", 1, 150.0, notes="Great company")
        portfolio.save_trade_thesis(r["transaction_id"], "AAPL", "BUY", "Strong thesis")
        mem = portfolio.get_investment_memory()
        theses = mem["current_holdings_theses"]
        assert len(theses) == 1
        assert theses[0]["original_buy_thesis"] == "Strong thesis"

    def test_no_thesis_returns_default(self):
        portfolio.buy_stock("AAPL", 1, 150.0)
        mem = portfolio.get_investment_memory()
        assert mem["current_holdings_theses"][0]["original_buy_thesis"] == "(no thesis recorded)"


# ══════════════════════════════════════════════════════════════════════════════
# Reflections & snapshots
# ══════════════════════════════════════════════════════════════════════════════

class TestReflections:

    def test_save_and_retrieve_reflection(self):
        portfolio.save_reflection("Great session today", portfolio_value=105_000.0)
        refs = portfolio.get_reflections(5)
        assert len(refs) == 1
        assert refs[0]["reflection"] == "Great session today"
        assert refs[0]["portfolio_value"] == pytest.approx(105_000.0)

    def test_multiple_reflections_ordered_newest_first(self):
        portfolio.save_reflection("Session 1", 100_000.0)
        portfolio.save_reflection("Session 2", 102_000.0)
        refs = portfolio.get_reflections(5)
        assert refs[0]["reflection"] == "Session 2"

    def test_limit_respected(self):
        for i in range(5):
            portfolio.save_reflection(f"Session {i}", float(100_000 + i))
        refs = portfolio.get_reflections(3)
        assert len(refs) == 3


class TestSnapshots:

    def test_save_and_retrieve_snapshot(self):
        portfolio.save_portfolio_snapshot(105_000.0, 20_000.0, 85_000.0, benchmark_price=4800.0)
        snaps = portfolio.get_portfolio_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["portfolio_value"] == pytest.approx(105_000.0)
        assert snaps[0]["benchmark_price"] == pytest.approx(4800.0)

    def test_snapshots_ordered_oldest_first(self):
        portfolio.save_portfolio_snapshot(100_000.0, 100_000.0, 0.0)
        portfolio.save_portfolio_snapshot(105_000.0, 80_000.0, 25_000.0)
        snaps = portfolio.get_portfolio_snapshots()
        assert snaps[0]["portfolio_value"] == pytest.approx(100_000.0)
        assert snaps[-1]["portfolio_value"] == pytest.approx(105_000.0)
