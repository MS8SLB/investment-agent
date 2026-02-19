"""
Portfolio manager with SQLite persistence.
Tracks cash, holdings, transaction history, investment theses, and session reflections.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "portfolio.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_portfolio(starting_cash: float = 100_000.0) -> None:
    """Create tables and seed initial cash balance if not already done."""
    conn = _get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                ticker TEXT PRIMARY KEY,
                shares REAL NOT NULL,
                avg_cost REAL NOT NULL,
                first_bought TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                action TEXT NOT NULL,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                total REAL NOT NULL,
                realized_pnl REAL,
                notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)
        # Stores the agent's reasoning at the time of each trade
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_thesis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                thesis TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # Stores the agent's post-session reflections and lessons
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                reflection TEXT NOT NULL,
                portfolio_value REAL,
                session_type TEXT DEFAULT 'review'
            )
        """)

        # Migrate existing DBs: add realized_pnl column if missing
        try:
            conn.execute("ALTER TABLE transactions ADD COLUMN realized_pnl REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Seed initial state only if not present
        existing = conn.execute("SELECT id FROM portfolio_state WHERE id = 1").fetchone()
        if not existing:
            now = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO portfolio_state (id, cash, created_at, updated_at) VALUES (1, ?, ?, ?)",
                (starting_cash, now, now),
            )
    conn.close()


def get_cash() -> float:
    conn = _get_connection()
    row = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()
    conn.close()
    return row["cash"] if row else 0.0


def get_holdings() -> list[dict]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT ticker, shares, avg_cost, first_bought, last_updated FROM holdings ORDER BY ticker"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_holding(ticker: str) -> Optional[dict]:
    conn = _get_connection()
    row = conn.execute(
        "SELECT ticker, shares, avg_cost, first_bought, last_updated FROM holdings WHERE ticker = ?",
        (ticker.upper(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def buy_stock(ticker: str, shares: float, price_per_share: float, notes: str = "") -> dict:
    """
    Execute a paper buy order.
    Returns result dict with success/error info, including transaction_id on success.
    """
    ticker = ticker.upper()
    total_cost = shares * price_per_share
    now = datetime.utcnow().isoformat()

    conn = _get_connection()
    try:
        with conn:
            cash = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()["cash"]
            if total_cost > cash:
                return {
                    "success": False,
                    "error": f"Insufficient funds. Need ${total_cost:,.2f}, have ${cash:,.2f}",
                }

            # Update cash
            conn.execute(
                "UPDATE portfolio_state SET cash = cash - ?, updated_at = ? WHERE id = 1",
                (total_cost, now),
            )

            # Upsert holding with updated avg cost
            existing = conn.execute(
                "SELECT shares, avg_cost FROM holdings WHERE ticker = ?", (ticker,)
            ).fetchone()
            if existing:
                old_shares = existing["shares"]
                old_cost = existing["avg_cost"]
                new_shares = old_shares + shares
                new_avg = (old_shares * old_cost + shares * price_per_share) / new_shares
                conn.execute(
                    "UPDATE holdings SET shares = ?, avg_cost = ?, last_updated = ? WHERE ticker = ?",
                    (new_shares, new_avg, now, ticker),
                )
            else:
                conn.execute(
                    "INSERT INTO holdings (ticker, shares, avg_cost, first_bought, last_updated) VALUES (?, ?, ?, ?, ?)",
                    (ticker, shares, price_per_share, now, now),
                )

            # Record transaction
            cursor = conn.execute(
                "INSERT INTO transactions (ts, action, ticker, shares, price, total, notes) VALUES (?, 'BUY', ?, ?, ?, ?, ?)",
                (now, ticker, shares, price_per_share, total_cost, notes),
            )
            transaction_id = cursor.lastrowid

        return {
            "success": True,
            "transaction_id": transaction_id,
            "ticker": ticker,
            "shares": shares,
            "price": price_per_share,
            "total_cost": total_cost,
            "cash_remaining": cash - total_cost,
        }
    finally:
        conn.close()


def sell_stock(ticker: str, shares: float, price_per_share: float, notes: str = "") -> dict:
    """
    Execute a paper sell order.
    Returns result dict with success/error info, including transaction_id on success.
    """
    ticker = ticker.upper()
    total_proceeds = shares * price_per_share
    now = datetime.utcnow().isoformat()

    conn = _get_connection()
    try:
        with conn:
            holding = conn.execute(
                "SELECT shares, avg_cost FROM holdings WHERE ticker = ?", (ticker,)
            ).fetchone()
            if not holding:
                return {"success": False, "error": f"No position in {ticker}"}
            if shares > holding["shares"]:
                return {
                    "success": False,
                    "error": f"Cannot sell {shares} shares, only own {holding['shares']:.4f}",
                }

            realized_pnl = (price_per_share - holding["avg_cost"]) * shares

            # Update or remove holding
            remaining = holding["shares"] - shares
            if remaining < 1e-6:
                conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
            else:
                conn.execute(
                    "UPDATE holdings SET shares = ?, last_updated = ? WHERE ticker = ?",
                    (remaining, now, ticker),
                )

            # Update cash
            conn.execute(
                "UPDATE portfolio_state SET cash = cash + ?, updated_at = ? WHERE id = 1",
                (total_proceeds, now),
            )

            # Record transaction
            cursor = conn.execute(
                "INSERT INTO transactions (ts, action, ticker, shares, price, total, realized_pnl, notes) VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?)",
                (now, ticker, shares, price_per_share, total_proceeds, realized_pnl, notes),
            )
            transaction_id = cursor.lastrowid

        cash = get_cash()
        return {
            "success": True,
            "transaction_id": transaction_id,
            "ticker": ticker,
            "shares": shares,
            "price": price_per_share,
            "total_proceeds": total_proceeds,
            "realized_pnl": realized_pnl,
            "cash_after": cash,
        }
    finally:
        conn.close()


def save_trade_thesis(transaction_id: int, ticker: str, action: str, thesis: str) -> None:
    """Record the agent's reasoning for a buy or sell at the time of the trade."""
    conn = _get_connection()
    with conn:
        conn.execute(
            "INSERT INTO trade_thesis (transaction_id, ticker, action, thesis, created_at) VALUES (?, ?, ?, ?, ?)",
            (transaction_id, ticker.upper(), action.upper(), thesis, datetime.utcnow().isoformat()),
        )
    conn.close()


def get_investment_memory() -> dict:
    """
    Return past investment theses for current holdings and recently closed positions.
    Gives the agent context about its past reasoning so it can evaluate whether
    original theses are playing out and apply lessons from closed positions.
    """
    conn = _get_connection()

    # Most recent buy thesis for each current holding
    holdings = conn.execute(
        "SELECT ticker, shares, avg_cost, first_bought FROM holdings ORDER BY ticker"
    ).fetchall()

    holding_theses = []
    for h in holdings:
        ticker = h["ticker"]
        thesis_row = conn.execute("""
            SELECT thesis, created_at FROM trade_thesis
            WHERE ticker = ? AND action = 'BUY'
            ORDER BY created_at DESC LIMIT 1
        """, (ticker,)).fetchone()

        holding_theses.append({
            "ticker": ticker,
            "shares": h["shares"],
            "avg_cost_per_share": h["avg_cost"],
            "first_bought": h["first_bought"],
            "original_buy_thesis": thesis_row["thesis"] if thesis_row else "(no thesis recorded)",
            "thesis_date": thesis_row["created_at"] if thesis_row else None,
        })

    # Recently closed positions with their sell thesis and realized P&L
    closed_rows = conn.execute("""
        SELECT t.ticker, t.shares, t.price as sell_price, t.total as proceeds,
               t.realized_pnl, t.ts as sold_at, t.notes as sell_notes,
               tt.thesis as sell_thesis
        FROM transactions t
        LEFT JOIN trade_thesis tt ON tt.transaction_id = t.id
        WHERE t.action = 'SELL'
          AND t.ticker NOT IN (SELECT ticker FROM holdings)
        ORDER BY t.ts DESC
        LIMIT 10
    """).fetchall()

    conn.close()

    closed = []
    for r in closed_rows:
        row = dict(r)
        # Format realized P&L direction clearly
        if row.get("realized_pnl") is not None:
            row["realized_pnl"] = round(row["realized_pnl"], 2)
            row["outcome"] = "profit" if row["realized_pnl"] >= 0 else "loss"
        closed.append(row)

    return {
        "current_holdings_theses": holding_theses,
        "recently_closed_positions": closed,
        "instruction": (
            "Review your original theses for current holdings — are they still valid? "
            "Review closed positions to identify patterns in what worked and what didn't."
        ),
    }


def save_reflection(
    reflection: str,
    portfolio_value: Optional[float] = None,
    session_type: str = "review",
) -> None:
    """Save the agent's post-session reflection."""
    conn = _get_connection()
    with conn:
        conn.execute(
            "INSERT INTO reflections (created_at, reflection, portfolio_value, session_type) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), reflection, portfolio_value, session_type),
        )
    conn.close()


def get_reflections(limit: int = 5) -> list[dict]:
    """Return the most recent post-session reflections."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT created_at, reflection, portfolio_value, session_type FROM reflections ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_transactions(limit: int = 50) -> list[dict]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_agent_message(message: str) -> None:
    conn = _get_connection()
    with conn:
        conn.execute(
            "INSERT INTO agent_log (ts, message) VALUES (?, ?)",
            (datetime.utcnow().isoformat(), message),
        )
    conn.close()


def get_agent_log(limit: int = 20) -> list[dict]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT ts, message FROM agent_log ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
