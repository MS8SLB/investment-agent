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
        # Point-in-time portfolio value snapshots for benchmark comparison
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                cash REAL NOT NULL,
                invested_value REAL NOT NULL,
                benchmark_price REAL,
                session_type TEXT DEFAULT 'review'
            )
        """)
        # Stocks to monitor and buy later at a better price or after a catalyst
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                reason TEXT NOT NULL,
                target_entry_price REAL,
                added_at TEXT NOT NULL
            )
        """)
        # Screener signal snapshot at time of each buy — enables performance attribution
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                screener_score REAL,
                peg_ratio REAL,
                relative_momentum_pct REAL,
                week52_return_pct REAL,
                revenue_growth_pct REAL,
                profit_margin_pct REAL,
                roe_pct REAL,
                fcf_yield_pct REAL,
                sector TEXT,
                recorded_at TEXT NOT NULL
            )
        """)

        # Stocks the agent analyzed and decided NOT to buy or watchlist —
        # tracked so we can measure whether passing was the right call.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_positions (
                ticker TEXT PRIMARY KEY,
                considered_at TEXT NOT NULL,
                price_at_consideration REAL NOT NULL,
                reason_passed TEXT NOT NULL,
                notes TEXT
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


def save_portfolio_snapshot(
    portfolio_value: float,
    cash: float,
    invested_value: float,
    benchmark_price: Optional[float] = None,
    session_type: str = "review",
) -> None:
    """Record a point-in-time portfolio value alongside the benchmark price."""
    conn = _get_connection()
    with conn:
        conn.execute(
            """INSERT INTO portfolio_snapshots
               (ts, portfolio_value, cash, invested_value, benchmark_price, session_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), portfolio_value, cash, invested_value, benchmark_price, session_type),
        )
    conn.close()


def get_portfolio_snapshots(limit: int = 52) -> list[dict]:
    """Return snapshots ordered oldest-first (suitable for charting and % change calcs)."""
    conn = _get_connection()
    rows = conn.execute(
        """SELECT ts, portfolio_value, cash, invested_value, benchmark_price, session_type
           FROM portfolio_snapshots ORDER BY ts ASC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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


# ── Watchlist ─────────────────────────────────────────────────────────────────

def add_to_watchlist(
    ticker: str,
    reason: str,
    target_entry_price: Optional[float] = None,
    company_name: Optional[str] = None,
) -> dict:
    """Add or update a stock on the watchlist."""
    ticker = ticker.upper()
    conn = _get_connection()
    with conn:
        conn.execute(
            """INSERT INTO watchlist (ticker, company_name, reason, target_entry_price, added_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                   reason = excluded.reason,
                   company_name = excluded.company_name,
                   target_entry_price = excluded.target_entry_price,
                   added_at = excluded.added_at""",
            (ticker, company_name, reason, target_entry_price, datetime.utcnow().isoformat()),
        )
    conn.close()
    return {"success": True, "ticker": ticker}


def get_watchlist() -> list[dict]:
    """Return all current watchlist entries, newest first."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT ticker, company_name, reason, target_entry_price, added_at FROM watchlist ORDER BY added_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def remove_from_watchlist(ticker: str) -> dict:
    """Remove a stock from the watchlist."""
    ticker = ticker.upper()
    conn = _get_connection()
    with conn:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
    conn.close()
    return {"success": True, "ticker": ticker}


# ── Trade signal log ──────────────────────────────────────────────────────────

def save_trade_signals(transaction_id: int, ticker: str, signals: dict) -> None:
    """
    Save the screener signal snapshot at time of a buy.
    Call this immediately after a successful buy, passing the screen_stocks
    result row for the purchased ticker. Enables future performance attribution.
    """
    conn = _get_connection()
    with conn:
        conn.execute(
            """INSERT INTO trade_signals (
                transaction_id, ticker, screener_score, peg_ratio,
                relative_momentum_pct, week52_return_pct, revenue_growth_pct,
                profit_margin_pct, roe_pct, fcf_yield_pct, sector, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                transaction_id,
                ticker.upper(),
                signals.get("score"),
                signals.get("peg_ratio"),
                signals.get("relative_momentum_pct"),
                signals.get("week52_return_pct"),
                signals.get("revenue_growth_pct"),
                signals.get("profit_margin_pct"),
                signals.get("roe_pct"),
                signals.get("fcf_yield_pct"),
                signals.get("sector"),
                datetime.utcnow().isoformat(),
            ),
        )
    conn.close()


def get_trade_outcomes() -> list[dict]:
    """
    Return all buy signal snapshots with their actual outcomes.

    For closed positions: buy price → sell price → return %.
    For open positions: status='open', no return yet.

    Use this to identify which screener signals correlate with positive returns
    over time. As the portfolio accumulates history this becomes a genuine
    feedback loop for improving screening criteria.
    """
    conn = _get_connection()
    rows = conn.execute(
        """SELECT
               ts.transaction_id, ts.ticker, ts.recorded_at AS signal_date,
               ts.screener_score, ts.peg_ratio, ts.relative_momentum_pct,
               ts.week52_return_pct, ts.revenue_growth_pct,
               ts.profit_margin_pct, ts.roe_pct, ts.fcf_yield_pct, ts.sector,
               t.price AS buy_price, t.ts AS buy_date
           FROM trade_signals ts
           JOIN transactions t ON t.id = ts.transaction_id
           ORDER BY ts.recorded_at DESC
           LIMIT 20"""
    ).fetchall()

    results = []
    for row in rows:
        r = dict(row)
        # Find first sell after this buy for the same ticker
        sell = conn.execute(
            """SELECT price AS sell_price, ts AS sell_date, realized_pnl
               FROM transactions
               WHERE action = 'SELL' AND ticker = ? AND ts > ?
               ORDER BY ts ASC LIMIT 1""",
            (r["ticker"], r["buy_date"]),
        ).fetchone()

        if sell:
            return_pct = (sell["sell_price"] - r["buy_price"]) / r["buy_price"] * 100
            r.update({
                "status": "closed",
                "sell_price": sell["sell_price"],
                "sell_date": sell["sell_date"][:10],
                "return_pct": round(return_pct, 2),
                "realized_pnl": sell["realized_pnl"],
            })
        else:
            r["status"] = "open"

        r["buy_date"] = r["buy_date"][:10]
        r["signal_date"] = r["signal_date"][:10]
        results.append(r)

    conn.close()
    return results


def get_signal_performance() -> dict:
    """
    Analyze which screener signals have predicted positive returns across closed trades.
    Returns per-signal statistics split by whether the threshold was met at buy time.
    """
    conn = _get_connection()
    rows = conn.execute("""
        SELECT ts.peg_ratio, ts.fcf_yield_pct, ts.relative_momentum_pct,
               ts.revenue_growth_pct, ts.screener_score,
               t.price AS buy_price, t.ticker,
               (SELECT s2.price FROM transactions s2
                WHERE s2.action = 'SELL' AND s2.ticker = t.ticker AND s2.ts > t.ts
                ORDER BY s2.ts ASC LIMIT 1) AS sell_price
        FROM trade_signals ts
        JOIN transactions t ON t.id = ts.transaction_id
        ORDER BY ts.recorded_at DESC
        LIMIT 20
    """).fetchall()
    conn.close()

    rows = [dict(r) for r in rows]
    closed = [r for r in rows if r["sell_price"] is not None]
    open_count = len(rows) - len(closed)

    if not closed:
        return {
            "message": (
                "No closed trades yet — signal performance will build over time. "
                f"{len(rows)} open position(s) tracked."
            ),
            "total_closed_trades": 0,
            "total_open_trades": open_count,
        }

    thresholds = {
        "peg_lt_1_5": lambda r: r["peg_ratio"] is not None and r["peg_ratio"] < 1.5,
        "fcf_yield_gt_3pct": lambda r: r["fcf_yield_pct"] is not None and r["fcf_yield_pct"] > 3.0,
        "positive_momentum": lambda r: r["relative_momentum_pct"] is not None and r["relative_momentum_pct"] > 0,
        "revenue_growth_gt_10pct": lambda r: r["revenue_growth_pct"] is not None and r["revenue_growth_pct"] > 10.0,
    }

    def _stats(group: list) -> dict:
        if not group:
            return {"count": 0, "positive_rate_pct": None, "avg_return_pct": None}
        returns = [(r["sell_price"] - r["buy_price"]) / r["buy_price"] * 100 for r in group]
        positive = sum(1 for x in returns if x > 0)
        return {
            "count": len(group),
            "positive_rate_pct": round(positive / len(group) * 100, 1),
            "avg_return_pct": round(sum(returns) / len(returns), 2),
        }

    signal_stats = {}
    for name, passes in thresholds.items():
        met = [r for r in closed if passes(r)]
        not_met = [r for r in closed if not passes(r)]
        signal_stats[name] = {
            "threshold_met": _stats(met),
            "threshold_not_met": _stats(not_met),
        }

    return {
        "total_closed_trades": len(closed),
        "total_open_trades": open_count,
        "signal_performance": signal_stats,
        "interpretation": (
            "Where threshold_met shows higher positive_rate_pct and avg_return_pct than "
            "threshold_not_met, that signal is genuinely predictive — weight it more heavily. "
            "Where there is no difference, the signal adds little edge."
        ),
    }


def add_to_shadow_portfolio(
    ticker: str,
    price_at_consideration: float,
    reason_passed: str,
    notes: str = "",
) -> dict:
    """
    Record a stock that was analyzed but not bought or watchlisted.
    Tracks whether the decision to pass was correct over time.
    """
    conn = _get_connection()
    now = datetime.utcnow().isoformat()
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO shadow_positions
                (ticker, considered_at, price_at_consideration, reason_passed, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (ticker.upper(), now, price_at_consideration, reason_passed, notes))
    conn.close()
    return {"status": "recorded", "ticker": ticker.upper(), "price_at_consideration": price_at_consideration}


def get_shadow_positions() -> list[dict]:
    """Return all shadow portfolio positions (stocks analyzed and passed on)."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT ticker, considered_at, price_at_consideration, reason_passed, notes
        FROM shadow_positions
        ORDER BY considered_at DESC
        LIMIT 30
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_portfolio_metrics() -> dict:
    """
    Compute risk and return metrics from portfolio snapshot history.
    Returns Sharpe ratio, max drawdown, annualised volatility, and
    rolling 1/3/6-month returns vs S&P 500.
    Assumes snapshots are roughly monthly (one per review session).
    """
    snapshots = get_portfolio_snapshots(limit=200)
    if len(snapshots) < 2:
        return {
            "message": "Need at least 2 review sessions to compute metrics. Run more reviews.",
            "n_snapshots": len(snapshots),
        }

    values = [s["portfolio_value"] for s in snapshots]
    spy_prices = [s.get("benchmark_price") for s in snapshots]
    n = len(values)

    # Period-over-period returns (between consecutive snapshots)
    returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, n)]

    avg_r = sum(returns) / len(returns)
    variance = sum((r - avg_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
    std_r = variance ** 0.5

    # Annualise assuming monthly snapshots
    annualised_return = ((1 + avg_r) ** 12 - 1) * 100
    annualised_vol = std_r * (12 ** 0.5) * 100
    risk_free = 4.5  # % annual, approx current 10yr treasury

    sharpe = (
        round((annualised_return - risk_free) / annualised_vol, 2)
        if annualised_vol > 0 else None
    )

    # Max drawdown
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Rolling returns — portfolio
    def _port_return(n_periods: int) -> Optional[float]:
        if n < n_periods + 1:
            return None
        start, end = values[-(n_periods + 1)], values[-1]
        return round((end - start) / start * 100, 2)

    # Rolling returns — S&P 500 (only use snapshots where benchmark exists)
    spy_valid = [(s["ts"], s["benchmark_price"]) for s in snapshots if s.get("benchmark_price")]

    def _spy_return(n_periods: int) -> Optional[float]:
        if len(spy_valid) < n_periods + 1:
            return None
        start = spy_valid[-(n_periods + 1)][1]
        end = spy_valid[-1][1]
        return round((end - start) / start * 100, 2)

    total_return = round((values[-1] - values[0]) / values[0] * 100, 2)

    return {
        "n_snapshots": n,
        "total_return_pct": total_return,
        "annualised_return_pct": round(annualised_return, 2) if n >= 3 else None,
        "annualised_volatility_pct": round(annualised_vol, 2) if len(returns) > 1 else None,
        "sharpe_ratio": sharpe if len(returns) > 1 else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "rolling": {
            "1m": {"portfolio_pct": _port_return(1), "benchmark_pct": _spy_return(1)},
            "3m": {"portfolio_pct": _port_return(3), "benchmark_pct": _spy_return(3)},
            "6m": {"portfolio_pct": _port_return(6), "benchmark_pct": _spy_return(6)},
        },
        "note": (
            f"Based on {n} snapshots. "
            "Annualised figures assume monthly snapshot frequency. "
            "Sharpe uses 4.5% annual risk-free rate."
        ),
    }
