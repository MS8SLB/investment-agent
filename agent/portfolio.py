"""
Portfolio manager with SQLite persistence.
Tracks cash, holdings, transaction history, investment theses, and session reflections.
"""

import sqlite3
import os
from datetime import datetime, timedelta
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS dividends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                amount_per_share REAL NOT NULL,
                shares_held REAL NOT NULL,
                total_amount REAL NOT NULL,
                ex_date TEXT,
                recorded_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS universe_scores (
                ticker      TEXT PRIMARY KEY,
                universe    TEXT NOT NULL,
                name        TEXT,
                sector      TEXT,
                industry    TEXT,
                quality_score   REAL NOT NULL,
                revenue_growth  REAL,
                profit_margin   REAL,
                roe             REAL,
                debt_to_equity  REAL,
                scored_at   TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                price       REAL,
                target_price REAL,
                pct_vs_target REAL,
                notes       TEXT,
                recorded_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_cache (
                ticker          TEXT PRIMARY KEY,
                report_json     TEXT NOT NULL,
                recommendation  TEXT,
                conviction_score REAL,
                price_at_research REAL,
                researched_at   TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                regime TEXT NOT NULL,
                previous_regime TEXT,
                vix REAL,
                gdp_growth REAL,
                core_cpi REAL,
                ten_yr_yield REAL,
                yield_inverted INTEGER,
                detected_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_value REAL NOT NULL,
                spy_price REAL NOT NULL,
                spy_shares_equivalent REAL NOT NULL,
                recorded_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                conviction_score INTEGER,
                predicted_iv REAL,
                price_at_decision REAL,
                mos_at_decision REAL,
                decision_date TEXT NOT NULL,
                outcome_price REAL,
                outcome_return_pct REAL,
                outcome_vs_spy_pct REAL,
                outcome_date TEXT,
                outcome_notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS earnings_call_sentiment (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT NOT NULL,
                quarter      TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                tone_direction  TEXT NOT NULL,
                beat_miss       TEXT,
                key_signals     TEXT NOT NULL DEFAULT '[]',
                raw_summary     TEXT,
                analyzed_at     TEXT NOT NULL,
                UNIQUE(ticker, quarter)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_efficiency (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date  TEXT NOT NULL,
                total_tool_calls  INTEGER DEFAULT 0,
                unique_tools_used INTEGER DEFAULT 0,
                stocks_researched INTEGER DEFAULT 0,
                stocks_bought     INTEGER DEFAULT 0,
                stocks_watchlisted INTEGER DEFAULT 0,
                duration_seconds  INTEGER,
                notes         TEXT,
                recorded_at   TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_audit (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date                TEXT NOT NULL,
                tickers_screened            INTEGER DEFAULT 0,
                tickers_researched          INTEGER DEFAULT 0,
                buys_made                   INTEGER DEFAULT 0,
                watchlist_added             INTEGER DEFAULT 0,
                shadow_added                INTEGER DEFAULT 0,
                workflow_issues_logged      INTEGER DEFAULT 0,
                re_researched_watchlist     INTEGER DEFAULT 0,
                deviated_from_matrix        INTEGER DEFAULT 0,
                duplicate_tool_calls        INTEGER DEFAULT 0,
                contradicted_prior_session  INTEGER DEFAULT 0,
                audit_notes                 TEXT,
                recorded_at                 TEXT NOT NULL
            )
        """)

        # Migrate existing DBs: add realized_pnl column if missing
        try:
            conn.execute("ALTER TABLE transactions ADD COLUMN realized_pnl REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Migrate existing DBs: add sector and archetype columns to prediction_tracking
        try:
            conn.execute("ALTER TABLE prediction_tracking ADD COLUMN sector TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE prediction_tracking ADD COLUMN archetype TEXT")
        except sqlite3.OperationalError:
            pass

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
    """Record a point-in-time portfolio value alongside the benchmark price.
    One snapshot per day — subsequent calls on the same date update the existing row."""
    conn = _get_connection()
    today = datetime.utcnow().date().isoformat()
    with conn:
        existing = conn.execute(
            "SELECT id FROM portfolio_snapshots WHERE ts LIKE ? AND session_type = ?",
            (f"{today}%", session_type),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE portfolio_snapshots
                   SET portfolio_value=?, cash=?, invested_value=?, benchmark_price=?, ts=?
                   WHERE id=?""",
                (portfolio_value, cash, invested_value, benchmark_price,
                 datetime.utcnow().isoformat(), existing[0]),
            )
        else:
            conn.execute(
                """INSERT INTO portfolio_snapshots
                   (ts, portfolio_value, cash, invested_value, benchmark_price, session_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (datetime.utcnow().isoformat(), portfolio_value, cash,
                 invested_value, benchmark_price, session_type),
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


def get_first_trade_date() -> Optional[str]:
    """Return the ISO date (YYYY-MM-DD) of the first transaction, or None if no trades have been made."""
    conn = _get_connection()
    row = conn.execute("SELECT MIN(ts) FROM transactions").fetchone()
    conn.close()
    if row and row[0]:
        return row[0][:10]
    return None


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


def reset_portfolio(starting_cash: float) -> None:
    """
    Full reset: delete the DB file entirely and reinitialise from scratch.
    This guarantees a clean state regardless of any in-memory or WAL caching.
    """
    import shutil
    # Remove the DB file (and any WAL / SHM sidecar files)
    for suffix in ("", "-wal", "-shm"):
        path = DB_PATH + suffix
        if os.path.exists(path):
            os.remove(path)
    # Recreate tables and seed fresh cash balance
    initialize_portfolio(starting_cash)


def get_portfolio_metrics() -> dict:
    """
    Compute risk and return metrics from portfolio snapshot history.
    Returns Sharpe ratio, max drawdown, annualised volatility, and
    rolling 1/3/6-month returns vs S&P 500.
    Only considers snapshots from the first trade date onward.
    Annualisation uses the actual elapsed time between snapshots.
    """
    first_trade_date = get_first_trade_date()
    if not first_trade_date:
        return {
            "message": "No trades made yet. Performance metrics are computed after the first trade.",
            "n_snapshots": 0,
        }
    all_snapshots = get_portfolio_snapshots(limit=200)
    snapshots = [s for s in all_snapshots if s["ts"][:10] >= first_trade_date]
    if len(snapshots) < 2:
        return {
            "message": "Need at least 2 snapshots after the first trade to compute metrics.",
            "n_snapshots": len(snapshots),
        }

    values = [s["portfolio_value"] for s in snapshots]
    n = len(values)

    # ── Actual time span ───────────────────────────────────────────────────────
    from datetime import datetime as _dt
    try:
        first_dt = _dt.fromisoformat(snapshots[0]["ts"])
        last_dt  = _dt.fromisoformat(snapshots[-1]["ts"])
        span_days = max((last_dt - first_dt).total_seconds() / 86400, 0.01)
    except Exception:
        span_days = 0.0

    # Annualised stats are only meaningful once we have >= 30 days of data
    enough_data = span_days >= 30

    # Period-over-period returns
    returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, n)]
    avg_r    = sum(returns) / len(returns)
    variance = sum((r - avg_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
    std_r    = variance ** 0.5

    # Annualise using ACTUAL inter-snapshot frequency
    if span_days > 0 and len(returns) > 0:
        avg_period_days = span_days / len(returns)        # avg days between snapshots
        periods_per_year = 365.0 / avg_period_days
    else:
        periods_per_year = 12.0                           # fallback

    annualised_return = ((1 + avg_r) ** periods_per_year - 1) * 100
    annualised_vol    = std_r * (periods_per_year ** 0.5) * 100
    risk_free         = 4.5   # % annual, approx 10yr treasury

    sharpe = (
        round((annualised_return - risk_free) / annualised_vol, 2)
        if annualised_vol > 0 else None
    )

    # Max drawdown
    peak   = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Rolling returns — portfolio (measured in snapshots, not calendar time)
    def _port_return(n_periods: int) -> Optional[float]:
        if n < n_periods + 1:
            return None
        start, end = values[-(n_periods + 1)], values[-1]
        return round((end - start) / start * 100, 2)

    # Rolling returns — S&P 500
    spy_valid = [(s["ts"], s["benchmark_price"]) for s in snapshots if s.get("benchmark_price")]

    def _spy_return(n_periods: int) -> Optional[float]:
        if len(spy_valid) < n_periods + 1:
            return None
        start = spy_valid[-(n_periods + 1)][1]
        end   = spy_valid[-1][1]
        return round((end - start) / start * 100, 2)

    total_return = round((values[-1] - values[0]) / values[0] * 100, 2)

    return {
        "n_snapshots":             n,
        "span_days":               round(span_days, 1),
        "total_return_pct":        total_return,
        "annualised_return_pct":   round(annualised_return, 2) if enough_data else None,
        "annualised_volatility_pct": round(annualised_vol, 2)  if enough_data else None,
        "sharpe_ratio":            sharpe                       if enough_data else None,
        "max_drawdown_pct":        round(max_dd * 100, 2),
        "rolling": {
            "1m": {"portfolio_pct": _port_return(1), "benchmark_pct": _spy_return(1)},
            "3m": {"portfolio_pct": _port_return(3), "benchmark_pct": _spy_return(3)},
            "6m": {"portfolio_pct": _port_return(6), "benchmark_pct": _spy_return(6)},
        },
        "note": (
            f"Based on {n} snapshots over {round(span_days, 1)} days. "
            + ("Annualised figures use actual snapshot frequency. Sharpe uses 4.5% risk-free rate."
               if enough_data else
               "Annualised stats require 30+ days of data.")
        ),
    }


def close_prediction(ticker: str, outcome_price: float) -> None:
    """
    Record the outcome of the most recent open prediction for a ticker.
    Called automatically when a position is sold.
    """
    conn = _get_connection()
    # Find the most recent open prediction for this ticker
    row = conn.execute(
        """SELECT id, price_at_decision FROM prediction_tracking
           WHERE ticker = ? AND outcome_price IS NULL
           ORDER BY id DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if row:
        pred_id = row["id"]
        price_at_decision = row["price_at_decision"] or outcome_price
        outcome_return_pct = (outcome_price / price_at_decision - 1) * 100 if price_at_decision else None
        conn.execute(
            """UPDATE prediction_tracking
               SET outcome_price = ?,
                   outcome_date = datetime('now'),
                   outcome_return_pct = ?
               WHERE id = ?""",
            (outcome_price, outcome_return_pct, pred_id),
        )
        conn.commit()
    conn.close()

# ── Dividend tracking ─────────────────────────────────────────────────────────

def log_dividend(ticker: str, amount_per_share: float, shares_held: float, ex_date: str = None) -> float:
    """Log a dividend payment and return total amount received."""
    total = amount_per_share * shares_held
    conn = _get_connection()
    conn.execute(
        "INSERT INTO dividends (ticker, amount_per_share, shares_held, total_amount, ex_date, recorded_at) VALUES (?,?,?,?,?,?)",
        (ticker, amount_per_share, shares_held, total, ex_date, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return total


def get_dividends(limit: int = 50) -> list:
    """Return recent dividend records."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT ticker, amount_per_share, shares_held, total_amount, ex_date, recorded_at FROM dividends ORDER BY recorded_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [{"ticker": r[0], "amount_per_share": r[1], "shares_held": r[2],
             "total_amount": r[3], "ex_date": r[4], "recorded_at": r[5]} for r in rows]


def get_total_dividends_received() -> float:
    """Return total dividends received across all time."""
    conn = _get_connection()
    result = conn.execute("SELECT COALESCE(SUM(total_amount), 0) FROM dividends").fetchone()
    conn.close()
    return result[0]


# ── Universe quality score cache ──────────────────────────────────────────────

def save_universe_scores(scores: list[dict]) -> None:
    """Persist quality scores for the full ticker universe. Overwrites existing rows."""
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    with conn:
        for s in scores:
            conn.execute("""
                INSERT INTO universe_scores
                    (ticker, universe, name, sector, industry,
                     quality_score, revenue_growth, profit_margin,
                     roe, debt_to_equity, scored_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                    universe=excluded.universe,
                    name=excluded.name,
                    sector=excluded.sector,
                    industry=excluded.industry,
                    quality_score=excluded.quality_score,
                    revenue_growth=excluded.revenue_growth,
                    profit_margin=excluded.profit_margin,
                    roe=excluded.roe,
                    debt_to_equity=excluded.debt_to_equity,
                    scored_at=excluded.scored_at
            """, (
                s["ticker"], s["universe"], s.get("name"), s.get("sector"), s.get("industry"),
                s["quality_score"], s.get("revenue_growth"), s.get("profit_margin"),
                s.get("roe"), s.get("debt_to_equity"), now,
            ))
    conn.close()


def get_universe_scores(top_n: int = 150) -> list[dict]:
    """Return top_n tickers ranked by quality_score, with cache metadata."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT ticker, universe, name, sector, industry,
               quality_score, revenue_growth, profit_margin, roe, debt_to_equity, scored_at
        FROM universe_scores
        ORDER BY quality_score DESC
        LIMIT ?
    """, (top_n,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_universe_scores_meta() -> dict:
    """Return count and age of the screener cache (JSON file written by screen_stocks)."""
    import json as _json
    cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "screener_cache.json")
    try:
        with open(cache_file, "r") as _f:
            _cache = _json.load(_f)
        _date = _cache.get("date")
        _count = len(_cache.get("results") or [])
        if _count > 0:
            try:
                _days = (datetime.utcnow().date() - datetime.fromisoformat(_date).date()).days
            except Exception:
                _days = None
            return {"count": _count, "oldest": _date, "newest": _date, "days_since_refresh": _days}
    except Exception:
        pass

    # Fallback: legacy universe_scores DB table
    conn = _get_connection()
    row = conn.execute("""
        SELECT COUNT(*) as count, MIN(scored_at) as oldest, MAX(scored_at) as newest
        FROM universe_scores
    """).fetchone()
    conn.close()
    if not row or row["count"] == 0:
        return {"count": 0, "oldest": None, "newest": None, "days_since_refresh": None}
    newest = row["newest"]
    try:
        delta = datetime.utcnow() - datetime.fromisoformat(newest)
        days = delta.days
    except Exception:
        days = None
    return {"count": row["count"], "oldest": row["oldest"], "newest": newest, "days_since_refresh": days}


# ── Watchlist history ─────────────────────────────────────────────────────────

def log_watchlist_event(ticker: str, event_type: str, price: float = None,
                        target_price: float = None, notes: str = None) -> None:
    """Log a watchlist lifecycle event (TRIGGERED, APPROACHING, BOUGHT, REMOVED)."""
    pct = round((price - target_price) / target_price * 100, 1) if (price and target_price) else None
    conn = _get_connection()
    with conn:
        conn.execute(
            """INSERT INTO watchlist_history (ticker, event_type, price, target_price,
               pct_vs_target, notes, recorded_at) VALUES (?,?,?,?,?,?,?)""",
            (ticker.upper(), event_type, price, target_price, pct, notes,
             datetime.utcnow().isoformat()),
        )
    conn.close()


def get_watchlist_history(ticker: str = None, limit: int = 50) -> list[dict]:
    """Return watchlist lifecycle events, optionally filtered by ticker."""
    conn = _get_connection()
    if ticker:
        rows = conn.execute(
            """SELECT ticker, event_type, price, target_price, pct_vs_target, notes, recorded_at
               FROM watchlist_history WHERE ticker=? ORDER BY recorded_at DESC LIMIT ?""",
            (ticker.upper(), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT ticker, event_type, price, target_price, pct_vs_target, notes, recorded_at
               FROM watchlist_history ORDER BY recorded_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stale_watchlist_entries(min_age_days: int = 60) -> list:
    """Return watchlist entries that haven't been re-evaluated in min_age_days days."""
    import datetime as _dt
    cutoff = (datetime.utcnow() - _dt.timedelta(days=min_age_days)).isoformat()
    conn = _get_connection()
    rows = conn.execute(
        "SELECT ticker, company_name, reason, target_entry_price, added_at FROM watchlist WHERE added_at < ? ORDER BY added_at ASC",
        (cutoff,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        try:
            added = datetime.fromisoformat(r["added_at"])
            age_days = (datetime.utcnow() - added).days
        except Exception:
            age_days = min_age_days
        result.append({
            "ticker": r["ticker"],
            "company_name": r["company_name"],
            "reason": r["reason"],
            "target_entry_price": r["target_entry_price"],
            "added_at": r["added_at"],
            "age_days": age_days,
            "staleness": "very_stale" if age_days > 180 else "stale",
            "action_needed": "Re-research: thesis may have changed or price target may have been hit",
        })
    return result


# ── Research cache ────────────────────────────────────────────────────────────

def save_research_cache(ticker: str, report: dict, price: float = None) -> None:
    """Cache a completed research report for a ticker."""
    import json as _json
    conn = _get_connection()
    with conn:
        conn.execute("""
            INSERT INTO research_cache
                (ticker, report_json, recommendation, conviction_score, price_at_research, researched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                report_json=excluded.report_json,
                recommendation=excluded.recommendation,
                conviction_score=excluded.conviction_score,
                price_at_research=excluded.price_at_research,
                researched_at=excluded.researched_at
        """, (
            ticker.upper(),
            _json.dumps(report),
            report.get("recommendation"),
            report.get("conviction_score"),
            price,
            datetime.utcnow().isoformat(),
        ))
    conn.close()


def get_research_cache(ticker: str) -> Optional[dict]:
    """Return a cached research report for ticker, or None if not found."""
    import json as _json
    conn = _get_connection()
    row = conn.execute(
        """SELECT report_json, recommendation, conviction_score,
                  price_at_research, researched_at
           FROM research_cache WHERE ticker = ?""",
        (ticker.upper(),),
    ).fetchone()
    conn.close()
    if not row:
        return None
    try:
        report = _json.loads(row["report_json"])
    except Exception:
        return None
    report["_cached"] = True
    report["_researched_at"] = row["researched_at"]
    report["_price_at_research"] = row["price_at_research"]
    return report


def is_research_cache_valid(
    ticker: str,
    current_price: float = None,
    max_age_days: int = 30,
    max_price_move_pct: float = 10.0,
) -> tuple[bool, str]:
    """
    Check whether the cached research report for ticker is still usable.
    Returns (is_valid: bool, reason: str).
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT price_at_research, researched_at FROM research_cache WHERE ticker = ?",
        (ticker.upper(),),
    ).fetchone()
    conn.close()

    if not row:
        return False, "no cache"

    try:
        age = datetime.utcnow() - datetime.fromisoformat(row["researched_at"])
        age_days = age.days
    except Exception:
        return False, "invalid timestamp"

    if age_days > max_age_days:
        return False, f"cache is {age_days} days old (max {max_age_days})"

    if current_price and row["price_at_research"]:
        move_pct = abs(current_price - row["price_at_research"]) / row["price_at_research"] * 100
        if move_pct > max_price_move_pct:
            return False, f"price moved {move_pct:.1f}% since research (max {max_price_move_pct}%)"

    return True, f"cache valid ({age_days}d old)"


# ── Regime history ────────────────────────────────────────────────────────────

def save_regime(regime: str, previous_regime: str = None, indicators: dict = None) -> None:
    """Persist the current detected regime to history."""
    conn = _get_connection()
    ind = indicators or {}
    conn.execute(
        """INSERT INTO regime_history
           (regime, previous_regime, vix, gdp_growth, core_cpi, ten_yr_yield, yield_inverted, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            regime,
            previous_regime,
            ind.get("vix"),
            ind.get("gdp_growth"),
            ind.get("core_cpi"),
            ind.get("ten_yr_yield"),
            1 if ind.get("yield_inverted") else 0,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_last_regime() -> Optional[dict]:
    """Return the most recently saved regime record, or None."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT regime, previous_regime, detected_at FROM regime_history ORDER BY detected_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return {"regime": row[0], "previous_regime": row[1], "detected_at": row[2]}
    return None


def get_regime_history(limit: int = 10) -> list[dict]:
    """Return recent regime history records."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT regime, previous_regime, detected_at FROM regime_history ORDER BY detected_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [{"regime": r[0], "previous_regime": r[1], "detected_at": r[2]} for r in rows]


# ── SPY benchmark tracking ────────────────────────────────────────────────────

def save_benchmark_snapshot(portfolio_value: float, spy_price: float) -> None:
    """Save a daily portfolio vs SPY snapshot. One row per day — upserts on same date."""
    conn = _get_connection()
    today = datetime.utcnow().date().isoformat()

    # Purge any legacy rows stored with SPY ETF prices before writing new data.
    # SPY ETF was ~$400-800; ^GSPC has been above 2 000 since 2017.
    with conn:
        conn.execute("DELETE FROM benchmark_snapshots WHERE spy_price < 2000")
        conn.execute(
            "UPDATE portfolio_snapshots SET benchmark_price = NULL "
            "WHERE benchmark_price IS NOT NULL AND benchmark_price < 2000"
        )

    existing = conn.execute(
        "SELECT id FROM benchmark_snapshots WHERE recorded_at LIKE ?",
        (f"{today}%",),
    ).fetchone()
    with conn:
        if existing:
            conn.execute(
                """UPDATE benchmark_snapshots
                   SET portfolio_value=?, spy_price=?, recorded_at=?
                   WHERE id=?""",
                (portfolio_value, spy_price, datetime.utcnow().isoformat(), existing[0]),
            )
        else:
            conn.execute(
                "INSERT INTO benchmark_snapshots (portfolio_value, spy_price, spy_shares_equivalent, recorded_at) VALUES (?, ?, ?, ?)",
                (portfolio_value, spy_price, portfolio_value / spy_price if spy_price > 0 else 0,
                 datetime.utcnow().isoformat()),
            )
    conn.close()


def get_benchmark_snapshots(limit: int = 500) -> list[dict]:
    """Return benchmark_snapshots rows oldest-first for charting.
    Only rows on/after the first trade date are included.
    Legacy SPY-ETF-priced rows (spy_price < 2000) are excluded.
    If the table is sparse (e.g. after a wipe), backfills daily ^GSPC closes
    from yfinance so charts always have a complete S&P 500 history."""
    first_trade_date = get_first_trade_date()
    if not first_trade_date:
        return []
    conn = _get_connection()

    def _fetch_rows():
        return conn.execute(
            """SELECT recorded_at AS ts, portfolio_value, spy_price AS benchmark_price
               FROM benchmark_snapshots
               WHERE recorded_at >= ? AND spy_price >= 2000
               ORDER BY recorded_at ASC LIMIT ?""",
            (first_trade_date, limit),
        ).fetchall()

    rows = _fetch_rows()

    # Backfill when we have fewer rows than trading days since the first trade.
    # This happens after a DB wipe — we only have today's snapshot, leaving the
    # S&P 500 chart line empty.  We insert one row per market day using yfinance
    # closes; portfolio_value=0 is intentional (chart only needs spy_price).
    trade_date_obj = datetime.strptime(first_trade_date[:10], "%Y-%m-%d")
    days_since_trade = (datetime.utcnow() - trade_date_obj).days
    # Expect roughly 5/7 of calendar days to be trading days; use len < days/2 as trigger.
    if len(rows) < max(2, days_since_trade // 2):
        try:
            import yfinance as yf
            end_str = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
            hist = yf.Ticker("^GSPC").history(start=first_trade_date[:10], end=end_str)
            if not hist.empty:
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_convert(None)
                existing_dates = {r["ts"][:10] for r in [dict(r) for r in rows]}
                inserts = []
                for idx, hrow in hist.iterrows():
                    ds = idx.date().isoformat()
                    price = float(hrow["Close"])
                    if ds not in existing_dates and price >= 2000:
                        inserts.append((0.0, price, 0.0, ds + "T16:00:00"))
                if inserts:
                    conn.executemany(
                        "INSERT OR IGNORE INTO benchmark_snapshots "
                        "(portfolio_value, spy_price, spy_shares_equivalent, recorded_at) "
                        "VALUES (?, ?, ?, ?)",
                        inserts,
                    )
                    conn.commit()
                    rows = _fetch_rows()
        except Exception:
            pass  # silently fall back to whatever rows exist

    conn.close()
    return [dict(r) for r in rows]


def get_benchmark_comparison() -> dict:
    """Compare current portfolio value vs what S&P 500 would be worth since first trade."""
    first_trade_date = get_first_trade_date()
    if not first_trade_date:
        return {"available": False, "message": "No trades made yet — performance tracking vs S&P 500 begins after the first trade"}

    conn = _get_connection()

    # Purge legacy rows stored with SPY ETF prices.
    # SPY ETF trades ~$400-800; ^GSPC has been above 2 000 since 2017.
    conn.execute("DELETE FROM benchmark_snapshots WHERE spy_price < 2000")
    conn.execute("UPDATE portfolio_snapshots SET benchmark_price = NULL WHERE benchmark_price IS NOT NULL AND benchmark_price < 2000")
    conn.commit()

    # Get the latest benchmark snapshot for current values.
    latest = conn.execute(
        "SELECT portfolio_value, spy_price, recorded_at FROM benchmark_snapshots WHERE recorded_at >= ? ORDER BY recorded_at DESC LIMIT 1",
        (first_trade_date,),
    ).fetchone()

    # Get the oldest portfolio snapshot for the starting portfolio value.
    first_portfolio = conn.execute(
        "SELECT portfolio_value, ts FROM portfolio_snapshots WHERE ts >= ? ORDER BY ts ASC LIMIT 1",
        (first_trade_date,),
    ).fetchone()
    conn.close()

    if not latest:
        return {"available": False, "message": "No benchmark data yet — run the agent at least once after making a trade"}

    current_portfolio = latest[0]
    current_spy_price = latest[1]
    current_date      = latest[2][:10]

    # Always fetch the historical ^GSPC price at first_trade_date from yfinance.
    # This gives a stable, DB-independent baseline that survives any future wipe.
    start_spy_price = None
    try:
        import yfinance as yf
        trade_dt = datetime.strptime(first_trade_date[:10], "%Y-%m-%d")
        # Fetch a ±7-day window around the first trade to handle weekends/holidays.
        hist_start = (trade_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        hist_end   = (trade_dt + timedelta(days=7)).strftime("%Y-%m-%d")
        gspc = yf.Ticker("^GSPC")
        hist = gspc.history(start=hist_start, end=hist_end)
        if not hist.empty:
            hist.index = hist.index.tz_convert(None) if hist.index.tz is not None else hist.index
            # Use the close on or just after the first trade date.
            after = hist[hist.index.date >= trade_dt.date()]
            start_spy_price = float((after if not after.empty else hist).iloc[0]["Close"])
    except Exception:
        pass  # fallback below

    # Fallback: use oldest benchmark snapshot if yfinance unavailable.
    if start_spy_price is None or start_spy_price <= 0:
        conn3 = _get_connection()
        first_bench = conn3.execute(
            "SELECT spy_price FROM benchmark_snapshots WHERE recorded_at >= ? ORDER BY recorded_at ASC LIMIT 1",
            (first_trade_date,),
        ).fetchone()
        conn3.close()
        start_spy_price = first_bench[0] if first_bench else current_spy_price

    # Sanity check: >5× ratio is impossible in normal markets — still-corrupt data.
    if start_spy_price > 0 and current_spy_price > 0:
        ratio = current_spy_price / start_spy_price
        if ratio > 5 or ratio < 0.2:
            conn2 = _get_connection()
            conn2.execute("DELETE FROM benchmark_snapshots")
            conn2.execute("UPDATE portfolio_snapshots SET benchmark_price = NULL WHERE benchmark_price IS NOT NULL")
            conn2.commit()
            conn2.close()
            return {"available": False, "message": "Benchmark data inconsistency detected and cleared. Will re-establish baseline on next page load."}

    # Starting portfolio value: oldest snapshot on/after first trade date.
    if first_portfolio:
        start_portfolio = first_portfolio[0]
        start_date      = first_portfolio[1][:10]
    else:
        start_portfolio = current_portfolio
        start_date      = first_trade_date[:10]

    # Use pure % return — immune to unit changes.
    portfolio_return = (current_portfolio - start_portfolio) / start_portfolio if start_portfolio else 0
    sp500_return     = (current_spy_price - start_spy_price) / start_spy_price if start_spy_price else 0
    alpha            = portfolio_return - sp500_return

    # What the starting portfolio value would be worth if invested in S&P 500
    sp500_equivalent_value = start_portfolio * (1 + sp500_return)

    return {
        "available": True,
        "start_date": start_date,
        "current_date": current_date,
        "portfolio_return_pct":  round(portfolio_return * 100, 2),
        "spy_return_pct":        round(sp500_return * 100, 2),
        "alpha_pct":             round(alpha * 100, 2),
        "beating_market":        alpha > 0,
        "portfolio_value":       round(current_portfolio, 2),
        "spy_equivalent_value":  round(sp500_equivalent_value, 2),
        "start_portfolio_value": round(start_portfolio, 2),
    }


# ── Earnings call sentiment ───────────────────────────────────────────────────

def save_earnings_sentiment(ticker: str, quarter: str, sentiment_score: float,
                             tone_direction: str, beat_miss: str = None,
                             key_signals: list = None, raw_summary: str = None) -> None:
    """Upsert earnings call sentiment for a ticker/quarter."""
    import json as _json
    conn = _get_connection()
    with conn:
        conn.execute("""
            INSERT INTO earnings_call_sentiment
                (ticker, quarter, sentiment_score, tone_direction, beat_miss,
                 key_signals, raw_summary, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, quarter) DO UPDATE SET
                sentiment_score=excluded.sentiment_score,
                tone_direction=excluded.tone_direction,
                beat_miss=excluded.beat_miss,
                key_signals=excluded.key_signals,
                raw_summary=excluded.raw_summary,
                analyzed_at=excluded.analyzed_at
        """, (
            ticker.upper(), quarter, sentiment_score, tone_direction, beat_miss,
            _json.dumps(key_signals or []), raw_summary,
            datetime.utcnow().isoformat(),
        ))
    conn.close()


def get_earnings_tone_delta(ticker: str, quarters: int = 4) -> dict:
    """
    Return sentiment trend for last N quarters.
    Returns trend direction and delta vs oldest quarter in window.
    """
    import json as _json
    conn = _get_connection()
    rows = conn.execute("""
        SELECT quarter, sentiment_score, tone_direction, beat_miss, key_signals, analyzed_at
        FROM earnings_call_sentiment
        WHERE ticker = ?
        ORDER BY analyzed_at DESC
        LIMIT ?
    """, (ticker.upper(), quarters)).fetchall()
    conn.close()

    if not rows:
        return {
            "ticker": ticker.upper(),
            "quarters": [],
            "delta": None,
            "trend": "insufficient_data",
            "signal": "No earnings call sentiment data available for this ticker.",
        }

    q_list = []
    for r in rows:
        try:
            signals = _json.loads(r["key_signals"]) if r["key_signals"] else []
        except Exception:
            signals = []
        q_list.append({
            "quarter": r["quarter"],
            "sentiment_score": r["sentiment_score"],
            "tone_direction": r["tone_direction"],
            "beat_miss": r["beat_miss"],
            "key_signals": signals,
            "analyzed_at": r["analyzed_at"],
        })

    if len(q_list) < 2:
        return {
            "ticker": ticker.upper(),
            "quarters": q_list,
            "delta": None,
            "trend": "insufficient_data",
            "signal": "Need >=2 quarters of data to compute trend.",
        }

    delta = round(q_list[0]["sentiment_score"] - q_list[-1]["sentiment_score"], 3)
    if delta > 0.1:
        trend = "improving"
    elif delta < -0.1:
        trend = "deteriorating"
    else:
        trend = "stable"

    latest_score = q_list[0]["sentiment_score"]
    latest_tone = q_list[0]["tone_direction"]
    signal = (
        f"Management tone is {trend} over last {len(q_list)} quarters "
        f"(delta={delta:+.3f}). Latest quarter: {latest_tone} "
        f"(score={latest_score:.3f})."
    )

    return {
        "ticker": ticker.upper(),
        "quarters": q_list,
        "delta": delta,
        "trend": trend,
        "signal": signal,
    }


# ── Prediction tracking / quarterly reconciliation ────────────────────────────

def log_prediction(ticker: str, action: str, conviction_score: int = None,
                   predicted_iv: float = None, price_at_decision: float = None,
                   mos_pct: float = None) -> None:
    """Log an agent decision for later reconciliation."""
    conn = _get_connection()
    conn.execute(
        """INSERT INTO prediction_tracking
           (ticker, action, conviction_score, predicted_iv, price_at_decision,
            mos_at_decision, decision_date)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ticker, action, conviction_score, predicted_iv, price_at_decision,
         mos_pct, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_pending_reconciliations(min_age_days: int = 90) -> list:
    """Return predictions older than min_age_days that haven't been reconciled."""
    import datetime as _datetime
    conn = _get_connection()
    cutoff = (datetime.utcnow().replace(hour=0, minute=0, second=0) -
              _datetime.timedelta(days=min_age_days)).isoformat()
    rows = conn.execute(
        """SELECT id, ticker, action, conviction_score, predicted_iv,
                  price_at_decision, mos_at_decision, decision_date
           FROM prediction_tracking
           WHERE outcome_date IS NULL AND decision_date < ?
           ORDER BY decision_date ASC""",
        (cutoff,)
    ).fetchall()
    conn.close()
    return [{"id": r[0], "ticker": r[1], "action": r[2], "conviction_score": r[3],
             "predicted_iv": r[4], "price_at_decision": r[5], "mos_at_decision": r[6],
             "decision_date": r[7]} for r in rows]


def update_prediction_outcome(prediction_id: int, outcome_price: float,
                               outcome_return_pct: float, outcome_vs_spy_pct: float,
                               notes: str = None) -> None:
    """Fill in the actual outcome for a prediction."""
    conn = _get_connection()
    conn.execute(
        """UPDATE prediction_tracking SET outcome_price=?, outcome_return_pct=?,
           outcome_vs_spy_pct=?, outcome_date=?, outcome_notes=? WHERE id=?""",
        (outcome_price, outcome_return_pct, outcome_vs_spy_pct,
         datetime.utcnow().isoformat(), notes, prediction_id)
    )
    conn.commit()
    conn.close()


def get_prediction_accuracy() -> dict:
    """Summarise historical prediction accuracy."""
    conn = _get_connection()
    rows = conn.execute(
        """SELECT action, conviction_score, outcome_return_pct, outcome_vs_spy_pct
           FROM prediction_tracking WHERE outcome_date IS NOT NULL"""
    ).fetchall()
    conn.close()
    if not rows:
        return {"available": False, "message": "No reconciled predictions yet"}
    buys = [r for r in rows if r[0] == "buy"]
    avg_return = sum(r[2] for r in buys) / len(buys) if buys else 0
    avg_alpha = sum(r[3] for r in buys) / len(buys) if buys else 0
    win_rate = sum(1 for r in buys if r[2] and r[2] > 0) / len(buys) if buys else 0
    high_conv = [r for r in buys if r[1] and r[1] >= 8]
    return {
        "available": True,
        "total_predictions": len(rows),
        "buy_predictions": len(buys),
        "avg_return_pct": round(avg_return, 2),
        "avg_alpha_pct": round(avg_alpha, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "high_conviction_count": len(high_conv),
        "high_conviction_avg_return": round(sum(r[2] for r in high_conv) / len(high_conv), 2) if high_conv else None,
    }


def get_stored_thesis(ticker: str) -> dict:
    """Retrieve the stored investment thesis for a ticker from previous research sessions."""
    from agent.knowledge_base import query_kb
    ticker = ticker.upper()
    conn = _get_connection()
    thesis_row = conn.execute(
        "SELECT ticker, action, thesis, created_at FROM trade_thesis WHERE ticker=? ORDER BY created_at DESC LIMIT 1",
        (ticker,)
    ).fetchone()
    watchlist_row = conn.execute(
        "SELECT ticker, company_name, reason, target_entry_price, added_at FROM watchlist WHERE ticker=?",
        (ticker,)
    ).fetchone()
    conn.close()

    price_row = None
    if thesis_row:
        inner_conn = _get_connection()
        price_row = inner_conn.execute(
            "SELECT price_at_decision FROM prediction_tracking WHERE ticker=? ORDER BY decision_date DESC LIMIT 1",
            (ticker,)
        ).fetchone()
        inner_conn.close()

    kb = query_kb(ticker, max_results=3)
    kb_notes = kb if isinstance(kb, list) else kb.get("results", []) if isinstance(kb, dict) else []

    return {
        "ticker": ticker,
        "has_trade_thesis": thesis_row is not None,
        "thesis": thesis_row["thesis"] if thesis_row else None,
        "action": thesis_row["action"] if thesis_row else None,
        "recorded_at": thesis_row["created_at"] if thesis_row else None,
        "price_at_decision": price_row["price_at_decision"] if price_row else None,
        "watchlist_entry": dict(watchlist_row) if watchlist_row else None,
        "kb_notes": kb_notes,
    }


# ── Concentration limits ──────────────────────────────────────────────────────

def check_concentration_limits(
    ticker: str,
    sector: str,
    buy_amount: float,
    max_position_pct: float = 0.10,
    max_sector_pct: float = 0.30,
) -> dict:
    """
    Check if a proposed buy would breach concentration limits.
    Call this before executing any buy order.
    """
    from agent import market_data as _market_data

    cash = get_cash()
    holdings = get_holdings()

    total_market_value = 0.0
    enriched = []
    for h in holdings:
        try:
            quote = _market_data.get_stock_quote(h["ticker"])
            price = quote.get("price") if "error" not in quote else None
            h_sector = quote.get("sector", "") if "error" not in quote else ""
        except Exception:
            price = None
            h_sector = ""

        if price:
            mv = h["shares"] * price
        else:
            mv = h["shares"] * h["avg_cost"]

        total_market_value += mv
        enriched.append({
            "ticker": h["ticker"],
            "market_value": mv,
            "sector": h_sector,
        })

    total_value = cash + total_market_value

    if total_value <= 0:
        return {
            "allowed": True,
            "violations": [],
            "max_allowed_buy": buy_amount,
            "current_position_pct": 0,
            "current_sector_pct": 0,
            "post_buy_position_pct": 0,
            "post_buy_sector_pct": 0,
        }

    ticker_upper = ticker.upper()
    current_pos_value = sum(
        h["market_value"] for h in enriched if h["ticker"] == ticker_upper
    )

    current_sector_value = sum(
        h["market_value"]
        for h in enriched
        if h["sector"] and h["sector"].lower() == sector.lower()
    )

    new_total = total_value + buy_amount
    post_pos_pct = (current_pos_value + buy_amount) / new_total
    post_sector_pct = (current_sector_value + buy_amount) / new_total

    violations = []
    if post_pos_pct > max_position_pct:
        violations.append(
            f"Position limit breach: {ticker_upper} would be {post_pos_pct:.1%} of portfolio "
            f"(max {max_position_pct:.0%})"
        )
    if post_sector_pct > max_sector_pct:
        violations.append(
            f"Sector limit breach: {sector} would be {post_sector_pct:.1%} of portfolio "
            f"(max {max_sector_pct:.0%})"
        )

    max_by_position = max(0.0, max_position_pct * new_total - current_pos_value)
    max_by_sector = max(0.0, max_sector_pct * new_total - current_sector_value)
    max_allowed = min(max_by_position, max_by_sector, buy_amount)
    if max_allowed < buy_amount and max_allowed > 0:
        new_total2 = total_value + max_allowed
        max_allowed = min(
            max_position_pct * new_total2 - current_pos_value,
            max_sector_pct * new_total2 - current_sector_value,
            buy_amount,
        )
        max_allowed = max(0.0, max_allowed)

    return {
        "allowed": len(violations) == 0,
        "violations": violations,
        "current_position_pct": round(current_pos_value / total_value, 4),
        "current_sector_pct": round(current_sector_value / total_value, 4),
        "post_buy_position_pct": round(post_pos_pct, 4),
        "post_buy_sector_pct": round(post_sector_pct, 4),
        "max_allowed_buy": round(max_allowed, 2),
    }


# ── Position drift ────────────────────────────────────────────────────────────

def check_position_drift() -> dict:
    """Flag positions that have drifted above concentration limits through price appreciation."""
    holdings = get_holdings()
    if not holdings:
        return {"drifted_positions": [], "drifted_sectors": [], "has_drift": False}

    cash = get_cash()
    from agent import market_data as _md

    enriched = []
    total_mv = 0.0
    for h in holdings:
        try:
            quote = _md.get_stock_quote(h["ticker"])
            price = quote.get("price") if "error" not in quote else h["avg_cost"]
            sector = quote.get("sector", "Unknown")
        except Exception:
            price = h["avg_cost"]
            sector = "Unknown"
        mv = h["shares"] * (price or h["avg_cost"])
        total_mv += mv
        enriched.append({**h, "current_price": price, "market_value": mv, "sector": sector})

    total_portfolio = total_mv + cash
    if total_portfolio <= 0:
        return {"drifted_positions": [], "drifted_sectors": [], "has_drift": False}

    drifted_positions = []
    sector_mv: dict = {}
    for h in enriched:
        pct = h["market_value"] / total_portfolio * 100
        sector_mv[h["sector"]] = sector_mv.get(h["sector"], 0.0) + h["market_value"]
        if pct > 12.0:
            drifted_positions.append({
                "ticker": h["ticker"],
                "current_weight_pct": round(pct, 1),
                "target_max_pct": 10.0,
                "excess_pct": round(pct - 10.0, 1),
                "market_value": round(h["market_value"], 2),
                "recommendation": f"Consider trimming ${round((pct-10)/100*total_portfolio,0):.0f} to restore 10% max weight",
            })

    drifted_sectors = []
    for sector, mv in sector_mv.items():
        pct = mv / total_portfolio * 100
        if pct > 33.0:
            drifted_sectors.append({
                "sector": sector,
                "current_weight_pct": round(pct, 1),
                "target_max_pct": 30.0,
                "excess_pct": round(pct - 30.0, 1),
            })

    return {
        "drifted_positions": drifted_positions,
        "drifted_sectors": drifted_sectors,
        "has_drift": bool(drifted_positions or drifted_sectors),
        "total_portfolio_value": round(total_portfolio, 2),
        "checked_at": datetime.utcnow().isoformat(),
    }


# ── Session efficiency ────────────────────────────────────────────────────────

def save_session_efficiency(session_date: str, total_tool_calls: int,
                             unique_tools_used: int, stocks_researched: int = 0,
                             stocks_bought: int = 0, stocks_watchlisted: int = 0,
                             duration_seconds: int = None, notes: str = None) -> int:
    conn = _get_connection()
    with conn:
        cur = conn.execute(
            """INSERT INTO session_efficiency
               (session_date, total_tool_calls, unique_tools_used, stocks_researched,
                stocks_bought, stocks_watchlisted, duration_seconds, notes, recorded_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (session_date, total_tool_calls, unique_tools_used, stocks_researched,
             stocks_bought, stocks_watchlisted, duration_seconds, notes,
             datetime.utcnow().isoformat())
        )
    return cur.lastrowid


def get_session_efficiency_history(limit: int = 10) -> list:
    conn = _get_connection()
    rows = conn.execute(
        """SELECT session_date, total_tool_calls, unique_tools_used, stocks_researched,
                  stocks_bought, stocks_watchlisted, duration_seconds, notes, recorded_at
           FROM session_efficiency ORDER BY recorded_at DESC LIMIT ?""", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]



def get_watchlist_accuracy() -> dict:
    """
    Analyse historical watchlist trigger accuracy.
    For each TRIGGERED event, check if a subsequent BOUGHT event exists.
    Returns hit rate, avg pct vs target at trigger, and per-ticker history.
    """
    conn = _get_connection()
    rows = conn.execute(
        """SELECT ticker, event_type, price, target_price, pct_vs_target, recorded_at
           FROM watchlist_history ORDER BY ticker, recorded_at"""
    ).fetchall()
    conn.close()
    events = [dict(r) for r in rows]

    triggered = [e for e in events if e["event_type"] == "TRIGGERED"]
    bought    = {e["ticker"] for e in events if e["event_type"] == "BOUGHT"}

    converted = [t for t in triggered if t["ticker"] in bought]
    hit_rate  = round(len(converted) / len(triggered) * 100, 1) if triggered else None

    return {
        "total_triggers": len(triggered),
        "converted_to_buy": len(converted),
        "hit_rate_pct": hit_rate,
        "triggered_items": triggered,
        "note": (
            "hit_rate_pct = % of TRIGGERED events that led to a buy. "
            "Low hit rate means target prices are consistently missed or too conservative."
        ),
    }

# ── Session audit + behaviour tracking ───────────────────────────────────────

def save_session_audit(
    tickers_screened: int = 0,
    tickers_researched: int = 0,
    buys_made: int = 0,
    watchlist_added: int = 0,
    shadow_added: int = 0,
    workflow_issues_logged: int = 0,
    re_researched_watchlist: int = 0,
    deviated_from_matrix: int = 0,
    duplicate_tool_calls: int = 0,
    contradicted_prior_session: int = 0,
    audit_notes: str = "",
) -> dict:
    """Save structured self-audit metrics for the current session."""
    conn = _get_connection()
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO session_audit
           (session_date, tickers_screened, tickers_researched, buys_made,
            watchlist_added, shadow_added, workflow_issues_logged,
            re_researched_watchlist, deviated_from_matrix, duplicate_tool_calls,
            contradicted_prior_session, audit_notes, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (now[:10], tickers_screened, tickers_researched, buys_made,
         watchlist_added, shadow_added, workflow_issues_logged,
         re_researched_watchlist, deviated_from_matrix, duplicate_tool_calls,
         contradicted_prior_session, audit_notes, now),
    )
    conn.commit()
    conn.close()
    flags = []
    if re_researched_watchlist:
        flags.append(f"re_researched_watchlist={re_researched_watchlist}")
    if deviated_from_matrix:
        flags.append(f"deviated_from_matrix={deviated_from_matrix}")
    if duplicate_tool_calls:
        flags.append(f"duplicate_tool_calls={duplicate_tool_calls}")
    if contradicted_prior_session:
        flags.append(f"contradicted_prior_session={contradicted_prior_session}")
    return {
        "saved": True,
        "session_date": now[:10],
        "flags": flags if flags else ["none — clean session"],
    }


def log_workflow_issue(issue: str, suggestion: str, severity: str = "low") -> dict:
    """
    Log a workflow inefficiency noticed during the session.
    Stored in kb_entries with category='workflow_issue' for later review.
    severity: 'low' | 'medium' | 'high'
    """
    import json as _json
    conn = _get_connection()
    now = datetime.utcnow().isoformat()
    content = _json.dumps({
        "issue": issue,
        "suggestion": suggestion,
        "severity": severity,
        "session_date": now[:10],
    })
    conn.execute(
        """INSERT INTO kb_entries (category, ticker, content, source, created_at)
           VALUES ('workflow_issue', NULL, ?, 'agent_self_audit', ?)""",
        (content, now),
    )
    conn.commit()
    conn.close()
    return {"logged": True, "severity": severity, "issue": issue[:80]}


def get_behaviour_summary(n_sessions: int = 10) -> dict:
    """
    Summarise agent behaviour patterns across recent sessions.
    Surfaces trends: avg tool calls, re-research rate, deviation rate,
    most common workflow issues. Loaded at session start so the agent
    can compare its current behaviour to its own history.
    """
    import json as _json
    conn = _get_connection()

    rows = conn.execute(
        """SELECT * FROM session_audit ORDER BY recorded_at DESC LIMIT ?""",
        (n_sessions,),
    ).fetchall()
    rows = [dict(r) for r in rows]

    issues = conn.execute(
        """SELECT content FROM kb_entries
           WHERE category = 'workflow_issue'
           ORDER BY created_at DESC LIMIT 20"""
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "available": False,
            "message": "No session audit history yet — will build after first audited session.",
        }

    n = len(rows)

    def _avg(field):
        vals = [r[field] for r in rows if r[field] is not None]
        return round(sum(vals) / len(vals), 1) if vals else 0

    # Trend: is the agent improving or regressing?
    re_research_trend = [r["re_researched_watchlist"] for r in rows]
    deviation_trend   = [r["deviated_from_matrix"] for r in rows]

    # Parse workflow issues
    parsed_issues = []
    for row in issues:
        try:
            parsed_issues.append(_json.loads(row["content"]))
        except Exception:
            pass
    high_issues = [i for i in parsed_issues if i.get("severity") == "high"]
    recent_suggestions = [i["suggestion"] for i in parsed_issues[:5]]

    return {
        "available": True,
        "sessions_analysed": n,
        "averages": {
            "tickers_screened":     _avg("tickers_screened"),
            "tickers_researched":   _avg("tickers_researched"),
            "buys_per_session":     _avg("buys_made"),
            "watchlist_per_session":_avg("watchlist_added"),
            "re_researched_watchlist": _avg("re_researched_watchlist"),
            "deviated_from_matrix": _avg("deviated_from_matrix"),
            "duplicate_tool_calls": _avg("duplicate_tool_calls"),
        },
        "flags": {
            "re_research_last_3": sum(re_research_trend[:3]),
            "deviation_last_3":   sum(deviation_trend[:3]),
            "high_severity_issues": len(high_issues),
        },
        "recent_workflow_suggestions": recent_suggestions,
        "interpretation": (
            "Compare your current session behaviour against these averages. "
            "If re_researched_watchlist > 0, you are wasting research budget on known stocks. "
            "If deviated_from_matrix > 0, your buy/pass decisions were not rules-based. "
            "If duplicate_tool_calls > 0, you called the same tool twice unnecessarily."
        ),
    }
