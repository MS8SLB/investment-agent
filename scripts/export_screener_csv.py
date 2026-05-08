"""
Export a CSV of the full screener universe ranked by score,
showing the research/decision status of each ticker.

Run from the project root:
    python scripts/export_screener_csv.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.portfolio import _get_connection, DB_PATH

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "screener_status.csv")


def main():
    print(f"Reading from: {DB_PATH}")
    conn = _get_connection()

    # 1. Full screener universe (universe_scores)
    universe = {
        row["ticker"]: dict(row)
        for row in conn.execute("""
            SELECT ticker, universe, name, sector, industry,
                   quality_score, revenue_growth, profit_margin,
                   roe, debt_to_equity, scored_at
            FROM universe_scores
            ORDER BY quality_score DESC
        """).fetchall()
    }

    # 2. Research cache — tickers that have been fully researched
    researched = {
        row["ticker"]: dict(row)
        for row in conn.execute("""
            SELECT ticker, recommendation, conviction_score,
                   price_at_research, researched_at
            FROM research_cache
        """).fetchall()
    }

    # 3. Decisions logged via log_decision (prediction_tracking)
    decisions = {}
    for row in conn.execute("""
        SELECT ticker, action, conviction_score, price_at_decision, decision_date
        FROM prediction_tracking
        ORDER BY decision_date DESC
    """).fetchall():
        if row["ticker"] not in decisions:
            decisions[row["ticker"]] = dict(row)

    # 4. Current watchlist
    watchlist = {
        row["ticker"]: row["tier"]
        for row in conn.execute(
            "SELECT ticker, tier FROM watchlist"
        ).fetchall()
    }

    # 5. Current holdings
    holdings = {
        row["ticker"]
        for row in conn.execute(
            "SELECT ticker FROM holdings WHERE shares > 0"
        ).fetchall()
    }

    # 6. Shadow portfolio (passed)
    shadow = {
        row["ticker"]
        for row in conn.execute(
            "SELECT ticker FROM shadow_positions"
        ).fetchall()
    }

    conn.close()

    # Build rows: universe first, then any researched tickers not in universe
    all_tickers = dict(universe)
    for t in list(researched) + list(decisions):
        if t not in all_tickers:
            all_tickers[t] = {"ticker": t}

    rows = []
    for ticker, u in all_tickers.items():
        r = researched.get(ticker, {})
        d = decisions.get(ticker, {})

        # Determine status
        if ticker in holdings:
            status = "HOLDING"
        elif ticker in watchlist:
            status = f"WATCHLIST ({watchlist[ticker]})"
        elif ticker in shadow:
            status = "PASSED"
        elif ticker in researched:
            rec = r.get("recommendation", "").upper()
            status = f"RESEARCHED ({rec})" if rec else "RESEARCHED"
        elif ticker in decisions:
            status = f"DECIDED ({d.get('action','').upper()})"
        elif ticker in universe:
            status = "SCREENED"
        else:
            status = "UNKNOWN"

        rows.append({
            "ticker":           ticker,
            "status":           status,
            "screener_score":   u.get("quality_score", ""),
            "sector":           u.get("sector", ""),
            "universe":         u.get("universe", ""),
            "revenue_growth":   u.get("revenue_growth", ""),
            "profit_margin":    u.get("profit_margin", ""),
            "roe":              u.get("roe", ""),
            "debt_to_equity":   u.get("debt_to_equity", ""),
            "researched_at":    r.get("researched_at", ""),
            "recommendation":   r.get("recommendation", ""),
            "conviction":       r.get("conviction_score") or d.get("conviction_score", ""),
            "decision":         d.get("action", ""),
            "decision_date":    d.get("decision_date", ""),
            "price_at_decision":d.get("price_at_decision", ""),
        })

    # Sort: holdings first, then watchlist, then by screener score desc
    STATUS_ORDER = {"HOLDING": 0, "WATCHLIST": 1, "RESEARCHED": 2, "DECIDED": 3, "PASSED": 4, "SCREENED": 5, "UNKNOWN": 6}
    rows.sort(key=lambda r: (
        STATUS_ORDER.get(r["status"].split(" ")[0], 9),
        -(r["screener_score"] if r["screener_score"] != "" else 0)
    ))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to: {OUTPUT}")

    # Print a quick summary
    from collections import Counter
    counts = Counter(r["status"].split(" ")[0] for r in rows)
    for status, count in sorted(counts.items(), key=lambda x: STATUS_ORDER.get(x[0], 9)):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
