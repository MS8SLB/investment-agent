"""
Export a CSV of the full ticker universe ranked by screener score.
Tickers that have been run through screen_stocks are tagged SCREENED with their score.
Tickers not yet screened are tagged UNSCREENED and appear at the bottom.

Also tags: HOLDING, WATCHLIST, PASSED.

Run from the project root:
    python3 scripts/export_screener_csv.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.portfolio import _get_connection, DB_PATH
from agent.market_data import get_stock_universe, get_international_universe

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "screener_status.csv")


def main():
    print("Fetching live universe (S&P 500 + international)...")
    sp500 = get_stock_universe("sp500")
    intl  = get_international_universe()

    # Build {ticker: sector} for sp500, {ticker: region} for international
    universe = {}
    for entry in sp500.get("tickers", []):
        t = entry if isinstance(entry, str) else entry.get("ticker", entry)
        sector = entry.get("sector", "") if isinstance(entry, dict) else ""
        universe[t] = {"ticker": t, "universe": "sp500", "sector": sector, "region": ""}

    for entry in intl.get("tickers", []):
        t = entry if isinstance(entry, str) else entry.get("ticker", entry)
        region = entry.get("region", "") if isinstance(entry, dict) else ""
        if t not in universe:
            universe[t] = {"ticker": t, "universe": "international", "sector": "", "region": region}

    print(f"  {len(universe)} tickers in live universe")

    # Pull screened scores from DB
    conn = _get_connection()
    screened = {
        row["ticker"]: dict(row)
        for row in conn.execute("""
            SELECT ticker, quality_score, revenue_growth, profit_margin,
                   roe, debt_to_equity, scored_at
            FROM universe_scores
        """).fetchall()
    }

    holdings = {
        row["ticker"]
        for row in conn.execute("SELECT ticker FROM holdings WHERE shares > 0").fetchall()
    }

    watchlist = {
        row["ticker"]: row["tier"]
        for row in conn.execute("SELECT ticker, tier FROM watchlist").fetchall()
    }

    shadow = {
        row["ticker"]
        for row in conn.execute("SELECT ticker FROM shadow_positions").fetchall()
    }

    conn.close()

    rows = []
    for ticker, u in universe.items():
        s = screened.get(ticker, {})
        score = s.get("quality_score", None)

        if ticker in holdings:
            tag = "HOLDING"
        elif ticker in watchlist:
            tag = f"WATCHLIST-{watchlist[ticker]}"
        elif ticker in shadow:
            tag = "PASSED"
        elif score is not None:
            tag = "SCREENED"
        else:
            tag = "UNSCREENED"

        rows.append({
            "ticker":          ticker,
            "tag":             tag,
            "screener_score":  score if score is not None else "",
            "universe":        u["universe"],
            "sector":          u["sector"],
            "region":          u["region"],
            "revenue_growth":  s.get("revenue_growth", ""),
            "profit_margin":   s.get("profit_margin", ""),
            "roe":             s.get("roe", ""),
            "debt_to_equity":  s.get("debt_to_equity", ""),
            "scored_at":       s.get("scored_at", ""),
        })

    # Sort: by score desc (UNSCREENED/empty score goes to bottom)
    rows.sort(key=lambda r: (
        0 if r["screener_score"] != "" else 1,
        -(r["screener_score"] if r["screener_score"] != "" else 0)
    ))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    counts = Counter(r["tag"].split("-")[0] for r in rows)
    print(f"Written {len(rows)} rows to: {OUTPUT}")
    for tag, count in sorted(counts.items()):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
