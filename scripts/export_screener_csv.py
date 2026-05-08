"""
Export a CSV of the full ticker universe ranked by screener score.
Scores come from screener_cache.json (written by screen_stocks).
Tickers not yet screened appear at the bottom with no score.

Tags: HOLDING / WATCHLIST-<tier> / PASSED / SCREENED / UNSCREENED

Run from the project root:
    python3 scripts/export_screener_csv.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.portfolio import _get_connection, DB_PATH
from agent.market_data import get_stock_universe, get_international_universe

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "screener_cache.json")
OUTPUT     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "screener_status.csv")


def main():
    print("Fetching live universe (S&P 500 + international)...")
    sp500_raw = get_stock_universe("sp500")
    intl_raw  = get_international_universe()

    # sp500: tickers is a list of plain strings; sector info in available_sectors only
    sp500_sector_map = {}
    for entry in sp500_raw.get("tickers", []):
        if isinstance(entry, dict):
            sp500_sector_map[entry["ticker"]] = entry.get("sector", "")

    universe = {}
    for t in sp500_raw.get("tickers", []):
        ticker = t if isinstance(t, str) else t["ticker"]
        universe[ticker] = {"universe": "sp500", "sector": sp500_sector_map.get(ticker, ""), "region": ""}
    for t in intl_raw.get("tickers", []):
        ticker = t if isinstance(t, str) else t["ticker"]
        if ticker not in universe:
            universe[ticker] = {"universe": "international", "sector": "", "region": ""}

    print(f"  {len(universe)} tickers in live universe")

    # Load screener scores from cache
    screened = {}
    cache_date = ""
    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        cache_date = cache.get("date", "")
        for r in cache.get("results", []):
            screened[r["ticker"]] = r
        print(f"  {len(screened)} tickers in screener cache (dated {cache_date})")
    except FileNotFoundError:
        print("  screener_cache.json not found — scores will be empty")

    # DB lookups
    conn = _get_connection()
    holdings = {
        r["ticker"] for r in conn.execute("SELECT ticker FROM holdings WHERE shares > 0").fetchall()
    }
    watchlist = {
        r["ticker"]: r["tier"]
        for r in conn.execute("SELECT ticker, tier FROM watchlist").fetchall()
    }
    shadow = {
        r["ticker"] for r in conn.execute("SELECT ticker FROM shadow_positions").fetchall()
    }
    conn.close()

    rows = []
    for ticker, u in universe.items():
        s = screened.get(ticker, {})
        score = s.get("score", None)

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
            "score":           round(score, 2) if score is not None else "",
            "universe":        u["universe"],
            "sector":          u["sector"],
            "peg_ratio":       s.get("peg_ratio", ""),
            "fcf_yield_pct":   s.get("fcf_yield_pct", ""),
            "revenue_growth":  s.get("revenue_growth_pct", ""),
            "profit_margin":   s.get("profit_margin_pct", ""),
            "roe":             s.get("roe_pct", ""),
            "momentum_pct":    s.get("relative_momentum_pct", ""),
            "screened_date":   s.get("cached_date", cache_date if score is not None else ""),
        })

    # Sort: scored rows highest-to-lowest, unscored at bottom
    rows.sort(key=lambda r: (0 if r["score"] != "" else 1, -(r["score"] if r["score"] != "" else 0)))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    counts = Counter(r["tag"].split("-")[0] for r in rows)
    print(f"Written {len(rows)} rows → {OUTPUT}")
    for tag in ["HOLDING", "WATCHLIST", "SCREENED", "PASSED", "UNSCREENED"]:
        if tag in counts:
            print(f"  {tag}: {counts[tag]}")


if __name__ == "__main__":
    main()
