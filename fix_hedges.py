import sqlite3
import re
from datetime import datetime

HEDGE_TICKERS = {"GLD", "TIP", "GSG", "TLT", "IEF", "SHV"}
DB_PATH = "data/portfolio.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# ── 1. Find all hedge holdings ────────────────────────────────────────────────
placeholders = ",".join("?" * len(HEDGE_TICKERS))
holdings = [
    dict(r) for r in
    conn.execute(f"SELECT * FROM holdings WHERE ticker IN ({placeholders})", list(HEDGE_TICKERS)).fetchall()
]

# ── 2. Find all hedge transactions ───────────────────────────────────────────
transactions = [
    dict(r) for r in
    conn.execute(f"SELECT * FROM transactions WHERE ticker IN ({placeholders}) ORDER BY ts ASC", list(HEDGE_TICKERS)).fetchall()
]

if not holdings and not transactions:
    print("No hedge positions found in the database.")
    conn.close()
    raise SystemExit(0)

print("=== Holdings to remove ===")
for h in holdings:
    print(f"  {h['ticker']}: {h['shares']} shares @ ${h['avg_cost']:.2f}")

print("\n=== Transactions to delete ===")
total_restore = 0.0
for t in transactions:
    action = t["action"].upper()
    net = float(t["total"]) if action == "BUY" else -float(t["total"])
    total_restore += net
    print(f"  {t['ts'][:10]}  {action:4s}  {t['ticker']:4s}  {t['shares']:.4f} @ ${t['price']:.2f}  total=${t['total']:.2f}")

state = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()
current_cash = float(state["cash"])
new_cash = current_cash + total_restore

print(f"\nCash to restore: ${total_restore:,.2f}")
print(f"Cash: ${current_cash:,.2f}  ->  ${new_cash:,.2f}")

# ── 3. Check reflections ──────────────────────────────────────────────────────
reflections = conn.execute(
    "SELECT id, created_at, reflection FROM reflections ORDER BY created_at ASC"
).fetchall()

cleaned = []
for r in reflections:
    original = r["reflection"]
    text = original
    text = re.sub(
        r"(?im)^[^\n]*(GLD|TIP\b|GSG\b|TLT\b|IEF\b|SHV\b|[Hh]edge\s+[Pp]osition|[Mm]acro\s+[Hh]edge|[Ii]nflationary\s+[Rr]egime\s+hedge)[^\n]*\n?",
        "", text
    )
    text = re.sub(r"(?im)^#+?\s*New Hedge Positions?\s*\n", "", text)
    text = re.sub(r"(?im)^New Hedge Positions?[^\n]*\n?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if text != original:
        cleaned.append((r["id"], r["created_at"], original, text))

print(f"\n=== Reflections to patch: {len(cleaned)} ===")
for rid, ts, orig, new in cleaned:
    removed_lines = set(orig.splitlines()) - set(new.splitlines())
    print(f"\n  Reflection {rid} ({ts[:10]}) -- removing {len(removed_lines)} line(s):")
    for line in sorted(removed_lines):
        if line.strip():
            print(f"    - {line.strip()[:120]}")

# ── 4. Confirm and apply ──────────────────────────────────────────────────────
print("\n" + "-" * 60)
confirm = input("Apply all changes? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted -- no changes made.")
    conn.close()
    raise SystemExit(0)

with conn:
    conn.execute(f"DELETE FROM holdings WHERE ticker IN ({placeholders})", list(HEDGE_TICKERS))
    conn.execute(f"DELETE FROM transactions WHERE ticker IN ({placeholders})", list(HEDGE_TICKERS))
    conn.execute(
        "UPDATE portfolio_state SET cash = ?, updated_at = ? WHERE id = 1",
        (new_cash, datetime.utcnow().isoformat())
    )
    for rid, _, _, new_text in cleaned:
        conn.execute("UPDATE reflections SET reflection = ? WHERE id = ?", (new_text, rid))

conn.close()
print(f"\nDone.")
print(f"  Removed {len(holdings)} holding(s), {len(transactions)} transaction(s)")
print(f"  Restored ${total_restore:,.2f} to cash (new balance: ${new_cash:,.2f})")
print(f"  Patched {len(cleaned)} reflection(s)")
