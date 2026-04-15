"""
remove_gen_thesis_error.py
--------------------------
One-time migration: remove GEN from live portfolio (thesis error correction),
revert original cost to available cash, add to shadow portfolio, log reflection.

Run from the project root:
    python3 scripts/remove_gen_thesis_error.py

What this script does:
  1. Reads GEN holding (shares, avg_cost)
  2. Calculates original_cost = shares × avg_cost
  3. DELETEs GEN from holdings
  4. ADDs original_cost back to portfolio cash  (cost-basis reversion, NOT market sale)
  5. INSERTs a THESIS_ERROR_CORRECTION transaction
  6. INSERTs GEN into shadow_positions via add_to_shadow_portfolio()
  7. INSERTs a reflection post-mortem via save_reflection()
  8. Prints a summary of all changes
"""

import os
import sys
from datetime import datetime, timezone

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.portfolio import DB_PATH, _get_connection, add_to_shadow_portfolio, save_reflection

# Current approximate market price — used only for the shadow position record
GEN_MARKET_PRICE_AT_REMOVAL = 18.89  # USD, ~April 2026

NOW = datetime.now(timezone.utc).isoformat()


def run():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Make sure the project data directory exists and you have run the agent at least once.")
        return

    conn = _get_connection()
    cur = conn.cursor()

    # ── Read GEN holding ──────────────────────────────────────────────────────
    cur.execute("SELECT ticker, shares, avg_cost, first_bought FROM holdings WHERE ticker = 'GEN'")
    row = cur.fetchone()

    if row is None:
        print("GEN is not in the holdings table — nothing to do.")
        conn.close()
        return

    ticker = row["ticker"]
    shares = row["shares"]
    avg_cost = row["avg_cost"]
    first_bought = row["first_bought"]
    original_cost = round(shares * avg_cost, 2)

    print(f"\n{'='*60}")
    print("GEN THESIS ERROR CORRECTION")
    print(f"{'='*60}")
    print(f"  Ticker       : {ticker}")
    print(f"  Shares       : {shares}")
    print(f"  Avg cost     : ${avg_cost:.4f}")
    print(f"  Original cost: ${original_cost:,.2f}  (cash to be returned)")
    print(f"  First bought : {first_bought}")
    print(f"  Market price : ${GEN_MARKET_PRICE_AT_REMOVAL:.2f}  (for shadow record only)")
    print(f"{'='*60}\n")

    # ── Read current cash ─────────────────────────────────────────────────────
    cur.execute("SELECT cash FROM portfolio_state WHERE id = 1")
    state = cur.fetchone()
    if state is None:
        print("ERROR: portfolio_state row not found.")
        conn.close()
        return

    cash_before = state["cash"]
    cash_after = round(cash_before + original_cost, 2)

    # ── Atomic: DELETE holding + UPDATE cash + INSERT transaction ─────────────
    # `with conn:` commits on success and rolls back on any exception,
    # preventing a partial-update if any SQL statement fails.
    transaction_notes = (
        "THESIS ERROR CORRECTION — GEN Digital Safety. "
        "Original purchase assumed switching-cost moat in consumer cybersecurity. "
        "Morningstar assigns No Economic Moat. Retention in mid-80% range (insufficient for "
        "switching-cost moat; enterprise threshold is >95%). "
        "Net Debt/EBITDA ~3.6x in a no-moat consumer business. "
        "Cash reverted at avg_cost (not market price) per thesis-error protocol. "
        "Position moved to shadow portfolio. New rules added to agent framework to prevent recurrence."
    )
    with conn:
        conn.execute("DELETE FROM holdings WHERE ticker = 'GEN'")
        conn.execute(
            "UPDATE portfolio_state SET cash = ?, updated_at = ? WHERE id = 1",
            (cash_after, NOW),
        )
        conn.execute(
            "INSERT INTO transactions (ts, action, ticker, shares, price, total, realized_pnl, notes)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (NOW, "THESIS_ERROR_CORRECTION", ticker, shares, avg_cost, original_cost, 0.0, transaction_notes),
        )

    conn.close()
    print("[1/3] Removed GEN from holdings, reverted cash, logged transaction.")

    # ── Shadow position ───────────────────────────────────────────────────────
    shadow_reason = (
        "thesis_error: no economic moat. "
        "Consumer cybersecurity (antivirus, VPN, identity protection) — not enterprise. "
        "Gross retention ~mid-80s (moat threshold is >=90% for 3 consecutive years). "
        "Switching cost moat assumed at purchase was never confirmed by data. "
        "Morningstar: No Moat, Very High Uncertainty. "
        "Net Debt/EBITDA ~3.6x without FCF durability to support leverage."
    )
    shadow_notes = (
        f"Removed from live portfolio {NOW[:10]} as thesis error. "
        f"Original cost basis: ${original_cost:,.2f} ({shares} shares @ ${avg_cost:.4f}). "
        f"Market price at removal: ~${GEN_MARKET_PRICE_AT_REMOVAL:.2f}. "
        "Keep in shadow to monitor: if gross retention reaches >=90% AND ARPU grows above "
        "inflation for 3+ years, thesis can be re-evaluated. "
        "Do NOT re-initiate without confirmed moat evidence."
    )
    add_to_shadow_portfolio(ticker, GEN_MARKET_PRICE_AT_REMOVAL, shadow_reason, shadow_notes)
    print("[2/3] Added GEN to shadow_positions.")

    # ── Reflection post-mortem ────────────────────────────────────────────────
    # Portfolio value: cash_after + remaining holdings at cost (best available without prices)
    pf_conn = _get_connection()
    holdings_val = pf_conn.execute("SELECT SUM(shares * avg_cost) FROM holdings").fetchone()[0] or 0.0
    pf_conn.close()
    approx_portfolio_value = round(cash_after + holdings_val, 2)

    reflection = f"""THESIS ERROR POST-MORTEM — GEN Digital Safety

DATE: {NOW[:10]}

WHAT HAPPENED:
GEN was purchased with an assumption that consumer cybersecurity products (Norton antivirus,
LifeLock identity protection, consumer VPN) create switching-cost moats. This assumption was
wrong. The position has been removed and cash reverted at cost basis.

WHY THE ORIGINAL THESIS WAS WRONG:
1. Moat type confusion: Consumer software switching costs are driven by inconvenience, not
   mission-critical workflow dependency (which is the enterprise standard). "Hard to cancel"
   != switching cost moat.
2. Retention data not checked: Gross retention in the mid-80s is insufficient for a
   switching-cost moat. Enterprise standard is >95%. Mid-80s = normal consumer churn.
3. Category error: "Cybersecurity benefits from AI" -- true for enterprise endpoint/SIEM/cloud.
   Does NOT apply to consumer antivirus/VPN. The two segments are structurally different.
4. Leverage risk ignored: Net Debt/EBITDA ~3.6x is dangerous in a no-moat consumer business
   with no pricing power. A 20% revenue decline scenario was not stress-tested.
5. Bundling misread as moat: Norton 360 bundles antivirus + VPN + identity = retention tactic,
   not evidence of a durable competitive advantage.

MORNINGSTAR ASSESSMENT (confirmed thesis error):
- No Economic Moat (explicitly assigned)
- Very High Uncertainty
- Quant report used -- no analyst-driven DCF

RULES ADDED TO AGENT FRAMEWORK (prevent recurrence):
1. Consumer software moat test: requires gross retention >=90% AND ARPU growing above
   inflation for 3+ consecutive years. Brand != moat. Market share != moat.
2. Cybersecurity segmentation: AI-driven demand expansion applies to enterprise (endpoint,
   SIEM, cloud security). Does not automatically extend to consumer antivirus/VPN.
3. Leverage in no-moat businesses: No moat + Net Debt/EBITDA >3x = mandatory 20% revenue
   decline stress test before maintaining or initiating.
4. Thesis error protocol: If moat assumed at purchase is contradicted by data, recalculate
   IV at no-moat assumptions. Do not add to position. Document error explicitly.

WHAT TO WATCH IN SHADOW:
GEN could be re-evaluated if: gross retention reaches >=90% sustained for 3 years AND
ARPU grows above inflation for 3 years AND leverage drops below 2x Net Debt/EBITDA.
This is a high bar intentionally -- consumer software moats are rare."""

    save_reflection(reflection, portfolio_value=approx_portfolio_value, session_type="thesis_error")
    print("[3/3] Logged reflection post-mortem.")

    print(f"\n{'='*60}")
    print("COMPLETE — Summary of changes")
    print(f"{'='*60}")
    print(f"  Holdings    : GEN removed")
    print(f"  Cash        : ${cash_before:,.2f}  ->  ${cash_after:,.2f}")
    print("  Transaction : THESIS_ERROR_CORRECTION logged (realized_pnl = 0)")
    print("  Shadow      : GEN added to shadow_positions")
    print("  Reflection  : Post-mortem written")
    print(f"{'='*60}\n")
    print("Run the Streamlit dashboard to verify the changes look correct.")


if __name__ == "__main__":
    run()
