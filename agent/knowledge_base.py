"""
Queryable investment knowledge base stored in SQLite.

Provides persistent, searchable notes on companies, sectors, investment
lessons, and framework details. Agents query on-demand — only the relevant
chunk is loaded, rather than baking all knowledge into every system prompt.

Usage:
    from agent.knowledge_base import query_kb, save_kb_note
    results = query_kb("Constellation Software capital allocation", max_results=3)
    save_kb_note("company", "Topicus Q4 update", "...content...", ["TOI", "VMS"])
"""

import json
import re
import sqlite3
from datetime import datetime
from typing import Optional

from agent.portfolio import DB_PATH, _get_connection


# ── Schema ────────────────────────────────────────────────────────────────────

def _init_kb() -> None:
    """Create kb_entries table and pre-seed with default knowledge if empty."""
    with _get_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS kb_entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                topic       TEXT    NOT NULL,
                title       TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                tags        TEXT    NOT NULL DEFAULT '[]',
                source      TEXT    NOT NULL DEFAULT 'manual',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_kb_topic ON kb_entries(topic)")
        count = con.execute("SELECT COUNT(*) FROM kb_entries").fetchone()[0]
        if count == 0:
            _seed(con)
        con.commit()


# ── Public API ────────────────────────────────────────────────────────────────

def query_kb(
    query: str,
    topic: Optional[str] = None,
    max_results: int = 3,
) -> list[dict]:
    """
    Search the knowledge base for entries matching the query.

    Scoring: title match = 3 pts, tag match = 2 pts, content match = 1 pt.
    Returns top max_results entries sorted by score descending.

    Args:
        query: Free-text search query (e.g. "VMS serial acquirer reinvestment")
        topic: Optional filter — one of: company, sector, framework, failure_mode,
               lesson, vms_playbook, iv_methodology
        max_results: Maximum entries to return (default 3, max 10)

    Returns:
        List of dicts with id, topic, title, content, tags, score.
    """
    _init_kb()
    max_results = min(max_results, 10)
    terms = _tokenize(query)
    if not terms:
        return []

    with _get_connection() as con:
        sql = "SELECT id, topic, title, content, tags FROM kb_entries"
        params: list = []
        if topic:
            sql += " WHERE topic = ?"
            params.append(topic)
        rows = con.execute(sql, params).fetchall()

    results = []
    for row in rows:
        score = _score(terms, row["title"], row["content"], json.loads(row["tags"] or "[]"))
        if score > 0:
            results.append({
                "id": row["id"],
                "topic": row["topic"],
                "title": row["title"],
                "content": row["content"],
                "tags": json.loads(row["tags"] or "[]"),
                "score": score,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    # Drop score from output — it's internal
    for r in results[:max_results]:
        r.pop("score", None)
    return results[:max_results]


def save_kb_note(
    topic: str,
    title: str,
    content: str,
    tags: Optional[list] = None,
    source: str = "agent",
) -> dict:
    """
    Save a new knowledge entry or update an existing one with the same title.

    Args:
        topic: Category — company | sector | framework | failure_mode |
               lesson | vms_playbook | iv_methodology
        title: Short title (also used for deduplication)
        content: The knowledge content text
        tags: List of tag strings (tickers, company names, keywords)
        source: "agent" (written by AI), "manual" (written by user), "seed"

    Returns:
        {"id": int, "action": "created"|"updated"}
    """
    _init_kb()
    tags_json = json.dumps(tags or [])
    with _get_connection() as con:
        existing = con.execute(
            "SELECT id FROM kb_entries WHERE title = ?", (title,)
        ).fetchone()
        if existing:
            con.execute(
                "UPDATE kb_entries SET content=?, tags=?, source=?, updated_at=? WHERE id=?",
                (content, tags_json, source, datetime.utcnow().isoformat(), existing["id"]),
            )
            con.commit()
            return {"id": existing["id"], "action": "updated"}
        else:
            cur = con.execute(
                "INSERT INTO kb_entries (topic, title, content, tags, source) VALUES (?,?,?,?,?)",
                (topic, title, content, tags_json, source),
            )
            con.commit()
            return {"id": cur.lastrowid, "action": "created"}


def list_kb_topics() -> dict:
    """Return entry counts grouped by topic."""
    _init_kb()
    with _get_connection() as con:
        rows = con.execute(
            "SELECT topic, COUNT(*) as count FROM kb_entries GROUP BY topic ORDER BY topic"
        ).fetchall()
    return {row["topic"]: row["count"] for row in rows}


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, min length 2, deduplicated."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    seen = set()
    result = []
    for t in tokens:
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _score(terms: list[str], title: str, content: str, tags: list) -> int:
    """Score a KB entry against the query terms."""
    title_lower = title.lower()
    content_lower = content.lower()
    tags_lower = " ".join(t.lower() for t in tags)
    score = 0
    for term in terms:
        if term in title_lower:
            score += 3
        if term in tags_lower:
            score += 2
        if term in content_lower:
            score += 1
    return score


# ── Default seed knowledge ────────────────────────────────────────────────────

def _seed(con: sqlite3.Connection) -> None:
    """Pre-populate with core investment framework knowledge."""
    for entry in _DEFAULT_ENTRIES:
        con.execute(
            "INSERT INTO kb_entries (topic, title, content, tags, source) VALUES (?,?,?,?,?)",
            (
                entry["topic"],
                entry["title"],
                entry["content"].strip(),
                json.dumps(entry.get("tags", [])),
                "seed",
            ),
        )


_DEFAULT_ENTRIES = [
    # ── VMS / Serial acquirer playbook ────────────────────────────────────────
    {
        "topic": "vms_playbook",
        "title": "Constellation Software family: FCF2S valuation model",
        "tags": ["CSU", "TOI", "LMN", "Sygnity", "Asseco", "VMS", "serial acquirer", "FCF2S"],
        "content": """
FCF2S growth = ROIC × reinvestment_rate + organic_growth_rate

Key tension: higher reinvestment rate → lower average ROIC (harder to find quality deals at scale).
The best operators defend the ROIC hurdle (20-25%) and let reinvestment rate flex.
NEVER model 90%+ reinvestment at 25%+ ROIC simultaneously — it is unrealistic.

Benchmarks:
- Topicus: 20% ROIC × 75% reinvestment + 4% organic → ~19% FCF CAGR (base case)
- Lumine:  20% ROIC × 80% reinvestment + 2% organic → ~18% FCF CAGR (carve-out drag)
- CSU at scale: ROIC pressure as deal size must increase — award PEMS discount if >10% capital in minority stakes

Terminal exit multiples (apply at year 10 FCF2S):
- Topicus/CSU quality (20+ yr track record, pan-European TAM): 22-25×
- Lumine quality (single vertical, carve-out model): 18-22×
- Emerging/turnaround VMS (Sygnity, early-stage): 14-20× depending on execution proof

Incentive structure checklist (required for buy):
1. Bonus tied to BOTH ROIC AND net revenue growth
2. ≥75% of after-tax bonus reinvested in company shares held 4+ years
3. No stock options or grants (aligns with owner-operator culture)
4. Organic growth tracked separately for organic vs. M&A segments

Sidecar sizing rule:
- Start with tracker position (1-2%) for any new sidecar opportunity
- Build only after 2-3 years of demonstrated execution
- A sidecar is a bet on management quality, not the numbers
""",
    },
    {
        "topic": "vms_playbook",
        "title": "Lumine Group: carve-out model nuances",
        "tags": ["LMN", "Lumine", "carve-out", "WideOrbit"],
        "content": """
Lumine carves out divisions from large tech conglomerates (unlike Topicus/CSU bolt-ons).

Carve-out dynamics:
- Post-acquisition organic drag is EXPECTED and NOT a thesis-breaker
  (divisions were cross-subsidised by parent; Lumine rebuilds from scratch)
- WideOrbit-style integration years may show negative organic growth
- Evaluate on post-integration margins, not near-term organic growth
- Deal flow is lumpy/episodic (6 deals in year 1, 2 in year 2 is normal)

Key metrics to track: post-integration EBITDA margin, invested capital CAGR
""",
    },
    {
        "topic": "vms_playbook",
        "title": "Sygnity: TSS turnaround sidecar thesis",
        "tags": ["Sygnity", "TSS", "Poland", "CEE", "sidecar", "turnaround"],
        "content": """
Sygnity (Warsaw: SGN) is a TSS/Constellation turnaround sidecar in Poland.

TSS acquired 70%+ stake, installed management, exited unprofitable contracts.
Gross margin expanded from 28% → 47%+ over 2.5 years (proof of operational playbook).
M&A engine being built — first acquisitions beginning.

Bull case (30%+ CAGR): M&A engine materialises, Poland/CEE VMS TAM exploited
Bear case (single-digit): Operational improvement stalls, M&A does not materialise

Poland tailwind: EU cohesion funds €76B (2021-2027) earmarked for digital transformation
of public infrastructure → direct demand for Sygnity's public sector verticals.

Sizing: tracker position (1-2%) until M&A engine demonstrates consistent deal flow.
Bull case requires 3+ acquisitions per year at 20%+ ROIC before building to full position.
""",
    },
    {
        "topic": "vms_playbook",
        "title": "Asseco Poland: operational sidecar thesis",
        "tags": ["Asseco", "Poland", "ACP", "TSS", "sidecar"],
        "content": """
Asseco Poland (Warsaw: ACP) is a TSS operational sidecar.
TSS took a minority stake to apply superior capital allocation discipline.

Profile: 130+ acquisitions, 62 countries, proven acquirer at lower risk than Sygnity.
Valuation entry: ~14× earnings with 12-13% organic growth = embedded 20%+ IRR.

Key difference from Sygnity: Asseco is already profitable and proven.
The sidecar bet is improved capital allocation discipline, not operational turnaround.
Lower risk, less upside than turnaround sidecars.

Geographic VMS opportunity — Germany/DACH:
30,000-40,000 VMS targets, aging founder cohort, highly fragmented.
Nordics/Benelux: high regulatory complexity = high switching costs = superior moat quality.
""",
    },

    # ── IV failure modes ──────────────────────────────────────────────────────
    {
        "topic": "failure_mode",
        "title": "Intrinsic value failure mode checklist (10 items)",
        "tags": ["IV", "valuation", "failure", "bear case", "mistakes"],
        "content": """
1. PEAK EARNINGS TRAP
   Margins at historical highs → normalised FCF is 30-50% lower than TTM.
   Test: is current FCF margin >5pp above the 5-year average?
   If yes, model reversion. Paying 25× "peak FCF" may be 40× normalised.

2. MOAT MISCLASSIFICATION
   The claimed moat does not hold under scrutiny.
   Test: "Has any well-funded competitor tried and failed in the last 5 years?"
   If no credible test, the moat may be market share, not structural advantage.

3. FCF INFLATION
   Aggressive working capital management, capex underspend, or deferred payments
   flatter near-term FCF. These are one-time boosts, not recurring.
   Test: compare FCF margin to free cash conversion ratio (FCF / net income).
   >110% = likely working capital harvest. Normalise.

4. MANAGEMENT NARRATIVE ACCEPTANCE
   Accepting management's "adjusted EBITDA" without verifying adjustments.
   Common inflation: stock-based compensation, "one-time" restructuring repeated annually,
   acquired intangible amortisation excluded (real cost if growth requires acquisitions).

5. TERMINAL VALUE DOMINANCE
   >60% of DCF value in terminal year → the model is not a DCF, it's a multiple guess.
   If terminal value >60% of IV, apply a 20% terminal value discount and report the range.

6. GROWTH + MARGIN SIMULTANEOUSLY
   Modelling both revenue growth acceleration AND margin expansion at the same time.
   These usually don't compound simultaneously — growth requires investment (margins fall).
   Build them as independent scenarios, not combined base case.

7. CURRENCY / ADR MISMATCH
   For ADRs or cross-listed stocks, intrinsic value must be in the functional currency
   of the business, then converted to USD. If functional currency is weakening, USD IV
   falls even if the business is healthy.

8. COMPETITIVE RESPONSE IGNORED
   High-margin businesses attract competition. If margins are >25% above sector average,
   model a 5-year margin normalisation path. Incumbents rarely sustain structurally
   excessive margins without reinforcing moat evidence.

9. REGULATORY RISK UNDERWEIGHTED
   Especially for: data/index providers (antitrust), fintech (regulatory capital),
   healthcare (reimbursement), consumer (data privacy). Model a 20% probability
   adverse scenario in the bear case, not as a footnote.

10. CONVICTION INFLATION
    Retrospective attribution — after a stock goes up, the thesis sounds more convincing.
    Test: re-read the original thesis without knowing the outcome. Would you buy today?
    If conviction has risen primarily because the price has risen, it is conviction inflation.
""",
    },

    # ── IV methodology ────────────────────────────────────────────────────────
    {
        "topic": "iv_methodology",
        "title": "FCF normalisation: 3-year average method",
        "tags": ["FCF", "normalisation", "DCF", "intrinsic value"],
        "content": """
Use 3-year average FCF as the DCF base, not the most recent year.

Why: single-year FCF is volatile (capex lumps, working capital swings, one-time items).
Method:
1. Pull annual FCF for years T-1, T-2, T-3 from yfinance annual statements
2. Average the three figures
3. Sanity check: normalised FCF should be within 3× of TTM FCF
   - If TTM is >3× higher: likely a working capital harvest year — use normalised
   - If TTM is <1/3 of normalised: likely one-time capex spike — use normalised
4. Use normalised FCF as the base for the 5-year DCF projection

Margin normalisation:
- Same 3-year average for FCF margin, gross margin
- If current FCF margin is >5pp above 3-yr average: flag and model reversion

Reporting: always disclose in valuation_inputs:
  normalized_fcf, ttm_fcf, normalization_ratio (TTM / 3yr_avg)
""",
    },
    {
        "topic": "iv_methodology",
        "title": "Margin of safety framework: when to buy",
        "tags": ["margin of safety", "IRR", "entry price"],
        "content": """
IV-based entry discipline:
- Required margin of safety: ≥20% discount to conservative (bear) IV
- Required IRR at entry price: ≥15% including dividends over 5 years
- IRR 12-15%: watchlist only (price needs to fall)
- IRR < 12%: pass unless moat quality is exceptional (wide moat + long runway)

IRR calculation includes:
- Entry price as T=0 outflow
- Annual dividends as interim cash flows (FCF × payout ratio)
- Terminal value = year-5 FCF2S × exit multiple + final dividend

Entry price discipline:
- Never anchor entry to a recent high or low — anchor to IV calculation
- The stock being "historically cheap" is irrelevant if the historical baseline was overvalued
- Relative cheapness vs. peers is not margin of safety

Conviction sizing (from conviction_score):
- Score 9-10: up to 1.0× regime base size
- Score 7-8: up to 0.75× regime base size
- Score 5-6: up to 0.5× regime base size
- Score <5: pass (do not invest)
""",
    },

    # ── Sector / framework ────────────────────────────────────────────────────
    {
        "topic": "framework",
        "title": "Moat durability test: the 10-year retest",
        "tags": ["moat", "competitive advantage", "durability", "switching costs"],
        "content": """
Before classifying a moat, apply the 10-year retest:

"In 10 years, will this competitive advantage be stronger, weaker, or gone?"

Strongest moats (likely stronger in 10 years):
- Benchmark entrenchment with derivatives liquidity flywheel (S&P 500, ICE benchmarks)
- Mission-critical VMS with deeply embedded data (cemetery software, dental practice mgmt)
- Network effects with high data moat (Visa/Mastercard two-sided)

Medium durability moats (likely same in 10 years, need monitoring):
- Brand moats without ongoing investment (can be diluted — see luxury brand dilution test)
- Cost advantage moats (technology can erode)
- Switching cost moats in software without vertical lock-in

Weak or time-limited moats (likely weaker or gone):
- First-mover advantage without follow-on barriers
- Regulatory protection without barriers to entry once regulation changes
- Proprietary data sets that commoditise as AI lowers data-collection cost

AI disruption test for moat classification:
- Switching costs + regulated workflows + niche verticals → HIGH AI resistance
- Discovery/aggregation layers without proprietary supply → LOW AI resistance (could be disrupted)
""",
    },
    {
        "topic": "framework",
        "title": "Serial acquirer capital allocation: red flags vs. green flags",
        "tags": ["capital allocation", "M&A", "acquirer", "red flags"],
        "content": """
GREEN FLAGS (disciplined capital allocators):
- Describe acquisitions in terms of standalone IRR and ROIC
- Disclose post-close ROIC tracking vs. acquisition-time underwriting
- Management bonus tied to ROIC + growth (not just growth)
- 75%+ of bonus reinvested in company shares held 4+ years
- No stock options; founder-style aligned compensation
- Organic growth positive and measured separately from M&A growth

RED FLAGS (empire builders):
- "Synergies" as primary acquisition justification (rarely verified post-close)
- "Market share" as primary goal (concentrates risk)
- "Diversification" as rationale (shareholders can diversify themselves)
- Adjusted EBITDA as compensation metric (non-standardised, gameable)
- Tenure-based RSU vesting with no performance component
- Growing acquisition pace faster than integration capacity
- Goodwill as % of total assets rising each year without ROIC improvement

Anti-dilution test (share cannibal check):
- Verify actual share count trend over 3-5 years (not just buyback announcements)
- Net buyback = gross buyback spend - annual SBC expense
- If net buyback ≈ 0, the programme is a pass-through with no net benefit
""",
    },

    # ── Lessons learned ───────────────────────────────────────────────────────
    {
        "topic": "lesson",
        "title": "Long-term portfolio doctrine: exits are never price-based",
        "tags": ["exit", "sell discipline", "long term", "fundamentals"],
        "content": """
This is a long-term, intrinsic value portfolio. Price moves are never a reason to sell.
A stock that has risen 3× is not a sell unless the business has fundamentally changed.
A stock that has fallen 50% is not a sell unless the business has fundamentally changed.

Valid exit triggers (business deterioration, not price):
1. Revenue decline: 2+ consecutive years of organic revenue decline
2. Negative FCF: FCF turns negative and shows no recovery trajectory
3. Margin compression: gross margin down >5pp in <2 years without tactical explanation
4. Leverage spike: net debt/EBITDA > 4× without clear deleveraging plan
5. Management quality deterioration: major governance failure, fraud signals, or
   management turnover removing the specific people underpinning the investment thesis
6. Moat erosion: new competitor demonstrating sustained market share gains
7. Thesis invalidation: the specific reasons you bought the stock are no longer true

Never exit because:
- The stock is up and you want to "take profits"
- The stock is down and you want to "cut losses"
- The market is falling (unless the business itself is deteriorating)
- You found something more attractive (add to portfolio instead; don't rotate on price)
""",
    },
    {
        "topic": "lesson",
        "title": "Bimodal outcome stocks: position sizing discipline",
        "tags": ["bimodal", "speculative", "sizing", "options thinking"],
        "content": """
When two informed analysts reach polar opposite conclusions on a stock
("essentially worthless" vs. "2-3× in 3 years"), standard DCF produces false precision.

Discipline for bimodal outcome stocks:
1. Build probability-weighted scenario models for both poles
2. Assign scenario probabilities (be honest — if uncertain, assign 50/50)
3. Compute probability-weighted expected value
4. Size at 1-2% max (never 5%+ concentration)
5. Require 50%+ margin of safety — bear case is equity impairment, not mild underperformance

If you cannot assign meaningful probabilities with reasonable confidence:
→ Recommend "pass (circle of competence)" — outcome range is too wide to underwrite.

Flag as "bimodal outcome — speculative position sizing required" in full_thesis.
""",
    },
]
