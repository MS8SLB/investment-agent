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

    # ── CoStar Group (CSGP) ───────────────────────────────────────────────────
    {
        "topic": "company",
        "title": "CoStar Group: 40-year moat in commercial real estate data",
        "tags": ["CSGP", "CoStar", "SaaS", "moat", "network effects", "data", "CRE", "commercial real estate"],
        "content": """
BUSINESS MODEL:
CoStar digitises commercial real estate through 40 years of on-the-ground data collection.
Core product: CoStar Suite (CRE analytics) - ~270k subscribers, $1B+ revenue annually.
59 consecutive quarters of double-digit revenue growth (through crises, COVID, rate hikes).

MOAT: Hybrid physical + digital approach
1. Physical boots-on-the-ground: Thousands of field researchers visit properties, verify data, photograph.
2. Accumulated proprietary database: 37+ years of manually-earned data cannot be replicated quickly.
3. Network effects flywheel: More brokers use CoStar → more listings → investors must also use it.
4. Switching costs: CoStar Suite is like "Bloomberg Terminal for CRE" — essentially required.
5. Pricing power: Sticky product where churn comes from customers going out of business, not competition.
6. AI-resistant: On-the-ground human verification cannot be fully automated.

Competitor landscape: MSCI, Real Capital Analytics, Reonomy exist but lack breadth/depth.
CoStar estimates it has more data on US commercial buildings than the US government.

PORTFOLIO OF BUSINESSES (diversification as resilience):
- CoStar Suite: CRE analytics & data ($1.02B revenue 2024)
- Apartments.com: Rental listings ($1.07B revenue 2024) — key success story
- STR: Hospitality data provider
- LoopNet: CRE marketplace (11M monthly visitors)
- Visual Lease & Real Estate Manager: Lease portfolio SaaS
- Ten-X: Commercial property auction platform
- Land.com: Land sales portal

Geographic TAM: Currently penetrating only 3-4% of global addressable market (primarily North America).
International expansion underway: Domain Holdings (Australia acquisition, $1.9B, 2025).
Matterport acquisition ($1.6B): 3D virtual tours of office/industrial spaces — deepens moat.
""",
    },
    {
        "topic": "framework",
        "title": "CoStar marketplace playbook: Apartments.com → Homes.com",
        "tags": ["marketplace", "playbook", "acquisition", "content", "network effects", "CSGP", "Apartments.com", "Homes.com"],
        "content": """
THREE-PHASE MARKETPLACE EXECUTION (proven at Apartments.com, being replicated at Homes.com):

PHASE 1: CONTENT DOMINANCE
- Build rich, proprietary content on every property (photos, floor plans, amenities, school ratings, walkability).
- Send field researchers to document properties before virtual tours were commonplace.
- Goal: Win organic search traffic through superior content (Google ranks you first for "apartments in Phoenix").
- This traffic is free and compounding — no paid marketing needed initially.

PHASE 2: BRAND ADVERTISING & CATEGORY OWNERSHIP
- National TV campaigns (e.g., Jeff Goldblum for Apartments.com).
- Goal: Own the category mindshare so when anyone thinks "renting," they think Apartments.com first.

PHASE 3: MONETIZATION VIA ENHANCED LISTINGS
- Landlords/property managers pay for enhanced listings (better placement, more photos, rental applications).
- Model similar to Google search ads — more you pay, more visibility you get.
- Requires dedicated sales force with deep market relationships (boots-on-ground critical).
- Once you have supply (landlords paying), demand (renters) naturally follows.

APARTMENTS.COM SUCCESS STORY:
- Acquisition cost: $585M (2014)
- Current revenue: $1.2B (2024)
- Timeline: ~10 years to fully build out
- IRR on acquisition: ~39%

HOMES.COM REPLICATION ATTEMPT:
CoStar is applying the same playbook to residential single-family homes (vs. rentals).
- Started: Content-first strategy (building profiles on every home in America)
- Market dynamics: Homes.com has "Your Listing, Your Lead" pitch vs. Zillow's "We sell your lead to competitors"
- Status (Feb 2026): 31k agent subscribers, ~$100M annualized revenue, 48% YoY lead growth
- Problem: CoStar has invested $5B in residential marketing with still-negative operating profits
- Challenge: Zillow has 230M+ monthly visitors and entrenched network effects (harder than Apartments market was)

KEY INSIGHT:
The fact that Apartments.com worked doesn't guarantee Homes.com will work.
Residential has an entrenched incumbent (Zillow) with stronger network effects.
But the playbook execution has proven disciplined and value-creating.
""",
    },
    {
        "topic": "framework",
        "title": "Andy Florance (CoStar CEO): Capital allocation track record & disciplined approach",
        "tags": ["capital allocation", "management", "founder-CEO", "CSGP", "Andy Florance", "buyback", "discipline"],
        "content": """
BACKGROUND:
Founder/still-CEO of CoStar (40+ years). Started company at age 24 by identifying a frustration:
commercial real estate was one of the most opaque markets in the world. Sent researchers with clipboards
and cameras to physically visit properties and build a database. Started selling data on floppy disks.

CAPITAL ALLOCATION TRACK RECORD:
IRRs on major investments: 17-39%
- This is EXCEPTIONAL and demonstrates disciplined capital allocation
- Apartments.com acquisition was a major bet; proved to be 39% IRR play
- Track record gives confidence that management deserves benefit of doubt on large bets

CURRENT FINANCIAL POSITION:
- Cash: $4B
- Debt: $0
- Share buyback authorized: $700M (largest in company history)
- 2026 net investment reduction: $300M vs. prior spending (trimming Homes.com spend)

DISCIPLINE SIGNALS:
1. No debt despite aggressive investment — financial fortress
2. Willingness to cut Homes.com spending by $300M when markets criticized it (shows responsiveness)
3. Massive buyback program despite Homes.com losses — signal of confidence + capital return
4. Founder-driven with decades of tenure — owner mentality

ACTIVIST RESPONSE:
Third Point hedge fund published letter (Jan 2026) criticizing the Homes.com investment.
Demanded board overhaul, less independent directors.
CoStar responded with $700M buyback + $300M spending cut.
Loeb (Third Point) subsequently exited position — suggests CoStar's pragmatic moves addressed concerns.

INVESTMENT INSIGHT:
Founder-CEOs with decades of track record and fortress balance sheets earn credibility.
Prefer to back great capital allocators taking speculative bets than mediocre operators with "safe" strategies.
""",
    },
    {
        "topic": "company",
        "title": "CoStar CSGP: recent developments (2025-2026)",
        "tags": ["CSGP", "CoStar", "Third Point", "Homes.com", "buyback", "Matterport", "Domain"],
        "content": """
RECENT DEVELOPMENTS AS OF APRIL 2026:

HOMES.COM STATUS (Feb 2026 update):
- 31,000+ agent subscribers
- ~$100M annual run-rate revenue (up from ~$0 in 2021)
- Lead volume to listing agents: +48% YoY
- Lead volume to paying member agents: +187% YoY
- Still spending $1B+ on marketing; original 2027 targets abandoned
- 2026 net residential investment being cut by $300M vs. prior pace

THIRD POINT ACTIVIST CAMPAIGN:
- Dan Loeb (Third Point) published open letter to CoStar board (Jan 2026)
- Demands: board overhaul, cut residential investment, return capital via buybacks
- Corporate response: $700M share buyback (largest in company history), $300M residential spend cut
- OUTCOME: Third Point exited its position entirely — activist campaign effectively concluded
- Interpretation: Management made tangible concessions; Loeb accepted partial win and moved on

CAPITAL RETURN:
- $700M share buyback authorized (unprecedented scale for CoStar)
- $4B cash, zero debt — one of the strongest balance sheets in software

RECENT ACQUISITIONS:
- Domain Holdings (Australia): ~$1.9B (2025). Australia #2 residential portal. International expansion.
- Matterport: ~$1.6B. Creates 3D digital twins of physical spaces.
  → Will allow CoStar subscribers to virtually tour any office/industrial building.
  → Deepens moat and addresses CRE broker workflow.

LOOPNET DETAILS:
- #1 CRE marketplace by traffic: ~11M monthly visitors
- Only 3.8% of top 1,000 US commercial properties are enhanced listing clients
- Institutional strategy (Brookfield, Blackstone etc.): Andy Florance says 2-3 more years to fully build
- When institutional clients come on: large contracts with high renewal rates
- Note: low % of trophy properties as clients reflects that prime assets don't need help selling

AI IMPACT ASSESSMENT:
- CoStar Suite labeled as "AI loser" by market — stock down 50%+ from peak
- Counter-argument: on-the-ground researchers cannot be replaced by AI
- Data collection, physical verification, field photography remain AI-resistant
- Most exposed business: Homes.com (consumer-facing, Zillow already has mindshare)
""",
    },
    {
        "topic": "failure_mode",
        "title": "CoStar Homes.com: Bear case and why disruption might fail",
        "tags": ["CSGP", "Homes.com", "bear case", "Zillow", "network effects", "capital allocation", "failure mode"],
        "content": """
BEAR CASE (Third Point perspective):

1. CAPITAL DESTRUCTION:
   - $5B invested across residential portals since 2021
   - Generated just $60M in revenue in 2024
   - IRR on capital deployed: effectively negative so far
   - Original 2027 targets abandoned without replacement timeline

2. NETWORK EFFECTS ARE ASYMMETRIC:
   - Apartments.com was a more fragmented market; Zillow dominated but not invincibly
   - Residential single-family homes: Zillow has 230M+ monthly visitors
   - Network effects in residential are stronger than in multifamily
   - By the time CoStar finishes building, Zillow has moved further ahead

3. STRUCTURAL CHALLENGE:
   - "Your Listing, Your Lead" pitch is compelling but not differentiated
   - Zillow's model still works despite recent NAR commission settlement
   - Only 27% of recent home buyers negotiated agent fees — old habits persist
   - Even if more buyers demand fee negotiation, Zillow's buy-side model survives

4. BETTER CAPITAL ALLOCATION:
   - Return capital to shareholders via buybacks and dividends
   - Refocus on core CRE business (CoStar Suite + LoopNet + STR)
   - Core business has 3-4% global penetration — massive TAM upside without Homes.com bet

BULL CASE COUNTERARGUMENT:
- "This is exactly what people said about Apartments.com, and look how it turned out"
- If CoStar can replicate that playbook, current pain is temporary
- Management's track record (17-39% IRRs) deserves deference

VALUATION INSIGHT:
Even with 60% probability on the "pivot" scenario, fair value is ~$55.
At ~$35, 30% discount to fair value with 17% IRR — skewed risk-reward.
""",
    },
    {
        "topic": "iv_methodology",
        "title": "CoStar: Two-scenario valuation with probability weighting",
        "tags": ["CSGP", "CoStar", "valuation", "DCF", "Homes.com", "IV"],
        "content": """
VALUATION SCENARIO FRAMEWORK (April 2026, Intrinsic Value Newsletter):

SCENARIO 1: PIVOT / CORE BUSINESS (60% probability)
Cut Homes.com promo spend ~80%, refocus on core CRE data + Apartments.com + LoopNet.

Revenue model (80% promo cut):
  2025: $3.25B → 2026: $3.59B → 2027: $3.78B → 2028: $4.01B → 2029: $4.26B → 2030: $4.55B

EBIT margin path:
  2025: -2.2% → 2026: 4.6% → 2027: 11.5% → 2028: 18.3% → 2029: 25.2% → 2030: 28.0%

Exit multiple analysis (2030 EV/EBIT weighted average):
  Range: 16x-30x EV/EBIT; probability-weighted avg ~24x
  Weighted Avg Present EV Per Share: $47
  Add: net cash per share (2025E): $3.10
  → Fair Value (Scenario 1): $50
  → 20% Margin of Safety Target Price: $40.12

IRR analysis:
  At $35 current price: 17% implied IRR → BUY
  At $40.12 (MoS price): 13.6% implied IRR → ENTRY ZONE
  At $50: ~9% IRR → HOLD

SCENARIO 2: HOMES.COM SCALES (40% probability)
Homes.com hits $500M-$1B revenue on largely fixed cost base → enormously value-creating.
If this scenario plays out, current price is clearly undervalued.

COMBINED PROBABILITY-WEIGHTED FAIR VALUE: ~$55 (author estimate)
Represents ~30% upside from ~$35 current price.

KEY INSIGHT:
CoStar is worth MORE if it stops Homes.com spending immediately than if Homes.com succeeds in 5 years.
The time-value of capital destruction makes the pivot scenario higher IV despite missing upside.

POSITION SIZING:
IV newsletter: 2% starter position, build to 5% if conviction increases.
Entry target: ~$40 (20% MoS to $50 fair value).
""",
    },

    # ── OTC Markets Group (OTCM) ─────────────────────────────────────────────
    {
        "topic": "company",
        "title": "OTC Markets Group (OTCM): core moat and business model",
        "tags": ["OTCM", "OTC Markets", "regulatory moat", "picks and shovels", "capital markets", "toll collector"],
        "content": """
BUSINESS MODEL:
OTC Markets Group (OTCM) operates the infrastructure for the over-the-counter (OTC) equity market in the US.
Covers 12,000+ securities — more than NYSE and Nasdaq combined — with only ~130 employees.
Founded 1913 (National Quotation Bureau), rebranded OTC Markets 2008. CEO Cromwell Coulson since 1997.

SCALE (2025):
- Market cap: ~$640M (small-cap, but outsized operational scope)
- Revenue: $125M; Operating margin: 34%; Gross margin: ~60%
- Revenue per employee: $961k (added only 28 employees as revenue doubled, 2020-2025)
- Zero debt since 2016; share count essentially flat (11.1M to 11.8M over a decade)

10-YEAR FINANCIAL TRACK RECORD:
- Revenue CAGR: 11%; Earnings CAGR: 13%; FCF CAGR: 14%
- FCF grows 3pp faster than revenue — operating leverage is expanding margins over time

COMPETITIVE MOAT (multi-layered):

1. REGULATORY MOAT (primary and strongest):
NYSE and Nasdaq are LEGALLY BARRED from listing non-SEC-registered foreign companies.
For any company that doesn't meet SEC registration requirements, OTC Markets is often THE ONLY option.
OTCM doesn't compete on price or features — regulations eliminated the competition.
This is the rarest moat type: a legally enforced monopoly on a specific market segment.

2. NETWORK EFFECTS:
12,000+ securities dataset becomes more valuable as more participants join.
More data -> more users -> more data -> more broker-dealers -> more issuers. Classic flywheel.

3. SWITCHING COSTS:
Broker-dealers: re-certification with SEC and FINRA, rebuild compliance systems, retrain staff.
Corporate issuers: lose market visibility, investor relations infrastructure, reputational standing.
Leaving OTCM signals uncertainty to the market — the exit itself hurts the company's credibility.

4. REGULATORY RELATIONSHIPS (non-replicable):
Decades of integration with FINRA and SEC. A new competitor cannot buy these relationships.
They take decades to establish and represent a true barrier to entry that money alone cannot overcome.

CONTEXT: Companies like Universal Music Group (UMGNF) and Exor (EXXRF) trade on OTC markets.
For many international companies, OTC is the only way to reach US public investors.

KEY RISK:
If SEC creates a "venture exchange" framework (like Canada's TSX Venture) or allows NYSE/Nasdaq to list
non-registered companies, OTCM's fundamental regulatory advantage evaporates. Low probability but binary.
""",
    },
    {
        "topic": "company",
        "title": "OTC Markets Group (OTCM): three revenue segments and unit economics",
        "tags": ["OTCM", "OTC Markets", "revenue segments", "unit economics", "OTC Link", "Corporate Services", "MDL"],
        "content": """
THREE REVENUE SEGMENTS:

1. OTC LINK — Trading Infrastructure (21% of revenue, most cyclical)
- Routes quotes, processes trades, reports to FINRA. Broker-dealers MUST use it — no meaningful alternative.
- Revenue: broker-dealer subscriptions, per-user fees, per-quote/message transaction fees.
- HIGHLY CYCLICAL: 2021 bull market — ECN transactions 11,500/day -> 48,000/day; revenue +87%.
  In down markets, subscription base provides revenue floor. Losses are cushioned, never catastrophic.
- RISK: Broker-dealer count falling (116 -> 77 over last decade) due to industry consolidation.

2. CORPORATE SERVICES — Listing Fees (39% of revenue, most stable)
- Flat annual fees from ~12,000 listed OTC companies. Three tiers:
  * OTCQX (highest transparency): ~$26k/yr (was $15k in 2017, +73% over 8 years)
  * OTCQB (mid-tier); Pink Limited Market (minimal disclosure)
- FLAT FEE STRUCTURE: Fees don't depend on market cap, trading volume, or share price.
  A company in free fall pays the same as one surging. This makes the segment essentially recession-proof.
- Renewal rates: OTCQX 95%, OTCQB 90%. Companies almost never leave.
- Pricing power: 73% fee increase over 8 years with <10% annual churn. Comparable to Verisign on .com domains.
- Future pricing growth: ~3-5% annual fee increases expected.

3. MARKET DATA LICENSING — MDL (40% of revenue, partially cyclical)
- Real-time quotes, order book data, compliance screening to Bloomberg, Refinitiv, broker-dealers, retail.
- Professional users: grown 35%+ over last decade.
- CYCLICAL: Retail subscribers surge in bull markets and evaporate in bear markets.
- RISK: Top customer = 9% of MDL revenue. One broker-dealer policy change in 2025 -> non-professional users -18%.

UNIT ECONOMICS (Corporate Services):
- Customer acquisition cost: ~$3,700 (inferred: $1.6M total marketing / 430 new subscribers)
- Annual fee: $26k with 3-5% annual price increases; average tenure 14-20 years
- Lifetime value per customer: $350k-$500k
- Return on acquisition spend: ~100x-150x

GROWTH CONSTRAINT:
OTCM cannot spend 10x more on marketing to acquire more issuers. The supply of listing candidates is
constrained by macroeconomic conditions — not marketing spend. IPO/listing activity is cyclical and exogenous.
""",
    },
    {
        "topic": "framework",
        "title": "OTC Markets CEO Cromwell Coulson: 29-year track record and capital allocation",
        "tags": ["OTCM", "OTC Markets", "Cromwell Coulson", "founder-CEO", "capital allocation", "management"],
        "content": """
BACKGROUND:
Cromwell Coulson has been CEO of OTC Markets since 1997 (29 years). Bought the business in 1997
and transformed it from floppy-disk data distribution to digital market infrastructure.
Has navigated the GFC, COVID, and 2022 rate hike cycle without using debt.

BUFFETT'S RULE #1 — VALUE CREATED PER DOLLAR RETAINED:
- Retained earnings (last decade): $18.2M
- Value created: $342M market cap increase + $129M in dividends = $471M total
- Value per $1 retained: $26 (Buffett's hurdle is simply >$1.00 — this is exceptional)

COMPENSATION:
- Cromwell Coulson total comp: ~$800k/year
- vs. Nasdaq CEO: $21.5M; ICE CEO (NYSE parent): $19.8M
- Running an operation that structurally competes with major exchanges at 1/25th the pay.
- Modest comp relative to scale is a green flag for owner mentality.

BONUS STRUCTURE:
Two KPIs: EPS growth and revenue growth.
EPS component prevents management from buying revenue growth at the expense of earnings quality.

CAPITAL RETURNS:
- Zero debt since 2016; share count flat (minimal dilution)
- Consistent dividends alongside reinvestment
- 11-14% CAGRs achieved without leverage — pure operational execution

RISK TO MONITOR:
29-year tenure is a double-edged sword. Green: proven multi-cycle performance, institutional knowledge.
Amber: potential for stale thinking. Evidence so far clearly supports the green interpretation.
""",
    },
    {
        "topic": "iv_methodology",
        "title": "OTC Markets (OTCM): valuation model and risk-adjusted return",
        "tags": ["OTCM", "OTC Markets", "valuation", "IRR", "DCF", "exit multiple", "picks and shovels"],
        "content": """
VALUATION (May 2026, Intrinsic Value Newsletter — Kyle Grieve):

CURRENT PRICE: ~$54/share | P/E: ~21x trailing earnings

5-YEAR DCF MODEL:
- Net income (2025): $31M; Annual growth rate: 12%
- Terminal net income (2030): ~$55M
- Exit multiple: 25x (historical average for this business)
- Enterprise value at 25x: $1.38B; Shares (2030): 12.2M (modest SBC dilution)
- Implied fair value (2030): ~$113/share

IMPLIED IRR FROM ~$54: ~20% annually including dividends

GROWTH DRIVER DECOMPOSITION (12% through-cycle earnings CAGR):
- Organic growth (new listings, subscriptions): ~8-9%
- Pricing power / margin expansion: ~2-3% incremental
- Cyclical bull market tailwinds: temporary lifts, not structural

SECTOR COMPARISON:
- OTCM: ~15x operating profits; operating margin 34% (higher than Alphabet)
- ICE (NYSE parent): ~23x operating profits
- OTCM covers more securities than NYSE + Nasdaq combined; lower multiple than the sector
- At 15x vs. 23x, OTCM appears significantly undervalued relative to exchange sector peers

OPERATING LEVERAGE PROOF:
Revenue doubled ($68M -> $125M, 2020-2025) while adding only 28 employees.
FCF CAGR (14%) exceeds revenue CAGR (11%) by 3pp. Capex: <$1.5M/year. Fully capital-light.

KEY RISKS:
1. REGULATORY (binary, primary): SEC creates "venture exchange" or allows NYSE/Nasdaq to list
   non-registered companies. Low probability but would impair all three segments simultaneously.
2. BROKER-DEALER CONSOLIDATION: OTC Link subscribers 116 -> 77 over last decade.
3. RETAIL CYCLICALITY: Non-professional MDL users evaporate in bear markets. Not controllable.
4. CUSTOMER CONCENTRATION: Top MDL customer = 9% of segment revenue.

POSITION SIZING (IV newsletter): 2% tracking position.
Will evaluate regulatory risk more deeply before building to full 5% position.
""",
    },
]
