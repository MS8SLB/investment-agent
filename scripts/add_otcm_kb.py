"""Add OTC Markets Group (OTCM) knowledge base entries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.knowledge_base import save_kb_note

save_kb_note(
    topic="company",
    title="OTC Markets Group (OTCM): core moat and business model",
    tags=["OTCM", "OTC Markets", "regulatory moat", "picks and shovels", "capital markets", "toll collector"],
    source="manual",
    content="""BUSINESS MODEL:
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
)

save_kb_note(
    topic="company",
    title="OTC Markets Group (OTCM): three revenue segments and unit economics",
    tags=["OTCM", "OTC Markets", "revenue segments", "unit economics", "OTC Link", "Corporate Services", "MDL"],
    source="manual",
    content="""THREE REVENUE SEGMENTS:

1. OTC LINK — Trading Infrastructure (21% of revenue, most cyclical)
- Routes quotes, processes trades, reports to FINRA. Broker-dealers MUST use it — no meaningful alternative.
- Revenue: broker-dealer subscriptions, per-user fees, per-quote/message transaction fees.
- HIGHLY CYCLICAL: 2021 bull market — ECN transactions 11,500/day -> 48,000/day; revenue +87%.
  In down markets, subscription base provides revenue floor. Losses are cushioned, never catastrophic.
- RISK: Broker-dealer count falling (116 -> 77 over last decade) due to industry consolidation.

2. CORPORATE SERVICES — Listing Fees (39% of revenue, most stable)
- Flat annual fees from ~12,000 listed OTC companies. Three tiers:
  * OTCQX (highest transparency): ~$26k/yr (was $15k in 2017, +73% over 8 years)
  * OTCQB (mid-tier)
  * Pink Limited Market (minimal disclosure)
- FLAT FEE STRUCTURE: Fees don't depend on market cap, trading volume, or share price.
  A company in free fall pays the same as one surging. This makes the segment essentially recession-proof.
- Renewal rates: OTCQX 95%, OTCQB 90%. Companies almost never leave — exit means losing all market visibility.
- Pricing power: 73% fee increase over 8 years with <10% annual churn. Comparable to Verisign on .com domains.
- Future pricing growth: ~3-5% annual fee increases expected.

3. MARKET DATA LICENSING — MDL (40% of revenue, partially cyclical)
- Real-time quotes, order book data, compliance screening data to Bloomberg, Refinitiv, broker-dealers, retail.
- Professional users: grown 35%+ over last decade. Sold directly and via distributors.
- CYCLICAL: Retail subscribers surge in bull markets and evaporate in bear markets.
- RISK: Top customer = 9% of MDL revenue. One broker-dealer policy change in 2025 -> non-professional users -18%.
- Acquisitions: EDGAR Online (limited value-add) and Blue Sky Data (accretive).

UNIT ECONOMICS (Corporate Services):
- Customer acquisition cost: ~$3,700 (inferred: $1.6M total marketing / 430 new subscribers)
- Annual fee: $26k with 3-5% annual price increases
- Churn: 5-7% annually -> implied average tenure: 14-20 years
- Lifetime value per customer: $350k-$500k
- Return on acquisition spend: ~100x-150x

GROWTH CONSTRAINT:
OTCM cannot spend 10x more on marketing to acquire more issuers. The supply of listing candidates is
constrained by macroeconomic and market conditions — not marketing spend. IPO/listing activity is cyclical
and exogenous; no amount of sales effort can manufacture demand in a cold market.
""",
)

save_kb_note(
    topic="framework",
    title="OTC Markets CEO Cromwell Coulson: 29-year track record and capital allocation",
    tags=["OTCM", "OTC Markets", "Cromwell Coulson", "founder-CEO", "capital allocation", "management"],
    source="manual",
    content="""BACKGROUND:
Cromwell Coulson has been CEO of OTC Markets since 1997 (29 years). Bought the business in 1997
with a group of investors and transformed it from floppy-disk data distribution to digital market
infrastructure. Has navigated the GFC, COVID, and 2022 rate hike cycle without using debt.

BUFFETT'S RULE #1 — VALUE CREATED PER DOLLAR RETAINED:
- Retained earnings (last decade): $18.2M
- Value created: $342M market cap increase + $129M in dividends = $471M total
- Value per $1 retained: $26 (Buffett's hurdle is simply >$1.00; this is exceptional)

COMPENSATION:
- Cromwell Coulson total comp: ~$800k/year
- Nasdaq CEO: $21.5M; ICE CEO (NYSE parent): $19.8M
- Running an operation that structurally competes with major exchanges at 1/25th the pay.
- Modest comp relative to scale is a green flag for owner mentality and long-term alignment.

BONUS STRUCTURE:
Two KPIs: EPS growth and revenue growth.
EPS component prevents management from buying revenue growth at the expense of earnings quality.
Correct incentive structure for a capital-light, high-margin business.

SHAREHOLDER RETURNS UNDER HIS TENURE:
- Zero debt since 2016
- Share count essentially flat (minimal dilution despite no buyback announcements)
- Consistent dividends alongside business reinvestment
- 11-14% CAGRs achieved without leverage — pure operational execution

RISK TO MONITOR:
29-year tenure is a double-edged sword. Green: proven multi-cycle performance, institutional knowledge.
Amber: potential for stale thinking or resistance to change. Evidence so far clearly supports green.
""",
)

save_kb_note(
    topic="iv_methodology",
    title="OTC Markets (OTCM): valuation model and risk-adjusted return",
    tags=["OTCM", "OTC Markets", "valuation", "IRR", "DCF", "exit multiple", "picks and shovels"],
    source="manual",
    content="""VALUATION (May 2026, Intrinsic Value Newsletter — Kyle Grieve):

CURRENT PRICE: ~$54/share | P/E: ~21x trailing earnings

5-YEAR DCF MODEL:
- Net income (2025): $31M
- Annual growth rate: 12%
- Terminal net income (2030): ~$55M
- Exit multiple: 25x (historical average for this business)
- Enterprise value at 25x: $1.38B
- Shares (2030): 12.2M (modest SBC dilution)
- Implied fair value (2030): ~$113/share

IMPLIED IRR FROM ~$54: ~20% annually including dividends

GROWTH DRIVER DECOMPOSITION (12% through-cycle earnings CAGR):
- Organic growth (new listings, subscriptions): ~8-9%
- Pricing power / margin expansion: ~2-3% incremental
- Cyclical bull market tailwinds: temporary lifts, not structural

SECTOR COMPARISON:
- OTCM: ~15x operating profits; operating margin 34%
- ICE (NYSE parent): ~23x operating profits
- OTCM covers more securities than NYSE + Nasdaq combined, higher margins than Alphabet (~30%)
- At 15x vs. 23x, OTCM appears significantly undervalued relative to the exchange sector

OPERATING LEVERAGE PROOF:
Revenue doubled ($68M -> $125M, 2020-2025) while adding only 28 employees.
FCF CAGR (14%) exceeds revenue CAGR (11%) by 3pp — compounding margin improvement.
Capex: <$1.5M/year. D&A: ~$2.6M. Essentially zero maintenance capex. Fully capital-light.

RISKS:
1. REGULATORY (binary, primary): SEC creates "venture exchange" or allows NYSE/Nasdaq to list non-registered
   companies. Low probability but would impair all three segments simultaneously.
2. BROKER-DEALER CONSOLIDATION: OTC Link subscribers 116 -> 77 over last decade. Structural or cyclical?
3. RETAIL CYCLICALITY: Non-professional MDL users evaporate in bear markets. Not controllable.
4. CUSTOMER CONCENTRATION: Top MDL customer = 9% of segment revenue.

POSITION SIZING (IV newsletter):
2% tracking position. Will evaluate regulatory risk more deeply before building to full position.
Potential to build to 5% if regulatory risk confirmed low and thesis strengthens.
""",
)

print("Done — 4 OTCM entries saved.")
