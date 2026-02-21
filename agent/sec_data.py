"""
SEC EDGAR data fetching for enhanced investment intelligence.

Provides five capabilities:
  1. Earnings call transcript retrieval (8-K filings)
  2. Annual/quarterly report analysis (10-K / 10-Q)
  3. Material corporate event monitoring (8-K item watch)
  4. Competitive positioning via peer-group screening
  5. Superinvestor 13F holding checks

All data is sourced from the SEC EDGAR public API (no key required)
and yfinance (already a project dependency).
"""

import re
import time
import datetime
from functools import lru_cache
from typing import Optional

import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# EDGAR constants
# ---------------------------------------------------------------------------
_EDGAR_API      = "https://data.sec.gov"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_HEADERS = {"User-Agent": "investment-agent-research sec-data@example.com"}

# Curated superinvestors: display name → zero-padded CIK
_SUPERINVESTORS = {
    "Berkshire Hathaway (Buffett)":   "0001067983",
    "Pershing Square (Ackman)":       "0001336528",
    "Appaloosa (Tepper)":             "0001100329",
    "Viking Global (Halvorsen)":      "0001103804",
    "Duquesne (Druckenmiller)":       "0001536411",
    "Third Point (Loeb)":             "0001040273",
    "Greenlight Capital (Einhorn)":   "0001079114",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ticker_cik_map() -> dict:
    """Load SEC ticker → zero-padded CIK mapping (cached for the process lifetime)."""
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_HEADERS, timeout=15,
    )
    resp.raise_for_status()
    return {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in resp.json().values()
    }


def _get_cik(ticker: str) -> Optional[str]:
    try:
        return _ticker_cik_map().get(ticker.upper())
    except Exception:
        return None


def _get_recent_filings(ticker: str, form_type: str, count: int = 5) -> list:
    """Return metadata for the most recent *count* filings of *form_type*."""
    cik = _get_cik(ticker)
    if not cik:
        return []
    try:
        url  = f"{_EDGAR_API}/submissions/CIK{cik}.json"
        data = requests.get(url, headers=_HEADERS, timeout=15).json()
        recent = data.get("filings", {}).get("recent", {})
        forms  = recent.get("form", [])
        dates  = recent.get("filingDate", [])
        accs   = recent.get("accessionNumber", [])
        docs   = recent.get("primaryDocument", [])
        results = []
        for form, date, acc, doc in zip(forms, dates, accs, docs):
            if form_type in form:
                acc_nd = acc.replace("-", "")
                int_cik = int(cik)
                results.append({
                    "form":      form,
                    "date":      date,
                    "accession": acc,
                    "index_url": f"https://www.sec.gov/Archives/edgar/data/{int_cik}/{acc_nd}/{acc}-index.htm",
                    "doc_url":   f"{_EDGAR_ARCHIVES}/{int_cik}/{acc_nd}/{doc}",
                })
                if len(results) >= count:
                    break
        return results
    except Exception:
        return []


def _fetch_text(url: str, max_chars: int = 8000) -> str:
    """Fetch *url*, strip HTML/XML tags, collapse whitespace, truncate."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"[fetch error: {e}]"


def _extract_section(full_text: str,
                     start_pats: list[str],
                     end_pats:   list[str],
                     max_chars:  int = 2500) -> str:
    """Extract a named section from a large block of filing text."""
    upper = full_text.upper()
    start = -1
    for pat in start_pats:
        idx = upper.find(pat.upper())
        if idx != -1:
            start = idx
            break
    if start == -1:
        return ""
    end = len(full_text)
    for pat in end_pats:
        idx = upper.find(pat.upper(), start + 200)
        if idx != -1:
            end = min(end, idx)
    return full_text[start:end][:max_chars].strip()


# ---------------------------------------------------------------------------
# 1. Earnings call transcript
# ---------------------------------------------------------------------------

def get_earnings_transcript(ticker: str) -> dict:
    """
    Retrieve the most recent earnings call transcript filed with the SEC.

    Earnings transcripts are typically filed as 8-K exhibits under
    Items 7.01 or 9.01.  Falls back to the primary 8-K document if
    a dedicated transcript exhibit cannot be located.
    """
    filings = _get_recent_filings(ticker, "8-K", count=10)
    if not filings:
        return {"error": f"No 8-K filings found for {ticker} on EDGAR"}

    cik     = _get_cik(ticker)
    int_cik = int(cik)

    for filing in filings:
        try:
            acc_nd     = filing["accession"].replace("-", "")
            index_url  = filing["index_url"]
            index_resp = requests.get(index_url, headers=_HEADERS, timeout=12)
            index_html = index_resp.text

            is_transcript_filing = any(
                kw in index_html.lower()
                for kw in ["transcript", "earnings call", "conference call", "prepared remarks"]
            )
            if not is_transcript_filing:
                continue

            # Try to find an exhibit document that looks like a transcript
            exhibit_matches = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.(htm|txt))"[^>]*>([^<]*(?:transcript|earnings|conference|remarks)[^<]*)',
                index_html, re.IGNORECASE,
            )
            if exhibit_matches:
                doc_url = "https://www.sec.gov" + exhibit_matches[0][0]
            else:
                doc_url = filing["doc_url"]

            text = _fetch_text(doc_url, max_chars=6000)
            return {
                "ticker":             ticker.upper(),
                "filing_date":        filing["date"],
                "source":             "SEC EDGAR 8-K",
                "transcript_excerpt": text,
                "note": (
                    "Earnings call transcript from SEC EDGAR. "
                    "Analyse: management confidence vs hedging language, "
                    "changes in forward guidance wording, analyst Q&A tension points, "
                    "and any topics management avoided."
                ),
            }
        except Exception:
            continue

    # Fallback: return the primary document of the most recent 8-K
    filing = filings[0]
    text   = _fetch_text(filing["doc_url"], max_chars=4000)
    return {
        "ticker":             ticker.upper(),
        "filing_date":        filing["date"],
        "source":             "SEC EDGAR 8-K (primary doc — transcript not isolated)",
        "transcript_excerpt": text,
        "note": (
            "Dedicated earnings call transcript was not found. "
            "This is the primary 8-K document — review for earnings press-release "
            "language and any management commentary."
        ),
    }


# ---------------------------------------------------------------------------
# 2. 10-K / 10-Q deep reading
# ---------------------------------------------------------------------------

def get_sec_filing_analysis(ticker: str, form_type: str = "10-K") -> dict:
    """
    Fetch the latest annual (10-K) or quarterly (10-Q) report and return
    three high-signal sections: Business overview, Risk Factors, and MD&A.

    These sections surface qualitative signals that pure-number screeners miss:
    - New risk factors = management flagging emerging threats
    - MD&A hedging language = caution ahead of guidance cut
    - Business moat language = management confidence in competitive position
    """
    filings = _get_recent_filings(ticker, form_type, count=1)
    if not filings:
        return {"error": f"No {form_type} filings found for {ticker} on EDGAR"}

    filing   = filings[0]
    raw_text = _fetch_text(filing["doc_url"], max_chars=60000)

    sections: dict = {}

    business = _extract_section(
        raw_text,
        start_pats=["ITEM 1.", "ITEM 1 BUSINESS", "OUR BUSINESS"],
        end_pats=["ITEM 1A", "RISK FACTORS"],
        max_chars=2000,
    )
    if business:
        sections["business_overview"] = business

    risks = _extract_section(
        raw_text,
        start_pats=["ITEM 1A", "RISK FACTORS"],
        end_pats=["ITEM 1B", "ITEM 2", "UNRESOLVED STAFF COMMENTS"],
        max_chars=3000,
    )
    if risks:
        sections["risk_factors"] = risks

    mda = _extract_section(
        raw_text,
        start_pats=[
            "ITEM 7.", "MANAGEMENT'S DISCUSSION",
            "MANAGEMENT S DISCUSSION", "RESULTS OF OPERATIONS",
        ],
        end_pats=["ITEM 7A", "ITEM 8", "QUANTITATIVE AND QUALITATIVE"],
        max_chars=3000,
    )
    if mda:
        sections["management_discussion_analysis"] = mda

    if not sections:
        sections["filing_excerpt"] = raw_text[:5000]

    return {
        "ticker":      ticker.upper(),
        "form_type":   form_type,
        "filing_date": filing["date"],
        "source":      "SEC EDGAR",
        "sections":    sections,
        "filing_url":  filing["doc_url"],
        "note": (
            f"Key sections from the latest {form_type}. "
            "Flag: new risk factors not in prior year, MD&A language shifts "
            "(e.g. 'we believe' → 'we cannot guarantee'), and moat durability "
            "signals in the business description."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Material event monitoring (8-K watch)
# ---------------------------------------------------------------------------

# 8-K item codes and their plain-English meaning
_8K_ITEMS = {
    "1.01": "Material definitive agreement",
    "1.02": "Terminated material agreement",
    "1.03": "Bankruptcy / receivership",
    "2.01": "Asset acquisition or disposal",
    "2.02": "Results of operations (earnings)",
    "2.05": "Departure of key employees / cost reduction plan",
    "2.06": "Asset impairment",
    "3.01": "Securities deregistration",
    "4.01": "Auditor change",
    "4.02": "Non-reliance on prior financials (restatement risk)",
    "5.01": "Change in controlling shareholder",
    "5.02": "Director / officer appointment or departure",
    "5.03": "Amendments to charter or by-laws",
    "7.01": "Regulation FD disclosure (earnings call / guidance)",
    "8.01": "Other material events",
    "9.01": "Financial statements and exhibits",
}


def get_material_events(ticker: str, days: int = 90) -> dict:
    """
    Return recent SEC 8-K material event filings for a ticker.

    8-Ks must be filed within 4 business days of a material event —
    making this a real-time signal source that surfaces CEO departures,
    M&A activity, restatements, and regulatory actions well before
    they appear in quarterly earnings calls.
    """
    filings = _get_recent_filings(ticker, "8-K", count=15)
    if not filings:
        return {"ticker": ticker, "events": [], "note": "No 8-K filings found on EDGAR."}

    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    recent = [f for f in filings if f["date"] >= cutoff]

    events = []
    for filing in recent[:7]:   # cap HTTP calls
        event = {"date": filing["date"], "form": filing["form"], "url": filing["index_url"]}
        try:
            resp  = requests.get(filing["index_url"], headers=_HEADERS, timeout=10)
            items = re.findall(
                r"Item\s+(\d+\.\d+)[^<\n]{0,5}[–\-]?\s*([^<\n]{5,80})",
                resp.text, re.IGNORECASE,
            )
            if items:
                event["items"] = [
                    {
                        "item":        i[0],
                        "description": _8K_ITEMS.get(i[0], i[1].strip()[:60]),
                    }
                    for i in items[:5]
                ]
        except Exception:
            pass
        events.append(event)
        time.sleep(0.15)

    # Flag high-severity events
    high_severity_items = {"1.03", "2.06", "4.01", "4.02", "5.01"}
    flags = []
    for ev in events:
        for item in ev.get("items", []):
            if item["item"] in high_severity_items:
                flags.append(f"{ev['date']} — Item {item['item']}: {item['description']}")

    return {
        "ticker":              ticker.upper(),
        "window_days":         days,
        "event_count":         len(events),
        "high_severity_flags": flags,
        "events":              events,
        "note": (
            "SEC 8-K filings capture material events within 4 business days. "
            "High-severity flags (auditor changes, restatements, impairments, "
            "bankruptcy) should trigger immediate thesis review. "
            "Item 5.02 (exec departure) is especially important — "
            "a CFO exit is a stronger warning signal than a CEO exit."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Competitive positioning
# ---------------------------------------------------------------------------

def get_competitor_analysis(ticker: str) -> dict:
    """
    Identify the stock's closest S&P 500 peers by GICS industry and return
    a side-by-side fundamental comparison.

    This reveals whether the stock's valuation premium or discount is
    justified relative to its actual competitors — not just the broad market.
    """
    try:
        info     = yf.Ticker(ticker).info
        sector   = info.get("sector", "")
        industry = info.get("industry", "")
        name     = info.get("shortName", ticker)
    except Exception:
        return {"error": f"Could not fetch sector info for {ticker}"}

    if not sector:
        return {"error": f"No sector data available for {ticker}"}

    # Import here to avoid circular imports
    from agent import market_data

    sp500     = market_data.get_stock_universe("sp500")
    sp500_tickers = sp500.get("tickers", []) if isinstance(sp500, dict) else []
    if not sp500_tickers:
        return {"error": "Could not load S&P 500 universe for peer comparison"}

    # Screen the whole sector (max 100 for performance)
    sector_peers = [t for t in sp500_tickers if t != ticker.upper()]
    screened     = market_data.screen_stocks(sector_peers[:100], top_n=50)

    # Pull the subject ticker's fundamentals
    subject_data = None
    all_results  = market_data.screen_stocks([ticker.upper()], top_n=1)
    if all_results and isinstance(all_results, list):
        subject_data = all_results[0]

    peer_rows = screened if isinstance(screened, list) else []

    return {
        "ticker":            ticker.upper(),
        "company_name":      name,
        "sector":            sector,
        "industry":          industry,
        "subject_metrics":   subject_data,
        "peer_count":        len(peer_rows),
        "top_peers":         peer_rows[:10],
        "note": (
            f"Peer comparison within the '{sector}' sector (S&P 500 universe). "
            "Key questions: Is the subject's PEG above or below the peer median? "
            "Does its FCF yield justify its P/E premium? "
            "Is its revenue growth rate above the sector average? "
            "Stocks that rank in the top quartile on both quality AND valuation "
            "vs peers have the strongest long-term track record."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Superinvestor 13F positions
# ---------------------------------------------------------------------------

def get_superinvestor_positions(ticker: str) -> dict:
    """
    Check whether prominent value investors hold this stock based on
    their most recent 13F-HR filing with the SEC.

    13F filings are submitted quarterly and have a 45-day lag, so
    they reflect positioning as of the previous quarter-end.
    Convergence among multiple superinvestors is a meaningful
    confirmation signal; a superinvestor *exiting* is a warning.
    """
    # Resolve ticker → company name (used to match 13F issuer names)
    try:
        info         = yf.Ticker(ticker).info
        company_name = (info.get("shortName") or info.get("longName") or ticker).upper()
        # Simplify: take first meaningful word(s) for fuzzy matching
        name_tokens  = [w for w in re.split(r"[\s,\.]+", company_name) if len(w) > 2]
        search_name  = name_tokens[0] if name_tokens else ticker.upper()
    except Exception:
        search_name = ticker.upper()

    holders     = []
    not_found   = []

    for investor, cik in _SUPERINVESTORS.items():
        try:
            filings = _get_recent_filings_by_cik(cik, "13F-HR", count=1)
            if not filings:
                not_found.append(investor)
                continue

            filing  = filings[0]
            int_cik = int(cik)
            acc_nd  = filing["accession"].replace("-", "")

            # Fetch the filing index to find the infotable document
            index_resp = requests.get(filing["index_url"], headers=_HEADERS, timeout=12)
            # Prefer infotable XML; fall back to any XML
            xml_matches = re.findall(
                r'href="(/Archives/edgar/data/[^"]+(?:infotable|information_table)[^"]*\.xml)"',
                index_resp.text, re.IGNORECASE,
            )
            if not xml_matches:
                xml_matches = re.findall(
                    r'href="(/Archives/edgar/data/[^"]+\.xml)"',
                    index_resp.text, re.IGNORECASE,
                )

            if not xml_matches:
                not_found.append(investor)
                time.sleep(0.2)
                continue

            xml_url  = "https://www.sec.gov" + xml_matches[0]
            xml_resp = requests.get(xml_url, headers=_HEADERS, timeout=15)
            xml_text = xml_resp.text

            # Search for the company name in the infotable
            if search_name in xml_text.upper():
                # Extract value (in thousands) from the surrounding context
                pattern = (
                    rf"<nameOfIssuer>[^<]*{re.escape(search_name)}[^<]*</nameOfIssuer>"
                    r".*?<value>(\d+)</value>"
                    r".*?<sshPrnamt>(\d+)</sshPrnamt>"
                )
                match = re.search(pattern, xml_text, re.IGNORECASE | re.DOTALL)
                value_k = int(match.group(1)) if match else None
                shares  = int(match.group(2)) if match else None

                holders.append({
                    "investor":       investor,
                    "filing_date":    filing["date"],
                    "market_value_k": value_k,   # USD thousands
                    "shares":         shares,
                    "filing_url":     filing["index_url"],
                })

            time.sleep(0.2)   # respect EDGAR rate limits

        except Exception:
            not_found.append(investor)
            continue

    conviction = (
        "STRONG — multiple smart-money investors are aligned on this position."
        if len(holders) >= 3 else
        "MODERATE — at least one prominent investor holds a position."
        if len(holders) >= 1 else
        "None of the tracked superinvestors appear to hold this stock."
    )

    return {
        "ticker":               ticker.upper(),
        "company_search_term":  search_name,
        "superinvestor_holders": holders,
        "holder_count":         len(holders),
        "conviction_signal":    conviction,
        "investors_checked":    list(_SUPERINVESTORS.keys()),
        "data_lag_note":        "13F filings have up to a 45-day lag after quarter-end.",
        "note": (
            "Superinvestor 13F holdings are a confirmation signal, not a buy trigger. "
            "Check the filing date — a position may have been reduced or exited since. "
            "The *absence* of smart money is not a negative signal on its own."
        ),
    }


def _get_recent_filings_by_cik(cik: str, form_type: str, count: int = 1) -> list:
    """Same as _get_recent_filings but takes a CIK directly (for superinvestors)."""
    try:
        cik_padded = cik.zfill(10)
        url        = f"{_EDGAR_API}/submissions/CIK{cik_padded}.json"
        data       = requests.get(url, headers=_HEADERS, timeout=15).json()
        recent     = data.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        dates      = recent.get("filingDate", [])
        accs       = recent.get("accessionNumber", [])
        docs       = recent.get("primaryDocument", [])
        results    = []
        int_cik    = int(cik)
        for form, date, acc, doc in zip(forms, dates, accs, docs):
            if form_type in form:
                acc_nd = acc.replace("-", "")
                results.append({
                    "form":      form,
                    "date":      date,
                    "accession": acc,
                    "index_url": f"https://www.sec.gov/Archives/edgar/data/{int_cik}/{acc_nd}/{acc}-index.htm",
                    "doc_url":   f"{_EDGAR_ARCHIVES}/{int_cik}/{acc_nd}/{doc}",
                })
                if len(results) >= count:
                    break
        return results
    except Exception:
        return []
