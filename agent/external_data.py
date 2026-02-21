"""
External data sources for enhanced investment intelligence.

Four tools — all free, no paid subscriptions required:

  get_economic_indicators()
      Key US macroeconomic data from the St. Louis Federal Reserve FRED API.
      Requires a free API key: https://fredapi.stlouisfed.org/docs/api/api_key.html
      Add FRED_API_KEY=<key> to .env.

  get_google_trends(ticker, keywords)
      Google search-interest trends for a company's brand or products.
      Uses pytrends — no API key required.

  get_retail_sentiment(ticker)
      Bullish/bearish ratio and recent posts from StockTwits and Reddit
      (r/investing, r/wallstreetbets, r/stocks). No API key required.

  get_rss_news(ticker)
      Recent headlines from RSS feeds: Yahoo Finance, MarketWatch,
      and Seeking Alpha — broader coverage than yfinance alone.
      Uses stdlib xml.etree.ElementTree — no extra dependencies required.
"""

import os
import re
import datetime
import requests
import yfinance as yf
from typing import Optional

_HEADERS = {"User-Agent": "investment-agent-research data@example.com"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FRED MACROECONOMIC INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# (series_id, plain-English label, unit hint)
_FRED_SERIES = [
    ("A191RL1Q225SBEA", "real_gdp_growth_pct",    "Real GDP growth rate (% QoQ annualised)"),
    ("CPIAUCSL",        "cpi_yoy_pct",             "CPI all items (YoY %)"),
    ("CPILFESL",        "core_cpi_yoy_pct",        "Core CPI ex food & energy (YoY %)"),
    ("UNRATE",          "unemployment_rate",        "Unemployment rate (%)"),
    ("ICSA",            "initial_jobless_claims_k", "Initial jobless claims (thousands, weekly)"),
    ("RSAFS",           "retail_sales_yoy_pct",    "Retail & food-service sales (YoY %)"),
    ("UMCSENT",         "consumer_sentiment",       "U. of Michigan consumer sentiment index"),
    ("INDPRO",          "industrial_production_yoy","Industrial production index (YoY %)"),
    ("HOUST",           "housing_starts_k",         "Housing starts (thousands, annualised)"),
    ("FEDFUNDS",        "fed_funds_rate",           "Federal funds effective rate (%)"),
]

# Series where we report YoY change instead of the raw level
_YOY_CALC = {"CPIAUCSL", "CPILFESL", "RSAFS", "INDPRO"}


def _fred_obs(series_id: str, api_key: str, limit: int = 14) -> list:
    """Fetch the last *limit* FRED observations for *series_id*."""
    try:
        resp = requests.get(
            _FRED_BASE,
            params={
                "series_id":  series_id,
                "api_key":    api_key,
                "file_type":  "json",
                "limit":      limit,
                "sort_order": "desc",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("observations", [])
    except Exception:
        return []


def _yoy_change(obs: list) -> Optional[float]:
    """YoY % change from a newest-first list of FRED observations."""
    valid = [o for o in obs if o.get("value") not in (".", "", None)]
    if len(valid) < 13:
        return None
    try:
        cur  = float(valid[0]["value"])
        prev = float(valid[12]["value"])
        return round((cur - prev) / abs(prev) * 100, 2) if prev else None
    except (ValueError, ZeroDivisionError):
        return None


def get_economic_indicators() -> dict:
    """
    Fetch key US macroeconomic indicators from the Federal Reserve FRED API.

    Returns real-economy signals that LEAD the market by 1-2 quarters —
    complementing get_macro_environment() which covers market-price signals
    (yields, VIX).  Requires FRED_API_KEY in .env (free registration).
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return {
            "error": "FRED_API_KEY not configured",
            "setup": (
                "1. Register for a free key at https://fredapi.stlouisfed.org/docs/api/api_key.html\n"
                "2. Add  FRED_API_KEY=<your_key>  to your .env file and restart the app."
            ),
        }

    indicators = {}
    for series_id, label, description in _FRED_SERIES:
        obs   = _fred_obs(series_id, api_key)
        valid = [o for o in obs if o.get("value") not in (".", "", None)]
        if not valid:
            indicators[label] = {"description": description, "value": None}
            continue

        try:
            raw = float(valid[0]["value"])
        except ValueError:
            indicators[label] = {"description": description, "value": None}
            continue

        entry = {
            "description":  description,
            "latest_date":  valid[0]["date"],
            "value":        raw,
        }
        if series_id in _YOY_CALC:
            yoy = _yoy_change(obs)
            entry["yoy_pct"]       = yoy
            entry["display_value"] = f"{yoy:+.2f}% YoY" if yoy is not None else "N/A"
        else:
            entry["display_value"] = f"{raw:.2f}"

        indicators[label] = entry

    # ── Synthesised investment signals ────────────────────────────────────────
    signals = []

    gdp = indicators.get("real_gdp_growth_pct", {}).get("value")
    if gdp is not None:
        if gdp < 0:
            signals.append("⚠ CONTRACTION: GDP growth negative — favour defensives (staples, utilities, healthcare); reduce cyclicals")
        elif gdp < 1.5:
            signals.append(f"↘ SLOW GROWTH: GDP {gdp:.1f}% — cautious on cyclicals; quality/dividend names preferred")
        elif gdp > 3:
            signals.append(f"↑ STRONG GROWTH: GDP {gdp:.1f}% — broad equity supportive; cyclicals and small-caps tend to outperform")
        else:
            signals.append(f"→ MODERATE GROWTH: GDP {gdp:.1f}% — balanced environment")

    core = indicators.get("core_cpi_yoy_pct", {}).get("yoy_pct")
    if core is not None:
        if core > 4:
            signals.append(f"⚠ HIGH INFLATION: Core CPI +{core:.1f}% — Fed likely hawkish; favour value, energy, financials; avoid long-duration growth")
        elif core > 2.5:
            signals.append(f"↑ ELEVATED INFLATION: Core CPI +{core:.1f}% — watch margin compression in consumer-facing companies")
        elif core < 1.5:
            signals.append(f"↓ LOW INFLATION: Core CPI +{core:.1f}% — accommodative Fed environment likely; growth/tech can sustain higher multiples")

    unemp = indicators.get("unemployment_rate", {}).get("value")
    if unemp is not None:
        if unemp > 6:
            signals.append(f"⚠ LABOUR WEAKNESS: Unemployment {unemp:.1f}% — consumer discretionary headwind; monitor retail and travel stocks closely")
        elif unemp < 3.8:
            signals.append(f"↑ TIGHT LABOUR: Unemployment {unemp:.1f}% — wage-push inflation risk; scrutinise labour-intensive companies' margin guidance")

    sentiment = indicators.get("consumer_sentiment", {}).get("value")
    if sentiment is not None:
        if sentiment < 65:
            signals.append(f"⚠ WEAK CONSUMER CONFIDENCE: {sentiment:.0f} — headwind for discretionary spending; defensive tilt recommended")
        elif sentiment > 95:
            signals.append(f"↑ STRONG CONSUMER CONFIDENCE: {sentiment:.0f} — tailwind for consumer discretionary and retail")

    claims = indicators.get("initial_jobless_claims_k", {}).get("value")
    if claims is not None:
        if claims > 300:
            signals.append(f"⚠ RISING CLAIMS: {claims:,.0f}k initial jobless claims — labour market softening; watch consumer spending names")
        elif claims < 220:
            signals.append(f"↓ LOW CLAIMS: {claims:,.0f}k — robust labour market; consumer spending likely resilient")

    return {
        "source":             "St. Louis Federal Reserve FRED API",
        "as_of":              str(datetime.date.today()),
        "indicators":         indicators,
        "investment_signals": signals,
        "note": (
            "Real-economy data that leads equity markets by 1-2 quarters. "
            "Combine with get_macro_environment() (market-price signals: yields, VIX, dollar) "
            "for a complete macro picture before making sector allocation decisions."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. GOOGLE TRENDS
# ══════════════════════════════════════════════════════════════════════════════

def get_google_trends(ticker: str, keywords: Optional[list] = None) -> dict:
    """
    Fetch Google search-interest trends for a company's brand or products.

    Search interest is a leading indicator of consumer demand — rising
    interest often precedes revenue beats for consumer-facing companies.
    Requires pytrends (pip install pytrends).  No API key needed.

    If no keywords are supplied the company's short name from yfinance is used.
    For product companies, passing product names gives better signal
    (e.g. ['iPhone', 'Mac'] for AAPL rather than 'Apple Inc').
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError:
        return {
            "error": "pytrends not installed",
            "setup": "Run: pip install pytrends  (then restart the app)",
        }

    # Resolve keywords
    if not keywords:
        try:
            info     = yf.Ticker(ticker).info
            keywords = [info.get("shortName") or ticker]
        except Exception:
            keywords = [ticker]
    keywords = [str(k) for k in keywords[:5]]

    try:
        pt = TrendReq(hl="en-US", tz=300)
        pt.build_payload(keywords, timeframe="today 12-m", geo="US")
        df = pt.interest_over_time()
    except Exception as e:
        return {"error": f"Google Trends request failed: {e}", "ticker": ticker}

    if df is None or df.empty:
        return {"error": "No data returned from Google Trends", "ticker": ticker, "keywords": keywords}

    trends = {}
    for kw in keywords:
        if kw not in df.columns:
            continue
        series     = df[kw]
        current    = int(series.iloc[-1])
        avg_3m     = int(series.iloc[-13:].mean())
        avg_12m    = int(series.mean())
        peak_12m   = int(series.max())
        # Trend: compare most-recent 8 weeks to the prior 16 weeks
        recent_avg = series.iloc[-8:].mean()
        prior_avg  = series.iloc[-24:-8].mean() if len(series) >= 24 else series.iloc[:-8].mean()
        if prior_avg > 0:
            change_pct = (recent_avg - prior_avg) / prior_avg * 100
            direction  = "rising" if change_pct > 10 else ("falling" if change_pct < -10 else "stable")
        else:
            change_pct, direction = 0, "stable"

        vs_avg = ((current - avg_12m) / max(avg_12m, 1)) * 100

        trends[kw] = {
            "current_week":       current,
            "3_month_avg":        avg_3m,
            "12_month_avg":       avg_12m,
            "peak_12m":           peak_12m,
            "trend_direction":    direction,
            "recent_8w_change":   f"{change_pct:+.0f}%",
            "vs_12m_avg":         f"{vs_avg:+.0f}%",
            "signal": (
                f"ACCELERATING interest (+{change_pct:.0f}% vs prior period) — potential demand tailwind"
                if change_pct > 20 else
                f"DECLINING interest ({change_pct:.0f}% vs prior period) — monitor revenue guidance"
                if change_pct < -20 else
                "Stable search interest — no significant demand shift detected"
            ),
        }

    return {
        "ticker":    ticker.upper(),
        "keywords":  keywords,
        "timeframe": "Past 12 months (US geography)",
        "source":    "Google Trends via pytrends",
        "trends":    trends,
        "note": (
            "Google Trends measures relative search interest (0–100 scale). "
            "Best used for CONSUMER-FACING companies where brand search interest "
            "correlates with revenue. Less signal for B2B companies — in those cases "
            "try product/service keywords (e.g. 'Azure' for MSFT, 'AWS' for AMZN). "
            "A rising trend 4–8 weeks before earnings is a leading demand indicator."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. RETAIL SENTIMENT  (StockTwits + Reddit)
# ══════════════════════════════════════════════════════════════════════════════

_POS_WORDS = {"buy", "bull", "long", "undervalued", "strong", "growth", "beat", "upgrade", "love", "hold"}
_NEG_WORDS = {"sell", "bear", "short", "overvalued", "weak", "miss", "downgrade", "avoid", "crash", "dump"}


def _stocktwits(ticker: str) -> dict:
    """Fetch StockTwits message stream and compute bull/bear ratio."""
    try:
        url  = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json"
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        if resp.status_code == 404:
            return {"available": False, "reason": "Ticker not found on StockTwits"}
        resp.raise_for_status()
        msgs = resp.json().get("messages", [])

        bullish = bearish = 0
        recent  = []
        for m in msgs:
            sent = (m.get("entities") or {}).get("sentiment") or {}
            tag  = sent.get("basic", "")
            if tag == "Bullish":
                bullish += 1
            elif tag == "Bearish":
                bearish += 1
            if len(recent) < 6:
                recent.append({
                    "text":      (m.get("body") or "")[:140],
                    "sentiment": tag or "neutral",
                    "date":      (m.get("created_at") or "")[:10],
                })

        labeled      = bullish + bearish
        bullish_pct  = round(bullish / labeled * 100, 1) if labeled else None
        return {
            "available":    True,
            "message_count": len(msgs),
            "bullish":      bullish,
            "bearish":      bearish,
            "neutral":      len(msgs) - labeled,
            "bullish_pct":  bullish_pct,
            "recent":       recent,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _reddit(ticker: str) -> dict:
    """Search recent Reddit posts across investing subreddits."""
    subreddits = ["investing", "wallstreetbets", "stocks", "SecurityAnalysis"]
    posts      = []
    headers    = {"User-Agent": "investment-agent/1.0 (research tool)"}

    for sub in subreddits:
        try:
            resp = requests.get(
                f"https://www.reddit.com/r/{sub}/search.json",
                params={"q": ticker, "restrict_sr": "true", "sort": "new", "t": "week", "limit": 10},
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            for child in resp.json().get("data", {}).get("children", []):
                pd = child.get("data", {})
                posts.append({
                    "subreddit": sub,
                    "title":     (pd.get("title") or "")[:120],
                    "score":     pd.get("score", 0),
                    "comments":  pd.get("num_comments", 0),
                    "date":      datetime.datetime.fromtimestamp(
                        pd.get("created_utc", 0)
                    ).strftime("%Y-%m-%d") if pd.get("created_utc") else "",
                    "url":       "https://reddit.com" + (pd.get("permalink") or ""),
                })
        except Exception:
            continue

    pos = sum(1 for p in posts if any(w in p["title"].lower() for w in _POS_WORDS))
    neg = sum(1 for p in posts if any(w in p["title"].lower() for w in _NEG_WORDS))
    top = sorted(posts, key=lambda x: x["score"], reverse=True)[:6]

    return {
        "mention_count":     len(posts),
        "subreddits":        subreddits,
        "positive_titles":   pos,
        "negative_titles":   neg,
        "top_posts":         top,
    }


def get_retail_sentiment(ticker: str) -> dict:
    """
    Aggregate retail investor sentiment from StockTwits and Reddit.

    Retail sentiment is a CONTRARIAN signal for long-term investors:
    - Extreme bullishness (>80% bulls) often precedes corrections.
    - Extreme bearishness (<25% bulls) can mark a bottom.
    Use alongside fundamentals, not instead of them.
    """
    st = _stocktwits(ticker)
    rd = _reddit(ticker)

    # Synthesise overall signal
    signals = []
    bull_pct = st.get("bullish_pct") if st.get("available") else None
    if bull_pct is not None:
        if bull_pct > 80:
            signals.append(
                f"⚠ RETAIL EUPHORIA: {bull_pct:.0f}% bullish on StockTwits — "
                "contrarian caution; retail peaks often precede short-term corrections"
            )
        elif bull_pct > 60:
            signals.append(f"↑ MODERATELY BULLISH: {bull_pct:.0f}% bulls — retail sentiment supportive, not extreme")
        elif bull_pct < 25:
            signals.append(
                f"⚠ EXTREME PESSIMISM: {bull_pct:.0f}% bullish — potential contrarian buy signal; "
                "verify fundamentals are intact before acting"
            )
        elif bull_pct < 45:
            signals.append(f"↓ BEARISH LEAN: {bull_pct:.0f}% bulls — retail cautious; "
                           "check whether fear is fundamental or sentiment-driven")

    mentions = rd.get("mention_count", 0)
    if mentions > 50:
        signals.append(
            f"🔊 HIGH REDDIT VOLUME: {mentions} posts this week — elevated retail attention "
            "may drive short-term volatility; avoid chasing momentum"
        )
    elif mentions == 0:
        signals.append("Low Reddit visibility — stock is flying under retail radar (can be a positive for long-term holders)")

    return {
        "ticker":       ticker.upper(),
        "stocktwits":   st,
        "reddit":       rd,
        "signals":      signals,
        "note": (
            "Retail sentiment is a sentiment THERMOMETER, not a compass. "
            "For a long-term investor: extreme readings (very bullish or very bearish) "
            "are more actionable than moderate ones. Always cross-check with "
            "fundamentals (get_stock_fundamentals) and SEC filings (get_material_events)."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. RSS NEWS AGGREGATION  (stdlib xml — no extra dependencies)
# ══════════════════════════════════════════════════════════════════════════════

import xml.etree.ElementTree as _ET

# Feed URL templates — {ticker} is replaced at call time
_RSS_FEEDS = [
    ("Yahoo Finance",  "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"),
    ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/"),
    ("Seeking Alpha",  "https://seekingalpha.com/api/sa/combined/{ticker}.xml"),
]


def _parse_rss(xml_text: str) -> list:
    """Parse RSS/Atom XML and return a list of (title, link, published, summary) tuples."""
    try:
        root = _ET.fromstring(xml_text)
    except _ET.ParseError:
        # Strip potential namespace declarations that confuse ElementTree
        clean = re.sub(r'\sxmlns[^"]*"[^"]*"', "", xml_text)
        try:
            root = _ET.fromstring(clean)
        except _ET.ParseError:
            return []

    items = []
    # RSS 2.0: <channel><item>…</item></channel>
    for item in root.iter("item"):
        title   = (item.findtext("title") or "").strip()
        link    = (item.findtext("link")  or "").strip()
        pub     = (item.findtext("pubDate") or item.findtext("published") or "").strip()
        summary = re.sub(r"<[^>]+>", "", item.findtext("description") or "").strip()
        if title:
            items.append((title, link, pub[:20], summary[:300]))
    # Atom: <entry>…</entry>
    if not items:
        for entry in root.iter("entry"):
            title   = (entry.findtext("title") or "").strip()
            link_el = entry.find("link")
            link    = (link_el.get("href") if link_el is not None else "") or ""
            pub     = (entry.findtext("published") or entry.findtext("updated") or "").strip()
            summary = re.sub(r"<[^>]+>", "", entry.findtext("summary") or "").strip()
            if title:
                items.append((title, link, pub[:20], summary[:300]))
    return items


def get_rss_news(ticker: str, max_per_feed: int = 5) -> dict:
    """
    Aggregate recent news headlines from multiple RSS feeds beyond yfinance.

    Covers Yahoo Finance, MarketWatch, and Seeking Alpha to surface
    analyst commentary, earnings previews, and sector news that
    yfinance's built-in news endpoint may miss.
    Uses stdlib xml.etree.ElementTree — no extra dependencies required.
    """
    ticker_up = ticker.upper()
    all_items  = []
    sources    = []

    for source_name, url_tpl in _RSS_FEEDS:
        url = url_tpl.replace("{ticker}", ticker_up)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=12)
            if resp.status_code != 200:
                continue
            parsed = _parse_rss(resp.text)
            if not parsed:
                continue
            count = 0
            for title, link, pub, summary in parsed:
                # For the non-ticker-specific MarketWatch feed, filter by ticker mention
                if source_name == "MarketWatch" and ticker_up not in title.upper():
                    continue
                all_items.append({
                    "source":    source_name,
                    "title":     title[:150],
                    "published": pub,
                    "link":      link,
                    "summary":   summary,
                })
                count += 1
                if count >= max_per_feed:
                    break
            if count:
                sources.append(source_name)
        except Exception:
            continue

    # Deduplicate by title similarity (keep first occurrence)
    seen   = set()
    unique = []
    for item in all_items:
        key = item["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Sort newest-first where dates are parseable
    try:
        unique.sort(key=lambda x: x["published"], reverse=True)
    except Exception:
        pass

    return {
        "ticker":        ticker_up,
        "article_count": len(unique),
        "sources_hit":   sources,
        "articles":      unique[:20],
        "note": (
            "RSS headlines aggregated from multiple financial news sources. "
            "Look for: recurring negative themes (multiple sources = higher signal), "
            "earnings preview tone, analyst rating changes, and any M&A / regulatory news "
            "not yet reflected in get_stock_news()."
        ),
    }
