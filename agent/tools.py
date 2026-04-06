"""
Tool definitions and handlers for the Claude investment agent.
Each tool maps to a market data or portfolio action.
"""

import json
from typing import Any, Optional

from agent import market_data, portfolio, sec_data, external_data, ml_insights



def _reset_session_tracker() -> None:
    global _SESSION_TOOL_CALLS, _SESSION_TOOL_NAMES, _SESSION_START_TS
    _SESSION_TOOL_CALLS = 0
    _SESSION_TOOL_NAMES = set()
    _SESSION_START_TS = _time.time()


def _get_session_stats() -> dict:
    elapsed = int(_time.time() - _SESSION_START_TS) if _SESSION_START_TS else 0
    return {
        "total_tool_calls": _SESSION_TOOL_CALLS,
        "unique_tools_used": len(_SESSION_TOOL_NAMES),
        "duration_seconds": elapsed,
    }


# ── Tool schemas for Claude ────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_portfolio_status",
        "description": (
            "Returns the current portfolio: cash balance, all stock holdings with "
            "their shares, average cost, and current unrealized P&L. Use this to "
            "understand what you currently own before making decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_stock_quote",
        "description": (
            "Get the current market price and basic stats for a stock ticker. "
            "Use this to check the current price before buying or selling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT, NVDA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_fundamentals",
        "description": (
            "Get detailed fundamental data for a stock: P/E ratio, profit margins, "
            "ROE, debt levels, dividend yield, analyst ratings, and more. "
            "Use this to evaluate whether a stock is a good long-term investment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_price_history",
        "description": (
            "Get historical price performance for a stock over a given period. "
            "Shows price change %, highs, lows, and average volume."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "period": {
                    "type": "string",
                    "description": "Time period: 1mo, 3mo, 6mo, 1y, 2y, 5y",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": (
            "Compute technical indicators for a stock to assess entry timing and trend health. "
            "Returns RSI-14, MACD (12/26/9), Bollinger Bands (20-day, 2σ), EMA-50/200 "
            "(golden/death cross), volume vs 20-day average, and an overall signal summary.\n\n"
            "Use this AFTER fundamentals confirm a great business at a fair price — "
            "technicals answer 'is now a good entry point?' not 'should I buy this business?'. "
            "A fundamentally sound stock with bearish technicals may warrant waiting for a "
            "better entry; a fundamentally weak stock with bullish technicals is still a pass.\n\n"
            "Key signals to act on:\n"
            "  - RSI < 30 (oversold): stock may be at a near-term bottom — favorable entry\n"
            "  - RSI > 70 (overbought): wait for a pullback before buying\n"
            "  - MACD bullish crossover: momentum turning up — confirms entry timing\n"
            "  - MACD bearish crossover: momentum turning down — wait or reduce size\n"
            "  - Price below EMA-50 and EMA-200 (death cross): stock in downtrend — be cautious\n"
            "  - Price at lower Bollinger Band: near-term oversold, may bounce\n"
            "  - Price at upper Bollinger Band: extended, high risk of short-term pullback\n"
            "  - High volume on up days: institutional buying — supportive of thesis\n\n"
            "No API key required — computed from yfinance 1-year daily history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_short_interest",
        "description": (
            "Retrieve short interest data for a stock: short % of float, days-to-cover "
            "(short ratio), month-over-month change in shares short, and interpreted signals.\n\n"
            "Short interest has two readings — use both:\n"
            "  (A) Institutional bear signal: high short % + rising trend means smart money "
            "has researched this stock and is actively betting against it. Treat as a red flag "
            "that warrants extra scrutiny — what do they know that the bull case missed?\n"
            "  (B) Squeeze catalyst: high short % on a stock with a strong fundamental thesis "
            "and an upcoming catalyst means short covering will amplify any price move higher. "
            "This is upside optionality on top of the thesis — not a standalone buy reason.\n\n"
            "Thresholds:\n"
            "  short_level 'high' (15-25%) or 'very_high' (>25%): investigate short thesis\n"
            "  mom_direction 'rising': bears adding — dig deeper before buying\n"
            "  mom_direction 'falling': short covering — potential near-term tailwind\n"
            "  days_to_cover > 7: elevated squeeze risk if positive catalyst emerges\n\n"
            "Data is sourced from FINRA via yfinance and updated twice monthly — "
            "figures typically lag ~2 weeks. No API key required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_options_flow",
        "description": (
            "Retrieve options market data for a stock: put/call volume and OI ratios, "
            "ATM implied volatility vs. 30-day realized volatility, and unusual contract "
            "activity (large fresh directional bets). Analyzes the nearest 3 expiries.\n\n"
            "Three distinct signals — read each separately:\n\n"
            "1. Put/Call ratio (sentiment):\n"
            "   < 0.7 = bullish (heavy call buying vs. puts)\n"
            "   0.7–1.0 = neutral\n"
            "   > 1.0 = bearish (heavy put buying — hedging or directional short)\n"
            "   > 1.5 = strongly bearish\n\n"
            "2. IV vs. realized volatility (event/fear pricing):\n"
            "   IV >> realized vol: market pricing in an upcoming event or fear — options\n"
            "   are expensive; buying stock outright is more capital-efficient than buying calls\n"
            "   IV ≈ realized vol: normal; no special event risk priced in\n"
            "   IV << realized vol: options unusually cheap — low implied risk\n\n"
            "3. Unusual contracts (volume ≥ 3× open interest, ≥ 500 contracts):\n"
            "   Fresh directional bets by large traders, not hedges or rolls.\n"
            "   Call-skewed unusual activity = bullish smart-money signal\n"
            "   Put-skewed unusual activity = bearish smart-money signal or large hedge\n\n"
            "Use this as a positioning/sentiment check, NOT as a buy/sell trigger. "
            "A fundamentally great stock with bearish options positioning is still a buy — "
            "but unusually heavy put buying or very high IV warrants investigation. "
            "No API key required — data from yfinance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "search_stocks",
        "description": (
            "Search for stocks by company name or keyword. Returns matching tickers. "
            "Use this when you know a company name but not its ticker symbol."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Company name or keyword to search for",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_market_summary",
        "description": (
            "Get a snapshot of major market indices (S&P 500, NASDAQ, Dow Jones, "
            "Russell 2000, VIX). Use this to gauge overall market conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "buy_stock",
        "description": (
            "Execute a paper (simulated) buy order. Purchases shares at the current "
            "market price and deducts from your cash balance. Always call get_stock_quote "
            "first to confirm the current price. Specify either shares OR dollar_amount."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to buy",
                },
                "shares": {
                    "type": "number",
                    "description": "Number of shares to buy (use this OR dollar_amount)",
                },
                "dollar_amount": {
                    "type": "number",
                    "description": "Dollar amount to invest (use this OR shares). Will be converted to shares at current price.",
                },
                "notes": {
                    "type": "string",
                    "description": "Reasoning for this buy decision",
                },
                "screener_snapshot": {
                    "type": "object",
                    "description": (
                        "Optional: pass the screen_stocks result row for this ticker "
                        "(the dict containing score, peg_ratio, relative_momentum_pct, etc.). "
                        "Saves the signal state at purchase time for future performance attribution."
                    ),
                },
            },
            "required": ["ticker", "notes"],
        },
    },
    {
        "name": "sell_stock",
        "description": (
            "Execute a paper (simulated) sell order. Sells shares and adds proceeds "
            "to your cash balance. Specify either shares OR 'all' to sell everything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to sell",
                },
                "shares": {
                    "type": "number",
                    "description": "Number of shares to sell",
                },
                "sell_all": {
                    "type": "boolean",
                    "description": "Set to true to sell all shares of this ticker",
                },
                "notes": {
                    "type": "string",
                    "description": "Reasoning for this sell decision",
                },
            },
            "required": ["ticker", "notes"],
        },
    },
    {
        "name": "get_transaction_history",
        "description": "View recent buy/sell transaction history for the portfolio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent transactions to show (default 20)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_stock_news",
        "description": (
            "Fetch recent news headlines for a stock from Yahoo Finance. "
            "Use this to stay aware of earnings surprises, CEO changes, product launches, "
            "lawsuits, regulatory actions, or any event that could affect the investment thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of articles to return (default 8, max 15)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_earnings_calendar",
        "description": (
            "Get the next scheduled earnings date for a stock, consensus EPS and revenue estimates, "
            "and the last 4 quarters of beat/miss history. "
            "Use this before buying — earnings are the single biggest short-term risk for any position."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_analyst_upgrades",
        "description": (
            "Get recent analyst upgrades and downgrades for a stock: which firm acted, "
            "whether it was an upgrade/downgrade/initiation, and the grade change. "
            "A cluster of downgrades is a warning sign; upgrades can signal improving sentiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of recent analyst actions to return (default 10)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_insider_activity",
        "description": (
            "Get recent insider transactions (buys and sells by executives, directors, and major shareholders). "
            "Significant insider buying — especially by the CEO or CFO — is one of the strongest "
            "bullish signals available. Heavy insider selling can be a warning sign."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_macro_environment",
        "description": (
            "Get key macroeconomic indicators: 10-year and 2-year Treasury yields, yield curve status, "
            "dollar index, oil, gold, and VIX. Includes synthesised signals so you can adjust "
            "sector allocation accordingly (e.g. high rates → favour value over growth, "
            "strong dollar → avoid multinationals, inverted curve → recession risk)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_hedge_recommendations",
        "description": (
            "Translate the current macro regime into concrete defensive ETF hedge recommendations: "
            "which ETFs to buy, how much to allocate, when to enter, and when to unwind.\n\n"
            "Call this during portfolio rebalancing when the macro environment looks stressed. "
            "Triggers: VIX > 25, inverted yield curve, oil > $85, or any RISK_OFF / INFLATIONARY "
            "/ STAGFLATION / HIGH_RATES regime signal from get_macro_environment.\n\n"
            "Hedge universe (plain, non-leveraged, non-inverse ETFs only):\n"
            "  TLT — 20+ Year Treasury Bonds (RISK_OFF flight-to-safety)\n"
            "  IEF — 7-10 Year Treasury Bonds (moderate duration, also HIGH_RATES)\n"
            "  SHV — Short Treasury Bonds <1yr (cash equivalent, earns short-term yield)\n"
            "  GLD — Gold (inflation + crisis hedge; useful in STAGFLATION)\n"
            "  TIP — TIPS Bonds (inflation-protected; INFLATIONARY regime)\n"
            "  GSG — Broad Commodity ETF (energy/metals/agriculture; pure INFLATIONARY)\n\n"
            "Hard rules:\n"
            "  - Hedges are ALWAYS funded from cash, never by selling equity positions\n"
            "  - Maximum hedge allocation: 20% of total portfolio\n"
            "  - No recommendation is made in NORMAL / RISK_ON regimes\n"
            "  - Unwind when the triggering regime resolves\n\n"
            "Pass equity_pct and cash_pct from get_portfolio_status so recommendations "
            "are sized to your actual portfolio composition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "equity_pct": {
                    "type": "number",
                    "description": (
                        "Current equity allocation as % of total portfolio value (0-100). "
                        "From get_portfolio_status: equity_value / (equity_value + cash) × 100."
                    ),
                },
                "cash_pct": {
                    "type": "number",
                    "description": (
                        "Current cash as % of total portfolio value (0-100). "
                        "Hedges will be scaled to fit available cash."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_benchmark_comparison",
        "description": (
            "Compare the portfolio's total return since inception against the S&P 500. "
            "Shows alpha (outperformance or underperformance) and historical snapshots. "
            "Call this during portfolio reviews to understand whether the strategy is "
            "actually beating a simple index fund."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_portfolio_metrics",
        "description": (
            "Return risk and return metrics computed from portfolio snapshot history: "
            "Sharpe ratio, max drawdown, annualised volatility, and rolling 1/3/6-month "
            "returns vs S&P 500. "
            "Call this in Step 2 to understand whether the portfolio is taking too much "
            "risk relative to its returns, and how recent momentum compares to the benchmark."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_investment_memory",
        "description": (
            "Retrieve your past investment theses for current holdings and recently closed positions. "
            "Call this at the start of each session to understand why you made past decisions, "
            "whether original theses are still valid, and what worked or didn't in closed positions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_session_reflections",
        "description": (
            "Retrieve your past post-session reflections — lessons and observations you documented "
            "from previous portfolio reviews. Call this at the start of sessions to apply past learnings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent reflections to retrieve (default 5)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_international_universe",
        "description": (
            "Return a curated list of ~200 major international stocks for screening — "
            "companies NOT already in the US S&P 500 / broad universe.\n\n"
            "Covers top companies from Europe (UK, Germany, France, Switzerland, Netherlands, "
            "Nordics), Asia-Pacific (Japan, South Korea, HK/China, Taiwan, Australia, Singapore), "
            "Latin America (Brazil, Mexico, Chile), Canada, and India/Israel.\n\n"
            "Mix of US-listed ADRs (NYSE/NASDAQ — best data quality, e.g. TSM, ASML, NVO, SAP) "
            "and direct foreign-listed tickers with exchange suffixes (e.g. NESN.SW, 005930.KS, "
            "0700.HK). All accessible via yfinance.\n\n"
            "WORKFLOW: call this once per session, then pass the returned tickers list to "
            "screen_stocks in batches of 50-80. Foreign-suffix tickers that lack data are "
            "silently skipped by screen_stocks.\n\n"
            "Use the optional 'region' parameter to focus on one geography if the macro "
            "environment favours it (e.g. 'europe' during USD weakness, 'asia' for EM exposure)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": (
                        "Optional: narrow to one region. "
                        "Valid: 'europe', 'asia', 'latam', 'canada', 'india'. "
                        "Omit (or pass null) for all regions combined."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "discover_universe_parallel",
        "description": (
            "Two-layer universe screen. Uses a cached quality score (stable fundamentals: "
            "revenue growth, margins, ROE, debt) to shortlist the top 150 names from ~700 tickers, "
            "then fetches fresh valuation metrics (FCF yield, PEG, momentum) for those 150 only. "
            "Returns 'top_candidates': top 60 globally ranked by combined quality+valuation score.\n\n"
            "Each candidate includes quality_score, valuation_score, combined_score, and a "
            "'universe' field ('us_sp500' or 'international'). "
            "Pass top_candidates directly to research_stocks_parallel — no screening calls needed.\n\n"
            "If the quality cache is empty it auto-runs a full refresh (~10 min first time). "
            "Call refresh_universe_scores to manually rebuild the cache (do quarterly or when "
            "quality_cache_age_days > 90)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "refresh_universe_scores",
        "description": (
            "Rebuild the quality score cache for the full universe (~700 tickers: S&P 500 + international). "
            "Scores every ticker on stable, non-price-dependent factors: revenue growth, profit margins, "
            "ROE, and debt/equity. Results are saved to the local database and used by "
            "discover_universe_parallel on every subsequent run.\n\n"
            "This is a slow call (~5-10 minutes for 700 tickers). Run it:\n"
            "  • On first use (cache is empty)\n"
            "  • Quarterly, after earnings season, when business fundamentals may have shifted\n"
            "  • When discover_universe_parallel reports quality_cache_age_days > 90\n\n"
            "Do NOT call this on every session — the quality cache is designed to be stable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_stock_universe",
        "description": (
            "Fetch tickers from major US stock universes. "
            "When index='sp500', returns ALL ~500 S&P 500 constituents (no sampling) — "
            "use this for exhaustive coverage. Optionally filter by GICS sector. "
            "When index='broad' or 'all', returns a random sample of 'sample_n' tickers "
            "from the ~2700-stock universe; call multiple times with different random_seed "
            "values to cover more of the universe."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "description": "Which universe to sample from: 'sp500' (~500 large caps), 'broad' (~2700 US-listed stocks), or 'all' (combined)",
                    "enum": ["sp500", "broad", "all"],
                },
                "sample_n": {
                    "type": "integer",
                    "description": "Number of tickers to return in this sample (default 200, max 300).",
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Seed for random sampling. Use different values (0, 1, 2, 3...) across calls to get different batches.",
                },
                "sector": {
                    "type": "string",
                    "description": (
                        "Optional GICS sector filter. Only returns tickers from this sector. "
                        "Valid values: 'Information Technology', 'Health Care', 'Financials', "
                        "'Consumer Discretionary', 'Communication Services', 'Industrials', "
                        "'Consumer Staples', 'Energy', 'Utilities', 'Real Estate', 'Materials'. "
                        "Sector data is only available for S&P 500 constituents."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "screen_stocks",
        "description": (
            "Run a fast parallel fundamental screen across a list of tickers. "
            "Fetches key metrics (P/E, revenue growth, profit margin, ROE, debt) for each ticker "
            "and returns ALL scored candidates ranked by a composite quality + value score — "
            "no artificial cap on results. "
            "Pass 50-100 tickers at a time (from get_stock_universe) for best results. "
            "Use this to discover high-quality opportunities across the full market — "
            "not just popular large caps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to screen",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_sector_exposure",
        "description": (
            "Show current portfolio weights broken down by GICS sector. "
            "Call this before buying new positions to see where you're already concentrated "
            "and where you have room to add without over-weighting a sector."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_to_watchlist",
        "description": (
            "Add a stock to the watchlist when you like the business but the timing isn't right — "
            "e.g. earnings in the next 2 weeks, valuation slightly too high, or waiting for a "
            "pullback to a target price. The watchlist is reviewed at the start of every session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "company_name": {"type": "string", "description": "Company name (optional)"},
                "reason": {
                    "type": "string",
                    "description": "Why you like this stock and what would trigger a buy (e.g. 'strong FCF, waiting for post-earnings dip below $X')",
                },
                "target_entry_price": {
                    "type": "number",
                    "description": "Optional price at or below which you'd be a buyer",
                },
            },
            "required": ["ticker", "reason"],
        },
    },
    {
        "name": "get_watchlist",
        "description": (
            "Retrieve the current watchlist — stocks you've flagged for future purchase. "
            "Call this at the start of each session to check if any watchlist candidates "
            "have reached their target entry price or had a meaningful pullback."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_from_watchlist",
        "description": (
            "Remove a stock from the watchlist. Call this after buying the stock, "
            "or if the investment thesis has broken down and it's no longer worth watching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol to remove"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_trade_outcomes",
        "description": (
            "Return all past buy signal snapshots with their actual outcomes. "
            "For each recorded buy, shows the screener signals that were present at purchase "
            "(PEG, momentum, FCF yield, etc.) alongside the eventual return. "
            "Use this to identify which signals have historically predicted positive returns "
            "and weight them more heavily in future screening decisions."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_signal_performance",
        "description": (
            "Analyze which screener signals (PEG < 1.5, FCF yield > 3%, positive momentum, "
            "revenue growth > 10%) have historically predicted positive returns in this portfolio. "
            "Returns per-signal statistics split by whether the threshold was met at buy time. "
            "Call this in Step 1 alongside get_trade_outcomes to calibrate signal weights for this session."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_to_shadow_portfolio",
        "description": (
            "Record a stock you analyzed and decided NOT to buy or watchlist. "
            "Use this for stocks that were seriously considered but rejected (too expensive, "
            "weak moat, sector crowded, thesis uncertain). "
            "Tracked so future sessions can review whether passing was the right call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "price_at_consideration": {
                    "type": "number",
                    "description": "Current stock price at the time you considered it",
                },
                "reason_passed": {
                    "type": "string",
                    "description": "Primary reason for passing (e.g. 'overvalued', 'weak FCF', 'sector too heavy', 'thesis unclear')",
                },
                "notes": {
                    "type": "string",
                    "description": "Additional context on the decision",
                },
            },
            "required": ["ticker", "price_at_consideration", "reason_passed"],
        },
    },
    {
        "name": "run_backtest",
        "description": (
            "Run a strategy backtest in one of three modes to validate whether the "
            "screening and trading approach is actually working.\n\n"
            "Modes:\n\n"
            "  'trade_history' — Replays ALL closed trades (no 20-trade limit). "
            "Computes win rate, avg return, Sharpe ratio, max drawdown, and "
            "S&P 500 alpha per trade. Segments results by market regime at entry "
            "(VIX level: low_volatility / normal / elevated / high_fear). "
            "Shows best and worst trades. Call this in Step 1 periodically (every "
            "5+ closed trades) to validate that the strategy is generating alpha.\n\n"
            "  'signal_cohorts' — Breaks all closed trades into signal cohorts: "
            "PEG < 1.5 vs ≥ 1.5, FCF yield > 3% vs ≤ 3%, positive vs negative "
            "momentum, score ≥ 8 vs < 8, and bull entry (VIX < 20) vs bear entry. "
            "Shows win rate and avg return for each cohort so you can see which "
            "signals are actually predictive in YOUR portfolio's specific history. "
            "Use this to update signal weights in get_ml_factor_weights.\n\n"
            "  'momentum' — Simulates buying the top-momentum tercile of a "
            "provided ticker list at a point holding_days ago, and measures "
            "actual forward return vs. S&P 500 buy-and-hold. Uses price data "
            "only — no look-ahead on fundamentals. Pass your current screener "
            "candidates as tickers to validate the momentum signal. "
            "Requires tickers list.\n\n"
            "Call 'trade_history' and 'signal_cohorts' in Step 1 once you have "
            "≥5 closed trades. Call 'momentum' in Step 4 after screening to "
            "validate momentum as a factor before weighting it in decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["trade_history", "signal_cohorts", "momentum"],
                    "description": "Which backtest to run.",
                },
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required for mode='momentum'. Pass screener candidate tickers.",
                },
                "holding_days": {
                    "type": "integer",
                    "description": (
                        "For mode='momentum': how many days ago the simulated entry was. "
                        "Default 90 (one quarter). Use 180 or 365 for longer horizons."
                    ),
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "get_shadow_performance",
        "description": (
            "Call this in Step 1 to audit past pass decisions — if a rejected stock is up 30%, "
            "understand why you were wrong; if it's down 20%, your thesis was validated. "
            "Use these lessons to sharpen your screening judgment."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── ML insights (learned from portfolio history) ──────────────────────────
    {
        "name": "get_regime_change_status",
        "description": (
            "Detect the current macro regime and compare it to the last recorded regime. "
            "Returns whether the regime has changed since the previous run, how many days ago "
            "the last regime was recorded, and a human-readable change summary. "
            "Call this at the start of every session to check for macro shifts that may require "
            "portfolio rebalancing — a regime change from RISK_ON to RISK_OFF, for example, "
            "should trigger a full portfolio review even mid-cycle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_ml_factor_weights",
        "description": (
            "Learn which screener signals have actually predicted returns in THIS portfolio's history. "
            "Returns data-driven factor weights blended with regime-adjusted priors. "
            "Weights are 100% regime-prior when no closed trades exist, then shift smoothly toward "
            "data-driven as closed trades accumulate (25% at 5 trades → 75% at 25+ trades). "
            "Also detects the current macro regime (RISK_ON / RISK_OFF / INFLATIONARY / NORMAL) "
            "and returns feature correlations with returns. "
            "Call this at the start of each session to understand which signals to trust most "
            "when evaluating screener candidates. Use blended_weights to manually re-rank results."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_watchlist_triggers",
        "description": (
            "Fetch live prices for every watchlist item and compare against target entry prices.\n\n"
            "Returns four buckets:\n"
            "  TRIGGERED   — price is AT or BELOW the target. Run deep research immediately.\n"
            "  APPROACHING — price is within 10% above target. Watch closely this session.\n"
            "  WAITING     — price is more than 10% above target. No action needed.\n"
            "  NO_TARGET   — no target price set; current price reported for reference.\n\n"
            "Call this at the start of every session BEFORE the universe screen. "
            "If any items are TRIGGERED, prioritise them for deep research in Step 4 — "
            "they may already be actionable without needing to screen the full universe."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_watchlist_earnings",
        "description": (
            "Fetch upcoming earnings dates for every watchlist item and bucket by urgency:\n"
            "  IMMINENT — earnings within 7 days: research the stock NOW, before results land\n"
            "  UPCOMING — earnings within 30 days: prepare thesis, set target entry\n"
            "  DISTANT  — earnings >30 days away: no immediate action needed\n\n"
            "Call this in Step 1 every session alongside check_watchlist_triggers. "
            "IMMINENT items must be reviewed in Step 4 even if their price hasn't hit target — "
            "earnings can create sudden entry opportunities or confirm thesis breaks."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_fundamental_deterioration",
        "description": (
            "Check all held positions for fundamental deterioration that may warrant exit. "
            "Flags: revenue declining YoY, FCF turned negative, gross margin < 20%, "
            "high leverage (D/E > 2x), ROE < 8%, earnings declining (fwd PE >> trailing PE). "
            "Returns severity ratings: WATCH (1 flag), REVIEW (2 flags), EXIT (3+ flags). "
            "Call this at the start of each session alongside portfolio status. "
            "This is a long-term portfolio — exits are fundamentals-driven, not price-driven."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "description": "List of held positions, each with at least a 'ticker' key",
                    "items": {"type": "object"},
                }
            },
            "required": ["holdings"],
        },
    },
    {
        "name": "check_earnings_surprises",
        "description": (
            "Check held positions for recent earnings surprises (actual EPS vs estimate). "
            "Flags significant beats (>15% above estimate) and misses (>15% below). "
            "A large miss on a held position triggers re-research — the original thesis "
            "may be breaking. A large beat may warrant adding to the position. "
            "Call this alongside check_fundamental_deterioration in Step 2."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {"type": "array", "items": {"type": "object"},
                            "description": "List of holdings with at least a ticker key"}
            },
            "required": ["holdings"],
        },
    },
    {
        "name": "check_dividend_payments",
        "description": (
            "Check held positions for dividend payments. Returns annual dividend rate, "
            "yield %, estimated annual income per holding, and ex-dividend dates. "
            "Dividends are treated as additional cash income for reinvestment. "
            "Call during portfolio review to account for dividend income in total return calculations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {"type": "array", "items": {"type": "object"},
                            "description": "List of holdings with ticker and shares keys"}
            },
            "required": ["holdings"],
        },
    },
    {
        "name": "calculate_intrinsic_value",
        "description": (
            "Standardised 3-stage DCF model for a single stock. Produces consistent, "
            "comparable intrinsic value estimates across all sessions and subagents.\n\n"
            "Model parameters (fixed, non-negotiable):\n"
            "  Stage 1 (yr 1-5):  conservative FCF/earnings growth, haircutted by 20%\n"
            "  Stage 2 (yr 6-10): linear fade from Stage 1 rate → 2.5% terminal\n"
            "  Stage 3 (terminal): 2.5% perpetuity growth\n"
            "  Discount rate:     10.0%\n\n"
            "Returns bear/base/bull IV per share and margin of safety at current price. "
            "Use the BASE scenario as the primary IV reference. "
            "Call this for every finalist before making a buy or watchlist decision — "
            "it replaces informal IV estimates with a standardised, auditable number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_watchlist_history",
        "description": (
            "Return the history of watchlist lifecycle events: when items were TRIGGERED "
            "(price hit target), APPROACHING, BOUGHT, or REMOVED. "
            "Use this periodically to audit IV estimate accuracy — if TSM was triggered at $230 "
            "but never bought and is now $180, that tells you the IV estimate was wrong. "
            "Pass an optional ticker to filter to one stock's history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Optional: filter to a specific ticker's history",
                },
            },
            "required": [],
        },
    },
    {
        "name": "prioritize_watchlist_ml",
        "description": (
            "Score every watchlist item using ML-derived factor weights and return a ranked list. "
            "For each item: fetches current fundamentals, applies learned factor weights, "
            "computes an ML score (0-10), flags strengths and risk factors, and highlights items "
            "near their target entry price (promoted to top of their score tier). "
            "Call this alongside get_watchlist() to prioritise which candidates deserve the deepest "
            "research this session. Score ≥7: strong candidate — investigate thoroughly. "
            "Score <4: weak fit for current regime — lower priority unless thesis is compelling."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_conviction_position_size",
        "description": (
            "Calculate the appropriate position size for a stock given its conviction score "
            "and the current macro regime. Call this BEFORE placing any buy order to determine "
            "how many dollars to allocate. Conviction 9-10 = full position, 7-8 = 75%, "
            "5-6 = 50%. Regime sets the base size (RISK_ON=15%, NORMAL=12%, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conviction_score": {"type": "integer", "description": "Conviction score 1-10 from the research report"},
                "regime": {"type": "string", "description": "Current macro regime (RISK_ON/NORMAL/INFLATIONARY/RISK_OFF/STAGFLATION)"},
                "portfolio_equity": {"type": "number", "description": "Total portfolio equity value in dollars (cash + holdings market value)"},
            },
            "required": ["conviction_score", "regime", "portfolio_equity"],
        },
    },
    {
        "name": "get_position_size_recommendation",
        "description": (
            "Estimate drawdown risk for a specific stock and recommend an appropriate position size "
            "(% of portfolio). Combines three inputs: (1) feature-based risk flags on valuation, "
            "FCF, momentum, growth, and profitability; (2) a logistic regression drawdown model "
            "trained on the portfolio's own closed trade history (when ≥5 closed trades exist); "
            "(3) regime-adjusted base size (smaller in RISK_OFF / STAGFLATION). "
            "Call this just before executing a buy to calibrate position size. "
            "Pass the screener_snapshot features from screen_stocks as the 'features' argument. "
            "The recommendation respects the 20% maximum position cap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol being considered for purchase",
                },
                "features": {
                    "type": "object",
                    "description": (
                        "Screener features for this stock — pass the screen_stocks result dict. "
                        "Expected keys: peg_ratio, fcf_yield_pct, relative_momentum_pct, "
                        "revenue_growth_pct, profit_margin_pct, roe_pct."
                    ),
                },
            },
            "required": ["ticker", "features"],
        },
    },
    # ── External data sources ─────────────────────────────────────────────────
    {
        "name": "get_economic_indicators",
        "description": (
            "Fetch key US macroeconomic indicators from the Federal Reserve FRED API: "
            "real GDP growth, CPI inflation, core CPI, unemployment rate, initial jobless "
            "claims, retail sales, consumer sentiment, industrial production, housing starts, "
            "and the federal funds rate. "
            "Returns synthesised investment signals (e.g. 'GDP negative → favour defensives'). "
            "These are LEADING real-economy indicators that typically lead equity markets by "
            "1-2 quarters — use alongside get_macro_environment() (which covers yield curve, "
            "VIX, dollar) for a complete macro picture. "
            "Requires FRED_API_KEY in .env (free at https://fredapi.stlouisfed.org)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_google_trends",
        "description": (
            "Fetch Google search-interest trends for a company's brand or products. "
            "Search interest is a leading indicator of consumer demand — rising interest "
            "4-8 weeks before earnings often predicts revenue beats for consumer-facing companies. "
            "Returns 12-month trend, recent 8-week direction (rising/falling/stable), "
            "and current interest vs 12-month average. "
            "Best used for consumer-facing companies (AAPL, AMZN, NFLX). "
            "For B2B companies, pass specific product keywords (e.g. ['Azure'] for MSFT). "
            "Uses pytrends — no API key required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of specific keywords to track instead of the company name. "
                        "Use product/service names for better signal (e.g. ['iPhone', 'Mac'] for AAPL). "
                        "Max 5 keywords."
                    ),
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_retail_sentiment",
        "description": (
            "Aggregate retail investor sentiment from StockTwits and Reddit "
            "(r/investing, r/wallstreetbets, r/stocks, r/SecurityAnalysis). "
            "Returns StockTwits bull/bear ratio and recent Reddit posts with scoring. "
            "IMPORTANT: Retail sentiment is a CONTRARIAN indicator for long-term investors. "
            "Extreme bullishness (>80% bulls) often precedes corrections; "
            "extreme bearishness (<25% bulls) can mark bottoms. "
            "Use as a sentiment thermometer alongside fundamentals — not as a buy/sell trigger. "
            "No API key required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_rss_news",
        "description": (
            "Aggregate recent news headlines from multiple RSS feeds: "
            "Yahoo Finance, MarketWatch, and Seeking Alpha. "
            "Provides broader coverage than get_stock_news(), surfacing analyst commentary, "
            "earnings previews, sector rotation themes, and M&A / regulatory news. "
            "Use when get_stock_news() returns few results or you want a second opinion on "
            "news coverage. Look for recurring negative themes across multiple sources — "
            "that cross-source agreement is a stronger signal than a single headline. "
            "Requires feedparser (pip install feedparser) — no API key needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    # ── GenAI intelligence tools (SEC EDGAR) ──────────────────────────────────
    {
        "name": "analyze_earnings_call",
        "description": (
            "Fetch the most recent earnings call transcript from SEC EDGAR (8-K filing) "
            "and return it for analysis. Use this to assess management tone, changes in "
            "forward guidance language, analyst Q&A tension points, and topics management "
            "avoided. A confident, specific management tone is bullish; vague or heavily "
            "hedged language often precedes a guidance cut. Call this when you want deeper "
            "qualitative insight beyond EPS beat/miss numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_sec_filing",
        "description": (
            "Fetch and return key sections of the latest 10-K (annual) or 10-Q (quarterly) "
            "report from SEC EDGAR. Returns three high-signal sections: Business overview "
            "(moat description), Risk Factors (management-flagged threats), and MD&A "
            "(management discussion and analysis). "
            "Key signals to look for: new risk factors vs prior year = emerging threats; "
            "MD&A language shifting from confident to hedged = caution ahead; "
            "moat language becoming defensive = competitive pressure. "
            "Use this for deep-dive research on high-conviction candidates or to validate "
            "the thesis on existing holdings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "form_type": {
                    "type": "string",
                    "description": "Filing type: '10-K' for annual report or '10-Q' for quarterly",
                    "enum": ["10-K", "10-Q"],
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_material_events",
        "description": (
            "Fetch recent SEC 8-K material event filings for a stock. Companies must file "
            "an 8-K within 4 business days of any material event — making this a real-time "
            "signal source for: CEO/CFO departures (Item 5.02), M&A activity (Item 2.01), "
            "asset impairments (Item 2.06), auditor changes (Item 4.01), restatements "
            "(Item 4.02), and bankruptcy (Item 1.03). "
            "Use this to catch thesis-breaking events between quarterly earnings calls. "
            "A CFO exit is typically a stronger warning signal than a CEO exit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days back to search for 8-K filings (default 90)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_competitor_analysis",
        "description": (
            "Identify the stock's closest S&P 500 peers by sector and return a "
            "side-by-side fundamental comparison (PEG, FCF yield, margins, ROE, momentum). "
            "Use this to determine whether a valuation premium is justified vs actual "
            "competitors — not just the broad market. "
            "Key questions: Is the subject's PEG above or below the peer median? "
            "Does its revenue growth rate justify a higher multiple? "
            "Stocks ranking in the top quartile on both quality AND valuation vs peers "
            "have the strongest long-term outperformance record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to analyse vs its sector peers",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_superinvestor_positions",
        "description": (
            "Check whether prominent long-term value investors hold this stock based on "
            "their latest SEC 13F-HR filings. Investors tracked: Buffett (Berkshire), "
            "Ackman (Pershing Square), Tepper (Appaloosa), Halvorsen (Viking Global), "
            "Druckenmiller (Duquesne), Loeb (Third Point), Einhorn (Greenlight). "
            "Multiple superinvestors converging on the same position is a strong "
            "independent confirmation signal. Note: 13F filings have up to a 45-day lag "
            "after quarter-end, so positions may have changed since the filing date. "
            "The absence of smart money is not a negative signal on its own."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    # ── Multi-agent parallel research ─────────────────────────────────────────
    {
        "name": "research_stocks_parallel",
        "description": (
            "Launch parallel research subagents to deep-dive multiple stocks simultaneously. "
            "Each subagent independently researches one ticker — running all 15 research tools "
            "(fundamentals, earnings call, SEC filings, insider activity, competitor analysis, "
            "superinvestor positions, sentiment, material events, etc.) — and returns a "
            "structured JSON report with a recommendation (buy/watchlist/pass), "
            "conviction score 1-10, key positives, key risks, and a full thesis. "
            "\n\n"
            "Use this in Step 4 instead of researching finalists one-by-one. "
            "Pass 3-6 tickers from your screener results. All reports arrive at once, "
            "already sorted by conviction score. The coordinator then decides which to "
            "buy, watchlist, or shadow-record based on the synthesised reports. "
            "\n\n"
            "Provide 'context' with the current macro regime, sector exposure, and "
            "available cash — subagents use this to calibrate recommendations. "
            "Each screener_data dict (from screen_stocks) should include the full "
            "screener row for that ticker so subagents can see the screener signals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers_with_data": {
                    "type": "array",
                    "description": (
                        "List of stocks to research in parallel. Each item must have 'ticker' "
                        "and optionally 'screener_data' (the screen_stocks result row for that ticker)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol",
                            },
                            "screener_data": {
                                "type": "object",
                                "description": "Screener metrics from screen_stocks for this ticker",
                            },
                        },
                        "required": ["ticker"],
                    },
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Portfolio context for the subagents: current macro regime, sector exposure "
                        "weights, available cash, and any signals from get_ml_factor_weights or "
                        "get_signal_performance that should bias the research. "
                        "Example: Macro RISK_OFF, yield curve inverted. Tech 35%, Health Care 18% overweight. "
                        "Cash $145k. Require PEG < 1.5 and positive FCF yield."
                    ),
                },
            },
            "required": ["tickers_with_data", "context"],
        },
    },
    # ── Bear case adversarial challenge ───────────────────────────────────────
    {
        "name": "challenge_buy_theses",
        "description": (
            "Launch adversarial bear case subagents to challenge bull buy recommendations "
            "before committing capital. For each stock the research agent recommended 'buy', "
            "a separate bear case subagent is given the full bull report and tasked with "
            "finding every flaw: overestimated moat, valuation errors, missed risks, "
            "accounting red flags, competitive threats, and macro sensitivity.\n\n"
            "Call this AFTER research_stocks_parallel, passing in the reports it returned. "
            "Only 'buy'-rated reports are challenged; others pass through unchanged.\n\n"
            "Each bear report returns:\n"
            "  - verdict: 'proceed' (bull thesis holds), 'caution' (real issues found), "
            "    or 'reject' (fundamental flaw — do not buy)\n"
            "  - bear_conviction: 1-10 (how strongly the bear argues against buying)\n"
            "  - key_objections: specific flaws found\n"
            "  - risks_missed_by_bull: risks the bull report glossed over\n"
            "  - recommended_action: final call after weighing both sides\n\n"
            "Decision rule:\n"
            "  - verdict='proceed' → buy is confirmed, proceed with normal position sizing\n"
            "  - verdict='caution' → consider half-size position or watchlist pending resolution\n"
            "  - verdict='reject' → do NOT buy; shadow-record instead\n\n"
            "Provide 'context' with the current macro regime and sector weights so the bear "
            "agent can assess macro sensitivity of each thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bull_reports": {
                    "type": "array",
                    "description": (
                        "List of research report dicts returned by research_stocks_parallel. "
                        "Pass the full list — non-buy recommendations are skipped automatically."
                    ),
                    "items": {"type": "object"},
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Portfolio context: macro regime, sector weights, cash available, "
                        "and any signals that should inform the bear's macro sensitivity check. "
                        "Example: Macro RISK_OFF, yield curve inverted. Tech 35% of portfolio. "
                        "Cash $145k. VIX elevated at 28."
                    ),
                },
            },
            "required": ["bull_reports", "context"],
        },
    },
    {
        "name": "check_concentration_limits",
        "description": (
            "Check if a proposed buy order would breach portfolio concentration limits. "
            "Hard limits: max 10% in any single position, max 30% in any one sector. "
            "Returns whether the buy is allowed, any violations, and the max_allowed_buy "
            "amount that stays within limits. "
            "MUST be called before every buy_stock execution. If not allowed, either reduce "
            "the buy amount to max_allowed_buy or skip the trade."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "sector": {
                    "type": "string",
                    "description": "Sector of the stock (e.g. Technology, Healthcare)",
                },
                "buy_amount": {
                    "type": "number",
                    "description": "Proposed dollar amount to invest",
                },
            },
            "required": ["ticker", "sector", "buy_amount"],
        },
    },
    {
        "name": "log_prediction",
        "description": (
            "Log an investment decision for later reconciliation. Call this AFTER every "
            "buy, watchlist, or pass decision with the conviction score, intrinsic value, "
            "and current price. After 90 days, reconcile_predictions will compare the "
            "prediction against actual outcomes to measure the agent's accuracy over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "action": {"type": "string", "enum": ["buy", "watchlist", "pass"]},
                "conviction_score": {"type": "integer"},
                "predicted_iv": {"type": "number", "description": "Intrinsic value from DCF"},
                "price_at_decision": {"type": "number"},
                "mos_pct": {"type": "number", "description": "Margin of safety %"},
            },
            "required": ["ticker", "action"],
        },
    },
    {
        "name": "reconcile_predictions",
        "description": (
            "Reconcile past predictions against actual outcomes. Finds all decisions made "
            ">90 days ago that haven't been reviewed, fetches current prices, and updates "
            "the record with actual return and alpha vs SPY. Call this quarterly to measure "
            "prediction accuracy and identify systematic biases."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_news_alerts",
        "description": (
            "Scan recent news headlines for held positions and flag material events "
            "that may warrant unscheduled re-research: CEO/CFO departures, accounting "
            "restatements, M&A activity, bankruptcy risk, guidance cuts, regulatory actions. "
            "Call alongside check_fundamental_deterioration in Step 2. "
            "Any flagged holding should be re-researched before the next scheduled run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {"type": "array", "items": {"type": "object"},
                            "description": "List of holdings with ticker key"}
            },
            "required": ["holdings"],
        },
    },
    {
        "name": "query_kb",
        "description": (
            "Search the investment knowledge base for relevant notes, frameworks, and company insights. "
            "Use this when you need specific knowledge about: a company (e.g. 'Sygnity TSS sidecar thesis'), "
            "a valuation methodology (e.g. 'FCF normalisation'), a failure mode (e.g. 'peak earnings trap'), "
            "or a sector framework (e.g. 'serial acquirer red flags'). "
            "Returns the top matching KB entries ranked by relevance. "
            "More efficient than re-deriving from first principles — check the KB first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search query, e.g. 'VMS serial acquirer reinvestment rate'",
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Optional topic filter: vms_playbook | iv_methodology | failure_mode | "
                        "framework | company | sector | lesson"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum entries to return (default 3, max 10)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_kb_note",
        "description": (
            "Save a new note to the investment knowledge base, or update an existing one. "
            "Use this to persist insights discovered during research that would be valuable "
            "in future sessions: company-specific observations, updated theses, sector patterns, "
            "or lessons from mistakes. Notes are searchable via query_kb. "
            "If a note with the same title already exists, it will be updated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "Category: vms_playbook | iv_methodology | failure_mode | "
                        "framework | company | sector | lesson"
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Short title for the note (used for deduplication), e.g. 'Topicus Q4 2025 update'",
                },
                "content": {
                    "type": "string",
                    "description": "The knowledge content to store",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tags for retrieval: ticker symbols, company names, keywords",
                },
            },
            "required": ["topic", "title", "content"],
        },
    },
    {
        "name": "run_iv_postmortem",
        "description": (
            "Analyse past IV (intrinsic value) predictions vs actual outcomes. "
            "For each reconciled prediction with a predicted_iv, computes iv_accuracy_pct = "
            "(outcome_price / predicted_iv - 1) * 100. Groups by conviction bucket (5-6, 7-8, 9-10) "
            "and by sector. Saves calibration insights to the knowledge base automatically. "
            "Call this quarterly alongside reconcile_predictions to refine your IV methodology. "
            "Returns: total_analysed, conviction_calibration, sector_calibration, insights, kb_notes_saved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_conviction_calibration",
        "description": (
            "Analyse whether conviction scores actually predict returns in this portfolio. "
            "Groups reconciled predictions by conviction bucket (low 5-6, medium 7-8, high 9-10) "
            "and computes avg_return, win_rate, avg_alpha per bucket. "
            "Diagnoses whether high conviction (9-10) is genuinely more accurate than medium (7-8). "
            "Returns calibration_status: 'well_calibrated', 'miscalibrated_high', or 'insufficient_data'. "
            "Use this to validate position sizing multipliers — if miscalibrated, reduce high-conviction sizing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_decision_thresholds",
        "description": (
            "Return the permanent decision thresholds for the buy/watchlist/pass matrix. "
            "CALL THIS ONCE at the start of Step 5 before acting on any research report. "
            "Returns mos_threshold_pct (regime-adjusted, permanent — 20% NORMAL to 28% STAGFLATION), "
            "bear_override_conviction, a ready-to-use decision_matrix, and ml_factor_guidance "
            "showing which screener signals have predicted returns in this portfolio. "
            "The MoS threshold is a permanent discipline rule for a 20+ year IV portfolio — "
            "it does not change based on trade history. ML only influences candidate ranking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_earnings_tone_trend",
        "description": (
            "Get the earnings call sentiment trend for a ticker over the last 3-4 quarters. "
            "Returns tone direction (positive/negative/neutral) per quarter plus a delta signal. "
            "Use this to detect management confidence deterioration before it shows in financials. "
            "A deteriorating trend (delta < -0.1) is an early warning sign that warrants thesis review."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_market_iv_context",
        "description": (
            "Aggregate recent research cache to quantify market-level valuation. "
            "Answers: 'What % of the researched universe currently offers ≥20% margin of safety?' "
            "High % = cheap market, easier to find opportunities. "
            "Low % = expensive market, require higher quality bar. "
            "Returns: at_mos_pct, avg_mos_pct, market_signal ('cheap'/'fair'/'expensive'), sector_breakdown. "
            "Call this in Step 1 to calibrate the opportunity bar for the session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "detect_financial_anomalies",
        "description": (
            "Detect statistical anomalies in a company's financial ratios by computing z-scores vs "
            "its own 5-year history and sector peers. Use this to catch peak earnings traps "
            "(margins at multi-year highs) or potential dislocations (margins at multi-year lows). "
            "Call before finalising your intrinsic value estimate. "
            "Returns: anomalies (temporal z-scores), sector_anomalies (cross-sectional), "
            "overall_flag ('clean'/'watch'/'flagged'), summary. "
            "If overall_flag is 'flagged', incorporate the interpretations into your normalisation assumptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "screener_data": {
                    "type": "object",
                    "description": "Optional screener row for cross-sectional peer comparison (profit_margin_pct, revenue_growth_pct, roe_pct)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "save_session_reflection",
        "description": (
            "Save a structured reflection at the end of your portfolio review session. "
            "Always call this at the end of each review. "
            "Use exactly this markdown template:\n\n"
            "## Actions Taken\n"
            "[Buys, sells, watchlist additions, shadow records — with one-line rationale each]\n\n"
            "## Thesis Validation\n"
            "[For each holding: is the original thesis playing out? Any cracks?]\n\n"
            "## Signal Performance\n"
            "[Which signals (PEG, FCF, momentum) appear to be working this session? "
            "Any surprises from get_signal_performance?]\n\n"
            "## Macro Observations\n"
            "[Rate environment, dollar, VIX, sector rotation — what the regime is telling you]\n\n"
            "## Shadow Portfolio Review\n"
            "[Any passed-on stocks that moved significantly? Was the pass correct?]\n\n"
            "## Lessons for Next Session\n"
            "[2-4 specific, actionable rules to apply next time]"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reflection": {
                    "type": "string",
                    "description": "Structured reflection using the template above.",
                }
            },
            "required": ["reflection"],
        },
    },
    {
        "name": "get_stored_thesis",
        "description": (
            "Retrieve the stored investment thesis for a ticker from previous research sessions. "
            "Call this when re-researching a watchlist or portfolio stock to check consistency "
            "with your original reasoning. Returns original moat assessment, IV estimate, and "
            "key thesis points so you can explicitly note what changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "check_portfolio_correlation",
        "description": (
            "Compute 1-year price correlation between a candidate stock and all current portfolio "
            "holdings. Returns a warning + sizing adjustment if the candidate is highly correlated "
            "with existing positions (avg correlation > 0.6 or any pair > 0.7). Call before "
            "buy_stock to avoid accidentally concentrating factor exposure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol of the candidate to check",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "check_watchlist_staleness",
        "description": (
            "Return watchlist entries that haven't been re-evaluated in 60+ days. These entries "
            "may have hit their target price (missed entry), had their thesis invalidated, or "
            "represent opportunities that need fresh research. Use this at the start of a session "
            "to prioritise re-evaluation of aging watchlist entries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_age_days": {
                    "type": "integer",
                    "description": "Minimum age in days to consider stale (default 60)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "save_management_profile",
        "description": (
            "Save a management quality assessment to the knowledge base for future sessions. "
            "Call this after completing the capital allocation / management quality step of "
            "research. The profile will be retrieved next time this company is researched, "
            "avoiding redundant re-derivation of management quality from scratch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (required)",
                },
                "company_name": {
                    "type": "string",
                    "description": "Company name",
                },
                "quality_score": {
                    "type": "integer",
                    "description": "Management quality score 1-10 (required)",
                },
                "capital_allocation_rating": {
                    "type": "string",
                    "description": "Capital allocation rating: excellent|good|average|poor",
                    "enum": ["excellent", "good", "average", "poor"],
                },
                "key_green_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of positive management qualities",
                },
                "key_red_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of negative management qualities or concerns",
                },
                "summary": {
                    "type": "string",
                    "description": "2-4 sentence management assessment (required)",
                },
            },
            "required": ["ticker", "quality_score", "summary"],
        },
    },
    {
        "name": "get_behaviour_summary",
        "description": (
            "Load agent behaviour patterns across recent sessions. "
            "Call at session start alongside get_investment_memory. "
            "Returns averages for: tickers screened/researched, buys per session, "
            "re_researched_watchlist (wasted research budget), deviated_from_matrix "
            "(rules not followed), duplicate_tool_calls, and recent workflow improvement "
            "suggestions logged by the agent itself. "
            "Compare your current session behaviour against these averages in real time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_sessions": {
                    "type": "integer",
                    "description": "Number of past sessions to include (default 10)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "log_workflow_issue",
        "description": (
            "Log a workflow inefficiency you notice during the session. "
            "Call immediately when you spot something suboptimal — redundant tool calls, "
            "a missing step, an instruction that produced unexpected results, a decision "
            "you made that deviated from the rules and why. "
            "These accumulate across sessions and surface in get_behaviour_summary so "
            "patterns can be identified and the workflow improved. "
            "severity: 'low' (minor friction), 'medium' (recurring waste), 'high' (caused wrong decision)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "issue": {
                    "type": "string",
                    "description": "What the inefficiency or problem was",
                },
                "suggestion": {
                    "type": "string",
                    "description": "Concrete suggestion to fix it in future sessions",
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Impact level: low=minor, medium=recurring waste, high=caused wrong decision",
                },
            },
            "required": ["issue", "suggestion"],
        },
    },
    {
        "name": "save_session_audit",
        "description": (
            "Save structured self-audit metrics at the end of every session. "
            "Call this BEFORE save_session_reflection as part of the Step 6 self-audit. "
            "Records objective counts and boolean flags that accumulate into behaviour patterns "
            "visible via get_behaviour_summary in future sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers_screened":           {"type": "integer", "description": "Total tickers passed to screen_stocks"},
                "tickers_researched":         {"type": "integer", "description": "Tickers sent to research_stocks_parallel"},
                "buys_made":                  {"type": "integer", "description": "Positions opened this session"},
                "watchlist_added":            {"type": "integer", "description": "Tickers added to watchlist"},
                "shadow_added":               {"type": "integer", "description": "Tickers added to shadow portfolio"},
                "workflow_issues_logged":     {"type": "integer", "description": "Times log_workflow_issue was called"},
                "re_researched_watchlist":    {"type": "integer", "description": "Tickers already on watchlist that were re-researched (should be 0)"},
                "deviated_from_matrix":       {"type": "integer", "description": "Times buy/pass decision deviated from decision matrix (should be 0)"},
                "duplicate_tool_calls":       {"type": "integer", "description": "Times same tool was called twice for same input"},
                "contradicted_prior_session": {"type": "integer", "description": "Times a conclusion contradicted a prior session without new evidence"},
                "audit_notes":                {"type": "string",  "description": "Any additional self-audit observations"},
            },
            "required": [],
        },
    },
    {
        "name": "get_session_efficiency",
        "description": (
            "Get current session efficiency statistics (total tool calls, unique tools used, "
            "duration) and the last 10 sessions' history for comparison. Call near the end of "
            "a session before save_session_reflection to log efficiency and identify waste "
            "patterns over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Optional notes about this session (e.g. why tool count was high)",
                },
                "stocks_researched": {
                    "type": "integer",
                    "description": "Number of stocks researched this session",
                },
                "stocks_bought": {
                    "type": "integer",
                    "description": "Number of stocks bought this session",
                },
                "stocks_watchlisted": {
                    "type": "integer",
                    "description": "Number of stocks added to watchlist this session",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_position_drift",
        "description": (
            "Check if any existing portfolio positions have drifted above concentration limits "
            "through price appreciation. A position bought at 8% can grow to 15% without "
            "triggering the buy-time check. Returns positions > 12% (limit: 10%) and sectors "
            "> 33% (limit: 30%) with trim recommendations. Call at session start alongside "
            "check_fundamental_deterioration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_sector_rotation_signal",
        "description": (
            "Analyse sector distribution across portfolio, recently researched stocks, and the "
            "full universe to detect availability bias (systematically over/under-researching "
            "certain sectors) and portfolio tilts vs. the opportunity set. Use at session start "
            "to ensure the session explores underweight sectors if warranted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ── Tool handlers ──────────────────────────────────────────────────────────────


def _save_call_sentiment(ticker: str, result: Any) -> None:
    """
    Extract and persist earnings call sentiment from the transcript result dict.
    Called automatically after analyze_earnings_call to build tone trend history.
    """
    from datetime import datetime
    from agent.portfolio import save_earnings_sentiment

    if not result or isinstance(result, dict) and result.get("error"):
        return

    # Determine quarter
    quarter = None
    if isinstance(result, dict):
        quarter = result.get("quarter") or result.get("period") or result.get("date")
    if not quarter:
        # Approximate from current date: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
        now = datetime.utcnow()
        q_num = (now.month - 1) // 3 + 1
        quarter = f"Q{q_num}-{now.year}"

    # Stringify entire result for keyword analysis
    text = str(result).lower()
    words = text.split()
    total_words = max(len(words), 1)

    positive_keywords = ["beat", "raised guidance", "strong", "exceeded", "ahead",
                         "accelerating", "growth", "record", "confident", "raised"]
    negative_keywords = ["miss", "missed", "lowered guidance", "headwinds", "slowdown",
                         "challenging", "below", "disappointing", "cut", "reduced"]
    hedging_keywords = ["uncertain", "monitoring", "cautious", "we'll see", "depends", "volatile"]

    pos_count = sum(text.count(kw) for kw in positive_keywords)
    neg_count = sum(text.count(kw) for kw in negative_keywords)

    raw_score = (pos_count - neg_count) / max(total_words / 100, 1)
    sentiment_score = max(-1.0, min(1.0, raw_score))

    if sentiment_score > 0.1:
        tone_direction = "positive"
    elif sentiment_score < -0.1:
        tone_direction = "negative"
    else:
        tone_direction = "neutral"

    # Collect top signals (keywords actually found)
    key_signals = []
    for kw in positive_keywords:
        if kw in text and text.count(kw) >= 2:
            key_signals.append(f"+{kw}")
    for kw in negative_keywords:
        if kw in text and text.count(kw) >= 2:
            key_signals.append(f"-{kw}")
    for kw in hedging_keywords:
        if kw in text:
            key_signals.append(f"~{kw}")
    key_signals = key_signals[:10]

    save_earnings_sentiment(
        ticker=ticker.upper(),
        quarter=str(quarter),
        sentiment_score=round(sentiment_score, 4),
        tone_direction=tone_direction,
        key_signals=key_signals,
        raw_summary=str(result)[:500],
    )

# ── Model tier context ─────────────────────────────────────────────────────────
# Set once per session before running the agentic loop so handle_tool_call
# can forward the right model to research and bear-case subagents.

_CURRENT_MODEL: Optional[str] = None

_MODEL_TIERS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    # ── restored: calculate_intrinsic_value

    {
        "name": "calculate_intrinsic_value",
        "description": (
            "Standardised 3-stage DCF model for a single stock. Produces consistent, "
            "comparable intrinsic value estimates across all sessions and subagents.\n\n"
            "Model parameters (fixed, non-negotiable):\n"
            "  Stage 1 (yr 1-5):  conservative FCF/earnings growth, haircutted by 20%\n"
            "  Stage 2 (yr 6-10): linear fade from Stage 1 rate → 2.5% terminal\n"
            "  Stage 3 (terminal): 2.5% perpetuity growth\n"
            "  Discount rate:     10.0%\n\n"
            "Returns bear/base/bull IV per share and margin of safety at current price. "
            "Use the BASE scenario as the primary IV reference. "
            "Call this for every finalist before making a buy or watchlist decision — "
            "it replaces informal IV estimates with a standardised, auditable number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    # ── restored: check_concentration_limits

    {
        "name": "check_concentration_limits",
        "description": (
            "Check if a proposed buy order would breach portfolio concentration limits. "
            "Hard limits: max 10% in any single position, max 30% in any one sector. "
            "Returns whether the buy is allowed, any violations, and the max_allowed_buy "
            "amount that stays within limits. "
            "MUST be called before every buy_stock execution. If not allowed, either reduce "
            "the buy amount to max_allowed_buy or skip the trade."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "sector": {
                    "type": "string",
                    "description": "Sector of the stock (e.g. Technology, Healthcare)",
                },
                "buy_amount": {
                    "type": "number",
                    "description": "Proposed dollar amount to invest",
                },
            },
            "required": ["ticker", "sector", "buy_amount"],
        },
    },
    # ── restored: check_dividend_payments

    {
        "name": "check_dividend_payments",
        "description": (
            "Check held positions for dividend payments. Returns annual dividend rate, "
            "yield %, estimated annual income per holding, and ex-dividend dates. "
            "Dividends are treated as additional cash income for reinvestment. "
            "Call during portfolio review to account for dividend income in total return calculations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {"type": "array", "items": {"type": "object"},
                            "description": "List of holdings with ticker and shares keys"}
            },
            "required": ["holdings"],
        },
    },
    # ── restored: check_earnings_surprises

    {
        "name": "check_earnings_surprises",
        "description": (
            "Check held positions for recent earnings surprises (actual EPS vs estimate). "
            "Flags significant beats (>15% above estimate) and misses (>15% below). "
            "A large miss on a held position triggers re-research — the original thesis "
            "may be breaking. A large beat may warrant adding to the position. "
            "Call this alongside check_fundamental_deterioration in Step 2."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {"type": "array", "items": {"type": "object"},
                            "description": "List of holdings with at least a ticker key"}
            },
            "required": ["holdings"],
        },
    },
    # ── restored: check_fundamental_deterioration

    {
        "name": "check_fundamental_deterioration",
        "description": (
            "Check all held positions for fundamental deterioration that may warrant exit. "
            "Flags: revenue declining YoY, FCF turned negative, gross margin < 20%, "
            "high leverage (D/E > 2x), ROE < 8%, earnings declining (fwd PE >> trailing PE). "
            "Returns severity ratings: WATCH (1 flag), REVIEW (2 flags), EXIT (3+ flags). "
            "Call this at the start of each session alongside portfolio status. "
            "This is a long-term portfolio — exits are fundamentals-driven, not price-driven."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "description": "List of held positions, each with at least a 'ticker' key",
                    "items": {"type": "object"},
                }
            },
            "required": ["holdings"],
        },
    },
    # ── restored: check_portfolio_correlation

    {
        "name": "check_portfolio_correlation",
        "description": (
            "Compute 1-year price correlation between a candidate stock and all current portfolio "
            "holdings. Returns a warning + sizing adjustment if the candidate is highly correlated "
            "with existing positions (avg correlation > 0.6 or any pair > 0.7). Call before "
            "buy_stock to avoid accidentally concentrating factor exposure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol of the candidate to check",
                }
            },
            "required": ["ticker"],
        },
    },
    # ── restored: check_position_drift

    {
        "name": "check_position_drift",
        "description": (
            "Check if any existing portfolio positions have drifted above concentration limits "
            "through price appreciation. A position bought at 8% can grow to 15% without "
            "triggering the buy-time check. Returns positions > 12% (limit: 10%) and sectors "
            "> 33% (limit: 30%) with trim recommendations. Call at session start alongside "
            "check_fundamental_deterioration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ── restored: check_watchlist_staleness

    {
        "name": "check_watchlist_staleness",
        "description": (
            "Return watchlist entries that haven't been re-evaluated in 60+ days. These entries "
            "may have hit their target price (missed entry), had their thesis invalidated, or "
            "represent opportunities that need fresh research. Use this at the start of a session "
            "to prioritise re-evaluation of aging watchlist entries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_age_days": {
                    "type": "integer",
                    "description": "Minimum age in days to consider stale (default 60)",
                }
            },
            "required": [],
        },
    },
    # ── restored: check_watchlist_triggers

    {
        "name": "check_watchlist_triggers",
        "description": (
            "Fetch live prices for every watchlist item and compare against target entry prices.\n\n"
            "Returns four buckets:\n"
            "  TRIGGERED   — price is AT or BELOW the target. Run deep research immediately.\n"
            "  APPROACHING — price is within 10% above target. Watch closely this session.\n"
            "  WAITING     — price is more than 10% above target. No action needed.\n"
            "  NO_TARGET   — no target price set; current price reported for reference.\n\n"
            "Call this at the start of every session BEFORE the universe screen. "
            "If any items are TRIGGERED, prioritise them for deep research in Step 4 — "
            "they may already be actionable without needing to screen the full universe."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── restored: detect_financial_anomalies

    {
        "name": "detect_financial_anomalies",
        "description": (
            "Detect statistical anomalies in a company's financial ratios by computing z-scores vs "
            "its own 5-year history and sector peers. Use this to catch peak earnings traps "
            "(margins at multi-year highs) or potential dislocations (margins at multi-year lows). "
            "Call before finalising your intrinsic value estimate. "
            "Returns: anomalies (temporal z-scores), sector_anomalies (cross-sectional), "
            "overall_flag ('clean'/'watch'/'flagged'), summary. "
            "If overall_flag is 'flagged', incorporate the interpretations into your normalisation assumptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "screener_data": {
                    "type": "object",
                    "description": "Optional screener row for cross-sectional peer comparison (profit_margin_pct, revenue_growth_pct, roe_pct)",
                },
            },
            "required": ["ticker"],
        },
    },
    # ── restored: discover_universe_parallel

    {
        "name": "discover_universe_parallel",
        "description": (
            "Two-layer universe screen. Uses a cached quality score (stable fundamentals: "
            "revenue growth, margins, ROE, debt) to shortlist the top 150 names from ~700 tickers, "
            "then fetches fresh valuation metrics (FCF yield, PEG, momentum) for those 150 only. "
            "Returns 'top_candidates': top 60 globally ranked by combined quality+valuation score.\n\n"
            "Each candidate includes quality_score, valuation_score, combined_score, and a "
            "'universe' field ('us_sp500' or 'international'). "
            "Pass top_candidates directly to research_stocks_parallel — no screening calls needed.\n\n"
            "If the quality cache is empty it auto-runs a full refresh (~10 min first time). "
            "Call refresh_universe_scores to manually rebuild the cache (do quarterly or when "
            "quality_cache_age_days > 90)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ── restored: get_conviction_calibration

    {
        "name": "get_conviction_calibration",
        "description": (
            "Analyse whether conviction scores actually predict returns in this portfolio. "
            "Groups reconciled predictions by conviction bucket (low 5-6, medium 7-8, high 9-10) "
            "and computes avg_return, win_rate, avg_alpha per bucket. "
            "Diagnoses whether high conviction (9-10) is genuinely more accurate than medium (7-8). "
            "Returns calibration_status: 'well_calibrated', 'miscalibrated_high', or 'insufficient_data'. "
            "Use this to validate position sizing multipliers — if miscalibrated, reduce high-conviction sizing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_decision_thresholds",
        "description": (
            "Return the permanent decision thresholds for the buy/watchlist/pass matrix. "
            "CALL THIS ONCE at the start of Step 5 before acting on any research report. "
            "Returns mos_threshold_pct (regime-adjusted, permanent — 20% NORMAL to 28% STAGFLATION), "
            "bear_override_conviction, a ready-to-use decision_matrix, and ml_factor_guidance "
            "showing which screener signals have predicted returns in this portfolio. "
            "The MoS threshold is a permanent discipline rule for a 20+ year IV portfolio — "
            "it does not change based on trade history. ML only influences candidate ranking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ── restored: get_conviction_position_size

    {
        "name": "get_conviction_position_size",
        "description": (
            "Calculate the appropriate position size for a stock given its conviction score "
            "and the current macro regime. Call this BEFORE placing any buy order to determine "
            "how many dollars to allocate. Conviction 9-10 = full position, 7-8 = 75%, "
            "5-6 = 50%. Regime sets the base size (RISK_ON=15%, NORMAL=12%, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conviction_score": {"type": "integer", "description": "Conviction score 1-10 from the research report"},
                "regime": {"type": "string", "description": "Current macro regime (RISK_ON/NORMAL/INFLATIONARY/RISK_OFF/STAGFLATION)"},
                "portfolio_equity": {"type": "number", "description": "Total portfolio equity value in dollars (cash + holdings market value)"},
            },
            "required": ["conviction_score", "regime", "portfolio_equity"],
        },
    },
    # ── restored: get_earnings_tone_trend

    {
        "name": "get_earnings_tone_trend",
        "description": (
            "Get the earnings call sentiment trend for a ticker over the last 3-4 quarters. "
            "Returns tone direction (positive/negative/neutral) per quarter plus a delta signal. "
            "Use this to detect management confidence deterioration before it shows in financials. "
            "A deteriorating trend (delta < -0.1) is an early warning sign that warrants thesis review."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT",
                }
            },
            "required": ["ticker"],
        },
    },
    # ── restored: get_market_iv_context

    {
        "name": "get_market_iv_context",
        "description": (
            "Aggregate recent research cache to quantify market-level valuation. "
            "Answers: 'What % of the researched universe currently offers ≥20% margin of safety?' "
            "High % = cheap market, easier to find opportunities. "
            "Low % = expensive market, require higher quality bar. "
            "Returns: at_mos_pct, avg_mos_pct, market_signal ('cheap'/'fair'/'expensive'), sector_breakdown. "
            "Call this in Step 1 to calibrate the opportunity bar for the session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ── restored: get_regime_change_status

    {
        "name": "get_regime_change_status",
        "description": (
            "Detect the current macro regime and compare it to the last recorded regime. "
            "Returns whether the regime has changed since the previous run, how many days ago "
            "the last regime was recorded, and a human-readable change summary. "
            "Call this at the start of every session to check for macro shifts that may require "
            "portfolio rebalancing — a regime change from RISK_ON to RISK_OFF, for example, "
            "should trigger a full portfolio review even mid-cycle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ── restored: get_sector_rotation_signal

    {
        "name": "get_sector_rotation_signal",
        "description": (
            "Analyse sector distribution across portfolio, recently researched stocks, and the "
            "full universe to detect availability bias (systematically over/under-researching "
            "certain sectors) and portfolio tilts vs. the opportunity set. Use at session start "
            "to ensure the session explores underweight sectors if warranted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ── restored: get_session_efficiency

    {
        "name": "get_session_efficiency",
        "description": (
            "Get current session efficiency statistics (total tool calls, unique tools used, "
            "duration) and the last 10 sessions' history for comparison. Call near the end of "
            "a session before save_session_reflection to log efficiency and identify waste "
            "patterns over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Optional notes about this session (e.g. why tool count was high)",
                },
                "stocks_researched": {
                    "type": "integer",
                    "description": "Number of stocks researched this session",
                },
                "stocks_bought": {
                    "type": "integer",
                    "description": "Number of stocks bought this session",
                },
                "stocks_watchlisted": {
                    "type": "integer",
                    "description": "Number of stocks added to watchlist this session",
                },
            },
            "required": [],
        },
    },
    # ── restored: get_stored_thesis

    {
        "name": "get_stored_thesis",
        "description": (
            "Retrieve the stored investment thesis for a ticker from previous research sessions. "
            "Call this when re-researching a watchlist or portfolio stock to check consistency "
            "with your original reasoning. Returns original moat assessment, IV estimate, and "
            "key thesis points so you can explicitly note what changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    # ── restored: get_watchlist_earnings

    {
        "name": "get_watchlist_earnings",
        "description": (
            "Fetch upcoming earnings dates for every watchlist item and bucket by urgency:\n"
            "  IMMINENT — earnings within 7 days: research the stock NOW, before results land\n"
            "  UPCOMING — earnings within 30 days: prepare thesis, set target entry\n"
            "  DISTANT  — earnings >30 days away: no immediate action needed\n\n"
            "Call this in Step 1 every session alongside check_watchlist_triggers. "
            "IMMINENT items must be reviewed in Step 4 even if their price hasn't hit target — "
            "earnings can create sudden entry opportunities or confirm thesis breaks."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── restored: get_watchlist_history

    {
        "name": "get_watchlist_history",
        "description": (
            "Return the history of watchlist lifecycle events: when items were TRIGGERED "
            "(price hit target), APPROACHING, BOUGHT, or REMOVED. "
            "Use this periodically to audit IV estimate accuracy — if TSM was triggered at $230 "
            "but never bought and is now $180, that tells you the IV estimate was wrong. "
            "Pass an optional ticker to filter to one stock's history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Optional: filter to a specific ticker's history",
                },
            },
            "required": [],
        },
    },
    # ── restored: log_prediction

    {
        "name": "log_prediction",
        "description": (
            "Log an investment decision for later reconciliation. Call this AFTER every "
            "buy, watchlist, or pass decision with the conviction score, intrinsic value, "
            "and current price. After 90 days, reconcile_predictions will compare the "
            "prediction against actual outcomes to measure the agent's accuracy over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "action": {"type": "string", "enum": ["buy", "watchlist", "pass"]},
                "conviction_score": {"type": "integer"},
                "predicted_iv": {"type": "number", "description": "Intrinsic value from DCF"},
                "price_at_decision": {"type": "number"},
                "mos_pct": {"type": "number", "description": "Margin of safety %"},
            },
            "required": ["ticker", "action"],
        },
    },
    # ── restored: query_kb

    {
        "name": "query_kb",
        "description": (
            "Search the investment knowledge base for relevant notes, frameworks, and company insights. "
            "Use this when you need specific knowledge about: a company (e.g. 'Sygnity TSS sidecar thesis'), "
            "a valuation methodology (e.g. 'FCF normalisation'), a failure mode (e.g. 'peak earnings trap'), "
            "or a sector framework (e.g. 'serial acquirer red flags'). "
            "Returns the top matching KB entries ranked by relevance. "
            "More efficient than re-deriving from first principles — check the KB first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search query, e.g. 'VMS serial acquirer reinvestment rate'",
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Optional topic filter: vms_playbook | iv_methodology | failure_mode | "
                        "framework | company | sector | lesson"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum entries to return (default 3, max 10)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    # ── restored: reconcile_predictions

    {
        "name": "reconcile_predictions",
        "description": (
            "Reconcile past predictions against actual outcomes. Finds all decisions made "
            ">90 days ago that haven't been reviewed, fetches current prices, and updates "
            "the record with actual return and alpha vs SPY. Call this quarterly to measure "
            "prediction accuracy and identify systematic biases."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── restored: run_iv_postmortem

    {
        "name": "run_iv_postmortem",
        "description": (
            "Analyse past IV (intrinsic value) predictions vs actual outcomes. "
            "For each reconciled prediction with a predicted_iv, computes iv_accuracy_pct = "
            "(outcome_price / predicted_iv - 1) * 100. Groups by conviction bucket (5-6, 7-8, 9-10) "
            "and by sector. Saves calibration insights to the knowledge base automatically. "
            "Call this quarterly alongside reconcile_predictions to refine your IV methodology. "
            "Returns: total_analysed, conviction_calibration, sector_calibration, insights, kb_notes_saved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ── restored: save_kb_note

    {
        "name": "save_kb_note",
        "description": (
            "Save a new note to the investment knowledge base, or update an existing one. "
            "Use this to persist insights discovered during research that would be valuable "
            "in future sessions: company-specific observations, updated theses, sector patterns, "
            "or lessons from mistakes. Notes are searchable via query_kb. "
            "If a note with the same title already exists, it will be updated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "Category: vms_playbook | iv_methodology | failure_mode | "
                        "framework | company | sector | lesson"
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Short title for the note (used for deduplication), e.g. 'Topicus Q4 2025 update'",
                },
                "content": {
                    "type": "string",
                    "description": "The knowledge content to store",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tags for retrieval: ticker symbols, company names, keywords",
                },
            },
            "required": ["topic", "title", "content"],
        },
    },
    # ── restored: save_management_profile

    {
        "name": "save_management_profile",
        "description": (
            "Save a management quality assessment to the knowledge base for future sessions. "
            "Call this after completing the capital allocation / management quality step of "
            "research. The profile will be retrieved next time this company is researched, "
            "avoiding redundant re-derivation of management quality from scratch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (required)",
                },
                "company_name": {
                    "type": "string",
                    "description": "Company name",
                },
                "quality_score": {
                    "type": "integer",
                    "description": "Management quality score 1-10 (required)",
                },
                "capital_allocation_rating": {
                    "type": "string",
                    "description": "Capital allocation rating: excellent|good|average|poor",
                    "enum": ["excellent", "good", "average", "poor"],
                },
                "key_green_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of positive management qualities",
                },
                "key_red_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of negative management qualities or concerns",
                },
                "summary": {
                    "type": "string",
                    "description": "2-4 sentence management assessment (required)",
                },
            },
            "required": ["ticker", "quality_score", "summary"],
        },
    },
]


def set_model_context(model: str) -> None:
    """Store the coordinator model so subagent dispatchers can use it."""
    global _CURRENT_MODEL
    _CURRENT_MODEL = model


def _bear_case_model(model: str) -> str:
    """
    Return the model to use for bear case subagents.
    Opus stays Opus — bear case is a critical decision gate and shouldn't be
    downgraded when the user explicitly chose the best model.
    Sonnet drops to Haiku (genuine cost save on focused adversarial work).
    Haiku stays Haiku.
    """
    _BEAR_TIER = {
        "claude-opus-4-6":           "claude-opus-4-6",
        "claude-sonnet-4-6":         "claude-haiku-4-5-20251001",
        "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
    }
    return _BEAR_TIER.get(model, model)


def _effective_model() -> Optional[str]:
    """Return the currently set model context, or None (subagent will use its own default)."""
    return _CURRENT_MODEL


# ── Tool schemas for Claude ────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_portfolio_status",
        "description": (
            "Returns the current portfolio: cash balance, all stock holdings with "
            "their shares, average cost, and current unrealized P&L. Use this to "
            "understand what you currently own before making decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_stock_quote",
        "description": (
            "Get the current market price and basic stats for a stock ticker. "
            "Use this to check the current price before buying or selling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT, NVDA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_fundamentals",
        "description": (
            "Get detailed fundamental data for a stock: P/E ratio, profit margins, "
            "ROE, debt levels, dividend yield, analyst ratings, and more. "
            "Use this to evaluate whether a stock is a good long-term investment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_price_history",
        "description": (
            "Get historical price performance for a stock over a given period. "
            "Shows price change %, highs, lows, and average volume."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "period": {
                    "type": "string",
                    "description": "Time period: 1mo, 3mo, 6mo, 1y, 2y, 5y",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": (
            "Compute technical indicators for a stock to assess entry timing and trend health. "
            "Returns RSI-14, MACD (12/26/9), Bollinger Bands (20-day, 2σ), EMA-50/200 "
            "(golden/death cross), volume vs 20-day average, and an overall signal summary.\n\n"
            "Use this AFTER fundamentals confirm a great business at a fair price — "
            "technicals answer 'is now a good entry point?' not 'should I buy this business?'. "
            "A fundamentally sound stock with bearish technicals may warrant waiting for a "
            "better entry; a fundamentally weak stock with bullish technicals is still a pass.\n\n"
            "Key signals to act on:\n"
            "  - RSI < 30 (oversold): stock may be at a near-term bottom — favorable entry\n"
            "  - RSI > 70 (overbought): wait for a pullback before buying\n"
            "  - MACD bullish crossover: momentum turning up — confirms entry timing\n"
            "  - MACD bearish crossover: momentum turning down — wait or reduce size\n"
            "  - Price below EMA-50 and EMA-200 (death cross): stock in downtrend — be cautious\n"
            "  - Price at lower Bollinger Band: near-term oversold, may bounce\n"
            "  - Price at upper Bollinger Band: extended, high risk of short-term pullback\n"
            "  - High volume on up days: institutional buying — supportive of thesis\n\n"
            "No API key required — computed from yfinance 1-year daily history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_short_interest",
        "description": (
            "Retrieve short interest data for a stock: short % of float, days-to-cover "
            "(short ratio), month-over-month change in shares short, and interpreted signals.\n\n"
            "Short interest has two readings — use both:\n"
            "  (A) Institutional bear signal: high short % + rising trend means smart money "
            "has researched this stock and is actively betting against it. Treat as a red flag "
            "that warrants extra scrutiny — what do they know that the bull case missed?\n"
            "  (B) Squeeze catalyst: high short % on a stock with a strong fundamental thesis "
            "and an upcoming catalyst means short covering will amplify any price move higher. "
            "This is upside optionality on top of the thesis — not a standalone buy reason.\n\n"
            "Thresholds:\n"
            "  short_level 'high' (15-25%) or 'very_high' (>25%): investigate short thesis\n"
            "  mom_direction 'rising': bears adding — dig deeper before buying\n"
            "  mom_direction 'falling': short covering — potential near-term tailwind\n"
            "  days_to_cover > 7: elevated squeeze risk if positive catalyst emerges\n\n"
            "Data is sourced from FINRA via yfinance and updated twice monthly — "
            "figures typically lag ~2 weeks. No API key required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_options_flow",
        "description": (
            "Retrieve options market data for a stock: put/call volume and OI ratios, "
            "ATM implied volatility vs. 30-day realized volatility, and unusual contract "
            "activity (large fresh directional bets). Analyzes the nearest 3 expiries.\n\n"
            "Three distinct signals — read each separately:\n\n"
            "1. Put/Call ratio (sentiment):\n"
            "   < 0.7 = bullish (heavy call buying vs. puts)\n"
            "   0.7–1.0 = neutral\n"
            "   > 1.0 = bearish (heavy put buying — hedging or directional short)\n"
            "   > 1.5 = strongly bearish\n\n"
            "2. IV vs. realized volatility (event/fear pricing):\n"
            "   IV >> realized vol: market pricing in an upcoming event or fear — options\n"
            "   are expensive; buying stock outright is more capital-efficient than buying calls\n"
            "   IV ≈ realized vol: normal; no special event risk priced in\n"
            "   IV << realized vol: options unusually cheap — low implied risk\n\n"
            "3. Unusual contracts (volume ≥ 3× open interest, ≥ 500 contracts):\n"
            "   Fresh directional bets by large traders, not hedges or rolls.\n"
            "   Call-skewed unusual activity = bullish smart-money signal\n"
            "   Put-skewed unusual activity = bearish smart-money signal or large hedge\n\n"
            "Use this as a positioning/sentiment check, NOT as a buy/sell trigger. "
            "A fundamentally great stock with bearish options positioning is still a buy — "
            "but unusually heavy put buying or very high IV warrants investigation. "
            "No API key required — data from yfinance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "search_stocks",
        "description": (
            "Search for stocks by company name or keyword. Returns matching tickers. "
            "Use this when you know a company name but not its ticker symbol."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Company name or keyword to search for",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_market_summary",
        "description": (
            "Get a snapshot of major market indices (S&P 500, NASDAQ, Dow Jones, "
            "Russell 2000, VIX). Use this to gauge overall market conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "buy_stock",
        "description": (
            "Execute a paper (simulated) buy order. Purchases shares at the current "
            "market price and deducts from your cash balance. Always call get_stock_quote "
            "first to confirm the current price. Specify either shares OR dollar_amount."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to buy",
                },
                "shares": {
                    "type": "number",
                    "description": "Number of shares to buy (use this OR dollar_amount)",
                },
                "dollar_amount": {
                    "type": "number",
                    "description": "Dollar amount to invest (use this OR shares). Will be converted to shares at current price.",
                },
                "notes": {
                    "type": "string",
                    "description": "Reasoning for this buy decision",
                },
                "screener_snapshot": {
                    "type": "object",
                    "description": (
                        "Optional: pass the screen_stocks result row for this ticker "
                        "(the dict containing score, peg_ratio, relative_momentum_pct, etc.). "
                        "Saves the signal state at purchase time for future performance attribution."
                    ),
                },
            },
            "required": ["ticker", "notes"],
        },
    },
    {
        "name": "sell_stock",
        "description": (
            "Execute a paper (simulated) sell order. Sells shares and adds proceeds "
            "to your cash balance. Specify either shares OR 'all' to sell everything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to sell",
                },
                "shares": {
                    "type": "number",
                    "description": "Number of shares to sell",
                },
                "sell_all": {
                    "type": "boolean",
                    "description": "Set to true to sell all shares of this ticker",
                },
                "notes": {
                    "type": "string",
                    "description": "Reasoning for this sell decision",
                },
            },
            "required": ["ticker", "notes"],
        },
    },
    {
        "name": "get_transaction_history",
        "description": "View recent buy/sell transaction history for the portfolio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent transactions to show (default 20)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_stock_news",
        "description": (
            "Fetch recent news headlines for a stock from Yahoo Finance. "
            "Use this to stay aware of earnings surprises, CEO changes, product launches, "
            "lawsuits, regulatory actions, or any event that could affect the investment thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of articles to return (default 8, max 15)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_earnings_calendar",
        "description": (
            "Get the next scheduled earnings date for a stock, consensus EPS and revenue estimates, "
            "and the last 4 quarters of beat/miss history. "
            "Use this before buying — earnings are the single biggest short-term risk for any position."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_analyst_upgrades",
        "description": (
            "Get recent analyst upgrades and downgrades for a stock: which firm acted, "
            "whether it was an upgrade/downgrade/initiation, and the grade change. "
            "A cluster of downgrades is a warning sign; upgrades can signal improving sentiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of recent analyst actions to return (default 10)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_insider_activity",
        "description": (
            "Get recent insider transactions (buys and sells by executives, directors, and major shareholders). "
            "Significant insider buying — especially by the CEO or CFO — is one of the strongest "
            "bullish signals available. Heavy insider selling can be a warning sign."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_macro_environment",
        "description": (
            "Get key macroeconomic indicators: 10-year and 2-year Treasury yields, yield curve status, "
            "dollar index, oil, gold, and VIX. Includes synthesised signals so you can adjust "
            "sector allocation accordingly (e.g. high rates → favour value over growth, "
            "strong dollar → avoid multinationals, inverted curve → recession risk)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_hedge_recommendations",
        "description": (
            "Translate the current macro regime into concrete defensive ETF hedge recommendations: "
            "which ETFs to buy, how much to allocate, when to enter, and when to unwind.\n\n"
            "Call this during portfolio rebalancing when the macro environment looks stressed. "
            "Triggers: VIX > 25, inverted yield curve, oil > $85, or any RISK_OFF / INFLATIONARY "
            "/ STAGFLATION / HIGH_RATES regime signal from get_macro_environment.\n\n"
            "Hedge universe (plain, non-leveraged, non-inverse ETFs only):\n"
            "  TLT — 20+ Year Treasury Bonds (RISK_OFF flight-to-safety)\n"
            "  IEF — 7-10 Year Treasury Bonds (moderate duration, also HIGH_RATES)\n"
            "  SHV — Short Treasury Bonds <1yr (cash equivalent, earns short-term yield)\n"
            "  GLD — Gold (inflation + crisis hedge; useful in STAGFLATION)\n"
            "  TIP — TIPS Bonds (inflation-protected; INFLATIONARY regime)\n"
            "  GSG — Broad Commodity ETF (energy/metals/agriculture; pure INFLATIONARY)\n\n"
            "Hard rules:\n"
            "  - Hedges are ALWAYS funded from cash, never by selling equity positions\n"
            "  - Maximum hedge allocation: 20% of total portfolio\n"
            "  - No recommendation is made in NORMAL / RISK_ON regimes\n"
            "  - Unwind when the triggering regime resolves\n\n"
            "Pass equity_pct and cash_pct from get_portfolio_status so recommendations "
            "are sized to your actual portfolio composition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "equity_pct": {
                    "type": "number",
                    "description": (
                        "Current equity allocation as % of total portfolio value (0-100). "
                        "From get_portfolio_status: equity_value / (equity_value + cash) × 100."
                    ),
                },
                "cash_pct": {
                    "type": "number",
                    "description": (
                        "Current cash as % of total portfolio value (0-100). "
                        "Hedges will be scaled to fit available cash."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_benchmark_comparison",
        "description": (
            "Compare the portfolio's total return since inception against the S&P 500. "
            "Shows alpha (outperformance or underperformance) and historical snapshots. "
            "Call this during portfolio reviews to understand whether the strategy is "
            "actually beating a simple index fund."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_portfolio_metrics",
        "description": (
            "Return risk and return metrics computed from portfolio snapshot history: "
            "Sharpe ratio, max drawdown, annualised volatility, and rolling 1/3/6-month "
            "returns vs S&P 500. "
            "Call this in Step 2 to understand whether the portfolio is taking too much "
            "risk relative to its returns, and how recent momentum compares to the benchmark."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_investment_memory",
        "description": (
            "Retrieve your past investment theses for current holdings and recently closed positions. "
            "Call this at the start of each session to understand why you made past decisions, "
            "whether original theses are still valid, and what worked or didn't in closed positions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_session_reflections",
        "description": (
            "Retrieve your past post-session reflections — lessons and observations you documented "
            "from previous portfolio reviews. Call this at the start of sessions to apply past learnings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent reflections to retrieve (default 5)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_international_universe",
        "description": (
            "Return a curated list of ~200 major international stocks for screening — "
            "companies NOT already in the US S&P 500 / broad universe.\n\n"
            "Covers top companies from Europe (UK, Germany, France, Switzerland, Netherlands, "
            "Nordics), Asia-Pacific (Japan, South Korea, HK/China, Taiwan, Australia, Singapore), "
            "Latin America (Brazil, Mexico, Chile), Canada, and India/Israel.\n\n"
            "Mix of US-listed ADRs (NYSE/NASDAQ — best data quality, e.g. TSM, ASML, NVO, SAP) "
            "and direct foreign-listed tickers with exchange suffixes (e.g. NESN.SW, 005930.KS, "
            "0700.HK). All accessible via yfinance.\n\n"
            "WORKFLOW: call this once per session, then pass the returned tickers list to "
            "screen_stocks in batches of 50-80. Foreign-suffix tickers that lack data are "
            "silently skipped by screen_stocks.\n\n"
            "Use the optional 'region' parameter to focus on one geography if the macro "
            "environment favours it (e.g. 'europe' during USD weakness, 'asia' for EM exposure)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": (
                        "Optional: narrow to one region. "
                        "Valid: 'europe', 'asia', 'latam', 'canada', 'india'. "
                        "Omit (or pass null) for all regions combined."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_stock_universe",
        "description": (
            "Fetch tickers from major US stock universes. "
            "When index='sp500', returns ALL ~500 S&P 500 constituents (no sampling) — "
            "use this for exhaustive coverage. Optionally filter by GICS sector. "
            "When index='broad' or 'all', returns a random sample of 'sample_n' tickers "
            "from the ~2700-stock universe; call multiple times with different random_seed "
            "values to cover more of the universe."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "description": "Which universe to sample from: 'sp500' (~500 large caps), 'broad' (~2700 US-listed stocks), or 'all' (combined)",
                    "enum": ["sp500", "broad", "all"],
                },
                "sample_n": {
                    "type": "integer",
                    "description": "Number of tickers to return in this sample (default 200, max 300).",
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Seed for random sampling. Use different values (0, 1, 2, 3...) across calls to get different batches.",
                },
                "sector": {
                    "type": "string",
                    "description": (
                        "Optional GICS sector filter. Only returns tickers from this sector. "
                        "Valid values: 'Information Technology', 'Health Care', 'Financials', "
                        "'Consumer Discretionary', 'Communication Services', 'Industrials', "
                        "'Consumer Staples', 'Energy', 'Utilities', 'Real Estate', 'Materials'. "
                        "Sector data is only available for S&P 500 constituents."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "screen_stocks",
        "description": (
            "Run a fast parallel fundamental screen across a list of tickers. "
            "Fetches key metrics (P/E, revenue growth, profit margin, ROE, debt) for each ticker "
            "and returns ALL scored candidates ranked by a composite quality + value score — "
            "no artificial cap on results. "
            "Pass 50-100 tickers at a time (from get_stock_universe) for best results. "
            "Use this to discover high-quality opportunities across the full market — "
            "not just popular large caps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to screen",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_sector_exposure",
        "description": (
            "Show current portfolio weights broken down by GICS sector. "
            "Call this before buying new positions to see where you're already concentrated "
            "and where you have room to add without over-weighting a sector."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_to_watchlist",
        "description": (
            "Add a stock to the watchlist when you like the business but the timing isn't right — "
            "e.g. earnings in the next 2 weeks, valuation slightly too high, or waiting for a "
            "pullback to a target price. The watchlist is reviewed at the start of every session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "company_name": {"type": "string", "description": "Company name (optional)"},
                "reason": {
                    "type": "string",
                    "description": "Why you like this stock and what would trigger a buy (e.g. 'strong FCF, waiting for post-earnings dip below $X')",
                },
                "target_entry_price": {
                    "type": "number",
                    "description": "Optional price at or below which you'd be a buyer",
                },
            },
            "required": ["ticker", "reason"],
        },
    },
    {
        "name": "get_watchlist",
        "description": (
            "Retrieve the current watchlist — stocks you've flagged for future purchase. "
            "Call this at the start of each session to check if any watchlist candidates "
            "have reached their target entry price or had a meaningful pullback."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_from_watchlist",
        "description": (
            "Remove a stock from the watchlist. Call this after buying the stock, "
            "or if the investment thesis has broken down and it's no longer worth watching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol to remove"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_trade_outcomes",
        "description": (
            "Return all past buy signal snapshots with their actual outcomes. "
            "For each recorded buy, shows the screener signals that were present at purchase "
            "(PEG, momentum, FCF yield, etc.) alongside the eventual return. "
            "Use this to identify which signals have historically predicted positive returns "
            "and weight them more heavily in future screening decisions."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_signal_performance",
        "description": (
            "Analyze which screener signals (PEG < 1.5, FCF yield > 3%, positive momentum, "
            "revenue growth > 10%) have historically predicted positive returns in this portfolio. "
            "Returns per-signal statistics split by whether the threshold was met at buy time. "
            "Call this in Step 1 alongside get_trade_outcomes to calibrate signal weights for this session."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_to_shadow_portfolio",
        "description": (
            "Record a stock you analyzed and decided NOT to buy or watchlist. "
            "Use this for stocks that were seriously considered but rejected (too expensive, "
            "weak moat, sector crowded, thesis uncertain). "
            "Tracked so future sessions can review whether passing was the right call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "price_at_consideration": {
                    "type": "number",
                    "description": "Current stock price at the time you considered it",
                },
                "reason_passed": {
                    "type": "string",
                    "description": "Primary reason for passing (e.g. 'overvalued', 'weak FCF', 'sector too heavy', 'thesis unclear')",
                },
                "notes": {
                    "type": "string",
                    "description": "Additional context on the decision",
                },
            },
            "required": ["ticker", "price_at_consideration", "reason_passed"],
        },
    },
    {
        "name": "run_backtest",
        "description": (
            "Run a strategy backtest in one of three modes to validate whether the "
            "screening and trading approach is actually working.\n\n"
            "Modes:\n\n"
            "  'trade_history' — Replays ALL closed trades (no 20-trade limit). "
            "Computes win rate, avg return, Sharpe ratio, max drawdown, and "
            "S&P 500 alpha per trade. Segments results by market regime at entry "
            "(VIX level: low_volatility / normal / elevated / high_fear). "
            "Shows best and worst trades. Call this in Step 1 periodically (every "
            "5+ closed trades) to validate that the strategy is generating alpha.\n\n"
            "  'signal_cohorts' — Breaks all closed trades into signal cohorts: "
            "PEG < 1.5 vs ≥ 1.5, FCF yield > 3% vs ≤ 3%, positive vs negative "
            "momentum, score ≥ 8 vs < 8, and bull entry (VIX < 20) vs bear entry. "
            "Shows win rate and avg return for each cohort so you can see which "
            "signals are actually predictive in YOUR portfolio's specific history. "
            "Use this to update signal weights in get_ml_factor_weights.\n\n"
            "  'momentum' — Simulates buying the top-momentum tercile of a "
            "provided ticker list at a point holding_days ago, and measures "
            "actual forward return vs. S&P 500 buy-and-hold. Uses price data "
            "only — no look-ahead on fundamentals. Pass your current screener "
            "candidates as tickers to validate the momentum signal. "
            "Requires tickers list.\n\n"
            "  'fundamental_history' — Simulates the screener running at every "
            "quarter-end since start_year (default 2015) using real historical "
            "fundamentals from FMP. For each quarter where a stock scores ≥ "
            "score_threshold, records a simulated buy and measures 1q/2q/4q "
            "forward returns vs S&P 500. Answers: would this screening approach "
            "have generated alpha over 10 years? Requires FMP_API_KEY. Uses up "
            "to max_tickers (default 20) from universe_scores or a passed list. "
            "Run this once after building a universe to validate the strategy.\n\n"
            "Call 'trade_history' and 'signal_cohorts' in Step 1 once you have "
            "≥5 closed trades. Call 'momentum' in Step 4 after screening to "
            "validate momentum as a factor before weighting it in decisions. "
            "Call 'fundamental_history' once to validate the strategy historically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["trade_history", "signal_cohorts", "momentum", "fundamental_history"],
                    "description": "Which backtest to run.",
                },
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required for mode='momentum'. Optional for 'fundamental_history' (defaults to universe_scores).",
                },
                "holding_days": {
                    "type": "integer",
                    "description": (
                        "For mode='momentum': how many days ago the simulated entry was. "
                        "Default 90 (one quarter). Use 180 or 365 for longer horizons."
                    ),
                },
                "start_year": {
                    "type": "integer",
                    "description": "For mode='fundamental_history': earliest year to simulate from. Default 2015.",
                },
                "score_threshold": {
                    "type": "number",
                    "description": "For mode='fundamental_history': minimum screener score to count as a buy signal. Default 6.0.",
                },
                "max_tickers": {
                    "type": "integer",
                    "description": "For mode='fundamental_history': max tickers to process (default 20, respects FMP free tier).",
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "get_shadow_performance",
        "description": (
            "Call this in Step 1 to audit past pass decisions — if a rejected stock is up 30%, "
            "understand why you were wrong; if it's down 20%, your thesis was validated. "
            "Use these lessons to sharpen your screening judgment."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── ML insights (learned from portfolio history) ──────────────────────────
    {
        "name": "get_ml_factor_weights",
        "description": (
            "Learn which screener signals have actually predicted returns in THIS portfolio's history. "
            "Returns data-driven factor weights blended with regime-adjusted priors. "
            "Weights are 100% regime-prior when no closed trades exist, then shift smoothly toward "
            "data-driven as closed trades accumulate (25% at 5 trades → 75% at 25+ trades). "
            "Also detects the current macro regime (RISK_ON / RISK_OFF / INFLATIONARY / NORMAL) "
            "and returns feature correlations with returns. "
            "Call this at the start of each session to understand which signals to trust most "
            "when evaluating screener candidates. Use blended_weights to manually re-rank results."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "prioritize_watchlist_ml",
        "description": (
            "Score every watchlist item using ML-derived factor weights and return a ranked list. "
            "For each item: fetches current fundamentals, applies learned factor weights, "
            "computes an ML score (0-10), flags strengths and risk factors, and highlights items "
            "near their target entry price (promoted to top of their score tier). "
            "Call this alongside get_watchlist() to prioritise which candidates deserve the deepest "
            "research this session. Score ≥7: strong candidate — investigate thoroughly. "
            "Score <4: weak fit for current regime — lower priority unless thesis is compelling."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_position_size_recommendation",
        "description": (
            "Estimate drawdown risk for a specific stock and recommend an appropriate position size "
            "(% of portfolio). Combines three inputs: (1) feature-based risk flags on valuation, "
            "FCF, momentum, growth, and profitability; (2) a logistic regression drawdown model "
            "trained on the portfolio's own closed trade history (when ≥5 closed trades exist); "
            "(3) regime-adjusted base size (smaller in RISK_OFF / STAGFLATION). "
            "Call this just before executing a buy to calibrate position size. "
            "Pass the screener_snapshot features from screen_stocks as the 'features' argument. "
            "The recommendation respects the 20% maximum position cap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol being considered for purchase",
                },
                "features": {
                    "type": "object",
                    "description": (
                        "Screener features for this stock — pass the screen_stocks result dict. "
                        "Expected keys: peg_ratio, fcf_yield_pct, relative_momentum_pct, "
                        "revenue_growth_pct, profit_margin_pct, roe_pct."
                    ),
                },
            },
            "required": ["ticker", "features"],
        },
    },
    # ── External data sources ─────────────────────────────────────────────────
    {
        "name": "get_economic_indicators",
        "description": (
            "Fetch key US macroeconomic indicators from the Federal Reserve FRED API: "
            "real GDP growth, CPI inflation, core CPI, unemployment rate, initial jobless "
            "claims, retail sales, consumer sentiment, industrial production, housing starts, "
            "and the federal funds rate. "
            "Returns synthesised investment signals (e.g. 'GDP negative → favour defensives'). "
            "These are LEADING real-economy indicators that typically lead equity markets by "
            "1-2 quarters — use alongside get_macro_environment() (which covers yield curve, "
            "VIX, dollar) for a complete macro picture. "
            "Requires FRED_API_KEY in .env (free at https://fredapi.stlouisfed.org)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_google_trends",
        "description": (
            "Fetch Google search-interest trends for a company's brand or products. "
            "Search interest is a leading indicator of consumer demand — rising interest "
            "4-8 weeks before earnings often predicts revenue beats for consumer-facing companies. "
            "Returns 12-month trend, recent 8-week direction (rising/falling/stable), "
            "and current interest vs 12-month average. "
            "Best used for consumer-facing companies (AAPL, AMZN, NFLX). "
            "For B2B companies, pass specific product keywords (e.g. ['Azure'] for MSFT). "
            "Uses pytrends — no API key required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of specific keywords to track instead of the company name. "
                        "Use product/service names for better signal (e.g. ['iPhone', 'Mac'] for AAPL). "
                        "Max 5 keywords."
                    ),
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_retail_sentiment",
        "description": (
            "Aggregate retail investor sentiment from StockTwits and Reddit "
            "(r/investing, r/wallstreetbets, r/stocks, r/SecurityAnalysis). "
            "Returns StockTwits bull/bear ratio and recent Reddit posts with scoring. "
            "IMPORTANT: Retail sentiment is a CONTRARIAN indicator for long-term investors. "
            "Extreme bullishness (>80% bulls) often precedes corrections; "
            "extreme bearishness (<25% bulls) can mark bottoms. "
            "Use as a sentiment thermometer alongside fundamentals — not as a buy/sell trigger. "
            "No API key required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_rss_news",
        "description": (
            "Aggregate recent news headlines from multiple RSS feeds: "
            "Yahoo Finance, MarketWatch, and Seeking Alpha. "
            "Provides broader coverage than get_stock_news(), surfacing analyst commentary, "
            "earnings previews, sector rotation themes, and M&A / regulatory news. "
            "Use when get_stock_news() returns few results or you want a second opinion on "
            "news coverage. Look for recurring negative themes across multiple sources — "
            "that cross-source agreement is a stronger signal than a single headline. "
            "Requires feedparser (pip install feedparser) — no API key needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    # ── GenAI intelligence tools (SEC EDGAR) ──────────────────────────────────
    {
        "name": "analyze_earnings_call",
        "description": (
            "Fetch the most recent earnings call transcript from SEC EDGAR (8-K filing) "
            "and return it for analysis. Use this to assess management tone, changes in "
            "forward guidance language, analyst Q&A tension points, and topics management "
            "avoided. A confident, specific management tone is bullish; vague or heavily "
            "hedged language often precedes a guidance cut. Call this when you want deeper "
            "qualitative insight beyond EPS beat/miss numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_sec_filing",
        "description": (
            "Fetch and return key sections of the latest 10-K (annual) or 10-Q (quarterly) "
            "report from SEC EDGAR. Returns three high-signal sections: Business overview "
            "(moat description), Risk Factors (management-flagged threats), and MD&A "
            "(management discussion and analysis). "
            "Key signals to look for: new risk factors vs prior year = emerging threats; "
            "MD&A language shifting from confident to hedged = caution ahead; "
            "moat language becoming defensive = competitive pressure. "
            "Use this for deep-dive research on high-conviction candidates or to validate "
            "the thesis on existing holdings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "form_type": {
                    "type": "string",
                    "description": "Filing type: '10-K' for annual report or '10-Q' for quarterly",
                    "enum": ["10-K", "10-Q"],
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_material_events",
        "description": (
            "Fetch recent SEC 8-K material event filings for a stock. Companies must file "
            "an 8-K within 4 business days of any material event — making this a real-time "
            "signal source for: CEO/CFO departures (Item 5.02), M&A activity (Item 2.01), "
            "asset impairments (Item 2.06), auditor changes (Item 4.01), restatements "
            "(Item 4.02), and bankruptcy (Item 1.03). "
            "Use this to catch thesis-breaking events between quarterly earnings calls. "
            "A CFO exit is typically a stronger warning signal than a CEO exit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days back to search for 8-K filings (default 90)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_competitor_analysis",
        "description": (
            "Identify the stock's closest S&P 500 peers by sector and return a "
            "side-by-side fundamental comparison (PEG, FCF yield, margins, ROE, momentum). "
            "Use this to determine whether a valuation premium is justified vs actual "
            "competitors — not just the broad market. "
            "Key questions: Is the subject's PEG above or below the peer median? "
            "Does its revenue growth rate justify a higher multiple? "
            "Stocks ranking in the top quartile on both quality AND valuation vs peers "
            "have the strongest long-term outperformance record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to analyse vs its sector peers",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_superinvestor_positions",
        "description": (
            "Check whether prominent long-term value investors hold this stock based on "
            "their latest SEC 13F-HR filings. Investors tracked: Buffett (Berkshire), "
            "Ackman (Pershing Square), Tepper (Appaloosa), Halvorsen (Viking Global), "
            "Druckenmiller (Duquesne), Loeb (Third Point), Einhorn (Greenlight). "
            "Multiple superinvestors converging on the same position is a strong "
            "independent confirmation signal. Note: 13F filings have up to a 45-day lag "
            "after quarter-end, so positions may have changed since the filing date. "
            "The absence of smart money is not a negative signal on its own."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    # ── Multi-agent parallel research ─────────────────────────────────────────
    {
        "name": "research_stocks_parallel",
        "description": (
            "Launch parallel research subagents to deep-dive multiple stocks simultaneously. "
            "Each subagent independently researches one ticker — running all 15 research tools "
            "(fundamentals, earnings call, SEC filings, insider activity, competitor analysis, "
            "superinvestor positions, sentiment, material events, etc.) — and returns a "
            "structured JSON report with a recommendation (buy/watchlist/pass), "
            "conviction score 1-10, key positives, key risks, and a full thesis. "
            "\n\n"
            "Use this in Step 4 instead of researching finalists one-by-one. "
            "Pass 3-6 tickers from your screener results. All reports arrive at once, "
            "already sorted by conviction score. The coordinator then decides which to "
            "buy, watchlist, or shadow-record based on the synthesised reports. "
            "\n\n"
            "Provide 'context' with the current macro regime, sector exposure, and "
            "available cash — subagents use this to calibrate recommendations. "
            "Each screener_data dict (from screen_stocks) should include the full "
            "screener row for that ticker so subagents can see the screener signals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers_with_data": {
                    "type": "array",
                    "description": (
                        "List of stocks to research in parallel. Each item must have 'ticker' "
                        "and optionally 'screener_data' (the screen_stocks result row for that ticker)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol",
                            },
                            "screener_data": {
                                "type": "object",
                                "description": "Screener metrics from screen_stocks for this ticker",
                            },
                        },
                        "required": ["ticker"],
                    },
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Portfolio context for the subagents: current macro regime, sector exposure "
                        "weights, available cash, and any signals from get_ml_factor_weights or "
                        "get_signal_performance that should bias the research. "
                        "Example: Macro RISK_OFF, yield curve inverted. Tech 35%, Health Care 18% overweight. "
                        "Cash $145k. Require PEG < 1.5 and positive FCF yield."
                    ),
                },
            },
            "required": ["tickers_with_data", "context"],
        },
    },
    # ── Bear case adversarial challenge ───────────────────────────────────────
    {
        "name": "challenge_buy_theses",
        "description": (
            "Launch adversarial bear case subagents to challenge bull buy recommendations "
            "before committing capital. For each stock the research agent recommended 'buy', "
            "a separate bear case subagent is given the full bull report and tasked with "
            "finding every flaw: overestimated moat, valuation errors, missed risks, "
            "accounting red flags, competitive threats, and macro sensitivity.\n\n"
            "Call this AFTER research_stocks_parallel, passing in the reports it returned. "
            "Only 'buy'-rated reports are challenged; others pass through unchanged.\n\n"
            "Each bear report returns:\n"
            "  - verdict: 'proceed' (bull thesis holds), 'caution' (real issues found), "
            "    or 'reject' (fundamental flaw — do not buy)\n"
            "  - bear_conviction: 1-10 (how strongly the bear argues against buying)\n"
            "  - key_objections: specific flaws found\n"
            "  - risks_missed_by_bull: risks the bull report glossed over\n"
            "  - recommended_action: final call after weighing both sides\n\n"
            "Decision rule:\n"
            "  - verdict='proceed' → buy is confirmed, proceed with normal position sizing\n"
            "  - verdict='caution' → consider half-size position or watchlist pending resolution\n"
            "  - verdict='reject' → do NOT buy; shadow-record instead\n\n"
            "Provide 'context' with the current macro regime and sector weights so the bear "
            "agent can assess macro sensitivity of each thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bull_reports": {
                    "type": "array",
                    "description": (
                        "List of research report dicts returned by research_stocks_parallel. "
                        "Pass the full list — non-buy recommendations are skipped automatically."
                    ),
                    "items": {"type": "object"},
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Portfolio context: macro regime, sector weights, cash available, "
                        "and any signals that should inform the bear's macro sensitivity check. "
                        "Example: Macro RISK_OFF, yield curve inverted. Tech 35% of portfolio. "
                        "Cash $145k. VIX elevated at 28."
                    ),
                },
            },
            "required": ["bull_reports", "context"],
        },
    },
    {
        "name": "save_session_reflection",
        "description": (
            "Save a structured reflection at the end of your portfolio review session. "
            "Always call this at the end of each review. "
            "Use exactly this markdown template:\n\n"
            "## Actions Taken\n"
            "[Buys, sells, watchlist additions, shadow records — with one-line rationale each]\n\n"
            "## Thesis Validation\n"
            "[For each holding: is the original thesis playing out? Any cracks?]\n\n"
            "## Signal Performance\n"
            "[Which signals (PEG, FCF, momentum) appear to be working this session? "
            "Any surprises from get_signal_performance?]\n\n"
            "## Macro Observations\n"
            "[Rate environment, dollar, VIX, sector rotation — what the regime is telling you]\n\n"
            "## Shadow Portfolio Review\n"
            "[Any passed-on stocks that moved significantly? Was the pass correct?]\n\n"
            "## Lessons for Next Session\n"
            "[2-4 specific, actionable rules to apply next time]"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reflection": {
                    "type": "string",
                    "description": "Structured reflection using the template above.",
                }
            },
            "required": ["reflection"],
        },
    },
    {
        "name": "get_business_trajectory",
        "description": "Analyze 8-quarter business trajectory trends for gross margin, FCF margin, ROIC, and revenue growth. Returns slope-based trend classification.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "identify_weakest_link",
        "description": "Identify the weakest holding in the portfolio for capital recycling. Scores each position on conviction, IV upside remaining, and research freshness.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_preearnings_briefing",
        "description": "Build a pre-earnings preparation briefing for a stock including thesis confirms/denies checklist, estimates, and beat/miss history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "recalibrate_universe_scores",
        "description": "Re-score the screener universe using ML-learned prediction accuracy weights. Updates quality scores in the database.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_portfolio_stress_test",
        "description": "Run scenario-based stress tests across the full portfolio: AI disruption, rate spike, recession, sector concentration shock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "description": "Scenario name: ai_disruption, rate_spike_200bps, recession_revenue_20pct, sector_concentration_shock, or all",
                    "default": "all",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_triaged_alerts",
        "description": "Get news alerts triaged by severity: thesis_breaking, watch, or noise. Returns actionable alerts first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of tickers to check. If omitted, checks portfolio positions.",
                }
            },
            "required": [],
        },
    },
]


# ── Tool handlers ──────────────────────────────────────────────────────────────

def handle_tool_call(tool_name: str, tool_input: dict) -> Any:
    """Dispatch a tool call and return the result as a JSON-serializable value."""

    if tool_name == "get_portfolio_status":
        return _get_portfolio_status()

    elif tool_name == "get_stock_quote":
        return market_data.get_stock_quote(tool_input["ticker"])

    elif tool_name == "get_stock_fundamentals":
        return market_data.get_stock_fundamentals(tool_input["ticker"])

    elif tool_name == "get_price_history":
        return market_data.get_price_history(
            tool_input["ticker"], tool_input.get("period", "1y")
        )

    elif tool_name == "get_technical_indicators":
        return market_data.get_technical_indicators(tool_input["ticker"])

    elif tool_name == "get_short_interest":
        return market_data.get_short_interest(tool_input["ticker"])

    elif tool_name == "get_options_flow":
        return market_data.get_options_flow(tool_input["ticker"])

    elif tool_name == "search_stocks":
        return market_data.search_stocks(tool_input["query"])

    elif tool_name == "get_market_summary":
        return market_data.get_market_summary()

    elif tool_name == "buy_stock":
        return _handle_buy(tool_input)

    elif tool_name == "sell_stock":
        return _handle_sell(tool_input)

    elif tool_name == "get_transaction_history":
        limit = tool_input.get("limit", 20)
        return portfolio.get_transactions(limit)

    elif tool_name == "get_stock_news":
        limit = min(tool_input.get("limit", 5), 10)
        return market_data.get_stock_news(tool_input["ticker"], limit)

    elif tool_name == "get_earnings_calendar":
        return market_data.get_earnings_calendar(tool_input["ticker"])

    elif tool_name == "get_analyst_upgrades":
        limit = tool_input.get("limit", 10)
        return market_data.get_analyst_upgrades(tool_input["ticker"], limit)

    elif tool_name == "get_insider_activity":
        return market_data.get_insider_activity(tool_input["ticker"])

    elif tool_name == "get_macro_environment":
        return market_data.get_macro_environment()

    elif tool_name == "get_hedge_recommendations":
        # Compute equity/cash % from live portfolio state if not supplied by caller
        equity_pct = tool_input.get("equity_pct")
        cash_pct = tool_input.get("cash_pct")
        if equity_pct is None or cash_pct is None:
            holdings = portfolio.get_holdings()
            cash = portfolio.get_cash()
            equity_value = 0.0
            for h in holdings:
                try:
                    q = market_data.get_stock_quote(h["ticker"])
                    price = q.get("price") or h.get("avg_cost", 0)
                    equity_value += h["shares"] * price
                except Exception:
                    equity_value += h["shares"] * h.get("avg_cost", 0)
            total = equity_value + cash
            if total > 0:
                equity_pct = round(equity_value / total * 100, 1)
                cash_pct = round(cash / total * 100, 1)
        return market_data.get_hedge_recommendations(equity_pct=equity_pct, cash_pct=cash_pct)

    elif tool_name == "get_international_universe":
        return market_data.get_international_universe(region=tool_input.get("region"))

    elif tool_name == "get_stock_universe":
        index = tool_input.get("index", "all")
        sample_n = min(tool_input.get("sample_n", 200), 300)
        random_seed = tool_input.get("random_seed", None)
        sector = tool_input.get("sector", None)
        return market_data.get_stock_universe(index, sample_n=sample_n, random_seed=random_seed, sector=sector)

    elif tool_name == "screen_stocks":
        tickers = tool_input.get("tickers", [])
        return market_data.screen_stocks(tickers)

    elif tool_name == "get_benchmark_comparison":
        return _handle_benchmark_comparison()

    elif tool_name == "get_portfolio_metrics":
        return portfolio.get_portfolio_metrics()

    elif tool_name == "get_sector_exposure":
        holdings = portfolio.get_holdings()
        return market_data.get_sector_exposure(holdings)

    elif tool_name == "add_to_watchlist":
        return portfolio.add_to_watchlist(
            ticker=tool_input["ticker"],
            reason=tool_input["reason"],
            target_entry_price=tool_input.get("target_entry_price"),
            company_name=tool_input.get("company_name"),
        )

    elif tool_name == "get_watchlist":
        return portfolio.get_watchlist()

    elif tool_name == "remove_from_watchlist":
        return portfolio.remove_from_watchlist(tool_input["ticker"])

    elif tool_name == "get_trade_outcomes":
        return portfolio.get_trade_outcomes()

    elif tool_name == "get_investment_memory":
        return portfolio.get_investment_memory()

    elif tool_name == "get_session_reflections":
        limit = tool_input.get("limit", 5)
        return portfolio.get_reflections(limit)

    elif tool_name == "save_session_reflection":
        status = _get_portfolio_status()
        portfolio_value = status.get("total_portfolio_value")
        cash = status.get("cash")
        invested = status.get("total_invested_value")
        portfolio.save_reflection(tool_input["reflection"], portfolio_value, "review")
        # Auto-snapshot for benchmark tracking
        spy = market_data.get_stock_quote("^GSPC")
        spy_price = spy.get("price") if "error" not in spy else None
        portfolio.save_portfolio_snapshot(portfolio_value, cash, invested, spy_price, "review")
        return {"success": True, "message": "Reflection and portfolio snapshot saved."}

    elif tool_name == "get_signal_performance":
        return portfolio.get_signal_performance()

    elif tool_name == "run_backtest":
        from agent.backtest import run_backtest
        return run_backtest(
            mode=tool_input["mode"],
            tickers=tool_input.get("tickers"),
            holding_days=tool_input.get("holding_days", 90),
            start_year=tool_input.get("start_year", 2015),
            score_threshold=tool_input.get("score_threshold", 6.0),
            max_tickers=tool_input.get("max_tickers", 20),
        )

    elif tool_name == "add_to_shadow_portfolio":
        return portfolio.add_to_shadow_portfolio(
            ticker=tool_input["ticker"],
            price_at_consideration=tool_input["price_at_consideration"],
            reason_passed=tool_input["reason_passed"],
            notes=tool_input.get("notes", ""),
        )

    elif tool_name == "get_shadow_performance":
        positions = portfolio.get_shadow_positions()
        if not positions:
            return {"message": "No shadow positions recorded yet.", "positions": []}
        enriched = []
        for p in positions:
            quote = market_data.get_stock_quote(p["ticker"])
            current_price = quote.get("price") if "error" not in quote else None
            entry = dict(p)
            if current_price and p["price_at_consideration"]:
                change_pct = (current_price - p["price_at_consideration"]) / p["price_at_consideration"] * 100
                entry["current_price"] = current_price
                entry["change_since_pass_pct"] = round(change_pct, 2)
                entry["verdict"] = "pass_validated" if change_pct <= 0 else "missed_gain"
            enriched.append(entry)
        return {"positions": enriched}

    elif tool_name == "research_stocks_parallel":
        # Local import to avoid circular dependency (research_agent imports tools)
        from agent.research_agent import research_stocks_parallel
        return research_stocks_parallel(
            tickers_with_data=tool_input["tickers_with_data"],
            context=tool_input.get("context", ""),
            model=_effective_model(),
        )

    elif tool_name == "challenge_buy_theses":
        # Local import to avoid circular dependency (bear_case_agent imports tools)
        from agent.bear_case_agent import challenge_buy_theses
        # Opus stays Opus (critical gate); Sonnet drops to Haiku (cost save)
        _bear_model = _bear_case_model(_effective_model()) if _effective_model() else None
        return challenge_buy_theses(
            bull_reports=tool_input["bull_reports"],
            context=tool_input.get("context", ""),
            model=_bear_model,
        )

    elif tool_name == "get_ml_factor_weights":
        return ml_insights.get_ml_factor_weights()

    elif tool_name == "prioritize_watchlist_ml":
        return ml_insights.prioritize_watchlist_ml()

    elif tool_name == "get_position_size_recommendation":
        return ml_insights.get_position_size_recommendation(
            ticker=tool_input["ticker"],
            features=tool_input.get("features", {}),
        )

    elif tool_name == "get_economic_indicators":
        return external_data.get_economic_indicators()

    elif tool_name == "get_google_trends":
        keywords = tool_input.get("keywords")
        return external_data.get_google_trends(tool_input["ticker"], keywords)

    elif tool_name == "get_retail_sentiment":
        return external_data.get_retail_sentiment(tool_input["ticker"])

    elif tool_name == "get_rss_news":
        return external_data.get_rss_news(tool_input["ticker"])

    elif tool_name == "analyze_earnings_call":
        return sec_data.get_earnings_transcript(tool_input["ticker"])

    elif tool_name == "analyze_sec_filing":
        form_type = tool_input.get("form_type", "10-K")
        return sec_data.get_sec_filing_analysis(tool_input["ticker"], form_type)

    elif tool_name == "get_material_events":
        days = tool_input.get("days", 90)
        return sec_data.get_material_events(tool_input["ticker"], days)

    elif tool_name == "get_competitor_analysis":
        return sec_data.get_competitor_analysis(tool_input["ticker"])

    elif tool_name == "get_superinvestor_positions":
        return sec_data.get_superinvestor_positions(tool_input["ticker"])

    elif tool_name == "get_business_trajectory":
        from agent.market_data import get_business_trajectory
        ticker = tool_input.get("ticker", "")
        result = get_business_trajectory(ticker)

    elif tool_name == "identify_weakest_link":
        from agent.ml_insights import identify_weakest_link
        result = identify_weakest_link()

    elif tool_name == "get_preearnings_briefing":
        from agent.market_data import get_preearnings_briefing
        ticker = tool_input.get("ticker", "")
        result = get_preearnings_briefing(ticker)

    elif tool_name == "recalibrate_universe_scores":
        from agent.ml_insights import recalibrate_universe_scores
        result = recalibrate_universe_scores()

    elif tool_name == "run_portfolio_stress_test":
        from agent.ml_insights import run_portfolio_stress_test
        scenario = tool_input.get("scenario", "all")
        result = run_portfolio_stress_test(scenario)

    elif tool_name == "get_triaged_alerts":
        from agent.market_data import get_triaged_alerts
        tickers = tool_input.get("tickers")
        result = get_triaged_alerts(tickers)

    elif tool_name == "calculate_intrinsic_value":
        return market_data.calculate_intrinsic_value(tool_input["ticker"])
    elif tool_name == "check_concentration_limits":
        return portfolio.check_concentration_limits(
            ticker=tool_input["ticker"],
            sector=tool_input["sector"],
            buy_amount=tool_input["buy_amount"],
        )
    elif tool_name == "check_dividend_payments":
        from agent.market_data import check_dividend_payments
        return {"dividends": check_dividend_payments(tool_input["holdings"])}
    elif tool_name == "check_earnings_surprises":
        from agent.market_data import check_earnings_surprises
        return {"surprises": check_earnings_surprises(tool_input["holdings"])}
    elif tool_name == "check_fundamental_deterioration":
        from agent.market_data import check_fundamental_deterioration
        return {"alerts": check_fundamental_deterioration(tool_input["holdings"])}
    elif tool_name == "check_portfolio_correlation":
        from agent.ml_insights import check_portfolio_correlation
        return check_portfolio_correlation(tool_input["ticker"])
    elif tool_name == "check_position_drift":
        return portfolio.check_position_drift()
    elif tool_name == "check_watchlist_staleness":
        return {"stale_entries": portfolio.get_stale_watchlist_entries(tool_input.get("min_age_days", 60))}
    elif tool_name == "check_watchlist_triggers":
        watchlist = portfolio.get_watchlist()
        result = market_data.check_watchlist_triggers(watchlist)
        # Log any newly triggered items
        for item in result.get("triggered", []):
            portfolio.log_watchlist_event(
                ticker=item["ticker"],
                event_type="TRIGGERED",
                price=item.get("current_price"),
                target_price=item.get("target_entry_price"),
                notes=f"Price {item.get('current_price')} hit target {item.get('target_entry_price')}",
            )
        return result
    elif tool_name == "detect_financial_anomalies":
        return market_data.detect_financial_anomalies(
            ticker=tool_input["ticker"],
            screener_data=tool_input.get("screener_data"),
        )
    elif tool_name == "discover_universe_parallel":
        meta = portfolio.get_universe_scores_meta()

        # Auto-refresh if cache is empty or stale (>90 days)
        age = meta.get("days_since_refresh")
        if meta["count"] == 0 or (age is not None and age > 90):
            sp500 = market_data.get_stock_universe("sp500")
            intl  = market_data.get_international_universe()
            us_tickers   = sp500.get("tickers", [])
            intl_tickers = intl.get("tickers", [])
            scored = market_data.score_quality_universe(us_tickers, intl_tickers)
            portfolio.save_universe_scores(scored)
            meta = portfolio.get_universe_scores_meta()

        # Load top quality names from cache
        quality_candidates = portfolio.get_universe_scores(top_n=150)

        # Fetch fresh valuation metrics and re-rank by combined score
        enriched = market_data.get_fresh_valuation(quality_candidates)

        top_candidates = enriched[:60]

        return {
            "top_candidates": top_candidates,
            "total_candidates_evaluated": len(enriched),
            "quality_cache_size": meta["count"],
            "quality_cache_age_days": meta["days_since_refresh"],
            "note": (
                f"Quality scores cached for {meta['count']} tickers "
                f"(last refreshed {meta['days_since_refresh']} days ago). "
                f"Fetched fresh valuation for top {len(quality_candidates)} quality names. "
                f"Returning top {len(top_candidates)} by combined quality+valuation score. "
                "Call refresh_universe_scores to rebuild the quality cache (do this quarterly "
                "or when the cache is >90 days old)."
            ),
        }
    elif tool_name == "get_conviction_calibration":
        return ml_insights.get_conviction_calibration()
    elif tool_name == "get_decision_thresholds":
        return ml_insights.get_decision_thresholds()
    elif tool_name == "get_conviction_position_size":
        from agent.ml_insights import conviction_position_size
        return conviction_position_size(
            conviction_score=tool_input["conviction_score"],
            regime=tool_input["regime"],
            portfolio_equity=tool_input["portfolio_equity"],
        )
    elif tool_name == "get_earnings_tone_trend":
        return portfolio.get_earnings_tone_delta(tool_input["ticker"])
    elif tool_name == "get_market_iv_context":
        return ml_insights.get_market_iv_context()
    elif tool_name == "get_regime_change_status":
        from agent.ml_insights import detect_regime_change
        return detect_regime_change()
    elif tool_name == "get_sector_rotation_signal":
        from agent.ml_insights import get_sector_rotation_signal
        return get_sector_rotation_signal()
    elif tool_name == "get_behaviour_summary":
        return portfolio.get_behaviour_summary(n_sessions=tool_input.get("n_sessions", 10))

    elif tool_name == "log_workflow_issue":
        return portfolio.log_workflow_issue(
            issue=tool_input["issue"],
            suggestion=tool_input["suggestion"],
            severity=tool_input.get("severity", "low"),
        )

    elif tool_name == "save_session_audit":
        return portfolio.save_session_audit(
            tickers_screened=tool_input.get("tickers_screened", 0),
            tickers_researched=tool_input.get("tickers_researched", 0),
            buys_made=tool_input.get("buys_made", 0),
            watchlist_added=tool_input.get("watchlist_added", 0),
            shadow_added=tool_input.get("shadow_added", 0),
            workflow_issues_logged=tool_input.get("workflow_issues_logged", 0),
            re_researched_watchlist=tool_input.get("re_researched_watchlist", 0),
            deviated_from_matrix=tool_input.get("deviated_from_matrix", 0),
            duplicate_tool_calls=tool_input.get("duplicate_tool_calls", 0),
            contradicted_prior_session=tool_input.get("contradicted_prior_session", 0),
            audit_notes=tool_input.get("audit_notes", ""),
        )

    elif tool_name == "get_session_efficiency":
        stats = _get_session_stats()
        import datetime as _dt
        session_date = _dt.date.today().isoformat()
        sid = portfolio.save_session_efficiency(
            session_date=session_date,
            total_tool_calls=stats["total_tool_calls"],
            unique_tools_used=stats["unique_tools_used"],
            stocks_researched=tool_input.get("stocks_researched", 0),
            stocks_bought=tool_input.get("stocks_bought", 0),
            stocks_watchlisted=tool_input.get("stocks_watchlisted", 0),
            duration_seconds=stats["duration_seconds"],
            notes=tool_input.get("notes"),
        )
        history = portfolio.get_session_efficiency_history(limit=5)
        return {"saved_id": sid, "current_session": stats, "recent_sessions": history}
    elif tool_name == "get_stored_thesis":
        return portfolio.get_stored_thesis(tool_input["ticker"])
    elif tool_name == "get_watchlist_earnings":
        watchlist = portfolio.get_watchlist()
        return market_data.get_watchlist_earnings(watchlist)
    elif tool_name == "get_watchlist_history":
        ticker = tool_input.get("ticker")
        return portfolio.get_watchlist_history(ticker=ticker)
    elif tool_name == "log_prediction":
        from agent.portfolio import log_prediction
        log_prediction(
            ticker=tool_input["ticker"],
            action=tool_input["action"],
            conviction_score=tool_input.get("conviction_score"),
            predicted_iv=tool_input.get("predicted_iv"),
            price_at_decision=tool_input.get("price_at_decision"),
            mos_pct=tool_input.get("mos_pct"),
        )
        return {"logged": True, "ticker": tool_input["ticker"], "action": tool_input["action"]}
    elif tool_name == "query_kb":
        from agent.knowledge_base import query_kb
        return {
            "results": query_kb(
                tool_input["query"],
                topic=tool_input.get("topic"),
                max_results=tool_input.get("max_results", 3),
            )
        }
    elif tool_name == "reconcile_predictions":
        import yfinance as yf
        from agent.portfolio import get_pending_reconciliations, update_prediction_outcome, get_prediction_accuracy
        from agent.market_data import get_spy_price
        pending = get_pending_reconciliations(min_age_days=90)
        reconciled = []
        spy_now = get_spy_price() or 0
        for pred in pending:
            try:
                info = yf.Ticker(pred["ticker"]).info
                current_price = info.get("regularMarketPrice") or info.get("previousClose")
                if not current_price or not pred["price_at_decision"]:
                    continue
                ret_pct = (current_price - pred["price_at_decision"]) / pred["price_at_decision"] * 100
                # Approximate SPY return over same period (rough — would need historical SPY)
                update_prediction_outcome(pred["id"], current_price, round(ret_pct, 2), 0)
                reconciled.append({"ticker": pred["ticker"], "return_pct": round(ret_pct, 2)})
            except Exception:
                continue
        accuracy = get_prediction_accuracy()
        return {"reconciled_count": len(reconciled), "reconciled": reconciled, "accuracy": accuracy}
    elif tool_name == "run_iv_postmortem":
        return ml_insights.run_iv_postmortem()
    elif tool_name == "save_kb_note":
        from agent.knowledge_base import save_kb_note
        return save_kb_note(
            topic=tool_input["topic"],
            title=tool_input["title"],
            content=tool_input["content"],
            tags=tool_input.get("tags"),
        )
    elif tool_name == "save_management_profile":
        from agent.knowledge_base import save_kb_note
        ticker = tool_input["ticker"].upper()
        name = tool_input.get("company_name", ticker)
        score = tool_input["quality_score"]
        rating = "⭐" * min(score, 5)
        content = f"Quality score: {score}/10 | Capital allocation: {tool_input.get('capital_allocation_rating', 'N/A')}\n\n"
        content += f"Summary: {tool_input['summary']}\n\n"
        if tool_input.get("key_green_flags"):
            content += "Green flags:\n" + "\n".join(f"- {f}" for f in tool_input["key_green_flags"]) + "\n\n"
        if tool_input.get("key_red_flags"):
            content += "Red flags:\n" + "\n".join(f"- {f}" for f in tool_input["key_red_flags"]) + "\n"
        return save_kb_note(
            topic="management",
            title=f"{ticker} management quality",
            content=content,
            tags=[ticker, name, f"score_{score}", tool_input.get("capital_allocation_rating", "")],
        )

    else:
        return {"error": f"Unknown tool: {tool_name}"}

    return result


def _get_portfolio_status() -> dict:
    cash = portfolio.get_cash()
    holdings = portfolio.get_holdings()

    enriched_holdings = []
    total_market_value = 0.0
    total_cost_basis = 0.0

    for h in holdings:
        quote = market_data.get_stock_quote(h["ticker"])
        current_price = quote.get("price") if "error" not in quote else None

        if current_price:
            market_value = h["shares"] * current_price
            cost_basis = h["shares"] * h["avg_cost"]
            unrealized_pnl = market_value - cost_basis
            unrealized_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else 0
            total_market_value += market_value
            total_cost_basis += cost_basis
        else:
            market_value = None
            unrealized_pnl = None
            unrealized_pct = None

        enriched_holdings.append({
            **h,
            "name": quote.get("name", ""),
            "sector": quote.get("sector", ""),
            "industry": quote.get("industry", ""),
            "current_price": current_price,
            "market_value": round(market_value, 2) if market_value else None,
            "unrealized_pnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
            "unrealized_pct": round(unrealized_pct, 2) if unrealized_pct is not None else None,
        })

    total_portfolio_value = cash + total_market_value
    total_unrealized = total_market_value - total_cost_basis

    return {
        "cash": round(cash, 2),
        "holdings": enriched_holdings,
        "total_invested_value": round(total_market_value, 2),
        "total_cost_basis": round(total_cost_basis, 2),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_portfolio_value": round(total_portfolio_value, 2),
        "number_of_positions": len(enriched_holdings),
    }


def _handle_buy(tool_input: dict) -> dict:
    ticker = tool_input["ticker"]
    notes = tool_input.get("notes", "")

    # Get current price
    quote = market_data.get_stock_quote(ticker)
    if "error" in quote:
        return {"error": f"Could not get price for {ticker}: {quote['error']}"}

    price = quote["price"]

    # Determine shares
    if "shares" in tool_input and tool_input["shares"]:
        shares = float(tool_input["shares"])
    elif "dollar_amount" in tool_input and tool_input["dollar_amount"]:
        shares = round(tool_input["dollar_amount"] / price, 6)
    else:
        return {"error": "Must specify either 'shares' or 'dollar_amount'"}

    if shares <= 0:
        return {"error": "Shares must be positive"}

    result = portfolio.buy_stock(ticker, shares, price, notes)
    if result["success"]:
        txn_id = result["transaction_id"]
        portfolio.log_agent_message(
            f"BUY {shares:.4f} shares of {ticker} @ ${price:.2f} | Reason: {notes}"
        )
        if notes:
            portfolio.save_trade_thesis(txn_id, ticker, "BUY", notes)
        # Save screener signals if provided — enables performance attribution later
        screener = tool_input.get("screener_snapshot")
        if screener and isinstance(screener, dict):
            portfolio.save_trade_signals(txn_id, ticker, screener)
        # Auto-remove from watchlist when bought
        portfolio.remove_from_watchlist(ticker)
    return result


def _handle_sell(tool_input: dict) -> dict:
    ticker = tool_input["ticker"]
    notes = tool_input.get("notes", "")

    # Get current price
    quote = market_data.get_stock_quote(ticker)
    if "error" in quote:
        return {"error": f"Could not get price for {ticker}: {quote['error']}"}

    price = quote["price"]

    if tool_input.get("sell_all"):
        holding = portfolio.get_holding(ticker)
        if not holding:
            return {"error": f"No position in {ticker}"}
        shares = holding["shares"]
    elif "shares" in tool_input and tool_input["shares"]:
        shares = float(tool_input["shares"])
    else:
        return {"error": "Must specify 'shares' or set 'sell_all' to true"}

    result = portfolio.sell_stock(ticker, shares, price, notes)
    if result["success"]:
        portfolio.log_agent_message(
            f"SELL {shares:.4f} shares of {ticker} @ ${price:.2f} | Reason: {notes}"
        )
        if notes:
            portfolio.save_trade_thesis(result["transaction_id"], ticker, "SELL", notes)
        # Auto-close any open IV prediction so backtest has outcome data
        portfolio.close_prediction(ticker, price)
    return result


def _handle_benchmark_comparison() -> dict:
    snapshots = portfolio.get_portfolio_snapshots()
    if not snapshots:
        return {
            "note": (
                "No snapshots yet. A snapshot is automatically saved at the end of each "
                "review session when you call save_session_reflection."
            )
        }

    first = snapshots[0]   # oldest
    latest = snapshots[-1]  # most recent

    portfolio_return = (
        (latest["portfolio_value"] - first["portfolio_value"]) / first["portfolio_value"] * 100
    )

    spy = market_data.get_stock_quote("^GSPC")
    current_spy = spy.get("price") if "error" not in spy else None
    benchmark_return = None
    alpha = None
    is_beating = None

    if current_spy and first.get("benchmark_price"):
        benchmark_return = (current_spy - first["benchmark_price"]) / first["benchmark_price"] * 100
        alpha = portfolio_return - benchmark_return
        is_beating = alpha > 0

    # Recent snapshots for context (last 5, newest first)
    recent = [
        {
            "date": s["ts"][:10],
            "portfolio_value": s["portfolio_value"],
            "sp500_price": s.get("benchmark_price"),
        }
        for s in reversed(snapshots[-5:])
    ]

    return {
        "start_date": first["ts"][:10],
        "portfolio_start_value": round(first["portfolio_value"], 2),
        "portfolio_current_value": round(latest["portfolio_value"], 2),
        "portfolio_return_pct": round(portfolio_return, 2),
        "benchmark": "S&P 500 (^GSPC)",
        "benchmark_start_price": first.get("benchmark_price"),
        "benchmark_current_price": current_spy,
        "benchmark_return_pct": round(benchmark_return, 2) if benchmark_return is not None else None,
        "alpha_pct": round(alpha, 2) if alpha is not None else None,
        "is_beating_benchmark": is_beating,
        "total_snapshots": len(snapshots),
        "recent_history": recent,
    }
