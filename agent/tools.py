"""
Tool definitions and handlers for the Claude investment agent.
Each tool maps to a market data or portfolio action.
"""

import json
from typing import Any

from agent import market_data, portfolio, sec_data, external_data, ml_insights


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
            "and returns the top candidates ranked by a composite quality + value score. "
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
                    "description": "List of ticker symbols to screen (max 100 per call)",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top candidates to return (default 10)",
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
        "name": "get_shadow_performance",
        "description": (
            "Review all stocks you previously passed on, showing their price change since consideration. "
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

    elif tool_name == "get_stock_universe":
        index = tool_input.get("index", "all")
        sample_n = min(tool_input.get("sample_n", 200), 300)
        random_seed = tool_input.get("random_seed", None)
        sector = tool_input.get("sector", None)
        return market_data.get_stock_universe(index, sample_n=sample_n, random_seed=random_seed, sector=sector)

    elif tool_name == "screen_stocks":
        tickers = tool_input.get("tickers", [])
        top_n = tool_input.get("top_n", 10)
        return market_data.screen_stocks(tickers, top_n)

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

    else:
        return {"error": f"Unknown tool: {tool_name}"}


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
