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
            "Compute RSI-14, MACD (12/26/9), Bollinger Bands (20-day, 2σ), EMA-50/200, "
            "and volume vs 20-day average for a stock to assess entry timing and momentum."
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
            "Return short % of float, days-to-cover, and month-over-month change in shares "
            "short for a stock. Data sourced from FINRA via yfinance (~2-week lag)."
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
            "Return put/call volume and OI ratios, ATM IV vs 30-day realized volatility, "
            "and unusual contract activity (large fresh directional bets) for the nearest 3 expiries."
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
            "Return the next earnings date, consensus EPS and revenue estimates, "
            "and the last 4 quarters of beat/miss history for a stock."
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
            "Return recent analyst upgrades, downgrades, and initiations with firm name and grade change."
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
        "name": "get_analyst_consensus",
        "description": (
            "Return aggregated Wall Street analyst consensus: rating distribution (strong_buy/buy/hold/sell counts), "
            "price targets (mean/high/low), upside % to mean target, and EPS revision momentum "
            "(# analysts raising vs cutting estimates last 30 days)."
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
        "name": "get_financial_history",
        "description": (
            "Return 4-5 years of annual financial history: revenue, gross/operating/net margins, "
            "YoY revenue growth %, free cash flow, operating cash flow, total debt, and cash. "
            "Use to assess long-run business quality and FCF trends."
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
        "name": "get_insider_activity",
        "description": (
            "Return recent insider buy/sell transactions by executives, directors, and major shareholders."
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
            "Return key macro indicators: 10-year and 2-year Treasury yields, yield curve status, "
            "dollar index, oil, gold, and VIX with synthesised regime signals."
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
            "Return regime-appropriate defensive ETF hedge recommendations (TLT, IEF, SHV, GLD, TIP, GSG) "
            "sized to the portfolio. Only produces recommendations in RISK_OFF, INFLATIONARY, "
            "STAGFLATION, or HIGH_RATES regimes; always funded from cash, max 20% of portfolio."
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
        "description": "Compare portfolio total return vs S&P 500: alpha, historical snapshots, start/current values.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_portfolio_metrics",
        "description": (
            "Return Sharpe ratio, max drawdown, annualised volatility, and rolling 1/3/6-month "
            "returns vs S&P 500 computed from portfolio snapshot history."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_investment_memory",
        "description": "Retrieve past investment theses for current holdings and recently closed positions.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_session_reflections",
        "description": "Return past post-session reflections and lessons from previous portfolio reviews.",
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
            "Return ~200 major international stocks (US-listed ADRs and foreign-listed tickers) "
            "for screening. Optionally filter by region: europe, asia, latam, canada, india."
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
            "Two-layer screen: quality-scores ~700 tickers on stable fundamentals, then re-screens "
            "the top 150 on fresh valuation metrics. Returns the top 60 candidates ranked by "
            "combined quality+valuation score. Auto-refreshes cache if empty."
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
            "Rebuild the quality score cache for ~700 tickers on stable fundamentals "
            "(revenue growth, margins, ROE, debt). Slow (~5-10 min) — run quarterly or "
            "when discover_universe_parallel reports quality_cache_age_days > 90."
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
        "description": "Return current portfolio weights broken down by GICS sector.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_to_watchlist",
        "description": "Add a stock to the watchlist with a reason and optional target entry price.",
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
        "description": "Return the current watchlist with target entry prices and notes.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_from_watchlist",
        "description": "Remove a stock from the watchlist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol to remove"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "prune_watchlist",
        "description": (
            "Archive watchlist entries where the current price is more than 40% above the target "
            "entry price into the shadow portfolio. Stocks that have run far past their entry target "
            "are not actionable — they clutter the watchlist and get re-researched unnecessarily. "
            "Call this once at session start immediately after get_watchlist."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_trade_outcomes",
        "description": (
            "Return all past buy signal snapshots with screener signals at purchase time "
            "and actual return outcomes for closed and open positions."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_signal_performance",
        "description": (
            "Return per-signal statistics (PEG, FCF yield, momentum, revenue growth) showing "
            "positive_rate_pct and avg_return_pct split by whether each threshold was met at buy time."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_to_shadow_portfolio",
        "description": "Record a rejected stock with price at consideration and reason for passing.",
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
            "Run a strategy backtest in one of three modes: "
            "'trade_history' (replay all closed trades — win rate, Sharpe, max drawdown, S&P alpha, regime segmentation), "
            "'signal_cohorts' (break trades by signal threshold to find predictive signals), "
            "or 'momentum' (simulate top-momentum buy from holding_days ago vs S&P 500; requires tickers list)."
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
        "description": "Return shadow portfolio performance: current price vs price at consideration for all rejected stocks.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── ML insights (learned from portfolio history) ──────────────────────────
    {
        "name": "get_regime_change_status",
        "description": (
            "Return the current macro regime, whether it has changed since the previous session, "
            "days since last regime, and a change summary."
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
            "Return ML-derived factor weights for screener signals, blended with regime-adjusted "
            "priors and calibrated to this portfolio's closed trade history."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_watchlist_triggers",
        "description": (
            "Return watchlist items bucketed by distance from target entry: "
            "TRIGGERED (at/below target), APPROACHING (within 10%), WAITING, or NO_TARGET."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_watchlist_earnings",
        "description": (
            "Return upcoming earnings dates for all watchlist items bucketed as "
            "IMMINENT (≤7 days), UPCOMING (≤30 days), or DISTANT."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_fundamental_deterioration",
        "description": (
            "Check held positions for fundamental deterioration: revenue decline, negative FCF, "
            "low margins, high leverage, low ROE. Returns severity: WATCH (1 flag), REVIEW (2), EXIT (3+)."
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
            "Return recent earnings surprises for held positions: flags beats (>15%) and misses (>15%)."
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
            "Return annual dividend rate, yield %, estimated annual income, and ex-dividend dates "
            "for held positions."
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
            "Run a standardised 3-stage DCF (10% discount rate, 2.5% terminal, 20% growth haircut) "
            "and return bear/base/bull intrinsic value per share and margin of safety at current price."
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
            "Return watchlist lifecycle events: when items were TRIGGERED, APPROACHING, BOUGHT, or REMOVED. "
            "Optionally filter to a specific ticker."
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
            "Score every watchlist item using ML factor weights and return a ranked list with "
            "ML score (0-10), strengths, risk factors, and proximity to target entry price."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_conviction_position_size",
        "description": (
            "Return position size in dollars for a given conviction score (1-10) and macro regime."
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
            "Return recommended position size (% of portfolio) based on feature risk flags, "
            "a drawdown model trained on closed trade history, and current macro regime. "
            "Pass screener_snapshot features from screen_stocks as the 'features' argument."
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
            "Fetch US macro indicators from FRED: GDP, CPI, unemployment, jobless claims, "
            "retail sales, consumer sentiment, industrial production, housing starts, and fed funds rate."
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
            "Return 12-month Google search-interest trend for a company's brand or products, "
            "including recent 8-week direction and current interest vs 12-month average."
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
            "Return retail investor sentiment from StockTwits and Reddit: bull/bear ratio and recent posts."
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
        "description": "Return recent news headlines from Yahoo Finance, MarketWatch, and Seeking Alpha RSS feeds.",
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
            "Fetch the most recent earnings call transcript from SEC EDGAR (8-K) for analysis "
            "of management tone, guidance language, and Q&A content."
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
            "Fetch the Business overview, Risk Factors, and MD&A sections from the latest "
            "10-K or 10-Q on SEC EDGAR."
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
            "Fetch recent SEC 8-K material event filings: CEO/CFO changes (5.02), M&A (2.01), "
            "asset impairments (2.06), auditor changes (4.01), restatements (4.02), bankruptcy (1.03)."
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
            "Return a side-by-side fundamental comparison (PEG, FCF yield, margins, ROE, momentum) "
            "of the stock vs its closest S&P 500 sector peers."
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
            "Return 13F-based positions for Buffett, Ackman, Tepper, Druckenmiller, Loeb, "
            "Einhorn, and Halvorsen. Note: 13F filings have up to a 45-day lag."
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
            "Launch parallel research subagents to deep-dive multiple tickers simultaneously. "
            "Each subagent runs all research tools and returns a structured JSON report with "
            "recommendation (buy/watchlist/pass), conviction score, key positives/risks, and thesis."
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
            "Launch adversarial bear case subagents for each 'buy'-rated report from "
            "research_stocks_parallel. Returns verdict (proceed/caution/reject), bear_conviction, "
            "key_objections, risks_missed_by_bull, and recommended_action for each stock."
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
            "Return whether a proposed buy would breach position (>10%) or sector (>30%) "
            "concentration limits, and the max_allowed_buy that stays within limits."
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
            "Log a buy/watchlist/pass decision with conviction score, predicted IV, and price "
            "for later reconciliation against actual outcomes."
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
        "description": "Reconcile past predictions (>90 days old) against actual prices and record return and alpha vs SPY.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_news_alerts",
        "description": (
            "Scan recent news for held positions and flag material events: CEO/CFO departures, "
            "restatements, M&A, bankruptcy risk, guidance cuts, regulatory actions."
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
        "description": "Search the investment knowledge base for notes on companies, frameworks, valuation methods, or lessons.",
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
        "description": "Save or update a note in the investment knowledge base (searchable via query_kb).",
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
            "Analyse past IV predictions vs actual outcomes. Returns conviction_calibration, "
            "sector_calibration, and insights; saves results to the knowledge base."
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
            "Return avg_return, win_rate, and avg_alpha per conviction bucket (5-6, 7-8, 9-10) "
            "with calibration_status: well_calibrated, miscalibrated_high, or insufficient_data."
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
            "Return the buy/watchlist/pass decision matrix: regime-adjusted MoS threshold (20-28%), "
            "bear_override_conviction threshold, and ML factor guidance for candidate ranking."
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
            "Return earnings call sentiment direction (positive/negative/neutral) for the last "
            "3-4 quarters with a delta signal indicating trend direction."
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
            "Return what % of the researched universe offers ≥20% margin of safety, avg_mos_pct, "
            "market_signal (cheap/fair/expensive), and sector breakdown."
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
            "Return z-scores for financial ratios vs the company's 5-year history and sector peers, "
            "with overall_flag (clean/watch/flagged) and interpretation summary."
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
            "Save a structured post-session reflection. Required sections: Actions Taken "
            "(include a 'New Hedge Positions' subsection ONLY if hedge ETFs were actually "
            "purchased this session — omit entirely if no hedges were bought), "
            "Thesis Validation, Signal Performance, Macro Observations, Shadow Portfolio Review, "
            "Lessons for Next Session."
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
        "description": "Return the stored investment thesis for a ticker from previous research sessions.",
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
            "Return 1-year price correlation between a candidate and all held positions, "
            "with a warning and sizing adjustment if avg correlation > 0.6 or any pair > 0.7."
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
        "description": "Return watchlist entries that haven't been re-evaluated in 60+ days (stale).",
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
        "description": "Save a management quality assessment (score, capital_allocation_rating, flags, summary) to the KB.",
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
            "Return average session behaviour stats: tickers screened/researched, buys per session, "
            "rule deviations, duplicate tool calls, and workflow improvement suggestions."
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
            "Log a workflow inefficiency with a fix suggestion and severity "
            "(low=minor friction, medium=recurring waste, high=caused wrong decision)."
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
        "description": "Save structured self-audit metrics at the end of a session (counts for screens, buys, deviations, duplicates, etc.).",
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
        "description": "Return current session tool call count, duration, unique tools used, and last 10 sessions' history.",
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
            "Return positions that have drifted above concentration limits (>12% single position, "
            ">33% sector) through price appreciation, with trim recommendations."
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
            "Return sector distribution across portfolio, recently researched stocks, and the universe "
            "to detect availability bias and portfolio tilts vs the opportunity set."
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
            "Run a standardised 3-stage DCF (10% discount rate, 2.5% terminal, 20% growth haircut) "
            "and return bear/base/bull intrinsic value per share and margin of safety at current price."
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
            "Return whether a proposed buy would breach position (>10%) or sector (>30%) "
            "concentration limits, and the max_allowed_buy that stays within limits."
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
            "Return annual dividend rate, yield %, estimated annual income, and ex-dividend dates "
            "for held positions."
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
            "Return recent earnings surprises for held positions: flags beats (>15%) and misses (>15%)."
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
            "Check held positions for fundamental deterioration: revenue decline, negative FCF, "
            "low margins, high leverage, low ROE. Returns severity: WATCH (1 flag), REVIEW (2), EXIT (3+)."
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
            "Return 1-year price correlation between a candidate and all held positions, "
            "with a warning and sizing adjustment if avg correlation > 0.6 or any pair > 0.7."
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
            "Return positions that have drifted above concentration limits (>12% single position, "
            ">33% sector) through price appreciation, with trim recommendations."
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
        "description": "Return watchlist entries that haven't been re-evaluated in 60+ days (stale).",
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
            "Return watchlist items bucketed by distance from target entry: "
            "TRIGGERED (at/below target), APPROACHING (within 10%), WAITING, or NO_TARGET."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── restored: detect_financial_anomalies

    {
        "name": "detect_financial_anomalies",
        "description": (
            "Return z-scores for financial ratios vs the company's 5-year history and sector peers, "
            "with overall_flag (clean/watch/flagged) and interpretation summary."
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
            "Two-layer screen: quality-scores ~700 tickers on stable fundamentals, then re-screens "
            "the top 150 on fresh valuation metrics. Returns the top 60 candidates ranked by "
            "combined quality+valuation score. Auto-refreshes cache if empty."
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
            "Return avg_return, win_rate, and avg_alpha per conviction bucket (5-6, 7-8, 9-10) "
            "with calibration_status: well_calibrated, miscalibrated_high, or insufficient_data."
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
            "Return the buy/watchlist/pass decision matrix: regime-adjusted MoS threshold (20-28%), "
            "bear_override_conviction threshold, and ML factor guidance for candidate ranking."
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
            "Return position size in dollars for a given conviction score (1-10) and macro regime."
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
            "Return earnings call sentiment direction (positive/negative/neutral) for the last "
            "3-4 quarters with a delta signal indicating trend direction."
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
            "Return what % of the researched universe offers ≥20% margin of safety, avg_mos_pct, "
            "market_signal (cheap/fair/expensive), and sector breakdown."
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
            "Return the current macro regime, whether it has changed since the previous session, "
            "days since last regime, and a change summary."
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
            "Return sector distribution across portfolio, recently researched stocks, and the universe "
            "to detect availability bias and portfolio tilts vs the opportunity set."
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
        "description": "Return current session tool call count, duration, unique tools used, and last 10 sessions' history.",
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
        "description": "Return the stored investment thesis for a ticker from previous research sessions.",
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
            "Return upcoming earnings dates for all watchlist items bucketed as "
            "IMMINENT (≤7 days), UPCOMING (≤30 days), or DISTANT."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── restored: get_watchlist_history

    {
        "name": "get_watchlist_history",
        "description": (
            "Return watchlist lifecycle events: when items were TRIGGERED, APPROACHING, BOUGHT, or REMOVED. "
            "Optionally filter to a specific ticker."
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
            "Log a buy/watchlist/pass decision with conviction score, predicted IV, and price "
            "for later reconciliation against actual outcomes."
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
        "description": "Search the investment knowledge base for notes on companies, frameworks, valuation methods, or lessons.",
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
        "description": "Reconcile past predictions (>90 days old) against actual prices and record return and alpha vs SPY.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── restored: run_iv_postmortem

    {
        "name": "run_iv_postmortem",
        "description": (
            "Analyse past IV predictions vs actual outcomes. Returns conviction_calibration, "
            "sector_calibration, and insights; saves results to the knowledge base."
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
        "description": "Save or update a note in the investment knowledge base (searchable via query_kb).",
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
        "description": "Save a management quality assessment (score, capital_allocation_rating, flags, summary) to the KB.",
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
            "Compute RSI-14, MACD (12/26/9), Bollinger Bands (20-day, 2σ), EMA-50/200, "
            "and volume vs 20-day average for a stock to assess entry timing and momentum."
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
            "Return short % of float, days-to-cover, and month-over-month change in shares "
            "short for a stock. Data sourced from FINRA via yfinance (~2-week lag)."
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
            "Return put/call volume and OI ratios, ATM IV vs 30-day realized volatility, "
            "and unusual contract activity (large fresh directional bets) for the nearest 3 expiries."
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
            "Return the next earnings date, consensus EPS and revenue estimates, "
            "and the last 4 quarters of beat/miss history for a stock."
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
            "Return recent analyst upgrades, downgrades, and initiations with firm name and grade change."
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
        "name": "get_analyst_consensus",
        "description": (
            "Return aggregated Wall Street analyst consensus: rating distribution (strong_buy/buy/hold/sell counts), "
            "price targets (mean/high/low), upside % to mean target, and EPS revision momentum "
            "(# analysts raising vs cutting estimates last 30 days)."
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
        "name": "get_financial_history",
        "description": (
            "Return 4-5 years of annual financial history: revenue, gross/operating/net margins, "
            "YoY revenue growth %, free cash flow, operating cash flow, total debt, and cash. "
            "Use to assess long-run business quality and FCF trends."
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
        "name": "get_insider_activity",
        "description": (
            "Return recent insider buy/sell transactions by executives, directors, and major shareholders."
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
            "Return key macro indicators: 10-year and 2-year Treasury yields, yield curve status, "
            "dollar index, oil, gold, and VIX with synthesised regime signals."
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
            "Return regime-appropriate defensive ETF hedge recommendations (TLT, IEF, SHV, GLD, TIP, GSG) "
            "sized to the portfolio. Only produces recommendations in RISK_OFF, INFLATIONARY, "
            "STAGFLATION, or HIGH_RATES regimes; always funded from cash, max 20% of portfolio."
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
        "description": "Compare portfolio total return vs S&P 500: alpha, historical snapshots, start/current values.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_portfolio_metrics",
        "description": (
            "Return Sharpe ratio, max drawdown, annualised volatility, and rolling 1/3/6-month "
            "returns vs S&P 500 computed from portfolio snapshot history."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_investment_memory",
        "description": "Retrieve past investment theses for current holdings and recently closed positions.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_session_reflections",
        "description": "Return past post-session reflections and lessons from previous portfolio reviews.",
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
            "Return ~200 major international stocks (US-listed ADRs and foreign-listed tickers) "
            "for screening. Optionally filter by region: europe, asia, latam, canada, india."
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
        "description": "Return current portfolio weights broken down by GICS sector.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_to_watchlist",
        "description": "Add a stock to the watchlist with a reason and optional target entry price.",
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
        "description": "Return the current watchlist with target entry prices and notes.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_from_watchlist",
        "description": "Remove a stock from the watchlist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol to remove"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "prune_watchlist",
        "description": (
            "Archive watchlist entries where the current price is more than 40% above the target "
            "entry price into the shadow portfolio. Stocks that have run far past their entry target "
            "are not actionable — they clutter the watchlist and get re-researched unnecessarily. "
            "Call this once at session start immediately after get_watchlist."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_trade_outcomes",
        "description": (
            "Return all past buy signal snapshots with screener signals at purchase time "
            "and actual return outcomes for closed and open positions."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_signal_performance",
        "description": (
            "Return per-signal statistics (PEG, FCF yield, momentum, revenue growth) showing "
            "positive_rate_pct and avg_return_pct split by whether each threshold was met at buy time."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_to_shadow_portfolio",
        "description": "Record a rejected stock with price at consideration and reason for passing.",
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
        "description": "Return shadow portfolio performance: current price vs price at consideration for all rejected stocks.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── ML insights (learned from portfolio history) ──────────────────────────
    {
        "name": "get_ml_factor_weights",
        "description": (
            "Return ML-derived factor weights for screener signals, blended with regime-adjusted "
            "priors and calibrated to this portfolio's closed trade history."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "prioritize_watchlist_ml",
        "description": (
            "Score every watchlist item using ML factor weights and return a ranked list with "
            "ML score (0-10), strengths, risk factors, and proximity to target entry price."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_position_size_recommendation",
        "description": (
            "Return recommended position size (% of portfolio) based on feature risk flags, "
            "a drawdown model trained on closed trade history, and current macro regime. "
            "Pass screener_snapshot features from screen_stocks as the 'features' argument."
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
            "Fetch US macro indicators from FRED: GDP, CPI, unemployment, jobless claims, "
            "retail sales, consumer sentiment, industrial production, housing starts, and fed funds rate."
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
            "Return 12-month Google search-interest trend for a company's brand or products, "
            "including recent 8-week direction and current interest vs 12-month average."
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
            "Return retail investor sentiment from StockTwits and Reddit: bull/bear ratio and recent posts."
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
        "description": "Return recent news headlines from Yahoo Finance, MarketWatch, and Seeking Alpha RSS feeds.",
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
            "Fetch the most recent earnings call transcript from SEC EDGAR (8-K) for analysis "
            "of management tone, guidance language, and Q&A content."
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
            "Fetch the Business overview, Risk Factors, and MD&A sections from the latest "
            "10-K or 10-Q on SEC EDGAR."
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
            "Fetch recent SEC 8-K material event filings: CEO/CFO changes (5.02), M&A (2.01), "
            "asset impairments (2.06), auditor changes (4.01), restatements (4.02), bankruptcy (1.03)."
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
            "Return a side-by-side fundamental comparison (PEG, FCF yield, margins, ROE, momentum) "
            "of the stock vs its closest S&P 500 sector peers."
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
            "Return 13F-based positions for Buffett, Ackman, Tepper, Druckenmiller, Loeb, "
            "Einhorn, and Halvorsen. Note: 13F filings have up to a 45-day lag."
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
            "Launch parallel research subagents to deep-dive multiple tickers simultaneously. "
            "Each subagent runs all research tools and returns a structured JSON report with "
            "recommendation (buy/watchlist/pass), conviction score, key positives/risks, and thesis."
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
            "Launch adversarial bear case subagents for each 'buy'-rated report from "
            "research_stocks_parallel. Returns verdict (proceed/caution/reject), bear_conviction, "
            "key_objections, risks_missed_by_bull, and recommended_action for each stock."
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
            "Save a structured post-session reflection. Required sections: Actions Taken "
            "(include a 'New Hedge Positions' subsection ONLY if hedge ETFs were actually "
            "purchased this session — omit entirely if no hedges were bought), "
            "Thesis Validation, Signal Performance, Macro Observations, Shadow Portfolio Review, "
            "Lessons for Next Session."
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

    elif tool_name == "get_analyst_consensus":
        return market_data.get_analyst_consensus(tool_input["ticker"])

    elif tool_name == "get_financial_history":
        return market_data.get_financial_history(tool_input["ticker"])

    elif tool_name == "get_insider_activity":
        return market_data.get_insider_activity(tool_input["ticker"])

    elif tool_name == "get_macro_environment":
        return market_data.get_macro_environment()

    elif tool_name == "get_hedge_recommendations":
        # Compute equity/cash % from live portfolio state if not supplied by caller
        equity_pct = tool_input.get("equity_pct")
        cash_pct = tool_input.get("cash_pct")
        holdings = portfolio.get_holdings()
        if equity_pct is None or cash_pct is None:
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
        # If any hedge ETF is already held, don't layer additional insurance on top
        _HEDGE_ETFS = {"GLD", "TIP", "TLT", "IEF", "SHV", "GSG"}
        existing_hedges = [h["ticker"] for h in holdings if h["ticker"].upper() in _HEDGE_ETFS]
        if existing_hedges:
            return {
                "hedge_warranted": False,
                "no_hedge_rationale": (
                    f"Portfolio already holds hedge position(s): {', '.join(existing_hedges)}. "
                    "Do not add new hedges while existing ones are open. "
                    "Review the exit conditions for the existing hedge(s) instead — "
                    "sell them if the triggering regime has resolved."
                ),
                "existing_hedges": existing_hedges,
                "recommendations": [],
                "total_recommended_hedge_pct": 0,
            }
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
        results = market_data.screen_stocks(tickers)
        # Show top 100 — each result is ~200 chars so this stays well within context.
        # Gives ample buffer to find 15 finalists even with a large watchlist/shadow portfolio.
        # The full sorted list is cached in data/screener_cache.json.
        return {
            "total_screened": len(results),
            "showing_top": min(100, len(results)),
            "note": (
                "Top 100 results shown out of the full screened universe. "
                "Walk this list mechanically: take the first 15 tickers that are "
                "NOT held, NOT on watchlist, NOT in shadow portfolio."
            ),
            "results": results[:100],
        }

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

    elif tool_name == "prune_watchlist":
        items = portfolio.get_watchlist()
        price_lookup: dict = {}
        for item in items:
            if item.get("target_entry_price"):
                try:
                    q = market_data.get_stock_quote(item["ticker"])
                    p = q.get("price")
                    if p:
                        price_lookup[item["ticker"]] = float(p)
                except Exception:
                    pass
        return portfolio.prune_watchlist(price_lookup)

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
        # Auto-snapshot for chart + benchmark tracking — only after the first trade,
        # since performance comparison vs S&P is meaningless on an all-cash portfolio.
        if portfolio.get_first_trade_date():
            sp500 = market_data.get_stock_quote("^GSPC")
            spy_price = sp500.get("price") if "error" not in sp500 else None
            portfolio.save_portfolio_snapshot(portfolio_value, cash, invested, spy_price, "review")
            if spy_price and portfolio_value:
                portfolio.save_benchmark_snapshot(portfolio_value, spy_price)
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
    first_trade_date = portfolio.get_first_trade_date()
    if not first_trade_date:
        return {
            "note": "No trades have been made yet. Performance tracking vs S&P 500 begins after the first trade."
        }

    snapshots = portfolio.get_portfolio_snapshots()
    # Only consider snapshots from the first trade date onward
    snapshots = [s for s in snapshots if s["ts"][:10] >= first_trade_date]
    if not snapshots:
        return {
            "note": (
                "No snapshots recorded since the first trade. A snapshot is automatically saved "
                "at the end of each review session when you call save_session_reflection."
            )
        }

    first = snapshots[0]   # oldest post-trade snapshot
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
