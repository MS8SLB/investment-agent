"""
Tool definitions and handlers for the Claude investment agent.
Each tool maps to a market data or portfolio action.
"""

import json
from typing import Any

from agent import market_data, portfolio


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
            "Fetch a random sample of tickers from major US stock universes. "
            "Returns 'sample_n' tickers (default 200) to keep response size manageable. "
            "Call multiple times with different random_seed values (0, 1, 2, ...) to cover "
            "different parts of the universe across multiple screen_stocks calls. "
            "Use this to discover investment opportunities beyond well-known names."
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
                    "description": "Number of tickers to return in this sample (default 200, max 300). Keep this reasonable to avoid large responses.",
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Seed for random sampling. Use different values (0, 1, 2, 3...) across calls to get different batches from the full universe.",
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
                    "description": "Number of top candidates to return (default 25)",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "save_session_reflection",
        "description": (
            "Save a reflection at the end of your portfolio review session. "
            "Document: what actions you took, whether past theses are playing out, "
            "market patterns you observed, and specific lessons for future sessions. "
            "Always call this at the end of each review."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reflection": {
                    "type": "string",
                    "description": (
                        "Your written reflection on this session. Include: actions taken, "
                        "thesis validation for existing positions, market observations, "
                        "and key lessons to apply in future sessions."
                    ),
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
        limit = min(tool_input.get("limit", 8), 15)
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
        return market_data.get_stock_universe(index, sample_n=sample_n, random_seed=random_seed)

    elif tool_name == "screen_stocks":
        tickers = tool_input.get("tickers", [])
        top_n = tool_input.get("top_n", 25)
        return market_data.screen_stocks(tickers, top_n)

    elif tool_name == "get_benchmark_comparison":
        return _handle_benchmark_comparison()

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
        portfolio.log_agent_message(
            f"BUY {shares:.4f} shares of {ticker} @ ${price:.2f} | Reason: {notes}"
        )
        if notes:
            portfolio.save_trade_thesis(result["transaction_id"], ticker, "BUY", notes)
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
