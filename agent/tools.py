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
    return result
