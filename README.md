# AI Investment Portfolio Agent

An AI-powered paper trading portfolio manager built with Claude and real market data. Start with **$100,000 in virtual money** and let the AI build a long-term investment portfolio.

## Features

- **Claude-powered AI agent** — analyzes fundamentals, valuations, and market conditions
- **Real market data** — live stock prices and fundamentals via `yfinance` (free, no API key needed)
- **Paper trading** — buy/sell with fake money, no real money at risk
- **Persistent portfolio** — SQLite database stores your holdings and transaction history across sessions
- **Long-term strategy** — focuses on quality companies, diversification, and value discipline
- **Rich CLI** — beautiful terminal interface with color-coded P&L

## Investment Philosophy

The agent invests like a long-term value investor:
- 3–10 year time horizon
- Fundamental analysis (P/E, ROE, margins, debt levels)
- Diversification across sectors (8–15 positions)
- Max 20% of portfolio in any single stock
- Minimum 10% cash buffer
- Avoids panic selling on short-term volatility

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Run
python main.py
```

## Usage

```bash
# Interactive menu
python main.py

# Run a full AI portfolio review (agent researches + trades autonomously)
python main.py review

# View your current portfolio
python main.py portfolio

# Ask the agent anything
python main.py ask "Should I buy more tech stocks right now?"
python main.py ask "Analyze NVDA and tell me if it's a good buy"
python main.py ask "Rebalance the portfolio to reduce tech exposure"

# View transaction history
python main.py history

# View market summary
python main.py market
```

## Project Structure

```
investment-agent/
├── main.py                      # CLI entry point
├── requirements.txt
├── .env.example                 # API key config template
├── agent/
│   ├── investment_agent.py      # Claude agentic loop
│   ├── portfolio.py             # SQLite portfolio management
│   ├── market_data.py           # yfinance market data
│   └── tools.py                 # Agent tool definitions & handlers
├── utils/
│   └── display.py               # Rich terminal display
└── data/
    └── portfolio.db             # Persistent portfolio (auto-created)
```

## Agent Tools

The AI agent has access to these tools:

| Tool | Description |
|------|-------------|
| `get_portfolio_status` | Current holdings, cash, unrealized P&L |
| `get_stock_quote` | Live price and basic stats |
| `get_stock_fundamentals` | P/E, margins, ROE, debt, dividends, analyst ratings |
| `get_price_history` | Historical performance (1mo to 5y) |
| `search_stocks` | Find tickers by company name |
| `get_market_summary` | S&P 500, NASDAQ, Dow, VIX snapshot |
| `buy_stock` | Execute a paper buy order |
| `sell_stock` | Execute a paper sell order |
| `get_transaction_history` | Recent trade log |

## Example Session

```
$ python main.py review

AI Portfolio Review
The agent is analyzing markets and managing your portfolio...

  ⚙ Tool: get_portfolio_status
  ⚙ Tool: get_market_summary
    S&P 500: $5,890.23 (+0.34%)
  ⚙ Tool: get_stock_fundamentals
    AAPL
  ⚙ Tool: get_stock_quote
    Apple Inc (AAPL): $228.50
  ⚙ Tool: buy_stock
    ticker=AAPL, dollar_amount=10000
    Bought 43.7900 shares of AAPL @ $228.50 (Total: $10,008.65)

Based on my analysis, I've initiated a position in Apple (AAPL) at $228.50/share...
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Your Anthropic API key |
| `STARTING_CASH` | `100000` | Virtual starting balance |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Claude model to use |
