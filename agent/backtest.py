"""
Backtesting harness for the investment agent.

Three modes, each answering a distinct question:

  trade_history  — How have the agent's actual closed trades performed?
                   Computes win rate, avg return, Sharpe, max drawdown, and
                   benchmark comparison; segments by market regime at entry.

  signal_cohorts — Which signal combinations at entry time predicted winners?
                   Breaks closed trades into cohorts (PEG, FCF yield, momentum,
                   score) and compares win rates and avg returns across cohorts
                   and market regimes (bull vs. bear at entry).

  momentum       — Does a price-momentum screen of a given ticker list beat
                   buy-and-hold? Simulates buying top-momentum stocks at a
                   point in the past and measures forward returns vs. S&P 500.
                   Uses price data only — no look-ahead on fundamentals.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "portfolio.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _sp500_return(start_date: str, end_date: str) -> Optional[float]:
    """Return S&P 500 total return % between two ISO date strings."""
    try:
        hist = yf.Ticker("^GSPC").history(start=start_date, end=end_date)
        if len(hist) < 2:
            return None
        return round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100, 2)
    except Exception:
        return None


def _vix_on_date(date_str: str) -> Optional[float]:
    """Return VIX closing price on or just before a given date."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        hist = yf.Ticker("^VIX").history(start=start, end=date_str[:10])
        if hist.empty:
            return None
        return round(float(hist["Close"].iloc[-1]), 1)
    except Exception:
        return None


def _classify_regime(vix: Optional[float]) -> str:
    if vix is None:
        return "unknown"
    if vix < 15:
        return "low_volatility"
    if vix < 20:
        return "normal"
    if vix < 28:
        return "elevated"
    return "high_fear"


# ── Mode 1: Trade history analysis ────────────────────────────────────────────

def _trade_history_backtest() -> dict:
    """
    Replay ALL closed trades in the portfolio DB and compute full performance stats.

    For each closed trade: holding period, return %, benchmark return over same
    period, alpha. Then aggregate into portfolio-level Sharpe, win rate, drawdown,
    and avg alpha. Segments results by market regime (VIX level) at entry.
    """
    db = _conn()

    # Pull every buy with a matching sell (no LIMIT); join prediction_tracking for IV data
    rows = db.execute("""
        SELECT
            b.id AS buy_id, b.ticker, b.price AS buy_price, b.ts AS buy_date,
            s.price AS sell_price, s.ts AS sell_date,
            ts_sig.screener_score, ts_sig.peg_ratio, ts_sig.fcf_yield_pct,
            ts_sig.relative_momentum_pct, ts_sig.sector,
            pt.conviction_score, pt.predicted_iv, pt.mos_at_decision
        FROM transactions b
        JOIN transactions s
            ON  s.ticker = b.ticker
            AND s.action = 'SELL'
            AND s.ts > b.ts
        LEFT JOIN trade_signals ts_sig ON ts_sig.transaction_id = b.id
        LEFT JOIN prediction_tracking pt
               ON pt.ticker = b.ticker
              AND date(pt.decision_date) = date(b.ts)
        WHERE b.action = 'BUY'
        ORDER BY b.ts ASC
    """).fetchall()
    db.close()

    if not rows:
        return {
            "mode": "trade_history",
            "message": "No closed trades found. Run more portfolio review sessions first.",
            "closed_trade_count": 0,
        }

    # Deduplicate: keep only the first sell for each buy
    seen = set()
    trades = []
    for row in [dict(r) for r in rows]:
        key = (row["buy_id"],)
        if key not in seen:
            seen.add(key)
            trades.append(row)

    # Compute per-trade stats; fetch VIX at entry date
    enriched = []
    for t in trades:
        buy_dt = t["buy_date"][:10]
        sell_dt = t["sell_date"][:10]
        ret = (t["sell_price"] - t["buy_price"]) / t["buy_price"] * 100
        holding_days = (
            datetime.strptime(sell_dt, "%Y-%m-%d") - datetime.strptime(buy_dt, "%Y-%m-%d")
        ).days
        spy_ret = _sp500_return(buy_dt, sell_dt)
        alpha = round(ret - spy_ret, 2) if spy_ret is not None else None
        vix = _vix_on_date(buy_dt)
        regime = _classify_regime(vix)

        # IV accuracy: did the stock reach its predicted intrinsic value?
        predicted_iv = t.get("predicted_iv")
        reached_iv = bool(predicted_iv and t["sell_price"] >= predicted_iv)
        iv_capture_pct = (
            round((t["sell_price"] / predicted_iv - 1) * 100, 1)
            if predicted_iv else None
        )

        # Conviction label
        conv_score = t.get("conviction_score")
        if conv_score is None:
            conviction = "unknown"
        elif conv_score >= 8:
            conviction = "high"
        elif conv_score >= 5:
            conviction = "medium"
        else:
            conviction = "low"

        enriched.append({
            "ticker": t["ticker"],
            "buy_date": buy_dt,
            "sell_date": sell_dt,
            "buy_price": round(t["buy_price"], 2),
            "sell_price": round(t["sell_price"], 2),
            "return_pct": round(ret, 2),
            "holding_days": holding_days,
            "spy_return_pct": spy_ret,
            "alpha_pct": alpha,
            "vix_at_entry": vix,
            "regime_at_entry": regime,
            "screener_score": t["screener_score"],
            "peg_ratio": t["peg_ratio"],
            "fcf_yield_pct": t["fcf_yield_pct"],
            "sector": t["sector"] or "Unknown",
            "conviction": conviction,
            "conviction_score": conv_score,
            "predicted_iv": predicted_iv,
            "mos_at_decision": t.get("mos_at_decision"),
            "reached_iv": reached_iv,
            "iv_capture_pct": iv_capture_pct,
        })

    n = len(enriched)
    returns = [t["return_pct"] for t in enriched]
    winners = [r for r in returns if r > 0]
    win_rate = round(len(winners) / n * 100, 1)
    avg_return = round(sum(returns) / n, 2)
    avg_holding = round(sum(t["holding_days"] for t in enriched) / n, 0)

    # Sharpe from trade returns (annualised — assume 252 trading days)
    if n >= 2:
        avg_r = sum(returns) / n
        std_r = (sum((r - avg_r) ** 2 for r in returns) / (n - 1)) ** 0.5
        # Scale to annual assuming avg holding period
        avg_hold_yrs = avg_holding / 252
        periods_per_year = 1 / avg_hold_yrs if avg_hold_yrs > 0 else 12
        ann_return = ((1 + avg_r / 100) ** periods_per_year - 1) * 100
        ann_vol = std_r * (periods_per_year ** 0.5)
        sharpe = round((ann_return - 4.5) / ann_vol, 2) if ann_vol > 0 else None
    else:
        sharpe = None

    # Max drawdown from cumulative returns (equal-weight each trade)
    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cum *= (1 + r / 100)
        peak = max(peak, cum)
        max_dd = max(max_dd, (peak - cum) / peak)

    # Benchmark: total S&P 500 return across the full period
    if enriched:
        earliest = min(t["buy_date"] for t in enriched)
        latest = max(t["sell_date"] for t in enriched)
        spy_period = _sp500_return(earliest, latest)
    else:
        spy_period = None

    # Regime segmentation
    regime_stats: dict[str, list] = {}
    for t in enriched:
        r = t["regime_at_entry"]
        regime_stats.setdefault(r, []).append(t["return_pct"])

    regime_breakdown = {}
    for regime, rets in sorted(regime_stats.items()):
        pos = sum(1 for r in rets if r > 0)
        regime_breakdown[regime] = {
            "trade_count": len(rets),
            "win_rate_pct": round(pos / len(rets) * 100, 1),
            "avg_return_pct": round(sum(rets) / len(rets), 2),
        }

    # Best and worst trades
    enriched_sorted = sorted(enriched, key=lambda t: t["return_pct"])
    worst = [{"ticker": t["ticker"], "return_pct": t["return_pct"], "buy_date": t["buy_date"],
              "conviction": t["conviction"], "sector": t["sector"]}
             for t in enriched_sorted[:3]]
    best = [{"ticker": t["ticker"], "return_pct": t["return_pct"], "buy_date": t["buy_date"],
             "conviction": t["conviction"], "sector": t["sector"]}
            for t in enriched_sorted[-3:]][::-1]

    # Conviction breakdown
    conviction_groups: dict[str, list] = {}
    for t in enriched:
        conviction_groups.setdefault(t["conviction"], []).append(t["return_pct"])

    def _group_stats(returns_list: list) -> dict:
        if not returns_list:
            return {"count": 0}
        wins = sum(1 for r in returns_list if r > 0)
        avg = sum(returns_list) / len(returns_list)
        return {
            "count": len(returns_list),
            "win_rate_pct": round(wins / len(returns_list) * 100, 1),
            "avg_return_pct": round(avg, 2),
        }

    conviction_breakdown = {k: _group_stats(v) for k, v in conviction_groups.items()}

    # Calibration check: high > medium > low?
    h = conviction_breakdown.get("high", {}).get("avg_return_pct")
    m = conviction_breakdown.get("medium", {}).get("avg_return_pct")
    lo = conviction_breakdown.get("low", {}).get("avg_return_pct")
    if h is not None and m is not None and lo is not None:
        calibration = "well_calibrated" if h > m > lo else "miscalibrated"
    elif h is not None and m is not None:
        calibration = "well_calibrated" if h > m else "miscalibrated"
    else:
        calibration = "insufficient_data"

    # Sector breakdown
    sector_groups: dict[str, list] = {}
    for t in enriched:
        sector_groups.setdefault(t["sector"], []).append(t["return_pct"])
    sector_breakdown = {k: _group_stats(v) for k, v in sector_groups.items()}

    # IV accuracy
    iv_trades = [t for t in enriched if t["predicted_iv"] is not None]
    if iv_trades:
        reached = sum(1 for t in iv_trades if t["reached_iv"])
        iv_accuracy = {
            "trades_with_iv_target": len(iv_trades),
            "pct_reached_iv": round(reached / len(iv_trades) * 100, 1),
            "avg_iv_capture_pct": round(
                sum(t["iv_capture_pct"] for t in iv_trades if t["iv_capture_pct"] is not None)
                / len([t for t in iv_trades if t["iv_capture_pct"] is not None]), 1
            ) if any(t["iv_capture_pct"] is not None for t in iv_trades) else None,
            "note": (
                "iv_capture_pct > 0 means sold above predicted intrinsic value; "
                "< 0 means sold before reaching IV target."
            ),
        }
    else:
        iv_accuracy = {
            "trades_with_iv_target": 0,
            "note": "No IV targets set yet. Add predicted_iv when making buy decisions.",
        }

    return {
        "mode": "trade_history",
        "closed_trade_count": n,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_return,
        "avg_holding_days": int(avg_holding),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe_ratio": sharpe,
        "portfolio_period_return_pct": round(sum(returns), 2),
        "sp500_period_return_pct": spy_period,
        "regime_breakdown": regime_breakdown,
        "conviction_breakdown": conviction_breakdown,
        "calibration_status": calibration,
        "sector_breakdown": sector_breakdown,
        "iv_accuracy": iv_accuracy,
        "best_trades": best,
        "worst_trades": worst,
        "all_trades": enriched,
    }


# ── Mode 2: Signal cohort analysis ────────────────────────────────────────────

def _signal_cohort_analysis() -> dict:
    """
    Break ALL closed trades into signal cohorts and compare win rates.

    Cohorts:
      - PEG < 1.5 vs PEG ≥ 1.5
      - FCF yield > 3% vs ≤ 3%
      - Positive momentum vs negative
      - Screener score ≥ 8 vs < 8
      - Bull entry (VIX < 20) vs Bear entry (VIX ≥ 20)

    Each combination shows trade count, win rate %, avg return %.
    """
    db = _conn()
    rows = db.execute("""
        SELECT
            b.ticker, b.price AS buy_price, b.ts AS buy_date,
            s.price AS sell_price,
            ts_sig.screener_score, ts_sig.peg_ratio, ts_sig.fcf_yield_pct,
            ts_sig.relative_momentum_pct, ts_sig.revenue_growth_pct,
            ts_sig.profit_margin_pct, ts_sig.roe_pct, ts_sig.sector
        FROM transactions b
        JOIN transactions s ON s.ticker = b.ticker AND s.action = 'SELL' AND s.ts > b.ts
        LEFT JOIN trade_signals ts_sig ON ts_sig.transaction_id = b.id
        WHERE b.action = 'BUY'
        ORDER BY b.ts ASC
    """).fetchall()
    db.close()

    if not rows:
        return {"mode": "signal_cohorts", "message": "No closed trades found.", "closed_trade_count": 0}

    seen = set()
    trades = []
    for row in [dict(r) for r in rows]:
        k = (row["ticker"], row["buy_date"])
        if k not in seen:
            seen.add(k)
            trades.append(row)

    # Enrich with return and regime
    for t in trades:
        t["return_pct"] = (t["sell_price"] - t["buy_price"]) / t["buy_price"] * 100
        vix = _vix_on_date(t["buy_date"][:10])
        t["vix_at_entry"] = vix
        t["bull_entry"] = vix is not None and vix < 20

    def _cohort_stats(subset: list) -> dict:
        if not subset:
            return {"count": 0, "win_rate_pct": None, "avg_return_pct": None}
        rets = [t["return_pct"] for t in subset]
        wins = sum(1 for r in rets if r > 0)
        return {
            "count": len(subset),
            "win_rate_pct": round(wins / len(subset) * 100, 1),
            "avg_return_pct": round(sum(rets) / len(rets), 2),
        }

    cohort_definitions = {
        "peg_lt_1_5":           lambda t: t.get("peg_ratio") is not None and t["peg_ratio"] < 1.5,
        "peg_gte_1_5":          lambda t: t.get("peg_ratio") is not None and t["peg_ratio"] >= 1.5,
        "fcf_yield_gt_3pct":    lambda t: t.get("fcf_yield_pct") is not None and t["fcf_yield_pct"] > 3.0,
        "fcf_yield_lte_3pct":   lambda t: t.get("fcf_yield_pct") is not None and t["fcf_yield_pct"] <= 3.0,
        "positive_momentum":    lambda t: t.get("relative_momentum_pct") is not None and t["relative_momentum_pct"] > 0,
        "negative_momentum":    lambda t: t.get("relative_momentum_pct") is not None and t["relative_momentum_pct"] <= 0,
        "score_gte_8":          lambda t: t.get("screener_score") is not None and t["screener_score"] >= 8,
        "score_lt_8":           lambda t: t.get("screener_score") is not None and t["screener_score"] < 8,
        "bull_entry_vix_lt_20": lambda t: t.get("bull_entry") is True,
        "bear_entry_vix_gte_20": lambda t: t.get("bull_entry") is False,
    }

    cohorts = {}
    for name, fn in cohort_definitions.items():
        subset = [t for t in trades if fn(t)]
        cohorts[name] = _cohort_stats(subset)

    # Best signal combo: PEG < 1.5 AND FCF yield > 3% AND bull entry
    best_combo = [t for t in trades
                  if (t.get("peg_ratio") or 99) < 1.5
                  and (t.get("fcf_yield_pct") or 0) > 3.0
                  and t.get("bull_entry") is True]
    cohorts["peg_lt_1_5_AND_fcf_gt_3pct_AND_bull"] = _cohort_stats(best_combo)

    return {
        "mode": "signal_cohorts",
        "closed_trade_count": len(trades),
        "cohorts": cohorts,
        "interpretation": (
            "Compare win_rate_pct and avg_return_pct across cohorts to identify "
            "which signal thresholds have historically separated winners from losers. "
            "Cohorts with count < 5 should be treated cautiously — insufficient sample. "
            "bull_entry vs bear_entry split shows whether regime timing matters."
        ),
    }


# ── Mode 3: Price-momentum strategy backtest ──────────────────────────────────

def _momentum_backtest(tickers: list[str], holding_days: int = 90) -> dict:
    """
    Simulate a price-momentum strategy on the given ticker list.

    At a point `holding_days` ago, rank the provided tickers by their
    prior-12-month return (standard momentum signal, excluding last month
    to avoid short-term reversal). Buy the top third. Measure actual return
    from that simulated entry to today. Compare to S&P 500 buy-and-hold.

    Uses price data only — no look-ahead on fundamentals.

    holding_days: how far back the simulated entry was (default 90 days = ~1 quarter)
    """
    if not tickers:
        return {"mode": "momentum", "error": "No tickers provided"}
    if holding_days < 30:
        return {"mode": "momentum", "error": "holding_days must be >= 30"}

    today = datetime.now().date()
    entry_date = today - timedelta(days=holding_days)
    lookback_start = entry_date - timedelta(days=365)

    entry_str = entry_date.strftime("%Y-%m-%d")
    lookback_str = lookback_start.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    results = []
    errors = []

    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(start=lookback_str, end=today_str)
            if len(hist) < 60:
                errors.append({"ticker": ticker, "reason": "insufficient history"})
                continue

            # Price at lookback start, entry date (~1yr before entry), and today
            # Filter to before entry for momentum calculation
            pre_entry = hist[hist.index.date <= entry_date]
            post_entry = hist[hist.index.date >= entry_date]

            if len(pre_entry) < 21 or len(post_entry) < 2:
                errors.append({"ticker": ticker, "reason": "insufficient pre/post entry data"})
                continue

            # Momentum: 12M return excluding last month (standard momentum signal)
            # Use price from ~252 days before entry to ~21 days before entry
            lookback_prices = pre_entry["Close"]
            if len(lookback_prices) < 42:
                errors.append({"ticker": ticker, "reason": "need at least 42 days pre-entry"})
                continue

            momentum_start_price = float(lookback_prices.iloc[0])
            momentum_end_price = float(lookback_prices.iloc[-22])  # exclude last month
            momentum = (momentum_end_price - momentum_start_price) / momentum_start_price * 100

            entry_price = float(post_entry["Close"].iloc[0])
            current_price = float(post_entry["Close"].iloc[-1])
            forward_return = (current_price - entry_price) / entry_price * 100

            results.append({
                "ticker": ticker,
                "momentum_12m_pct": round(momentum, 1),
                "entry_price": round(entry_price, 2),
                "current_price": round(current_price, 2),
                "forward_return_pct": round(forward_return, 2),
            })
        except Exception as e:
            errors.append({"ticker": ticker, "reason": str(e)})

    if not results:
        return {
            "mode": "momentum",
            "error": "Could not fetch price data for any ticker",
            "ticker_errors": errors,
        }

    # Sort by momentum and pick top third
    results.sort(key=lambda r: r["momentum_12m_pct"], reverse=True)
    top_n = max(1, len(results) // 3)
    top_momentum = results[:top_n]
    bottom_momentum = results[top_n:]

    def _stats(group: list) -> dict:
        if not group:
            return {}
        rets = [t["forward_return_pct"] for t in group]
        wins = sum(1 for r in rets if r > 0)
        avg_r = sum(rets) / len(rets)
        std_r = (sum((r - avg_r) ** 2 for r in rets) / max(len(rets) - 1, 1)) ** 0.5
        return {
            "count": len(group),
            "win_rate_pct": round(wins / len(group) * 100, 1),
            "avg_return_pct": round(avg_r, 2),
            "std_return_pct": round(std_r, 2),
            "best_return_pct": round(max(rets), 2),
            "worst_return_pct": round(min(rets), 2),
        }

    spy_return = _sp500_return(entry_str, today_str)

    top_stats = _stats(top_momentum)
    bottom_stats = _stats(bottom_momentum)

    return {
        "mode": "momentum",
        "simulation_entry_date": entry_str,
        "simulation_end_date": today_str,
        "holding_days": holding_days,
        "tickers_analyzed": len(results),
        "ticker_errors": len(errors),
        "sp500_return_pct": spy_return,
        "top_momentum_portfolio": {
            **top_stats,
            "alpha_vs_spy_pct": round(top_stats.get("avg_return_pct", 0) - (spy_return or 0), 2),
            "tickers": top_momentum,
        },
        "bottom_momentum_portfolio": {
            **bottom_stats,
            "alpha_vs_spy_pct": round(bottom_stats.get("avg_return_pct", 0) - (spy_return or 0), 2),
        },
        "momentum_premium_pct": round(
            top_stats.get("avg_return_pct", 0) - bottom_stats.get("avg_return_pct", 0), 2
        ),
        "interpretation": (
            f"Top-momentum tercile returned {top_stats.get('avg_return_pct')}% avg vs "
            f"{bottom_stats.get('avg_return_pct')}% for bottom tercile over the {holding_days}-day hold. "
            f"S&P 500 returned {spy_return}% over the same period. "
            "Positive momentum_premium_pct confirms momentum effect in this universe. "
            "Use this to weight relative_momentum_pct more heavily in screening when the "
            "premium is consistently positive across multiple backtest windows."
        ),
    }


# ── Mode 4: Fundamental history backtest (FMP-powered) ───────────────────────

def _score_quarter(q_inc: dict, q_bal: dict, q_cf: dict,
                   prev_inc: Optional[dict], price: float) -> tuple[float, dict]:
    """
    Compute the screener score for one historical quarter using the same
    logic as screen_stocks(), but limited to what FMP historical statements
    provide. Returns (score, signal_dict).
    """
    score = 0.0
    signals: dict = {}

    revenue      = q_inc.get("revenue") or 0
    gross_profit = q_inc.get("grossProfit") or 0
    net_income   = q_inc.get("netIncome") or 0
    op_income    = q_inc.get("operatingIncome") or 0
    op_cf        = q_cf.get("operatingCashFlow") or 0
    capex        = q_cf.get("capitalExpenditure") or 0      # usually negative in FMP
    total_debt   = q_bal.get("totalDebt") or 0
    equity       = q_bal.get("totalStockholdersEquity") or 1
    shares       = q_bal.get("commonStock") or q_bal.get("sharesOutstanding") or 0
    current_assets  = q_bal.get("totalCurrentAssets") or 0
    current_liabs   = q_bal.get("totalCurrentLiabilities") or 1
    eps          = q_inc.get("eps") or q_inc.get("epsdiluted") or None

    # Revenue growth YoY (requires prior year same quarter)
    revenue_growth = None
    if prev_inc and revenue and prev_inc.get("revenue"):
        revenue_growth = (revenue - prev_inc["revenue"]) / abs(prev_inc["revenue"])
        signals["revenue_growth"] = round(revenue_growth * 100, 1)
        if revenue_growth > 0.08:
            score += 2
        if revenue_growth > 0.20:
            score += 1

    # Net profit margin
    profit_margin = net_income / revenue if revenue else None
    if profit_margin is not None:
        signals["profit_margin_pct"] = round(profit_margin * 100, 1)
        if profit_margin > 0.10:
            score += 2

    # ROE
    roe = net_income / equity if equity else None
    if roe is not None:
        signals["roe_pct"] = round(roe * 100, 1)
        if roe > 0.15:
            score += 2

    # P/E ratio (trailing, using quarter EPS annualised)
    pe = None
    if eps and eps > 0 and price:
        pe = price / (eps * 4)
        signals["pe"] = round(pe, 1)
        if 5 < pe < 20:
            score += 2
        elif pe < 30:
            score += 1

    # Debt / equity
    de = total_debt / equity if equity else None
    if de is not None:
        signals["debt_to_equity"] = round(de, 2)
        if de < 1.0:
            score += 1

    # FCF yield (market cap = price * shares if available)
    market_cap = price * shares if (price and shares) else None
    fcf = op_cf + capex if capex < 0 else op_cf - capex
    fcf_yield = fcf / market_cap if (market_cap and market_cap > 0) else None
    if fcf_yield is not None:
        signals["fcf_yield_pct"] = round(fcf_yield * 100, 1)
        if fcf_yield > 0.05:
            score += 2
        elif fcf_yield > 0.02:
            score += 1

    # Current ratio
    cr = current_assets / current_liabs if current_liabs else None
    if cr is not None:
        signals["current_ratio"] = round(cr, 2)

    signals["score"] = score
    return score, signals


def _get_price_on_date(ticker: str, date_str: str,
                       price_cache: dict) -> Optional[float]:
    """Return closing price on or just after date_str. Caches full history per ticker."""
    if ticker not in price_cache:
        try:
            hist = yf.Ticker(ticker).history(period="max")
            price_cache[ticker] = hist["Close"] if not hist.empty else None
        except Exception:
            price_cache[ticker] = None

    series = price_cache.get(ticker)
    if series is None or series.empty:
        return None

    target = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    # Find the first trading day on or after target
    for ts, price in series.items():
        ts_date = ts.date() if hasattr(ts, "date") else ts
        if ts_date >= target:
            return round(float(price), 2)
    return None


def _fundamental_history_backtest(
    tickers: Optional[list[str]] = None,
    start_year: int = 2015,
    score_threshold: float = 6.0,
    holding_quarters: int = 4,
    max_tickers: int = 20,
) -> dict:
    """
    Simulate the agent's screener running at every quarter-end since start_year
    using real historical fundamentals from FMP.

    For each quarter where a ticker scores ≥ score_threshold:
      - Record a simulated buy at that quarter's closing price
      - Measure 1q / 2q / 4q forward returns vs S&P 500

    Answers: "Would this screening approach have generated alpha historically?"
    """
    import os
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        return {
            "mode": "fundamental_history",
            "error": "FMP_API_KEY not set. This mode requires Financial Modeling Prep API access.",
        }

    # Default universe: tickers from universe_scores DB + any passed in
    db = _conn()
    db_tickers = [r[0] for r in db.execute(
        "SELECT ticker FROM universe_scores ORDER BY quality_score DESC LIMIT 50"
    ).fetchall()]
    db.close()

    universe = list(dict.fromkeys((tickers or []) + db_tickers))[:max_tickers]
    if not universe:
        return {"mode": "fundamental_history", "error": "No tickers to backtest. Pass tickers or run screener first."}

    import requests
    FMP_BASE = "https://financialmodelingprep.com/api/v3"

    def _fmp(path, **params):
        params["apikey"] = fmp_key
        try:
            r = requests.get(f"{FMP_BASE}/{path}", params=params, timeout=15)
            return r.json() if r.status_code == 200 else []
        except Exception:
            return []

    price_cache: dict = {}
    signals_by_quarter: list[dict] = []
    tickers_processed = 0
    tickers_skipped = 0

    for ticker in universe:
        inc_rows  = _fmp(f"income-statement/{ticker}",        period="quarter", limit=40)
        bal_rows  = _fmp(f"balance-sheet-statement/{ticker}", period="quarter", limit=40)
        cf_rows   = _fmp(f"cash-flow-statement/{ticker}",     period="quarter", limit=40)

        if not inc_rows or len(inc_rows) < 5:
            tickers_skipped += 1
            continue

        # Index by date for alignment
        bal_map = {r["date"]: r for r in bal_rows}
        cf_map  = {r["date"]: r for r in cf_rows}

        # Newest first → reverse for chronological iteration
        inc_chrono = list(reversed(inc_rows))

        tickers_processed += 1

        for idx, q_inc in enumerate(inc_chrono):
            date_str = q_inc.get("date", "")
            if not date_str or int(date_str[:4]) < start_year:
                continue

            q_bal = bal_map.get(date_str, {})
            q_cf  = cf_map.get(date_str, {})
            # Same quarter prior year for YoY growth
            prev_inc = inc_chrono[idx - 4] if idx >= 4 else None

            price = _get_price_on_date(ticker, date_str, price_cache)
            if not price:
                continue

            score, signal_detail = _score_quarter(q_inc, q_bal, q_cf, prev_inc, price)
            if score < score_threshold:
                continue

            # Measure forward returns at 1q (~63 days), 2q (~126 days), 4q (~252 days)
            fwd_returns: dict = {}
            spy_returns: dict = {}
            for label, days in [("1q", 63), ("2q", 126), ("4q", 252)]:
                fwd_date = (datetime.strptime(date_str[:10], "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")
                fwd_price = _get_price_on_date(ticker, fwd_date, price_cache)
                if fwd_price:
                    fwd_returns[label] = round((fwd_price / price - 1) * 100, 2)
                spy_ret = _sp500_return(date_str[:10], fwd_date)
                if spy_ret is not None:
                    spy_returns[label] = spy_ret

            vix   = _vix_on_date(date_str[:10])
            regime = _classify_regime(vix)

            signals_by_quarter.append({
                "ticker":   ticker,
                "date":     date_str[:10],
                "price":    price,
                "score":    score,
                "signals":  signal_detail,
                "regime":   regime,
                "vix":      vix,
                "fwd_returns":  fwd_returns,
                "spy_returns":  spy_returns,
                "alphas": {
                    k: round(fwd_returns[k] - spy_returns.get(k, 0), 2)
                    for k in fwd_returns
                },
            })

    if not signals_by_quarter:
        return {
            "mode": "fundamental_history",
            "tickers_processed": tickers_processed,
            "tickers_skipped":   tickers_skipped,
            "message": f"No buy signals generated above score threshold {score_threshold}. "
                       "Try lowering score_threshold or adding more tickers.",
        }

    # ── Aggregate stats ──────────────────────────────────────────────────────

    def _agg(subset: list, horizon: str = "4q") -> dict:
        rets = [s["fwd_returns"].get(horizon) for s in subset if horizon in s["fwd_returns"]]
        alps = [s["alphas"].get(horizon) for s in subset if horizon in s["alphas"]]
        if not rets:
            return {"signals": len(subset), "insufficient_data": True}
        wins = sum(1 for r in rets if r > 0)
        return {
            "signals":        len(subset),
            "win_rate_pct":   round(wins / len(rets) * 100, 1),
            "avg_return_pct": round(sum(rets) / len(rets), 2),
            "avg_alpha_pct":  round(sum(alps) / len(alps), 2) if alps else None,
            "best_pct":       round(max(rets), 2),
            "worst_pct":      round(min(rets), 2),
        }

    overall_1q = _agg(signals_by_quarter, "1q")
    overall_2q = _agg(signals_by_quarter, "2q")
    overall_4q = _agg(signals_by_quarter, "4q")

    # By score bucket
    high_score  = [s for s in signals_by_quarter if s["score"] >= 9]
    med_score   = [s for s in signals_by_quarter if 6 <= s["score"] < 9]

    # By regime
    by_regime: dict[str, list] = {}
    for s in signals_by_quarter:
        by_regime.setdefault(s["regime"], []).append(s)
    regime_breakdown = {k: _agg(v, "4q") for k, v in by_regime.items()}

    # By ticker
    by_ticker: dict[str, list] = {}
    for s in signals_by_quarter:
        by_ticker.setdefault(s["ticker"], []).append(s)
    ticker_breakdown = {k: _agg(v, "4q") for k, v in by_ticker.items()}

    # Best / worst individual signals (4q horizon)
    ranked = sorted(
        [s for s in signals_by_quarter if "4q" in s["fwd_returns"]],
        key=lambda s: s["fwd_returns"]["4q"],
    )
    best_signals  = [{"ticker": s["ticker"], "date": s["date"], "score": s["score"],
                      "return_4q_pct": s["fwd_returns"]["4q"], "alpha_4q_pct": s["alphas"].get("4q")}
                     for s in ranked[-5:][::-1]]
    worst_signals = [{"ticker": s["ticker"], "date": s["date"], "score": s["score"],
                      "return_4q_pct": s["fwd_returns"]["4q"], "alpha_4q_pct": s["alphas"].get("4q")}
                     for s in ranked[:5]]

    avg_alpha_4q = overall_4q.get("avg_alpha_pct")
    summary = (
        f"Fundamental history backtest: {len(signals_by_quarter)} buy signals across "
        f"{tickers_processed} tickers since {start_year} (score ≥ {score_threshold}). "
        f"4-quarter forward return: avg {overall_4q.get('avg_return_pct', 'n/a')}%, "
        f"win rate {overall_4q.get('win_rate_pct', 'n/a')}%, "
        f"avg alpha vs SPY {avg_alpha_4q:+.1f}%." if avg_alpha_4q is not None
        else f"Processed {tickers_processed} tickers."
    )

    return {
        "mode": "fundamental_history",
        "parameters": {
            "start_year": start_year,
            "score_threshold": score_threshold,
            "holding_quarters": holding_quarters,
            "tickers_processed": tickers_processed,
            "tickers_skipped": tickers_skipped,
            "total_signals": len(signals_by_quarter),
        },
        "overall": {
            "1_quarter_hold": overall_1q,
            "2_quarter_hold": overall_2q,
            "4_quarter_hold": overall_4q,
        },
        "score_buckets": {
            "high_score_gte_9": _agg(high_score,  "4q"),
            "med_score_6_to_9": _agg(med_score,   "4q"),
        },
        "regime_breakdown": regime_breakdown,
        "ticker_breakdown":  ticker_breakdown,
        "best_signals":      best_signals,
        "worst_signals":     worst_signals,
        "summary":           summary,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_backtest(
    mode: str,
    tickers: Optional[list[str]] = None,
    holding_days: int = 90,
    start_year: int = 2015,
    score_threshold: float = 6.0,
    holding_quarters: int = 4,
    max_tickers: int = 20,
) -> dict:
    """
    Run a backtest in one of four modes.

    Args:
        mode: "trade_history" | "signal_cohorts" | "momentum" | "fundamental_history"
        tickers: for mode="momentum" (required) or "fundamental_history" (optional)
        holding_days: for mode="momentum"
        start_year: for mode="fundamental_history" (default 2015)
        score_threshold: for mode="fundamental_history" (default 6.0)
        holding_quarters: for mode="fundamental_history" (default 4)
        max_tickers: for mode="fundamental_history" (default 20, stays within FMP free tier)
    """
    if mode == "trade_history":
        return _trade_history_backtest()
    elif mode == "signal_cohorts":
        return _signal_cohort_analysis()
    elif mode == "momentum":
        if not tickers:
            return {"mode": "momentum", "error": "tickers list required for momentum mode"}
        return _momentum_backtest(tickers, holding_days=holding_days)
    elif mode == "fundamental_history":
        return _fundamental_history_backtest(
            tickers=tickers,
            start_year=start_year,
            score_threshold=score_threshold,
            holding_quarters=holding_quarters,
            max_tickers=max_tickers,
        )
    else:
        return {"error": f"Unknown mode '{mode}'. Use: trade_history, signal_cohorts, momentum, fundamental_history"}
