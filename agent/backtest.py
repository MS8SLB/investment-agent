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


# ── Public entry point ────────────────────────────────────────────────────────

def run_backtest(mode: str, tickers: Optional[list[str]] = None, holding_days: int = 90) -> dict:
    """
    Run a backtest in one of three modes.

    Args:
        mode: "trade_history" | "signal_cohorts" | "momentum"
        tickers: required for mode="momentum"; ignored otherwise
        holding_days: for mode="momentum", how far back the simulated entry was

    Returns:
        Structured backtest results dict.
    """
    if mode == "trade_history":
        return _trade_history_backtest()
    elif mode == "signal_cohorts":
        return _signal_cohort_analysis()
    elif mode == "momentum":
        if not tickers:
            return {"mode": "momentum", "error": "tickers list required for momentum mode"}
        return _momentum_backtest(tickers, holding_days=holding_days)
    else:
        return {"error": f"Unknown mode '{mode}'. Use: trade_history, signal_cohorts, or momentum"}
