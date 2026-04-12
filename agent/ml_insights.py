"""
ML insights derived entirely from the agent's own historical trade data.

Three tools — zero new data sources required, everything is already in portfolio.db:

  get_ml_factor_weights()
      Learns which screener signals have actually predicted returns in THIS
      portfolio's history.  Returns data-driven factor weights blended with
      regime-adjusted priors (literature-backed).  Works immediately — returns
      pure regime priors when no closed trades exist, then transitions smoothly
      to data-driven weights as the trade history grows.

  prioritize_watchlist_ml()
      Fetches current fundamentals for every watchlist item, scores each using
      the learned factor weights, and returns a ranked list that surfaces the
      highest-conviction candidates to research first.

  get_position_size_recommendation(ticker, features)
      Estimates drawdown risk from historical patterns and the current feature
      set, then recommends a position size (% of portfolio).  Larger positions
      when features resemble past winners; smaller when risk flags are present.
"""

import numpy as np
import yfinance as yf
from typing import Optional

from agent import portfolio, market_data


# ══════════════════════════════════════════════════════════════════════════════
# Regime-prior factor weights  (finance-literature based)
# ══════════════════════════════════════════════════════════════════════════════
#
# Features (all transformed so HIGHER = MORE ATTRACTIVE):
#   peg_inv            = 1 / max(peg, 0.1)    lower PEG → higher score
#   fcf_yield_pct      raw                     higher FCF yield → better
#   relative_momentum  raw                     positive momentum preferred
#   revenue_growth_pct raw                     higher growth → better
#   profit_margin_pct  raw                     higher margin → better
#   roe_pct            raw                     higher ROE → better
#
_FEATURES = [
    "peg_inv",
    "fcf_yield_pct",
    "relative_momentum_pct",
    "revenue_growth_pct",
    "profit_margin_pct",
    "roe_pct",
]

_REGIME_PRIORS: dict[str, dict[str, float]] = {
    # Risk-on: low VIX, solid GDP — momentum & growth outperform
    "RISK_ON": {
        "peg_inv":               0.12,
        "fcf_yield_pct":         0.15,
        "relative_momentum_pct": 0.28,
        "revenue_growth_pct":    0.25,
        "profit_margin_pct":     0.13,
        "roe_pct":               0.07,
    },
    # Risk-off: high VIX / recession — quality & cash generation dominate
    "RISK_OFF": {
        "peg_inv":               0.22,
        "fcf_yield_pct":         0.30,
        "relative_momentum_pct": 0.08,
        "revenue_growth_pct":    0.10,
        "profit_margin_pct":     0.20,
        "roe_pct":               0.10,
    },
    # Inflationary: high CPI — value & real cash flow beat long-duration growth
    "INFLATIONARY": {
        "peg_inv":               0.28,
        "fcf_yield_pct":         0.27,
        "relative_momentum_pct": 0.12,
        "revenue_growth_pct":    0.10,
        "profit_margin_pct":     0.16,
        "roe_pct":               0.07,
    },
    # Stagflation: contraction + high inflation — maximum quality & value discipline
    "STAGFLATION": {
        "peg_inv":               0.30,
        "fcf_yield_pct":         0.33,
        "relative_momentum_pct": 0.05,
        "revenue_growth_pct":    0.05,
        "profit_margin_pct":     0.19,
        "roe_pct":               0.08,
    },
    # Normal / baseline
    "NORMAL": {
        "peg_inv":               0.20,
        "fcf_yield_pct":         0.22,
        "relative_momentum_pct": 0.20,
        "revenue_growth_pct":    0.20,
        "profit_margin_pct":     0.12,
        "roe_pct":               0.06,
    },
}

# Base position size (% of portfolio) per regime before risk adjustment
_REGIME_BASE_POSITION = {
    "RISK_ON":     15.0,
    "NORMAL":      12.0,
    "INFLATIONARY": 10.0,
    "RISK_OFF":     8.0,
    "STAGFLATION":  6.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _detect_regime() -> tuple[str, str, dict]:
    """
    Classify the current macro regime using live market data.
    Returns (regime_name, rationale_string, raw_macro_dict).
    """
    try:
        macro = market_data.get_macro_environment()
    except Exception:
        return "NORMAL", "Could not fetch macro data — using balanced defaults", {}

    vix            = (macro.get("sentiment") or {}).get("vix")
    rates          = macro.get("rates") or {}
    yield_spread   = rates.get("yield_curve_spread")          # 10yr - 2yr (%)
    ten_yr         = rates.get("ten_yr_treasury_yield_pct")
    yield_inverted = yield_spread is not None and yield_spread < -0.3

    rationale_parts = []
    if vix:
        rationale_parts.append(f"VIX={vix:.1f}")
    if yield_spread is not None:
        rationale_parts.append(f"yield spread={yield_spread:.2f}%")

    # Also check FRED economic data if available
    gdp_growth  = None
    core_cpi    = None
    try:
        from agent import external_data
        econ = external_data.get_economic_indicators()
        if "error" not in econ:
            gdp_growth = (econ.get("indicators", {})
                          .get("real_gdp_growth_pct", {})
                          .get("value"))
            core_cpi   = (econ.get("indicators", {})
                          .get("core_cpi_yoy_pct", {})
                          .get("yoy_pct"))
            if gdp_growth is not None:
                rationale_parts.append(f"GDP={gdp_growth:.1f}%")
            if core_cpi is not None:
                rationale_parts.append(f"CoreCPI={core_cpi:.1f}%YoY")
    except Exception:
        pass

    # Classification logic
    high_vix   = vix is not None and vix > 25
    low_vix    = vix is not None and vix < 16
    recession  = gdp_growth is not None and gdp_growth < 0
    slow_growth = gdp_growth is not None and gdp_growth < 1.5
    high_infl  = core_cpi is not None and core_cpi > 3.5
    high_rates = ten_yr is not None and ten_yr > 4.5

    if (recession or high_vix) and high_infl:
        regime = "STAGFLATION"
    elif recession or (high_vix and yield_inverted):
        regime = "RISK_OFF"
    elif high_infl or (high_rates and not low_vix):
        regime = "INFLATIONARY"
    elif low_vix and not yield_inverted and not high_infl:
        regime = "RISK_ON"
    else:
        regime = "NORMAL"

    rationale = f"{regime}: " + (", ".join(rationale_parts) if rationale_parts else "baseline indicators")
    return regime, rationale, macro


def _to_feature_vec(row: dict) -> Optional[np.ndarray]:
    """
    Convert a trade_signals / screener row into a feature vector.
    Returns None if too many values are missing.
    """
    peg = row.get("peg_ratio")
    vals = {
        "peg_inv":               1.0 / max(peg, 0.1) if peg and peg > 0 else None,
        "fcf_yield_pct":         row.get("fcf_yield_pct"),
        "relative_momentum_pct": row.get("relative_momentum_pct"),
        "revenue_growth_pct":    row.get("revenue_growth_pct"),
        "profit_margin_pct":     row.get("profit_margin_pct"),
        "roe_pct":                row.get("roe_pct"),
    }
    missing = sum(1 for v in vals.values() if v is None)
    if missing > 3:
        return None
    return np.array([vals[f] if vals[f] is not None else 0.0 for f in _FEATURES])


def _normalize_cols(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise columns; return (X_norm, mean, std)."""
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mu) / std, mu, std


def _blend_weights(prior: dict, learned: dict, n_trades: int) -> dict:
    """
    Blend regime-prior weights with data-driven learned weights.
    The learned component grows from 0 → 0.75 as n_trades goes 5 → 25+.
    """
    if n_trades < 5:
        return prior
    alpha = min(0.75, (n_trades - 5) / 20 * 0.75)   # 0 at 5 trades, 0.75 at 25+
    blended = {}
    for f in _FEATURES:
        blended[f] = (1 - alpha) * prior.get(f, 0) + alpha * learned.get(f, 0)
    total = sum(blended.values())
    return {f: round(v / total, 4) for f, v in blended.items()} if total else prior


def _score_features(features: dict, weights: dict) -> float:
    """Score a screener row using the given weights. Returns 0–10 score."""
    vec = _to_feature_vec(features)
    if vec is None:
        return 0.0
    raw_score = sum(vec[i] * weights.get(f, 0) for i, f in enumerate(_FEATURES))
    # Clamp to a 0-10 scale using a soft sigmoid-like mapping
    return round(min(10.0, max(0.0, raw_score * 5 + 5)), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Factor Weight Learning
# ══════════════════════════════════════════════════════════════════════════════

def get_ml_factor_weights() -> dict:
    """
    Learn which screener signals have actually predicted returns in this portfolio.

    Algorithm:
    1. Detect current macro regime → select regime-prior weights.
    2. Fetch closed trade history from portfolio.db.
    3. If ≥ 5 closed trades: fit Ridge regression on (features → return_pct)
       and compute per-feature importance from coefficients + correlations.
    4. Blend regime priors with learned weights — the data-driven component
       grows from 0% at 5 trades to 75% at 25+ trades.
    5. Return weights, per-feature diagnostics, and actionable guidance.

    Use these weights when interpreting screener results — features with high
    learned weight are the ones that have actually predicted returns for YOU.
    """
    regime, regime_rationale, _ = _detect_regime()
    prior_weights = _REGIME_PRIORS[regime]

    # ── Fetch closed trade history ────────────────────────────────────────────
    outcomes = portfolio.get_trade_outcomes()
    closed   = [t for t in outcomes if t.get("status") == "closed" and t.get("return_pct") is not None]

    result_base = {
        "regime":           regime,
        "regime_rationale": regime_rationale,
        "n_closed_trades":  len(closed),
        "regime_weights":   prior_weights,
    }

    if len(closed) < 5:
        return {
            **result_base,
            "data_confidence":  "low",
            "blended_weights":  prior_weights,
            "learned_weights":  None,
            "feature_correlations": None,
            "actionable_guidance": [
                f"Using regime-prior weights ({regime}) — insufficient trade history for data-driven learning.",
                f"Need at least 5 closed trades; currently have {len(closed)}.",
                "Weights will automatically become data-driven as the portfolio accumulates history.",
            ],
            "note": (
                "Regime-prior weights are based on historical factor performance in similar "
                "macro environments (academic literature + practitioner consensus). "
                "They will be updated with your own portfolio's history once 5+ trades close."
            ),
        }

    # ── Build feature matrix ──────────────────────────────────────────────────
    rows, returns = [], []
    for t in closed:
        vec = _to_feature_vec(t)
        if vec is not None:
            rows.append(vec)
            returns.append(float(t["return_pct"]))

    if len(rows) < 5:
        return {
            **result_base,
            "data_confidence": "low",
            "blended_weights": prior_weights,
            "learned_weights": None,
            "feature_correlations": None,
            "actionable_guidance": [
                "Closed trades found but most lack screener signal data.",
                "Only trades executed WITH a screener_snapshot contribute to ML learning.",
            ],
            "note": "Ensure screener_snapshot is passed to buy_stock() for future trades to enable ML.",
        }

    X = np.array(rows)
    y = np.array(returns)
    X_norm, mu, std = _normalize_cols(X)

    # ── Feature correlations (simple, interpretable) ──────────────────────────
    corr = {}
    for i, feat in enumerate(_FEATURES):
        col = X_norm[:, i]
        if col.std() > 0:
            corr[feat] = round(float(np.corrcoef(col, y)[0, 1]), 3)
        else:
            corr[feat] = 0.0

    # ── Ridge regression for multi-feature weights ────────────────────────────
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        alpha = max(1.0, 10.0 / len(rows))   # stronger regularisation with less data
        model = Ridge(alpha=alpha)
        model.fit(X_norm, y)

        # Cross-validated R² (only meaningful with ≥8 samples)
        if len(rows) >= 8:
            cv_r2 = float(np.mean(cross_val_score(model, X_norm, y, cv=min(5, len(rows)), scoring="r2")))
        else:
            cv_r2 = None

        raw_coefs = model.coef_
    except Exception:
        # Fallback: use correlations as proxy weights
        raw_coefs = np.array([max(0.0, corr.get(f, 0)) for f in _FEATURES])
        cv_r2 = None

    # Convert coefficients to non-negative weights (clip negatives, then normalise)
    # Note: peg_inv is already inverted, so positive coef = lower PEG preferred ✓
    weights_raw  = np.clip(raw_coefs, 0, None)
    weight_total = weights_raw.sum()
    if weight_total == 0:
        weights_raw = np.ones(len(_FEATURES))
        weight_total = len(_FEATURES)
    learned_weights = {f: round(float(weights_raw[i] / weight_total), 4) for i, f in enumerate(_FEATURES)}

    # ── Blend regime priors with learned weights ──────────────────────────────
    blended = _blend_weights(prior_weights, learned_weights, len(rows))
    blend_pct = min(75, int((len(rows) - 5) / 20 * 75))

    # ── Actionable guidance ───────────────────────────────────────────────────
    guidance = []
    # Find strongest predictor
    strongest = max(corr, key=lambda f: abs(corr[f]))
    if abs(corr[strongest]) > 0.2:
        direction = "higher" if corr[strongest] > 0 else "lower"
        guidance.append(
            f"'{strongest}' shows strongest correlation ({corr[strongest]:+.2f}) with your returns — "
            f"{direction} values predict better outcomes in your history."
        )
    # Warn if momentum is negatively correlated (mean reversion in your portfolio)
    if corr.get("relative_momentum_pct", 0) < -0.15:
        guidance.append(
            "Negative momentum correlation detected — momentum has been a contrarian signal in your "
            "portfolio. Consider down-weighting it vs the default screener."
        )
    # Warn if FCF is not predictive
    if abs(corr.get("fcf_yield_pct", 0)) < 0.1 and len(rows) >= 8:
        guidance.append(
            "FCF yield shows weak correlation in your trade history. "
            "This may reflect your sector mix — energy/financials compute FCF differently."
        )
    if cv_r2 is not None:
        guidance.append(
            f"Model cross-validated R²={cv_r2:.2f} — "
            + ("reasonable predictive power; learned weights are meaningful."
               if cv_r2 > 0.15 else
               "low predictive power; regime-prior blend still dominant.")
        )
    if not guidance:
        guidance.append("Insufficient variation in outcomes to draw strong signal conclusions yet.")

    return {
        **result_base,
        "data_confidence":      "high" if len(rows) >= 20 else "moderate" if len(rows) >= 8 else "low-moderate",
        "n_trades_with_signals": len(rows),
        "learned_weights":      learned_weights,
        "blended_weights":      blended,
        "data_weight_pct":      blend_pct,
        "prior_weight_pct":     100 - blend_pct,
        "feature_correlations": corr,
        "cross_val_r2":         cv_r2,
        "actionable_guidance":  guidance,
        "note": (
            f"Blended weights = {blend_pct}% data-driven + {100-blend_pct}% regime prior ({regime}). "
            "Use blended_weights when manually scoring screener candidates. "
            "Weights automatically shift toward your own trade history as it grows."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. ML Watchlist Prioritisation
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_screener_features(ticker: str) -> dict:
    """Fetch current screener-compatible features for a ticker via yfinance."""
    try:
        info = yf.Ticker(ticker).info
        pe   = info.get("trailingPE") or info.get("forwardPE")
        eps_growth = info.get("earningsGrowth") or info.get("revenueGrowth")
        peg  = pe / (eps_growth * 100) if pe and eps_growth and eps_growth > 0 else info.get("pegRatio")

        fcf       = info.get("freeCashflow")
        mktcap    = info.get("marketCap")
        fcf_yield = (fcf / mktcap * 100) if fcf and mktcap and mktcap > 0 else None

        rev_growth = (info.get("revenueGrowth") or 0) * 100
        margin     = (info.get("profitMargins") or 0) * 100
        roe        = (info.get("returnOnEquity") or 0) * 100

        # 52-week momentum vs S&P 500
        momentum = None
        try:
            import yfinance as _yf
            hist = _yf.Ticker(ticker).history(period="1y")
            sp   = _yf.Ticker("^GSPC").history(period="1y")
            if len(hist) > 10 and len(sp) > 10:
                stk_ret = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100
                sp_ret  = (sp["Close"].iloc[-1]  - sp["Close"].iloc[0])  / sp["Close"].iloc[0]  * 100
                momentum = round(float(stk_ret - sp_ret), 2)
        except Exception:
            pass

        return {
            "peg_ratio":             round(peg, 2) if peg else None,
            "fcf_yield_pct":         round(fcf_yield, 2) if fcf_yield else None,
            "relative_momentum_pct": momentum,
            "revenue_growth_pct":    round(rev_growth, 2) if rev_growth else None,
            "profit_margin_pct":     round(margin, 2) if margin else None,
            "roe_pct":               round(roe, 2) if roe else None,
            "current_price":         info.get("regularMarketPrice") or info.get("currentPrice"),
            "company_name":          info.get("shortName") or ticker,
        }
    except Exception:
        return {}


def prioritize_watchlist_ml() -> dict:
    """
    Score every watchlist item using ML-derived factor weights and return
    a ranked list surfacing the highest-conviction candidates to research first.

    For each item: fetches current fundamentals, applies learned weights,
    flags items near their target entry price, and highlights key
    strengths and risk flags.

    Call this at the start of a session after get_watchlist() to decide
    which watchlist candidates to research most deeply.
    """
    watchlist = portfolio.get_watchlist()
    if not watchlist:
        return {"message": "Watchlist is empty.", "ranked_items": []}

    # Get learned weights
    fw_result = get_ml_factor_weights()
    weights   = fw_result["blended_weights"]
    regime    = fw_result["regime"]

    # Pre-filter: get current prices cheaply to skip items trading far above target entry.
    # If price > target × 1.15 (more than 15% above our desired entry), the opportunity
    # has drifted away — skip the expensive fundamental fetch entirely.
    import json as _json
    try:
        from agent import market_data as _md
        tickers_list = [item["ticker"] for item in watchlist if item.get("target_entry_price")]
        price_map: dict = {}
        if tickers_list:
            for _t in tickers_list:
                try:
                    _q = _md.get_stock_quote(_t)
                    if _q and not _q.get("error"):
                        price_map[_t.upper()] = _q.get("price")
                except Exception:
                    pass
    except Exception:
        price_map = {}

    ranked = []
    for item in watchlist:
        ticker = item["ticker"]
        target = item.get("target_entry_price")
        current_price_quick = price_map.get(ticker.upper())

        # Proximity filter: skip deep fundamental fetch if price is >15% above target entry
        if target and target > 0 and current_price_quick:
            pct_above = (current_price_quick - target) / target * 100
            if pct_above > 15:
                ranked.append({
                    "rank":             None,
                    "ticker":           ticker,
                    "ml_score":         None,
                    "current_price":    current_price_quick,
                    "target_entry":     target,
                    "vs_target":        f"+{pct_above:.1f}% vs target ${target:.2f}",
                    "near_entry_price": False,
                    "watchlist_reason": item.get("reason", ""),
                    "features":         {},
                    "strengths":        [],
                    "risk_flags":       [f"Price {pct_above:.1f}% above target entry — not actionable yet"],
                    "skipped_reason":   "proximity_filter: >15% above target entry",
                })
                continue

        feats  = _fetch_screener_features(ticker)
        if not feats:
            ranked.append({
                "rank":         None,
                "ticker":       ticker,
                "ml_score":     None,
                "error":        "Could not fetch current fundamentals",
                "watchlist_reason": item.get("reason", ""),
            })
            continue

        score = _score_features(feats, weights)

        # Price vs target
        current  = feats.get("current_price")
        target   = item.get("target_entry_price")
        vs_target = None
        near_entry = False
        if current and target and target > 0:
            diff_pct  = (current - target) / target * 100
            vs_target = f"{diff_pct:+.1f}% vs target ${target:.2f}"
            near_entry = abs(diff_pct) < 5

        # Identify strengths and flags
        strengths, flags = [], []

        peg = feats.get("peg_ratio")
        if peg and peg < 1.5:
            strengths.append(f"PEG {peg:.2f} (excellent — below 1.5)")
        elif peg and peg > 3.0:
            flags.append(f"PEG {peg:.2f} (expensive relative to growth)")

        fcf = feats.get("fcf_yield_pct")
        if fcf and fcf > 4:
            strengths.append(f"FCF yield {fcf:.1f}% (strong cash generation)")
        elif fcf is not None and fcf < 0:
            flags.append(f"Negative FCF yield {fcf:.1f}% (cash burn)")

        mom = feats.get("relative_momentum_pct")
        if mom and mom > 10:
            strengths.append(f"Momentum +{mom:.0f}% vs S&P 500")
        elif mom is not None and mom < -20:
            flags.append(f"Weak momentum {mom:.0f}% vs S&P 500 — needs fundamental catalyst")

        rev = feats.get("revenue_growth_pct")
        if rev and rev > 15:
            strengths.append(f"Revenue growth {rev:.0f}% YoY")

        ranked.append({
            "ticker":           ticker,
            "company_name":     feats.get("company_name", ticker),
            "ml_score":         score,
            "current_price":    current,
            "target_entry":     target,
            "vs_target":        vs_target,
            "near_entry_price": near_entry,
            "watchlist_reason": item.get("reason", ""),
            "features":         {k: v for k, v in feats.items() if k not in ("current_price", "company_name")},
            "strengths":        strengths,
            "risk_flags":       flags,
        })

    # Sort by score (nulls last), then surface near-target items
    ranked = [r for r in ranked if r["ml_score"] is not None] + \
             [r for r in ranked if r["ml_score"] is None]
    ranked_scored = sorted(
        [r for r in ranked if r["ml_score"] is not None],
        key=lambda x: (not x["near_entry_price"], -x["ml_score"]),
    )
    ranked_no_score = [r for r in ranked if r["ml_score"] is None]

    for i, r in enumerate(ranked_scored):
        r["rank"] = i + 1

    return {
        "regime":              regime,
        "factor_weights_used": weights,
        "n_items":             len(watchlist),
        "ranked_watchlist":    ranked_scored + ranked_no_score,
        "note": (
            "Items near target entry price are promoted to the top within their score tier. "
            "ML score (0-10) reflects quality relative to learned factor weights for this "
            f"portfolio in a {regime} regime. Score ≥7: strong candidate; <4: weak fit."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Position Size Recommendation (Drawdown Risk)
# ══════════════════════════════════════════════════════════════════════════════

def get_position_size_recommendation(ticker: str, features: dict) -> dict:
    """
    Estimate drawdown risk and recommend an appropriate position size.

    Combines three inputs:
    1. Current feature risk flags (valuation, FCF, momentum quality).
    2. Historical drawdown patterns from closed trades (if ≥5 exist).
    3. Current macro regime (risk-off → smaller base sizes).

    Returns a recommended % of portfolio with clear risk reasoning.
    Use this just before executing a buy to size the position correctly.
    High-conviction, low-risk positions warrant 12-20%;
    uncertain or risky positions should be 3-8% test positions.
    """
    regime, regime_rationale, _ = _detect_regime()
    base_size = _REGIME_BASE_POSITION[regime]

    # ── Rule-based risk score from features ──────────────────────────────────
    risk_score   = 0
    risk_factors = []

    peg = features.get("peg_ratio")
    if peg and peg > 3.5:
        risk_score += 2
        risk_factors.append(f"High valuation: PEG {peg:.1f} (>3.5 — earnings risk if growth disappoints)")
    elif peg and peg > 2.5:
        risk_score += 1
        risk_factors.append(f"Elevated valuation: PEG {peg:.1f} (>2.5 — limited margin of safety)")

    fcf = features.get("fcf_yield_pct")
    if fcf is not None and fcf < 0:
        risk_score += 2
        risk_factors.append(f"Negative FCF yield {fcf:.1f}% — company burning cash; no valuation floor")
    elif fcf is not None and fcf < 1.5:
        risk_score += 1
        risk_factors.append(f"Low FCF yield {fcf:.1f}% (<1.5%) — limited downside protection")

    mom = features.get("relative_momentum_pct")
    if mom is not None and mom < -25:
        risk_score += 2
        risk_factors.append(f"Sharply negative momentum ({mom:.0f}% vs S&P) — needs strong catalyst to reverse")
    elif mom is not None and mom < -10:
        risk_score += 1
        risk_factors.append(f"Negative momentum ({mom:.0f}% vs S&P) — trend working against position")
    elif mom is not None and mom > 60:
        risk_score += 1
        risk_factors.append(f"Extreme momentum (+{mom:.0f}% vs S&P) — mean reversion risk if sentiment shifts")

    rev = features.get("revenue_growth_pct")
    if rev is not None and rev < 0:
        risk_score += 1
        risk_factors.append(f"Negative revenue growth {rev:.1f}% — business contracting")

    margin = features.get("profit_margin_pct")
    if margin is not None and margin < 0:
        risk_score += 1
        risk_factors.append(f"Negative profit margin {margin:.1f}% — unprofitable business")

    roe = features.get("roe_pct")
    if roe is not None and roe < 5:
        risk_score += 1
        risk_factors.append(f"Low ROE {roe:.1f}% (<5%) — weak capital efficiency")

    # ── Data-driven drawdown calibration ────────────────────────────────────
    ml_adjustment = 0.0
    drawdown_probability = None
    try:
        outcomes = portfolio.get_trade_outcomes()
        closed   = [t for t in outcomes if t.get("return_pct") is not None]

        if len(closed) >= 5:
            from sklearn.linear_model import LogisticRegression

            rows, labels = [], []
            for t in closed:
                vec = _to_feature_vec(t)
                if vec is not None:
                    rows.append(vec)
                    # Label: 1 = drew down significantly (return < -15%) or negative
                    labels.append(1 if float(t["return_pct"]) < -10 else 0)

            if len(rows) >= 5 and sum(labels) >= 1 and sum(labels) < len(labels):
                X = np.array(rows)
                X_norm, _, _ = _normalize_cols(X)
                clf = LogisticRegression(C=0.5, max_iter=500)
                clf.fit(X_norm, labels)

                vec_new = _to_feature_vec(features)
                if vec_new is not None:
                    X_new = (vec_new - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
                    drawdown_probability = round(float(clf.predict_proba([X_new])[0][1]), 3)
                    if drawdown_probability > 0.6:
                        ml_adjustment = -3.0
                        risk_factors.append(
                            f"ML drawdown model: {drawdown_probability*100:.0f}% probability of >10% loss "
                            f"(based on {len(rows)} historical trades) — reducing position size"
                        )
                    elif drawdown_probability < 0.25:
                        ml_adjustment = +2.0
    except Exception:
        pass

    # ── Compute final recommended size ───────────────────────────────────────
    # risk_score 0 → full base; each point reduces by ~2%; floor at 3%
    size_reduction  = risk_score * 2.0
    recommended_pct = max(3.0, min(20.0, base_size - size_reduction + ml_adjustment))
    recommended_pct = round(recommended_pct, 1)

    # Map score to label
    total_risk = risk_score + (2 if drawdown_probability and drawdown_probability > 0.6 else 0)
    if total_risk == 0:
        risk_level = "LOW"
    elif total_risk <= 2:
        risk_level = "LOW-MODERATE"
    elif total_risk <= 4:
        risk_level = "MODERATE"
    elif total_risk <= 6:
        risk_level = "ELEVATED"
    else:
        risk_level = "HIGH"

    return {
        "ticker":                    ticker.upper(),
        "recommended_pct":           recommended_pct,
        "risk_level":                risk_level,
        "risk_score":                risk_score,
        "risk_factors":              risk_factors,
        "ml_drawdown_probability":   drawdown_probability,
        "regime":                    regime,
        "regime_base_size_pct":      base_size,
        "size_guide": {
            "LOW (0 flags)":          f"{_REGIME_BASE_POSITION[regime]:.0f}%  (full position)",
            "LOW-MODERATE (1-2)":     f"{max(3, _REGIME_BASE_POSITION[regime]-4):.0f}%",
            "MODERATE (3-4)":         f"{max(3, _REGIME_BASE_POSITION[regime]-8):.0f}%",
            "ELEVATED (5-6)":         f"{max(3, _REGIME_BASE_POSITION[regime]-12):.0f}%",
            "HIGH (7+)":              "3%  (test position only)",
        },
        "note": (
            f"Base size {base_size:.0f}% for {regime} regime. "
            f"Reduced by {size_reduction:.0f}pp for {risk_score} risk factor(s)"
            + (f", adjusted {ml_adjustment:+.0f}pp by ML drawdown model" if ml_adjustment != 0 else "")
            + f". Final recommendation: {recommended_pct:.1f}% of portfolio. "
            "Always subject to 20% maximum position cap and available cash."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Capital Recycling / Weakest Link
# ══════════════════════════════════════════════════════════════════════════════

def identify_weakest_link() -> dict:
    """
    Identify the weakest holding in the portfolio for potential capital recycling.

    Scores each position on 3 dimensions (0-10 each):
      - conviction_score: from prediction_tracking (high=8, medium=5, low=2)
      - iv_upside_remaining: (iv_price - current_price) / current_price from research_cache
      - research_freshness: days since last research

    Returns ranked positions (weakest first) with a top recycle candidate.
    """
    from agent.portfolio import get_holdings, get_research_cache, _get_connection
    from datetime import datetime

    holdings = get_holdings()
    if not holdings:
        return {
            "positions_ranked": [],
            "weakest_link": None,
            "recommendation": "No positions in portfolio.",
        }

    # Load prediction_tracking data for conviction levels
    conn = _get_connection()
    pred_rows = conn.execute(
        "SELECT ticker, conviction_score FROM prediction_tracking ORDER BY id DESC"
    ).fetchall()
    conn.close()

    # Build conviction map: ticker → last conviction_score
    conviction_map = {}
    for row in pred_rows:
        t = row["ticker"]
        if t not in conviction_map and row["conviction_score"] is not None:
            conviction_map[t] = row["conviction_score"]

    # Get current prices
    current_prices = {}
    try:
        import yfinance as yf
        for h in holdings:
            try:
                info = yf.Ticker(h["ticker"]).info
                price = (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("previousClose")
                )
                current_prices[h["ticker"]] = price
            except Exception:
                current_prices[h["ticker"]] = None
    except ImportError:
        pass

    scored = []
    for h in holdings:
        ticker = h["ticker"]

        # Conviction score
        conv_raw = conviction_map.get(ticker)
        if conv_raw is None:
            conviction_dim = 5  # missing → neutral
        elif conv_raw >= 8:
            conviction_dim = 8  # high
        elif conv_raw >= 5:
            conviction_dim = 5  # medium
        else:
            conviction_dim = 2  # low

        # IV upside remaining from research cache
        upside_pct = None
        iv_dim = 5  # missing → neutral
        try:
            cache = get_research_cache(ticker)
            if cache:
                # Look for iv_price in the report
                iv_price = None
                for key in ("intrinsic_value", "iv_price", "base_iv", "fair_value"):
                    iv_price = cache.get(key)
                    if iv_price:
                        break
                # Also check nested valuation keys
                if iv_price is None:
                    for key in ("valuation", "valuation_inputs"):
                        sub = cache.get(key)
                        if isinstance(sub, dict):
                            for subkey in ("base_iv", "intrinsic_value", "fair_value"):
                                iv_price = sub.get(subkey)
                                if iv_price:
                                    break
                        if iv_price:
                            break

                current_price = current_prices.get(ticker)
                if iv_price and current_price and current_price > 0:
                    upside_pct = (float(iv_price) - float(current_price)) / float(current_price) * 100
                    if upside_pct > 30:
                        iv_dim = 8
                    elif upside_pct >= 15:
                        iv_dim = 5
                    else:
                        iv_dim = 2
        except Exception:
            pass

        # Research freshness
        days_since = None
        freshness_dim = 2  # missing → stale
        try:
            cache = get_research_cache(ticker)
            if cache:
                ts_str = cache.get("_researched_at")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
                    days_since = (datetime.utcnow() - ts).days
                    if days_since < 30:
                        freshness_dim = 8
                    elif days_since <= 90:
                        freshness_dim = 5
                    else:
                        freshness_dim = 2
        except Exception:
            pass

        total_score = round((conviction_dim + iv_dim + freshness_dim) / 3, 2)

        scored.append({
            "ticker": ticker,
            "score": total_score,
            "conviction": conv_raw,
            "upside_pct": round(upside_pct, 1) if upside_pct is not None else None,
            "days_since_research": days_since,
            "recycle_candidate": total_score < 4,
        })

    # Sort weakest first
    scored.sort(key=lambda x: x["score"])
    weakest = scored[0] if scored else None

    if weakest and weakest["recycle_candidate"]:
        recommendation = (
            f"Consider recycling {weakest['ticker']} (score={weakest['score']:.1f}/10). "
            f"Low combined conviction/upside/freshness. Re-research or redeploy capital."
        )
    elif weakest:
        recommendation = (
            f"No strong recycle candidates. Weakest position: {weakest['ticker']} "
            f"(score={weakest['score']:.1f}/10). Portfolio appears reasonably positioned."
        )
    else:
        recommendation = "No positions to evaluate."

    return {
        "positions_ranked": scored,
        "weakest_link": weakest,
        "recommendation": recommendation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Screener ML Recalibration
# ══════════════════════════════════════════════════════════════════════════════

def recalibrate_universe_scores() -> dict:
    """
    Re-score the screener universe using prediction accuracy weights.

    For each ticker with prediction history:
      new_score = old_score * 0.7 + (accuracy * 10) * 0.3
    where accuracy = 1 - abs(outcome_price / predicted_iv - 1) clamped to [0, 1].

    Also applies per-conviction-bucket multipliers based on historical postmortem data.
    Writes updated scores back to universe_scores table.
    """
    from agent.portfolio import _get_connection

    conn = _get_connection()

    # Load universe scores
    universe_rows = conn.execute(
        "SELECT ticker, quality_score FROM universe_scores WHERE quality_score IS NOT NULL"
    ).fetchall()
    universe_map = {r["ticker"]: r["quality_score"] for r in universe_rows}

    # Load prediction tracking with outcomes
    pred_rows = conn.execute("""
        SELECT ticker, predicted_iv, outcome_price, outcome_date, conviction_score AS conviction
        FROM prediction_tracking
        WHERE outcome_price IS NOT NULL
          AND predicted_iv IS NOT NULL
          AND predicted_iv > 0
    """).fetchall()
    pred_rows = [dict(r) for r in pred_rows]

    # Compute accuracy per ticker (use most recent prediction)
    ticker_accuracy = {}
    ticker_conviction = {}
    for r in pred_rows:
        t = r["ticker"]
        raw_acc = 1.0 - abs(r["outcome_price"] / r["predicted_iv"] - 1.0)
        acc = max(0.0, min(1.0, raw_acc))
        # Keep best accuracy per ticker (most recent is fine since rows come ordered)
        if t not in ticker_accuracy:
            ticker_accuracy[t] = acc
            ticker_conviction[t] = r.get("conviction")

    # Conviction-level multipliers from IV postmortem logic
    # Group actual returns (outcome_price/predicted_iv) by conviction
    conviction_groups: dict[str, list] = {}
    for r in pred_rows:
        conv = r.get("conviction") or "medium"
        actual_ratio = r["outcome_price"] / r["predicted_iv"]
        conviction_groups.setdefault(conv, []).append(actual_ratio)

    conviction_multipliers: dict[str, float] = {}
    for conv, ratios in conviction_groups.items():
        avg_ratio = sum(ratios) / len(ratios)
        # Multiplier: if avg_ratio > 1 → stock tends to exceed IV → boost; < 1 → discount
        conviction_multipliers[conv] = round(min(1.3, max(0.7, avg_ratio)), 3)

    # Apply adjustments
    adjustments = []
    updated_count = 0
    skipped_count = 0

    for ticker, old_score in universe_map.items():
        if ticker not in ticker_accuracy:
            skipped_count += 1
            continue

        acc = ticker_accuracy[ticker]
        new_score = old_score * 0.7 + (acc * 10) * 0.3

        # Apply conviction multiplier if available
        conv = ticker_conviction.get(ticker)
        if conv and conv in conviction_multipliers:
            new_score *= conviction_multipliers[conv]

        new_score = round(min(10.0, max(0.0, new_score)), 3)
        delta = round(new_score - old_score, 3)

        conn.execute(
            "UPDATE universe_scores SET quality_score = ? WHERE ticker = ?",
            (new_score, ticker),
        )
        conn.commit()

        adjustments.append({
            "ticker": ticker,
            "old_score": old_score,
            "new_score": new_score,
            "delta": delta,
        })
        updated_count += 1

    conn.close()

    adjustments.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "adjustments": adjustments[:20],  # top 20 most impacted
        "conviction_multipliers": conviction_multipliers,
        "summary": (
            f"Recalibrated {updated_count} tickers using prediction accuracy weights. "
            f"{skipped_count} tickers skipped (no prediction history). "
            f"Top adjustment: {adjustments[0]['ticker']} Δ{adjustments[0]['delta']:+.2f}"
            if adjustments else f"Recalibrated {updated_count} tickers. {skipped_count} skipped."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio Stress Testing
# ══════════════════════════════════════════════════════════════════════════════

def run_portfolio_stress_test(scenario: str = "all") -> dict:
    """
    Run scenario-based stress tests across the portfolio.

    Scenarios:
      - ai_disruption: AI commoditizes core products
      - rate_spike_200bps: Risk-free rate rises 200bps, multiple compression
      - recession_revenue_20pct: Recession with revenue -20%, margin compression
      - sector_concentration_shock: Worst sector drops 40%
      - all: run all four

    Returns per-scenario portfolio impact estimates and most vulnerable scenario.
    """
    from agent.portfolio import get_holdings, get_research_cache

    holdings = get_holdings()
    if not holdings:
        return {
            "scenario_results": {},
            "most_vulnerable_scenario": None,
            "summary": "No holdings to stress test.",
        }

    SCENARIOS = {
        "ai_disruption": {
            "description": "AI commoditizes core products",
            "sector_hits": {"Technology": -0.35, "Software": -0.40, "Services": -0.20},
            "default_hit": -0.10,
            "ai_risk_multiplier": 1.5,
        },
        "rate_spike_200bps": {
            "description": "Risk-free rate rises 200bps, multiple compression",
            "sector_hits": {"Real Estate": -0.25, "Utilities": -0.20, "Financial": -0.15},
            "growth_premium_hit": -0.20,
            "default_hit": -0.08,
        },
        "recession_revenue_20pct": {
            "description": "Recession: revenue -20%, margins compress",
            "sector_hits": {
                "Consumer Cyclical": -0.35,
                "Industrial": -0.30,
                "Energy": -0.25,
            },
            "defensive_protection": {"Healthcare": 0.05, "Consumer Defensive": 0.05},
            "default_hit": -0.15,
        },
        "sector_concentration_shock": {
            "description": "Worst sector drops 40%",
            "dynamic": True,
        },
    }

    # Get sector and ai_disruption_risk per ticker from research_cache
    ticker_sector = {}
    ticker_ai_risk = {}
    ticker_value = {}

    try:
        import yfinance as yf
        for h in holdings:
            ticker = h["ticker"]
            # Try research cache first
            try:
                cache = get_research_cache(ticker)
                if cache:
                    sector = cache.get("sector")
                    if not sector:
                        for key in ("business_overview", "fundamentals"):
                            sub = cache.get(key) or {}
                            sector = sub.get("sector") if isinstance(sub, dict) else None
                            if sector:
                                break
                    ticker_sector[ticker] = sector or "Unknown"
                    # Check for ai_disruption_risk flag
                    ticker_ai_risk[ticker] = bool(
                        cache.get("ai_disruption_risk")
                        or (isinstance(cache.get("key_risks"), list) and
                            any("ai" in str(r).lower() for r in cache.get("key_risks", [])))
                    )
                else:
                    ticker_sector[ticker] = "Unknown"
                    ticker_ai_risk[ticker] = False
            except Exception:
                ticker_sector[ticker] = "Unknown"
                ticker_ai_risk[ticker] = False

            # Estimate position value
            try:
                info = yf.Ticker(ticker).info
                price = (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("previousClose")
                    or h.get("avg_cost", 0)
                )
                ticker_value[ticker] = float(price) * float(h["shares"])
            except Exception:
                ticker_value[ticker] = float(h.get("avg_cost", 0)) * float(h["shares"])
    except ImportError:
        for h in holdings:
            ticker = h["ticker"]
            ticker_sector[ticker] = "Unknown"
            ticker_ai_risk[ticker] = False
            ticker_value[ticker] = float(h.get("avg_cost", 0)) * float(h["shares"])

    total_value = sum(ticker_value.values()) or 1.0

    def _run_scenario(scen_name: str, scen_cfg: dict) -> dict:
        positions_detail = []

        # Handle sector_concentration_shock dynamically
        if scen_cfg.get("dynamic"):
            # Find heaviest sector
            sector_values: dict[str, float] = {}
            for h in holdings:
                t = h["ticker"]
                s = ticker_sector.get(t, "Unknown")
                sector_values[s] = sector_values.get(s, 0) + ticker_value.get(t, 0)
            worst_sector = max(sector_values, key=lambda s: sector_values[s]) if sector_values else "Unknown"

            for h in holdings:
                t = h["ticker"]
                pv = ticker_value.get(t, 0)
                s = ticker_sector.get(t, "Unknown")
                impact = -0.40 if s == worst_sector else 0.0
                positions_detail.append({
                    "ticker": t,
                    "sector": s,
                    "position_value": round(pv, 2),
                    "estimated_impact_pct": round(impact * 100, 1),
                })

            portfolio_impact = sum(
                (p["estimated_impact_pct"] / 100) * (ticker_value.get(p["ticker"], 0) / total_value)
                for p in positions_detail
            )
            worst_pos = min(positions_detail, key=lambda p: p["estimated_impact_pct"])
            return {
                "description": f"Worst sector ({worst_sector}) drops 40%",
                "portfolio_impact_pct": round(portfolio_impact * 100, 2),
                "worst_position": worst_pos,
                "positions_detail": positions_detail,
            }

        # Standard scenarios
        sector_hits = scen_cfg.get("sector_hits", {})
        defensive = scen_cfg.get("defensive_protection", {})
        default_hit = scen_cfg.get("default_hit", -0.10)

        for h in holdings:
            t = h["ticker"]
            pv = ticker_value.get(t, 0)
            s = ticker_sector.get(t, "Unknown")

            # Base impact from sector
            impact = sector_hits.get(s)
            if impact is None:
                impact = defensive.get(s, default_hit)

            # AI disruption multiplier
            if scen_name == "ai_disruption" and ticker_ai_risk.get(t):
                impact *= scen_cfg.get("ai_risk_multiplier", 1.5)
                impact = max(-0.80, impact)  # floor at -80%

            positions_detail.append({
                "ticker": t,
                "sector": s,
                "position_value": round(pv, 2),
                "estimated_impact_pct": round(impact * 100, 1),
            })

        portfolio_impact = sum(
            (p["estimated_impact_pct"] / 100) * (ticker_value.get(p["ticker"], 0) / total_value)
            for p in positions_detail
        )
        worst_pos = min(positions_detail, key=lambda p: p["estimated_impact_pct"])

        return {
            "description": scen_cfg["description"],
            "portfolio_impact_pct": round(portfolio_impact * 100, 2),
            "worst_position": worst_pos,
            "positions_detail": positions_detail,
        }

    # Determine which scenarios to run
    if scenario == "all":
        run_scenarios = list(SCENARIOS.keys())
    elif scenario in SCENARIOS:
        run_scenarios = [scenario]
    else:
        return {"error": f"Unknown scenario '{scenario}'. Choose from: {', '.join(SCENARIOS.keys())}, all"}

    scenario_results = {}
    for sn in run_scenarios:
        scenario_results[sn] = _run_scenario(sn, SCENARIOS[sn])

    # Most vulnerable scenario
    most_vulnerable = min(scenario_results, key=lambda s: scenario_results[s]["portfolio_impact_pct"])

    impacts = {sn: scenario_results[sn]["portfolio_impact_pct"] for sn in scenario_results}
    summary = (
        f"Stress test complete ({len(scenario_results)} scenarios). "
        f"Most vulnerable: {most_vulnerable} "
        f"({scenario_results[most_vulnerable]['portfolio_impact_pct']:+.1f}% portfolio impact). "
        + " | ".join(f"{sn}: {v:+.1f}%" for sn, v in impacts.items())
    )

    return {
        "scenario_results": scenario_results,
        "most_vulnerable_scenario": most_vulnerable,
        "summary": summary,
    }



def detect_regime_change() -> dict:
    """
    Detect the current macro regime, compare to last persisted regime,
    and save if it changed (or if no history exists).

    Returns:
        {
          "current_regime": str,
          "previous_regime": Optional[str],
          "changed": bool,
          "days_since_last_detection": Optional[int],
          "change_summary": str,   # human-readable description
          "indicators": dict,      # raw indicator values
        }
    """
    from agent.portfolio import get_last_regime, save_regime
    from datetime import datetime, timezone

    # Run current detection
    regime, _rationale, macro = _detect_regime()
    current = regime
    rates = (macro.get("rates") or {})
    indicators = {
        "vix": (macro.get("sentiment") or {}).get("vix"),
        "gdp_growth": None,
        "core_cpi": None,
        "ten_yr_yield": rates.get("ten_yr_treasury_yield_pct"),
        "yield_inverted": (rates.get("yield_curve_spread") or 0) < -0.3,
    }
    # Try to pull GDP/CPI from FRED if available (already attempted in _detect_regime)
    try:
        from agent import external_data
        econ = external_data.get_economic_indicators()
        if "error" not in econ:
            indicators["gdp_growth"] = (
                econ.get("indicators", {}).get("real_gdp_growth_pct", {}).get("value")
            )
            indicators["core_cpi"] = (
                econ.get("indicators", {}).get("core_cpi_yoy_pct", {}).get("yoy_pct")
            )
    except Exception:
        pass

    last = get_last_regime()
    previous = last["regime"] if last else None

    # Compute days since last detection
    days_since = None
    if last:
        try:
            last_dt = datetime.fromisoformat(last["detected_at"])
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            days_since = (datetime.now(timezone.utc) - last_dt).days
        except Exception:
            pass

    changed = previous is None or current != previous

    # Always save if no history, or if regime changed, or if >7 days since last save
    if changed or days_since is None or days_since >= 7:
        save_regime(current, previous, indicators)

    if changed and previous:
        change_summary = f"REGIME CHANGE: {previous} → {current}"
    elif previous is None:
        change_summary = f"Initial regime detected: {current}"
    else:
        change_summary = f"Regime stable: {current} (last checked {days_since}d ago)"

    return {
        "current_regime": current,
        "previous_regime": previous,
        "changed": changed and previous is not None,
        "days_since_last_detection": days_since,
        "change_summary": change_summary,
        "indicators": indicators,
    }


def conviction_position_size(conviction_score: int, regime: str, portfolio_equity: float) -> dict:
    """
    Calculate position size scaling with conviction score.

    Base sizes by regime (already defined in _REGIME_PRIORS):
      RISK_ON: 15%, NORMAL: 12%, INFLATIONARY: 10%, RISK_OFF: 8%, STAGFLATION: 6%

    Conviction multipliers:
      9-10: 1.0x (full position)
       7-8: 0.75x
       5-6: 0.5x (minimum viable — agent should rarely buy below 6)

    Returns:
      {
        "base_pct": float,          # regime base %
        "conviction_multiplier": float,
        "final_pct": float,         # base * multiplier
        "dollar_amount": float,     # final_pct * portfolio_equity
        "rationale": str,
      }
    """
    _BASE = {
        "RISK_ON": 0.15, "NORMAL": 0.12, "INFLATIONARY": 0.10,
        "RISK_OFF": 0.08, "STAGFLATION": 0.06,
    }
    base = _BASE.get(regime, 0.10)

    if conviction_score >= 9:
        multiplier = 1.0
    elif conviction_score >= 7:
        multiplier = 0.75
    else:
        multiplier = 0.5

    final_pct = base * multiplier
    dollar = portfolio_equity * final_pct

    rationale = (
        f"Conviction {conviction_score}/10 → {multiplier:.0%} of {regime} base "
        f"({base:.0%}) = {final_pct:.1%} of equity = ${dollar:,.0f}"
    )
    return {
        "base_pct": round(base, 4),
        "conviction_multiplier": multiplier,
        "final_pct": round(final_pct, 4),
        "dollar_amount": round(dollar, 2),
        "rationale": rationale,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. IV Post-mortem Loop + KB Feedback
# ══════════════════════════════════════════════════════════════════════════════


def check_portfolio_correlation(ticker: str) -> dict:
    """
    Compute 1-year price correlation between a candidate stock and all current
    portfolio holdings. Returns a warning + sizing guidance if the candidate is
    highly correlated with existing positions.
    """
    try:
        holdings = portfolio.get_holdings()
        if not holdings:
            return {"ticker": ticker, "warning": False, "message": "No existing holdings"}

        held_tickers = [h["ticker"] for h in holdings]
        all_tickers = list(dict.fromkeys([ticker] + held_tickers))  # deduplicate, preserve order

        data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)

        # Handle MultiIndex columns (multiple tickers) vs single-level (one ticker)
        if hasattr(data.columns, "levels"):
            prices = data["Close"]
        else:
            prices = data

        returns = prices.pct_change().dropna()

        correlations = {}
        candidate_col = ticker if ticker in returns.columns else None
        if candidate_col is None:
            return {"ticker": ticker, "warning": False, "error": "Candidate ticker not found in downloaded data"}

        for t in held_tickers:
            if t in returns.columns and t != ticker:
                corr = returns[candidate_col].corr(returns[t])
                if not np.isnan(corr):
                    correlations[t] = round(float(corr), 3)

        high_corr = {t: c for t, c in correlations.items() if abs(c) > 0.7}
        avg_corr = float(np.mean(list(correlations.values()))) if correlations else 0.0
        warning = bool(high_corr) or avg_corr > 0.6

        if warning:
            sizing_guidance = (
                f"REDUCE to 0.5x normal size — portfolio correlation {avg_corr:.2f}, "
                f"high pairs: {high_corr}"
            )
        else:
            sizing_guidance = f"Correlation acceptable (avg={avg_corr:.2f}). Normal sizing."

        return {
            "ticker": ticker,
            "warning": warning,
            "correlations": correlations,
            "high_correlation_pairs": high_corr,
            "avg_portfolio_correlation": round(avg_corr, 3),
            "sizing_guidance": sizing_guidance,
        }
    except Exception as e:
        return {"ticker": ticker, "warning": False, "error": str(e)}


def get_conviction_calibration() -> dict:
    """
    Analyse prediction accuracy grouped by conviction score bucket.

    Uses prediction_tracking to answer: "Is conviction=8 actually more accurate
    than conviction=6 in this portfolio?"

    Returns calibration table + recommended sizing adjustments.
    """
    from agent.portfolio import _get_connection

    conn = _get_connection()
    rows = conn.execute("""
        SELECT conviction_score, outcome_return_pct, outcome_vs_spy_pct
        FROM prediction_tracking
        WHERE outcome_date IS NOT NULL
          AND outcome_return_pct IS NOT NULL
    """).fetchall()
    conn.close()

    rows = [dict(r) for r in rows]

    if not rows:
        return {
            "available": False,
            "calibration_status": "insufficient_data",
            "buckets": {},
            "sizing_guidance": "No reconciled predictions available yet.",
            "note": "Call reconcile_predictions to populate outcome data.",
        }

    def _bucket(score):
        if score is None:
            return None
        if score >= 9:
            return "high_9_10"
        elif score >= 7:
            return "medium_7_8"
        elif score >= 5:
            return "low_5_6"
        return None

    groups = {}
    for r in rows:
        b = _bucket(r.get("conviction_score"))
        if b:
            groups.setdefault(b, []).append(r)

    buckets = {}
    for bucket_name, bucket_rows in groups.items():
        if len(bucket_rows) < 3:
            buckets[bucket_name] = {"count": len(bucket_rows), "note": "insufficient_data (<3)"}
            continue
        returns = [float(r["outcome_return_pct"]) for r in bucket_rows]
        alphas = [float(r["outcome_vs_spy_pct"]) for r in bucket_rows if r.get("outcome_vs_spy_pct") is not None]
        avg_return = round(sum(returns) / len(returns), 2)
        win_rate = round(sum(1 for x in returns if x > 0) / len(returns) * 100, 1)
        avg_alpha = round(sum(alphas) / len(alphas), 2) if alphas else None
        buckets[bucket_name] = {
            "count": len(bucket_rows),
            "avg_return": avg_return,
            "win_rate": win_rate,
            "avg_alpha": avg_alpha,
        }

    # Determine calibration status
    high = buckets.get("high_9_10", {})
    medium = buckets.get("medium_7_8", {})
    low = buckets.get("low_5_6", {})

    h_return = high.get("avg_return")
    m_return = medium.get("avg_return")
    l_return = low.get("avg_return")

    any_insufficient = any(
        b.get("note") == "insufficient_data (<3)"
        for b in [high, medium, low]
        if b
    ) or len(groups) < 2

    if any_insufficient or (h_return is None and m_return is None):
        calibration_status = "insufficient_data"
        sizing_guidance = "Need ≥3 predictions per bucket to assess calibration."
    elif h_return is not None and m_return is not None and h_return > m_return:
        if l_return is None or m_return > l_return:
            calibration_status = "well_calibrated"
        else:
            calibration_status = "well_calibrated"
        h_wr = high.get("win_rate", 0)
        h_avg = h_return
        sizing_guidance = (
            f"conviction_9+ historically delivers {h_avg:.1f}% avg return with "
            f"{h_wr:.0f}% win rate — full sizing appropriate"
        )
    elif h_return is not None and m_return is not None and h_return < m_return:
        calibration_status = "miscalibrated_high"
        sizing_guidance = (
            "High conviction (9-10) has underperformed medium conviction (7-8) historically. "
            "Consider capping position size at 0.85x base for conviction 9-10 until recalibrated."
        )
    else:
        calibration_status = "insufficient_data"
        sizing_guidance = "Insufficient cross-bucket data to assess calibration."

    note = (
        "Calibration status: "
        + ("conviction correctly ranks returns — sizing multipliers are well-grounded." if calibration_status == "well_calibrated"
           else "high conviction overestimates returns — review whether 9+ scores are being assigned too liberally." if calibration_status == "miscalibrated_high"
           else "more reconciled data needed before drawing conclusions.")
    )

    return {
        "available": True,
        "buckets": buckets,
        "calibration_status": calibration_status,
        "sizing_guidance": sizing_guidance,
        "note": note,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Market-level MoS Context
# ══════════════════════════════════════════════════════════════════════════════


def get_market_iv_context() -> dict:
    """
    Aggregate recent research cache to quantify market-level valuation.

    Answers: "What % of the researched universe currently offers ≥20% MoS?"
    High % = cheap market, easier to find opportunities.
    Low % = expensive market, require higher quality bar.

    Returns market valuation signal + sector breakdown.
    """
    import json as _json
    from datetime import datetime, timedelta
    from agent.portfolio import _get_connection

    cutoff = (datetime.utcnow() - timedelta(days=90)).isoformat()

    conn = _get_connection()
    rows = conn.execute("""
        SELECT rc.ticker, rc.report_json, rc.recommendation, rc.conviction_score,
               us.sector
        FROM research_cache rc
        LEFT JOIN universe_scores us ON us.ticker = rc.ticker
        WHERE rc.researched_at >= ?
    """, (cutoff,)).fetchall()
    conn.close()

    rows = [dict(r) for r in rows]

    if not rows:
        return {
            "available": False,
            "message": "No research cache entries in the last 90 days.",
            "market_signal": "unknown",
        }

    mos_values = []
    at_mos_count = 0
    sector_data = {}

    for r in rows:
        mos = None
        try:
            report = _json.loads(r["report_json"]) if r["report_json"] else {}
            # Try valuation_inputs first, then top-level
            vi = report.get("valuation_inputs", {})
            mos = (
                vi.get("margin_of_safety_pct")
                or vi.get("margin_of_safety")
                or report.get("margin_of_safety_pct")
                or report.get("margin_of_safety")
            )
            if mos is None:
                # Try intrinsic_value section
                iv_section = report.get("intrinsic_value", {})
                mos = iv_section.get("margin_of_safety_pct") or iv_section.get("margin_of_safety")
        except Exception:
            pass

        if mos is not None:
            try:
                mos = float(mos)
            except (TypeError, ValueError):
                mos = None

        if mos is not None:
            mos_values.append(mos)
            at_mos = mos >= 20
            if at_mos:
                at_mos_count += 1

            sec = r.get("sector") or "Unknown"
            sector_data.setdefault(sec, {"total": 0, "at_mos": 0})
            sector_data[sec]["total"] += 1
            if at_mos:
                sector_data[sec]["at_mos"] += 1

    total_researched = len(rows)

    if not mos_values:
        return {
            "available": True,
            "total_researched": total_researched,
            "mos_data_available": 0,
            "message": "Research cache found but no MoS data extractable from reports.",
            "market_signal": "unknown",
        }

    mos_data_count = len(mos_values)
    at_mos_pct = round(at_mos_count / mos_data_count * 100, 1)
    avg_mos = round(sum(mos_values) / len(mos_values), 1)

    if at_mos_pct > 40:
        market_signal = "cheap"
    elif at_mos_pct >= 20:
        market_signal = "fair"
    else:
        market_signal = "expensive"

    sector_breakdown = {}
    for sec, d in sector_data.items():
        pct = round(d["at_mos"] / d["total"] * 100, 1) if d["total"] > 0 else 0
        sector_breakdown[sec] = {
            "total": d["total"],
            "at_mos": d["at_mos"],
            "pct_at_mos": pct,
        }

    return {
        "available": True,
        "total_researched": total_researched,
        "mos_data_available": mos_data_count,
        "at_mos_count": at_mos_count,
        "at_mos_pct": at_mos_pct,
        "avg_mos_pct": avg_mos,
        "market_signal": market_signal,
        "sector_breakdown": sector_breakdown,
        "interpretation": (
            f"{at_mos_pct:.0f}% of recently researched stocks offer ≥20% margin of safety. "
            f"Market appears {market_signal}. "
            + ("High availability of opportunities — be selective but active." if market_signal == "cheap"
               else "Moderate opportunity set — apply strict moat and quality filters." if market_signal == "fair"
               else "Few stocks at required margin of safety — raise the quality bar; prefer cash over stretching valuation.")
        ),
    }


def get_sector_rotation_signal() -> dict:
    """
    Detect sector availability bias and portfolio tilt vs. opportunity set.
    Compares: (1) portfolio sector weights, (2) recently researched sector distribution,
    (3) full universe sector distribution.
    """
    import sqlite3
    conn = portfolio._get_connection()

    # Portfolio sector weights (use cost basis as proxy for weight)
    holdings = portfolio.get_holdings()
    cash = portfolio.get_cash()
    portfolio_sector_mv: dict = {}
    total_portfolio_cost = cash
    for h in holdings:
        row = conn.execute("SELECT sector FROM universe_scores WHERE ticker=?", (h["ticker"],)).fetchone()
        sector = row["sector"] if row and row["sector"] else "Unknown"
        cost = h["shares"] * h["avg_cost"]
        portfolio_sector_mv[sector] = portfolio_sector_mv.get(sector, 0) + cost
        total_portfolio_cost += cost

    portfolio_weights = {
        s: round(v / total_portfolio_cost * 100, 1)
        for s, v in portfolio_sector_mv.items()
    } if total_portfolio_cost > 0 else {}

    # Recently researched sector distribution (last 60 days)
    cache_rows = conn.execute("""
        SELECT rc.ticker, us.sector
        FROM research_cache rc
        LEFT JOIN universe_scores us ON rc.ticker = us.ticker
        WHERE rc.researched_at > datetime('now', '-60 days')
    """).fetchall()
    researched_by_sector: dict = {}
    for r in cache_rows:
        s = r["sector"] or "Unknown"
        researched_by_sector[s] = researched_by_sector.get(s, 0) + 1
    total_researched = sum(researched_by_sector.values()) or 1
    researched_weights = {s: round(c / total_researched * 100, 1) for s, c in researched_by_sector.items()}

    # Universe sector distribution
    universe_rows = conn.execute(
        "SELECT sector, COUNT(*) as cnt FROM universe_scores WHERE sector IS NOT NULL GROUP BY sector"
    ).fetchall()
    conn.close()
    total_universe = sum(r["cnt"] for r in universe_rows) or 1
    universe_weights = {r["sector"]: round(r["cnt"] / total_universe * 100, 1) for r in universe_rows}

    # Availability bias: sectors researched significantly more/less than universe weight
    availability_bias = []
    for s, rw in researched_weights.items():
        uw = universe_weights.get(s, 0)
        if uw > 0:
            tilt = rw - uw
            if abs(tilt) > 10:
                availability_bias.append({
                    "sector": s,
                    "researched_pct": rw,
                    "universe_pct": uw,
                    "tilt": round(tilt, 1),
                    "signal": "over-researching" if tilt > 0 else "under-researching",
                })

    # Portfolio tilt vs. opportunity set
    portfolio_tilts = []
    for s, pw in portfolio_weights.items():
        rw = researched_weights.get(s, 0)
        tilt = pw - rw
        if abs(tilt) > 15:
            portfolio_tilts.append({
                "sector": s,
                "portfolio_pct": pw,
                "opportunity_set_pct": rw,
                "tilt": round(tilt, 1),
                "signal": "portfolio overweight vs opportunity set" if tilt > 0 else "portfolio underweight vs opportunity set",
            })

    bias_summary = (
        f"Availability bias in: {', '.join(f['sector'] + ' (' + ('+' if f['tilt'] > 0 else '') + str(f['tilt']) + '%)' for f in availability_bias)}"
        if availability_bias else "No significant availability bias detected."
    )

    return {
        "portfolio_sector_weights": portfolio_weights,
        "researched_sector_distribution": researched_weights,
        "universe_sector_distribution": universe_weights,
        "availability_bias_flags": availability_bias,
        "portfolio_tilt_flags": portfolio_tilts,
        "total_stocks_researched_60d": total_researched,
        "summary": bias_summary,
    }


def run_iv_postmortem() -> dict:
    """
    Analyse past IV predictions vs actual outcomes. Save calibration insights to KB.

    For each reconciled prediction (has outcome_price + predicted_iv):
    - Compute iv_accuracy_pct = (outcome_price / predicted_iv - 1) * 100
      positive = IV was too conservative (stock exceeded estimate)
      negative = IV was too high (stock never reached estimate)
    - Group by conviction bucket and by sector (via universe_scores join)
    - Save KB notes: per-bucket calibration, per-sector bias patterns
    - Return summary dict
    """
    from agent.portfolio import DB_PATH, _get_connection
    from agent.knowledge_base import save_kb_note

    conn = _get_connection()
    rows = conn.execute("""
        SELECT pt.ticker, pt.conviction_score, pt.predicted_iv, pt.outcome_price,
               pt.outcome_return_pct, us.sector
        FROM prediction_tracking pt
        LEFT JOIN universe_scores us ON us.ticker = pt.ticker
        WHERE pt.outcome_date IS NOT NULL
          AND pt.predicted_iv IS NOT NULL
          AND pt.predicted_iv > 0
          AND pt.outcome_price IS NOT NULL
    """).fetchall()
    conn.close()

    rows = [dict(r) for r in rows]

    if len(rows) < 3:
        return {
            "available": False,
            "message": "Insufficient data — need ≥3 reconciled predictions with IV estimates",
        }

    # Compute iv_accuracy_pct for each row
    for r in rows:
        r["iv_accuracy_pct"] = (r["outcome_price"] / r["predicted_iv"] - 1) * 100

    total_analysed = len(rows)

    # Group by conviction bucket
    def _bucket(score):
        if score is None:
            return None
        if score >= 9:
            return "9-10"
        elif score >= 7:
            return "7-8"
        elif score >= 5:
            return "5-6"
        return None

    conviction_groups = {}
    for r in rows:
        b = _bucket(r.get("conviction_score"))
        if b:
            conviction_groups.setdefault(b, []).append(r["iv_accuracy_pct"])

    conviction_calibration = {}
    kb_notes_saved = 0
    insights = []

    for bucket, accuracies in conviction_groups.items():
        avg = round(sum(accuracies) / len(accuracies), 2)
        std = round(float(np.std(accuracies)), 2) if len(accuracies) > 1 else None
        count = len(accuracies)
        conviction_calibration[bucket] = {"avg_iv_accuracy_pct": avg, "std": std, "count": count}
        if count >= 3:
            direction = "underestimating" if avg > 0 else "overestimating"
            insight = (
                f"Conviction {bucket}: {direction} IV by {abs(avg):.1f}% on average "
                f"(n={count}, std={std}). "
                + ("Stock typically exceeds IV estimate." if avg > 0
                   else "Stock typically falls short of IV estimate.")
            )
            insights.append(insight)
            save_kb_note(
                topic="iv_methodology",
                title=f"IV calibration: conviction {bucket}",
                content=(
                    f"Historical IV accuracy for conviction bucket {bucket}: avg_iv_accuracy_pct={avg}%, "
                    f"std={std}%, n={count}. "
                    f"Interpretation: when conviction is {bucket}, the IV estimate is typically "
                    f"{direction} actual outcome by {abs(avg):.1f}%. "
                    f"Consider {'raising' if avg > 5 else 'lowering' if avg < -5 else 'keeping'} "
                    f"IV estimates for this conviction range."
                ),
                tags=["conviction", "calibration", "postmortem"],
            )
            kb_notes_saved += 1

    # Group by sector (only sectors with ≥3 data points)
    sector_groups = {}
    for r in rows:
        sec = r.get("sector")
        if sec:
            sector_groups.setdefault(sec, []).append(r["iv_accuracy_pct"])

    sector_calibration = {}
    for sector, accuracies in sector_groups.items():
        count = len(accuracies)
        if count < 3:
            continue
        avg = round(sum(accuracies) / len(accuracies), 2)
        sector_calibration[sector] = {"avg_iv_accuracy_pct": avg, "count": count}
        direction = "underestimating" if avg > 0 else "overestimating"
        insight = (
            f"{sector}: {direction} IV by {abs(avg):.1f}% on average (n={count})"
        )
        insights.append(insight)
        save_kb_note(
            topic="iv_methodology",
            title=f"IV calibration: {sector} sector",
            content=(
                f"Historical IV accuracy for {sector} sector: avg_iv_accuracy_pct={avg}%, n={count}. "
                f"Interpretation: IV estimates for {sector} stocks are typically "
                f"{direction} actual outcome by {abs(avg):.1f}%. "
                f"Consider {'adding a premium to' if avg > 5 else 'applying a haircut to' if avg < -5 else 'no adjustment needed for'} "
                f"IV estimates in this sector."
            ),
            tags=[sector, "calibration", "postmortem"],
        )
        kb_notes_saved += 1

    return {
        "available": True,
        "total_analysed": total_analysed,
        "conviction_calibration": conviction_calibration,
        "sector_calibration": sector_calibration,
        "insights": insights,
        "kb_notes_saved": kb_notes_saved,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Conviction Calibration from Prediction History
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 7. Decision Thresholds — permanent rules + ML factor guidance
# ══════════════════════════════════════════════════════════════════════════════
#
# Design principle: this is a 20+ year intrinsic-value portfolio.
# The MoS threshold is a DISCIPLINE RULE, not a parameter to optimise.
# Buffett didn't tune his margin of safety on past trades — he set it as a
# permanent principle to protect against estimation error over long horizons.
#
# ML's role here is narrow and appropriate:
#   - Tell the agent which screener FACTORS have actually predicted returns
#     in this portfolio (kicks in at 5+ closed trades via get_ml_factor_weights)
#   - Calibrate position sizing confidence (conviction calibration)
#   - Surface candidate stocks faster (prioritize_watchlist_ml)
#
# What ML does NOT do:
#   - Adjust the MoS threshold (regime rules are permanent)
#   - Override the moat requirement (qualitative, not learnable from price data)
#   - Change the bear verdict handling (disciplined conservatism)
# ══════════════════════════════════════════════════════════════════════════════

# Regime-adjusted MoS thresholds — permanent, academically grounded.
# Higher regimes = more macro uncertainty = wider safety cushion required.
_REGIME_MOS_DEFAULTS = {
    "NORMAL":       20.0,   # standard Buffett/Munger threshold
    "INFLATIONARY": 22.0,   # inflation erodes real returns; need more cushion
    "HIGH_RATES":   23.0,   # higher discount rate compresses IV; be patient
    "RISK_OFF":     25.0,   # market stress = estimation error rises
    "STAGFLATION":  28.0,   # worst regime; capital preservation > deployment
}

def get_decision_thresholds() -> dict:
    """
    Return the permanent decision thresholds for the buy/watchlist/pass matrix,
    plus ML factor guidance to sharpen candidate ranking.

    This is a 20+ year intrinsic-value portfolio. The MoS threshold is a
    permanent discipline rule — not a parameter to tune from trade history.
    ML's role is limited to improving candidate RANKING (which factors have
    actually predicted returns), not changing the buy threshold.

    Returns:
        mos_threshold_pct       Minimum margin of safety required to buy.
                                Regime-adjusted. Permanent — does not change.
        bear_override_conviction Bear conviction ≥ this treats "caution" as "reject".
        decision_matrix         Ready-to-use action rules.
        factor_guidance         From get_ml_factor_weights — use when ranking
                                screener candidates, not when deciding to buy.
    """
    regime, _, _ = _detect_regime()
    mos_threshold = _REGIME_MOS_DEFAULTS.get(regime, 20.0)
    bear_override = 8  # permanent: high-conviction bear caution = reject

    # ── ML factor guidance (for ranking candidates, not for buy/no-buy) ───────
    try:
        fw = get_ml_factor_weights()
        factor_guidance = fw.get("actionable_guidance", [])
        blended_weights = fw.get("blended_weights", {})
        n_closed = fw.get("n_closed_trades", 0)
        data_note = (
            f"{n_closed} closed trade(s) in history. "
            + ("Factor weights are regime priors — more trades will sharpen candidate ranking."
               if n_closed < 5 else
               f"Factor weights are {fw.get('data_weight_pct', 0)}% data-driven.")
        )
    except Exception:
        factor_guidance = []
        blended_weights = {}
        data_note = "Factor weight data unavailable."

    return {
        "regime":                   regime,
        "mos_threshold_pct":        mos_threshold,
        "bear_override_conviction": bear_override,
        "half_size_on_caution":     True,
        "mos_rationale": (
            f"Permanent regime rule: {mos_threshold:.0f}% MoS required in {regime} regime. "
            "This threshold does not change — it protects against IV estimation error "
            "over a 20+ year holding horizon."
        ),
        "watchlist_formula": (
            f"target_price = intrinsic_value × {1 - mos_threshold/100:.3f}  "
            f"(IV discounted by {mos_threshold:.0f}%)"
        ),
        "decision_matrix": {
            "BUY_FULL":  f"moat=True AND MoS ≥ {mos_threshold:.0f}% AND bear='proceed'",
            "BUY_HALF":  f"moat=True AND MoS ≥ {mos_threshold:.0f}% AND bear='caution' AND bear_conviction < {bear_override}",
            "WATCHLIST": f"moat=True AND MoS < {mos_threshold:.0f}% → target = IV × {1 - mos_threshold/100:.3f}",
            "SHADOW":    f"moat=False OR bear='reject' OR (bear='caution' AND bear_conviction ≥ {bear_override})",
        },
        "ml_factor_guidance": {
            "purpose": "Use blended_weights when RANKING screener candidates — not when deciding whether to buy.",
            "data_note": data_note,
            "blended_weights": blended_weights,
            "actionable_guidance": factor_guidance,
        },
    }
