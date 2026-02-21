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

    ranked = []
    for item in watchlist:
        ticker = item["ticker"]
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
