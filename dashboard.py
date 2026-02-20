#!/usr/bin/env python3
"""
Bloomberg-style Investment Terminal Dashboard
Run with: streamlit run dashboard.py
"""

import html
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Bootstrap ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv, dotenv_values
load_dotenv(override=True)

# Read STARTING_CASH directly from .env file so it is never affected by
# stale shell environment variables inherited by the Streamlit process.
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_dotenv_vals = dotenv_values(_env_file)
STARTING_CASH = float(_dotenv_vals.get("STARTING_CASH") or os.environ.get("STARTING_CASH") or "100000")

from agent.portfolio import initialize_portfolio, get_reflections, reset_portfolio
initialize_portfolio(STARTING_CASH)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="INVESTMENT TERMINAL",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bloomberg CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

:root {
    --bg:      #000000;
    --panel:   #080808;
    --border:  #1a1a1a;
    --orange:  #FF8000;
    --orange2: #FFA040;
    --green:   #00E676;
    --red:     #FF3B3B;
    --text:    #C8C8C8;
    --dim:     #505050;
    --white:   #FFFFFF;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}

/* ─ Sidebar ─ */
[data-testid="stSidebar"] {
    background-color: #030303 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}
[data-testid="stSidebar"] label {
    color: var(--dim) !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: var(--text) !important;
    font-size: 13px !important;
    padding: 5px 0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    color: var(--orange) !important;
}

/* ─ Main area ─ */
.main .block-container {
    background-color: var(--bg) !important;
    padding: 1rem 2rem !important;
    max-width: 100% !important;
}

/* ─ Metrics ─ */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--orange) !important;
    padding: 14px 16px !important;
    border-radius: 0 !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--dim) !important;
    font-size: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
[data-testid="stMetricValue"] {
    color: var(--orange) !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ─ Buttons ─ */
.stButton > button {
    background-color: var(--orange) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background-color: var(--orange2) !important;
}
.stButton > button:disabled {
    background-color: #2a2a2a !important;
    color: #555 !important;
}

/* ─ Inputs ─ */
.stTextInput > div > div > input,
.stTextArea textarea {
    background-color: #050505 !important;
    color: var(--orange) !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    caret-color: var(--orange) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 1px var(--orange) !important;
}
.stTextInput label, .stTextArea label {
    color: var(--dim) !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ─ Dataframe ─ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] th {
    background: #0d0d0d !important;
    color: var(--dim) !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
    color: var(--text) !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    background: var(--panel) !important;
}

/* ─ Expander ─ */
[data-testid="stExpander"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    color: var(--orange) !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ─ Spinner ─ */
.stSpinner > div { border-top-color: var(--orange) !important; }

/* ─ Scrollbar ─ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #2a2a2a; }
::-webkit-scrollbar-thumb:hover { background: var(--orange); }

/* ─ Markdown ─ */
.stMarkdown p, .stMarkdown li { color: var(--text) !important; font-family: 'IBM Plex Mono', monospace !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: var(--orange) !important; }
.stMarkdown code { color: var(--orange2) !important; background: #0d0d0d !important; }
hr { border-color: var(--border) !important; }

/* ─ Success/error ─ */
[data-testid="stAlert"] { border-radius: 0 !important; font-family: 'IBM Plex Mono', monospace !important; }

/* ─ Hide ALL Streamlit chrome ─ */
header, header *, footer, footer *,
#MainMenu, #MainMenu *,
[data-testid="stHeader"], [data-testid="stHeader"] *,
[data-testid="stToolbar"], [data-testid="stToolbar"] *,
[data-testid="stToolbarActions"], [data-testid="stToolbarActions"] *,
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"],
[class*="toolbar"], [class*="Toolbar"],
[class*="StatusWidget"], [class*="keyboardShortcut"],
button[aria-label*="keyboard" i], button[aria-label*="shortcut" i],
button[title*="keyboard" i], button[title*="shortcut" i] {
    display:    none !important;
    visibility: hidden !important;
    height:     0 !important;
    max-height: 0 !important;
    overflow:   hidden !important;
    opacity:    0 !important;
    pointer-events: none !important;
}
.main .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Utility functions ─────────────────────────────────────────────────────────

def fmt_usd(val):
    if val is None:
        return "—"
    prefix = "-$" if val < 0 else "$"
    return f"{prefix}{abs(val):,.2f}"


def fmt_pct(val):
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def clr(val, text=None):
    """Wrap a value in green/red HTML based on sign."""
    if val is None:
        return "—"
    t = text if text is not None else str(val)
    color = "#00E676" if val >= 0 else "#FF3B3B"
    return f'<span style="color:{color};font-weight:500">{t}</span>'


def section(label):
    st.markdown(
        f'<div style="color:#505050;font-size:10px;letter-spacing:2px;'
        f'text-transform:uppercase;border-bottom:1px solid #1a1a1a;'
        f'padding-bottom:5px;margin:20px 0 12px 0;'
        f'font-family:\'IBM Plex Mono\',monospace;">◈ {label}</div>',
        unsafe_allow_html=True,
    )


def page_header(title):
    st.markdown(
        f'<div style="border-bottom:2px solid #FF8000;padding-bottom:8px;margin-bottom:20px;">'
        f'<span style="color:#FF8000;font-size:15px;font-weight:700;'
        f'letter-spacing:3px;font-family:\'IBM Plex Mono\',monospace;">◈ {title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def agent_box(lines):
    """Render the agent output box."""
    inner = "<br>".join(lines)
    return (
        f'<div style="background:#050505;border:1px solid #1a1a1a;border-left:3px solid #FF8000;'
        f'padding:16px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#C8C8C8;'
        f'max-height:600px;overflow-y:auto;line-height:1.8;white-space:pre-wrap;">'
        f'{inner}</div>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 20px;border-bottom:1px solid #1a1a1a;margin-bottom:20px;">
        <div style="color:#FF8000;font-size:20px;font-weight:700;letter-spacing:4px;
                    font-family:'IBM Plex Mono',monospace;">◈ TERMINAL</div>
        <div style="color:#505050;font-size:10px;letter-spacing:2px;margin-top:2px;
                    font-family:'IBM Plex Mono',monospace;">AI INVESTMENT PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    now = datetime.now()
    st.markdown(
        f'<div style="color:#505050;font-size:11px;letter-spacing:1px;margin-bottom:20px;'
        f'font-family:\'IBM Plex Mono\',monospace;">'
        f'{now.strftime("%a %b %d %Y")}<br>{now.strftime("%H:%M:%S")} LOCAL</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "SECTION",
        [
            "PORTFOLIO",
            "PERFORMANCE",
            "TRADES",
            "AI REVIEW",
            "ASK AGENT",
            "REFLECTIONS",
        ],
        key="page_selector",
    )

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    MODEL_OPTIONS = {
        "Haiku 4.5  (fast)": "claude-haiku-4-5-20251001",
        "Sonnet 4.6 (balanced)": "claude-sonnet-4-6",
        "Opus 4.6   (best)": "claude-opus-4-6",
    }
    selected_model_label = st.selectbox(
        "AI MODEL",
        list(MODEL_OPTIONS.keys()),
        index=1,
        key="model_selector",
    )
    selected_model = MODEL_OPTIONS[selected_model_label]

    st.markdown("""
    <div style="margin-top:32px;border-top:1px solid #1a1a1a;padding-top:12px;">
        <div style="color:#505050;font-size:9px;letter-spacing:1px;
                    font-family:'IBM Plex Mono',monospace;">PAPER TRADING ONLY<br>NO REAL MONEY AT RISK</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
if "PORTFOLIO" in page:
    page_header("PORTFOLIO — LIVE OVERVIEW")
    _, _rb = st.columns([6, 1])
    with _rb:
        if st.button("⟳  REFRESH", key="refresh_portfolio", use_container_width=True):
            st.rerun()

    from agent.tools import _get_portfolio_status

    with st.spinner("FETCHING LIVE MARKET DATA..."):
        status = _get_portfolio_status()

    import datetime as _dt
    st.caption(f"Data fetched at {_dt.datetime.now().strftime('%H:%M:%S')} — if this timestamp is current, you are seeing live data.")

    cash = status["cash"]
    total = status["total_portfolio_value"]
    invested = status["total_invested_value"]
    pnl = status["total_unrealized_pnl"]
    cost_basis = status.get("total_cost_basis") or 0
    pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0
    positions = status["number_of_positions"]
    cash_pct = (cash / total * 100) if total else 100.0

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TOTAL VALUE", fmt_usd(total))
    c2.metric("CASH", fmt_usd(cash), f"{cash_pct:.1f}% of portfolio")
    c3.metric("INVESTED", fmt_usd(invested))
    c4.metric(
        "UNREALIZED P&L",
        fmt_usd(pnl),
        fmt_pct(pnl_pct),
        delta_color="normal",
    )
    c5.metric("OPEN POSITIONS", str(positions))

    # ── Holdings ─────────────────────────────────────────────────────────────
    section("HOLDINGS")

    holdings = status.get("holdings", [])

    if not holdings:
        st.markdown(
            '<div style="color:#505050;font-size:13px;padding:30px 0;text-align:center;">'
            'NO POSITIONS OPEN — PORTFOLIO IS 100% CASH</div>',
            unsafe_allow_html=True,
        )
    else:
        rows = []
        for h in holdings:
            up = h.get("unrealized_pnl") or 0
            up_pct = h.get("unrealized_pct") or 0
            rows.append({
                "TICKER":        h["ticker"],
                "NAME":          h.get("name") or "",
                "INDUSTRY":      h.get("industry") or "",
                "SHARES":        f"{h['shares']:.4f}",
                "AVG COST":      fmt_usd(h["avg_cost"]),
                "CURRENT PRICE": fmt_usd(h.get("current_price")),
                "MKT VALUE":     fmt_usd(h.get("market_value")),
                "UNREAL P&L":    up,
                "RETURN":        up_pct,
                "FIRST BOUGHT":  (h.get("first_bought") or "")[:10],
            })

        df = pd.DataFrame(rows)

        def _color_signed(col):
            return col.apply(lambda v: "color: #00E676" if v >= 0 else "color: #FF3B3B")

        styled = (
            df.style
            .apply(_color_signed, subset=["UNREAL P&L", "RETURN"])
            .format({
                "UNREAL P&L": lambda v: ("-$" if v < 0 else "$") + f"{abs(v):,.2f}",
                "RETURN":     lambda v: ("+" if v >= 0 else "") + f"{v:.2f}%",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Allocation chart ─────────────────────────────────────────────────
        section("PORTFOLIO ALLOCATION")

        alloc = [{"label": h["ticker"], "value": h["market_value"]}
                 for h in holdings if h.get("market_value")]
        if cash > 0:
            alloc.append({"label": "CASH", "value": cash})

        if alloc:
            alloc_df = pd.DataFrame(alloc)
            palette = [
                "#FF8000", "#FFA040", "#FFB870", "#CC6600",
                "#FF6600", "#FFD700", "#00E676", "#40C4FF",
                "#EA80FC", "#FF5252", "#69F0AE", "#FFD740",
            ]
            fig = go.Figure(go.Pie(
                labels=alloc_df["label"],
                values=alloc_df["value"],
                hole=0.62,
                marker=dict(
                    colors=palette[:len(alloc_df)],
                    line=dict(color="#000", width=2),
                ),
                textfont=dict(family="IBM Plex Mono", size=11),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="#000",
                plot_bgcolor="#000",
                font=dict(family="IBM Plex Mono", color="#FF8000"),
                legend=dict(
                    bgcolor="#080808",
                    bordercolor="#1a1a1a",
                    font=dict(color="#C8C8C8", size=11),
                ),
                margin=dict(l=0, r=0, t=8, b=8),
                height=320,
                annotations=[dict(
                    text=f"<b>${total/1000:.1f}K</b>",
                    x=0.5, y=0.5,
                    font=dict(size=16, color="#FF8000", family="IBM Plex Mono"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig, use_container_width=True)


    # ── Danger zone ──────────────────────────────────────────────────────────
    section("DANGER ZONE")

    if "reset_confirm" not in st.session_state:
        st.session_state.reset_confirm = False

    if not st.session_state.reset_confirm:
        st.markdown(
            '<div style="color:#505050;font-size:11px;margin-bottom:10px;">'
            'Clears all positions and restores starting cash. Trade history is preserved.'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("⚠  RESET PORTFOLIO"):
            st.session_state.reset_confirm = True
            st.rerun()
    else:
        st.markdown(
            '<div style="color:#FF3B3B;font-size:12px;letter-spacing:1px;margin-bottom:12px;">'
            '⚠ THIS WILL WIPE ALL POSITIONS AND RESET CASH — ARE YOU SURE?'
            '</div>',
            unsafe_allow_html=True,
        )
        col_confirm, col_cancel, _ = st.columns([1, 1, 4])
        with col_confirm:
            if st.button("✓  CONFIRM RESET"):
                reset_portfolio(STARTING_CASH)
                # Also discard any interrupted session checkpoint
                _ckpt = "data/session_checkpoint.json"
                if os.path.exists(_ckpt):
                    os.remove(_ckpt)
                st.session_state.reset_confirm = False
                st.success(f"Portfolio reset — cash restored to ${STARTING_CASH:,.0f}.")
                st.rerun()
        with col_cancel:
            if st.button("✗  CANCEL"):
                st.session_state.reset_confirm = False
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "PERFORMANCE" in page:
    page_header("PERFORMANCE — VS S&P 500")
    _, _rb = st.columns([6, 1])
    with _rb:
        if st.button("⟳  REFRESH", key="refresh_performance", use_container_width=True):
            st.rerun()

    from agent.portfolio import get_portfolio_snapshots
    from agent.tools import _handle_benchmark_comparison

    with st.spinner("LOADING PERFORMANCE DATA..."):
        bench = _handle_benchmark_comparison()
        snapshots = get_portfolio_snapshots(limit=200)

    # ── Benchmark metrics ─────────────────────────────────────────────────────
    if bench.get("portfolio_return_pct") is not None:
        port_ret = bench.get("portfolio_return_pct", 0)
        bench_ret = bench.get("benchmark_return_pct")
        alpha = bench.get("alpha_pct")
        beating = bench.get("is_beating_benchmark")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PORTFOLIO RETURN", fmt_pct(port_ret), delta=fmt_pct(port_ret))
        c2.metric("S&P 500 RETURN", fmt_pct(bench_ret) if bench_ret is not None else "N/A")
        c3.metric(
            "ALPHA (OUTPERFORMANCE)",
            fmt_pct(alpha) if alpha is not None else "N/A",
            "BEATING INDEX" if beating else "LAGGING INDEX",
            delta_color="normal" if beating else "inverse",
        )
        c4.metric("CURRENT VALUE", fmt_usd(bench.get("portfolio_current_value")))
    else:
        st.info("No performance data yet — run an AI Review to start tracking.")

    # ── Chart ─────────────────────────────────────────────────────────────────
    section("PORTFOLIO VALUE OVER TIME")

    if snapshots:
        df = pd.DataFrame(snapshots)
        df["ts"] = pd.to_datetime(df["ts"]).dt.floor("D")
        df = df.drop_duplicates(subset="ts").sort_values("ts")

        time_span = df["ts"].max() - df["ts"].min()
        tick_fmt = "%b %d '%y" if time_span.days > 365 else "%b %d"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["ts"],
            y=df["portfolio_value"],
            mode="lines+markers",
            name="MY PORTFOLIO",
            line=dict(color="#FF8000", width=2),
            marker=dict(size=5, color="#FF8000"),
            fill="tozeroy",
            fillcolor="rgba(255,128,0,0.06)",
            hovertemplate="<b>Portfolio:</b> $%{y:,.2f}<extra></extra>",
        ))

        if df["benchmark_price"].notna().any():
            first_spy = df["benchmark_price"].dropna().iloc[0]
            first_val = df["portfolio_value"].iloc[0]
            df["spy_indexed"] = df["benchmark_price"] / first_spy * first_val
            fig.add_trace(go.Scatter(
                x=df["ts"],
                y=df["spy_indexed"],
                mode="lines",
                name="S&P 500 (INDEXED)",
                line=dict(color="#404040", width=1, dash="dot"),
                hovertemplate="<b>S&P 500:</b> $%{y:,.2f}<extra></extra>",
            ))

        fig.update_layout(
            paper_bgcolor="#000",
            plot_bgcolor="#000",
            font=dict(family="IBM Plex Mono", color="#FF8000"),
            xaxis=dict(
                gridcolor="#0d0d0d",
                tickfont=dict(color="#505050", size=10),
                showline=True,
                linecolor="#1a1a1a",
                title=None,
                tickformat=tick_fmt,
                nticks=10,
            ),
            yaxis=dict(
                gridcolor="#0d0d0d",
                tickfont=dict(color="#505050", size=10),
                tickprefix="$",
                showline=True,
                linecolor="#1a1a1a",
                title=None,
            ),
            legend=dict(
                bgcolor="#080808",
                bordercolor="#1a1a1a",
                font=dict(color="#C8C8C8", size=11),
            ),
            margin=dict(l=0, r=0, t=8, b=0),
            height=420,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Risk metrics ──────────────────────────────────────────────────────
        from agent.portfolio import get_portfolio_metrics
        metrics = get_portfolio_metrics()

        section("RISK METRICS")
        if metrics.get("sharpe_ratio") is not None or metrics.get("max_drawdown_pct") is not None:
            r1, r2, r3, r4 = st.columns(4)
            sharpe = metrics.get("sharpe_ratio")
            drawdown = metrics.get("max_drawdown_pct")
            vol = metrics.get("annualised_volatility_pct")
            ann_ret = metrics.get("annualised_return_pct")

            r1.metric(
                "SHARPE RATIO",
                f"{sharpe:.2f}" if sharpe is not None else "N/A",
                "GOOD" if sharpe is not None and sharpe > 1 else ("WEAK" if sharpe is not None and sharpe < 0 else None),
                delta_color="normal" if sharpe is not None and sharpe > 0 else "inverse",
            )
            r2.metric(
                "MAX DRAWDOWN",
                f"-{drawdown:.1f}%" if drawdown is not None else "N/A",
                delta_color="off",
            )
            r3.metric(
                "ANNUALISED VOL",
                f"{vol:.1f}%" if vol is not None else "N/A",
                delta_color="off",
            )
            r4.metric(
                "ANNUALISED RETURN",
                f"{ann_ret:+.1f}%" if ann_ret is not None else "N/A",
                delta_color="off",
            )
            if metrics.get("note"):
                st.markdown(
                    f'<div style="color:#505050;font-size:10px;margin-top:4px;">{metrics["note"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#505050;font-size:12px;">Not enough snapshots yet — metrics appear after 2+ reviews.</div>',
                unsafe_allow_html=True,
            )

        # ── Rolling returns ───────────────────────────────────────────────────
        section("ROLLING RETURNS VS S&P 500")
        rolling = metrics.get("rolling", {})
        periods = ["1m", "3m", "6m"]
        labels = ["1 MONTH", "3 MONTHS", "6 MONTHS"]
        port_vals = [rolling.get(p, {}).get("portfolio_pct") for p in periods]
        spy_vals  = [rolling.get(p, {}).get("benchmark_pct") for p in periods]

        has_rolling = any(v is not None for v in port_vals)
        if has_rolling:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="MY PORTFOLIO",
                x=labels,
                y=port_vals,
                marker_color=["#FF8000" if (v or 0) >= 0 else "#FF3B3B" for v in port_vals],
                text=[f"{v:+.1f}%" if v is not None else "N/A" for v in port_vals],
                textposition="outside",
                textfont=dict(color="#C8C8C8", size=11),
            ))
            fig2.add_trace(go.Bar(
                name="S&P 500",
                x=labels,
                y=spy_vals,
                marker_color="#303030",
                text=[f"{v:+.1f}%" if v is not None else "N/A" for v in spy_vals],
                textposition="outside",
                textfont=dict(color="#505050", size=11),
            ))
            fig2.update_layout(
                paper_bgcolor="#000",
                plot_bgcolor="#000",
                barmode="group",
                bargap=0.3,
                bargroupgap=0.05,
                font=dict(family="IBM Plex Mono", color="#FF8000"),
                xaxis=dict(gridcolor="#0d0d0d", tickfont=dict(color="#505050", size=11), title=None),
                yaxis=dict(gridcolor="#0d0d0d", tickfont=dict(color="#505050", size=10), ticksuffix="%", title=None),
                legend=dict(bgcolor="#080808", bordercolor="#1a1a1a", font=dict(color="#C8C8C8", size=11)),
                margin=dict(l=0, r=0, t=20, b=0),
                height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(
                '<div style="color:#505050;font-size:12px;">Rolling return data builds over time — check back after more monthly reviews.</div>',
                unsafe_allow_html=True,
            )

        # ── Snapshot table ────────────────────────────────────────────────────
        section("SNAPSHOT HISTORY")
        snap_rows = []
        for s in reversed(snapshots[-20:]):
            snap_rows.append({
                "DATE":            s["ts"][:16].replace("T", "  "),
                "PORTFOLIO VALUE": fmt_usd(s["portfolio_value"]),
                "CASH":            fmt_usd(s["cash"]),
                "INVESTED":        fmt_usd(s["invested_value"]),
                "S&P 500 PRICE":   fmt_usd(s.get("benchmark_price")),
            })
        st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)

    else:
        st.markdown(
            '<div style="color:#505050;text-align:center;padding:60px 0;font-size:13px;">'
            'NO SNAPSHOT DATA YET<br><br>'
            '<span style="font-size:11px;">Run an AI Review to begin tracking performance vs benchmark</span>'
            '</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRADES
# ═══════════════════════════════════════════════════════════════════════════════
elif "TRADES" in page:
    page_header("TRADE HISTORY")

    from agent.portfolio import get_transactions
    txs = get_transactions(limit=200)

    if not txs:
        st.markdown(
            '<div style="color:#505050;text-align:center;padding:60px 0;">NO TRADES EXECUTED YET</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Summary metrics ───────────────────────────────────────────────────
        buys  = [t for t in txs if t["action"] == "BUY"]
        sells = [t for t in txs if t["action"] == "SELL"]
        realized = [t["realized_pnl"] for t in sells if t.get("realized_pnl") is not None]
        total_realized = sum(realized)
        winners = [p for p in realized if p > 0]
        win_rate = len(winners) / len(realized) * 100 if realized else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("TOTAL TRADES", str(len(txs)))
        c2.metric("BUYS", str(len(buys)))
        c3.metric("SELLS", str(len(sells)))
        c4.metric("REALIZED P&L", fmt_usd(total_realized), delta=fmt_pct(0) if not realized else None)
        c5.metric("WIN RATE", f"{win_rate:.0f}%" if realized else "—")

        # ── Trade table ───────────────────────────────────────────────────────
        section("ALL TRANSACTIONS")

        rows = []
        for t in txs:
            pnl = t.get("realized_pnl")
            notes = t.get("notes") or ""
            rows.append({
                "TIMESTAMP":    t["ts"][:16].replace("T", "  "),
                "ACTION":       t["action"],
                "TICKER":       t["ticker"],
                "SHARES":       f"{t['shares']:.4f}",
                "PRICE":        f"${t['price']:,.2f}",
                "TOTAL":        f"${t['total']:,.2f}",
                "REALIZED P&L": fmt_usd(pnl) if pnl is not None else "—",
                "NOTES":        notes[:90] + ("…" if len(notes) > 90 else ""),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── P&L chart (if sells exist) ────────────────────────────────────────
        if sells and realized:
            section("REALIZED P&L PER TRADE")
            sell_rows = [
                {"trade": f"{t['ticker']} {t['ts'][:10]}", "pnl": t["realized_pnl"]}
                for t in sells if t.get("realized_pnl") is not None
            ]
            sdf = pd.DataFrame(sell_rows)
            colors = ["#00E676" if v >= 0 else "#FF3B3B" for v in sdf["pnl"]]
            fig = go.Figure(go.Bar(
                x=sdf["trade"],
                y=sdf["pnl"],
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="#000",
                plot_bgcolor="#000",
                font=dict(family="IBM Plex Mono", color="#FF8000"),
                xaxis=dict(gridcolor="#0d0d0d", tickfont=dict(color="#505050", size=10)),
                yaxis=dict(gridcolor="#0d0d0d", tickfont=dict(color="#505050", size=10), tickprefix="$"),
                margin=dict(l=0, r=0, t=8, b=0),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI REVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif "AI REVIEW" in page:
    page_header("AI PORTFOLIO REVIEW")

    st.markdown(
        '<div style="color:#505050;font-size:12px;line-height:2;margin-bottom:24px;">'
        'The AI agent will analyze market conditions, evaluate existing positions, '
        'research new opportunities, and execute trades autonomously.<br>'
        'This may take 2–5 minutes to complete.'
        '</div>',
        unsafe_allow_html=True,
    )

    if "agent_running" not in st.session_state:
        st.session_state.agent_running = False

    CHECKPOINT_PATH = "data/session_checkpoint.json"
    checkpoint_exists = os.path.exists(CHECKPOINT_PATH)

    # ── Last successful run timestamp ─────────────────────────────────────────
    last_reviews = [r for r in get_reflections(limit=20) if r.get("session_type") == "review"]
    if last_reviews:
        last_ts = last_reviews[0]["created_at"][:16].replace("T", " ") + " UTC"
        last_run_label = f'LAST SUCCESSFUL RUN: {last_ts}'
    else:
        last_run_label = "LAST SUCCESSFUL RUN: NEVER"
    st.markdown(
        f'<div style="color:#505050;font-size:11px;letter-spacing:1px;margin-bottom:8px;">'
        f'{last_run_label}</div>',
        unsafe_allow_html=True,
    )

    if checkpoint_exists:
        st.markdown(
            '<div style="color:#FFA040;font-size:11px;letter-spacing:1px;margin-bottom:12px;">'
            '⚠ INTERRUPTED SESSION FOUND — "RUN AI REVIEW" will resume it. '
            'Use "FRESH RUN" to discard it and start over.'
            '</div>',
            unsafe_allow_html=True,
        )

    col_btn, col_fresh, col_status = st.columns([1, 1, 2])
    with col_btn:
        run_btn = st.button(
            "▶  RUN AI REVIEW",
            disabled=st.session_state.agent_running,
        )
    with col_fresh:
        fresh_btn = st.button(
            "↺  FRESH RUN",
            disabled=st.session_state.agent_running,
        )
    with col_status:
        status_box = st.empty()
        if st.session_state.agent_running:
            status_box.markdown(
                '<span style="color:#FF8000;font-size:12px;letter-spacing:1px;">● AGENT RUNNING — PLEASE WAIT...</span>',
                unsafe_allow_html=True,
            )

    output_area = st.empty()

    if fresh_btn and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    if run_btn or fresh_btn:
        from agent.investment_agent import run_portfolio_review

        st.session_state.agent_running = True
        status_box.markdown(
            '<span style="color:#FF8000;font-size:12px;letter-spacing:1px;">● AGENT RUNNING — PLEASE WAIT...</span>',
            unsafe_allow_html=True,
        )

        session_label = "FRESH RUN" if fresh_btn else "SESSION STARTED"
        lines = [
            '<span style="color:#505050">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>',
            f'<span style="color:#505050">{session_label}  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>',
            '<span style="color:#505050">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>',
            "",
        ]

        def refresh():
            output_area.markdown(agent_box(lines), unsafe_allow_html=True)

        def on_text(text: str):
            safe = html.escape(text)
            lines.append(f'<span style="color:#C8C8C8">{safe}</span>')
            refresh()

        def on_tool_call(name: str, inp: dict):
            ticker = html.escape(str(inp.get("ticker", "")))
            dollar = inp.get("dollar_amount")
            extra = ""
            if ticker:
                extra += f' <span style="color:#888">{ticker}</span>'
            if dollar:
                extra += f' <span style="color:#888">${dollar:,.0f}</span>'
            lines.append(f'<span style="color:#FF8000">  ⚙ {html.escape(name)}</span>{extra}')
            refresh()

        def on_tool_result(name: str, result):
            pass  # Keep output clean; tool calls already shown above

        refresh()

        try:
            run_portfolio_review(
                model=selected_model,
                on_text=on_text,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
            )
            lines.append("")
            lines.append('<span style="color:#00E676">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>')
            lines.append('<span style="color:#00E676">✓ REVIEW COMPLETE — NAVIGATE TO PORTFOLIO TO SEE CHANGES</span>')
            lines.append('<span style="color:#00E676">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>')
        except Exception as e:
            lines.append(f'<span style="color:#FF3B3B">✗ ERROR: {html.escape(str(e))}</span>')

        refresh()
        st.session_state.agent_running = False
        status_box.markdown(
            '<span style="color:#00E676;font-size:12px;letter-spacing:1px;">✓ AGENT COMPLETE</span>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ASK AGENT
# ═══════════════════════════════════════════════════════════════════════════════
elif "ASK AGENT" in page:
    page_header("ASK THE AGENT")

    st.markdown(
        '<div style="color:#505050;font-size:12px;line-height:2;margin-bottom:20px;">'
        'Ask the AI anything about your portfolio, specific stocks, or market conditions.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Quick-prompt buttons ──────────────────────────────────────────────────
    section("QUICK QUERIES")
    examples = [
        "What is the current risk profile of my portfolio?",
        "Analyze NVDA — is it a good buy right now?",
        "Should I add more defensive positions?",
        "Rebalance the portfolio to reduce tech exposure",
        "Which of my holdings has the weakest thesis right now?",
        "What sectors am I missing?",
    ]

    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"quick_{i}"):
            st.session_state["ask_prefill"] = ex

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    section("YOUR QUERY")
    prompt = st.text_area(
        "TYPE YOUR QUESTION",
        value=st.session_state.pop("ask_prefill", ""),
        height=80,
        placeholder='e.g. "Should I buy Apple right now?"',
        key="ask_text",
        label_visibility="collapsed",
    )

    ask_btn = st.button("▶  SEND TO AGENT")
    output_area = st.empty()

    if ask_btn and prompt.strip():
        from agent.investment_agent import run_custom_prompt

        lines = [
            '<span style="color:#505050">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>',
            f'<span style="color:#888">USER › {html.escape(prompt.strip())}</span>',
            '<span style="color:#505050">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>',
            "",
        ]

        def refresh():
            output_area.markdown(agent_box(lines), unsafe_allow_html=True)

        def on_text(text: str):
            lines.append(f'<span style="color:#C8C8C8">{html.escape(text)}</span>')
            refresh()

        def on_tool_call(name: str, inp: dict):
            ticker = html.escape(str(inp.get("ticker", "")))
            extra = f' <span style="color:#888">{ticker}</span>' if ticker else ""
            lines.append(f'<span style="color:#FF8000">  ⚙ {html.escape(name)}</span>{extra}')
            refresh()

        refresh()

        try:
            run_custom_prompt(prompt.strip(), model=selected_model, on_text=on_text, on_tool_call=on_tool_call)
            lines.append("")
            lines.append('<span style="color:#00E676">✓ DONE</span>')
        except Exception as e:
            lines.append(f'<span style="color:#FF3B3B">✗ ERROR: {html.escape(str(e))}</span>')

        refresh()

    elif ask_btn:
        st.warning("Please enter a question before sending.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REFLECTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif "REFLECTIONS" in page:
    page_header("AGENT REFLECTIONS — SESSION LOG")

    from agent.portfolio import get_reflections
    reflections = get_reflections(limit=20)

    if not reflections:
        st.markdown(
            '<div style="color:#505050;text-align:center;padding:60px 0;font-size:13px;">'
            'NO REFLECTIONS YET<br><br>'
            '<span style="font-size:11px;">Run an AI Review — the agent saves a reflection at the end of each session</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="color:#505050;font-size:11px;margin-bottom:16px;">{len(reflections)} SESSION(S) RECORDED</div>',
            unsafe_allow_html=True,
        )
        for i, r in enumerate(reflections):
            date_str = r["created_at"][:16].replace("T", "  ")
            pv = r.get("portfolio_value")
            pv_str = fmt_usd(pv) if pv else "N/A"
            label = f"◈  {date_str}    |    PORTFOLIO: {pv_str}"

            with st.expander(label, expanded=(i == 0)):
                safe_reflection = html.escape(r["reflection"])
                st.markdown(
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                    f'color:#C8C8C8;white-space:pre-wrap;line-height:1.8;padding:8px 0;">'
                    f'{safe_reflection}</div>',
                    unsafe_allow_html=True,
                )
