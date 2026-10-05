"""Componentes visuais reutilizáveis (Streamlit/Plotly) das avaliações qualitativas.

Genéricos: recebem etiquetas e valores, não conhecem o Lançamento — servem as futuras competências.
"""

from __future__ import annotations

from datetime import date
from html import escape
from typing import Optional, Sequence

import plotly.graph_objects as go
import streamlit as st

from basketball_eval import qualitative as q
from basketball_eval.competencies import LEVELS

ACCENT = "#e8590c"
MUTED = "#8a94a3"
GRID = "rgba(128,128,128,.28)"
# Escala ordinal: do tom claro (Inicial) ao intenso (Muito bom).
LEVEL_STYLE = {1: ("#fde7d6", "#6b2a05"), 2: ("#fbc9a0", "#6b2a05"), 3: ("#f6a066", "#3b1500"),
               4: ("#ec7a31", "#ffffff"), 5: ("#b84a0b", "#ffffff")}

CSS = f"""
<style>
.block-container{{padding-top:2.2rem;max-width:1280px}}
.qe-title{{font-size:.8rem;letter-spacing:.14em;color:{ACCENT};font-weight:700;margin:0}}
.qe-h1{{font-size:1.75rem;font-weight:800;line-height:1.15;margin:.1rem 0 .3rem}}
.qe-cycle{{font-size:.8rem;opacity:.7;margin-bottom:.8rem}}
.qe-chip{{display:inline-block;padding:.15rem .55rem;border-radius:999px;font-size:.74rem;font-weight:600;white-space:nowrap}}
.qe-legend{{display:flex;flex-wrap:wrap;gap:.4rem;margin:.2rem 0 1rem}}
.qe-card{{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:1rem 1.2rem;background:rgba(128,128,128,.06)}}
.qe-score{{text-align:center}}
.qe-score .k{{font-size:.75rem;letter-spacing:.12em;opacity:.7;font-weight:700}}
.qe-score .v{{font-size:2.8rem;font-weight:800;color:{ACCENT};line-height:1.1}}
.qe-score .v small{{font-size:1.1rem;opacity:.6;color:inherit;font-weight:600}}
.qe-score .l{{font-size:1.15rem;font-weight:800;letter-spacing:.06em}}
.qe-bar{{height:8px;border-radius:99px;background:rgba(128,128,128,.22);overflow:hidden;margin:.35rem 0}}
.qe-bar>div{{height:100%;background:{ACCENT};border-radius:99px}}
.qe-row{{display:flex;justify-content:space-between;font-size:.85rem;font-weight:600}}
.qe-row span:last-child{{opacity:.75}}
.qe-na{{opacity:.5;font-size:.8rem;white-space:nowrap}}
.qe-stat .k{{font-size:.72rem;letter-spacing:.1em;opacity:.65;font-weight:700;text-transform:uppercase}}
.qe-stat .v{{font-size:1.7rem;font-weight:800}}
.qe-stat .s{{font-size:.8rem;opacity:.65}}
.qe-pos{{color:#1a8a4a}}.qe-neg{{color:#c2410c}}
.qe-banner{{padding:.55rem 1rem;border-radius:10px;font-weight:800;letter-spacing:.06em;text-align:center}}
.st-key-evalrow>div>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{{position:sticky;top:3.5rem;align-self:flex-start}}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{{background:{ACCENT};border-color:{ACCENT};color:#fff}}
.stButton button[kind="primary"]:hover,.stDownloadButton button[kind="primary"]:hover{{background:#c2490a;border-color:#c2490a}}
@media (max-width:1000px){{
  .st-key-evalrow>div>[data-testid="stHorizontalBlock"]{{flex-wrap:wrap}}
  .st-key-evalrow>div>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{{min-width:100%!important}}
  .st-key-evalrow>div>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{{position:static}}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def chip(score: Optional[int]) -> str:
    if score is None:
        return '<span class="qe-na">não avaliado</span>'
    bg, fg = LEVEL_STYLE[score]
    return f'<span class="qe-chip" style="background:{bg};color:{fg}">{escape(q.level_label(score))}</span>'


def legend() -> None:
    """Escala sempre visível: número + descrição."""
    st.markdown('<div class="qe-legend">' + "".join(chip(n) for n in LEVELS) + "</div>", unsafe_allow_html=True)


def score_card(name: str, mean: Optional[float]) -> None:
    if mean is None:
        body = '<div class="v">—</div><div class="qe-na">Avalie pelo menos um critério</div>'
    else:
        body = (f'<div class="v">{q.fmt_score(mean)}<small> / 5</small></div>'
                f'<div class="qe-bar"><div style="width:{mean / 5 * 100:.0f}%"></div></div>'
                f'<div class="l">{escape(q.classify(mean).upper())}</div>')
    st.markdown(f'<div class="qe-card qe-score"><div class="k">{escape(name.upper())}</div>{body}</div>',
                unsafe_allow_html=True)


def progress_rows(rows: Sequence[tuple[str, Optional[float], str]]) -> None:
    """rows = (nome, média, nota) — barras de progresso 0–5."""
    html = ""
    for name, val, note in rows:
        shown = "—" if val is None else q.fmt_score(val, compact=True)
        width = 0 if val is None else val / 5 * 100
        html += (f'<div class="qe-row"><span>{escape(name)}</span><span>{shown} · {escape(note)}</span></div>'
                 f'<div class="qe-bar"><div style="width:{width:.0f}%"></div></div>')
    st.markdown(html, unsafe_allow_html=True)


def stat(label: str, value: str, sub: str = "", cls: str = "") -> None:
    st.markdown(f'<div class="qe-stat"><div class="k">{escape(label)}</div><div class="v {cls}">{escape(value)}</div>'
                f'<div class="s">{escape(sub)}</div></div>', unsafe_allow_html=True)


def trend_banner(label: str, trend: str) -> None:
    bg, fg = {"positiva": ("#dcf5e5", "#14633a"), "negativa": ("#fde3d3", "#9a3412")}.get(trend, ("#e5e7eb", "#374151"))
    st.markdown(f'<div class="qe-banner" style="background:{bg};color:{fg}">{escape(label)}</div>',
                unsafe_allow_html=True)


def bullet_list(items: Sequence[str], empty: str) -> None:
    st.markdown("\n".join(f"- {i}" for i in items) if items else f"*{empty}*")


# ── Gráficos ────────────────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _wrap(label: str, width: int = 12) -> str:
    """Quebra rótulos longos em duas linhas (o radar não faz quebra automática)."""
    if len(label) <= width or " " not in label:
        return label
    cut = min((i for i, ch in enumerate(label) if ch == " "), key=lambda i: abs(i - len(label) / 2))
    return label[:cut] + "<br>" + label[cut + 1:]


def radar_fig(labels: Sequence[str], series: Sequence[tuple[str, Sequence[Optional[float]], str]],
              height: int = 360) -> go.Figure:
    """Radar 1–5. series = (nome, valores por eixo, cor). Eixos não avaliados ficam por desenhar."""
    labels = [_wrap(l) for l in labels]
    fig = go.Figure()
    # Traço invisível com todos os eixos: mantém o radar desenhado mesmo sem valores.
    fig.add_trace(go.Scatterpolar(r=[0] * len(labels), theta=labels, mode="markers", opacity=0,
                                  showlegend=False, hoverinfo="skip"))
    for name, values, color in series:
        pts = [(l, v) for l, v in zip(labels, values) if v is not None]
        if not pts:
            continue
        theta = [p[0] for p in pts] + [pts[0][0]]
        r = [p[1] for p in pts] + [pts[0][1]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta, name=name, fill="toself" if len(pts) >= 3 else None,
            mode="lines+markers", line=dict(color=color, width=3), marker=dict(size=7),
            fillcolor=_rgba(color, 0.18),
            hovertemplate="%{theta}: %{r:.1f}<extra>" + name + "</extra>"))
    fig.update_layout(
        height=height, margin=dict(l=90, r=90, t=25, b=25), showlegend=len(series) > 1,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        polar=dict(radialaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5], gridcolor=GRID, tickfont=dict(size=10),
                                   linecolor=GRID),
                   angularaxis=dict(tickfont=dict(size=13), gridcolor=GRID, rotation=90, direction="clockwise"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _short(d: str, with_year: bool) -> str:
    return date.fromisoformat(d).strftime("%d/%m/%y" if with_year else "%d/%m")


def evolution_fig(series: dict[str, Sequence[tuple[str, float]]], height: int = 320) -> go.Figure:
    """Linhas de evolução (data ISO, valor). A primeira série é a média técnica (destacada)."""
    all_dates = sorted({d for pts in series.values() for d, _ in pts})
    with_year = len({d[:4] for d in all_dates}) > 1
    labels = {d: _short(d, with_year) for d in all_dates}
    fig = go.Figure()
    for i, (name, pts) in enumerate(series.items()):
        main = i == 0
        fig.add_trace(go.Scatter(
            x=[labels[d] for d, _ in pts], y=[v for _, v in pts], name=name, mode="lines+markers" + ("+text" if main else ""),
            text=[q.fmt_score(v) for _, v in pts] if main else None, textposition="top center",
            line=dict(color=ACCENT if main else None, width=4 if main else 2, dash=None if main else "dot"),
            marker=dict(size=9 if main else 6),
            hovertemplate="%{x}: %{y:.1f}<extra>" + name + "</extra>"))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(labels.values()), showgrid=False)
    fig.update_yaxes(range=[0.8, 5.3], tickvals=[1, 2, 3, 4, 5], gridcolor=GRID, title="Escala 1–5")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10), showlegend=len(series) > 1,
                      legend=dict(orientation="h", y=-0.2), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
