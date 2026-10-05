"""Gráficos Plotly: Roda das Competências (radar).

Cores: azul/laranja da paleta de visualização validada (contraste e daltonismo OK em
modo claro e escuro). A identidade não depende só da cor: a referência é tracejada e a
atual tem preenchimento. A escala do radar vai de 0 ao máximo da escala ativa.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import plotly.graph_objects as go

from . import calc
from . import competencies as comp

# slot 1 / slot 2 da paleta categórica validada (claro, escuro)
COLORS = {"light": ("#2a78d6", "#eb6834"), "dark": ("#3987e5", "#d95926")}


def radar_figure(series: Sequence[dict], scale_max: int = 5, levels: Mapping[int, str] | None = None,
                 dark: bool = False, height: int = 480) -> go.Figure:
    """Roda das Competências.

    series: [{"name": str, "scores": {chave: valor|None}}, ...]. A primeira série é a
    atual/principal (azul, preenchida); a segunda é a referência (laranja, tracejada).
    """
    if not 1 <= len(series) <= 2:
        raise ValueError("A roda mostra uma ou duas avaliações.")
    primary, reference = COLORS["dark" if dark else "light"]
    theta = [c.short for c in comp.COMPETENCIES]
    theta_closed = theta + theta[:1]
    fig = go.Figure()
    # a referência desenha-se primeiro, para ficar por baixo da avaliação atual
    for i, s in sorted(enumerate(series), key=lambda x: -x[0]):
        is_primary = i == 0
        vals = [s["scores"].get(k) for k in comp.KEYS]
        r = vals + vals[:1]
        labels = [(levels or {}).get(v, "Não avaliada") if v is not None else "Não avaliada" for v in r]
        color = primary if is_primary else reference
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta_closed, name=s["name"], mode="lines+markers", connectgaps=False,
            line=dict(color=color, width=2, dash="solid" if is_primary else "dash"),
            marker=dict(color=color, size=8, line=dict(color="rgba(0,0,0,0)", width=0)),
            fill="toself" if is_primary else None,
            fillcolor=_rgba(color, 0.18) if is_primary else None,
            customdata=labels,
            hovertemplate="<b>%{theta}</b><br>%{fullData.name}: %{r}<extra>%{customdata}</extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, scale_max], tickvals=list(range(scale_max + 1)), showline=False,
                            gridcolor="rgba(128,128,128,0.30)", tickfont=dict(size=11)),
            angularaxis=dict(direction="clockwise", rotation=90, gridcolor="rgba(128,128,128,0.30)",
                             linecolor="rgba(128,128,128,0.30)",
                             tickfont=dict(size=13)),
        ),
        showlegend=len(series) >= 2,
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=70, r=70, t=30, b=40), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def line_figure(points: Sequence[dict], scale_max: int = 5, series_name: str = "", levels: Mapping[int, str] | None = None,
                dark: bool = False, height: int = 340, decimals: int = 2) -> go.Figure:
    """Evolução ao longo do tempo (uma série). points: [{"date": ISO, "moment": str, "value": float|None}].

    Eixo vertical de 0 ao máximo da escala. Etiquetas diretas só no primeiro e no último valor;
    valores em falta (competência não avaliada) interrompem a linha, nunca viram 0.
    """
    color = COLORS["dark" if dark else "light"][0]
    xs = [p["date"] for p in points]
    ys = [p["value"] for p in points]
    texts = [calc.fmt(v, decimals) if v is not None else "Não avaliada" for v in ys]
    rated = [i for i, v in enumerate(ys) if v is not None]
    label_idx = {rated[0], rated[-1]} if rated else set()
    shown = [texts[i] if i in label_idx else "" for i in range(len(ys))]
    level = [(levels or {}).get(round(v)) if v is not None and float(v).is_integer() else None for v in ys]
    custom = [[p.get("moment", ""), t, lv or ""] for p, t, lv in zip(points, texts, level)]
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, name=series_name, mode="lines+markers+text", connectgaps=False,
        line=dict(color=color, width=2), marker=dict(color=color, size=8, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=shown, textposition="top center", textfont=dict(size=12),
        customdata=custom,
        hovertemplate="<b>%{x|%d/%m/%Y}</b> · %{customdata[0]}<br>" + (series_name + ": " if series_name else "")
        + "%{customdata[1]}<extra>%{customdata[2]}</extra>"))
    fig.update_layout(
        xaxis=dict(type="date", tickvals=xs, ticktext=[f"{x[8:10]}/{x[5:7]}/{x[2:4]}" for x in xs],
                   gridcolor="rgba(128,128,128,0.20)", tickfont=dict(size=11)),
        yaxis=dict(range=[0, scale_max + 0.4], tickvals=list(range(scale_max + 1)), gridcolor="rgba(128,128,128,0.25)",
                   zeroline=False),
        showlegend=False, margin=dict(l=40, r=20, t=20, b=40), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_figure(stats: Sequence[dict], scale_max: int = 5, dark: bool = False, height: int = 400) -> go.Figure:
    """Média da equipa por competência (barras horizontais, pela ordem da roda, não por valor).

    stats: saída de `teamstats.competency_stats`. A dica mostra mediana, melhor, mais baixo e n.
    """
    color = COLORS["dark" if dark else "light"][0]
    names = [comp.short_of(r["key"]) for r in stats]
    means = [r["mean"] for r in stats]
    custom = [[calc.fmt(r["median"]), r["best"] if r["best"] is not None else "—",
               r["lowest"] if r["lowest"] is not None else "—", r["n"]] for r in stats]
    fig = go.Figure(go.Bar(
        y=names, x=means, orientation="h", marker=dict(color=color), name="Média da equipa",
        text=[calc.fmt(m) if m is not None else "" for m in means], textposition="outside", cliponaxis=False,
        customdata=custom,
        hovertemplate="<b>%{y}</b><br>Média: %{text}<br>Mediana: %{customdata[0]}<br>Melhor: %{customdata[1]}"
                      "<br>Mais baixo: %{customdata[2]}<br>Jogadores avaliados: %{customdata[3]}<extra></extra>"))
    fig.update_layout(
        xaxis=dict(range=[0, scale_max + 0.3], tickvals=list(range(scale_max + 1)), gridcolor="rgba(128,128,128,0.25)"),
        yaxis=dict(autorange="reversed"), bargap=0.45, showlegend=False,
        margin=dict(l=130, r=40, t=20, b=40), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
