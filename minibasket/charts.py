"""Gráficos Plotly: Roda das Competências (radar).

Cores: azul/laranja da paleta de visualização validada (contraste e daltonismo OK em
modo claro e escuro). A identidade não depende só da cor: a referência é tracejada e a
atual tem preenchimento. A escala do radar vai de 0 ao máximo da escala ativa.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import plotly.graph_objects as go

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
