"""Dashboard do treinador: equipas, jogadores, avaliações, médias e evolução por escalão."""

from fractions import Fraction

import pandas as pd
import streamlit as st

from minibasket import access, calc, charts, db, teamstats
from minibasket.db import CATEGORIES
from minibasket.views.common import current_user, fmt_date, is_dark


def _mean(xs):
    return float(Fraction(sum(Fraction(x).limit_denominator(10**6) for x in xs), len(xs))) if xs else None


def render() -> None:
    scale = db.active_scale()
    top = max(v for v, _ in scale["levels"])
    user = current_user()
    all_teams = access.list_teams(user)
    if not all_teams:
        st.info("Ainda não há equipas associadas à sua conta. Crie-as na secção «Equipas» ou peça ao administrador para as atribuir.")
        _cycle()
        return
    seasons = sorted({t["season"] for t in all_teams}, reverse=True)
    season = st.selectbox("Época", seasons, key="dash_season")
    teams = [t for t in all_teams if t["season"] == season]
    ov = {t["id"]: access.team_overview(user, t["id"]) for t in teams}

    n_eval = sum(o["n_evaluations"] for o in ov.values())
    last = max((o["last_date"] for o in ov.values() if o["last_date"]), default=None)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equipas", len(teams))
    m2.metric("Jogadores", sum(t["n_players"] for t in teams))
    m3.metric("Avaliações realizadas", n_eval)
    m4.metric("Última avaliação", fmt_date(last) if last else "—")

    st.markdown("#### Por escalão")
    cols = st.columns(len(CATEGORIES))
    for col, cat in zip(cols, CATEGORIES):
        ts = [t for t in teams if t["category"] == cat]
        avgs = [a for t in ts for a in ov[t["id"]]["player_averages"]]
        with col:
            st.markdown(f"**{cat}**")
            st.metric("Média global", f"{calc.fmt(_mean(avgs))} / {top}" if avgs else "—",
                      help="Média das médias individuais dos jogadores avaliados deste escalão.")
            st.caption(f"{len(ts)} equipa(s) · {sum(t['n_players'] for t in ts)} jogador(es) · "
                       f"{sum(ov[t['id']]['n_evaluations'] for t in ts)} avaliação(ões)")

    st.markdown("#### Equipas")
    rows = []
    for t in teams:
        o = ov[t["id"]]
        ch = o["change"]
        rows.append({"Equipa": t["name"] + (" (dados de teste)" if t["is_demo"] else ""), "Escalão": t["category"],
                     "Jogadores": t["n_players"], "Avaliados": o["n_evaluated"], "Média global": calc.fmt(o["average"]),
                     "Evolução": calc.fmt(ch["avg_delta"], signed=True) if ch and ch["avg_delta"] is not None else "—",
                     "Última avaliação": fmt_date(o["last_date"]) if o["last_date"] else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Evolução: da primeira à última data de avaliação, com os mesmos jogadores. "
               "Os escalões não se misturam e não há classificações entre equipas.")

    st.markdown("#### Competências e evolução")
    team = st.selectbox("Equipa", teams, key="dash_team",
                        format_func=lambda t: f"{t['name']} ({t['category']})")
    line = access.team_timeline(user, team["id"])
    if not line:
        st.info("Esta equipa ainda não tem avaliações.")
    else:
        c1, c2 = st.columns(2)
        c1.markdown("**Média por competência**")
        c1.plotly_chart(charts.bar_figure(teamstats.competency_stats(line[-1]["snapshot"]), top, is_dark(), height=360),
                        key="dash_bars")
        c2.markdown("**Evolução da média da equipa**")
        c2.plotly_chart(charts.line_figure(
            [{"date": p["date"], "moment": f"{p['n_evaluated']} jogadores avaliados", "value": p["average"]}
             for p in line], top, "Média da equipa", None, is_dark(), height=360), key="dash_line")
    _cycle()


def _cycle() -> None:
    with st.expander("Ciclo pedagógico", expanded=False):
        st.markdown("**Observar** → **Avaliar** → **Identificar necessidades** → **Definir objetivos** → "
                    "**Intervir no treino** → **Reavaliar** → **Verificar a evolução**")
        st.caption("Escala pedagógica de avaliação; não corresponde a normas científicas nem a percentis.")
