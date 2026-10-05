"""Vista «Evolução da Equipa»: estatísticas por competência e evolução coletiva."""

import pandas as pd
import streamlit as st

from minibasket import calc, charts, db, service, teamstats
from minibasket import competencies as comp
from minibasket.db import CATEGORIES
from minibasket.views.common import fmt_date, is_dark


def pick_team(prefix: str):
    cat = st.radio("Escalão", CATEGORIES, horizontal=True, key=f"{prefix}_cat")
    teams = service.list_teams(category=cat)
    if not teams:
        st.info("Sem equipas neste escalão. Crie-as na secção «Equipas».")
        return None
    return st.selectbox("Equipa", teams, key=f"{prefix}_team",
                        format_func=lambda t: f"{t['name']} ({t['category']}, {t['season']})")


def stats_section(team: dict, snap: dict) -> None:
    scale = db.active_scale()
    top = max(v for v, _ in scale["levels"])
    stats = teamstats.competency_stats(snap)
    st.caption(f"Avaliação mais recente de cada jogador em {fmt_date(snap['as_of'])}: "
               f"{len(snap['evaluated'])} de {snap['members']} jogadores avaliados.")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.bar_figure(stats, top, is_dark()), key="team_bars")
    c2.metric("Média global da equipa", f"{calc.fmt(teamstats.team_average(snap))} / {top}",
              help="Média das médias individuais; cada jogador conta uma vez.")
    c2.dataframe(pd.DataFrame([{
        "Competência": comp.short_of(r["key"]), "Média": calc.fmt(r["mean"]), "Mediana": calc.fmt(r["median"]),
        "Melhor": "—" if r["best"] is None else r["best"], "Mais baixo": "—" if r["lowest"] is None else r["lowest"],
        "Avaliados": r["n"]} for r in stats]), hide_index=True)


def evolution_section(team: dict, line: list[dict]) -> None:
    scale = db.active_scale()
    levels, top = dict(scale["levels"]), max(v for v, _ in scale["levels"])
    pts = [{"date": p["date"], "moment": f"{p['n_evaluated']} jogadores avaliados", "value": p["average"]} for p in line]
    st.caption("Média global da equipa em cada data de avaliação. Pode variar também porque mudam os jogadores "
               "avaliados; a comparação abaixo usa só jogadores avaliados nas duas ocasiões.")
    st.plotly_chart(charts.line_figure(pts, top, "Média da equipa", None, is_dark()), key="team_line")
    if len(line) < 2:
        st.info("A comparação inicial vs. atual surge quando a equipa tiver duas datas de avaliação.")
        return
    by = {p["date"]: p for p in line}
    dates = list(by)
    c1, c2 = st.columns(2)
    d0 = c1.selectbox("Data de referência (inicial)", dates, index=0, key="team_d0", format_func=fmt_date)
    d1 = c2.selectbox("Data a comparar (atual)", dates, index=len(dates) - 1, key="team_d1", format_func=fmt_date)
    if d0 >= d1:
        st.info("Escolha uma data inicial anterior à data atual.")
        return
    ch = teamstats.snapshot_change(by[d0]["snapshot"], by[d1]["snapshot"])
    if ch["n_common"] == 0:
        st.info("Nenhum jogador foi avaliado nas duas datas.")
        return
    st.metric("Evolução da média global", f"{calc.fmt(ch['avg_after'])} / {top}",
              delta=calc.fmt(ch["avg_delta"], signed=True),
              help=f"Média inicial {calc.fmt(ch['avg_before'])}. {ch['n_common']} jogadores avaliados nas duas datas.")
    left, right = st.columns([3, 2])
    series = [{"name": f"Atual · {fmt_date(d1)}", "scores": {r["key"]: r["after"] for r in ch["rows"]}},
              {"name": f"Inicial · {fmt_date(d0)}", "scores": {r["key"]: r["before"] for r in ch["rows"]}}]
    left.plotly_chart(charts.radar_figure(series, top, None, is_dark()), key="team_radar")
    right.dataframe(pd.DataFrame([{
        "Competência": comp.short_of(r["key"]), "Inicial": calc.fmt(r["before"]), "Atual": calc.fmt(r["after"]),
        "Evolução": calc.fmt(r["delta"], signed=True)} for r in ch["rows"]]), hide_index=True)
    right.caption(f"Mesmos {ch['n_common']} jogadores nas duas datas.")


def render() -> None:
    team = pick_team("tevo")
    if not team:
        return
    line = teamstats.timeline(team["id"])
    st.subheader(f"{team['name']} — {team['category']} · {team['club']} · {team['season']}"
                 + (" · Dados de teste" if team["is_demo"] else ""))
    if not line:
        st.info("Esta equipa ainda não tem avaliações.")
        return
    t_stats, t_evo = st.tabs(["Competências da equipa", "Evolução"])
    with t_stats:
        stats_section(team, line[-1]["snapshot"])
    with t_evo:
        evolution_section(team, line)
