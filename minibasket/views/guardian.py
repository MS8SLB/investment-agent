"""Área do encarregado de educação: só o(s) seu(s) educando(s)."""

import streamlit as st

from minibasket import access, charts, db, evolution
from minibasket import competencies as comp
from minibasket.views.common import current_user, fmt_date, is_dark
from minibasket.views.reports import render_parent


def render() -> None:
    user = current_user()
    kids = access.guardian_children(user)
    if not kids:
        st.info("Ainda não há nenhum educando associado à sua conta. Contacte o treinador ou o administrador.")
        return
    child = st.selectbox("Educando", kids, key="guardian_child", format_func=lambda k: k["name"]) if len(kids) > 1 else kids[0]
    evals = access.guardian_evaluations(user, child["id"])
    st.subheader(f"{child['name']} — {child['category'] or ''} · {child['team'] or ''}")
    if not evals:
        st.info("Ainda não há avaliações. Quando o treinador registar a primeira, aparece aqui.")
        return
    t_report, t_evo = st.tabs(["Relatório", "Evolução"])
    by_id = {e["id"]: e for e in evals}
    with t_report:
        eid = st.selectbox("Avaliação", list(by_id)[::-1], key="guardian_eval",
                           format_func=lambda i: f"{fmt_date(by_id[i]['evaluation_date'])} · {by_id[i]['moment']}")
        render_parent(access.parent_report(user, eid))
    with t_evo:
        scale = db.active_scale()
        levels, top = dict(scale["levels"]), max(v for v, _ in scale["levels"])
        st.markdown("#### Evolução geral")
        pts = [{"date": p["date"], "moment": p["moment"], "value": p["average"]} for p in evolution.evolution_series(evals)]
        st.plotly_chart(charts.line_figure(pts, top, "Evolução geral", None, is_dark()), key="guardian_line")
        key = st.selectbox("Competência", list(comp.KEYS), key="guardian_comp", format_func=comp.name_of)
        series = evolution.competency_series(evals, key)
        st.plotly_chart(charts.line_figure([{"date": p["date"], "moment": p["moment"], "value": p["score"]} for p in series],
                                           top, comp.name_of(key), levels, is_dark(), decimals=0), key="guardian_comp_line")
        st.caption("Cada jogador evolui ao seu ritmo.")
