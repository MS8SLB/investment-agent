"""Vista «Relatórios»: relatório individual e da equipa para o treinador."""

import pandas as pd
import streamlit as st

from minibasket import calc, charts, db, evaluations, reports
from minibasket.views.common import fmt_date, is_dark, pick_player
from minibasket.views.team_evolution import pick_team

ARROW = {1: "↑", 0: "=", -1: "↓"}


def _area_line(a: dict) -> str:
    return f"**{a['name']}** — {a['score']}" + (f" ({a['level']})" if a["level"] else "")


def render_individual(r: dict) -> None:
    top = r["scale_max"]
    st.markdown("## Relatório de avaliação individual")
    if r["is_demo"]:
        st.caption("Dados de teste")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Jogador:** {r['player']['name']}  \n**Escalão:** {r['category']}  \n**Equipa:** {r['team']} ({r['club']})")
    c2.markdown(f"**Data:** {fmt_date(r['date'])} · {r['moment']}  \n**Treinador:** {r['coach'] or '—'}")
    if not r["complete"]:
        st.warning("Avaliação incompleta — competências por avaliar: " + ", ".join(r["missing"]) + ".")

    st.markdown("### Roda das Competências")
    st.plotly_chart(charts.radar_figure([{"name": fmt_date(r["date"]), "scores": r["scores"]}], top, r["levels"],
                                        is_dark(), height=420), key=f"rep_radar_{r['evaluation_id']}")
    st.markdown("### Resultados")
    st.dataframe(pd.DataFrame([{"Competência": x["name"], "Classificação": "—" if x["score"] is None else f"{x['score']}/{top}",
                                "Nível": x["level"] or "—", "Observação": x["note"] or ""} for x in r["results"]]),
                 hide_index=True)
    st.metric("MÉDIA GLOBAL", f"{calc.fmt(r['average'])}/{top}")

    st.markdown("### Evolução")
    ev_ = r["evolution"]
    if not ev_:
        st.caption("Primeira avaliação: ainda não há avaliação anterior para comparar.")
    else:
        st.write(f"Comparação com a avaliação anterior ({fmt_date(ev_['previous_date'])} · {ev_['previous_moment']}): "
                 f"média {calc.fmt(ev_['previous_average'])} → {calc.fmt(r['average'])}"
                 + (f" ({calc.fmt(ev_['avg_delta'], signed=True)})" if ev_["avg_delta"] is not None else "") + ".")
        if ev_["improved"]:
            st.write("**Evolução em:** " + ", ".join(ev_["improved"]) + ".")
        if ev_["maintained"]:
            st.write("**Mantido em:** " + ", ".join(ev_["maintained"]) + ".")
        if ev_["to_consolidate"]:
            st.write("**A consolidar:** " + ", ".join(ev_["to_consolidate"]) + ".")
        if ev_["previous_objectives"]:
            st.info(f"**Objetivos definidos na avaliação anterior:** {ev_['previous_objectives']}")

    st.markdown("### Áreas fortes")
    if r["balanced"]:
        st.write("Perfil equilibrado: todas as competências avaliadas têm a mesma classificação.")
    for i, a in enumerate(r["strengths"], 1):
        st.write(f"{i}. {_area_line(a)}")
    if r["tie_strengths"]:
        st.caption("Há outras competências com a mesma classificação.")
    st.markdown("### Áreas a desenvolver")
    for i, a in enumerate(r["to_develop"], 1):
        st.write(f"{i}. {_area_line(a)}")
    if r["tie_to_develop"]:
        st.caption("Há outras competências com a mesma classificação.")

    st.markdown("### Observações do treinador")
    st.write(r["general_notes"] or "Sem observações gerais registadas.")
    st.markdown("### Objetivos para o próximo período")
    if r["objectives"]:
        st.write(r["objectives"])
    else:
        st.caption("Ainda não definidos."
                   + (" Sugestão de foco: " + ", ".join(r["suggested_focus"]) + "." if r["suggested_focus"] else ""))


def render_team(r: dict) -> None:
    t = r["team"]
    st.markdown("## Relatório da equipa")
    st.markdown(f"**{t['name']}** — {t['category']} · {t['club']} · {t['season']}" + (" · *Dados de teste*" if t["is_demo"] else ""))
    if not r["has_data"]:
        st.info("Esta equipa ainda não tem avaliações.")
        return
    top = max(v for v, _ in db.active_scale()["levels"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Jogadores avaliados", f"{r['n_evaluated']} de {r['members']}")
    m2.metric("Média global da equipa", f"{calc.fmt(r['average'])} / {top}")
    ev_ = r["evolution"]
    if ev_ and ev_["avg_delta"] is not None:
        m3.metric("Evolução da média", f"{calc.fmt(ev_['avg_after'])} / {top}", delta=calc.fmt(ev_["avg_delta"], signed=True),
                  help=f"Inicial {calc.fmt(ev_['avg_before'])}; {ev_['n_common']} jogadores avaliados nas duas datas.")
    else:
        m3.metric("Evolução da média", "—")
    st.caption(f"Avaliação mais recente de cada jogador em {fmt_date(r['as_of'])}.")

    st.markdown("### Média de cada competência")
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.bar_figure(r["stats"], top, is_dark(), height=380), key=f"rep_team_bars_{t['id']}")
    right.dataframe(pd.DataFrame([{"Competência": s["name"], "Média": calc.fmt(s["mean"]), "Mediana": calc.fmt(s["median"]),
                                   "Melhor": "—" if s["best"] is None else s["best"],
                                   "Mais baixo": "—" if s["lowest"] is None else s["lowest"], "Avaliados": s["n"]}
                                  for s in r["stats"]]), hide_index=True)

    st.markdown("### Evolução da equipa")
    if not ev_:
        st.caption("A evolução surge quando a equipa tiver duas datas de avaliação.")
    elif ev_["n_common"] == 0:
        st.caption("Nenhum jogador foi reavaliado entre as duas datas.")
    else:
        st.write(f"De {fmt_date(ev_['date_before'])} a {fmt_date(ev_['date_after'])}, com os {ev_['n_common']} jogadores "
                 f"reavaliados: média {calc.fmt(ev_['avg_before'])} → {calc.fmt(ev_['avg_after'])} "
                 f"({calc.fmt(ev_['avg_delta'], signed=True)}).")
        if ev_["homogeneous"]:
            st.write("A evolução foi semelhante em todas as competências.")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Maior evolução**")
            for a in ev_["most_improved"]:
                c1.write(f"{a['name']} {calc.fmt(a['delta'], signed=True)}")
            c2.markdown("**Menor evolução**")
            for a in ev_["least_improved"]:
                c2.write(f"{a['name']} {calc.fmt(a['delta'], signed=True)}")
        st.dataframe(pd.DataFrame([{"Competência": x["name"], "Inicial": calc.fmt(x["before"]), "Atual": calc.fmt(x["after"]),
                                    "Evolução": calc.fmt(x["delta"], signed=True)} for x in ev_["rows"]]), hide_index=True)

    st.markdown("### Competências que necessitam de maior atenção")
    if r["balanced_means"]:
        st.write("As médias são idênticas em todas as competências.")
    for i, a in enumerate(r["attention"], 1):
        st.write(f"{i}. **{a['name']}** — média {calc.fmt(a['mean'])}")
    if r["strong"]:
        st.markdown("### Competências em que a equipa está mais forte")
        for i, a in enumerate(r["strong"], 1):
            st.write(f"{i}. **{a['name']}** — média {calc.fmt(a['mean'])}")


def render_parent(r: dict) -> None:
    """Pré-visualização do relatório que o encarregado de educação vai ver."""
    w, sec = r["wheel"], r["sections"]
    st.markdown(f"## {r['player']['first_name']} — como está a correr")
    if r["is_demo"]:
        st.caption("Dados de teste")
    st.caption(f"{r['player']['name']} · {r['category']} · {r['team']} ({r['club']}) · "
               f"{fmt_date(r['date'])} · {r['moment']}" + (f" · Treinador: {r['coach']}" if r["coach"] else ""))
    if r["incomplete_note"]:
        st.caption(r["incomplete_note"])

    st.markdown(f"### {sec['evolution']['title']}")
    st.write(sec["evolution"]["text"])
    series = [{"name": "Esta avaliação", "scores": w["scores"]}]
    if w["previous_scores"]:
        series.append({"name": f"Avaliação anterior · {fmt_date(w['previous_date'])}", "scores": w["previous_scores"]})
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.radar_figure(series, w["scale_max"], w["levels"], is_dark(), height=400),
                      key=f"parent_radar_{r['evaluation_id']}")
    with right:
        for k in r["skills"]:
            st.write(f"**{k['name']}**  \n{k['dots']}  {k['level'] or ''}")

    st.markdown(f"### {sec['strengths']['title']}")
    st.write(sec["strengths"]["text"])
    st.markdown(f"### {sec['working']['title']}")
    st.write(sec["working"]["text"])
    st.markdown(f"### {sec['goals']['title']}")
    st.write(sec["goals"]["text"])
    if r["message"]:
        st.info(f"**Mensagem do treinador:** {r['message']}")
    st.caption("Cada jogador evolui ao seu ritmo. Obrigado pelo acompanhamento e apoio.")


def render() -> None:
    t_ind, t_team, t_parent = st.tabs(["Treinador · Individual", "Treinador · Equipa", "Pais"])
    with t_ind:
        p = pick_player("rep")
        if p:
            history = evaluations.list_evaluations(p["id"])
            if not history:
                st.info("Este jogador ainda não tem avaliações.")
            else:
                by_id = {e["id"]: e for e in history}
                eid = st.selectbox("Avaliação", list(by_id)[::-1], key="rep_eval",
                                   format_func=lambda i: f"{fmt_date(by_id[i]['evaluation_date'])} · {by_id[i]['moment']}"
                                                          f" · média {calc.fmt(by_id[i]['average'])}")
                render_individual(reports.individual_report(eid))
    with t_team:
        team = pick_team("rep_t")
        if team:
            render_team(reports.team_report(team["id"]))
    with t_parent:
        st.caption("Pré-visualização do que o encarregado de educação vê: só o próprio jogador, sem comparações com a "
                   "equipa e sem as notas internas do treinador.")
        p = pick_player("rep_p")
        if p:
            history = evaluations.list_evaluations(p["id"])
            if not history:
                st.info("Este jogador ainda não tem avaliações.")
            else:
                by_id = {e["id"]: e for e in history}
                eid = st.selectbox("Avaliação", list(by_id)[::-1], key="rep_p_eval",
                                   format_func=lambda i: f"{fmt_date(by_id[i]['evaluation_date'])} · {by_id[i]['moment']}")
                render_parent(reports.parent_report(eid))
