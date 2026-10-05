"""Vista «Evolução do Jogador»: Roda das Competências (e, nas fases seguintes, histórico e gráficos)."""

import pandas as pd
import streamlit as st

from minibasket import calc, charts, db, evaluations, evolution, teamstats
from minibasket import competencies as comp
from minibasket.views.common import fmt_date, is_dark, pick_player

ARROW = {1: "↑", 0: "=", -1: "↓"}


def _label(e: dict) -> str:
    return f"{fmt_date(e['evaluation_date'])} · {e['moment']} · média {calc.fmt(e['average'])}"


def radar_section(history: list[dict]) -> None:
    """Roda do jogador; com 2+ avaliações permite comparar (por omissão: inicial vs atual)."""
    scale = db.active_scale()
    levels, top = dict(scale["levels"]), max(v for v, _ in scale["levels"])
    by_id = {e["id"]: e for e in history}
    ids = [e["id"] for e in history]

    if len(history) == 1:
        e = history[0]
        st.caption("Só existe uma avaliação; a comparação surge a partir da segunda.")
        series = [{"name": f"{fmt_date(e['evaluation_date'])} · {e['moment']}", "scores": e["scores"]}]
        st.plotly_chart(charts.radar_figure(series, top, levels, is_dark()))
        return

    c1, c2 = st.columns(2)
    ref_id = c1.selectbox("Avaliação de referência (inicial)", ids, index=0, key="radar_ref",
                          format_func=lambda i: _label(by_id[i]))
    cur_id = c2.selectbox("Avaliação a comparar (atual)", ids, index=len(ids) - 1, key="radar_cur",
                          format_func=lambda i: _label(by_id[i]))
    ref, cur = by_id[ref_id], by_id[cur_id]
    if ref_id == cur_id:
        st.info("Escolha duas avaliações diferentes para comparar.")
        series = [{"name": f"{fmt_date(cur['evaluation_date'])} · {cur['moment']}", "scores": cur["scores"]}]
        st.plotly_chart(charts.radar_figure(series, top, levels, is_dark()))
        return

    series = [{"name": f"Atual · {fmt_date(cur['evaluation_date'])}", "scores": cur["scores"]},
              {"name": f"Inicial · {fmt_date(ref['evaluation_date'])}", "scores": ref["scores"]}]
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.radar_figure(series, top, levels, is_dark()))

    cmp_ = calc.compare(ref["scores"], cur["scores"])
    with right:
        st.metric("Média global", f"{calc.fmt(cmp_['avg_after'])} / {top}",
                  delta=calc.fmt(cmp_["avg_delta"], signed=True) if cmp_["avg_delta"] is not None else None,
                  help="Comparada só nas competências avaliadas nas duas ocasiões.")
        if cmp_["n_common"] < len(comp.COMPETENCIES):
            st.caption(f"Comparação feita em {cmp_['n_common']} de {len(comp.COMPETENCIES)} competências "
                       "(avaliação incompleta).")
        rows = [{"Competência": comp.short_of(r["key"]),
                 "Inicial": r["before"], "Atual": r["after"],
                 "Evolução": "—" if r["delta"] is None else
                 f"{ARROW[(r['delta'] > 0) - (r['delta'] < 0)]} {calc.fmt(r['delta'], 0, signed=True) if r['delta'] else '0'}"}
                for r in cmp_["rows"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    grew = sorted((r for r in cmp_["rows"] if r["delta"] and r["delta"] > 0), key=lambda r: -r["delta"])
    if grew:
        st.success("Onde houve evolução: " + ", ".join(f"{comp.short_of(r['key'])} "
                                                       f"({calc.fmt(r['delta'], 0, signed=True)})" for r in grew) + ".")
    else:
        st.info("Sem aumentos de classificação neste intervalo; é um bom momento para rever os objetivos de treino.")


def average_section(history: list[dict]) -> None:
    scale = db.active_scale()
    pts = [{"date": p["date"], "moment": p["moment"], "value": p["average"]}
           for p in evolution.evolution_series(history)]
    st.caption("Média global de cada avaliação (calculada automaticamente).")
    st.plotly_chart(charts.line_figure(pts, max(v for v, _ in scale["levels"]), "Média global", None, is_dark()))
    if len(pts) == 1:
        st.caption("Com a segunda avaliação passa a ser possível acompanhar a evolução.")


def competency_section(history: list[dict]) -> None:
    scale = db.active_scale()
    levels, top = dict(scale["levels"]), max(v for v, _ in scale["levels"])
    options = list(comp.KEYS) + ["all"]
    key = st.selectbox("Competência", options, key="evo_comp",
                       format_func=lambda k: "Todas as competências (resumo)" if k == "all" else comp.name_of(k))
    if key != "all":
        series = evolution.competency_series(history, key)
        pts = [{"date": p["date"], "moment": p["moment"], "value": p["score"]} for p in series]
        st.plotly_chart(charts.line_figure(pts, top, comp.name_of(key), levels, is_dark(), decimals=0))
        rated = [p for p in series if p["score"] is not None]
        if len(rated) >= 2:
            st.write(f"**{comp.name_of(key)}:** de {rated[0]['score']} ({levels[rated[0]['score']]}) para "
                     f"{rated[-1]['score']} ({levels[rated[-1]['score']]}).")
        elif not rated:
            st.caption("Esta competência ainda não foi avaliada.")
        return
    cols = st.columns(3)
    for i, c in enumerate(comp.COMPETENCIES):
        series = evolution.competency_series(history, c.key)
        pts = [{"date": p["date"], "moment": p["moment"], "value": p["score"]} for p in series]
        with cols[i % 3]:
            st.markdown(f"**{c.name}**")
            st.plotly_chart(charts.line_figure(pts, top, c.name, levels, is_dark(), height=230, decimals=0),
                            key=f"mini_{c.key}")


def team_section(history: list[dict]) -> None:
    scale = db.active_scale()
    levels, top = dict(scale["levels"]), max(v for v, _ in scale["levels"])
    st.caption("Serve para identificar necessidades de desenvolvimento do jogador. Não é uma classificação "
               "nem um ranking.")
    by_id = {e["id"]: e for e in history}
    ids = [e["id"] for e in history]
    eid = st.selectbox("Avaliação do jogador", ids, index=len(ids) - 1, key="vs_eval",
                       format_func=lambda i: _label(by_id[i]))
    e = by_id[eid]
    snap = teamstats.snapshot(e["team_id"], e["evaluation_date"])
    if not teamstats.can_compare(snap):
        st.info(f"A comparação com a equipa só aparece quando houver pelo menos {teamstats.MIN_TEAM_FOR_COMPARISON} "
                f"jogadores avaliados (agora: {len(snap['evaluated'])}).")
        return
    stats = teamstats.competency_stats(snap)
    rows = teamstats.compare_to_team(e["scores"], stats)
    series = [{"name": "Jogador", "scores": e["scores"]},
              {"name": "Média da equipa", "scores": {s["key"]: s["mean"] for s in stats}}]
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.radar_figure(series, top, levels, is_dark()), key="vs_radar")
    right.caption(f"Equipa {e['team']} ({e['category']}) em {fmt_date(snap['as_of'])}: "
                  f"{len(snap['evaluated'])} jogadores avaliados.")
    right.dataframe(pd.DataFrame([{
        "Competência": comp.short_of(r["key"]), "Jogador": r["player"],
        "Média equipa": calc.fmt(r["team_mean"], 1), "Diferença": calc.fmt(r["diff"], 1, signed=True)}
        for r in rows]), hide_index=True)


def history_section(pid: int, history: list[dict]) -> None:
    """Histórico cronológico: nenhuma avaliação é apagada nem substituída."""
    points = evolution.evolution_series(history)
    rows = [{"Data": fmt_date(p["date"]), "Momento": p["moment"], "Escalão": p["category"],
             "Média": calc.fmt(p["average"]) + ("" if p["complete"] else " (incompleta)"),
             "Variação": "—" if p["delta"] is None else calc.fmt(p["delta"], signed=True)}
            for p in points]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    ch = evolution.overall_change(history)
    if ch:
        st.write(f"**Do início ao momento atual:** média {calc.fmt(ch['avg_first'])} → {calc.fmt(ch['avg_last'])} "
                 f"({calc.fmt(ch['delta'], signed=True)}) em {ch['n_evaluations']} avaliações.")
    else:
        st.caption("Com a segunda avaliação passa a ser possível acompanhar a evolução.")

    st.markdown("#### Evolução por competência")
    cols = [fmt_date(p["date"]) for p in points]
    table = []
    for r in evolution.competency_table(history):
        row = {"Competência": comp.short_of(r["key"])}
        row.update({c: ("—" if s is None else s) for c, s in zip(cols, r["scores"])})
        row["Variação total"] = "—" if r["change"] is None else calc.fmt(r["change"], 0, signed=True) if r["change"] else "0"
        table.append(row)
    st.dataframe(pd.DataFrame(table), hide_index=True)

    st.markdown("#### Detalhe das avaliações")
    for e in reversed(history):
        with st.expander(f"{fmt_date(e['evaluation_date'])} · {e['moment']} · média {calc.fmt(e['average'])}"
                         + ("" if e["complete"] else " (incompleta)")):
            st.caption(f"{e['category']} · {e['team']} · Treinador: {e['coach'] or '—'}")
            for c in comp.COMPETENCIES:
                v, n = e["scores"].get(c.key), e["notes"].get(c.key)
                st.write(f"**{c.name}:** {v if v is not None else '—'}" + (f" — {n}" if n else ""))
            if e["general_notes"]:
                st.markdown(f"**Observações do treinador:** {e['general_notes']}")
            if e["next_objectives"]:
                st.markdown(f"**Objetivos para o período seguinte:** {e['next_objectives']}")
    versions = [e for e in evaluations.list_evaluations(pid, include_superseded=True) if e["superseded_by"]]
    if versions:
        with st.expander(f"Versões anteriores corrigidas ({len(versions)})"):
            st.caption("Ficam guardadas para auditoria e não entram na evolução.")
            for e in versions:
                st.write(f"{fmt_date(e['evaluation_date'])} · {e['moment']} · média {calc.fmt(e['average'])} "
                         f"(registada em {e['created_at'][:16]})")


def render() -> None:
    p = pick_player("evo")
    if not p:
        return
    history = evaluations.list_evaluations(p["id"])
    st.subheader(f"{p['name']} — {p['category']} · {p['team']}")
    if not history:
        st.info("Este jogador ainda não tem avaliações. Registe a primeira na secção «Avaliar».")
        return
    t_wheel, t_avg, t_comp, t_team, t_hist = st.tabs(
        ["Roda das Competências", "Evolução da média", "Evolução por competência", "Jogador vs. equipa",
         "Histórico"])
    with t_wheel:
        radar_section(history)
    with t_avg:
        average_section(history)
    with t_comp:
        competency_section(history)
    with t_team:
        team_section(history)
    with t_hist:
        history_section(p["id"], history)
