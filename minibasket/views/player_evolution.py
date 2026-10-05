"""Vista «Evolução do Jogador»: Roda das Competências (e, nas fases seguintes, histórico e gráficos)."""

import pandas as pd
import streamlit as st

from minibasket import calc, charts, db, evaluations
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


def render() -> None:
    p = pick_player("evo")
    if not p:
        return
    history = evaluations.list_evaluations(p["id"])
    st.subheader(f"{p['name']} — {p['category']} · {p['team']}")
    if not history:
        st.info("Este jogador ainda não tem avaliações. Registe a primeira na secção «Avaliar».")
        return
    st.markdown("### Roda das Competências")
    radar_section(history)
