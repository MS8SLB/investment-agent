"""Vista «Avaliar»: ficha de avaliação das nove competências."""

from datetime import date

import streamlit as st

from minibasket import calc
from minibasket import competencies as comp
from minibasket import charts, db, evaluations, service
from minibasket.db import CATEGORIES, MOMENTS
from minibasket.views.common import is_dark

NOT_RATED = 0   # opção «—» (não avaliada)


def _fmt_date(iso):
    return date.fromisoformat(iso).strftime("%d/%m/%Y")


def render() -> None:
    scale = db.active_scale()
    levels = dict(scale["levels"])
    st.caption(f"{scale['name']}: " + " · ".join(f"{v} {t}" for v, t in scale["levels"])
               + " — escala pedagógica, sem normas nem percentis.")

    cat = st.radio("Escalão", CATEGORIES, horizontal=True, key="eval_cat")
    players = service.search_players(category=cat)
    if not players:
        st.info("Sem jogadores neste escalão. Crie-os na secção «Jogadores».")
        return
    by_id = {p["id"]: p for p in players}
    pid = st.selectbox("Jogador", list(by_id), format_func=lambda i: by_id[i]["name"], key="eval_player")
    p = by_id[pid]
    history = evaluations.list_evaluations(pid)
    last = history[-1] if history else None

    # Contexto pedagógico: o que ficou definido na avaliação anterior.
    if last:
        with st.expander(f"Avaliação anterior ({_fmt_date(last['evaluation_date'])} · {last['moment']}) — "
                         f"média {calc.fmt(last['average'])}"):
            st.write(" · ".join(f"{comp.short_of(k)}: {v if v is not None else '—'}" for k, v in last["scores"].items()))
            if last["next_objectives"]:
                st.markdown(f"**Objetivos definidos:** {last['next_objectives']}")
            else:
                st.caption("Sem objetivos registados na avaliação anterior.")

    mode_opts = [None] + [e["id"] for e in reversed(history)]
    fix = st.selectbox("Tipo de registo", mode_opts, key=f"eval_mode_{pid}",
                       format_func=lambda i: "Nova avaliação" if i is None else
                       f"Corrigir: {_fmt_date(next(e for e in history if e['id'] == i)['evaluation_date'])} · "
                       f"{next(e for e in history if e['id'] == i)['moment']} (cria nova versão; a anterior fica guardada)")
    base = next((e for e in history if e["id"] == fix), None)
    ctx = f"{pid}_{fix or 'new'}"        # chaves novas ao mudar de jogador/modo → valores por omissão atualizados

    c1, c2, c3 = st.columns(3)
    c1.text_input("Jogador / Escalão / Equipa", f"{p['name']} · {p['category']} · {p['team']}", disabled=True,
                  key=f"ro_{ctx}")
    moment = c2.selectbox("Momento da época", MOMENTS, key=f"mom_{ctx}",
                          index=MOMENTS.index(base["moment"]) if base else 0)
    ev_date = c3.date_input("Data da avaliação (pode ser personalizada)", key=f"dt_{ctx}",
                            value=date.fromisoformat(base["evaluation_date"]) if base else date.today(),
                            max_value=date.today(), format="DD/MM/YYYY")
    coach_name = st.text_input("Treinador", key=f"coach_{ctx}",
                               value=(base or last or {}).get("coach") or "",
                               help="Nome do treinador que realizou a avaliação.")

    st.markdown("### Competências")
    scores, notes = {}, {}
    for c in comp.COMPETENCIES:
        prev = (base or {}).get("scores", {}).get(c.key)
        a, b = st.columns([3, 4])
        v = a.radio(c.name.upper(), [NOT_RATED, 1, 2, 3, 4, 5], horizontal=True, key=f"s_{c.key}_{ctx}",
                    index=prev if prev else 0, format_func=lambda x: "—" if x == NOT_RATED else str(x))
        scores[c.key] = None if v == NOT_RATED else v
        if scores[c.key]:
            a.caption(levels[scores[c.key]])
        notes[c.key] = b.text_area("Observação", key=f"n_{c.key}_{ctx}", height=68,
                                   value=(base or {}).get("notes", {}).get(c.key, ""), label_visibility="collapsed",
                                   placeholder=f"Observação sobre {c.short.lower()} (opcional)")

    avg = calc.global_average(scores)
    st.plotly_chart(charts.radar_figure([{"name": "Esta avaliação", "scores": scores}], max(levels), levels, is_dark(),
                                        height=420))
    st.metric("MÉDIA GLOBAL", f"{calc.fmt(avg)} / {max(levels)}",
              help="Calculada automaticamente a partir das competências classificadas.")
    if calc.missing(scores) and avg is not None:
        st.warning("Avaliação incompleta — a média usa só as competências classificadas: "
                   + ", ".join(comp.short_of(k) for k in calc.missing(scores)) + ".")

    general = st.text_area("Observações gerais do treinador", key=f"gen_{ctx}", value=(base or {}).get("general_notes") or "")
    objectives = st.text_area("Objetivos para o próximo período", key=f"obj_{ctx}",
                              value=(base or {}).get("next_objectives") or "")

    parent_msg = st.text_area("Mensagem para os pais (opcional)", key=f"pm_{ctx}", value=(base or {}).get("parent_message") or "",
                              help="Texto positivo e simples, visível no relatório para os encarregados de educação. "
                                   "As observações e notas acima são internas e não aparecem nesse relatório.")

    if st.button("GUARDAR AVALIAÇÃO", type="primary", key=f"save_{ctx}"):
        try:
            coach_id = evaluations.get_or_create_coach(coach_name) if coach_name.strip() else None
            if base:
                evaluations.correct_evaluation(base["id"], ev_date, moment, scores, notes, coach_id, general, objectives,
                                               parent_message=parent_msg)
            else:
                evaluations.create_evaluation(pid, ev_date, moment, scores, notes, coach_id, general, objectives,
                                              parent_message=parent_msg)
            st.success(f"Avaliação guardada. Média global: {calc.fmt(avg)} / {max(levels)}.")
        except service.ValidationError as e:
            st.error(str(e))
