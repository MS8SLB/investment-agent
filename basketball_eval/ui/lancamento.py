"""Avaliação → Técnica Individual → Lançamento (avaliação qualitativa 1–5).

Executar pela navegação: streamlit run basketball_eval/home.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from basketball_eval import competencies, qual_service as svc, qualitative as q, report
from basketball_eval import service as base
from basketball_eval.competencies import LEVELS
from basketball_eval.ui import components as ui

COMP = competencies.get("lancamento")
CKEY = COMP.key
GREY = "#8a94a3"
VIEWS = ["Avaliar", "Perfil", "Evolução", "Comparar", "Histórico", "Relatório"]
SC = {c.key: f"sc_{c.key}" for c in COMP.criteria}      # chave do widget de cada critério

st.set_page_config(page_title="Lançamento · Avaliação qualitativa", page_icon="🏀", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()


# ── Estado do formulário ────────────────────────────────────────────────────
def reset_form():
    for k in SC.values():
        st.session_state[k] = None
    st.session_state.update(f_obs="", edit_id=None)


def _init_state():
    st.session_state.setdefault("view", "Avaliar")
    st.session_state.setdefault("f_date", date.today())
    st.session_state.setdefault("f_coach", "")
    st.session_state.setdefault("flash", None)
    if "edit_id" not in st.session_state:
        reset_form()


def current_scores() -> dict:
    return {c: st.session_state.get(k) for c, k in SC.items()}


def load_for_edit(evaluation_id: int):
    ev = svc.get_evaluation(evaluation_id)
    for c, k in SC.items():
        st.session_state[k] = ev["scores"].get(c)
    st.session_state.update(f_obs=ev["observations"] or "", f_date=date.fromisoformat(ev["evaluation_date"]),
                            f_coach=ev["coach_name"] or "", edit_id=evaluation_id, view="Avaliar")


def save(player_id: int, team):
    try:
        args = (st.session_state.f_date, current_scores(), st.session_state.f_obs)
        if st.session_state.edit_id:
            svc.update_evaluation(st.session_state.edit_id, *args, coach=st.session_state.f_coach)
            msg = "Avaliação atualizada."
        else:
            svc.save_evaluation(CKEY, player_id, *args, team=team, age_group=st.session_state.ctx_age,
                                coach=st.session_state.f_coach)
            msg = "Avaliação guardada."
        reset_form()
        st.session_state.flash = ("success", msg)
    except ValueError as e:
        st.session_state.flash = ("error", str(e))


def add_player_cb():
    s = st.session_state
    try:
        pid = base.add_player(s.np_name, s.np_age, team=(s.np_team or "").strip() or None)
        if (s.np_team or "").strip():
            svc.get_or_create_team(s.np_team)
        s.update(ctx_age=s.np_age, ctx_team=(s.np_team or "").strip() or None, ctx_player=pid, np_name="",
                 flash=("success", "Jogador adicionado."))
    except ValueError as e:
        s.flash = ("error", str(e))


_init_state()

# ── Cabeçalho ───────────────────────────────────────────────────────────────
st.markdown('<p class="qe-title">AVALIAÇÃO QUALITATIVA</p>'
            '<div class="qe-h1">TÉCNICA INDIVIDUAL – LANÇAMENTO</div>'
            '<div class="qe-cycle">Avaliar → Identificar → Acompanhar → Intervir → Reavaliar</div>',
            unsafe_allow_html=True)

if st.session_state.flash:
    kind, text = st.session_state.flash
    getattr(st, kind)(text)
    st.session_state.flash = None

# ── Contexto: equipa · escalão · jogador ────────────────────────────────────
ages = svc.list_age_groups()
teams = svc.list_teams()
if "ctx_age" not in st.session_state:               # abre no primeiro escalão com jogadores
    st.session_state.ctx_age = next((a for a in ages if svc.players_for(a)), ages[0])
c1, c2, c3 = st.columns([2, 1.2, 2.4])
team = c1.selectbox("Equipa", [None] + teams, key="ctx_team", format_func=lambda t: "Todas as equipas" if t is None else t)
age = c2.selectbox("Escalão", ages, key="ctx_age")
players = svc.players_for(age, team)
by_id = {p["id"]: p for p in players}
pid = c3.selectbox("Jogador", list(by_id), key="ctx_player", format_func=lambda i: by_id[i]["name"],
                   placeholder="Sem jogadores neste escalão/equipa") if players else None
if not players:
    c3.selectbox("Jogador", [], disabled=True, placeholder="Sem jogadores neste escalão/equipa")

with st.expander("➕ Adicionar jogador", expanded=not players):
    with st.form("new_player", border=False):
        f1, f2, f3 = st.columns([2, 1.5, 1])
        f1.text_input("Nome do jogador", key="np_name")
        f2.text_input("Equipa", key="np_team")
        f3.selectbox("Escalão", ages, key="np_age")
        st.form_submit_button("Adicionar", on_click=add_player_cb)

if pid is None:
    st.info("Selecione ou adicione um jogador para começar.")
    st.stop()

player = by_id[pid]
if st.session_state.get("form_owner") != pid:       # outro jogador → formulário limpo
    reset_form()
    st.session_state.form_owner = pid

evs = svc.list_evaluations(pid, CKEY)
view = st.segmented_control("Secção", VIEWS, key="view", label_visibility="collapsed") or "Avaliar"
st.divider()


def eval_label(e):
    return f"{report.fmt_date(e['evaluation_date'])} · {q.fmt_score(e['mean'])} – {e['label']}"


def dim_names():
    return [d.name for d in COMP.dimensions]


def dim_values(e):
    return [e["dimension_means"][d.key] for d in COMP.dimensions]


def need_evals(minimum=1):
    if len(evs) < minimum:
        st.info("Ainda não há avaliações guardadas para este jogador." if minimum == 1
                else "São necessárias pelo menos duas avaliações.")
        return False
    return True


def insight_columns(scores):
    ins = q.insights(COMP, scores)
    a, b = st.columns(2)
    with a:
        st.markdown("##### 💪 Pontos fortes")
        ui.bullet_list([i.text for i in ins.strengths], "Nenhum critério avaliado com 4 ou 5.")
    with b:
        st.markdown("##### 🎯 Áreas de melhoria")
        ui.bullet_list([i.text for i in ins.improvements], "Nenhum critério avaliado com 1 ou 2.")


# ── Avaliar ─────────────────────────────────────────────────────────────────
if view == "Avaliar":
    editing = st.session_state.edit_id
    if editing:
        st.warning("A editar uma avaliação existente. Ao guardar, os dados anteriores são substituídos.")
    ui.legend()
    h1, h2, h3 = st.columns([1, 1, 1.4])
    h1.date_input("Data", key="f_date", format="DD/MM/YYYY")
    h2.text_input("Treinador", key="f_coach")
    h3.markdown(f"**Jogador:** {player['name']}  \n**Equipa:** {player['team'] or '—'} · **Escalão:** {age}")

    evalrow = st.container(key="evalrow")
    left, right = evalrow.columns([7, 4], gap="large")
    with left:
        for d in COMP.dimensions:
            with st.container(border=True):
                st.markdown(f"**{d.title}**")
                for c in d.criteria:
                    a, b, cc = st.columns([3, 4, 2.6], vertical_alignment="center")
                    a.markdown(c.label)
                    b.segmented_control(c.label, [1, 2, 3, 4, 5], key=SC[c.key], label_visibility="collapsed")
                    cc.markdown(ui.chip(st.session_state.get(SC[c.key])), unsafe_allow_html=True)
        st.text_area("Observações do treinador", key="f_obs", height=120,
                     placeholder="Notas livres sobre o gesto, o contexto, o que trabalhar a seguir…")
        st.caption("Clique de novo numa pontuação para a retirar (critério não avaliado).")
        st.button("Guardar alterações" if editing else "Guardar avaliação", type="primary", on_click=save,
                  args=(pid, team if team else None))
        if editing:
            st.button("Cancelar edição", on_click=reset_form)

    scores = q.validate_scores(COMP, current_scores())
    s = q.summarize(COMP, scores)
    with right:
        with st.container(key="live"):
            ui.score_card(COMP.name, s.mean)
            st.plotly_chart(ui.radar_fig(dim_names(), [("Atual", [s.dimension_means[d.key] for d in COMP.dimensions],
                                                        ui.ACCENT)], 300), width="stretch", key="live_radar")
            ui.progress_rows([(d.name, s.dimension_means[d.key], "{}/{} avaliados".format(*s.assessed[d.key]))
                              for d in COMP.dimensions])
    if scores:
        st.divider()
        insight_columns(scores)

# ── Perfil ──────────────────────────────────────────────────────────────────
elif view == "Perfil":
    st.markdown(f"### {player['name']}")
    st.caption(f"Equipa: {player['team'] or '—'} · Escalão: {player['category']}")
    if need_evals():
        last = evs[-1]
        a, b = st.columns([1, 1.4], gap="large")
        with a:
            st.caption(f"Última avaliação de Lançamento · {report.fmt_date(last['evaluation_date'])}")
            ui.score_card(COMP.name, last["mean"])
            st.markdown("")
            st.plotly_chart(ui.radar_fig(dim_names(), [("Última", dim_values(last), ui.ACCENT)], 320), width="stretch")
        with b:
            st.markdown("##### Evolução")
            st.plotly_chart(ui.evolution_fig({"Média": [(e["evaluation_date"], e["mean"]) for e in evs if e["mean"]]}),
                            width="stretch")
        insight_columns(last["scores"])
        st.markdown("##### Avaliações anteriores e observações")
        for e in reversed(evs):
            with st.expander(f"{eval_label(e)} · {e['coach_name'] or 'sem treinador'}", expanded=e is last):
                st.write(e["observations"] or "*Sem observações.*")

# ── Evolução ────────────────────────────────────────────────────────────────
elif view == "Evolução":
    st.markdown("### Evolução do Lançamento")
    if need_evals():
        evo = q.evolution([(e["evaluation_date"], e["mean"]) for e in evs if e["mean"] is not None])
        extra = st.multiselect("Mostrar também", dim_names(), placeholder="Dimensões")
        series = {"Média técnica": [(e["evaluation_date"], e["mean"]) for e in evs if e["mean"] is not None]}
        for d in COMP.dimensions:
            if d.name in extra:
                series[d.name] = [(e["evaluation_date"], e["dimension_means"][d.key]) for e in evs
                                  if e["dimension_means"][d.key] is not None]
        st.plotly_chart(ui.evolution_fig(series, 360), width="stretch")
        st.dataframe(pd.DataFrame({"Data": [report.fmt_date(e["evaluation_date"]) for e in evs],
                                   "Média": [q.fmt_score(e["mean"]) for e in evs],
                                   "Classificação": [e["label"] or "—" for e in evs]}),
                     hide_index=True, width="stretch")
        if evo.n < 2:
            st.info("Primeira avaliação guardada. A evolução aparece a partir da segunda.")
        else:
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                ui.stat("Primeira avaliação", q.fmt_score(evo.first), report.fmt_date(evo.first_date))
            with s2:
                ui.stat("Última avaliação", q.fmt_score(evo.last), report.fmt_date(evo.last_date))
            with s3:
                ui.stat("Evolução", f"{q.fmt_score(evo.change, signed=True)} pontos", f"{q.fmt_score(evo.first)} → {q.fmt_score(evo.last)}",
                        "qe-pos" if evo.trend == q.POSITIVE else "qe-neg" if evo.trend == q.NEGATIVE else "")
            with s4:
                ui.stat("Variação percentual", "—" if evo.change_pct is None else f"{q.fmt_score(evo.change_pct, signed=True)}%",
                        "indicativa; escala qualitativa")
            st.markdown("")
            ui.trend_banner(evo.trend_label, evo.trend)

# ── Comparar ────────────────────────────────────────────────────────────────
elif view == "Comparar":
    st.markdown("### Comparar avaliações")
    if need_evals(2):
        ids = [e["id"] for e in evs]
        lab = {e["id"]: eval_label(e) for e in evs}
        a, b = st.columns(2)
        ida = a.selectbox("Avaliação inicial", ids, index=0, format_func=lab.get)
        idb = b.selectbox("Avaliação atual", ids, index=len(ids) - 1, format_func=lab.get)
        if ida == idb:
            st.info("Escolha duas avaliações diferentes.")
        else:
            ev_a, ev_b, cmp = svc.compare_evaluations(ida, idb)
            st.caption(f"{report.fmt_date(ev_a['evaluation_date'])} (inicial) → {report.fmt_date(ev_b['evaluation_date'])} (atual)")
            t, r = st.columns([1.2, 1], gap="large")
            with t:
                rows = [(x.label, x.before, x.after, x.delta) for x in cmp.dimensions]
                st.dataframe(pd.DataFrame(
                    {"Dimensão": [x[0] for x in rows],
                     "Comparação": [f"{q.fmt_score(x[1], True)} → {q.fmt_score(x[2], True)}" for x in rows],
                     "Variação": [q.fmt_score(x[3], signed=True) if x[3] is not None else "—" for x in rows]}),
                    hide_index=True, width="stretch")
                st.markdown(f"**Média técnica: {q.fmt_score(cmp.overall.before)} → {q.fmt_score(cmp.overall.after)}**")
                if cmp.trend:
                    ui.trend_banner(cmp.trend_label, cmp.trend)
                st.caption("Comparação descritiva entre dois momentos; não indica causas.")
            with r:
                st.plotly_chart(ui.radar_fig(dim_names(), [
                    ("Inicial", dim_values(ev_a), GREY), ("Atual", dim_values(ev_b), ui.ACCENT)], 340), width="stretch")
            with st.expander("Detalhe por critério"):
                st.dataframe(pd.DataFrame([{"Critério": x.label, "Inicial": q.level_label(x.before),
                                            "Atual": q.level_label(x.after)} for x in cmp.criteria]),
                             hide_index=True, width="stretch")

# ── Histórico ───────────────────────────────────────────────────────────────
elif view == "Histórico":
    st.markdown("### Histórico de avaliações")
    if need_evals():
        st.dataframe(pd.DataFrame([{"Data": report.fmt_date(e["evaluation_date"]), "Treinador": e["coach_name"] or "—",
                                    "Média": q.fmt_score(e["mean"]), "Classificação": e["label"] or "—",
                                    "Observações": e["observations"] or ""} for e in reversed(evs)]),
                     hide_index=True, width="stretch")
        by_eid = {e["id"]: e for e in evs}
        eid = st.selectbox("Abrir avaliação", list(by_eid)[::-1], format_func=lambda i: eval_label(by_eid[i]))
        ev = by_eid[eid]
        with st.container(border=True):
            a, b = st.columns([1, 1.3], gap="large")
            with a:
                ui.score_card(COMP.name, ev["mean"])
                st.plotly_chart(ui.radar_fig(dim_names(), [("Avaliação", dim_values(ev), ui.ACCENT)], 300), width="stretch",
                                key="hist_radar")
            with b:
                for d in COMP.dimensions:
                    st.markdown(f"**{d.title}**")
                    for c in d.criteria:
                        x, y = st.columns([3, 2])
                        x.markdown(c.label)
                        y.markdown(ui.chip(ev["scores"].get(c.key)), unsafe_allow_html=True)
                st.markdown(f"**Observações:** {ev['observations'] or '—'}")
            st.button("✏️ Editar esta avaliação", on_click=load_for_edit, args=(eid,))
            with st.popover("🗑️ Apagar avaliação"):
                st.warning("Esta ação é definitiva.")
                if st.checkbox("Confirmo que quero apagar esta avaliação", key=f"del_ok_{eid}"):
                    if st.button("Apagar definitivamente", type="primary", key=f"del_{eid}"):
                        svc.delete_evaluation(eid, confirm=True)
                        st.session_state.flash = ("success", "Avaliação apagada.")
                        st.rerun()

# ── Relatório ───────────────────────────────────────────────────────────────
elif view == "Relatório":
    st.markdown("### Relatório individual")
    if need_evals():
        by_eid = {e["id"]: e for e in evs}
        eid = st.selectbox("Avaliação", list(by_eid)[::-1], format_func=lambda i: eval_label(by_eid[i]))
        data = report.build_report(eid)
        html = report.render_html(data)
        fname = f"relatorio_lancamento_{data.player.replace(' ', '_')}_{data.date}.html"
        st.download_button("⬇️ Descarregar relatório (HTML)", html, file_name=fname, mime="text/html", type="primary")
        st.caption("Para PDF: abra o ficheiro no navegador e use Imprimir → Guardar como PDF.")
        components.html(html, height=1500, scrolling=True)
