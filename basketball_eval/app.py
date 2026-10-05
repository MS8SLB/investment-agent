"""Interface do treinador — Teste de Movimentos Defensivos.

Executar: streamlit run basketball_eval/app.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from basketball_eval import defensive_movement as dm
from basketball_eval import norms, service
from basketball_eval import qualitative as ql

st.set_page_config(page_title="Movimentos Defensivos", layout="wide")


def _require_password():
    """Se existir APP_PASSWORD (st.secrets ou variável de ambiente), exige-a antes de mostrar dados."""
    try:
        expected = st.secrets.get("APP_PASSWORD")
    except Exception:  # sem ficheiro de secrets (uso local)
        expected = None
    expected = expected or os.environ.get("APP_PASSWORD")
    if not expected or st.session_state.get("auth_ok"):
        return
    pw = st.text_input("Palavra-passe", type="password")
    if pw:
        if pw == expected:
            st.session_state["auth_ok"] = True
            st.rerun()
        st.error("Palavra-passe incorreta.")
    st.stop()


_require_password()
st.title(dm.TEST_NAME_PT)
st.caption(f"{dm.TEST_NAME_EN} — {dm.REFERENCE} · Unidade: segundos · **Menor tempo = melhor desempenho**")

tab_eval, tab_qual, tab_player, tab_team, tab_players = st.tabs(
    ["Avaliar", "Avaliação qualitativa", "Ficha individual", "Equipa", "Jogadores"])


def _players(category=None):
    return service.list_players(category=category)


# ── Avaliar ─────────────────────────────────────────────────────────────────
with tab_eval:
    category = st.radio("Escalão", dm.CATEGORIES, horizontal=True, key="cat")
    players = _players(category)
    if not players:
        st.info("Sem jogadores neste escalão. Adicione-os no separador «Jogadores».")
    else:
        by_label = {f"{p['name']} (#{p['id']})": p for p in players}
        player = by_label[st.selectbox("Jogador", list(by_label))]
        c1, c2 = st.columns(2)
        ev_date = c1.date_input("Data do teste", date.today())
        session = c2.text_input("Sessão/momento", placeholder="início / intermédia / final")
        c3, c4 = st.columns(2)
        location = c3.text_input("Local")
        coach = c4.text_input("Treinador")
        practice = st.checkbox("A tentativa 1 é de familiarização (não conta para o melhor tempo)")

        times, valid = [], []
        for i in range(1, dm.N_TRIALS + 1):
            a, b = st.columns([3, 1])
            times.append(a.text_input(f"Tentativa {i}: tempo (s)", key=f"t{i}", placeholder="ex.: 12.35"))
            valid.append(not b.checkbox("Inválida", key=f"inv{i}"))
        notes = st.text_area("Notas")

        prev = service.previous_best(player["id"], ev_date)
        if st.button("CALCULAR", type="primary"):
            try:
                st.session_state["calc"] = service.calculate(times, valid, practice, prev)
                st.session_state["calc_err"] = None
            except ValueError as e:
                st.session_state["calc"], st.session_state["calc_err"] = None, str(e)

        if st.session_state.get("calc_err"):
            st.error(st.session_state["calc_err"])
        calc = st.session_state.get("calc")
        if calc:
            if not calc.valid:
                st.warning("Sem tentativa válida — o teste deve ser repetido (não entra no histórico).")
            else:
                st.metric("Melhor tempo", dm.fmt_seconds(calc.best_time), help=f"Tentativa {calc.best_trial}")
                if calc.status:
                    st.metric("Evolução", dm.fmt_pct(calc.change_percentage, signed=True),
                              delta=dm.fmt_seconds(calc.change_seconds, signed=True), delta_color="inverse")
                    st.write(calc.message)
                else:
                    st.caption("Sem avaliação anterior para comparar.")
        if st.button("GUARDAR AVALIAÇÃO"):
            try:
                coach_id = service.get_or_create_coach(coach) if coach.strip() else None
                service.save_test(player["id"], ev_date, times, valid, practice, category, location or None,
                                  session or None, notes or None, coach_id)
                st.success("Avaliação guardada.")
            except ValueError as e:
                st.error(str(e))

# ── Avaliação qualitativa ───────────────────────────────────────────────────
with tab_qual:
    st.subheader(ql.TEST_NAME_PT)
    st.caption("Observe a execução do percurso e classifique cada critério de 1 a 4 (mais alto = melhor). "
               "Critérios e descritores são uma proposta de trabalho, não uma norma validada.")
    qplayers = _players()
    if not qplayers:
        st.info("Sem jogadores. Adicione-os no separador «Jogadores».")
    else:
        qlab = {f"{p['name']} · {p['category']} (#{p['id']})": p for p in qplayers}
        qp = qlab[st.selectbox("Jogador", list(qlab), key="qp")]
        q1, q2, q3 = st.columns(3)
        q_date = q1.date_input("Data da observação", date.today(), key="qd")
        q_session = q2.text_input("Sessão/momento", key="qs")
        q_coach = q3.text_input("Treinador", key="qc")
        ratings = {}
        for key, crit in ql.CRITERIA.items():
            with st.expander(f"{crit['name']} · {crit['dimension']}", expanded=False):
                for lvl in ql.SCALE:
                    st.markdown(f"**{lvl} – {ql.SCALE_LABELS[lvl]}:** {crit['descriptors'][lvl]}")
                ratings[key] = st.radio("Classificação", [None, *ql.SCALE], horizontal=True, key=f"q_{key}",
                                        format_func=lambda v: "—" if v is None else str(v))
        s_note = st.text_area("Notas — pontos fortes", key="qsn")
        i_note = st.text_area("Notas — a melhorar", key="qin")
        if any(v is not None for v in ratings.values()):
            sm = ql.summarize(ratings)
            st.metric("Média", f"{sm.average:.2f} / 4", help=f"{sm.n_rated}/{sm.n_total} critérios")
            dims = list(sm.by_dimension)
            fig = go.Figure(go.Scatterpolar(r=[sm.by_dimension[d] for d in dims] + [sm.by_dimension[dims[0]]],
                                            theta=dims + [dims[0]], fill="toself"))
            fig.update_polars(radialaxis=dict(range=[0, 4]))
            st.plotly_chart(fig, width="stretch")
        if st.button("GUARDAR OBSERVAÇÃO", type="primary"):
            try:
                cid = service.get_or_create_coach(q_coach) if q_coach.strip() else None
                service.save_qualitative(qp["id"], q_date, ratings, s_note or None, i_note or None,
                                         q_session or None, cid)
                st.success("Observação guardada.")
            except ValueError as e:
                st.error(str(e))
        qh = service.qualitative_history(qp["id"])
        if qh:
            st.divider()
            st.markdown("**Relatório mais recente**")
            st.text(service.qualitative_report_text(qp["id"]))
            st.line_chart(pd.DataFrame({"Média": [r["average"] for r in qh]},
                                       index=[r["evaluation_date"] for r in qh]))

# ── Ficha individual ────────────────────────────────────────────────────────
with tab_player:
    allp = _players()
    if allp:
        lab = {f"{p['name']} · {p['category']} (#{p['id']})": p for p in allp}
        pl = lab[st.selectbox("Jogador", list(lab), key="fp")]
        rep = service.player_report(pl["id"])
        if rep is None:
            st.info("Sem avaliações válidas.")
        else:
            t, evo = rep["test"], rep["evolution"]
            st.subheader("TESTE DE MOVIMENTOS DEFENSIVOS")
            st.markdown(f"**Melhor resultado:** {dm.fmt_seconds(t['best_time'])}")
            st.markdown("**Tentativas:** " + " · ".join(
                f"T{i} — {dm.fmt_seconds(t[f'trial_{i}'])}"
                + (" (inválida)" if not t[f"trial_{i}_valid"] else "")
                + (" (familiarização)" if i == 1 and t["first_is_practice"] else "")
                for i in (1, 2, 3)))
            if evo:
                st.markdown(f"**Avaliação anterior:** {dm.fmt_seconds(t['previous_best_time'])}")
                st.markdown(f"**Evolução:** {dm.fmt_seconds(t['change_seconds'], True)}")
                st.markdown(f"**{'Melhoria' if evo.status == dm.IMPROVED else 'Variação'}:** "
                            f"{dm.fmt_pct(t['change_percentage'], True)}")
                st.info(evo.message)
            if pl.get("birth_date"):
                ref = norms.reference_position(t["best_time"], norms.age_at(pl["birth_date"], t["evaluation_date"]), pl.get("sex"))
                if ref:
                    st.caption(f"Referência ({ref['source']}), {ref['age']} anos: {ref['note']}. "
                               "Posição indicativa; não é uma classificação.")
            h = pd.DataFrame(rep["history"])
            fig = go.Figure(go.Scatter(x=h["evaluation_date"], y=h["best_time"], mode="lines+markers+text",
                                       text=[dm.fmt_seconds(x) for x in h["best_time"]], textposition="top center"))
            # Eixo invertido: descer no tempo = subir no gráfico = evolução positiva.
            fig.update_yaxes(autorange="reversed", title="Tempo (s) — menor é melhor")
            fig.update_xaxes(title="Data da avaliação", type="category")
            fig.add_annotation(xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
                               text="↑ Mais alto no gráfico = menor tempo = melhor desempenho")
            st.plotly_chart(fig, width="stretch")
            st.text(service.coach_report_text(pl["id"]))

# ── Equipa ──────────────────────────────────────────────────────────────────
with tab_team:
    cat = st.selectbox("Escalão", dm.CATEGORIES, key="tc")
    st.caption("Escalões não são comparados entre si (não existe norma única validada).")
    a1, a2, b1, b2 = st.columns(4)
    d1a, d1b = a1.date_input("Momento 1 — de", date(date.today().year, 1, 1)), a2.date_input("Momento 1 — até", date.today())
    d2a, d2b = b1.date_input("Momento 2 — de", date.today(), key="d2a"), b2.date_input("Momento 2 — até", date.today(), key="d2b")
    cmp = service.team_comparison(cat, (d1a, d1b), (d2a, d2b))
    rows = [{"": n, "Jogadores": s.n, "Média": dm.fmt_seconds(s.mean), "Mediana": dm.fmt_seconds(s.median),
             "Melhor": dm.fmt_seconds(s.best), "Pior": dm.fmt_seconds(s.worst)}
            for n, s in (("Momento 1", cmp.before), ("Momento 2", cmp.after))]
    st.table(pd.DataFrame(rows).set_index(""))
    if cmp.paired_n:
        st.write(f"Jogadores nos dois momentos: {cmp.paired_n} — melhoraram {cmp.paired_improved}, "
                 f"mantiveram {cmp.paired_unchanged}, pioraram {cmp.paired_worsened}. "
                 f"Variação média: {dm.fmt_seconds(cmp.paired_mean_change_seconds, True)} (negativo = melhoria).")

# ── Jogadores ───────────────────────────────────────────────────────────────
with tab_players:
    with st.form("newp"):
        n = st.text_input("Nome")
        c1, c2, c3, c4 = st.columns(4)
        cat_p = c1.selectbox("Escalão", dm.CATEGORIES)
        sex = c2.selectbox("Sexo", ["", "M", "F"])
        team = c3.text_input("Equipa")
        bd = c4.date_input("Data de nascimento", value=None, min_value=date(2000, 1, 1))
        if st.form_submit_button("Adicionar"):
            try:
                service.add_player(n, cat_p, sex or None, team or None, bd)
                st.success("Jogador adicionado.")
            except ValueError as e:
                st.error(str(e))
    st.dataframe(pd.DataFrame(_players()), width="stretch")
