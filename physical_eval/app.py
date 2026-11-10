"""Plataforma de registo de avaliação física — interface dos estagiários.

Executar: streamlit run physical_eval/app.py
Opcional: PHYSICAL_EVAL_PASSWORD=... (palavra-passe partilhada) · PHYSICAL_EVAL_DB=caminho.db
"""

import hmac
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from physical_eval import norms, service
from physical_eval.protocols import PROTOCOLS

st.set_page_config(page_title="Avaliação Física — ENB", layout="wide")

# ── Acesso (palavra-passe partilhada, opcional) ─────────────────────────────
_pw = os.environ.get("PHYSICAL_EVAL_PASSWORD")
if _pw and not st.session_state.get("auth"):
    st.title("Avaliação Física — ENB")
    typed = st.text_input("Palavra-passe", type="password")
    if typed:
        if hmac.compare_digest(typed.encode(), _pw.encode()):
            st.session_state["auth"] = True
            st.rerun()
        st.error("Palavra-passe incorreta.")
    st.stop()

st.title("Avaliação Física — Perfil Antropométrico e Motor")
st.caption("Os percentis são calculados automaticamente por idade a partir das tabelas de referência da ficha ENB. "
           "Percentil vazio = não existe referência para essa idade/sexo.")

with st.sidebar:
    st.header("Estagiário")
    intern = st.text_input("O meu nome", key="intern", placeholder="obrigatório para registar")
    st.caption("O nome fica associado a cada registo.")

PCTL_FIELDS = [("altura", "Altura", "pctl_altura", "height_cm"), ("peso", "Peso", "pctl_peso", "weight_kg"),
               ("envergadura", "Envergadura", "pctl_envergadura", "wingspan_cm"),
               ("salto", "Salto horizontal", "pctl_salto", "jump_best"),
               ("corrida", "20 m corrida", "pctl_corrida", "run_best"),
               ("drible", "20 m drible", "pctl_drible", "dribble_best")]


def _fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{nd}f}".replace(".", ",")


def _athlete_label(a):
    return f"{a['name']}" + (f" · {a['team']}" if a.get("team") else "") + f" (#{a['id']})"


def _txt(v):
    return "" if v is None else f"{v:g}".replace(".", ",")


tab_reg, tab_hist, tab_ath, tab_all, tab_ref = st.tabs(
    ["Registar avaliação", "Ficha do atleta", "Atletas", "Todos os registos", "Protocolos e referências"])

# ── Registar ────────────────────────────────────────────────────────────────
with tab_reg:
    athletes = service.list_athletes()
    if not athletes:
        st.info("Ainda não há atletas. Adicione-os no separador «Atletas».")
    else:
        lab = {_athlete_label(a): a for a in athletes}
        ath = lab[st.selectbox("Atleta", list(lab), key="reg_ath")]
        editing = None
        hist = service.athlete_history(ath["id"])
        if hist:
            opts = {"Nova avaliação": None} | {f"{h['eval_date']} (#{h['id']})": h for h in hist}
            editing = opts[st.selectbox("Nova avaliação ou corrigir uma existente", list(opts), key=f"edit_{ath['id']}")]
        e = editing or {}
        k = f"{ath['id']}_{e.get('id', 'new')}"          # reinicia os campos ao mudar de atleta/registo

        c1, c2, c3 = st.columns(3)
        ev_date = c1.date_input("Data da avaliação", date.fromisoformat(e["eval_date"]) if e else date.today(),
                                max_value=date.today(), key=f"d{k}")
        if ath.get("birth_date"):
            age = service.age_on(ath["birth_date"], ev_date)
            c2.metric("Idade (calculada)", f"{age} anos")
            manual_age = None
        else:
            manual_age = c2.number_input("Idade (anos)", 3, 60, int(e.get("age") or 10), key=f"a{k}",
                                         help="Atleta sem data de nascimento: indique a idade.")
            age = manual_age
        c3.metric("Sexo", {"M": "Masculino", "F": "Feminino"}.get(ath.get("sex"), "não definido"))
        if ath.get("sex") not in ("M", "F"):
            st.warning("Sem sexo definido, o percentil do salto não é calculado. Defina-o no separador «Atletas».")

        st.markdown("##### Medidas antropométricas")
        m1, m2, m3 = st.columns(3)
        height = m1.text_input("Altura (cm)", _txt(e.get("height_cm")), key=f"h{k}", placeholder="ex.: 150,5")
        weight = m2.text_input("Peso (kg)", _txt(e.get("weight_kg")), key=f"w{k}", placeholder="ex.: 39,7")
        wing = m3.text_input("Envergadura (cm)", _txt(e.get("wingspan_cm")), key=f"e{k}", placeholder="ex.: 153,7")

        st.markdown("##### Testes motores — registe todas as tentativas; fica o melhor")
        j = st.columns(3)
        jumps = [j[i].text_input(f"Salto horizontal — tentativa {i + 1} (m)", _txt(e.get(f"jump_{i + 1}")),
                                 key=f"j{i}{k}", placeholder="ex.: 1,54") for i in range(3)]
        r = st.columns(2)
        runs = [r[i].text_input(f"20 m corrida — tentativa {i + 1} (s)", _txt(e.get(f"run_{i + 1}")),
                                key=f"r{i}{k}", placeholder="ex.: 3,62") for i in range(2)]
        d = st.columns(2)
        dribs = [d[i].text_input(f"20 m drible — tentativa {i + 1} (s)", _txt(e.get(f"dribble_{i + 1}")),
                                 key=f"b{i}{k}", placeholder="ex.: 3,82") for i in range(2)]
        notes = st.text_area("Notas (opcional)", e.get("notes") or "", key=f"n{k}")

        # Pré-visualização dos percentis antes de guardar.
        rec, err = None, None
        try:
            rec = service.build_assessment(
                eval_date=ev_date, recorded_by=intern or e.get("recorded_by") or "?", age=age,
                height=height, weight=weight, wingspan=wing, jumps=jumps, runs=runs, dribbles=dribs, notes=notes)
        except ValueError as ex:
            err = str(ex)
        if err and "pelo menos uma medição" not in err:
            st.error(err)
        elif rec:
            pr = service.with_percentiles(rec, ath.get("sex"))
            cols = st.columns(6)
            for col, (tk, label, pk, vk) in zip(cols, PCTL_FIELDS):
                nd_ = 1 if tk in ("altura", "peso", "envergadura") else 2
                p = pr.get(pk)
                col.metric(label, _fmt(pr.get(vk), nd_), None if p is None else f"P{p}", delta_color="off")
            if norms.is_provisional("drible", age) and rec.get("dribble_best") is not None:
                st.caption("⚠️ Percentil do drible aos 12–15 anos: projeção provisória (a validar).")

        if st.button("GUARDAR AVALIAÇÃO" if not editing else "ATUALIZAR AVALIAÇÃO", type="primary", key=f"save{k}"):
            if not intern.strip():
                st.error("Escreva o seu nome na barra lateral antes de guardar.")
            elif err or rec is None:
                st.error(err or "Dados inválidos.")
            else:
                service.save_assessment(ath["id"], rec, editing["id"] if editing else None)
                st.success("Avaliação guardada." if not editing else "Avaliação atualizada.")
        if editing and st.button("Apagar esta avaliação", key=f"del{k}"):
            st.session_state[f"confirm_del_{k}"] = True
        if st.session_state.get(f"confirm_del_{k}"):
            st.warning("Apagar é definitivo.")
            if st.button("Confirmar apagar", key=f"cdel{k}"):
                service.delete_assessment(editing["id"])
                st.session_state[f"confirm_del_{k}"] = False
                st.rerun()

# ── Ficha do atleta ─────────────────────────────────────────────────────────
with tab_hist:
    athletes = service.list_athletes()
    if not athletes:
        st.info("Sem atletas.")
    else:
        lab = {_athlete_label(a): a for a in athletes}
        a = lab[st.selectbox("Atleta", list(lab), key="hist_ath")]
        h = service.athlete_history(a["id"])
        if not h:
            st.info("Sem avaliações registadas.")
        else:
            last = h[-1]
            st.subheader(f"{a['name']} — última avaliação: {last['eval_date']} ({last['age']} anos)")
            cols = st.columns(6)
            for col, (tk, label, pk, vk) in zip(cols, PCTL_FIELDS):
                p = last.get(pk)
                col.metric(label, _fmt(last.get(vk), 1 if tk in ("altura", "peso", "envergadura") else 2),
                           None if p is None else f"P{p}", delta_color="off")
            df = pd.DataFrame([{label: r.get(vk) for _, label, _, vk in PCTL_FIELDS} |
                               {"Data": r["eval_date"], "Idade": r["age"], "Registado por": r["recorded_by"]} for r in h])
            pdf = pd.DataFrame([{label: r.get(pk) for _, label, pk, _ in PCTL_FIELDS} | {"Data": r["eval_date"]} for r in h])
            st.markdown("**Valores**")
            st.dataframe(df.set_index("Data"), width="stretch")
            st.markdown("**Percentis**")
            st.dataframe(pdf.set_index("Data"), width="stretch")
            if len(h) > 1:
                pick = st.selectbox("Evolução", [(l, vk, tk) for tk, l, _, vk in PCTL_FIELDS],
                                    format_func=lambda t: t[0], key="evo")
                lbl, vk, tk = pick
                xs = [r["eval_date"] for r in h if r.get(vk) is not None]
                ys = [r[vk] for r in h if r.get(vk) is not None]
                fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
                fig.update_xaxes(type="category", title="Data")
                fig.update_yaxes(title=f"{lbl} ({norms.TESTS[tk][1]})")
                if norms.TESTS[tk][2] == "min":
                    fig.update_yaxes(autorange="reversed")
                    fig.add_annotation(xref="paper", yref="paper", x=0, y=1.1, showarrow=False,
                                       text="↑ Mais alto no gráfico = menor tempo = melhor")
                st.plotly_chart(fig, width="stretch")

# ── Atletas ─────────────────────────────────────────────────────────────────
with tab_ath:
    with st.form("new_ath", clear_on_submit=True):
        st.markdown("**Novo atleta**")
        f1, f2, f3, f4, f5 = st.columns([1, 3, 2, 2, 2])
        number = f1.text_input("N.º")
        name = f2.text_input("Nome")
        sex = f3.selectbox("Sexo", ["", "F", "M"], format_func=lambda v: {"": "—", "F": "Feminino", "M": "Masculino"}[v])
        bd = f4.date_input("Data de nascimento", value=None, min_value=date(1990, 1, 1), max_value=date.today())
        team = f5.text_input("Equipa / escalão")
        if st.form_submit_button("Adicionar"):
            try:
                service.add_athlete(name, sex or None, bd, team, number)
                st.success("Atleta adicionado.")
            except ValueError as ex:
                st.error(str(ex))
    ath_list = service.list_athletes()
    if ath_list:
        st.dataframe(pd.DataFrame(ath_list).drop(columns=["created_at"]).rename(columns={
            "number": "N.º", "name": "Nome", "sex": "Sexo", "birth_date": "Nascimento", "team": "Equipa"}),
            width="stretch", hide_index=True)
        with st.expander("Corrigir ou apagar atleta"):
            lab = {_athlete_label(a): a for a in ath_list}
            s = lab[st.selectbox("Atleta", list(lab), key="fix_ath")]
            with st.form(f"fix_{s['id']}"):
                g1, g2, g3, g4, g5 = st.columns([1, 3, 2, 2, 2])
                n2 = g1.text_input("N.º", s.get("number") or "")
                nm2 = g2.text_input("Nome", s["name"])
                sx2 = g3.selectbox("Sexo", ["", "F", "M"], index=["", "F", "M"].index(s.get("sex") or ""),
                                   format_func=lambda v: {"": "—", "F": "Feminino", "M": "Masculino"}[v])
                bd2 = g4.date_input("Data de nascimento", value=date.fromisoformat(s["birth_date"]) if s.get("birth_date") else None,
                                    min_value=date(1990, 1, 1), max_value=date.today())
                tm2 = g5.text_input("Equipa / escalão", s.get("team") or "")
                if st.form_submit_button("Guardar alterações"):
                    try:
                        service.update_athlete(s["id"], nm2, sx2 or None, bd2, tm2, n2)
                        st.success("Atleta atualizado.")
                    except ValueError as ex:
                        st.error(str(ex))
            if st.checkbox("Quero apagar este atleta e todas as suas avaliações", key=f"cd{s['id']}"):
                if st.button("Apagar atleta", key=f"da{s['id']}"):
                    service.delete_athlete(s["id"])
                    st.rerun()

# ── Todos os registos ───────────────────────────────────────────────────────
with tab_all:
    rows = service.export_rows()
    if not rows:
        st.info("Ainda não há registos.")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
        x1, x2 = st.columns(2)
        x1.download_button("Descarregar Excel (.xlsx)", service.export_xlsx(), "avaliacao_fisica.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        x2.download_button("Descarregar CSV", service.export_csv(), "avaliacao_fisica.csv", "text/csv")

# ── Protocolos e referências ────────────────────────────────────────────────
with tab_ref:
    st.markdown("#### Protocolos")
    st.dataframe(pd.DataFrame(PROTOCOLS, columns=["Teste", "Unidade", "Procedimento", "Fonte (ENB)"]),
                 width="stretch", hide_index=True)
    st.markdown("#### Tabelas de referência")
    which = st.selectbox("Teste", list(norms.TESTS), format_func=lambda t: norms.TESTS[t][0])
    rsex = st.radio("Sexo (salto)", ["F", "M"], horizontal=True,
                    format_func=lambda v: "Feminino" if v == "F" else "Masculino") if which == "salto" else None
    t = norms.reference_table(which, rsex)
    st.dataframe(pd.DataFrame.from_dict(t["rows"], orient="index", columns=[f"P{p}" for p in t["pcts"]])
                 .rename_axis("Idade"), width="stretch")
    st.caption("Fontes: Matulaitis (2013) e Matulaitis et al. (2019); salto: tabelas fornecidas (7–12 F) e Thomas et al. (2020) (13–15). "
               "Drible 12–15 anos: projeções provisórias. "
               + ("Maior valor = melhor." if norms.TESTS[which][2] == "max" else "Menor tempo = melhor."))
