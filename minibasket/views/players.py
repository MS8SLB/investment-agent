"""Vista «Jogadores»: pesquisa, filtros, criação e ficha individual."""

from datetime import date

import streamlit as st

from minibasket import service
from minibasket.db import CATEGORIES

SEX_LABELS = {"": "—", "M": "Masculino", "F": "Feminino"}


def _fmt_date(iso):
    return date.fromisoformat(iso).strftime("%d/%m/%Y") if iso else "—"


def _create_form(teams) -> None:
    with st.expander("➕ Novo jogador", expanded=False):
        if not teams:
            st.info("Crie primeiro uma equipa.")
            return
        with st.form("new_player", clear_on_submit=True):
            c1, c2 = st.columns(2)
            name = c1.text_input("Nome")
            team = c2.selectbox("Equipa", teams, format_func=lambda t: f"{t['name']} ({t['category']}, {t['season']})")
            c3, c4, c5 = st.columns(3)
            birth = c3.date_input("Data de nascimento", value=None, min_value=date(2005, 1, 1), max_value=date.today(),
                                  format="DD/MM/YYYY")
            sex = c4.selectbox("Sexo", list(SEX_LABELS), format_func=SEX_LABELS.get)
            jersey = c5.number_input("Nº da camisola", 0, 99, value=None, step=1)
            joined = st.date_input("Data de entrada na equipa", date.today(), format="DD/MM/YYYY")
            photo = st.file_uploader("Fotografia (opcional)", type=["png", "jpg", "jpeg", "webp"])
            notes = st.text_area("Observações")
            if st.form_submit_button("Criar jogador"):
                try:
                    path = service.save_photo(photo.getvalue(), photo.name) if photo else None
                    pid = service.create_player(name, team["id"], birth, sex, jersey, joined, notes, path)
                    st.session_state["selected_player"] = pid
                    st.rerun()
                except service.ValidationError as e:
                    st.error(str(e))


def _profile(pid: int, teams) -> None:
    p = service.get_player(pid)
    if not p:
        st.warning("Jogador inexistente.")
        return
    left, right = st.columns([1, 3])
    if p["photo_path"]:
        left.image(p["photo_path"], width=140)
    else:
        left.markdown("### 🏀")
    right.subheader(p["name"])
    if p["is_demo"]:
        right.caption("Dados de teste")
    right.write(f"**Escalão:** {p['category'] or '—'} · **Equipa:** {p['team'] or '—'} · **Clube:** {p['club'] or '—'}")
    right.write(f"**Época:** {p['season'] or '—'} · **Camisola:** {p['jersey_number'] if p['jersey_number'] is not None else '—'}"
                f" · **Entrada na equipa:** {_fmt_date(p['joined_on'])}")
    right.write(f"**Nascimento:** {_fmt_date(p['birth_date'])} · **Sexo:** {SEX_LABELS[p['sex'] or '']}")
    if p["notes"]:
        right.write(f"**Observações:** {p['notes']}")

    with st.expander("✏️ Editar ficha"):
        with st.form(f"edit_{pid}"):
            name = st.text_input("Nome", p["name"])
            birth = st.date_input("Data de nascimento",
                                  date.fromisoformat(p["birth_date"]) if p["birth_date"] else None,
                                  min_value=date(2005, 1, 1), max_value=date.today(), format="DD/MM/YYYY")
            sex = st.selectbox("Sexo", list(SEX_LABELS), index=list(SEX_LABELS).index(p["sex"] or ""),
                               format_func=SEX_LABELS.get)
            jersey = st.number_input("Nº da camisola", 0, 99, value=p["jersey_number"], step=1)
            photo = st.file_uploader("Nova fotografia", type=["png", "jpg", "jpeg", "webp"])
            notes = st.text_area("Observações", p["notes"] or "")
            if st.form_submit_button("Guardar"):
                try:
                    path = service.save_photo(photo.getvalue(), photo.name) if photo else None
                    service.update_player(pid, name, birth, sex, notes, path, jersey)
                    st.rerun()
                except service.ValidationError as e:
                    st.error(str(e))

    with st.expander("🔁 Mudar de equipa / escalão"):
        st.caption("O histórico anterior (equipas e avaliações) mantém-se.")
        options = [t for t in teams if t["id"] != p["team_id"]]
        if not options:
            st.info("Não há outras equipas.")
        else:
            with st.form(f"move_{pid}"):
                team = st.selectbox("Nova equipa", options,
                                    format_func=lambda t: f"{t['name']} ({t['category']}, {t['season']})")
                on = st.date_input("Data da mudança", date.today(), format="DD/MM/YYYY")
                jersey = st.number_input("Nº da camisola", 0, 99, value=None, step=1)
                if st.form_submit_button("Mudar"):
                    try:
                        service.change_team(pid, team["id"], on, jersey)
                        st.rerun()
                    except service.ValidationError as e:
                        st.error(str(e))

    st.markdown("**Percurso nas equipas**")
    for h in p["team_history"]:
        end = _fmt_date(h["left_on"]) if h["left_on"] else "atual"
        st.write(f"• {h['category']} — {h['team']} ({h['club']}, {h['season']}): "
                 f"{_fmt_date(h['joined_on'])} → {end}")


def render() -> None:
    teams = service.list_teams()
    _create_form(teams)

    c1, c2, c3 = st.columns([2, 1, 2])
    text = c1.text_input("🔍 Pesquisar por nome", key="player_search")
    cat = c2.selectbox("Escalão", ("Todos",) + CATEGORIES)
    pool = [t for t in teams if cat == "Todos" or t["category"] == cat]
    team = c3.selectbox("Equipa", [None] + pool,
                        format_func=lambda t: "Todas" if t is None else f"{t['name']} ({t['category']}, {t['season']})")
    players = service.search_players(text, None if cat == "Todos" else cat, team["id"] if team else None)

    if not players:
        st.info("Nenhum jogador encontrado.")
        return
    st.caption(f"{len(players)} jogador(es), por ordem alfabética.")
    ids = [p["id"] for p in players]
    if st.session_state.get("selected_player") not in ids:
        st.session_state["selected_player"] = ids[0]
    by_id = {p["id"]: p for p in players}
    pid = st.selectbox("Abrir ficha", ids, key="selected_player",
                       format_func=lambda i: f"{by_id[i]['name']} — {by_id[i]['category'] or 'sem equipa'}")
    st.divider()
    _profile(pid, teams)
