"""Vista «Equipas»: clubes e equipas Sub-8 / Sub-10 / Sub-12."""

from datetime import date

import streamlit as st

from minibasket import service
from minibasket.db import CATEGORIES


def default_season(today: date | None = None) -> str:
    today = today or date.today()
    start = today.year if today.month >= 8 else today.year - 1
    return f"{start}/{start + 1}"


def render() -> None:
    clubs = service.list_clubs()
    with st.expander("➕ Novo clube", expanded=not clubs):
        with st.form("new_club", clear_on_submit=True):
            name = st.text_input("Nome do clube")
            if st.form_submit_button("Criar clube"):
                try:
                    service.create_club(name)
                    st.rerun()
                except service.ValidationError as e:
                    st.error(str(e))
    if not clubs:
        st.info("Crie primeiro um clube.")
        return

    with st.expander("➕ Nova equipa", expanded=True):
        with st.form("new_team", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
            club = c1.selectbox("Clube", clubs, format_func=lambda c: c["name"])
            name = c2.text_input("Nome da equipa")
            category = c3.selectbox("Escalão", CATEGORIES)
            season = c4.text_input("Época", default_season())
            if st.form_submit_button("Criar equipa"):
                try:
                    service.create_team(club["id"], name, category, season)
                    st.success("Equipa criada.")
                    st.rerun()
                except service.ValidationError as e:
                    st.error(str(e))

    st.subheader("Equipas")
    cat = st.radio("Escalão", ("Todos",) + CATEGORIES, horizontal=True, key="teams_cat")
    teams = service.list_teams(category=None if cat == "Todos" else cat)
    if not teams:
        st.caption("Sem equipas neste escalão.")
    for t in teams:
        tag = " · *Dados de teste*" if t["is_demo"] else ""
        st.markdown(f"**{t['name']}** — {t['category']} · {t['club']} · {t['season']} · "
                    f"{t['n_players']} jogador(es){tag}")
