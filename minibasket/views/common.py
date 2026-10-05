"""Utilitários partilhados pelas vistas."""

from datetime import date

import streamlit as st

from minibasket import service
from minibasket.db import CATEGORIES


def fmt_date(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%d/%m/%Y")


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def pick_player(prefix: str):
    """Escalão → jogador. Devolve a ficha (dict) ou None se não houver jogadores."""
    cat = st.radio("Escalão", CATEGORIES, horizontal=True, key=f"{prefix}_cat")
    players = service.search_players(category=cat)
    if not players:
        st.info("Sem jogadores neste escalão. Crie-os na secção «Jogadores».")
        return None
    by_id = {p["id"]: p for p in players}
    pid = st.selectbox("Jogador", list(by_id), format_func=lambda i: by_id[i]["name"], key=f"{prefix}_player")
    return by_id[pid]
