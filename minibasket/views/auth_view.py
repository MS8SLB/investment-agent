"""Início de sessão, configuração inicial e conta do utilizador."""

import streamlit as st

from minibasket import auth


def render_setup() -> None:
    st.title("🏀 Configuração inicial")
    st.write("Ainda não existe nenhum administrador. Crie a conta que vai gerir a plataforma, "
             "os treinadores e os encarregados de educação.")
    with st.form("setup_admin"):
        username = st.text_input("Nome de utilizador", placeholder="ex.: admin")
        name = st.text_input("Nome")
        pw = st.text_input("Palavra-passe", type="password", help=f"Pelo menos {auth.MIN_PASSWORD} caracteres.")
        pw2 = st.text_input("Repetir palavra-passe", type="password")
        if st.form_submit_button("Criar administrador", type="primary"):
            try:
                if pw != pw2:
                    raise auth.AuthError("As palavras-passe não coincidem.")
                uid = auth.create_first_admin(username, name, pw)
                st.session_state["user_id"] = uid
                st.rerun()
            except auth.AuthError as e:
                st.error(str(e))


def render_login() -> None:
    st.title("🏀 Avaliação do Minibasquete")
    st.caption("Entre com a sua conta de treinador, administrador ou encarregado de educação.")
    with st.form("login"):
        username = st.text_input("Nome de utilizador")
        pw = st.text_input("Palavra-passe", type="password")
        if st.form_submit_button("Entrar", type="primary"):
            try:
                user = auth.authenticate(username, pw)
            except auth.AuthError as e:
                st.error(str(e))
                return
            if user is None:
                st.error("Nome de utilizador ou palavra-passe incorretos.")
            else:
                st.session_state["user_id"] = user["id"]
                st.rerun()
    st.caption("Se não tem conta ou perdeu a palavra-passe, peça ao administrador do seu clube.")


def require_login() -> dict:
    """Devolve o utilizador autenticado; caso contrário mostra o ecrã de entrada e pára a execução."""
    if auth.needs_setup():
        render_setup()
        st.stop()
    uid = st.session_state.get("user_id")
    user = auth.get_user(uid)                  # revalidado a cada execução (a conta pode ter sido desativada)
    if user is None:
        if uid is not None:                    # sessão antiga inválida: limpa seleções; nunca com o formulário a meio
            st.session_state.clear()
        render_login()
        st.stop()
    st.session_state["user"] = user
    return user


def sidebar_account(user: dict) -> None:
    st.sidebar.divider()
    st.sidebar.caption(f"Sessão: **{user['display_name']}** · {auth.role_label(user['role'])}")
    with st.sidebar.expander("Alterar palavra-passe"):
        with st.form("change_pw", clear_on_submit=True):
            cur = st.text_input("Palavra-passe atual", type="password")
            new = st.text_input("Nova palavra-passe", type="password")
            new2 = st.text_input("Repetir nova palavra-passe", type="password")
            if st.form_submit_button("Guardar"):
                try:
                    if new != new2:
                        raise auth.AuthError("As palavras-passe não coincidem.")
                    auth.change_own_password(user["id"], cur, new)
                    st.success("Palavra-passe alterada.")
                except auth.AuthError as e:
                    st.error(str(e))
    if st.sidebar.button("Terminar sessão", key="logout"):
        st.session_state.clear()          # limpa também seleções (jogador, equipa) da sessão
        st.rerun()
