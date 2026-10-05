"""A minha conta: alterar a própria palavra-passe."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from basketball_eval import auth

st.set_page_config(page_title="A minha conta", page_icon="👤", initial_sidebar_state="collapsed")
me = st.session_state.get("auth_user")
if not me:
    st.error("Sessão não iniciada.")
    st.stop()

st.markdown(f"### 👤 {me['display_name']}")
st.caption(f"Utilizador: {me['username']} · Perfil: {'administrador' if me['role'] == 'admin' else 'treinador'}")
st.markdown("##### Alterar palavra-passe")
with st.form("change_pw", clear_on_submit=True):
    old = st.text_input("Palavra-passe atual", type="password")
    new = st.text_input(f"Nova palavra-passe (mín. {auth.MIN_PASSWORD} caracteres)", type="password")
    again = st.text_input("Repita a nova palavra-passe", type="password")
    if st.form_submit_button("Guardar", type="primary"):
        if new != again:
            st.error("As palavras-passe não coincidem.")
        else:
            try:
                auth.change_password(me["id"], old, new)
                st.success("Palavra-passe alterada.")
            except auth.AuthError as e:
                st.error(str(e))
