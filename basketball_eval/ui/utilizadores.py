"""Administração de contas de treinadores (só administradores)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from basketball_eval import auth

st.set_page_config(page_title="Utilizadores", page_icon="🔑", layout="wide", initial_sidebar_state="collapsed")
me = st.session_state.get("auth_user")
if not me or me.get("role") != "admin":          # defesa em profundidade: a página nem é registada para treinadores
    st.error("Acesso reservado a administradores.")
    st.stop()

st.markdown("### 🔑 Utilizadores")
if flash := st.session_state.pop("adm_flash", None):
    st.success(flash)

users = auth.list_users()
st.dataframe(pd.DataFrame([{
    "Utilizador": u.username, "Nome": u.display_name, "Perfil": "Administrador" if u.is_admin else "Treinador",
    "Estado": "Ativo" if u.active else "Desativado",
    "Bloqueada até (UTC)": u.locked_until[:16] if u.locked_until else "",
    "Troca de palavra-passe pendente": "Sim" if u.must_change_password else "",
} for u in users]), hide_index=True, width="stretch")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("##### Nova conta")
    with st.form("new_user", clear_on_submit=True):
        username = st.text_input("Utilizador (ex.: rui.costa)")
        name = st.text_input("Nome do treinador")
        role = st.selectbox("Perfil", auth.ROLES, format_func=lambda r: "Administrador" if r == "admin" else "Treinador",
                            index=1)
        temp = st.text_input(f"Palavra-passe temporária (mín. {auth.MIN_PASSWORD})", type="password")
        if st.form_submit_button("Criar conta", type="primary"):
            try:
                auth.create_user(username, name, temp, role=role, must_change=True)
                auth.audit(me["username"], "criar_conta", detail=username.strip().lower())
                st.session_state.adm_flash = "Conta criada. Entregue a palavra-passe temporária ao treinador (será obrigado a mudá-la)."
                st.rerun()
            except auth.AuthError as e:
                st.error(str(e))
with right:
    st.markdown("##### Gerir conta")
    by_id = {u.id: u for u in users}
    uid = st.selectbox("Conta", list(by_id), format_func=lambda i: f"{by_id[i].username} · {by_id[i].display_name}")
    if uid:
        target = by_id[uid]
        with st.form("reset_pw", clear_on_submit=True):
            temp2 = st.text_input("Nova palavra-passe temporária", type="password")
            if st.form_submit_button("Repor palavra-passe"):
                try:
                    auth.reset_password(uid, temp2, must_change=True)
                    auth.audit(me["username"], "repor_palavra_passe", detail=target.username)
                    st.session_state.adm_flash = f"Palavra-passe de {target.username} reposta (e conta desbloqueada)."
                    st.rerun()
                except auth.AuthError as e:
                    st.error(str(e))
        label = "Desativar conta" if target.active else "Reativar conta"
        if st.button(label):
            try:
                auth.set_active(uid, not target.active)
                auth.audit(me["username"], "desativar_conta" if target.active else "reativar_conta", detail=target.username)
                st.session_state.adm_flash = f"Conta {target.username}: {'desativada' if target.active else 'reativada'}."
                st.rerun()
            except auth.AuthError as e:
                st.error(str(e))

with st.expander("Registo de ações (últimas 100)"):
    st.dataframe(pd.DataFrame(auth.list_audit(100)).rename(columns={
        "at": "Data (UTC)", "username": "Utilizador", "action": "Ação", "evaluation_id": "Avaliação",
        "player_id": "Jogador (id)", "detail": "Detalhe"}).drop(columns=["id"], errors="ignore"),
        hide_index=True, width="stretch")
