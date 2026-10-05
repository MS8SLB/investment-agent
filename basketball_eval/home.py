"""Ponto de entrada da aplicação de avaliação de Minibasquete (navegação + acesso).

Executar: streamlit run basketball_eval/home.py

Configuração (variáveis de ambiente ou Streamlit secrets — ver basketball_eval/DEPLOY.md):
  DATABASE_URL     URL Postgres (partilhado entre dispositivos). Sem ele usa SQLite local.
  ADMIN_PASSWORD   ativa as CONTAS individuais: cria o 1.º administrador (ADMIN_USERNAME, por omissão «admin»)
                   se ainda não existirem utilizadores. O administrador cria as contas dos treinadores.
  APP_PASSWORD     alternativa simples: uma palavra-passe única do clube (sem contas).
Com DATABASE_URL é obrigatório ADMIN_PASSWORD (ou já haver contas) ou APP_PASSWORD.

Cada fundamento técnico futuro (Drible, Passe…) acrescenta-se como nova página no grupo
«Avaliação · Técnica Individual». O módulo quantitativo existente (app.py) mantém-se intacto.
"""

import hmac
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from basketball_eval import auth

HERE = os.path.dirname(os.path.abspath(__file__))


def _setting(name: str):
    """Segredo do Streamlit (secrets.toml / painel da cloud) ou variável de ambiente."""
    try:
        value = st.secrets[name]
    except Exception:                      # sem ficheiro de segredos ou chave em falta
        value = None
    return value or os.environ.get(name)


def _login_form() -> None:
    st.markdown("### 🏀 Avaliação de Minibasquete")
    with st.form("login"):
        username = st.text_input("Utilizador")
        password = st.text_input("Palavra-passe", type="password")
        if st.form_submit_button("Entrar", type="primary"):
            try:
                user = auth.authenticate(username, password)
            except auth.AccountLocked as e:
                st.error(str(e))
                return
            if user is None:
                time.sleep(1)              # trava tentativas repetidas
                st.error("Utilizador ou palavra-passe incorretos.")
                return
            st.session_state.auth_user = {"id": user.id}
            st.rerun()


def _force_change(user: "auth.User") -> None:
    st.markdown(f"### Olá, {user.display_name}")
    st.info("Por segurança, defina agora a sua própria palavra-passe.")
    with st.form("force_change"):
        old = st.text_input("Palavra-passe atual (temporária)", type="password")
        new = st.text_input(f"Nova palavra-passe (mín. {auth.MIN_PASSWORD} caracteres)", type="password")
        again = st.text_input("Repita a nova palavra-passe", type="password")
        if st.form_submit_button("Guardar e continuar", type="primary"):
            if new != again:
                st.error("As palavras-passe não coincidem.")
                return
            try:
                auth.change_password(user.id, old, new)
            except auth.AuthError as e:
                st.error(str(e))
                return
            st.rerun()


def _shared_password_gate(password) -> None:
    if st.session_state.get("auth_ok"):
        if st.sidebar.button("Terminar sessão"):
            st.session_state.clear()
            st.rerun()
        return
    st.markdown("### 🏀 Avaliação de Minibasquete")
    with st.form("login_shared"):
        typed = st.text_input("Palavra-passe de acesso", type="password")
        if st.form_submit_button("Entrar", type="primary"):
            if hmac.compare_digest(typed.encode(), str(password).encode()):
                st.session_state.auth_ok = True
                st.rerun()
            time.sleep(1)
            st.error("Palavra-passe incorreta.")
    st.stop()


def _gate():
    """Devolve o utilizador autenticado (modo contas) ou None (modo local / palavra-passe única)."""
    db_url, admin_pw, shared = _setting("DATABASE_URL"), _setting("ADMIN_PASSWORD"), _setting("APP_PASSWORD")
    if db_url:
        os.environ["DATABASE_URL"] = str(db_url)
    try:
        if admin_pw:
            auth.bootstrap_admin(_setting("ADMIN_USERNAME") or "admin", str(admin_pw))
        accounts = bool(admin_pw) or auth.has_users()
    except Exception as e:                 # BD inacessível/URL errado: não mostrar detalhes
        print(f"Erro de ligação à base de dados: {type(e).__name__}", file=sys.stderr)
        st.error("Não foi possível ligar à base de dados. Verifique `DATABASE_URL` nos segredos.")
        st.stop()
    if not accounts:
        if db_url and not shared:          # dados de menores numa BD remota: nunca sem controlo de acesso
            st.error("Configuração incompleta: defina `ADMIN_PASSWORD` (contas) ou `APP_PASSWORD` nos segredos.")
            st.stop()
        if shared:
            _shared_password_gate(shared)
        return None
    sess = st.session_state.get("auth_user")
    user = auth.get_user(sess["id"]) if sess else None      # revalida: conta desativada/removida perde o acesso
    if user is None or not user.active:
        st.session_state.pop("auth_user", None)
        _login_form()
        st.stop()
    if user.must_change_password:
        _force_change(user)
        st.stop()
    st.session_state.auth_user = {"id": user.id, "username": user.username, "display_name": user.display_name,
                                  "role": user.role}
    st.sidebar.caption(f"👤 {user.display_name} · {'administrador' if user.is_admin else 'treinador'}")
    if st.sidebar.button("Terminar sessão"):
        st.session_state.clear()
        st.rerun()
    return user


_user = _gate()

pages = {
    "Avaliação · Técnica Individual": [
        st.Page(os.path.join(HERE, "ui", "lancamento.py"), title="Lançamento", icon="🏀",
                url_path="lancamento", default=True),
    ],
    "Avaliação quantitativa": [
        st.Page(os.path.join(HERE, "app.py"), title="Movimentos Defensivos", icon="🛡️",
                url_path="movimentos-defensivos"),
    ],
}
if _user is not None:
    pages["Conta"] = [st.Page(os.path.join(HERE, "ui", "conta.py"), title="A minha conta", icon="👤",
                              url_path="conta")]
    if _user.is_admin:
        pages["Conta"].append(st.Page(os.path.join(HERE, "ui", "utilizadores.py"), title="Utilizadores",
                                      icon="🔑", url_path="utilizadores"))

st.navigation(pages).run()
