"""Ponto de entrada da aplicação de avaliação de Minibasquete (navegação + acesso).

Executar: streamlit run basketball_eval/home.py

Configuração (variáveis de ambiente ou Streamlit secrets — ver basketball_eval/DEPLOY.md):
  DATABASE_URL  URL Postgres (partilhado entre dispositivos). Sem ele usa SQLite local.
  APP_PASSWORD  palavra-passe de acesso do clube. Obrigatória quando há DATABASE_URL.

Cada fundamento técnico futuro (Drible, Passe…) acrescenta-se como nova página no grupo
«Avaliação · Técnica Individual». O módulo quantitativo existente (app.py) mantém-se intacto.
"""

import hmac
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))


def _setting(name: str):
    """Segredo do Streamlit (secrets.toml / painel da cloud) ou variável de ambiente."""
    try:
        value = st.secrets[name]
    except Exception:                      # sem ficheiro de segredos ou chave em falta
        value = None
    return value or os.environ.get(name)


def _gate() -> None:
    """Acesso por palavra-passe do clube. Sem palavra-passe, só é permitido em modo local (SQLite)."""
    db_url, password = _setting("DATABASE_URL"), _setting("APP_PASSWORD")
    if db_url:
        os.environ["DATABASE_URL"] = str(db_url)
    if not password:
        if db_url:                         # dados de menores numa BD remota: nunca sem palavra-passe
            st.error("Configuração incompleta: defina `APP_PASSWORD` nos segredos da aplicação.")
            st.stop()
        return
    if st.session_state.get("auth_ok"):
        if st.sidebar.button("Terminar sessão"):
            st.session_state.clear()
            st.rerun()
        return
    st.markdown("### 🏀 Avaliação de Minibasquete")
    with st.form("login"):
        typed = st.text_input("Palavra-passe de acesso", type="password")
        if st.form_submit_button("Entrar", type="primary"):
            if hmac.compare_digest(typed.encode(), str(password).encode()):
                st.session_state.auth_ok = True
                st.rerun()
            time.sleep(1)                  # trava tentativas repetidas
            st.error("Palavra-passe incorreta.")
    st.stop()


_gate()

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

st.navigation(pages).run()
