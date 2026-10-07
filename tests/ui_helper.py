"""Auxiliar dos testes de interface: abre a app já autenticada."""

import os

APP = os.path.join(os.path.dirname(__file__), "..", "minibasket", "app.py")


def logged_in_app(role: str = "admin", username: str | None = None, run: bool = True):
    """AppTest com sessão iniciada (cria a conta na BD atual se ainda não existir).

    Chamar depois de redirecionar `minibasket.db.DB_PATH` para a BD temporária do teste.
    """
    from streamlit.testing.v1 import AppTest
    from minibasket import auth, db
    db.init_db()
    username = username or f"{role}ui"
    existing = next((u for u in auth.list_users() if u["username"] == username), None)
    uid = existing["id"] if existing else auth.create_user(username, f"{role.title()} UI", role, "palavra-passe-1")
    at = AppTest.from_file(APP)
    at.session_state["user_id"] = uid
    return at.run(timeout=30) if run else at
