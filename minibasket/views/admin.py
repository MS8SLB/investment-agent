"""Vista «Utilizadores» (só administrador): contas, perfis e ligações."""

import pandas as pd
import streamlit as st

from minibasket import access, auth
from minibasket.views.common import current_user

ROLE_OPTIONS = ["coach", "guardian", "admin"]


def _player_label(p: dict) -> str:
    return f"{p['name']} — {p['team'] or 'sem equipa'} ({p['category'] or '—'})"


def render() -> None:
    user = current_user()
    users = access.list_users(user)
    clubs = access.list_clubs(user)
    st.dataframe(pd.DataFrame([{
        "Utilizador": u["username"], "Nome": u["display_name"], "Perfil": auth.role_label(u["role"]),
        "Ativo": "Sim" if u["active"] else "Não", "Palavra-passe": "Definida" if u["has_password"] else "Por definir",
    } for u in users]), hide_index=True)

    with st.expander("➕ Nova conta"):
        with st.form("new_user", clear_on_submit=True):
            c1, c2 = st.columns(2)
            username = c1.text_input("Nome de utilizador")
            name = c2.text_input("Nome")
            c3, c4 = st.columns(2)
            role = c3.selectbox("Perfil", ROLE_OPTIONS, format_func=auth.role_label)
            club = c4.selectbox("Clube (treinadores)", [None] + clubs, format_func=lambda c: "—" if c is None else c["name"])
            pw = st.text_input("Palavra-passe inicial", type="password", help=f"Pelo menos {auth.MIN_PASSWORD} caracteres.")
            if st.form_submit_button("Criar conta"):
                try:
                    access.admin_create_user(user, username, name, role, pw, club["id"] if club else None)
                    st.success("Conta criada.")
                    st.rerun()
                except auth.AuthError as e:
                    st.error(str(e))

    st.subheader("Gerir conta")
    target = st.selectbox("Conta", users, key="admin_target",
                          format_func=lambda u: f"{u['display_name']} ({u['username']}) — {auth.role_label(u['role'])}")
    if not target:
        return
    tid = target["id"]
    cur = access.admin_assignments(user, tid)

    with st.form(f"pw_{tid}", clear_on_submit=True):
        pw = st.text_input("Nova palavra-passe", type="password", help="Também desbloqueia a conta.")
        if st.form_submit_button("Definir palavra-passe"):
            try:
                access.admin_set_password(user, tid, pw)
                st.success("Palavra-passe definida.")
            except auth.AuthError as e:
                st.error(str(e))

    active = st.toggle("Conta ativa", value=bool(target["active"]), key=f"active_{tid}")
    if active != bool(target["active"]):
        try:
            access.admin_set_active(user, tid, active)
            st.rerun()
        except auth.AuthError as e:
            st.error(str(e))

    teams = access.list_teams(user)
    players = access.search_players(user)
    by_team = {t["id"]: t for t in teams}
    by_player = {p["id"]: p for p in players}
    if target["role"] == "coach":
        club_ids = [None] + [c["id"] for c in clubs]
        club = st.selectbox("Clube", club_ids, index=club_ids.index(target["club_id"]) if target["club_id"] in club_ids else 0,
                            key=f"club_{tid}", format_func=lambda i: "—" if i is None else next(c["name"] for c in clubs if c["id"] == i))
        team_ids = st.multiselect("Equipas que acompanha", list(by_team), default=[i for i in cur["teams"] if i in by_team],
                                  key=f"teams_{tid}",
                                  format_func=lambda i: f"{by_team[i]['name']} ({by_team[i]['category']}, {by_team[i]['season']})")
        extra = st.multiselect("Acesso individual a jogadores (além das equipas)", list(by_player),
                               default=[i for i in cur["player_access"] if i in by_player], key=f"access_{tid}",
                               format_func=lambda i: _player_label(by_player[i]))
        if st.button("Guardar acessos", key=f"save_coach_{tid}"):
            access.admin_set_club(user, tid, club)
            access.admin_set_team_assignments(user, tid, team_ids)
            access.admin_set_player_access(user, tid, extra)
            st.success("Acessos guardados.")
    elif target["role"] == "guardian":
        kids = st.multiselect("Educandos associados", list(by_player), default=[i for i in cur["children"] if i in by_player],
                              key=f"kids_{tid}", format_func=lambda i: _player_label(by_player[i]))
        if st.button("Guardar associações", key=f"save_guardian_{tid}"):
            access.admin_set_guardian_links(user, tid, kids)
            st.success("Associações guardadas.")
        st.caption("O encarregado de educação só vê o relatório e a evolução dos educandos aqui indicados.")
    else:
        st.caption("O administrador tem acesso a tudo.")
