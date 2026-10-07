"""Fase 10 — interface: início de sessão, perfis e isolamento."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from ui_helper import APP, logged_in_app  # noqa: E402

from minibasket import auth, db, evaluations as ev, service  # noqa: E402
from minibasket import competencies as comp  # noqa: E402


def flat(n, **over):
    return {**{k: n for k in comp.KEYS}, **over}


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    db.init_db()
    return tmp_path


def raw_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP).run(timeout=30)


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def texts(at):
    return " ".join(x.value for kind in (at.markdown, at.caption, at.info, at.error, at.success, at.warning, at.subheader, at.title)
                    for x in kind)


# ── configuração inicial ────────────────────────────────────────────────────
def test_first_run_shows_setup_not_data(tmpdb):
    at = raw_app()
    assert not at.exception and "Configuração inicial" in texts(at)
    assert not at.sidebar.radio                                           # sem navegação antes de haver sessão


def test_setup_creates_admin_and_logs_in(tmpdb):
    at = raw_app()
    at.text_input[0].set_value("admin"); at.text_input[1].set_value("Maria Admin")
    at.text_input[2].set_value("palavra-longa-1"); at.text_input[3].set_value("diferente")
    click(at, "Criar administrador")
    assert any("não coincidem" in e.value for e in at.error) and auth.needs_setup()
    at.text_input[3].set_value("palavra-longa-1")
    click(at, "Criar administrador")
    assert not at.exception and not auth.needs_setup()
    assert "Maria Admin" in texts(at) and at.sidebar.radio[0].value == "Dashboard"


# ── início / fim de sessão ──────────────────────────────────────────────────
def test_login_flow_and_errors(tmpdb):
    auth.create_user("ana", "Ana Treinadora", "coach", "palavra-passe-1")
    auth.create_user("adm", "Adm", "admin", "palavra-passe-1")
    at = raw_app()
    assert "Entre com a sua conta" in texts(at) and not at.sidebar.radio
    at.text_input[0].set_value("ana"); at.text_input[1].set_value("errada")
    click(at, "Entrar")
    assert any("incorretos" in e.value for e in at.error) and not at.sidebar.radio
    at.text_input[0].set_value("naoexiste"); at.text_input[1].set_value("palavra-passe-1")
    click(at, "Entrar")
    assert any("incorretos" in e.value for e in at.error)                 # mesma mensagem: não revela se o utilizador existe
    at.text_input[0].set_value("ana"); at.text_input[1].set_value("palavra-passe-1")
    click(at, "Entrar")
    assert not at.exception and at.sidebar.radio and "Ana Treinadora" in texts(at)


def test_lockout_message_in_ui(tmpdb):
    auth.create_user("ana", "Ana", "coach", "palavra-passe-1")
    auth.create_user("adm", "Adm", "admin", "palavra-passe-1")
    at = raw_app()
    for _ in range(auth.MAX_FAILED):
        at.text_input[0].set_value("ana"); at.text_input[1].set_value("errada")
        click(at, "Entrar")
    at.text_input[0].set_value("ana"); at.text_input[1].set_value("palavra-passe-1")
    click(at, "Entrar")
    assert any("bloqueada" in e.value for e in at.error) and not at.sidebar.radio


def test_logout_clears_session_and_selections(tmpdb):
    auth.create_user("adm", "Adm", "admin", "palavra-passe-1")
    at = logged_in_app("admin", "adm")
    at.session_state["selected_player"] = 5
    click(at, "Terminar sessão")
    assert not at.sidebar.radio and "Entre com a sua conta" in texts(at)
    assert "user_id" not in at.session_state and "selected_player" not in at.session_state


def test_deactivated_account_is_kicked_out_on_next_run(tmpdb):
    auth.create_user("adm", "Adm", "admin", "palavra-passe-1")
    uid = auth.create_user("ana", "Ana", "coach", "palavra-passe-1")
    at = logged_in_app("coach", "ana")
    assert at.sidebar.radio
    auth.set_active(uid, False)
    at.run()
    assert not at.sidebar.radio and "Entre com a sua conta" in texts(at)


def test_change_password_from_sidebar(tmpdb):
    auth.create_user("adm", "Adm", "admin", "palavra-passe-1")
    at = logged_in_app("admin", "adm")
    inputs = {t.label: t for t in at.sidebar.text_input}
    inputs["Palavra-passe atual"].set_value("palavra-passe-1")
    inputs["Nova palavra-passe"].set_value("nova-palavra-2"); inputs["Repetir nova palavra-passe"].set_value("nova-palavra-2")
    click(at, "Guardar")
    assert auth.authenticate("adm", "nova-palavra-2") and not auth.authenticate("adm", "palavra-passe-1")


# ── perfis e navegação ──────────────────────────────────────────────────────
def test_navigation_per_role(tmpdb):
    auth.create_user("adm", "Adm", "admin", "palavra-passe-1")
    admin_sections = logged_in_app("admin", "adm").sidebar.radio[0].options
    coach_sections = logged_in_app("coach", "treino").sidebar.radio[0].options
    parent_sections = logged_in_app("guardian", "pais").sidebar.radio[0].options
    assert "Utilizadores" in admin_sections and "Utilizadores" not in coach_sections
    assert parent_sections == ["O meu educando"]
    assert {"Dashboard", "Jogadores", "Avaliar", "Relatórios"} <= set(coach_sections)


# ── isolamento de dados na interface ────────────────────────────────────────
@pytest.fixture
def school(tmpdb):
    auth.create_user("adm", "Adm", "admin", "palavra-passe-1")
    club = service.create_club("Clube")
    ta = service.create_team(club, "Sub-10 A", "Sub-10", "2024/2025")
    tb = service.create_team(club, "Sub-10 B", "Sub-10", "2024/2025")
    coach = auth.create_user("coacha", "Coach A", "coach", "palavra-passe-1", club_id=club)
    auth.set_team_assignments(coach, [ta])
    pais = auth.create_user("pais1", "Pai da Ana", "guardian", "palavra-passe-1")
    ids = {}
    for name, team in (("Ana", ta), ("Rui", ta), ("Eva", tb)):
        pid = service.create_player(name, team, sex="F", joined_on="2024-01-01")
        ids[name] = pid
        ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", flat(2), general_notes=f"INTERNA-{name}")
        ev.create_evaluation(pid, "2024-12-15", "2.º Período", flat(3), parent_message=f"MSG-{name}")
    auth.set_guardian_links(pais, [ids["Ana"]])
    return {"ids": ids, "ta": ta, "tb": tb, "pais": pais, "coach": coach}


def test_coach_ui_lists_only_own_players_and_teams(school):
    at = logged_in_app("coach", "coacha")
    at.sidebar.radio[0].set_value("Jogadores").run()
    assert not at.exception
    assert at.selectbox(key="selected_player").options == ["Ana — Sub-10", "Rui — Sub-10"]       # sem a Eva (outra equipa)
    at.sidebar.radio[0].set_value("Equipas").run()
    t = texts(at)
    assert "Sub-10 A" in t and "Sub-10 B" not in t
    at.sidebar.radio[0].set_value("Evolução da Equipa").run()
    at.radio(key="tevo_cat").set_value("Sub-10").run()
    assert not at.exception and at.selectbox(key="tevo_team").options == ["Sub-10 A (Sub-10, 2024/2025)"]


def test_coach_cannot_see_other_team_in_player_search(school):
    at = logged_in_app("coach", "coacha")
    at.sidebar.radio[0].set_value("Jogadores").run()
    at.text_input(key="player_search").set_value("Eva").run()
    assert any("Nenhum jogador encontrado" in i.value for i in at.info)
    at.text_input(key="player_search").set_value("Rui").run()
    assert not at.exception and "Rui" in " ".join(s.value for s in at.subheader)


def test_guardian_ui_only_own_child(school):
    at = logged_in_app("guardian", "pais1")
    assert not at.exception and at.sidebar.radio[0].options == ["O meu educando"]
    t = texts(at)
    assert "Ana" in t and "Rui" not in t and "Eva" not in t
    assert "MSG-Ana" in t and "INTERNA" not in t
    assert not at.dataframe                                              # nenhuma tabela coletiva/estatística
    assert not any(b.label == "Criar equipa" for b in at.button)
    assert len(at.tabs) == 2 and [x.label for x in at.tabs] == ["Relatório", "Evolução"]


def test_guardian_without_links_sees_friendly_message(school):
    auth.set_guardian_links(school["pais"], [])
    at = logged_in_app("guardian", "pais1")
    assert not at.exception and "nenhum educando associado" in texts(at)


def test_guardian_cannot_reach_staff_sections_by_state_tampering(school):
    at = logged_in_app("guardian", "pais1", run=False)
    at.session_state["selected_player"] = school["ids"]["Rui"]           # tentativa de forçar outro jogador
    at.run()
    assert not at.exception and "Rui" not in texts(at)


# ── administração ───────────────────────────────────────────────────────────
def test_admin_can_create_account_and_link_guardian(school):
    at = logged_in_app("admin", "adm")
    at.sidebar.radio[0].set_value("Utilizadores").run()
    assert not at.exception
    inputs = {t.label: t for t in at.text_input}
    inputs["Nome de utilizador"].set_value("pais2"); inputs["Nome"].set_value("Pai do Rui")
    inputs["Palavra-passe inicial"].set_value("palavra-passe-1")
    at.selectbox[0].set_value("guardian")
    click(at, "Criar conta")
    new = next(u for u in auth.list_users() if u["username"] == "pais2")
    assert new["role"] == "guardian" and auth.authenticate("pais2", "palavra-passe-1")
    at.selectbox(key="admin_target").set_value(next(u for u in auth.list_users() if u["username"] == "pais2")).run()
    at.multiselect(key=f"kids_{new['id']}").set_value([school["ids"]["Rui"]]).run()
    click(at, "Guardar associações")
    from minibasket import access
    assert [c["name"] for c in access.guardian_children(auth.get_user(new["id"]))] == ["Rui"]


def test_admin_assigns_teams_to_coach(school):
    at = logged_in_app("admin", "adm")
    at.sidebar.radio[0].set_value("Utilizadores").run()
    target = next(u for u in auth.list_users() if u["username"] == "coacha")
    at.selectbox(key="admin_target").set_value(target).run()
    at.multiselect(key=f"teams_{target['id']}").set_value([school["ta"], school["tb"]]).run()
    click(at, "Guardar acessos")
    from minibasket import access
    assert access.coached_team_ids(auth.get_user(target["id"])) == {school["ta"], school["tb"]}


def test_admin_cannot_deactivate_self_in_ui(school):
    at = logged_in_app("admin", "adm")
    at.sidebar.radio[0].set_value("Utilizadores").run()
    at.selectbox(key="admin_target").set_value(next(u for u in auth.list_users() if u["username"] == "adm")).run()
    uid = auth.get_user(next(u["id"] for u in auth.list_users() if u["username"] == "adm"))["id"]
    at.toggle(key=f"active_{uid}").set_value(False).run()
    assert any("própria conta" in e.value or "único administrador" in e.value for e in at.error)
    assert auth.get_user(uid) is not None
