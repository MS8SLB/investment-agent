"""Teste de fumo da interface Streamlit (AppTest) — Lançamento e navegação."""

import os
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from basketball_eval import auth, db, qual_service as svc
from basketball_eval import service as base

ROOT = os.path.dirname(os.path.dirname(__file__))
PAGE = os.path.join(ROOT, "basketball_eval", "ui", "lancamento.py")
HOME = os.path.join(ROOT, "basketball_eval", "home.py")


@pytest.fixture
def dbpath(tmp_path, monkeypatch):
    path = str(tmp_path / "ui.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def run(path=PAGE):
    at = AppTest.from_file(path, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_empty_state_prompts_to_add_player(dbpath):
    at = run()
    assert any("Selecione ou adicione" in i.value for i in at.info)


def test_full_flow_save_then_all_views(dbpath):
    pid = base.add_player("Ana", "Sub-10", team="Águias")
    at = run()
    assert at.session_state["ctx_player"] == pid
    for key, val in [("sc_equilibrio_corporal", 4), ("sc_posicao_pes", 3), ("sc_repetir_gesto", 2)]:
        next(w for w in at.get("segmented_control") if w.key == key).set_value(val)
    at.run()
    assert not at.exception
    next(t for t in at.text_area if t.key == "f_obs").set_value("Boa base")
    at.text_input(key="f_coach").set_value("Rui")
    next(b for b in at.button if b.label == "Guardar avaliação").click()
    at.run()
    assert not at.exception and any("guardada" in s.value for s in at.success)
    evs = svc.list_evaluations(pid, "lancamento")
    assert len(evs) == 1 and evs[0]["scores"]["posicao_pes"] == 3 and evs[0]["coach_name"] == "Rui"
    # formulário limpo após guardar
    assert all(at.session_state[f"sc_{k}"] is None for k in ("posicao_pes", "repetir_gesto"))
    # uma segunda avaliação para ativar evolução/comparação
    svc.save_evaluation("lancamento", pid, "2026-01-15", {"posicao_pes": 5, "repetir_gesto": 4})
    for view in ["Perfil", "Evolução", "Comparar", "Histórico", "Relatório", "Avaliar"]:
        at.session_state["view"] = view
        at.run()
        assert not at.exception, (view, [e.value for e in at.exception])


def test_saving_without_scores_shows_error(dbpath):
    base.add_player("Ana", "Sub-10")
    at = run()
    next(b for b in at.button if b.label == "Guardar avaliação").click()
    at.run()
    assert any("pelo menos um critério" in e.value for e in at.error)


def test_navigation_entrypoint_loads(dbpath):
    run(HOME)


# ── Acesso por palavra-passe ────────────────────────────────────────────────
def test_remote_db_without_password_refuses_to_start(dbpath, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x@invalido:5432/x")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(auth, "has_users", lambda *a, **k: False)     # BD remota sem contas configuradas
    at = run(HOME)
    assert any("APP_PASSWORD" in e.value for e in at.error)
    assert not at.get("segmented_control")             # nada da app foi mostrado


def test_password_gate_blocks_then_allows(dbpath, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "frase-secreta")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("time.sleep", lambda s: None)
    at = run(HOME)
    assert at.text_input[0].proto.type == 1 and not at.get("segmented_control")   # 1 = campo de palavra-passe
    at.text_input[0].set_value("errada")
    at.button[0].click()
    at.run()
    assert any("incorreta" in e.value for e in at.error)
    assert "auth_ok" not in at.session_state
    at.text_input[0].set_value("frase-secreta")
    at.button[0].click()
    at.run()
    assert not at.exception and at.session_state["auth_ok"] is True


# ── Contas individuais ──────────────────────────────────────────────────────
PW = "uma-frase-longa-1"


def login(at, user, pw):
    at.text_input[0].set_value(user)
    at.text_input[1].set_value(pw)
    at.button[0].click()
    at.run()


@pytest.fixture
def accounts(dbpath, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-temporaria-1")
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_accounts_login_forced_change_then_app(accounts):
    at = run(HOME)                                      # cria o administrador e mostra o login
    assert [t.label for t in at.text_input] == ["Utilizador", "Palavra-passe"]
    login(at, "admin", "errada-errada")
    assert any("incorretos" in e.value for e in at.error)
    login(at, "admin", "admin-temporaria-1")
    assert not at.exception and any("defina agora" in i.value for i in at.info)      # troca obrigatória
    at.text_input[0].set_value("admin-temporaria-1")
    at.text_input[1].set_value("nova-frase-longa-9")
    at.text_input[2].set_value("nova-frase-longa-9")
    at.button[0].click()
    at.run()
    assert not at.exception and all("defina agora" not in i.value for i in at.info)
    assert at.session_state["auth_user"]["role"] == "admin"


def test_coach_uses_own_name_and_actions_are_audited(accounts):
    pid = base.add_player("Ana", "Sub-10", team="Águias")
    auth.create_user("rui", "Rui Costa", PW, must_change=False)
    at = run(HOME)
    login(at, "rui", PW)
    assert not at.exception and at.session_state["auth_user"]["username"] == "rui"
    coach = next(t for t in at.text_input if t.label == "Treinador")
    assert coach.value == "Rui Costa" and coach.disabled
    next(w for w in at.get("segmented_control") if w.key == "sc_posicao_pes").set_value(4)
    at.run()
    next(b for b in at.button if b.label == "Guardar avaliação").click()
    at.run()
    assert not at.exception
    ev = svc.list_evaluations(pid, "lancamento")[0]
    assert ev["coach_name"] == "Rui Costa"
    log = auth.list_audit()
    assert log[0]["username"] == "rui" and log[0]["action"] == "criar_avaliacao" and log[0]["evaluation_id"] == ev["id"]


def test_coach_cannot_open_admin_page_and_deactivated_user_is_kicked_out(accounts):
    auth.create_user("rui", "Rui", PW, must_change=False)
    admin = auth.create_user("boss", "Boss", PW, role="admin", must_change=False)
    at = run(HOME)
    login(at, "rui", PW)
    admin_page = AppTest.from_file(os.path.join(ROOT, "basketball_eval", "ui", "utilizadores.py"))
    admin_page.session_state["auth_user"] = at.session_state["auth_user"]    # sessão de treinador
    admin_page.run()
    assert any("reservado a administradores" in e.value for e in admin_page.error)
    auth.set_active(auth.list_users()[1].id, False)                          # administrador desativa o treinador
    at.run()
    assert [t.label for t in at.text_input] == ["Utilizador", "Palavra-passe"]    # voltou ao login


def test_admin_page_creates_account(accounts):
    at = run(HOME)
    login(at, "admin", "admin-temporaria-1")
    at.text_input[0].set_value("admin-temporaria-1"); at.text_input[1].set_value("nova-frase-longa-9")
    at.text_input[2].set_value("nova-frase-longa-9"); at.button[0].click(); at.run()
    page = AppTest.from_file(os.path.join(ROOT, "basketball_eval", "ui", "utilizadores.py"), default_timeout=30)
    page.session_state["auth_user"] = at.session_state["auth_user"]
    page.run()
    assert not page.exception
    page.text_input[0].set_value("ana.silva"); page.text_input[1].set_value("Ana Silva")
    page.text_input[2].set_value("temporaria-123456")
    next(b for b in page.button if b.label == "Criar conta").click()
    page.run()
    assert not page.exception and any(u.username == "ana.silva" and u.must_change_password for u in auth.list_users())
