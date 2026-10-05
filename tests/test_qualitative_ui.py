"""Teste de fumo da interface Streamlit (AppTest) — Lançamento e navegação."""

import os
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from basketball_eval import db, qual_service as svc
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
