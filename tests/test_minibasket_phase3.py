"""Fase 3 — avaliação das nove competências e média global."""

import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from ui_helper import logged_in_app  # noqa: E402

from minibasket import calc
from minibasket import competencies as comp
from minibasket import db, evaluations as ev, service
from minibasket.service import ValidationError

EXAMPLE = dict(zip(comp.KEYS, (3, 4, 3, 4, 2, 3, 4, 3, 4)))     # exemplo do enunciado: média 3,33


@pytest.fixture
def w(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    club = service.create_club("C", db_path=p)
    t8 = service.create_team(club, "S8", "Sub-8", "2026/2027", db_path=p)
    t10 = service.create_team(club, "S10", "Sub-10", "2026/2027", db_path=p)
    pid = service.create_player("João", t8, joined_on="2024-09-01", db_path=p)
    return {"p": p, "pid": pid, "t8": t8, "t10": t10, "club": club}


# ── Cálculo da média ────────────────────────────────────────────────────────
def test_average_example():
    assert calc.global_average(EXAMPLE) == pytest.approx(30 / 9)
    assert calc.fmt(calc.global_average(EXAMPLE)) == "3,33"


def test_average_other_examples_from_spec():
    assert calc.fmt(22 / 9) == "2,44"        # Setembro
    assert calc.fmt(27 / 9) == "3,00"        # Dezembro
    assert calc.fmt(31 / 9) == "3,44"        # Março
    assert calc.fmt(36 / 9) == "4,00"        # Junho


def test_average_incomplete_and_empty():
    partial = {"shooting": 4, "passing": 2, "footwork": None}
    assert calc.global_average(partial) == 3.0
    assert not calc.is_complete(partial)
    assert calc.global_average({}) is None and calc.fmt(None) == "—"
    assert calc.missing(partial)[0] == "dribbling" and len(calc.missing(partial)) == 7


def test_fmt_rounding_half_up_and_sign():
    assert calc.fmt(2.675) == "2,68" and calc.round2(2.675) == 2.68
    assert calc.fmt(0.8, signed=True) == "+0,80" and calc.fmt(-0.5, signed=True) == "-0,50"
    assert calc.fmt(3.2, decimals=1) == "3,2"


# ── Criação ─────────────────────────────────────────────────────────────────
def test_create_evaluation_complete(w):
    notes = {"shooting": "Melhorou a preparação dos pés."}
    eid = ev.create_evaluation(w["pid"], "2024-09-15", "Avaliação Inicial", EXAMPLE, notes,
                               general_notes="Bom início", next_objectives="Melhorar equilíbrio", db_path=w["p"])
    e = ev.get_evaluation(eid, w["p"])
    assert e["scores"] == EXAMPLE and e["complete"] and e["missing"] == []
    assert calc.fmt(e["average"]) == "3,33"
    assert e["notes"] == notes and e["category"] == "Sub-8" and e["team"] == "S8"
    assert e["moment"] == "Avaliação Inicial" and e["next_objectives"] == "Melhorar equilíbrio"
    assert e["scale_id"] == db.active_scale(w["p"])["id"]


def test_incomplete_evaluation_is_stored_flagged(w):
    eid = ev.create_evaluation(w["pid"], "2024-09-15", "1.º Período", {"shooting": 3, "passing": 5}, db_path=w["p"])
    e = ev.get_evaluation(eid, w["p"])
    assert not e["complete"] and e["average"] == 4.0 and len(e["missing"]) == 7
    assert e["scores"]["dribbling"] is None


def test_average_is_never_stored(w):
    ev.create_evaluation(w["pid"], "2024-09-15", "Avaliação Inicial", EXAMPLE, db_path=w["p"])
    with db.connect(w["p"]) as c:
        vals = [v for t in ("evaluations", "evaluation_scores") for r in c.execute(f"SELECT * FROM {t}") for v in tuple(r)]
    assert not any(isinstance(v, float) for v in vals)


def test_validation(w):
    p, pid = w["p"], w["pid"]
    ok = dict(player_id=pid, evaluation_date="2024-09-15", moment="1.º Período", scores=EXAMPLE, db_path=p)
    for bad in ({"scores": {**EXAMPLE, "shooting": 0}}, {"scores": {**EXAMPLE, "shooting": 6}},
                {"scores": {**EXAMPLE, "shooting": 3.5}}, {"scores": {**EXAMPLE, "shooting": True}},
                {"scores": {"nonsense": 3}}, {"scores": {}}, {"scores": {k: None for k in comp.KEYS}},
                {"moment": "Natal"}, {"evaluation_date": "2024-13-40"}, {"evaluation_date": None},
                {"evaluation_date": (date.today() + timedelta(days=1)).isoformat()},
                {"player_id": 999}, {"coach_id": 999}):
        with pytest.raises(ValidationError):
            ev.create_evaluation(**{**ok, **bad})
    with db.connect(p) as c:
        assert c.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0   # nada ficou gravado


def test_custom_date_and_moment(w):
    eid = ev.create_evaluation(w["pid"], "2024-11-07", "Personalizada", EXAMPLE, db_path=w["p"])
    assert ev.get_evaluation(eid, w["p"])["evaluation_date"] == "2024-11-07"


def test_coach_reuse(w):
    a = ev.get_or_create_coach("Ana Costa", w["p"])
    assert ev.get_or_create_coach(" ana costa ", w["p"]) == a
    assert ev.get_or_create_coach("Ana  Costa2", w["p"]) != a
    with pytest.raises(ValidationError):
        ev.get_or_create_coach(" ", w["p"])
    eid = ev.create_evaluation(w["pid"], "2024-09-15", "1.º Período", EXAMPLE, coach_id=a, db_path=w["p"])
    assert ev.get_evaluation(eid, w["p"])["coach"] == "Ana Costa"


# ── Histórico e correções (nunca apagar) ────────────────────────────────────
def test_history_chronological_and_no_overwrite(w):
    p, pid = w["p"], w["pid"]
    ev.create_evaluation(pid, "2024-12-15", "2.º Período", {k: 3 for k in comp.KEYS}, db_path=p)
    ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", {k: 2 for k in comp.KEYS}, db_path=p)
    h = ev.list_evaluations(pid, db_path=p)
    assert [e["evaluation_date"] for e in h] == ["2024-09-15", "2024-12-15"]
    assert [e["average"] for e in h] == [2.0, 3.0]
    assert ev.latest_evaluation(pid, p)["evaluation_date"] == "2024-12-15"


def test_player_without_evaluations(w):
    assert ev.list_evaluations(w["pid"], db_path=w["p"]) == [] and ev.latest_evaluation(w["pid"], w["p"]) is None


def test_correction_creates_new_version_keeping_old(w):
    p, pid = w["p"], w["pid"]
    first = ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", EXAMPLE, db_path=p)
    fixed = ev.correct_evaluation(first, "2024-09-15", "Avaliação Inicial", {**EXAMPLE, "footwork": 3},
                                  db_path=p)
    assert fixed != first
    current = ev.list_evaluations(pid, db_path=p)
    assert [e["id"] for e in current] == [fixed] and current[0]["supersedes_id"] == first
    allv = ev.list_evaluations(pid, include_superseded=True, db_path=p)
    assert [e["id"] for e in allv] == [first, fixed]
    old = ev.get_evaluation(first, p)
    assert old["scores"]["footwork"] == 2 and old["superseded_by"] == fixed        # original intacto
    with pytest.raises(ValidationError):
        ev.correct_evaluation(first, "2024-09-15", "Avaliação Inicial", EXAMPLE, db_path=p)  # já corrigida
    again = ev.correct_evaluation(fixed, "2024-09-15", "Avaliação Inicial", EXAMPLE, db_path=p)
    assert [e["id"] for e in ev.list_evaluations(pid, db_path=p)] == [again]
    with pytest.raises(ValidationError):
        ev.correct_evaluation(999, "2024-09-15", "Avaliação Inicial", EXAMPLE, db_path=p)


def test_cannot_supersede_other_players_evaluation(w):
    other = service.create_player("Rui", w["t8"], db_path=w["p"])
    e = ev.create_evaluation(w["pid"], "2024-09-15", "1.º Período", EXAMPLE, db_path=w["p"])
    with pytest.raises(ValidationError):
        ev.create_evaluation(other, "2024-09-16", "1.º Período", EXAMPLE, supersedes_id=e, db_path=w["p"])


# ── Mudança de escalão ──────────────────────────────────────────────────────
def test_category_follows_team_at_evaluation_date(w):
    p, pid = w["p"], w["pid"]
    e1 = ev.create_evaluation(pid, "2024-10-01", "1.º Período", EXAMPLE, db_path=p)
    service.change_team(pid, w["t10"], "2025-09-01", db_path=p)
    e2 = ev.create_evaluation(pid, "2025-09-15", "Avaliação Inicial", EXAMPLE, db_path=p)
    e3 = ev.create_evaluation(pid, "2025-02-01", "2.º Período", EXAMPLE, db_path=p)   # retroativa, ainda Sub-8
    assert [ev.get_evaluation(i, p)["category"] for i in (e1, e2, e3)] == ["Sub-8", "Sub-10", "Sub-8"]
    assert len(ev.list_evaluations(pid, db_path=p)) == 3               # histórico completo mantido


# ── UI ──────────────────────────────────────────────────────────────────────
def test_ui_evaluation_flow(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    club = service.create_club("C")
    team = service.create_team(club, "S8", "Sub-8", "2026/2027")
    pid = service.create_player("Maria", team, joined_on="2024-09-01")

    at = logged_in_app()
    at.sidebar.radio[0].set_value("Avaliar").run()
    assert not at.exception
    ctx = f"{pid}_new"
    for k, v in EXAMPLE.items():
        at.radio(key=f"s_{k}_{ctx}").set_value(v)
    at.run()
    assert not at.exception
    assert any(m.label == "MÉDIA GLOBAL" and m.value.startswith("3,33") for m in at.metric)
    at.button(key=f"save_{ctx}").click().run()
    assert not at.exception and any("3,33" in s.value for s in at.success)
    saved = ev.list_evaluations(pid)
    assert len(saved) == 1 and saved[0]["scores"] == EXAMPLE and saved[0]["coach"] == "Admin UI"        # o avaliador é a conta autenticada

    # incompleta: aviso visível e média só das avaliadas
    at.radio(key=f"s_shooting_{ctx}").set_value(0).run()
    assert any("incompleta" in w.value for w in at.warning)
