"""Fase 1 — estrutura, competências, escala e esquema da BD."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minibasket import competencies as comp
from minibasket import db, scale


@pytest.fixture
def path(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    return p


def test_nine_competencies_in_order():
    assert len(comp.COMPETENCIES) == 9
    assert [c.position for c in comp.COMPETENCIES] == list(range(1, 10))
    assert comp.name_of("reception") == "Receção da Bola"
    assert comp.name_of("fast_break") == "Contra-Ataque"


def test_default_scale_is_1_to_5():
    assert scale.bounds() == (1, 5)
    assert scale.label(1) == "Inicial"
    assert scale.label(5) == "Muito bom"
    with pytest.raises(ValueError):
        scale.label(6)


def test_init_db_seeds_and_is_idempotent(path):
    db.init_db(path)
    with db.connect(path) as c:
        assert c.execute("SELECT COUNT(*) FROM competencies").fetchone()[0] == 9
        assert c.execute("SELECT COUNT(*) FROM scales").fetchone()[0] == 1
    cfg = db.active_scale(path)
    assert [v for v, _ in cfg["levels"]] == [1, 2, 3, 4, 5]


def test_categories_only_sub8_10_12(path):
    assert db.CATEGORIES == ("Sub-8", "Sub-10", "Sub-12")
    with db.connect(path) as c:
        c.execute("INSERT INTO clubs(name) VALUES ('C')")
        with pytest.raises(Exception):
            c.execute("INSERT INTO teams(club_id,name,category,season) VALUES (1,'X','Sub-14','2026/27')")


def test_no_stored_average_column(path):
    with db.connect(path) as c:
        cols = {r["name"] for t in ("evaluations", "evaluation_scores")
                for r in c.execute(f"PRAGMA table_info({t})")}
    assert not any("average" in x or "mean" in x or "media" in x for x in cols)


def test_foreign_keys_enforced(path):
    with db.connect(path) as c:
        with pytest.raises(Exception):
            c.execute("INSERT INTO team_memberships(player_id,team_id) VALUES (999,999)")
