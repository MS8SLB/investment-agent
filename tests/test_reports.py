"""Testes: nota percentil, folha/evolução de equipa e relatórios."""

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from basketball_eval import norms, reports, service
from basketball_eval.db import init_db


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "t.db")
    init_db(path)
    norms.load_matulaitis_2019(path)
    return path


def test_percentile_score_interpolates_and_clamps(db):
    assert norms.percentile_score(10.43, 8, None, db)["score"] == pytest.approx(50)
    mid = norms.percentile_score((10.43 + 10.5) / 2, 8, None, db)["score"]
    assert 40 < mid < 50
    assert norms.percentile_score(8.0, 8, None, db) == {"score": 90, "bound": "above", "age": 8}
    assert norms.percentile_score(15.0, 8, None, db)["bound"] == "below"
    assert norms.percentile_score(10.0, 5, None, db) is None


def _seed(db):
    a = service.add_player("Ana", "Sub-10", "F", "Clube X", date(2016, 3, 1), db)
    b = service.add_player("Rui", "Sub-10", "M", "Clube X", date(2016, 6, 1), db)
    for pid, d, t in ((a, date(2026, 1, 10), "11.00"), (a, date(2026, 6, 10), "10.00"),
                      (b, date(2026, 1, 10), "12.00"), (b, date(2026, 6, 10), "11.50")):
        service.save_test(pid, d, [t, None, None], db_path=db)
    return a, b


def test_team_sheet_and_evolution(db):
    _seed(db)
    sh = service.team_sheet("Sub-10", date(2026, 6, 1), date(2026, 6, 30), db_path=db)
    assert [r["name"] for r in sh["rows"]] == ["Ana", "Rui"]
    assert sh["fastest"] == 10.0 and sh["slowest"] == 11.5 and sh["mean"] == pytest.approx(10.75)
    assert all(r["age"] == 10 and r["percentile"] is not None for r in sh["rows"])
    evo = service.team_evolution("Sub-10", db_path=db)
    assert [e["date"] for e in evo] == ["2026-01-10", "2026-06-10"]
    assert evo[1]["mean"] < evo[0]["mean"]


def test_reports(db):
    a, _ = _seed(db)
    assert "Ana" in reports.coach_report(a, db)
    pr = reports.parent_report(a, db)
    assert "PAIS" in pr and "10,00 s" in pr and "percentil" not in pr.lower()
    assert "Ana" in reports.team_report("Sub-10", date(2026, 6, 1), date(2026, 6, 30), db_path=db)
    assert "&lt;" in reports.to_html("t", "<b>")
