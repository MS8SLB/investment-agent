"""Testes da avaliação qualitativa da defesa (basketball_eval)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from basketball_eval import qualitative as ql
from basketball_eval import service
from basketball_eval.db import init_db


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "t.db")
    init_db(path)
    return path


def all_rated(v):
    return {k: v for k in ql.CRITERIA}


def test_descriptors_cover_scale():
    for c in ql.CRITERIA.values():
        assert set(c["descriptors"]) == set(ql.SCALE)


def test_summary_and_partial():
    s = ql.summarize({**all_rated(3), "drop_step": 1, "atitude": None})
    assert s.n_rated == len(ql.CRITERIA) - 1 and s.to_improve == ["drop_step"]
    assert ql.summarize({"drop_step": 2, "atitude": 4}).average == 3.0
    assert not ql.summarize({"drop_step": 2}).complete


@pytest.mark.parametrize("bad", [{}, {"x": 3}, {"drop_step": 5}, {"drop_step": True}, {"drop_step": None}])
def test_invalid(bad):
    with pytest.raises(ValueError):
        ql.validate_ratings(bad)


def test_compare_sign():
    assert ql.compare({"drop_step": 2, "atitude": 4}, {"drop_step": 3, "atitude": 3}) == {"drop_step": 1, "atitude": -1}


def test_save_and_report(db):
    p = service.add_player("Rui", "Sub-12", db_path=db)
    service.save_qualitative(p, "2026-01-10", {"drop_step": 2, "atitude": 4}, db_path=db)
    service.save_qualitative(p, "2026-03-10", {"drop_step": 3, "atitude": 4}, "Empenho", db_path=db)
    h = service.qualitative_history(p, db)
    assert [r["average"] for r in h] == [3.0, 3.5] and h[0]["ratings"]["drop_step"] == 2
    txt = service.qualitative_report_text(p, db)
    assert "Melhorou: Drop step" in txt and "Empenho" in txt
    with pytest.raises(ValueError):
        service.save_qualitative(999, "2026-01-10", {"atitude": 3}, db_path=db)
