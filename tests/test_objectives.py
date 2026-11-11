"""Testes dos objetivos individuais (basketball_eval)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from basketball_eval import objectives as ob
from basketball_eval import qualitative as ql
from basketball_eval import service
from basketball_eval.db import init_db


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "t.db")
    init_db(path)
    return path


def test_library_links_valid():
    for o in ob.LIBRARY.values():
        for c in o.get("criteria", []):
            assert c in ql.CRITERIA


def test_no_data_keeps_library_order():
    s = ob.suggest_objectives("def")
    assert [o["key"] for o in s] == ob.DEFAULT_ORDER["def"] and all(o["need"] == 0 for o in s)


def test_qualitative_drives_priority():
    q = {"deslize_lateral": 1, "drop_step": 1, "velocidade_ritmo": 2, "posicao_base": 4, "maos_bracos": 4,
         "cabeca_olhar": 4, "atitude": 4, "toque_marcadores": 4}
    s = ob.suggest_objectives("def", q)
    assert s[0]["key"] == "def_closeout" and s[0]["reasons"]


def test_movement_worse_raises_closeout():
    s = ob.suggest_objectives("def", None, -5.0)
    assert s[0]["key"] == "def_closeout"
    assert ob.suggest_objectives("def", None, 3.0)[0]["need"] == 0  # melhorou → sem necessidade


def test_objective_scores_drive_offense_and_gap():
    last = {"off_pnr": {"execution": 2, "frequency": 2, "application": 1, "evolution": 2},
            "off_scan": {"execution": 5, "frequency": 5, "application": 5, "evolution": 5}}
    assert ob.suggest_objectives("off", last_scores=last)[0]["key"] == "off_pnr"
    assert ob.application_gap({"execution": 5, "application": 2})
    assert not ob.application_gap({"execution": 4, "application": 3})


@pytest.mark.parametrize("bad", [{}, {"x": 3}, {"execution": 6}, {"execution": 0}, {"execution": True}])
def test_invalid_scores(bad):
    with pytest.raises(ValueError):
        ob.validate_scores(bad)


def test_service_roundtrip(db):
    pid = service.add_player("Ana", "Sub-14", db_path=db)
    service.save_qualitative(pid, "2026-01-10", {"deslize_lateral": 1, "drop_step": 2}, db_path=db)
    sug = service.objective_suggestions(pid, db_path=db)
    assert sug["def"][0]["key"] == "def_closeout"
    service.set_objectives(pid, [("def_closeout", "sugerido"), ("off_scan", "manual")], db_path=db)
    service.set_objectives(pid, [("def_closeout", "sugerido")], db_path=db)  # off_scan desativado
    assert [o["objective_key"] for o in service.get_objectives(pid, db_path=db)] == ["def_closeout"]
    n = service.save_objective_evaluation(
        pid, "2026-02-01", {"def_closeout": {"execution": 3, "application": 1}, "off_scan": {}}, db_path=db)
    assert n == 1
    h = service.objective_history(pid, db_path=db)
    assert h[0]["average"] == 2.0
    # a avaliação de objetivos passa a influenciar as sugestões (lacuna de aplicação)
    top = service.objective_suggestions(pid, db_path=db)["def"][0]
    assert top["key"] == "def_closeout" and any("aplica pouco" in r for r in top["reasons"])
    with pytest.raises(ValueError):
        service.set_objectives(pid, [("nao_existe", "manual")], db_path=db)
    with pytest.raises(ValueError):
        service.save_objective_evaluation(999, "2026-02-01", {"def_closeout": {"execution": 3}}, db_path=db)


# ── IA (sem chamadas reais: cliente simulado) ───────────────────────────────

from types import SimpleNamespace as NS

from basketball_eval import ai_coach


class FakeClient:
    def __init__(self, text, stop="end_turn"):
        self.calls, self._r = [], NS(stop_reason=stop, content=[NS(type="text", text=text)])
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self._r


def test_ai_payload_is_anonymised(db):
    pid = service.add_player("Maria Segredo", "Sub-12", team="Clube Secreto", birth_date="2014-05-05", db_path=db)
    service.save_qualitative(pid, "2026-01-10", {"drop_step": 2}, "nota com nome João", "outra nota", db_path=db)
    blob = str(service.ai_payload(pid, db_path=db))
    for secret in ("Maria", "Segredo", "Clube Secreto", "2014", "João", "nota"):
        assert secret not in blob
    assert "Sub-12" in blob


def test_ai_propose_filters_unknown_and_uses_fallbacks():
    raw = {"summary": "ok", "caveats": [], "objectives": [
        {"key": "def_closeout", "reason": "r", "drills": ["a"]},
        {"key": "inventado", "reason": "x", "drills": []},
        {"key": "def_closeout", "reason": "dup", "drills": []}]}
    fc = FakeClient(json.dumps(raw))
    out = ai_coach.propose({"n_objetivos_a_propor": 4}, fc)
    assert [o["key"] for o in out["objectives"]] == ["def_closeout"]
    kw = fc.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"


@pytest.mark.parametrize("client", [FakeClient("", "refusal"), FakeClient("{", "end_turn"),
                                    FakeClient("{}", "max_tokens")])
def test_ai_failures_raise(client):
    with pytest.raises(RuntimeError):
        ai_coach.propose({"n_objetivos_a_propor": 4}, client)


def test_plan_sheet_text():
    t = ob.plan_sheet_text("Ana", "Sub-14", ["off_scan", "def_closeout"],
                           {"def_closeout": {"execution": 4, "application": 2}})
    assert "OBJETIVOS OFENSIVOS" in t and "OBJETIVOS DEFENSIVOS" in t
    assert "[3.0/5]" in t and "Execução\t1–5" in t and "Reavaliar" in t
