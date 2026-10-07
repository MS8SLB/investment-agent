"""Fase 8 — relatórios do treinador (individual e da equipa)."""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from ui_helper import logged_in_app  # noqa: E402

from minibasket import calc, db, evaluations as ev, reports, service
from minibasket import competencies as comp
from minibasket.service import ValidationError

BANNED = re.compile(r"mau jogador|fraco|fraca|sem capacidade|incapaz|péssim|inferior|ranking|pior", re.I)
EXAMPLE = dict(zip(comp.KEYS, (3, 4, 3, 4, 2, 3, 4, 3, 4)))


def flat(n, **over):
    return {**{k: n for k in comp.KEYS}, **over}


@pytest.fixture
def w(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    club = service.create_club("Clube", db_path=p)
    t = service.create_team(club, "S10", "Sub-10", "2024/2025", db_path=p)
    pid = service.create_player("João Silva", t, joined_on="2024-01-01", db_path=p)
    return {"p": p, "t": t, "pid": pid}


def evaluate(w, date, scores, pid=None, **kw):
    return ev.create_evaluation(pid or w["pid"], date, "Personalizada", scores, db_path=w["p"], **kw)


# ── rank_areas ──────────────────────────────────────────────────────────────
def test_rank_areas_example_from_spec_values():
    a = reports.rank_areas(EXAMPLE)
    assert [k for k, _ in a["top"]] == ["dribbling", "reception", "individual_defense"]    # 4,4,4 — desempate pela roda
    assert [k for k, _ in a["bottom"]] == ["footwork", "shooting", "passing"]              # 2,3,3
    assert a["tie_top"] and a["tie_bottom"] and not a["balanced"]


def test_rank_areas_disjoint_and_ordered():
    vals = {k: v for k, v in zip(comp.KEYS, (5, 1, 4, 2, 3, 3, 3, 3, 3))}
    a = reports.rank_areas(vals)
    assert [v for _, v in a["top"]] == [5, 4, 3] and [v for _, v in a["bottom"]] == [1, 2, 3]
    assert not {k for k, _ in a["top"]} & {k for k, _ in a["bottom"]}


def test_rank_areas_balanced_few_and_empty():
    assert reports.rank_areas(flat(3))["balanced"] and reports.rank_areas(flat(3))["top"] == []
    assert reports.rank_areas({})["top"] == [] and reports.rank_areas({"shooting": 4})["top"] == []
    a = reports.rank_areas({"shooting": 4, "passing": 2, "footwork": 3})     # 3 avaliadas → 1 de cada
    assert [k for k, _ in a["top"]] == ["shooting"] and [k for k, _ in a["bottom"]] == ["passing"]
    b = reports.rank_areas({"shooting": 4, "passing": 2})
    assert len(b["top"]) == 1 and len(b["bottom"]) == 1


def test_rank_areas_handles_partial_ties_without_overlap():
    a = reports.rank_areas({"shooting": 3, "dribbling": 3, "passing": 3, "reception": 3, "footwork": 4})
    assert len(a["top"]) == 2 and len(a["bottom"]) == 2
    assert not {k for k, _ in a["top"]} & {k for k, _ in a["bottom"]}


# ── relatório individual ────────────────────────────────────────────────────
def test_individual_report_first_evaluation(w):
    eid = evaluate(w, "2024-09-15", EXAMPLE, general_notes="Bom início", next_objectives="Melhorar equilíbrio",
                   notes={"shooting": "Melhorou a preparação dos pés."})
    r = reports.individual_report(eid, w["p"])
    assert (r["player"]["name"], r["category"], r["team"], r["club"], r["date"]) == \
        ("João Silva", "Sub-10", "S10", "Clube", "2024-09-15")
    assert calc.fmt(r["average"]) == "3,33" and r["complete"] and r["evolution"] is None
    assert [x["score"] for x in r["results"]] == list(EXAMPLE.values())
    assert r["results"][0]["note"] == "Melhorou a preparação dos pés." and r["results"][1]["level"] == "Bom"
    assert [a["name"] for a in r["strengths"]] == ["Drible / Domínio da Bola", "Receção da Bola", "Defesa Individual"]
    assert r["to_develop"][0]["name"] == "Trabalho de Pés" and r["to_develop"][0]["level"] == "Em desenvolvimento"
    assert r["objectives"] == "Melhorar equilíbrio" and r["general_notes"] == "Bom início"
    assert r["scale_max"] == 5 and r["suggested_focus"][0] == "Trabalho de Pés"


def test_individual_report_with_previous_evaluation(w):
    evaluate(w, "2024-09-15", flat(2, passing=3, footwork=3), next_objectives="Trabalhar passe")
    cur = evaluate(w, "2024-12-15", flat(3, passing=3, footwork=2, shooting=4))
    e = reports.individual_report(cur, w["p"])["evolution"]
    assert e["previous_date"] == "2024-09-15" and e["previous_objectives"] == "Trabalhar passe"
    assert e["improved"][0] == "Lançamento" and "Passe" in e["maintained"] and e["to_consolidate"] == ["Trabalho de Pés"]
    assert e["avg_delta"] == pytest.approx(7 / 9)          # (27 − 20) / 9
    assert e["n_common"] == 9 and e["previous_average"] == pytest.approx(2 + 2 / 9)


def test_individual_report_previous_uses_only_in_force_and_earlier(w):
    a = evaluate(w, "2024-09-15", flat(2))
    ev.correct_evaluation(a, "2024-09-15", "Personalizada", flat(3), db_path=w["p"])      # versão corrigida
    later = evaluate(w, "2025-03-15", flat(4))
    r = reports.individual_report(later, w["p"])
    assert r["evolution"]["previous_average"] == 3.0                                       # versão em vigor, não a original
    # uma avaliação posterior não é «anterior» de uma mais antiga
    first = ev.list_evaluations(w["pid"], db_path=w["p"])[0]["id"]
    assert reports.individual_report(first, w["p"])["evolution"] is None


def test_individual_report_incomplete_and_balanced(w):
    eid = evaluate(w, "2024-09-15", {"shooting": 4, "passing": 4})
    r = reports.individual_report(eid, w["p"])
    assert not r["complete"] and len(r["missing"]) == 7 and r["average"] == 4.0
    assert r["balanced"] and r["strengths"] == [] and r["to_develop"] == []


def test_individual_report_errors(w):
    with pytest.raises(ValidationError):
        reports.individual_report(999, w["p"])


def test_individual_report_without_objectives_suggests_focus(w):
    r = reports.individual_report(evaluate(w, "2024-09-15", EXAMPLE), w["p"])
    assert r["objectives"] is None and len(r["suggested_focus"]) == 3


def test_individual_report_language_is_pedagogical(w):
    r = reports.individual_report(evaluate(w, "2024-09-15", flat(1, shooting=5)), w["p"])
    text = json.dumps(r, ensure_ascii=False)
    assert not BANNED.search(text)
    assert {x["level"] for x in r["to_develop"]} == {"Inicial"}                  # «Inicial», nunca «fraco»


def test_report_keeps_scale_used_in_evaluation(w):
    eid = evaluate(w, "2024-09-15", EXAMPLE)
    with db.connect(w["p"]) as c:                                                  # escala nova passa a ser a ativa
        c.execute("UPDATE scales SET active=0")
        sid = c.execute("INSERT INTO scales(name, active) VALUES ('Nova', 1)").lastrowid
        c.executemany("INSERT INTO scale_levels VALUES (?,?,?)", [(sid, v, f"N{v}") for v in range(1, 11)])
    r = reports.individual_report(eid, w["p"])
    assert r["scale_max"] == 5 and r["results"][1]["level"] == "Bom"


# ── relatório da equipa ─────────────────────────────────────────────────────
def players(w, n):
    return [service.create_player(f"Jogador{i}", w["t"], joined_on="2024-01-01", db_path=w["p"]) for i in range(n)]


def test_team_report_no_evaluations(w):
    r = reports.team_report(w["t"], w["p"])
    assert r["has_data"] is False and r["team"]["name"] == "S10" and r["evolution"] is None and r["stats"] == []
    with pytest.raises(ValidationError):
        reports.team_report(999, w["p"])


def test_team_report_single_date_has_no_evolution(w):
    for pid in players(w, 3):
        evaluate(w, "2024-09-15", EXAMPLE, pid=pid)
    r = reports.team_report(w["t"], w["p"])
    assert r["has_data"] and r["n_evaluated"] == 3 and r["members"] == 4 and r["evolution"] is None
    assert calc.fmt(r["average"]) == "3,33" and len(r["stats"]) == 9
    assert [a["name"] for a in r["attention"]][0] == "Trabalho de Pés"


def test_team_report_evolution_example_shape(w):
    ps = players(w, 4)
    for pid in ps:
        evaluate(w, "2024-09-15", flat(2), pid=pid)
    for pid in ps:
        evaluate(w, "2024-12-15", flat(3, dribbling=5, individual_defense=4, finishing=4, footwork=2, individual_tactics=2),
                 pid=pid)
    r = reports.team_report(w["t"], w["p"])
    e = r["evolution"]
    assert e["date_before"] == "2024-09-15" and e["date_after"] == "2024-12-15" and e["n_common"] == 4
    assert calc.fmt(e["avg_before"]) == "2,00" and calc.fmt(e["avg_after"]) == "3,22" and calc.fmt(e["avg_delta"], signed=True) == "+1,22"
    assert [a["name"] for a in e["most_improved"]] == ["Drible / Domínio da Bola", "Finalizações", "Defesa Individual"]   # empate: ordem da roda
    assert [a["delta"] for a in e["most_improved"]] == [3, 2, 2]
    assert [a["name"] for a in e["least_improved"]] == ["Trabalho de Pés", "Tática Individual", "Lançamento"]
    assert [a["delta"] for a in e["least_improved"]] == [0, 0, 1]
    assert not e["homogeneous"]


def test_team_report_homogeneous_evolution(w):
    ps = players(w, 3)
    for pid in ps:
        evaluate(w, "2024-09-15", flat(2), pid=pid); evaluate(w, "2024-12-15", flat(3), pid=pid)
    e = reports.team_report(w["t"], w["p"])["evolution"]
    assert e["homogeneous"] and e["most_improved"] == [] and e["avg_delta"] == pytest.approx(1.0)


def test_team_report_has_no_player_names_or_language_issues(w):
    for pid in players(w, 3):
        evaluate(w, "2024-09-15", EXAMPLE, pid=pid); evaluate(w, "2024-12-15", flat(4), pid=pid)
    text = json.dumps(reports.team_report(w["t"], w["p"]), ensure_ascii=False)
    assert "Jogador0" not in text and "player_id" not in text and not BANNED.search(text)


# ── UI ──────────────────────────────────────────────────────────────────────
def test_ui_reports(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    club = service.create_club("C")
    t = service.create_team(club, "S10", "Sub-10", "2024/2025", is_demo=True)
    at = logged_in_app()
    at.sidebar.radio[0].set_value("Relatórios").run()
    at.radio(key="rep_cat").set_value("Sub-10").run()
    at.radio(key="rep_t_cat").set_value("Sub-10").run()
    assert not at.exception and any("ainda não tem avaliações" in i.value for i in at.info)

    for n in ("Ana", "Rui", "Inês"):
        pid = service.create_player(n, t, joined_on="2024-01-01")
        ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", flat(2), general_notes="Início")
        ev.create_evaluation(pid, "2024-12-15", "2.º Período", EXAMPLE, next_objectives="Melhorar equilíbrio",
                             coach_id=ev.get_or_create_coach("Marta"))
    at.run()
    assert not at.exception
    text = " ".join(x.value for kind in (at.markdown, at.caption, at.info, at.success, at.warning) for x in kind)
    assert "Relatório de avaliação individual" in text and "Relatório da equipa" in text
    assert "MÉDIA GLOBAL" in [m.label for m in at.metric]
    assert "Áreas fortes" in text and "Áreas a desenvolver" in text and "Objetivos para o próximo período" in text
    assert "Marta" in text and "Melhorar equilíbrio" in text and "Evolução em:" in text
    assert "Maior evolução" in text and "Menor evolução" in text and "Dados de teste" in text
    assert not BANNED.search(text)
