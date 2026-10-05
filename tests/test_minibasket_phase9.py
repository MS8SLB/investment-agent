"""Fase 9 — relatório para os pais."""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from ui_helper import logged_in_app  # noqa: E402

from minibasket import db, evaluations as ev, reports, service
from minibasket import competencies as comp

BANNED = re.compile(r"mau jogador|fraco|fraca|sem capacidade|incapaz|péssim|inferior|ranking|pior|dificuldade|"
                    r"regress|descid|falh|lacuna|défice|deficit", re.I)


def flat(n, **over):
    return {**{k: n for k in comp.KEYS}, **over}


@pytest.fixture
def w(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    club = service.create_club("Clube", db_path=p)
    t = service.create_team(club, "S10", "Sub-10", "2024/2025", db_path=p)
    return {"p": p, "t": t}


def mk(w, name="João Silva", sex="M"):
    return service.create_player(name, w["t"], sex=sex, joined_on="2024-01-01", db_path=w["p"])


def evaluate(w, pid, date, scores, **kw):
    return ev.create_evaluation(pid, date, "Personalizada", scores, db_path=w["p"], **kw)


def parent(w, eid):
    return reports.parent_report(eid, w["p"])


# ── as quatro secções pedidas ───────────────────────────────────────────────
def test_section_titles(w):
    r = parent(w, evaluate(w, mk(w), "2024-09-15", flat(3, footwork=2)))
    assert [s["title"] for s in r["sections"].values()] == [
        "Como está a evoluir?", "Os seus pontos fortes", "O que estamos a trabalhar", "Objetivos para a próxima etapa"]


def test_spec_example_sentences(w):
    pid = mk(w)
    evaluate(w, pid, "2024-09-15", flat(3, dribbling=2, individual_defense=3, footwork=2, individual_tactics=2))
    cur = evaluate(w, pid, "2024-12-15", flat(3, dribbling=4, individual_defense=4, footwork=2, individual_tactics=2))
    s = parent(w, cur)["sections"]
    assert s["evolution"]["text"] == ("O João apresentou uma evolução positiva ao longo deste período, "
                                      "particularmente no domínio da bola e na defesa individual.")
    assert s["working"]["text"] == "Estamos a trabalhar principalmente o trabalho de pés, a tática individual (tomada de decisão) e o passe."
    assert s["goals"]["text"].startswith("Para o próximo período, o objetivo será continuar a desenvolver o trabalho de pés e a tática individual")


def test_strengths_text(w):
    r = parent(w, evaluate(w, mk(w), "2024-09-15", flat(2, dribbling=5, passing=4, reception=4, finishing=4)))
    assert r["sections"]["strengths"]["text"] == "O João destaca-se, neste momento, no domínio da bola, no passe e na receção da bola."
    assert r["sections"]["strengths"]["items"] == ["dribbling", "passing", "reception"]


# ── nome / sexo ─────────────────────────────────────────────────────────────
def test_article_depends_on_recorded_sex_only(w):
    evs = {}
    for name, sex in (("Ana Costa", "F"), ("Rui Pires", "M"), ("Alex Reis", None)):
        pid = mk(w, name, sex)
        evaluate(w, pid, "2024-09-15", flat(2))
        evs[name] = evaluate(w, pid, "2024-12-15", flat(2, shooting=3))
    assert parent(w, evs["Ana Costa"])["sections"]["evolution"]["text"].startswith("A Ana apresentou")
    assert parent(w, evs["Rui Pires"])["sections"]["evolution"]["text"].startswith("O Rui apresentou")
    assert parent(w, evs["Alex Reis"])["sections"]["evolution"]["text"].startswith("Alex apresentou")   # sem adivinhar


# ── evolução ────────────────────────────────────────────────────────────────
def test_first_evaluation(w):
    r = parent(w, evaluate(w, mk(w), "2024-09-15", flat(3, footwork=2)))
    assert "primeira avaliação" in r["sections"]["evolution"]["text"]
    assert r["wheel"]["previous_scores"] is None


def test_maintained_level(w):
    pid = mk(w)
    evaluate(w, pid, "2024-09-15", flat(3, footwork=2))
    r = parent(w, evaluate(w, pid, "2024-12-15", flat(3, footwork=2)))
    assert "manteve o seu nível" in r["sections"]["evolution"]["text"]
    assert r["wheel"]["previous_scores"] is not None


def test_decrease_is_framed_neutrally_and_previous_wheel_hidden(w):
    pid = mk(w)
    evaluate(w, pid, "2024-09-15", flat(4))
    r = parent(w, evaluate(w, pid, "2024-12-15", flat(3)))
    assert "esteve a consolidar" in r["sections"]["evolution"]["text"]
    assert r["wheel"]["previous_scores"] is None                         # a roda comparativa só se mostra se não recuou
    assert not BANNED.search(json.dumps(r, ensure_ascii=False))


def test_previous_wheel_shown_on_improvement(w):
    pid = mk(w)
    evaluate(w, pid, "2024-09-15", flat(2))
    r = parent(w, evaluate(w, pid, "2024-12-15", flat(3)))
    assert r["wheel"]["previous_scores"] == flat(2) and r["wheel"]["previous_date"] == "2024-09-15"


# ── casos limite ────────────────────────────────────────────────────────────
def test_balanced_profile_and_few_scores(w):
    r = parent(w, evaluate(w, mk(w), "2024-09-15", flat(3)))
    s = r["sections"]
    assert "perfil equilibrado" in s["strengths"]["text"] and "de forma equilibrada" in s["working"]["text"]
    assert "todas as competências" in s["goals"]["text"]
    one = parent(w, evaluate(w, mk(w, "Zé Luís"), "2024-09-15", {"shooting": 4}))
    assert "dados suficientes" in one["sections"]["strengths"]["text"] and one["incomplete_note"]
    assert len(one["skills"]) == 1


def test_goals_from_coach_take_precedence(w):
    r = parent(w, evaluate(w, mk(w), "2024-09-15", flat(3, footwork=2),
                           next_objectives="Melhorar a qualidade das decisões e a utilização do espaço."))
    assert r["sections"]["goals"] == {"title": "Objetivos para a próxima etapa", "from_coach": True,
                                      "text": "Melhorar a qualidade das decisões e a utilização do espaço."}


def test_skills_visual_dots_and_levels(w):
    r = parent(w, evaluate(w, mk(w), "2024-09-15", flat(3, shooting=5, footwork=1)))
    by = {s["key"]: s for s in r["skills"]}
    assert by["shooting"]["dots"] == "●●●●●" and by["shooting"]["level"] == "Muito bom"
    assert by["footwork"]["dots"] == "●○○○○" and by["footwork"]["level"] == "Inicial"
    assert len(r["skills"]) == 9


# ── privacidade e linguagem ─────────────────────────────────────────────────
def test_no_internal_notes_team_data_or_other_players(w):
    pid, other = mk(w), mk(w, "Maria Outra", "F")
    evaluate(w, other, "2024-09-15", flat(5))
    evaluate(w, mk(w, "Terceiro Jogador"), "2024-09-15", flat(1))
    eid = evaluate(w, pid, "2024-09-15", flat(3, footwork=2), general_notes="NOTA-INTERNA-GERAL",
                   notes={"shooting": "NOTA-INTERNA-COMPETENCIA instabilidade"}, parent_message="Muito empenho nos treinos!")
    r = parent(w, eid)
    text = json.dumps(r, ensure_ascii=False)
    for forbidden in ("NOTA-INTERNA-GERAL", "NOTA-INTERNA-COMPETENCIA", "instabilidade", "Maria Outra", "Terceiro",
                      "team_mean", "average", "stats", "median", "rank", "attention"):
        assert forbidden not in text, forbidden
    assert r["message"] == "Muito empenho nos treinos!"
    assert not BANNED.search(text)


def test_lowest_levels_stay_positive(w):
    r = parent(w, evaluate(w, mk(w), "2024-09-15", flat(1, shooting=2)))
    assert not BANNED.search(json.dumps(r, ensure_ascii=False))
    assert r["sections"]["strengths"]["text"].startswith("O João destaca-se")


def test_parent_report_unknown_evaluation(w):
    with pytest.raises(service.ValidationError):
        reports.parent_report(999, w["p"])


# ── mensagem para os pais: persistência e migração ──────────────────────────
def test_parent_message_stored_and_carried_by_correction(w):
    pid = mk(w)
    a = evaluate(w, pid, "2024-09-15", flat(3), parent_message="  Parabéns!  ")
    assert ev.get_evaluation(a, w["p"])["parent_message"] == "Parabéns!"
    b = ev.correct_evaluation(a, "2024-09-15", "Personalizada", flat(3), parent_message="Parabéns pelo esforço!",
                              db_path=w["p"])
    assert ev.get_evaluation(b, w["p"])["parent_message"] == "Parabéns pelo esforço!"
    assert ev.get_evaluation(a, w["p"])["parent_message"] == "Parabéns!"             # versão original intacta


def test_migration_adds_column_to_existing_db(tmp_path):
    p = str(tmp_path / "old.db")
    with db.connect(p) as c:
        c.executescript(db.SCHEMA.replace(
            "    parent_message  TEXT,                         -- mensagem escrita para os encarregados de educação\n", ""))
        assert "parent_message" not in {r["name"] for r in c.execute("PRAGMA table_info(evaluations)")}
    db.init_db(p)
    with db.connect(p) as c:
        assert "parent_message" in {r["name"] for r in c.execute("PRAGMA table_info(evaluations)")}
    db.init_db(p)                                                                      # idempotente


# ── UI ──────────────────────────────────────────────────────────────────────
def test_ui_parent_tab_and_message_field(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    club = service.create_club("C")
    t = service.create_team(club, "S10", "Sub-10", "2024/2025")
    pid = service.create_player("João Silva", t, sex="M", joined_on="2024-01-01")
    ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", flat(2))
    ev.create_evaluation(pid, "2024-12-15", "2.º Período", flat(3, footwork=2), general_notes="INTERNA",
                         parent_message="Muito empenho!")
    at = logged_in_app()
    at.sidebar.radio[0].set_value("Relatórios").run()
    at.radio(key="rep_p_cat").set_value("Sub-10").run()
    assert not at.exception
    assert any(t_.label == "Pais" for t_ in at.tabs)
    text = " ".join(x.value for kind in (at.markdown, at.caption, at.info) for x in kind)
    for needle in ("Como está a evoluir?", "Os seus pontos fortes", "O que estamos a trabalhar",
                   "Objetivos para a próxima etapa", "O João apresentou uma evolução positiva", "Muito empenho!"):
        assert needle in text, needle
    parent_text = text[text.index("João — como está a correr"):]
    assert "INTERNA" not in parent_text and not BANNED.search(parent_text)

    at.sidebar.radio[0].set_value("Avaliar").run()                                      # campo no formulário
    at.radio(key="eval_cat").set_value("Sub-10").run()
    assert not at.exception and any(t_.label.startswith("Mensagem para os pais") for t_ in at.text_area)
