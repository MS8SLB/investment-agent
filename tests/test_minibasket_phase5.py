"""Fase 5 — histórico e evolução individual."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from ui_helper import logged_in_app  # noqa: E402

from minibasket import calc, db, evaluations as ev, evolution, service
from minibasket import competencies as comp


def sc(*vals):
    return dict(zip(comp.KEYS, vals))


# Exemplo do enunciado: Setembro 2,44 · Dezembro 3,00 · Março 3,44 · Junho 4,00
SEP, DEC, MAR, JUN = (sc(2, 3, 2, 3, 2, 2, 3, 2, 3), sc(3, 3, 3, 3, 3, 3, 3, 3, 3),
                      sc(3, 4, 3, 4, 3, 3, 4, 3, 4), sc(4, 4, 4, 4, 4, 4, 4, 4, 4))


@pytest.fixture
def w(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    club = service.create_club("C", db_path=p)
    t8 = service.create_team(club, "S8", "Sub-8", "2024/2025", db_path=p)
    t10 = service.create_team(club, "S10", "Sub-10", "2024/2025", db_path=p)
    pid = service.create_player("João", t8, joined_on="2024-01-01", db_path=p)
    return {"p": p, "pid": pid, "t8": t8, "t10": t10}


def hist(w, *items):
    for d, m, s in items:
        ev.create_evaluation(w["pid"], d, m, s, db_path=w["p"])
    return ev.list_evaluations(w["pid"], db_path=w["p"])


# ── jogador sem avaliações / com uma ────────────────────────────────────────
def test_no_evaluations(w):
    h = ev.list_evaluations(w["pid"], db_path=w["p"])
    assert evolution.evolution_series(h) == [] and evolution.overall_change(h) is None
    assert all(r["n_rated"] == 0 and r["change"] is None for r in evolution.competency_table(h))


def test_single_evaluation(w):
    h = hist(w, ("2024-09-15", "Avaliação Inicial", DEC))
    pts = evolution.evolution_series(h)
    assert len(pts) == 1 and pts[0]["delta"] is None and pts[0]["average"] == 3.0
    assert evolution.overall_change(h) is None
    assert evolution.competency_table(h)[0]["change"] is None and evolution.competency_table(h)[0]["last"] == 3


# ── várias avaliações ───────────────────────────────────────────────────────
def test_series_with_several_evaluations(w):
    h = hist(w, ("2024-12-15", "2.º Período", DEC), ("2024-09-15", "Avaliação Inicial", sc(2, 3, 2, 3, 2, 2, 3, 2, 3)),
             ("2025-06-15", "Avaliação Final", JUN), ("2025-03-15", "3.º Período", MAR))
    pts = evolution.evolution_series(h)
    assert [p["date"] for p in pts] == ["2024-09-15", "2024-12-15", "2025-03-15", "2025-06-15"]
    assert [calc.fmt(p["average"]) for p in pts] == ["2,44", "3,00", "3,44", "4,00"]
    assert pts[0]["delta"] is None
    assert [calc.fmt(p["delta"], signed=True) for p in pts[1:]] == ["+0,56", "+0,44", "+0,56"]
    ch = evolution.overall_change(h)
    assert ch["n_evaluations"] == 4 and calc.fmt(ch["delta"], signed=True) == "+1,56"


def test_competency_series_example_from_spec(w):
    # Evolução do Lançamento: Setembro 2, Dezembro 3, Março 3, Junho 4
    h = hist(w, ("2024-09-15", "Avaliação Inicial", {**DEC, "shooting": 2}), ("2024-12-15", "2.º Período", DEC),
             ("2025-03-15", "3.º Período", {**DEC, "shooting": 3}), ("2025-06-15", "Avaliação Final", {**DEC, "shooting": 4}))
    s = evolution.competency_series(h, "shooting")
    assert [p["score"] for p in s] == [2, 3, 3, 4] and [p["delta"] for p in s] == [None, 1, 0, 1]
    row = evolution.competency_table(h)[0]
    assert row["scores"] == [2, 3, 3, 4] and row["change"] == 2 and (row["first"], row["last"]) == (2, 4)
    with pytest.raises(ValueError):
        evolution.competency_series(h, "nonsense")


def test_decrease_is_reported_neutrally_as_negative_delta(w):
    h = hist(w, ("2024-09-15", "1.º Período", DEC), ("2024-12-15", "2.º Período", {**DEC, "passing": 2}))
    assert evolution.competency_series(h, "passing")[1]["delta"] == -1


# ── avaliação incompleta ────────────────────────────────────────────────────
def test_incomplete_evaluation_flagged_and_compared_like_for_like(w):
    h = hist(w, ("2024-09-15", "1.º Período", sc(2, 2, 2, 2, 2, 2, 2, 2, 2)),
             ("2024-12-15", "2.º Período", {"shooting": 4, "dribbling": 4}))
    p0, p1 = evolution.evolution_series(h)
    assert p1["complete"] is False and p1["n_rated"] == 2 and p1["average"] == 4.0
    assert p1["n_common"] == 2 and p1["delta"] == 2.0               # só as 2 competências em comum
    assert p0["complete"] is True


def test_competency_delta_skips_unrated_gaps(w):
    h = hist(w, ("2024-09-15", "1.º Período", DEC), ("2024-10-15", "Personalizada", {"shooting": None, "passing": 3}),
             ("2024-12-15", "2.º Período", {**DEC, "shooting": 4}))
    s = evolution.competency_series(h, "shooting")
    assert [p["score"] for p in s] == [3, None, 4] and [p["delta"] for p in s] == [None, None, 1]
    assert evolution.competency_table(h)[0]["n_rated"] == 2


# ── alteração de uma avaliação ──────────────────────────────────────────────
def test_corrected_evaluation_replaces_old_in_evolution_but_is_kept(w):
    a = ev.create_evaluation(w["pid"], "2024-09-15", "Avaliação Inicial", DEC, db_path=w["p"])
    ev.create_evaluation(w["pid"], "2024-12-15", "2.º Período", JUN, db_path=w["p"])
    ev.correct_evaluation(a, "2024-09-15", "Avaliação Inicial", sc(2, 2, 2, 2, 2, 2, 2, 2, 2), db_path=w["p"])
    h = ev.list_evaluations(w["pid"], db_path=w["p"])
    pts = evolution.evolution_series(h)
    assert [p["average"] for p in pts] == [2.0, 4.0] and len(h) == 2           # evolução usa a versão corrigida
    assert len(ev.list_evaluations(w["pid"], include_superseded=True, db_path=w["p"])) == 3   # original preservada


# ── mudança de escalão ──────────────────────────────────────────────────────
def test_history_survives_category_change(w):
    ev.create_evaluation(w["pid"], "2024-09-15", "Avaliação Inicial", SEP, db_path=w["p"])
    service.change_team(w["pid"], w["t10"], "2024-12-01", db_path=w["p"])
    ev.create_evaluation(w["pid"], "2024-12-15", "2.º Período", DEC, db_path=w["p"])
    pts = evolution.evolution_series(ev.list_evaluations(w["pid"], db_path=w["p"]))
    assert [p["category"] for p in pts] == ["Sub-8", "Sub-10"] and pts[1]["delta"] is not None


# ── UI ──────────────────────────────────────────────────────────────────────
def test_ui_history_and_versions(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    club = service.create_club("C")
    team = service.create_team(club, "S10", "Sub-10", "2024/2025")
    pid = service.create_player("João", team, joined_on="2024-01-01")
    at = logged_in_app()
    at.sidebar.radio[0].set_value("Evolução do Jogador").run()
    at.radio(key="evo_cat").set_value("Sub-10").run()
    assert not at.exception and not at.dataframe                       # sem avaliações

    a = ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", sc(2, 3, 2, 3, 2, 2, 3, 2, 3))
    at.run()
    assert not at.exception and len(at.dataframe) == 2
    assert any("segunda avaliação" in c.value for c in at.caption)

    ev.create_evaluation(pid, "2024-12-15", "2.º Período", DEC)
    ev.correct_evaluation(a, "2024-09-15", "Avaliação Inicial", sc(2, 3, 2, 3, 2, 2, 3, 2, 3), general_notes="v2")
    at.run()
    assert not at.exception
    hist_df = next(d.value for d in at.dataframe if "Média" in d.value.columns)
    assert list(hist_df["Média"]) == ["2,44", "3,00"] and list(hist_df["Variação"]) == ["—", "+0,56"]
    assert any("2,44 → 3,00" in m.value for m in at.markdown)
    assert any(e.label.startswith("Versões anteriores corrigidas (1)") for e in at.expander)
