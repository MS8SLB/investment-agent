"""Fase 7 — evolução da equipa e Dashboard."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minibasket import calc, charts, db, evaluations as ev, service, teamstats
from minibasket import competencies as comp


def flat(n, **over):
    return {**{k: n for k in comp.KEYS}, **over}


@pytest.fixture
def w(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    club = service.create_club("C", db_path=p)
    t = service.create_team(club, "S10", "Sub-10", "2024/2025", db_path=p)
    t8 = service.create_team(club, "S8", "Sub-8", "2024/2025", db_path=p)
    return {"p": p, "t": t, "t8": t8, "club": club}


def player(w, name, team=None):
    return service.create_player(name, team or w["t"], joined_on="2024-01-01", db_path=w["p"])


def evaluate(w, pid, date, scores):
    return ev.create_evaluation(pid, date, "Personalizada", scores, db_path=w["p"])


# ── timeline ────────────────────────────────────────────────────────────────
def test_timeline_empty_team(w):
    assert teamstats.timeline(w["t"], w["p"]) == []
    o = teamstats.overview(w["t"], w["p"])
    assert o["n_evaluations"] == 0 and o["average"] is None and o["last_date"] is None and o["change"] is None


def test_timeline_points(w):
    a, b, c = player(w, "A"), player(w, "B"), player(w, "C")
    for pid, v in ((a, 2), (b, 3)):
        evaluate(w, pid, "2024-09-15", flat(v))
    for pid, v in ((a, 3), (b, 4), (c, 3)):
        evaluate(w, pid, "2024-12-15", flat(v))
    tl = teamstats.timeline(w["t"], w["p"])
    assert [(p["date"], p["n_evaluated"], p["members"]) for p in tl] == [("2024-09-15", 2, 3), ("2024-12-15", 3, 3)]
    assert tl[0]["average"] == pytest.approx(2.5) and tl[1]["average"] == pytest.approx(10 / 3)


# ── evolução com os mesmos jogadores ────────────────────────────────────────
def test_snapshot_change_like_for_like(w):
    a, b, c = player(w, "A"), player(w, "B"), player(w, "C")
    evaluate(w, a, "2024-09-15", flat(2)); evaluate(w, b, "2024-09-15", flat(3))
    evaluate(w, a, "2024-12-15", flat(4)); evaluate(w, b, "2024-12-15", flat(3))
    evaluate(w, c, "2024-12-15", flat(5))                      # só avaliada na segunda data → fora da comparação
    tl = teamstats.timeline(w["t"], w["p"])
    ch = teamstats.snapshot_change(tl[0]["snapshot"], tl[1]["snapshot"])
    assert ch["n_common"] == 2
    assert ch["avg_before"] == pytest.approx(2.5) and ch["avg_after"] == pytest.approx(3.5)
    assert ch["avg_delta"] == pytest.approx(1.0)
    assert all(r["n"] == 2 and r["delta"] == pytest.approx(1.0) for r in ch["rows"])
    # a média simples da equipa subiu mais (inclui o jogador novo), mas a evolução é like-for-like
    assert tl[1]["average"] - tl[0]["average"] > ch["avg_delta"]


def test_player_not_reevaluated_is_not_counted_as_zero_change(w):
    a, b = player(w, "A"), player(w, "B")
    evaluate(w, a, "2024-09-15", flat(2)); evaluate(w, b, "2024-09-15", flat(3))
    evaluate(w, a, "2024-12-15", flat(4))                       # B não foi reavaliado
    tl = teamstats.timeline(w["t"], w["p"])
    ch = teamstats.snapshot_change(tl[0]["snapshot"], tl[1]["snapshot"])
    assert ch["n_common"] == 1 and ch["avg_delta"] == pytest.approx(2.0)


def test_snapshot_change_per_competency_and_incomplete(w):
    a, b = player(w, "A"), player(w, "B")
    evaluate(w, a, "2024-09-15", flat(2, shooting=1)); evaluate(w, b, "2024-09-15", {"shooting": 3, "passing": 2})
    evaluate(w, a, "2024-12-15", flat(3, shooting=4)); evaluate(w, b, "2024-12-15", {"shooting": 5})
    tl = teamstats.timeline(w["t"], w["p"])
    ch = {r["key"]: r for r in teamstats.snapshot_change(tl[0]["snapshot"], tl[1]["snapshot"])["rows"]}
    assert ch["shooting"]["before"] == 2.0 and ch["shooting"]["after"] == 4.5 and ch["shooting"]["delta"] == 2.5
    assert ch["passing"]["n"] == 1 and ch["passing"]["delta"] == 1.0        # só o A; o B não tem passe na 2.ª data
    assert ch["dribbling"]["n"] == 1 and ch["dribbling"]["delta"] == 1.0


def test_snapshot_change_without_common_players(w):
    a, b = player(w, "A"), player(w, "B")
    evaluate(w, a, "2024-09-15", flat(2)); evaluate(w, b, "2024-12-15", flat(3))
    tl = teamstats.timeline(w["t"], w["p"])
    ch = teamstats.snapshot_change(tl[0]["snapshot"], tl[1]["snapshot"])
    assert ch["n_common"] == 0 and ch["avg_delta"] is None and all(r["delta"] is None for r in ch["rows"])


def test_overview_values(w):
    a, b = player(w, "A"), player(w, "B"); player(w, "D")                      # D sem avaliações
    evaluate(w, a, "2024-09-15", flat(2)); evaluate(w, b, "2024-09-15", flat(4))
    evaluate(w, a, "2024-12-15", flat(3)); evaluate(w, b, "2024-12-15", flat(5))
    o = teamstats.overview(w["t"], w["p"])
    assert (o["members"], o["n_evaluated"], o["n_evaluations"]) == (3, 2, 4)
    assert o["first_date"] == "2024-09-15" and o["last_date"] == "2024-12-15"
    assert o["average"] == pytest.approx(4.0) and sorted(o["player_averages"]) == [3.0, 5.0]
    assert o["change"]["avg_delta"] == pytest.approx(1.0) and o["change"]["n_common"] == 2


def test_overview_single_date_has_no_change(w):
    evaluate(w, player(w, "A"), "2024-09-15", flat(3))
    assert teamstats.overview(w["t"], w["p"])["change"] is None


def test_overview_ignores_superseded_and_other_team(w):
    a = player(w, "A")
    first = evaluate(w, a, "2024-09-15", flat(2))
    ev.correct_evaluation(first, "2024-09-15", "Personalizada", flat(4), db_path=w["p"])
    assert teamstats.overview(w["t"], w["p"])["n_evaluations"] == 1
    assert teamstats.overview(w["t"], w["p"])["average"] == 4.0
    assert teamstats.overview(w["t8"], w["p"])["n_evaluations"] == 0


def test_player_moved_between_teams_kept_in_history(w):
    a = player(w, "A")
    evaluate(w, a, "2024-09-15", flat(2))
    service.change_team(a, w["t8"], "2024-11-01", db_path=w["p"])
    assert teamstats.overview(w["t"], w["p"])["n_evaluations"] == 1            # histórico da equipa antiga
    old = teamstats.timeline(w["t"], w["p"])
    assert old[0]["n_evaluated"] == 1                                         # na data da avaliação ainda era da equipa
    assert teamstats.snapshot(w["t"], "2024-12-01", w["p"])["evaluated"] == []


# ── gráfico de barras ───────────────────────────────────────────────────────
def test_bar_figure(w):
    a, b = player(w, "A"), player(w, "B")
    evaluate(w, a, "2024-09-15", flat(2, shooting=5)); evaluate(w, b, "2024-09-15", flat(3, shooting=4))
    stats = teamstats.competency_stats(teamstats.snapshot(w["t"], "2025-01-01", w["p"]))
    fig = charts.bar_figure(stats, 5)
    t = fig.data[0]
    assert list(t.y) == [c.short for c in comp.COMPETENCIES]                  # ordem da roda, não por valor
    assert t.x[0] == 4.5 and t.text[0] == "4,50" and t.orientation == "h"
    assert list(t.customdata[0]) == ["4,50", 5, 4, 2]                         # mediana, melhor, mais baixo, n
    assert tuple(fig.layout.xaxis.range)[0] == 0


def test_bar_figure_empty_team():
    stats = teamstats.competency_stats({"evaluated": []})
    t = charts.bar_figure(stats).data[0]
    assert all(x is None for x in t.x) and all(s == "" for s in t.text)


# ── UI ──────────────────────────────────────────────────────────────────────
def app_at():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "minibasket", "app.py")).run(timeout=30)


def test_ui_dashboard_empty_and_populated(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    at = app_at()
    assert not at.exception and any("Ainda não há equipas" in i.value for i in at.info)

    club = service.create_club("C")
    t10 = service.create_team(club, "S10", "Sub-10", "2024/2025")
    service.create_team(club, "S8", "Sub-8", "2024/2025")
    for n, (v1, v2) in zip("ABCD", ((2, 3), (3, 4), (2, 4), (3, 3))):
        pid = service.create_player(n, t10, joined_on="2024-01-01")
        ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", flat(v1))
        ev.create_evaluation(pid, "2024-12-15", "2.º Período", flat(v2))
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Equipas"] == "2" and metrics["Jogadores"] == "4" and metrics["Avaliações realizadas"] == "8"
    assert metrics["Última avaliação"] == "15/12/2024"
    table = next(d.value for d in at.dataframe if "Evolução" in d.value.columns)
    row = table[table["Equipa"] == "S10"].iloc[0]
    assert (row["Média global"], row["Evolução"], row["Avaliados"]) == ("3,50", "+1,00", 4)
    assert table[table["Equipa"] == "S8"].iloc[0]["Média global"] == "—"
    assert len(at.get("plotly_chart")) == 2


def test_ui_team_evolution(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui2.db"))
    club = service.create_club("C")
    t = service.create_team(club, "S10", "Sub-10", "2024/2025")
    at = app_at()
    at.sidebar.radio[0].set_value("Evolução da Equipa").run()
    at.radio(key="tevo_cat").set_value("Sub-10").run()
    assert not at.exception and any("ainda não tem avaliações" in i.value for i in at.info)

    pid = service.create_player("A", t, joined_on="2024-01-01")
    ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", flat(2))
    at.run()
    assert not at.exception and any("duas datas" in i.value for i in at.info)

    ev.create_evaluation(pid, "2024-12-15", "2.º Período", flat(4))
    at.run()
    assert not at.exception
    m = next(m for m in at.metric if m.label == "Evolução da média global")
    assert m.value.startswith("4,00") and m.delta == "+2,00"
    at.selectbox(key="team_d0").set_value("2024-12-15").run()
    assert any("anterior à data atual" in i.value for i in at.info)
