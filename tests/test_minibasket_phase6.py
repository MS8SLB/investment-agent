"""Fase 6 — gráficos individuais, estatísticas da equipa e comparação jogador vs equipa."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from ui_helper import logged_in_app  # noqa: E402

from minibasket import calc, charts, db, evaluations as ev, service, teamstats
from minibasket import competencies as comp

LEVELS = {1: "Inicial", 2: "Em desenvolvimento", 3: "Adequado", 4: "Bom", 5: "Muito bom"}


def sc(*vals):
    return dict(zip(comp.KEYS, vals))


def flat(n):
    return {k: n for k in comp.KEYS}


# ── line chart ──────────────────────────────────────────────────────────────
def pts(*vals):
    dates = ["2024-09-15", "2024-12-15", "2025-03-15", "2025-06-15"]
    return [{"date": d, "moment": f"M{i}", "value": v} for i, (d, v) in enumerate(zip(dates, vals))]


def test_line_figure_basic():
    fig = charts.line_figure(pts(2, 3, 3, 4), 5, "Lançamento", LEVELS, decimals=0)
    t = fig.data[0]
    assert list(t.y) == [2, 3, 3, 4] and list(t.x) == ["2024-09-15", "2024-12-15", "2025-03-15", "2025-06-15"]
    assert list(t.text) == ["2", "", "", "4"]                         # só primeiro e último rotulados
    assert fig.layout.yaxis.range[0] == 0 and fig.layout.yaxis.range[1] >= 5
    assert fig.layout.showlegend is False and t.line.width == 2 and t.marker.size >= 8
    assert t.customdata[1][2] == "Adequado" and t.customdata[0][0] == "M0"


def test_line_figure_portuguese_decimals_and_gap():
    fig = charts.line_figure(pts(2 + 4 / 9, None, 3 + 4 / 9, 4.0), 5, "Média global")
    t = fig.data[0]
    assert t.text[0] == "2,44" and t.text[3] == "4,00"
    assert t.y[1] is None and t.connectgaps is False and t.customdata[1][1] == "Não avaliada"


def test_line_figure_single_and_empty():
    one = charts.line_figure(pts(3), 5).data[0]
    assert list(one.text) == ["3,00"]
    none = charts.line_figure(pts(None, None), 5).data[0]
    assert list(none.text) == ["", ""]
    assert len(charts.line_figure([], 5).data[0].x) == 0


# ── estatísticas da equipa ──────────────────────────────────────────────────
@pytest.fixture
def w(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    club = service.create_club("C", db_path=p)
    t = service.create_team(club, "S10", "Sub-10", "2024/2025", db_path=p)
    t2 = service.create_team(club, "S12", "Sub-12", "2024/2025", db_path=p)
    return {"p": p, "t": t, "t2": t2}


def add(w, name, scores, date="2024-10-01", team=None):
    pid = service.create_player(name, team or w["t"], joined_on="2024-01-01", db_path=w["p"])
    if scores is not None:
        ev.create_evaluation(pid, date, "1.º Período", scores, db_path=w["p"])
    return pid


def test_team_without_evaluations(w):
    s = teamstats.snapshot(w["t"], "2025-01-01", w["p"])
    assert s["evaluated"] == [] and teamstats.team_average(s) is None
    st_ = teamstats.competency_stats(s)
    assert all(r["n"] == 0 and r["mean"] is None and r["median"] is None for r in st_)
    assert not teamstats.can_compare(s)
    empty = service.create_team(1, "Vazia", "Sub-8", "2024/2025", db_path=w["p"])
    assert teamstats.snapshot(empty, "2025-01-01", w["p"])["members"] == 0


def test_competency_stats_values(w):
    add(w, "A", flat(2)); add(w, "B", flat(3)); add(w, "C", flat(5)); add(w, "D", None)   # D sem avaliações
    s = teamstats.snapshot(w["t"], "2025-01-01", w["p"])
    assert s["members"] == 4 and len(s["evaluated"]) == 3
    r = teamstats.competency_stats(s)[0]
    assert r["n"] == 3 and r["mean"] == pytest.approx(10 / 3) and r["median"] == 3.0
    assert (r["best"], r["lowest"]) == (5, 2)
    assert teamstats.team_average(s) == pytest.approx(10 / 3)
    assert "name" not in s["evaluated"][0]                               # nada identifica o jogador


def test_stats_with_incomplete_evaluations(w):
    add(w, "A", {"shooting": 4}); add(w, "B", {"shooting": 2, "passing": 5})
    st_ = {r["key"]: r for r in teamstats.competency_stats(teamstats.snapshot(w["t"], "2025-01-01", w["p"]))}
    assert st_["shooting"]["n"] == 2 and st_["shooting"]["mean"] == 3.0
    assert st_["passing"]["n"] == 1 and st_["passing"]["mean"] == 5.0 and st_["dribbling"]["n"] == 0


def test_snapshot_uses_latest_evaluation_up_to_date(w):
    a = add(w, "A", flat(2), "2024-10-01")
    ev.create_evaluation(a, "2025-02-01", "2.º Período", flat(4), db_path=w["p"])
    assert teamstats.snapshot(w["t"], "2024-12-01", w["p"])["evaluated"][0]["scores"]["shooting"] == 2
    assert teamstats.snapshot(w["t"], "2025-03-01", w["p"])["evaluated"][0]["scores"]["shooting"] == 4


def test_snapshot_respects_team_membership_and_corrections(w):
    a = add(w, "A", flat(2), "2024-10-01")
    first = ev.list_evaluations(a, db_path=w["p"])[0]["id"]
    ev.correct_evaluation(first, "2024-10-01", "1.º Período", flat(3), db_path=w["p"])
    assert teamstats.snapshot(w["t"], "2024-11-01", w["p"])["evaluated"][0]["scores"]["shooting"] == 3   # versão em vigor
    service.change_team(a, w["t2"], "2025-01-10", db_path=w["p"])
    assert len(teamstats.snapshot(w["t"], "2025-02-01", w["p"])["evaluated"]) == 0       # já saiu
    assert len(teamstats.snapshot(w["t"], "2024-12-01", w["p"])["evaluated"]) == 1       # ainda lá estava
    assert len(teamstats.snapshot(w["t2"], "2025-02-01", w["p"])["evaluated"]) == 0      # sem avaliações na nova equipa


def test_compare_to_team_example_from_spec(w):
    # Lançamento do jogador 4 vs média da equipa 3,2 → +0,8
    stats = [{"key": k, "n": 5, "mean": 3.2 if k == "shooting" else None} for k in comp.KEYS]
    rows = teamstats.compare_to_team({"shooting": 4}, stats)
    r = rows[0]
    assert r["team_mean"] == 3.2 and calc.fmt(r["diff"], 1, signed=True) == "+0,8"
    assert rows[1]["diff"] is None                                       # sem dados → sem diferença


def test_min_team_for_comparison(w):
    add(w, "A", flat(3)); add(w, "B", flat(3))
    assert not teamstats.can_compare(teamstats.snapshot(w["t"], "2025-01-01", w["p"]))
    add(w, "C", flat(3))
    assert teamstats.can_compare(teamstats.snapshot(w["t"], "2025-01-01", w["p"]))


# ── UI ──────────────────────────────────────────────────────────────────────
def test_ui_charts_and_team_comparison(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    club = service.create_club("C")
    team = service.create_team(club, "S10", "Sub-10", "2024/2025")
    ids = []
    for n in ("João", "Rui", "Ana", "Zé"):
        pid = service.create_player(n, team, joined_on="2024-01-01")
        ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", sc(2, 3, 2, 3, 2, 2, 3, 2, 3))
        ev.create_evaluation(pid, "2024-12-15", "2.º Período", sc(4, 4, 3, 4, 2, 3, 4, 3, 4))
        ids.append(pid)
    at = logged_in_app()
    at.sidebar.radio[0].set_value("Evolução do Jogador").run()
    at.radio(key="evo_cat").set_value("Sub-10").run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Roda das Competências", "Evolução da média", "Evolução por competência",
                                          "Jogador vs. equipa", "Histórico"]
    n_default = len(at.get("plotly_chart"))
    at.selectbox(key="evo_comp").set_value("all").run()
    assert not at.exception and len(at.get("plotly_chart")) == n_default - 1 + 9      # nove miniaturas
    at.selectbox(key="evo_comp").set_value("shooting").run()
    assert any("de 2 (Em desenvolvimento) para 4 (Bom)" in m.value for m in at.markdown)
    assert any(d.value["Diferença"].iloc[0] == "+0,0" for d in at.dataframe if "Média equipa" in d.value.columns)
    assert not any("pelo menos" in i.value for i in at.info)                          # 4 avaliados ≥ mínimo


def test_ui_team_comparison_hidden_with_few_players(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui2.db"))
    club = service.create_club("C")
    team = service.create_team(club, "S10", "Sub-10", "2024/2025")
    pid = service.create_player("João", team, joined_on="2024-01-01")
    ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", flat(3))
    at = logged_in_app()
    at.sidebar.radio[0].set_value("Evolução do Jogador").run()
    at.radio(key="evo_cat").set_value("Sub-10").run()
    assert not at.exception and any("pelo menos 3" in i.value for i in at.info)
