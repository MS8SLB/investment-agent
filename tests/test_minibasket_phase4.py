"""Fase 4 — Roda das Competências (comparação e radar)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minibasket import calc, charts
from minibasket import competencies as comp
from minibasket import db, evaluations as ev, service

LEVELS = {1: "Inicial", 2: "Em desenvolvimento", 3: "Adequado", 4: "Bom", 5: "Muito bom"}
INITIAL = dict(zip(comp.KEYS, (2, 3, 2, 3, 2, 2, 3, 2, 3)))
CURRENT = dict(zip(comp.KEYS, (4, 4, 3, 4, 2, 3, 4, 3, 4)))


# ── calc.compare ────────────────────────────────────────────────────────────
def test_compare_deltas_and_averages():
    c = calc.compare(INITIAL, CURRENT)
    d = {r["key"]: r["delta"] for r in c["rows"]}
    assert d["shooting"] == 2 and d["footwork"] == 0 and d["dribbling"] == 1
    assert [r["key"] for r in c["rows"]] == list(comp.KEYS)
    assert c["n_common"] == 9
    assert c["avg_before"] == pytest.approx(22 / 9) and c["avg_after"] == pytest.approx(31 / 9)
    assert c["avg_delta"] == pytest.approx(1.0)


def test_compare_negative_delta_allowed():
    c = calc.compare({"shooting": 4}, {"shooting": 3})
    assert c["rows"][0]["delta"] == -1 and c["avg_delta"] == -1.0


def test_compare_incomplete_uses_common_competencies_only():
    before = {"shooting": 2, "passing": 2}                       # incompleta
    after = {**CURRENT, "passing": None}
    c = calc.compare(before, after)
    assert c["n_common"] == 1                                    # só o lançamento está nas duas
    assert c["avg_before"] == 2 and c["avg_after"] == 4 and c["avg_delta"] == 2
    p = next(r for r in c["rows"] if r["key"] == "passing")
    assert p["delta"] is None


def test_compare_no_overlap():
    c = calc.compare({"shooting": 3}, {"passing": 3})
    assert c["n_common"] == 0 and c["avg_delta"] is None and c["avg_before"] is None


# ── radar ───────────────────────────────────────────────────────────────────
def test_radar_single_series():
    fig = charts.radar_figure([{"name": "Atual", "scores": CURRENT}], 5, LEVELS)
    assert len(fig.data) == 1
    t = fig.data[0]
    assert list(t.theta[:9]) == [c.short for c in comp.COMPETENCIES]     # as nove competências, pela ordem
    assert list(t.r[:9]) == [4, 4, 3, 4, 2, 3, 4, 3, 4]
    assert t.r[-1] == t.r[0] and t.theta[-1] == t.theta[0]               # polígono fechado
    assert tuple(fig.layout.polar.radialaxis.range) == (0, 5)             # escala 0–5
    assert fig.layout.showlegend is False
    assert t.fill == "toself" and t.line.dash == "solid"


def test_radar_two_series_distinguishable_without_color():
    fig = charts.radar_figure([{"name": "Atual", "scores": CURRENT}, {"name": "Inicial", "scores": INITIAL}])
    assert fig.layout.showlegend is True and len(fig.data) == 2
    by = {t.name: t for t in fig.data}
    assert by["Atual"].line.color != by["Inicial"].line.color
    assert by["Atual"].line.dash == "solid" and by["Inicial"].line.dash == "dash"        # texto/forma, não só cor
    assert by["Atual"].fill == "toself" and by["Inicial"].fill is None
    assert fig.data[0].name == "Inicial"                                  # referência desenhada por baixo
    assert list(by["Inicial"].r[:9]) == [2, 3, 2, 3, 2, 2, 3, 2, 3]


def test_radar_missing_scores_not_drawn_as_zero():
    fig = charts.radar_figure([{"name": "X", "scores": {"shooting": 3}}], 5, LEVELS)
    t = fig.data[0]
    assert t.r[0] == 3 and all(v is None for v in t.r[1:9])
    assert t.customdata[0] == "Adequado" and t.customdata[1] == "Não avaliada"


def test_radar_dark_palette_and_scale_max():
    fig = charts.radar_figure([{"name": "X", "scores": CURRENT}], 10, dark=True)
    assert fig.data[0].line.color == charts.COLORS["dark"][0]
    assert tuple(fig.layout.polar.radialaxis.range) == (0, 10)
    assert list(fig.layout.polar.radialaxis.tickvals) == list(range(11))


def test_radar_rejects_bad_series_count():
    with pytest.raises(ValueError):
        charts.radar_figure([])
    with pytest.raises(ValueError):
        charts.radar_figure([{"name": str(i), "scores": CURRENT} for i in range(3)])


# ── UI ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def ui(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    club = service.create_club("C")
    team = service.create_team(club, "S10", "Sub-10", "2026/2027")
    pid = service.create_player("João", team, joined_on="2024-09-01")
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "minibasket", "app.py")).run(timeout=30)
    at.sidebar.radio[0].set_value("Evolução do Jogador").run()
    at.radio(key="evo_cat").set_value("Sub-10").run()
    return at, pid


def test_ui_player_without_evaluations(ui):
    at, _ = ui
    assert not at.exception and any("ainda não tem avaliações" in i.value for i in at.info)
    assert not at.get("plotly_chart")


def test_ui_single_evaluation_and_comparison(ui):
    at, pid = ui
    ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", INITIAL)
    at.run()
    assert not at.exception and len(at.get("plotly_chart")) == 3  # roda + média + competência
    assert any("Só existe uma avaliação" in c.value for c in at.caption)

    ev.create_evaluation(pid, "2024-12-15", "1.º Período", CURRENT)
    at.run()
    assert not at.exception and len(at.get("plotly_chart")) == 3
    m = next(m for m in at.metric if m.label == "Média global")
    assert m.value.startswith("3,44") and m.delta == "+1,00"
    assert any("Lançamento (+2)" in s.value for s in at.success)

    ids = [e["id"] for e in ev.list_evaluations(pid)]
    at.selectbox(key="radar_cur").set_value(ids[0]).run()           # mesma avaliação nos dois lados
    assert not at.exception and any("duas avaliações diferentes" in i.value for i in at.info)


def test_ui_evaluate_page_shows_live_radar(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui2.db"))
    club = service.create_club("C")
    team = service.create_team(club, "S8", "Sub-8", "2026/2027")
    service.create_player("Ana", team, joined_on="2024-09-01")
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "minibasket", "app.py")).run(timeout=30)
    at.sidebar.radio[0].set_value("Avaliar").run()
    assert not at.exception and len(at.get("plotly_chart")) == 1
