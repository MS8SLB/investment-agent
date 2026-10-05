"""Testes do Teste de Movimentos Defensivos (basketball_eval)."""

import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from basketball_eval import defensive_movement as dm
from basketball_eval import protocol, service
from basketball_eval.db import connect, init_db


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "t.db")
    init_db(path)
    return path


def T(*xs, valid=None):
    valid = valid or [True] * len(xs)
    return [dm.Trial(dm.parse_time(x), v) for x, v in zip(xs, valid)]


# ── parse_time ──────────────────────────────────────────────────────────────
def test_parse_decimal_and_comma():
    assert dm.parse_time("12.35") == Decimal("12.35")
    assert dm.parse_time("12,35") == Decimal("12.35")
    assert dm.parse_time(12.35) == Decimal("12.35")
    assert dm.parse_time("") is None and dm.parse_time(None) is None


@pytest.mark.parametrize("bad", ["abc", "-1", -0.5, 0, "0", "nan", "inf", "1.2.3", True])
def test_parse_rejects(bad):
    with pytest.raises(dm.InvalidTimeError):
        dm.parse_time(bad)


# ── melhor resultado ────────────────────────────────────────────────────────
def test_spec_example_best_is_minimum():
    r = dm.compute_best(T("12.84", "12.31", "12.56"))
    assert r.best_time == Decimal("12.31") and r.best_trial == 2


def test_invalid_trial_excluded_even_if_fastest():
    r = dm.compute_best(T("12.84", "10.00", "12.56", valid=[True, False, True]))
    assert r.best_time == Decimal("12.56") and r.best_trial == 3


def test_missing_trials_and_all_invalid():
    assert dm.compute_best(T("12.5", None, None)).best_trial == 1
    r = dm.compute_best(T("12.5", "12.1", valid=[False, False]) + [dm.Trial()])
    assert r.best_time is None and r.best_trial is None


def test_practice_first_trial_not_counted_but_kept():
    r = dm.compute_best(T("11.00", "12.31", "12.56"), first_is_practice=True)
    assert r.best_trial == 2 and r.counted_trials == [2, 3]


def test_tie_picks_earliest():
    assert dm.compute_best(T("12.0", "12.0", "12.5")).best_trial == 1


# ── evolução ────────────────────────────────────────────────────────────────
def test_spec_example_evolution():
    e = dm.compute_evolution(Decimal("12.31"), Decimal("12.78"))
    assert e.change_seconds == Decimal("-0.47")
    assert round(e.change_percentage, 2) == Decimal("3.68")
    assert e.status == dm.IMPROVED
    assert e.message == "Melhoria do desempenho relativamente à avaliação anterior."


def test_worse_and_unchanged():
    w = dm.compute_evolution(Decimal("13.00"), Decimal("12.50"))
    assert w.status == dm.WORSENED and w.change_seconds > 0 and w.change_percentage < 0
    u = dm.compute_evolution(Decimal("12.50"), Decimal("12.50"))
    assert u.status == dm.UNCHANGED and u.change_percentage == 0


def test_no_previous_gives_none():
    assert dm.compute_evolution(Decimal("12"), None) is None


def test_formatting_pt():
    assert dm.fmt_seconds(12.31) == "12,31 s"
    assert dm.fmt_seconds(-0.47, signed=True) == "-0,47 s"
    assert dm.fmt_pct(3.677, signed=True) == "+3,68 %"


# ── equipa ──────────────────────────────────────────────────────────────────
def test_group_stats():
    s = dm.group_stats([12.0, 13.0, 15.0, 11.0])
    assert (s.n, s.best, s.worst, s.median) == (4, 11.0, 15.0, 12.5)
    assert s.mean == pytest.approx(12.75)
    assert dm.group_stats([]).n == 0


def test_compare_groups_paired():
    c = dm.compare_groups({1: 13.0, 2: 12.0, 3: 14.0}, {1: 12.5, 2: 12.0, 4: 11.0})
    assert c.paired_n == 2 and (c.paired_improved, c.paired_unchanged, c.paired_worsened) == (1, 1, 0)
    assert c.paired_mean_change_seconds == pytest.approx(-0.25)


def test_percentile_band_only_with_norms():
    assert dm.percentile_band(10.0, []) is None
    rows = [(90, 9.0), (50, 10.0), (10, 12.0)]
    assert dm.percentile_band(9.5, rows) == 50
    assert dm.percentile_band(13.0, rows) is None


def test_protocol_default_valid():
    protocol.validate(protocol.DEFAULT_PROTOCOL)
    bad = dict(protocol.DEFAULT_PROTOCOL, sequence=["A", "Z"])
    with pytest.raises(ValueError):
        protocol.validate(bad)


# ── base de dados / serviço ─────────────────────────────────────────────────
def test_save_and_history_with_previous(db):
    p = service.add_player("Ana", "Sub-12", "F", "Equipa A", "2014-03-01", db)
    service.save_test(p, "2026-01-10", ["13.20", "13.50", None], db_path=db)
    service.save_test(p, "2026-03-10", ["12.80", "12.90", "13.00"], db_path=db)
    service.save_test(p, "2026-06-10", ["12.84", "12.31", "12.56"], db_path=db)
    h = service.history(p, db)
    assert [x["best_time"] for x in h] == [13.2, 12.8, 12.31]
    assert h[0]["previous_best_time"] is None and h[0]["change_seconds"] is None
    assert h[2]["previous_best_time"] == 12.8 and h[2]["change_seconds"] == -0.49
    rep = service.player_report(p, db)
    assert rep["evolution"].status == dm.IMPROVED
    assert "Tentativa 2" in service.coach_report_text(p, db)


def test_invalid_test_saved_but_excluded(db):
    p = service.add_player("Rui", "Sub-10", db_path=db)
    service.save_test(p, "2026-01-10", ["13.0", "13.1", "13.2"], db_path=db)
    service.save_test(p, "2026-02-10", ["9", "9", "9"], [False] * 3, db_path=db)  # a repetir
    service.save_test(p, "2026-03-10", ["12.5", None, None], db_path=db)
    assert len(service.history(p, db)) == 2
    last = service.history(p, db)[-1]
    assert last["previous_best_time"] == 13.0  # ignora a avaliação inválida


def test_db_rejects_bad_input(db):
    p = service.add_player("Eva", "Sub-8", db_path=db)
    with pytest.raises(dm.InvalidTimeError):
        service.save_test(p, "2026-01-01", ["-1", None, None], db_path=db)
    with pytest.raises(dm.InvalidTimeError):
        service.save_test(p, "2026-01-01", ["abc", None, None], db_path=db)
    with pytest.raises(ValueError):
        service.add_player("X", "Sub-16", db_path=db)
    with pytest.raises(ValueError):
        service.save_test(999, "2026-01-01", ["12", None, None], db_path=db)


def test_team_comparison_does_not_mix_categories(db):
    a = service.add_player("A", "Sub-10", team="T", db_path=db)
    b = service.add_player("B", "Sub-12", team="T", db_path=db)
    for pid, t1, t2 in [(a, "13.0", "12.0"), (b, "10.0", "9.0")]:
        service.save_test(pid, "2026-01-10", [t1, None, None], db_path=db)
        service.save_test(pid, "2026-06-10", [t2, None, None], db_path=db)
    cmp10 = service.team_comparison("Sub-10", ("2026-01-01", "2026-01-31"), ("2026-06-01", "2026-06-30"), db_path=db)
    assert cmp10.before.n == 1 and cmp10.before.mean == 13.0 and cmp10.after.mean == 12.0
    assert cmp10.mean_change_seconds == -1.0


def test_schema_has_spec_columns(db):
    with connect(db) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(defensive_movement_tests)")}
    spec = {"id", "player_id", "evaluation_date", "category", "trial_1", "trial_2", "trial_3", "best_time",
            "best_trial", "previous_best_time", "change_seconds", "change_percentage", "valid", "notes",
            "coach_id", "created_at"}
    assert spec <= cols


# ── normas e protocolo completo ─────────────────────────────────────────────
from basketball_eval import norms


def test_norms_empty_by_default_then_loaded(db):
    assert norms.reference_position(10.0, 10, db_path=db) is None
    assert norms.load_matulaitis_2019(db) == 90
    assert norms.load_matulaitis_2019(db) == 90  # idempotente
    with connect(db) as c:
        assert c.execute("SELECT COUNT(*) n FROM reference_norms").fetchone()["n"] == 90


def test_norms_values_spot_check(db):
    norms.load_matulaitis_2019(db)
    # Idade 12: 90> = 8.3, 50 = 9.26, <10 = 10.3 (tabela do PDF)
    assert norms.reference_position(8.3, 12, db_path=db)["percentile"] == 90
    assert norms.reference_position(9.0, 12, db_path=db)["percentile"] == 60  # <=9.03 e >8.9
    assert norms.reference_position(9.26, 12, db_path=db)["percentile"] == 50
    assert norms.reference_position(11.0, 12, db_path=db)["percentile"] is None  # acima de 10.3
    assert norms.reference_position(9.0, 20, db_path=db) is None  # sem norma para a idade


def test_norms_monotonic_per_age():
    """Menor tempo = melhor: os limites crescem ao descer o percentil.

    Exceção conhecida, fiel à tabela do PDF (verificada): aos 15 anos, p80 = 7.7 > p70 = 7.54.
    Não é «corrigida» na transcrição.
    """
    for i, age in enumerate(norms.AGES):
        col = [norms.TABLE[p][i] for p in norms.PERCENTILES]
        if age == 15:
            assert norms.TABLE[80][i] > norms.TABLE[70][i]
            continue
        assert col == sorted(col), f"idade {age} fora de ordem: {col}"


def test_age_at():
    assert norms.age_at("2014-03-01", "2026-02-28") == 11
    assert norms.age_at("2014-03-01", "2026-03-01") == 12


def test_seeded_protocol_complete_and_coherent(db):
    with connect(db) as c:
        cfg = protocol.load(c)
    assert cfg["sequence"] == list("ABCDEFA")
    assert [(l["from"], l["to"]) for l in cfg["legs"]] == list(zip("ABCDEF", "BCDEFA"))
    assert set(cfg["points"]) == set("ABCDEF")
