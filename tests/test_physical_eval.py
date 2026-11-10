"""Testes da plataforma de avaliação física (physical_eval)."""

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from physical_eval import norms, service


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "p.db")


# Valores da folha «Exemplo preenchido» (percentis calculados pela folha original).
@pytest.mark.parametrize("test,value,age,sex,expected", [
    ("altura", 150, 10, None, 70), ("peso", 39.7, 10, None, 70), ("envergadura", 153.7, 10, None, 70),
    ("corrida", 3.62, 10, None, 70), ("drible", 3.82, 10, None, 70),
    ("altura", 151.4, 11, None, 50), ("peso", 39.8, 11, None, 50), ("envergadura", 154.9, 11, None, 50),
    ("corrida", 3.69, 11, None, 50), ("drible", 4.36, 11, None, 50),
    ("altura", 170.3, 12, None, 90), ("peso", 60, 12, None, 90), ("envergadura", 180, 12, None, 90),
    ("corrida", 3.34, 12, None, 90),
])
def test_percentiles_match_sheet(test, value, age, sex, expected):
    assert norms.percentile(test, value, age, sex) == expected


def test_boundaries_and_floor():
    assert norms.percentile("altura", 100, 10) == 5          # abaixo do 1.º limiar
    assert norms.percentile("altura", 200, 10) == 90
    assert norms.percentile("corrida", 9.0, 10) == 5          # muito lento
    assert norms.percentile("corrida", 1.0, 10) == 90
    assert norms.percentile("salto", 0.5, 8, "F") == 10       # <= P10 -> 10
    assert norms.percentile("salto", 1.30, 8, "F") == 60


def test_missing_reference_is_none():
    assert norms.percentile("altura", 150, 6) is None
    assert norms.percentile("drible", 3.5, 16) is None        # drible só até 15
    assert norms.percentile("salto", 1.5, 10, "M") is None    # masculino só 13–15
    assert norms.percentile("salto", 1.5, 10, None) is None   # sem sexo
    assert norms.percentile("altura", None, 10) is None
    assert norms.percentile("altura", 150, None) is None


def test_jump_male_13():
    assert norms.percentile("salto", 1.70, 13, "M") == 50
    assert norms.percentile("salto", 1.30, 13, "M") == 10


def test_provisional_flag():
    assert norms.is_provisional("drible", 13) and not norms.is_provisional("drible", 10)


@pytest.mark.parametrize("raw,expected", [("1,52", 1.52), ("1.52", 1.52), (1.52, 1.52), ("", None), (None, None), ("  ", None)])
def test_parse_number(raw, expected):
    assert service.parse_number(raw, "jump") == expected


@pytest.mark.parametrize("raw", ["abc", "nan", "inf", "-1", "0", "15", True])
def test_parse_number_rejects(raw):
    with pytest.raises(ValueError):
        service.parse_number(raw, "jump")


def test_age_on():
    assert service.age_on("2014-06-15", "2024-06-14") == 9
    assert service.age_on("2014-06-15", "2024-06-15") == 10


def test_best_trials_and_validation(db):
    rec = service.build_assessment(eval_date=date.today(), recorded_by="Ana", age=10, height="150,5",
                                   jumps=["1,40", "1,54", ""], runs=["3,70", "3,62"], dribbles=["3,9", ""])
    assert rec["jump_best"] == 1.54 and rec["run_best"] == 3.62 and rec["dribble_best"] == 3.9
    assert rec["height_cm"] == 150.5
    with pytest.raises(ValueError, match="estagiário"):
        service.build_assessment(eval_date=date.today(), recorded_by=" ", age=10, height=150)
    with pytest.raises(ValueError, match="pelo menos uma"):
        service.build_assessment(eval_date=date.today(), recorded_by="Ana", age=10)
    with pytest.raises(ValueError, match="idade"):
        service.build_assessment(eval_date=date.today(), recorded_by="Ana", height=150)
    with pytest.raises(ValueError, match="plausível"):
        service.build_assessment(eval_date=date.today(), recorded_by="Ana", age=10, height=1500)
    with pytest.raises(ValueError, match="futura"):
        service.build_assessment(eval_date="2999-01-01", recorded_by="Ana", age=10, height=150)


def test_age_from_birth_date():
    rec = service.build_assessment(eval_date="2024-06-14", recorded_by="Ana", birth_date="2014-06-15", height=140)
    assert rec["age"] == 9


def test_roundtrip_edit_delete_export(db):
    aid = service.add_athlete("Atleta A", "F", "2014-01-01", "Sub-12", "1", db_path=db)
    rec = service.build_assessment(eval_date="2024-06-01", recorded_by="Ana", birth_date="2014-01-01",
                                   height=150, weight=39.7, wingspan=153.7, jumps=["1.54"], runs=["3.62"], dribbles=["3.82"])
    sid = service.save_assessment(aid, rec, db_path=db)
    h = service.athlete_history(aid, db)
    assert len(h) == 1 and h[0]["pctl_altura"] == 70 and h[0]["pctl_corrida"] == 70 and h[0]["pctl_salto"] is not None

    rec2 = service.build_assessment(eval_date="2024-06-01", recorded_by="Rui", birth_date="2014-01-01", height=160)
    service.save_assessment(aid, rec2, sid, db)
    h = service.athlete_history(aid, db)
    assert len(h) == 1 and h[0]["height_cm"] == 160 and h[0]["recorded_by"] == "Rui" and h[0]["weight_kg"] is None

    rows = service.export_rows(db)
    assert rows[0]["Nome"] == "Atleta A" and rows[0]["Registado por"] == "Rui"
    assert service.export_xlsx(db)[:2] == b"PK" and b"Nome" in service.export_csv(db)

    service.delete_assessment(sid, db)
    assert service.athlete_history(aid, db) == []
    service.delete_athlete(aid, db)
    assert service.list_athletes(db_path=db) == []


def test_add_athlete_validation(db):
    with pytest.raises(ValueError):
        service.add_athlete(" ", db_path=db)
    with pytest.raises(ValueError):
        service.add_athlete("X", sex="Z", db_path=db)
    with pytest.raises(ValueError, match="futura"):
        service.add_athlete("X", birth_date="2999-01-01", db_path=db)
