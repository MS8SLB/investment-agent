"""Fase 2 — clubes, equipas, jogadores, mudança de escalão, pesquisa e UI."""

import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minibasket import db, service
from minibasket.service import ValidationError


@pytest.fixture
def path(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    return p


@pytest.fixture
def world(path):
    club = service.create_club("Clube A", db_path=path)
    t8 = service.create_team(club, "Sub-8 A", "Sub-8", "2026/2027", db_path=path)
    t10 = service.create_team(club, "Sub-10 A", "Sub-10", "2026/2027", db_path=path)
    t12 = service.create_team(club, "Sub-12 A", "Sub-12", "2026/2027", db_path=path)
    return {"club": club, "t8": t8, "t10": t10, "t12": t12, "path": path}


# ── Clubes e equipas ────────────────────────────────────────────────────────
def test_club_rules(path):
    service.create_club("Clube", db_path=path)
    with pytest.raises(ValidationError):
        service.create_club("  clube ", db_path=path)   # duplicado (sem distinguir maiúsculas)
    with pytest.raises(ValidationError):
        service.create_club("  ", db_path=path)


def test_team_validation(world):
    p, club = world["path"], world["club"]
    with pytest.raises(ValidationError):
        service.create_team(club, "X", "Sub-14", "2026/2027", db_path=p)
    for bad in ("2026", "2026/2028", "26/27", ""):
        with pytest.raises(ValidationError):
            service.create_team(club, "X", "Sub-8", bad, db_path=p)
    with pytest.raises(ValidationError):
        service.create_team(club, "sub-8 a", "Sub-8", "2026/2027", db_path=p)   # duplicada
    with pytest.raises(ValidationError):
        service.create_team(999, "X", "Sub-8", "2026/2027", db_path=p)


def test_list_teams_filters(world):
    p = world["path"]
    assert len(service.list_teams(db_path=p)) == 3
    assert [t["category"] for t in service.list_teams(category="Sub-10", db_path=p)] == ["Sub-10"]


# ── Jogadores ───────────────────────────────────────────────────────────────
def test_create_player_full_record(world):
    p = world["path"]
    pid = service.create_player("João Silva", world["t10"], "2016-05-03", "M", 7, "2026-09-15",
                                "Canhoto", db_path=p)
    f = service.get_player(pid, p)
    assert (f["name"], f["category"], f["team"], f["club"], f["season"]) == \
        ("João Silva", "Sub-10", "Sub-10 A", "Clube A", "2026/2027")
    assert f["jersey_number"] == 7 and f["joined_on"] == "2026-09-15" and f["notes"] == "Canhoto"
    assert len(f["team_history"]) == 1 and f["team_history"][0]["left_on"] is None


def test_create_player_defaults_and_validation(world):
    p, t = world["path"], world["t8"]
    pid = service.create_player("Ana", t, db_path=p)             # só o essencial
    assert service.get_player(pid, p)["joined_on"] == date.today().isoformat()
    future = (date.today() + timedelta(days=1)).isoformat()
    for kw in ({"name": " "}, {"birth_date": future}, {"birth_date": "2016-13-01"},
               {"sex": "X"}, {"jersey_number": 100}, {"jersey_number": "abc"}):
        args = {"name": "Z", "team_id": t, "db_path": p, **kw}
        with pytest.raises(ValidationError):
            service.create_player(**args)
    with pytest.raises(ValidationError):
        service.create_player("Z", 999, db_path=p)


def test_update_player_keeps_photo_when_none(world):
    p = world["path"]
    pid = service.create_player("Rui", world["t8"], photo_path="data/photos/x.png", jersey_number=4, db_path=p)
    service.update_player(pid, "Rui Costa", "2018-02-01", "M", "nota", None, 9, db_path=p)
    f = service.get_player(pid, p)
    assert (f["name"], f["photo_path"], f["jersey_number"], f["notes"]) == ("Rui Costa", "data/photos/x.png", 9, "nota")
    with pytest.raises(ValidationError):
        service.update_player(999, "X", db_path=p)


def test_change_category_keeps_history(world):
    p = world["path"]
    pid = service.create_player("Inês", world["t8"], joined_on="2026-09-01", jersey_number=3, db_path=p)
    service.change_team(pid, world["t10"], "2027-09-01", 11, db_path=p)
    f = service.get_player(pid, p)
    assert (f["category"], f["team"], f["jersey_number"]) == ("Sub-10", "Sub-10 A", 11)
    old, new = f["team_history"][1], f["team_history"][0]
    assert old["category"] == "Sub-8" and old["left_on"] == "2027-09-01" and old["joined_on"] == "2026-09-01"
    assert new["category"] == "Sub-10" and new["left_on"] is None
    # deixa de aparecer no escalão antigo, aparece no novo
    assert pid not in [x["id"] for x in service.search_players(category="Sub-8", db_path=p)]
    assert pid in [x["id"] for x in service.search_players(category="Sub-10", db_path=p)]


def test_change_team_rules(world):
    p = world["path"]
    pid = service.create_player("Tiago", world["t8"], joined_on="2026-09-10", db_path=p)
    with pytest.raises(ValidationError):
        service.change_team(pid, world["t8"], db_path=p)            # mesma equipa
    with pytest.raises(ValidationError):
        service.change_team(pid, 999, db_path=p)
    with pytest.raises(ValidationError):
        service.change_team(pid, world["t10"], "2026-09-01", db_path=p)   # antes da entrada
    with pytest.raises(ValidationError):
        service.change_team(999, world["t10"], db_path=p)


def test_search_accent_insensitive_and_alphabetical(world):
    p = world["path"]
    for n in ("Zé Costa", "João Pires", "Joana Reis", "Álvaro Dias"):
        service.create_player(n, world["t10"], db_path=p)
    service.create_player("Rita", world["t12"], db_path=p)
    assert [x["name"] for x in service.search_players("joao", db_path=p)] == ["João Pires"]
    assert [x["name"] for x in service.search_players("JOA", db_path=p)] == ["Joana Reis", "João Pires"]
    assert [x["name"] for x in service.search_players(category="Sub-10", db_path=p)] == \
        ["Álvaro Dias", "Joana Reis", "João Pires", "Zé Costa"]
    assert [x["name"] for x in service.search_players(team_id=world["t12"], db_path=p)] == ["Rita"]
    assert service.search_players("inexistente", db_path=p) == []
    with pytest.raises(ValidationError):
        service.search_players(category="Sub-14", db_path=p)


def test_get_player_missing(path):
    assert service.get_player(123, path) is None


def test_save_photo(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "PHOTO_DIR", str(tmp_path / "ph"))
    f = service.save_photo(b"\x89PNG", "Joao Silva.PNG")
    assert os.path.exists(f) and "joao" not in os.path.basename(f).lower() and f.endswith(".png")
    with pytest.raises(ValidationError):
        service.save_photo(b"x", "a.exe")
    with pytest.raises(ValidationError):
        service.save_photo(b"x" * (5 * 1024 * 1024 + 1), "a.png")


# ── UI ──────────────────────────────────────────────────────────────────────
def test_ui_flow(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "minibasket", "app.py"))
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("Equipas").run()
    assert not at.exception
    at.text_input[0].set_value("Clube Teste")
    [b for b in at.button if b.label == "Criar clube"][0].click().run()
    assert not at.exception and service.list_clubs()[0]["name"] == "Clube Teste"
    at.text_input[1].set_value("Sub-10 Teste")
    [b for b in at.button if b.label == "Criar equipa"][0].click().run()
    assert not at.exception and service.list_teams()[0]["name"] == "Sub-10 Teste"

    service.create_player("Maria Luís", service.list_teams()[0]["id"])
    at.sidebar.radio[0].set_value("Jogadores").run()
    assert not at.exception
    assert any("Maria Luís" in s.value for s in at.subheader)
    at.text_input(key="player_search").set_value("zzz").run()
    assert any("Nenhum jogador" in i.value for i in at.info)
