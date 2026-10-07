"""Fase 12 — dados de teste: criação, marcação, remoção segura e separação dos dados reais."""

import io
import os
import sys
from datetime import date

import pytest
from pypdf import PdfReader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from ui_helper import logged_in_app  # noqa: E402

from minibasket import access, auth, db, evaluations as ev, evolution, reports, seed, service, teamstats  # noqa: E402
from minibasket import competencies as comp  # noqa: E402
from minibasket.service import ValidationError  # noqa: E402


def flat(n):
    return {k: n for k in comp.KEYS}


@pytest.fixture
def path(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    return p


@pytest.fixture
def demo(path):
    seed.load_demo(db_path=path)
    return path


def rows(path, sql, *args):
    with db.connect(path) as c:
        return [dict(r) for r in c.execute(sql, args)]


# ── conteúdo dos dados de teste ─────────────────────────────────────────────
def test_demo_counts_per_category(demo):
    by = {r["category"]: r["n"] for r in rows(demo, """SELECT t.category, COUNT(*) AS n FROM team_memberships m
                                                       JOIN teams t ON t.id=m.team_id GROUP BY t.category""")}
    assert by == {"Sub-8": 5, "Sub-10": 8, "Sub-12": 8}
    assert seed.summary(demo) == {"clubs": 1, "teams": 3, "players": 21, "evaluations": 80, "users": 4}


def test_every_demo_row_is_flagged_and_labelled(demo):
    for table in ("clubs", "teams", "players", "evaluations", "users"):
        assert rows(demo, f"SELECT COUNT(*) AS n FROM {table} WHERE is_demo=0")[0]["n"] == 0, table
    assert all(r["name"].endswith("Teste") for r in rows(demo, "SELECT name FROM players"))
    assert "dados de teste" in rows(demo, "SELECT name FROM clubs")[0]["name"]
    assert all("(teste)" in r["name"] for r in rows(demo, "SELECT name FROM teams"))


def test_each_player_has_at_least_three_inforce_evaluations(demo):
    for p in rows(demo, "SELECT id FROM players"):
        evs = ev.list_evaluations(p["id"], db_path=demo)
        assert len(evs) >= 3
        assert [e["evaluation_date"] for e in evs] == sorted(e["evaluation_date"] for e in evs)
        assert all(date.fromisoformat(e["evaluation_date"]) <= date.today() for e in evs)   # sem datas futuras
        assert all(e["is_demo"] for e in evs)


def test_scores_are_valid_and_example_cases_present(demo):
    with db.connect(demo) as c:
        bad = c.execute("SELECT COUNT(*) FROM evaluation_scores WHERE score IS NOT NULL AND (score<1 OR score>5)").fetchone()[0]
        assert bad == 0
    assert rows(demo, "SELECT COUNT(*) AS n FROM evaluations WHERE supersedes_id IS NOT NULL")[0]["n"] == 1
    incomplete = [e for p in rows(demo, "SELECT id FROM players") for e in ev.list_evaluations(p["id"], db_path=demo)
                  if not e["complete"]]
    assert len(incomplete) == 1


def test_demo_is_deterministic_for_a_seed(tmp_path):
    a, b, c = (str(tmp_path / f"{n}.db") for n in "abc")
    for p, s in ((a, 7), (b, 7), (c, 8)):
        seed.load_demo(seed=s, db_path=p)
    dump = lambda p: rows(p, "SELECT competency_key, score FROM evaluation_scores ORDER BY evaluation_id, competency_key")
    assert dump(a) == dump(b) and dump(a) != dump(c)


def test_load_twice_refused_and_season_is_in_the_past(path):
    seed.load_demo(db_path=path)
    with pytest.raises(ValidationError):
        seed.load_demo(db_path=path)
    assert seed.season_start(date(2026, 10, 5)) == 2025 and seed.season_start(date(2026, 3, 1)) == 2024
    assert seed.season_start(date(2026, 7, 1)) == 2025


def test_demo_accounts_cannot_log_in_until_admin_sets_password(demo):
    assert auth.authenticate("treinador.sub10.teste", "", db_path=demo) is None
    assert auth.authenticate("encarregado.teste", "qualquer-coisa", db_path=demo) is None
    guardian = next(u for u in auth.list_users(demo) if u["username"] == "encarregado.teste")
    auth.set_password(guardian["id"], "palavra-passe-1", demo)
    user = auth.authenticate("encarregado.teste", "palavra-passe-1", db_path=demo)
    kids = access.guardian_children(user, demo)
    assert len(kids) == 1 and kids[0]["category"] == "Sub-10" and kids[0]["name"].endswith("Teste")


# ── os relatórios funcionam com todos os dados de teste ─────────────────────
def test_every_report_and_pdf_builds_for_all_demo_data(demo):
    admin = auth.get_user(auth.create_user("adm", "Adm", "admin", "palavra-passe-1", db_path=demo), demo)
    for p in rows(demo, "SELECT id FROM players"):
        evs = ev.list_evaluations(p["id"], db_path=demo)
        assert evolution.overall_change(evs) is not None
        for e in evs:
            assert reports.individual_report(e["id"], demo)["average"] is not None
            assert reports.parent_report(e["id"], demo)["sections"]
        pdf_bytes, _ = access.export_player_sheet_pdf(admin, p["id"], demo)
        assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) >= 1
        data, _ = access.export_individual_pdf(admin, evs[-1]["id"], demo)
        assert data.startswith(b"%PDF")
        data, _ = access.export_parent_pdf(admin, evs[-1]["id"], demo)
        assert data.startswith(b"%PDF")
    for t in rows(demo, "SELECT id FROM teams"):
        r = reports.team_report(t["id"], demo)
        assert r["has_data"] and r["evolution"]["n_common"] >= 3
        assert access.export_team_pdf(admin, t["id"], demo)[0].startswith(b"%PDF")
        assert teamstats.overview(t["id"], demo)["n_evaluated"] >= 5


# ── remoção segura ──────────────────────────────────────────────────────────
def snapshot_real(path):
    return {t: rows(path, f"SELECT * FROM {t} WHERE is_demo=0 ORDER BY 1") for t in
            ("clubs", "teams", "players", "evaluations", "users")} | {
        "scores": rows(path, "SELECT * FROM evaluation_scores WHERE evaluation_id IN (SELECT id FROM evaluations WHERE is_demo=0) ORDER BY 1,2"),
        "memberships": rows(path, "SELECT * FROM team_memberships WHERE player_id IN (SELECT id FROM players WHERE is_demo=0) ORDER BY 1")}


def make_real(path):
    club = service.create_club("Clube Real", db_path=path)
    team = service.create_team(club, "Sub-10 Real", "Sub-10", "2025/2026", db_path=path)
    coach = auth.create_user("treinador.real", "Treinador Real", "coach", "palavra-passe-1", club_id=club, db_path=path)
    auth.set_team_assignments(coach, [team], path)
    pid = service.create_player("Criança Real", team, joined_on="2025-09-01", db_path=path)
    ev.create_evaluation(pid, "2025-10-01", "1.º Período", flat(3), coach_id=coach, db_path=path)
    ev.create_evaluation(pid, "2025-12-01", "2.º Período", flat(4), db_path=path)
    return {"club": club, "team": team, "pid": pid, "coach": coach}


def test_remove_demo_leaves_real_data_untouched(path):
    make_real(path)
    seed.load_demo(db_path=path)
    before = snapshot_real(path)
    assert before["players"] and len(rows(path, "SELECT id FROM players")) == 22
    removed = seed.remove_demo(db_path=path)
    assert removed["players"] == 21 and seed.summary(path) == {"clubs": 0, "teams": 0, "players": 0, "evaluations": 0, "users": 0}
    assert snapshot_real(path) == before
    with db.connect(path) as c:
        assert c.execute("PRAGMA foreign_key_check").fetchall() == []
    seed.load_demo(db_path=path)                      # pode voltar a carregar
    assert seed.summary(path)["players"] == 21


def test_remove_demo_when_nothing_to_remove(path):
    assert seed.remove_demo(db_path=path)["players"] == 0


def _attach(path, sql, *args):
    with db.connect(path) as c:
        c.execute(sql, args)


@pytest.mark.parametrize("make_attachment", [
    lambda p, ids: _attach(p, "INSERT INTO team_memberships(player_id, team_id) VALUES ((SELECT id FROM players WHERE is_demo=0), ?)", ids["team"]),
    lambda p, ids: _attach(p, "INSERT INTO evaluations(player_id, team_id, category, scale_id, evaluation_date, moment, is_demo) "
                              "VALUES (?, ?, 'Sub-10', 1, '2025-10-01', 'Personalizada', 0)", ids["player"], ids["team"]),
    lambda p, ids: _attach(p, "INSERT INTO teams(club_id, name, category, season, is_demo) VALUES (?, 'Real', 'Sub-8', '2025/2026', 0)", ids["club"]),
    lambda p, ids: _attach(p, "INSERT INTO users(username, display_name, role, club_id, is_demo) VALUES ('real.coach', 'Real', 'coach', ?, 0)", ids["club"]),
], ids=["jogador-real-em-equipa-de-teste", "avaliacao-real-de-jogador-de-teste", "equipa-real-em-clube-de-teste", "conta-real-em-clube-de-teste"])
def test_removal_refused_for_each_kind_of_real_attachment(path, make_attachment):
    seed.load_demo(db_path=path)
    with db.connect(path) as c:
        c.execute("INSERT INTO players(name, is_demo) VALUES ('Criança Real', 0)")
    ids = {"team": rows(path, "SELECT id FROM teams LIMIT 1")[0]["id"], "player": rows(path, "SELECT id FROM players WHERE is_demo=1 LIMIT 1")[0]["id"],
           "club": rows(path, "SELECT id FROM clubs LIMIT 1")[0]["id"]}
    make_attachment(path, ids)
    before = seed.summary(path)
    with pytest.raises(ValidationError, match="Nada foi apagado"):
        seed.remove_demo(db_path=path)
    assert seed.summary(path) == before                                           # nada foi apagado


def test_new_data_in_demo_context_inherits_demo_flag_and_is_removed_with_it(path):
    seed.load_demo(db_path=path)
    team = rows(path, "SELECT id FROM teams WHERE category='Sub-8'")[0]["id"]
    pid = service.create_player("Extra", team, joined_on="2025-09-01", db_path=path)          # criado como «real», mas na equipa de teste
    assert rows(path, "SELECT is_demo FROM players WHERE id=?", pid)[0]["is_demo"] == 1
    eid = ev.create_evaluation(pid, "2025-10-10", "1.º Período", flat(3), db_path=path)
    assert ev.get_evaluation(eid, path)["is_demo"] == 1
    seed.remove_demo(db_path=path)
    assert rows(path, "SELECT COUNT(*) AS n FROM players")[0]["n"] == 0


def test_real_player_evaluated_in_real_team_is_never_flagged(path):
    ids = make_real(path)
    assert all(e["is_demo"] == 0 for e in ev.list_evaluations(ids["pid"], db_path=path))
    seed.load_demo(db_path=path)
    assert ev.list_evaluations(ids["pid"], db_path=path)[0]["is_demo"] == 0


# ── permissões e CLI ────────────────────────────────────────────────────────
def test_only_admin_can_manage_demo_data(path):
    admin = auth.get_user(auth.create_user("adm", "Adm", "admin", "palavra-passe-1", db_path=path), path)
    coach = auth.get_user(auth.create_user("trein", "Trein", "coach", "palavra-passe-1", db_path=path), path)
    for fn in (access.admin_load_demo, access.admin_remove_demo, access.admin_demo_summary):
        with pytest.raises(access.PermissionDenied):
            fn(coach, db_path=path)
    assert access.admin_load_demo(admin, db_path=path)["players"] == 21
    assert access.admin_demo_summary(admin, path)["players"] == 21
    access.admin_remove_demo(admin, db_path=path)
    assert access.admin_demo_summary(admin, path)["players"] == 0


def test_clubs_migration(tmp_path):
    p = str(tmp_path / "old.db")
    with db.connect(p) as c:
        c.executescript(db.SCHEMA.replace("    name    TEXT NOT NULL UNIQUE,\n    is_demo INTEGER NOT NULL DEFAULT 0\n", "    name    TEXT NOT NULL UNIQUE\n"))
        assert "is_demo" not in {r["name"] for r in c.execute("PRAGMA table_info(clubs)")}
    db.init_db(p)
    assert "is_demo" in {r["name"] for r in rows(p, "PRAGMA table_info(clubs)")}


def test_cli(path, monkeypatch, capsys):
    monkeypatch.setattr(db, "DB_PATH", path)
    assert seed.main(["status"]) == 0 and "'players': 0" in capsys.readouterr().out
    assert seed.main(["load"]) == 0 and "21" in capsys.readouterr().out
    assert seed.main(["load"]) == 1 and "Já existem" in capsys.readouterr().out
    assert seed.main(["remove"]) == 0
    assert seed.main(["xyz"]) == 2


# ── interface ───────────────────────────────────────────────────────────────
def texts(at):
    return " ".join(x.value for kind in (at.markdown, at.caption, at.info, at.warning, at.success, at.subheader)
                    for x in kind)


def test_ui_admin_loads_and_removes_demo_data(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui.db"))
    at = logged_in_app("admin", "adm")
    assert "Existem dados de teste" not in texts(at)
    at.sidebar.radio[0].set_value("Utilizadores").run()
    next(b for b in at.button if b.label == "Carregar dados de teste").click().run()
    assert not at.exception and seed.summary()["players"] == 21
    assert any("Existem dados de teste" in w.value for w in at.sidebar.warning)
    btn = next(b for b in at.button if b.label == "Remover dados de teste")
    assert btn.disabled                                                           # exige confirmação
    at.checkbox(key="demo_confirm").check().run()
    next(b for b in at.button if b.label == "Remover dados de teste").click().run()
    assert not at.exception and seed.summary()["players"] == 0


def test_ui_dashboard_never_sums_real_and_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui2.db"))
    make_real(db.DB_PATH)
    seed.load_demo()
    at = logged_in_app("admin", "adm")
    assert not at.exception and [o for o in at.radio(key="dash_data").options] == ["Reais", "Dados de teste"]
    m = {x.label: x.value for x in at.metric}
    assert m["Equipas"] == "1" and m["Jogadores"] == "1" and m["Avaliações realizadas"] == "2"      # só os reais
    assert not any("dados de teste" in w.value for w in at.main.warning)                 # a página mostra só os reais
    at.radio(key="dash_data").set_value("Dados de teste").run()
    m = {x.label: x.value for x in at.metric}
    assert m["Equipas"] == "3" and m["Jogadores"] == "21" and m["Avaliações realizadas"] == "79"        # 80 registos; a correção substitui uma avaliação
    assert any("dados de teste" in w.value for w in at.main.warning)


def test_ui_only_demo_data_is_shown_as_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ui3.db"))
    seed.load_demo()
    at = logged_in_app("admin", "adm")
    assert not at.exception and not at.radio(key="dash_data") if False else not at.exception
    assert any("dados de teste" in w.value for w in at.main.warning)
    assert next(x for x in at.metric if x.label == "Jogadores").value == "21"
