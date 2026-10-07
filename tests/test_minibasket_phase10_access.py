"""Fase 10 — permissões: administrador, treinador e encarregado de educação."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minibasket import access, auth, db, evaluations as ev, service
from minibasket import competencies as comp
from minibasket.access import PermissionDenied


def flat(n, **over):
    return {**{k: n for k in comp.KEYS}, **over}


@pytest.fixture
def w(tmp_path):
    """Dois clubes, três equipas; treinador A (Sub-10 A), treinador B (Sub-10 B), pais de dois jogadores."""
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    c1, c2 = service.create_club("Clube 1", db_path=p), service.create_club("Clube 2", db_path=p)
    t10a = service.create_team(c1, "Sub-10 A", "Sub-10", "2024/2025", db_path=p)
    t10b = service.create_team(c1, "Sub-10 B", "Sub-10", "2024/2025", db_path=p)
    t12 = service.create_team(c2, "Sub-12", "Sub-12", "2024/2025", db_path=p)
    u = lambda name, role, **k: auth.get_user(auth.create_user(name, name.title(), role, "palavra-passe-1", db_path=p, **k), p)
    admin, coach_a, coach_b = u("admin", "admin"), u("coacha", "coach", club_id=c1), u("coachb", "coach", club_id=c1)
    g1, g2 = u("pai1", "guardian"), u("pai2", "guardian")
    auth.set_team_assignments(coach_a["id"], [t10a], p)
    auth.set_team_assignments(coach_b["id"], [t10b, t12], p)
    pl = lambda n, t: service.create_player(n, t, joined_on="2024-01-01", db_path=p)
    ana, rui, eva, leo = pl("Ana", t10a), pl("Rui", t10a), pl("Eva", t10b), pl("Leo", t12)
    auth.set_guardian_links(g1["id"], [ana], p)
    auth.set_guardian_links(g2["id"], [eva, leo], p)
    evals = {}
    for pid in (ana, rui, eva, leo):
        evals[pid] = [ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", flat(2), general_notes=f"INTERNA-{pid}",
                                           notes={"shooting": f"NOTA-{pid}"}, db_path=p),
                      ev.create_evaluation(pid, "2024-12-15", "2.º Período", flat(3), parent_message=f"MSG-{pid}", db_path=p)]
    return dict(p=p, admin=admin, a=coach_a, b=coach_b, g1=g1, g2=g2, t10a=t10a, t10b=t10b, t12=t12, c1=c1, c2=c2,
                ana=ana, rui=rui, eva=eva, leo=leo, evals=evals)


# ── conjuntos visíveis ──────────────────────────────────────────────────────
def test_visible_players_per_role(w):
    p = w["p"]
    assert access.visible_player_ids(w["admin"], p) == {w["ana"], w["rui"], w["eva"], w["leo"]}
    assert access.visible_player_ids(w["a"], p) == {w["ana"], w["rui"]}
    assert access.visible_player_ids(w["b"], p) == {w["eva"], w["leo"]}
    assert access.visible_player_ids(w["g1"], p) == {w["ana"]}
    assert access.visible_player_ids(w["g2"], p) == {w["eva"], w["leo"]}


def test_invalid_session_denied(w):
    for bad in (None, {}, {"id": 1, "role": "admin", "active": False}):
        with pytest.raises(PermissionDenied):
            access.list_teams(bad, db_path=w["p"])


def test_player_access_grants_single_player(w):
    auth.set_player_access(w["a"]["id"], [w["leo"]], w["p"])
    assert w["leo"] in access.visible_player_ids(w["a"], w["p"])
    assert access.get_player(w["a"], w["leo"], w["p"])["name"] == "Leo"
    with pytest.raises(PermissionDenied):                              # mas não ganha a equipa do Leo
        access.team_report(w["a"], w["t12"], w["p"])


# ── treinador ───────────────────────────────────────────────────────────────
def test_coach_sees_only_own_teams_and_players(w):
    p = w["p"]
    assert [t["name"] for t in access.list_teams(w["a"], db_path=p)] == ["Sub-10 A"]
    assert {t["name"] for t in access.list_teams(w["b"], db_path=p)} == {"Sub-10 B", "Sub-12"}
    assert {x["name"] for x in access.search_players(w["a"], db_path=p)} == {"Ana", "Rui"}
    assert access.search_players(w["a"], name="Eva", db_path=p) == []
    assert [c["name"] for c in access.list_clubs(w["a"], db_path=p)] == ["Clube 1"]


def test_coach_cannot_open_other_teams_player_or_data(w):
    p, a = w["p"], w["a"]
    with pytest.raises(PermissionDenied):
        access.get_player(a, w["eva"], p)
    with pytest.raises(PermissionDenied):
        access.list_evaluations(a, w["eva"], db_path=p)
    with pytest.raises(PermissionDenied):
        access.individual_report(a, w["evals"][w["eva"]][0], p)
    with pytest.raises(PermissionDenied):
        access.parent_report(a, w["evals"][w["eva"]][0], p)
    with pytest.raises(PermissionDenied):
        access.update_player(a, w["eva"], "Eva X", db_path=p)
    for fn in (access.team_report, access.team_overview, access.team_timeline, access.team_snapshot):
        with pytest.raises(PermissionDenied):
            fn(a, w["t10b"], db_path=p)


def test_coach_can_work_with_own_players(w):
    p, a = w["p"], w["a"]
    assert access.get_player(a, w["ana"], p)["name"] == "Ana"
    assert len(access.list_evaluations(a, w["ana"], db_path=p)) == 2
    assert access.individual_report(a, w["evals"][w["ana"]][1], p)["player"]["name"] == "Ana"
    assert access.parent_report(a, w["evals"][w["ana"]][1], p)["kind"] == "parent"
    assert access.team_report(a, w["t10a"], p)["n_evaluated"] == 2
    pid = access.create_player(a, "Novo Jogador", w["t10a"], joined_on="2024-02-01", db_path=p)
    assert pid in access.visible_player_ids(a, p)
    with pytest.raises(PermissionDenied):
        access.create_player(a, "Intruso", w["t10b"], db_path=p)


def test_evaluator_is_always_the_logged_in_user(w):
    p = w["p"]
    eid = access.create_evaluation(w["a"], w["ana"], "2025-01-10", "3.º Período", flat(4), db_path=p)
    assert ev.get_evaluation(eid, p)["coach"] == "Coacha"
    with pytest.raises(TypeError):                                      # não há como indicar outro treinador
        access.create_evaluation(w["a"], w["ana"], "2025-01-10", "3.º Período", flat(4), coach_id=w["b"]["id"], db_path=p)
    fixed = access.correct_evaluation(w["a"], eid, "2025-01-10", "3.º Período", flat(5), db_path=p)
    assert ev.get_evaluation(fixed, p)["coach"] == "Coacha"
    with pytest.raises(PermissionDenied):
        access.create_evaluation(w["a"], w["eva"], "2025-01-10", "3.º Período", flat(4), db_path=p)
    with pytest.raises(PermissionDenied):
        access.correct_evaluation(w["a"], w["evals"][w["eva"]][1], "2024-12-15", "2.º Período", flat(4), db_path=p)


def test_admin_can_evaluate_and_is_recorded(w):
    eid = access.create_evaluation(w["admin"], w["leo"], "2025-01-10", "3.º Período", flat(4), db_path=w["p"])
    assert ev.get_evaluation(eid, w["p"])["coach"] == "Admin"


def test_coach_creates_team_only_in_own_club_and_follows_it(w):
    p, a = w["p"], w["a"]
    tid = access.create_team(a, w["c1"], "Sub-8 Nova", "Sub-8", "2024/2025", db_path=p)
    assert tid in access.coached_team_ids(a, p)
    with pytest.raises(PermissionDenied):
        access.create_team(a, w["c2"], "Outra", "Sub-8", "2024/2025", db_path=p)
    with pytest.raises(PermissionDenied):
        access.create_club(a, "Clube 3", db_path=p)
    assert tid not in access.coached_team_ids(w["b"], p)


def test_change_team_needs_both_teams(w):
    p = w["p"]
    with pytest.raises(PermissionDenied):
        access.change_team(w["a"], w["ana"], w["t10b"], "2025-01-01", db_path=p)       # destino não acompanhado
    access.change_team(w["admin"], w["ana"], w["t10b"], "2025-01-01", db_path=p)
    assert w["ana"] in access.visible_player_ids(w["b"], p) and w["ana"] not in access.visible_player_ids(w["a"], p)
    assert len(access.list_evaluations(w["b"], w["ana"], db_path=p)) == 2                 # histórico acompanha o jogador


# ── administrador ───────────────────────────────────────────────────────────
def test_admin_sees_and_manages_everything(w):
    p, adm = w["p"], w["admin"]
    assert len(access.list_teams(adm, db_path=p)) == 3 and len(access.list_clubs(adm, db_path=p)) == 2
    assert len(access.search_players(adm, db_path=p)) == 4
    assert access.team_report(adm, w["t12"], p)["n_evaluated"] == 1
    access.create_club(adm, "Clube 3", db_path=p)
    assert len(access.list_users(adm, p)) == 5
    access.admin_set_team_assignments(adm, w["a"]["id"], [w["t10a"], w["t10b"]], p)
    assert w["eva"] in access.visible_player_ids(w["a"], p)
    assert access.admin_assignments(adm, w["a"]["id"], p)["teams"] == sorted([w["t10a"], w["t10b"]])


def test_only_admin_manages_accounts(w):
    p = w["p"]
    for user in (w["a"], w["g1"]):
        for call in (lambda u: access.list_users(u, p), lambda u: access.admin_create_user(u, "xx1", "X", "coach", db_path=p),
                     lambda u: access.admin_set_password(u, w["a"]["id"], "nova-palavra-1", p),
                     lambda u: access.admin_set_active(u, w["b"]["id"], False, p),
                     lambda u: access.admin_set_team_assignments(u, u["id"], [w["t10b"]], p),
                     lambda u: access.admin_set_guardian_links(u, u["id"], [w["eva"]], p),
                     lambda u: access.admin_set_player_access(u, u["id"], [w["eva"]], p),
                     lambda u: access.admin_set_club(u, u["id"], w["c2"], p)):
            with pytest.raises(PermissionDenied):
                call(user)


def test_admin_cannot_deactivate_self(w):
    with pytest.raises(auth.AuthError):
        access.admin_set_active(w["admin"], w["admin"]["id"], False, w["p"])


# ── encarregado de educação ─────────────────────────────────────────────────
def test_guardian_sees_only_own_child(w):
    p, g1, g2 = w["p"], w["g1"], w["g2"]
    assert [c["name"] for c in access.guardian_children(g1, p)] == ["Ana"]
    assert [c["name"] for c in access.guardian_children(g2, p)] == ["Eva", "Leo"]
    assert set(access.guardian_children(g1, p)[0]) == {"id", "name", "category", "team", "club"}   # sem dados pessoais extra
    assert [e["evaluation_date"] for e in access.guardian_evaluations(g1, w["ana"], p)] == ["2024-09-15", "2024-12-15"]
    with pytest.raises(PermissionDenied):
        access.guardian_evaluations(g1, w["rui"], p)                         # colega de equipa
    with pytest.raises(PermissionDenied):
        access.guardian_evaluations(g1, w["eva"], p)                         # filho de outro encarregado
    with pytest.raises(PermissionDenied):
        access.parent_report(g1, w["evals"][w["rui"]][1], p)
    with pytest.raises(PermissionDenied):
        access.parent_report(g1, w["evals"][w["eva"]][1], p)


def test_guardian_has_no_access_to_coach_functions(w):
    p, g = w["p"], w["g1"]
    calls = [lambda: access.list_teams(g, db_path=p), lambda: access.list_clubs(g, db_path=p),
             lambda: access.search_players(g, db_path=p), lambda: access.get_player(g, w["ana"], p),
             lambda: access.list_evaluations(g, w["ana"], db_path=p),
             lambda: access.individual_report(g, w["evals"][w["ana"]][1], p),
             lambda: access.team_report(g, w["t10a"], p), lambda: access.team_overview(g, w["t10a"], p),
             lambda: access.team_timeline(g, w["t10a"], p), lambda: access.team_snapshot(g, w["t10a"], db_path=p),
             lambda: access.create_player(g, "X", w["t10a"], db_path=p),
             lambda: access.update_player(g, w["ana"], "Ana X", db_path=p),
             lambda: access.change_team(g, w["ana"], w["t10b"], db_path=p),
             lambda: access.create_evaluation(g, w["ana"], "2025-01-10", "3.º Período", flat(5), db_path=p),
             lambda: access.correct_evaluation(g, w["evals"][w["ana"]][1], "2024-12-15", "2.º Período", flat(5), db_path=p),
             lambda: access.create_team(g, w["c1"], "X", "Sub-8", "2024/2025", db_path=p),
             lambda: access.create_club(g, "X", db_path=p)]
    for call in calls:
        with pytest.raises(PermissionDenied):
            call()


def test_coach_cannot_use_guardian_area(w):
    with pytest.raises(PermissionDenied):
        access.guardian_children(w["a"], w["p"])
    with pytest.raises(PermissionDenied):
        access.guardian_evaluations(w["a"], w["ana"], w["p"])


def test_guardian_data_has_no_internal_or_collective_information(w):
    p, g = w["p"], w["g1"]
    evs = access.guardian_evaluations(g, w["ana"], p)
    r = access.parent_report(g, w["evals"][w["ana"]][1], p)
    text = json.dumps([evs, r], ensure_ascii=False)
    for forbidden in (f"INTERNA-{w['ana']}", f"NOTA-{w['ana']}", "Rui", "Eva", "Leo", "team_mean", "stats"):
        assert forbidden not in text, forbidden
    assert f"MSG-{w['ana']}" in text                                      # só a mensagem escrita para os pais
    assert set(evs[0]) == {"id", "evaluation_date", "moment", "category", "scores", "average", "complete", "is_demo"}


def test_guardian_cannot_see_superseded_version(w):
    p = w["p"]
    old = w["evals"][w["ana"]][1]
    ev.correct_evaluation(old, "2024-12-15", "2.º Período", flat(4), db_path=p)
    with pytest.raises(PermissionDenied):
        access.parent_report(w["g1"], old, p)
    assert len(access.guardian_evaluations(w["g1"], w["ana"], p)) == 2     # só as versões em vigor


def test_child_keeps_access_after_changing_team(w):
    p = w["p"]
    service.change_team(w["ana"], w["t12"], "2025-01-01", db_path=p)
    assert [c["category"] for c in access.guardian_children(w["g1"], p)] == ["Sub-12"]
    assert len(access.guardian_evaluations(w["g1"], w["ana"], p)) == 2


def test_unlinking_guardian_removes_access(w):
    auth.set_guardian_links(w["g1"]["id"], [], w["p"])
    assert access.guardian_children(w["g1"], w["p"]) == []
    with pytest.raises(PermissionDenied):
        access.guardian_evaluations(w["g1"], w["ana"], w["p"])


def test_unknown_evaluation_is_validation_error_not_leak(w):
    with pytest.raises(service.ValidationError):
        access.parent_report(w["g1"], 9999, w["p"])
    with pytest.raises(service.ValidationError):
        access.individual_report(w["a"], 9999, w["p"])
