"""Contas de treinadores: hash, criação, autenticação, bloqueio, perfis e registo de ações."""

import pytest

from basketball_eval import auth
from basketball_eval.db import connect

PW = "uma-frase-longa-1"


def test_hash_is_salted_and_verifiable():
    h1, h2 = auth.hash_password(PW), auth.hash_password(PW)
    assert h1 != h2 and PW not in h1
    assert auth.verify_password(PW, h1) and not auth.verify_password("errada-errada", h1)
    assert not auth.verify_password(PW, "lixo") and not auth.verify_password(PW, "")


def test_create_and_authenticate(bb_db):
    uid = auth.create_user("Rui.Costa", "Rui Costa", PW, db_path=bb_db)
    u = auth.authenticate("  rui.costa ", PW, bb_db)                  # utilizador não distingue maiúsculas/espaços
    assert (u.id, u.username, u.display_name, u.role) == (uid, "rui.costa", "Rui Costa", "coach")
    assert u.must_change_password and not u.is_admin
    with connect(bb_db) as c:                                          # nunca em claro
        assert PW not in c.execute("SELECT password_hash FROM users").fetchone()[0]


def test_wrong_unknown_and_inactive_all_fail_the_same_way(bb_db):
    uid = auth.create_user("ana", "Ana", PW, db_path=bb_db)
    assert auth.authenticate("ana", "errada-errada", bb_db) is None
    assert auth.authenticate("ninguem", PW, bb_db) is None
    auth.create_user("admin1", "Admin", PW, role="admin", db_path=bb_db)    # permite desativar a Ana
    auth.set_active(uid, False, bb_db)
    assert auth.authenticate("ana", PW, bb_db) is None
    auth.set_active(uid, True, bb_db)
    assert auth.authenticate("ana", PW, bb_db) is not None


def test_lockout_after_repeated_failures_then_expires(bb_db):
    auth.create_user("ana", "Ana", PW, db_path=bb_db)
    for _ in range(auth.MAX_FAILED):
        assert auth.authenticate("ana", "errada-errada", bb_db) is None
    with pytest.raises(auth.AccountLocked):
        auth.authenticate("ana", PW, bb_db)                           # nem a certa entra enquanto bloqueada
    with connect(bb_db) as c:
        c.execute("UPDATE users SET locked_until='2000-01-01 00:00:00'")
    assert auth.authenticate("ana", PW, bb_db) is not None


def test_success_resets_failed_counter(bb_db):
    auth.create_user("ana", "Ana", PW, db_path=bb_db)
    for _ in range(auth.MAX_FAILED - 1):
        auth.authenticate("ana", "errada-errada", bb_db)
    assert auth.authenticate("ana", PW, bb_db) is not None
    for _ in range(auth.MAX_FAILED - 1):                              # recomeça do zero: não bloqueia
        auth.authenticate("ana", "errada-errada", bb_db)
    assert auth.authenticate("ana", PW, bb_db) is not None


@pytest.mark.parametrize("bad", ["ab", "Nome Com Espaços", "x" * 40, "ç-acento", ""])
def test_username_validation(bb_db, bad):
    with pytest.raises(auth.AuthError):
        auth.create_user(bad, "X", PW, db_path=bb_db)


def test_password_policy_duplicates_and_role(bb_db):
    with pytest.raises(auth.AuthError, match="pelo menos"):
        auth.create_user("ana", "Ana", "curta", db_path=bb_db)
    with pytest.raises(auth.AuthError, match="Perfil"):
        auth.create_user("ana", "Ana", PW, role="deus", db_path=bb_db)
    with pytest.raises(auth.AuthError, match="nome"):
        auth.create_user("ana", " ", PW, db_path=bb_db)
    auth.create_user("ana", "Ana", PW, db_path=bb_db)
    with pytest.raises(auth.AuthError, match="já existe"):
        auth.create_user("ANA", "Outra", PW, db_path=bb_db)


def test_change_password_requires_current_and_clears_flag(bb_db):
    uid = auth.create_user("ana", "Ana", PW, db_path=bb_db)
    with pytest.raises(auth.AuthError, match="atual"):
        auth.change_password(uid, "errada-errada", "nova-frase-longa", bb_db)
    with pytest.raises(auth.AuthError, match="diferente"):
        auth.change_password(uid, PW, PW, bb_db)
    auth.change_password(uid, PW, "nova-frase-longa", bb_db)
    assert auth.authenticate("ana", PW, bb_db) is None
    u = auth.authenticate("ana", "nova-frase-longa", bb_db)
    assert u is not None and not u.must_change_password


def test_admin_reset_forces_change_and_unlocks(bb_db):
    uid = auth.create_user("ana", "Ana", PW, db_path=bb_db)
    for _ in range(auth.MAX_FAILED):
        auth.authenticate("ana", "errada-errada", bb_db)
    auth.reset_password(uid, "temporaria-123456", db_path=bb_db)
    u = auth.authenticate("ana", "temporaria-123456", bb_db)
    assert u is not None and u.must_change_password


def test_last_active_admin_cannot_be_deactivated(bb_db):
    a1 = auth.create_user("admin1", "A1", PW, role="admin", db_path=bb_db)
    with pytest.raises(auth.AuthError, match="último administrador"):
        auth.set_active(a1, False, bb_db)
    a2 = auth.create_user("admin2", "A2", PW, role="admin", db_path=bb_db)
    auth.set_active(a1, False, bb_db)
    with pytest.raises(auth.AuthError, match="último administrador"):
        auth.set_active(a2, False, bb_db)


def test_bootstrap_admin_only_when_empty(bb_db):
    assert not auth.has_users(bb_db)
    assert auth.bootstrap_admin("admin", PW, bb_db) is True
    assert auth.bootstrap_admin("outro", PW, bb_db) is False
    users = auth.list_users(bb_db)
    assert [(u.username, u.role) for u in users] == [("admin", "admin")]


def test_audit_log_newest_first(bb_db):
    auth.audit("ana", "criar", evaluation_id=1, player_id=2, detail="Rui", db_path=bb_db)
    auth.audit("ana", "apagar", evaluation_id=1, db_path=bb_db)
    log = auth.list_audit(10, bb_db)
    assert [r["action"] for r in log] == ["apagar", "criar"] and log[1]["detail"] == "Rui" and log[0]["at"]
