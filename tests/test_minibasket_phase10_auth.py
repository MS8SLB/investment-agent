"""Fase 10 — autenticação, palavras-passe e bloqueio."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minibasket import auth, db


@pytest.fixture
def path(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    return p


# ── hash ────────────────────────────────────────────────────────────────────
def test_hash_is_salted_and_verifiable():
    a, b = auth.hash_password("palavra-passe-1"), auth.hash_password("palavra-passe-1")
    assert a != b and a.startswith("pbkdf2_sha256$") and "palavra-passe-1" not in a
    assert auth.verify_password("palavra-passe-1", a) and not auth.verify_password("outra", a)
    assert not auth.verify_password("x", None) and not auth.verify_password("x", "lixo") \
        and not auth.verify_password("x", "md5$1$a$b")


def test_password_never_stored_in_clear(path):
    auth.create_user("ana", "Ana", "coach", "segredo-forte-1", db_path=path)
    with db.connect(path) as c:
        raw = " ".join(str(v) for r in c.execute("SELECT * FROM users") for v in tuple(r))
    assert "segredo-forte-1" not in raw


# ── criação de contas ───────────────────────────────────────────────────────
def test_create_user_validation(path):
    uid = auth.create_user("Maria.Silva", "Maria", "coach", "abcdefgh", db_path=path)
    assert auth.get_user(uid, path)["username"] == "maria.silva"                   # normalizado
    for kw in ({"username": "ab"}, {"username": "com espaço"}, {"username": "MARIA.SILVA"}, {"display_name": " "},
               {"role": "dono"}, {"password": "curta"}, {"password": "ana.ferreira"}, {"club_id": 99}):
        args = {"username": "ana.ferreira", "display_name": "Ana", "role": "coach", "password": "abcdefgh", **kw}
        with pytest.raises(auth.AuthError):
            auth.create_user(db_path=path, **args)


def test_account_without_password_cannot_log_in(path):
    auth.create_user("sempass", "Sem Pass", "coach", None, db_path=path)
    assert auth.authenticate("sempass", "", db_path=path) is None
    assert auth.authenticate("sempass", "qualquer-coisa", db_path=path) is None


def test_get_user_never_exposes_hash(path):
    uid = auth.create_user("ana", "Ana", "coach", "abcdefgh", db_path=path)
    u = auth.get_user(uid, path)
    assert "password_hash" not in u and u["has_password"] == 1
    assert all("password_hash" not in x for x in auth.list_users(path))


# ── primeiro administrador ──────────────────────────────────────────────────
def test_first_admin_only_once(path):
    assert auth.needs_setup(path)
    auth.create_first_admin("admin", "Admin", "palavra-longa-1", db_path=path)
    assert not auth.needs_setup(path)
    with pytest.raises(auth.AuthError):
        auth.create_first_admin("outro", "Outro", "palavra-longa-1", db_path=path)      # sem tomada de controlo


def test_coach_without_password_does_not_end_setup(path):
    auth.create_user("treinador", "Treinador", "coach", "abcdefgh", db_path=path)
    assert auth.needs_setup(path)


# ── autenticação ────────────────────────────────────────────────────────────
def test_authenticate_success_and_failure_are_indistinguishable(path):
    auth.create_user("ana", "Ana", "coach", "abcdefgh", db_path=path)
    ok = auth.authenticate(" ANA ", "abcdefgh", db_path=path)
    assert ok["username"] == "ana" and ok["role"] == "coach" and "password_hash" not in ok
    assert auth.authenticate("ana", "errada", db_path=path) is None
    assert auth.authenticate("naoexiste", "abcdefgh", db_path=path) is None
    assert auth.authenticate("", "", db_path=path) is None


def test_lockout_after_repeated_failures_then_expires(path):
    auth.create_user("ana", "Ana", "coach", "abcdefgh", db_path=path)
    for _ in range(auth.MAX_FAILED):
        assert auth.authenticate("ana", "errada", now=1000, db_path=path) is None
    with pytest.raises(auth.AuthError):                                  # mesmo com a palavra-passe certa
        auth.authenticate("ana", "abcdefgh", now=1001, db_path=path)
    assert auth.authenticate("ana", "abcdefgh", now=1000 + auth.LOCK_SECONDS + 1, db_path=path) is not None


def test_success_resets_failed_counter(path):
    auth.create_user("ana", "Ana", "coach", "abcdefgh", db_path=path)
    for _ in range(auth.MAX_FAILED - 1):
        auth.authenticate("ana", "errada", now=1, db_path=path)
    assert auth.authenticate("ana", "abcdefgh", now=2, db_path=path)
    for _ in range(auth.MAX_FAILED - 1):
        auth.authenticate("ana", "errada", now=3, db_path=path)
    assert auth.authenticate("ana", "abcdefgh", now=4, db_path=path)      # não chegou ao limite


def test_inactive_account_cannot_log_in(path):
    uid = auth.create_user("ana", "Ana", "coach", "abcdefgh", db_path=path)
    auth.create_user("adm", "Adm", "admin", "abcdefgh", db_path=path)
    auth.set_active(uid, False, path)
    assert auth.authenticate("ana", "abcdefgh", db_path=path) is None and auth.get_user(uid, path) is None
    auth.set_active(uid, True, path)
    assert auth.authenticate("ana", "abcdefgh", db_path=path)


# ── palavras-passe ──────────────────────────────────────────────────────────
def test_change_own_password(path):
    uid = auth.create_user("ana", "Ana", "coach", "abcdefgh", db_path=path)
    with pytest.raises(auth.AuthError):
        auth.change_own_password(uid, "errada", "nova-palavra-1", path)
    with pytest.raises(auth.AuthError):
        auth.change_own_password(uid, "abcdefgh", "curta", path)
    auth.change_own_password(uid, "abcdefgh", "nova-palavra-1", path)
    assert auth.authenticate("ana", "abcdefgh", db_path=path) is None
    assert auth.authenticate("ana", "nova-palavra-1", db_path=path)


def test_admin_reset_unlocks_account(path):
    uid = auth.create_user("ana", "Ana", "coach", "abcdefgh", db_path=path)
    for _ in range(auth.MAX_FAILED):
        auth.authenticate("ana", "x", now=10, db_path=path)
    auth.set_password(uid, "redefinida-1", path)
    assert auth.authenticate("ana", "redefinida-1", now=11, db_path=path)
    with pytest.raises(auth.AuthError):
        auth.set_password(999, "redefinida-1", path)


def test_cannot_deactivate_last_admin(path):
    a = auth.create_user("adm", "Adm", "admin", "abcdefgh", db_path=path)
    with pytest.raises(auth.AuthError):
        auth.set_active(a, False, path)
    b = auth.create_user("adm2", "Adm2", "admin", "abcdefgh", db_path=path)
    auth.set_active(a, False, path)
    with pytest.raises(auth.AuthError):
        auth.set_active(b, False, path)


# ── ligações ────────────────────────────────────────────────────────────────
def test_assignments_replace_and_validate(path):
    from minibasket import service
    club = service.create_club("C", db_path=path)
    t1 = service.create_team(club, "A", "Sub-8", "2024/2025", db_path=path)
    t2 = service.create_team(club, "B", "Sub-10", "2024/2025", db_path=path)
    uid = auth.create_user("ana", "Ana", "coach", "abcdefgh", db_path=path)
    auth.set_team_assignments(uid, [t1, t2, t2], path)
    auth.set_team_assignments(uid, [t2], path)
    with db.connect(path) as c:
        assert [r[0] for r in c.execute("SELECT team_id FROM team_coaches WHERE user_id=?", (uid,))] == [t2]
    with pytest.raises(auth.AuthError):
        auth.set_team_assignments(uid, [999], path)
    with pytest.raises(auth.AuthError):
        auth.set_team_assignments(999, [t1], path)
    auth.set_club(uid, club, path)
    with pytest.raises(auth.AuthError):
        auth.set_club(uid, 99, path)


def test_migration_adds_lock_columns(tmp_path):
    p = str(tmp_path / "old.db")
    with db.connect(p) as c:
        c.executescript(db.SCHEMA.replace("    failed_attempts INTEGER NOT NULL DEFAULT 0,   -- tentativas de início de sessão falhadas seguidas\n", "")
                        .replace("    locked_until  REAL,                           -- epoch; conta bloqueada até esta hora\n", ""))
    db.init_db(p)
    with db.connect(p) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
    assert {"failed_attempts", "locked_until"} <= cols
