"""Autenticação: utilizadores, palavras-passe com hash e bloqueio por tentativas falhadas.

Palavras-passe: PBKDF2-HMAC-SHA256 com sal aleatório por utilizador (só biblioteca padrão);
nunca se guardam nem se registam em claro. Sem recuperação por e-mail nesta versão: um
administrador redefine a palavra-passe.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import time
from typing import Optional

from .db import ROLES, connect, init_db

ITERATIONS = 600_000
MIN_PASSWORD = 8
MAX_FAILED = 5
LOCK_SECONDS = 15 * 60
_USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,40}$")
_ROLE_LABELS = {"admin": "Administrador", "coach": "Treinador", "guardian": "Encarregado de educação"}
PUBLIC_COLUMNS = "id, username, display_name, role, club_id, active, is_demo, (password_hash IS NOT NULL) AS has_password"


class AuthError(Exception):
    """Falha de autenticação ou gestão de contas (mensagem apresentável ao utilizador)."""


def role_label(role: str) -> str:
    return _ROLE_LABELS[role]


# ── palavras-passe ──────────────────────────────────────────────────────────
def hash_password(password: str, iterations: int | None = None) -> str:
    iterations = iterations or ITERATIONS
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, stored: str | None) -> bool:
    try:
        algo, iters, salt, digest = (stored or "").split("$")
        if algo != "pbkdf2_sha256":
            return False
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(iters))
        return hmac.compare_digest(calc, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


def _check_password(password: str, username: str | None = None) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise AuthError(f"A palavra-passe deve ter pelo menos {MIN_PASSWORD} caracteres.")
    if username and password.lower() == username.lower():
        raise AuthError("A palavra-passe não pode ser igual ao nome de utilizador.")


def _username(value: str) -> str:
    u = (value or "").strip().lower()
    if not _USERNAME_RE.match(u):
        raise AuthError("Nome de utilizador inválido: use 3 a 40 letras minúsculas, números, '.', '_' ou '-'.")
    return u


# ── contas ──────────────────────────────────────────────────────────────────
def create_user(username: str, display_name: str, role: str, password: str | None = None,
                club_id: int | None = None, is_demo: bool = False, db_path: str | None = None) -> int:
    username = _username(username)
    name = (display_name or "").strip()
    if not name:
        raise AuthError("O nome é obrigatório.")
    if role not in ROLES:
        raise AuthError(f"Perfil inválido: {role!r}.")
    if password is not None:
        _check_password(password, username)
    init_db(db_path)
    with connect(db_path) as c:
        if c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            raise AuthError(f"O nome de utilizador «{username}» já existe.")
        if club_id is not None and not c.execute("SELECT 1 FROM clubs WHERE id=?", (club_id,)).fetchone():
            raise AuthError("Clube inexistente.")
        return c.execute(
            "INSERT INTO users(username, display_name, role, password_hash, club_id, is_demo) VALUES (?,?,?,?,?,?)",
            (username, name, role, hash_password(password) if password is not None else None, club_id,
             int(is_demo))).lastrowid


def needs_setup(db_path: str | None = None) -> bool:
    """True se ainda não existe nenhum administrador ativo com palavra-passe."""
    init_db(db_path)
    with connect(db_path) as c:
        return c.execute("SELECT 1 FROM users WHERE role='admin' AND active=1 AND password_hash IS NOT NULL").fetchone() is None


def create_first_admin(username: str, display_name: str, password: str, db_path: str | None = None) -> int:
    """Cria o administrador inicial; só funciona enquanto não houver nenhum (evita tomada de controlo)."""
    if not needs_setup(db_path):
        raise AuthError("Já existe um administrador.")
    return create_user(username, display_name, "admin", password, db_path=db_path)


def get_user(user_id: int | None, db_path: str | None = None) -> Optional[dict]:
    """Utilizador ativo (sem o hash). None se não existir ou estiver desativado."""
    if user_id is None:
        return None
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute(f"SELECT {PUBLIC_COLUMNS} FROM users WHERE id=? AND active=1", (user_id,)).fetchone()
    return dict(row) if row else None


def list_users(db_path: str | None = None) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(f"SELECT {PUBLIC_COLUMNS} FROM users ORDER BY role, display_name")]


_DUMMY_HASH = None


def authenticate(username: str, password: str, now: float | None = None, db_path: str | None = None) -> Optional[dict]:
    """Utilizador se as credenciais estiverem certas; None caso contrário (mensagem igual para qualquer causa).

    Após MAX_FAILED falhas seguidas a conta fica bloqueada LOCK_SECONDS (AuthError).
    """
    global _DUMMY_HASH
    now = time.time() if now is None else now
    init_db(db_path)
    uname = (username or "").strip().lower()
    with connect(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (uname,)).fetchone()
        if row and row["locked_until"] and row["locked_until"] > now:
            raise AuthError("Conta temporariamente bloqueada por tentativas falhadas. Tente mais tarde.")
        ok = False
        if row and row["active"] and row["password_hash"]:
            ok = verify_password(password or "", row["password_hash"])
        else:                                    # tempo semelhante quer a conta exista quer não
            _DUMMY_HASH = _DUMMY_HASH or hash_password("dummy-password")
            verify_password(password or "", _DUMMY_HASH)
        if not row or not row["active"] or not row["password_hash"]:
            return None
        if not ok:
            failed = row["failed_attempts"] + 1
            if failed >= MAX_FAILED:
                c.execute("UPDATE users SET failed_attempts=0, locked_until=? WHERE id=?", (now + LOCK_SECONDS, row["id"]))
            else:
                c.execute("UPDATE users SET failed_attempts=? WHERE id=?", (failed, row["id"]))
            return None
        c.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?", (row["id"],))
        out = c.execute(f"SELECT {PUBLIC_COLUMNS} FROM users WHERE id=?", (row["id"],)).fetchone()
    return dict(out)


def set_password(user_id: int, new_password: str, db_path: str | None = None) -> None:
    """Define/redefine a palavra-passe (também desbloqueia a conta). A autorização é verificada em `access`."""
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            raise AuthError("Utilizador inexistente.")
        _check_password(new_password, row["username"])
        c.execute("UPDATE users SET password_hash=?, failed_attempts=0, locked_until=NULL WHERE id=?",
                  (hash_password(new_password), user_id))


def change_own_password(user_id: int, current: str, new: str, db_path: str | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT password_hash FROM users WHERE id=? AND active=1", (user_id,)).fetchone()
    if not row or not verify_password(current or "", row["password_hash"]):
        raise AuthError("A palavra-passe atual está incorreta.")
    set_password(user_id, new, db_path)


def set_active(user_id: int, active: bool, db_path: str | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT role, active FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            raise AuthError("Utilizador inexistente.")
        if not active and row["role"] == "admin" and row["active"]:
            others = c.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1 AND password_hash IS NOT NULL "
                               "AND id<>?", (user_id,)).fetchone()[0]
            if others == 0:
                raise AuthError("Não é possível desativar o único administrador.")
        c.execute("UPDATE users SET active=? WHERE id=?", (int(active), user_id))


def set_club(user_id: int, club_id: int | None, db_path: str | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as c:
        if club_id is not None and not c.execute("SELECT 1 FROM clubs WHERE id=?", (club_id,)).fetchone():
            raise AuthError("Clube inexistente.")
        if not c.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone():
            raise AuthError("Utilizador inexistente.")
        c.execute("UPDATE users SET club_id=? WHERE id=?", (club_id, user_id))


def _sync(table: str, col: str, user_id: int, ids: list[int], db_path: str | None, ref: str) -> None:
    init_db(db_path)
    ids = sorted(set(ids))
    with connect(db_path) as c:
        if not c.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone():
            raise AuthError("Utilizador inexistente.")
        for i in ids:
            if not c.execute(f"SELECT 1 FROM {ref} WHERE id=?", (i,)).fetchone():
                raise AuthError("Elemento inexistente na lista.")
        c.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        c.executemany(f"INSERT INTO {table}(user_id, {col}) VALUES (?,?)", [(user_id, i) for i in ids])


def set_team_assignments(user_id: int, team_ids: list[int], db_path: str | None = None) -> None:
    """Equipas que o treinador acompanha (substitui a lista anterior)."""
    _sync("team_coaches", "team_id", user_id, team_ids, db_path, "teams")


def set_guardian_links(user_id: int, player_ids: list[int], db_path: str | None = None) -> None:
    """Educandos de um encarregado de educação (substitui a lista anterior)."""
    _sync("guardians_players", "player_id", user_id, player_ids, db_path, "players")


def set_player_access(user_id: int, player_ids: list[int], db_path: str | None = None) -> None:
    """Acessos individuais a jogadores (além das equipas) para um treinador."""
    _sync("player_access", "player_id", user_id, player_ids, db_path, "players")
