"""Contas de treinadores (sem Streamlit): utilizadores, palavras-passe com hash, bloqueio e registo de ações.

* Palavras-passe: scrypt com sal aleatório; nunca guardadas nem registadas em claro.
* Bloqueio temporário após MAX_FAILED tentativas falhadas (por utilizador).
* Perfis: «admin» (gere contas) e «coach» (avalia). Todos veem todas as equipas.
* Contas criadas/repostas por um administrador obrigam a trocar a palavra-passe no 1.º acesso.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import db
from .db import connect, init_db

MIN_PASSWORD = 10
MAX_FAILED = 5
LOCK_MINUTES = 5
ROLES = ("admin", "coach")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    username             TEXT NOT NULL UNIQUE,
    display_name         TEXT NOT NULL,
    password_hash        TEXT NOT NULL,
    role                 TEXT NOT NULL DEFAULT 'coach' CHECK (role IN ('admin','coach')),
    active               INTEGER NOT NULL DEFAULT 1,
    failed_count         INTEGER NOT NULL DEFAULT 0,
    locked_until         TEXT,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Quem fez o quê (sem chave estrangeira: o registo sobrevive a apagamentos).
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT NOT NULL DEFAULT (datetime('now')),
    username      TEXT NOT NULL,
    action        TEXT NOT NULL,
    evaluation_id INTEGER,
    player_id     INTEGER,
    detail        TEXT
);
"""


class AuthError(ValueError):
    """Erro de validação mostrável ao utilizador."""


class AccountLocked(AuthError):
    pass


@dataclass
class User:
    id: int
    username: str
    display_name: str
    role: str
    active: bool
    must_change_password: bool
    locked_until: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def init(db_path: str | None = None) -> None:
    if db.schema_ready(db_path, "auth"):
        return
    init_db(db_path)
    with connect(db_path) as c:
        c.executescript(SCHEMA)
    db.mark_schema_ready(db_path, "auth")


# ── Palavras-passe ──────────────────────────────────────────────────────────

_N, _R, _P = 2 ** 14, 8, 1


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(calc, bytes.fromhex(h))
    except (ValueError, TypeError):
        return False


_DUMMY = hash_password("dummy-para-igualar-tempos")


def _check_password_policy(password: str) -> None:
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"A palavra-passe deve ter pelo menos {MIN_PASSWORD} caracteres.")


def _norm(username: str) -> str:
    u = (username or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9._-]{3,32}", u):
        raise AuthError("Utilizador inválido: use 3 a 32 letras minúsculas, números, ponto, hífen ou _.")
    return u


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _user(row) -> User:
    return User(row["id"], row["username"], row["display_name"], row["role"], bool(row["active"]),
                bool(row["must_change_password"]), row["locked_until"])


# ── Gestão de contas ────────────────────────────────────────────────────────

def has_users(db_path: str | None = None) -> bool:
    init(db_path)
    with connect(db_path) as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0


def create_user(username: str, display_name: str, password: str, role: str = "coach",
                must_change: bool = True, db_path: str | None = None) -> int:
    u = _norm(username)
    name = (display_name or "").strip()
    if not name:
        raise AuthError("O nome do treinador é obrigatório.")
    if role not in ROLES:
        raise AuthError(f"Perfil inválido: {role!r}.")
    _check_password_policy(password)
    init(db_path)
    with connect(db_path) as c:
        if c.execute("SELECT 1 FROM users WHERE username=?", (u,)).fetchone():
            raise AuthError(f"O utilizador «{u}» já existe.")
        return c.execute(
            "INSERT INTO users(username, display_name, password_hash, role, must_change_password) "
            "VALUES (?,?,?,?,?) RETURNING id",
            (u, name, hash_password(password), role, int(must_change))).fetchone()[0]


def bootstrap_admin(username: str, password: str, db_path: str | None = None) -> bool:
    """Cria o 1.º administrador a partir dos segredos, só se ainda não existirem utilizadores."""
    if has_users(db_path):
        return False
    create_user(username, "Administrador", password, role="admin", must_change=True, db_path=db_path)
    return True


def list_users(db_path: str | None = None) -> list[User]:
    init(db_path)
    with connect(db_path) as c:
        return [_user(r) for r in c.execute("SELECT * FROM users ORDER BY username")]


def get_user(user_id: int, db_path: str | None = None) -> Optional[User]:
    init(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return _user(row) if row else None


def _active_admins(c, excluding: int | None = None) -> int:
    sql, args = "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1", []
    if excluding is not None:
        sql += " AND id<>?"
        args.append(excluding)
    return c.execute(sql, args).fetchone()[0]


def set_active(user_id: int, active: bool, db_path: str | None = None) -> None:
    init(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
        if row is None:
            raise AuthError("Utilizador inexistente.")
        if not active and row["role"] == "admin" and _active_admins(c, excluding=user_id) == 0:
            raise AuthError("Não é possível desativar o último administrador ativo.")
        c.execute("UPDATE users SET active=?, failed_count=0, locked_until=NULL WHERE id=?", (int(active), user_id))


def reset_password(user_id: int, new_password: str, must_change: bool = True, db_path: str | None = None) -> None:
    _check_password_policy(new_password)
    init(db_path)
    with connect(db_path) as c:
        if c.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone() is None:
            raise AuthError("Utilizador inexistente.")
        c.execute("UPDATE users SET password_hash=?, must_change_password=?, failed_count=0, locked_until=NULL "
                  "WHERE id=?", (hash_password(new_password), int(must_change), user_id))


def change_password(user_id: int, old: str, new: str, db_path: str | None = None) -> None:
    """O próprio utilizador troca a palavra-passe (exige a atual)."""
    init(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None or not verify_password(old, row["password_hash"]):
        raise AuthError("A palavra-passe atual está incorreta.")
    if hmac.compare_digest(old.encode(), new.encode()):
        raise AuthError("A nova palavra-passe tem de ser diferente da atual.")
    reset_password(user_id, new, must_change=False, db_path=db_path)


# ── Autenticação ────────────────────────────────────────────────────────────

def authenticate(username: str, password: str, db_path: str | None = None) -> Optional[User]:
    """User se as credenciais estão certas; None se não. AccountLocked se bloqueada temporariamente.

    A mensagem é a mesma para «utilizador inexistente», «palavra-passe errada» e «conta desativada».
    """
    init(db_path)
    u = (username or "").strip().lower()
    with connect(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (u,)).fetchone()
        if row is None:
            verify_password(password or "", _DUMMY)         # iguala o tempo de resposta
            return None
        if row["locked_until"] and row["locked_until"] > _fmt(_now()):
            raise AccountLocked(f"Conta bloqueada temporariamente até às {row['locked_until'][11:16]} (UTC).")
        ok = verify_password(password or "", row["password_hash"]) and bool(row["active"])
        if ok:
            c.execute("UPDATE users SET failed_count=0, locked_until=NULL WHERE id=?", (row["id"],))
            return _user(row)
        failed = row["failed_count"] + 1
        if failed >= MAX_FAILED:
            c.execute("UPDATE users SET failed_count=0, locked_until=? WHERE id=?",
                      (_fmt(_now() + timedelta(minutes=LOCK_MINUTES)), row["id"]))
        else:
            c.execute("UPDATE users SET failed_count=? WHERE id=?", (failed, row["id"]))
        return None


# ── Registo de ações ────────────────────────────────────────────────────────

def audit(username: str, action: str, evaluation_id: int | None = None, player_id: int | None = None,
          detail: str | None = None, db_path: str | None = None) -> None:
    init(db_path)
    with connect(db_path) as c:
        c.execute("INSERT INTO audit_log(username, action, evaluation_id, player_id, detail) VALUES (?,?,?,?,?)",
                  (username, action, evaluation_id, player_id, detail))


def list_audit(limit: int = 100, db_path: str | None = None) -> list[dict]:
    init(db_path)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]
