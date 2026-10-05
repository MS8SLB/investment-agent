"""Base de dados SQLite da plataforma (data/minibasket.db).

Ficheiro próprio, para não alterar as tabelas do Teste de Movimentos Defensivos
(basketball_eval). As avaliações são imutáveis: não há UPDATE/DELETE de resultados;
uma correção cria uma nova versão (supersedes_id).
"""

import os
import sqlite3
from contextlib import contextmanager

from . import competencies as comp
from . import scale

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "minibasket.db")

CATEGORIES = ("Sub-8", "Sub-10", "Sub-12")
MOMENTS = ("Avaliação Inicial", "1.º Período", "2.º Período", "3.º Período", "Avaliação Final", "Personalizada")
ROLES = ("admin", "coach", "guardian")

SCHEMA = """
CREATE TABLE IF NOT EXISTS clubs (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('admin','coach','guardian')),
    password_hash TEXT,
    club_id       INTEGER REFERENCES clubs(id),
    active        INTEGER NOT NULL DEFAULT 1,
    is_demo       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS teams (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    club_id  INTEGER NOT NULL REFERENCES clubs(id),
    name     TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('Sub-8','Sub-10','Sub-12')),
    season   TEXT NOT NULL,                       -- ex.: 2026/2027
    is_demo  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (club_id, name, season)
);

CREATE TABLE IF NOT EXISTS team_coaches (
    team_id INTEGER NOT NULL REFERENCES teams(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    PRIMARY KEY (team_id, user_id)
);

CREATE TABLE IF NOT EXISTS players (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    photo_path    TEXT,
    birth_date    TEXT,                           -- ISO yyyy-mm-dd
    sex           TEXT CHECK (sex IN ('M','F') OR sex IS NULL),
    notes         TEXT,
    is_demo       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Pertença a equipa por época. Mudar de escalão = nova linha; o histórico mantém-se.
CREATE TABLE IF NOT EXISTS team_memberships (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id    INTEGER NOT NULL REFERENCES players(id),
    team_id      INTEGER NOT NULL REFERENCES teams(id),
    jersey_number INTEGER,
    joined_on    TEXT,                            -- data de entrada na equipa
    left_on      TEXT,
    UNIQUE (player_id, team_id)
);

CREATE TABLE IF NOT EXISTS guardians_players (
    user_id   INTEGER NOT NULL REFERENCES users(id),
    player_id INTEGER NOT NULL REFERENCES players(id),
    PRIMARY KEY (user_id, player_id)
);

-- Treinadores com acesso a um jogador específico (além dos da equipa).
CREATE TABLE IF NOT EXISTS player_access (
    user_id   INTEGER NOT NULL REFERENCES users(id),
    player_id INTEGER NOT NULL REFERENCES players(id),
    PRIMARY KEY (user_id, player_id)
);

CREATE TABLE IF NOT EXISTS competencies (
    key      TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    short    TEXT NOT NULL,
    position INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS scales (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    active INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS scale_levels (
    scale_id INTEGER NOT NULL REFERENCES scales(id),
    value    INTEGER NOT NULL,
    label    TEXT NOT NULL,
    PRIMARY KEY (scale_id, value)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id       INTEGER NOT NULL REFERENCES players(id),
    team_id         INTEGER NOT NULL REFERENCES teams(id),
    category        TEXT NOT NULL CHECK (category IN ('Sub-8','Sub-10','Sub-12')),  -- escalão à data
    scale_id        INTEGER NOT NULL REFERENCES scales(id),
    evaluation_date TEXT NOT NULL,
    moment          TEXT NOT NULL,
    coach_id        INTEGER REFERENCES users(id),
    general_notes   TEXT,
    next_objectives TEXT,                         -- objetivos para o próximo período
    supersedes_id   INTEGER REFERENCES evaluations(id),   -- correção de uma avaliação anterior
    is_demo         INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_eval_player_date ON evaluations(player_id, evaluation_date);

-- Sem coluna de média: a média global calcula-se sempre a partir destes resultados.
CREATE TABLE IF NOT EXISTS evaluation_scores (
    evaluation_id  INTEGER NOT NULL REFERENCES evaluations(id),
    competency_key TEXT NOT NULL REFERENCES competencies(key),
    score          INTEGER,                       -- NULL = não avaliada (avaliação incompleta)
    note           TEXT,
    PRIMARY KEY (evaluation_id, competency_key)
);
"""


@contextmanager
def connect(path: str | None = None):
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: str | None = None) -> None:
    """Cria o esquema e semeia competências e escala por omissão (idempotente)."""
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT OR IGNORE INTO competencies(key, name, short, position) VALUES (?,?,?,?)",
            [(c.key, c.name, c.short, c.position) for c in comp.COMPETENCIES])
        if conn.execute("SELECT 1 FROM scales LIMIT 1").fetchone() is None:
            sid = conn.execute("INSERT INTO scales(name, active) VALUES (?, 1)",
                               (scale.DEFAULT_SCALE_NAME,)).lastrowid
            conn.executemany("INSERT INTO scale_levels(scale_id, value, label) VALUES (?,?,?)",
                             [(sid, v, t) for v, t in scale.DEFAULT_LEVELS])


def active_scale(path: str | None = None) -> dict:
    """Escala ativa: {'id', 'name', 'levels': [(valor, rótulo), ...]}."""
    init_db(path)
    with connect(path) as conn:
        s = conn.execute("SELECT id, name FROM scales WHERE active=1 ORDER BY id DESC LIMIT 1").fetchone()
        lv = conn.execute("SELECT value, label FROM scale_levels WHERE scale_id=? ORDER BY value",
                          (s["id"],)).fetchall()
    return {"id": s["id"], "name": s["name"], "levels": [(r["value"], r["label"]) for r in lv]}
