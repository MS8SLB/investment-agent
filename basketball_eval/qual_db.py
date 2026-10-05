"""Esquema SQLite da avaliação QUALITATIVA (junto de `players` e `coaches` já existentes).

Não altera tabelas existentes: só acrescenta `age_groups`, `teams`,
`evaluations` e `evaluation_items`. A coluna `competency` identifica o
fundamento técnico (hoje só «lancamento»), pelo que Drible, Passe, etc. usam
as mesmas tabelas sem migração.
"""

from __future__ import annotations

from .competencies import DEFAULT_AGE_GROUPS, SCALE_MAX, SCALE_MIN
from .db import connect, init_db

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS age_groups (
    name       TEXT PRIMARY KEY,
    sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS teams (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Cabeçalho da avaliação (uma por jogador, competência e momento).
CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    competency      TEXT NOT NULL,                       -- ex.: 'lancamento'
    player_id       INTEGER NOT NULL REFERENCES players(id),
    team_id         INTEGER REFERENCES teams(id),
    age_group       TEXT NOT NULL REFERENCES age_groups(name),
    coach_id        INTEGER REFERENCES coaches(id),
    evaluation_date TEXT NOT NULL,                       -- ISO yyyy-mm-dd
    observations    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_eval_player
    ON evaluations(player_id, competency, evaluation_date);

-- Pontuação original (1–5) de cada critério avaliado. Sem linha = não avaliado.
CREATE TABLE IF NOT EXISTS evaluation_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    category      TEXT NOT NULL,                         -- dimensão, ex.: 'preparacao'
    criterion     TEXT NOT NULL,                         -- ex.: 'equilibrio_corporal'
    score         INTEGER NOT NULL CHECK (score BETWEEN {SCALE_MIN} AND {SCALE_MAX}),
    UNIQUE (evaluation_id, criterion)
);
"""


def init(db_path: str | None = None) -> None:
    """Cria (se faltar) as tabelas base e as da avaliação qualitativa."""
    init_db(db_path)
    with connect(db_path) as c:
        c.executescript(SCHEMA)
        for i, name in enumerate(DEFAULT_AGE_GROUPS):
            c.execute("INSERT OR IGNORE INTO age_groups(name, sort_order) VALUES (?, ?)", (name, i))
