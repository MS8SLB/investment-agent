"""Base de dados SQLite da plataforma. Ficheiro por omissão: data/physical_eval.db
(substituível pela variável de ambiente PHYSICAL_EVAL_DB)."""

import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get("PHYSICAL_EVAL_DB") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "physical_eval.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS athletes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    number     TEXT,                                   -- N.º na ficha (opcional)
    name       TEXT NOT NULL,
    sex        TEXT CHECK (sex IN ('M','F') OR sex IS NULL),
    birth_date TEXT,                                   -- ISO yyyy-mm-dd
    team       TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS assessments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    athlete_id    INTEGER NOT NULL REFERENCES athletes(id) ON DELETE CASCADE,
    eval_date     TEXT NOT NULL,                       -- ISO yyyy-mm-dd
    age           INTEGER NOT NULL,                    -- idade (anos) na data da avaliação
    height_cm     REAL CHECK (height_cm   IS NULL OR height_cm   > 0),
    weight_kg     REAL CHECK (weight_kg   IS NULL OR weight_kg   > 0),
    wingspan_cm   REAL CHECK (wingspan_cm IS NULL OR wingspan_cm > 0),
    jump_1 REAL, jump_2 REAL, jump_3 REAL,             -- m (melhor = maior)
    jump_best     REAL,
    run_1  REAL, run_2  REAL,                          -- s (melhor = menor)
    run_best      REAL,
    dribble_1 REAL, dribble_2 REAL,
    dribble_best  REAL,
    recorded_by   TEXT NOT NULL,                       -- estagiário que registou
    notes         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_assess_athlete_date ON assessments(athlete_id, eval_date);
"""


@contextmanager
def connect(path: str | None = None):
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
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
    with connect(path) as c:
        c.executescript(SCHEMA)
