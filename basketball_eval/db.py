"""Base de dados SQLite (basketball_eval). Ficheiro por omissão: data/basketball_eval.db."""

import json
import os
import sqlite3
from contextlib import contextmanager

from . import protocol as proto

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "basketball_eval.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS coaches (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS players (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    sex        TEXT CHECK (sex IN ('M','F') OR sex IS NULL),
    category   TEXT NOT NULL CHECK (category IN ('Sub-8','Sub-10','Sub-12','Sub-14')),
    team       TEXT,
    birth_date TEXT,                         -- ISO yyyy-mm-dd
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS defensive_movement_tests (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id          INTEGER NOT NULL REFERENCES players(id),
    evaluation_date    TEXT NOT NULL,        -- ISO yyyy-mm-dd
    category           TEXT NOT NULL CHECK (category IN ('Sub-8','Sub-10','Sub-12','Sub-14')),
    trial_1            REAL CHECK (trial_1 IS NULL OR trial_1 > 0),   -- segundos
    trial_2            REAL CHECK (trial_2 IS NULL OR trial_2 > 0),
    trial_3            REAL CHECK (trial_3 IS NULL OR trial_3 > 0),
    trial_1_valid      INTEGER NOT NULL DEFAULT 1,
    trial_2_valid      INTEGER NOT NULL DEFAULT 1,
    trial_3_valid      INTEGER NOT NULL DEFAULT 1,
    first_is_practice  INTEGER NOT NULL DEFAULT 0,  -- T1 = familiarização (não conta)
    best_time          REAL,                 -- MENOR tempo das tentativas válidas
    best_trial         INTEGER CHECK (best_trial IN (1,2,3) OR best_trial IS NULL),
    previous_best_time REAL,
    change_seconds     REAL,                 -- atual - anterior (negativo = melhoria)
    change_percentage  REAL,                 -- (ant - atual)/ant*100 (positivo = melhoria)
    valid              INTEGER NOT NULL DEFAULT 1,  -- 0 = sem tentativa válida / teste a repetir
    location           TEXT,
    session_label      TEXT,                 -- sessão/momento (início, intermédia, final…)
    notes              TEXT,
    coach_id           INTEGER REFERENCES coaches(id),
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_dmt_player_date ON defensive_movement_tests(player_id, evaluation_date);

CREATE TABLE IF NOT EXISTS qualitative_evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id       INTEGER NOT NULL REFERENCES players(id),
    evaluation_date TEXT NOT NULL,           -- ISO yyyy-mm-dd
    category        TEXT NOT NULL CHECK (category IN ('Sub-8','Sub-10','Sub-12','Sub-14')),
    ratings         TEXT NOT NULL,           -- JSON {criterio: 1..4}
    average         REAL NOT NULL,
    strengths_note  TEXT,
    improve_note    TEXT,
    session_label   TEXT,
    coach_id        INTEGER REFERENCES coaches(id),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_qe_player_date ON qualitative_evaluations(player_id, evaluation_date);

-- Valores de referência (opcionais; vazia por omissão — nada é assumido).
CREATE TABLE IF NOT EXISTS reference_norms (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    test_key   TEXT NOT NULL,
    source     TEXT NOT NULL,                -- ex.: 'Matulaitis et al., 2019'
    category   TEXT,
    age        INTEGER,
    sex        TEXT,
    percentile REAL NOT NULL,
    threshold  REAL NOT NULL,                -- tempo limite em segundos
    level      TEXT                          -- nível de desempenho (opcional)
);

CREATE TABLE IF NOT EXISTS test_protocols (
    test_key TEXT PRIMARY KEY,
    config   TEXT NOT NULL                   -- JSON (pontos A–F configuráveis)
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
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        # Semeia o protocolo; só substitui um já guardado se for de versão anterior.
        row = conn.execute("SELECT config FROM test_protocols WHERE test_key='defensive_movement'").fetchone()
        stored = json.loads(row["config"]).get("version", 0) if row else -1
        if stored < proto.DEFAULT_PROTOCOL["version"]:
            conn.execute("INSERT OR REPLACE INTO test_protocols(test_key, config) VALUES (?, ?)",
                         ("defensive_movement", proto.dumps(proto.DEFAULT_PROTOCOL)))
