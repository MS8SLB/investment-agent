"""Base de dados SQLite (basketball_eval). Ficheiro por omissão: data/basketball_eval.db."""

import json
import os
import re
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


def _is_pg(target: str) -> bool:
    return target.startswith(("postgres://", "postgresql://"))


def target_for(path: str | None = None) -> str:
    """Destino efetivo: caminho/URL explícito > $DATABASE_URL (Postgres) > ficheiro SQLite por omissão."""
    return path or os.environ.get("DATABASE_URL") or DB_PATH


# ── Postgres (opcional): o resto do código escreve SQL «estilo SQLite» ──────
_PG_POOLS: dict = {}
_PG_READY: set = set()          # destinos cujo esquema já foi criado nesta sessão


def _pg_sql(sql: str) -> str:
    """Traduz o pouco SQL específico do SQLite usado nos esquemas/consultas."""
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    sql = sql.replace("datetime('now')", "to_char(now() at time zone 'utc', 'YYYY-MM-DD HH24:MI:SS')")
    sql = re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql)     # REAL em Postgres é float4 (perde precisão)
    return sql.replace("?", "%s")


class _PgConn:
    """Interface mínima compatível com sqlite3.Connection (execute / executescript)."""

    def __init__(self, raw):
        import psycopg2.extras
        self.raw = raw
        self._factory = psycopg2.extras.DictCursor    # linhas acessíveis por nome e por índice

    def execute(self, sql: str, args=()):
        cur = self.raw.cursor(cursor_factory=self._factory)
        cur.execute(_pg_sql(sql), tuple(args))
        return cur

    def executescript(self, script: str) -> None:
        script = re.sub(r"--[^\n]*", "", script)
        for stmt in filter(None, (x.strip() for x in script.split(";"))):
            self.execute(stmt)


def _pg_pool(url: str):
    if url not in _PG_POOLS:
        from psycopg2.pool import ThreadedConnectionPool
        _PG_POOLS[url] = ThreadedConnectionPool(1, 8, url)
    return _PG_POOLS[url]


@contextmanager
def connect(path: str | None = None):
    target = target_for(path)
    if _is_pg(target):
        pool = _pg_pool(target)
        raw = pool.getconn()
        try:
            yield _PgConn(raw)
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            pool.putconn(raw)
        return
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    conn = sqlite3.connect(target)
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


def schema_ready(path: str | None = None, part: str = "base") -> bool:
    """Em Postgres cada parte do esquema só é criada uma vez por sessão (evita ~20 comandos por pedido)."""
    target = target_for(path)
    return _is_pg(target) and (target, part) in _PG_READY


def mark_schema_ready(path: str | None = None, part: str = "base") -> None:
    target = target_for(path)
    if _is_pg(target):
        _PG_READY.add((target, part))


def init_db(path: str | None = None) -> None:
    if schema_ready(path):
        return
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        # Semeia o protocolo; só substitui um já guardado se for de versão anterior.
        row = conn.execute("SELECT config FROM test_protocols WHERE test_key='defensive_movement'").fetchone()
        stored = json.loads(row["config"]).get("version", 0) if row else -1
        if stored < proto.DEFAULT_PROTOCOL["version"]:
            conn.execute("INSERT INTO test_protocols(test_key, config) VALUES (?, ?) "
                         "ON CONFLICT(test_key) DO UPDATE SET config = excluded.config",
                         ("defensive_movement", proto.dumps(proto.DEFAULT_PROTOCOL)))
    mark_schema_ready(path, "base")
