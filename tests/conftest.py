"""Fixture partilhada: base de dados de teste em SQLite e, se TEST_DATABASE_URL existir, também em Postgres."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PG_URL = os.environ.get("TEST_DATABASE_URL")     # BD DESCARTÁVEL: cada teste apaga o esquema public


@pytest.fixture(params=["sqlite"] + (["postgres"] if PG_URL else []))
def bb_db(request, tmp_path):
    from basketball_eval import db as dbmod
    if request.param == "postgres":
        with dbmod.connect(PG_URL) as c:
            c.execute("DROP SCHEMA public CASCADE")
            c.execute("CREATE SCHEMA public")
        dbmod._PG_READY.clear()
        return PG_URL
    return str(tmp_path / "bb.db")
