import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DATABASE_URL"] = "sqlite:///" + tempfile.mkdtemp() + "/teste.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def db():
    from app.database import SessionLocal
    with SessionLocal() as s:
        yield s


@pytest.fixture(scope="session")
def cenario(client):
    """Dados de exemplo: 6 atletas Sub-10/Sub-12 com 2 avaliações em 2 testes."""
    from datetime import date
    from sqlalchemy import select
    from app.database import SessionLocal
    from app.models import Atleta, Equipa, Teste
    from app.services.avaliacao import guardar_resultado, nova_avaliacao
    with SessionLocal() as db:
        ts = {t.codigo: t for t in db.scalars(select(Teste))}
        eq = Equipa(nome="Análise", clube="C", escalao_id=3)
        db.add(eq)
        db.flush()
        for nome, sexo, escalao in [("A1", "F", 2), ("A2", "F", 2), ("A3", "F", 2), ("B1", "F", 3), ("B2", "F", 3), ("B3", "F", 3)]:
            a = Atleta(nome=nome, data_nascimento=date(2013, 1, 1), sexo=sexo, escalao_id=escalao, equipa_id=eq.id)
            db.add(a)
            db.flush()
            base = 13.0 if escalao == 2 else 12.0
            for d, delta in ((date(2025, 1, 10), 0), (date(2025, 6, 10), -0.5)):
                av = nova_avaliacao(db, a, d)
                guardar_resultado(db, av, ts["velocidade-coordenacao"], [(base + delta, ""), (base + delta + 0.4, "")])
                guardar_resultado(db, av, ts["lancamentos-livres"], [(10 + int(-delta * 4), "")])
        db.commit()
        return ts["velocidade-coordenacao"].id, ts["lancamentos-livres"].id, eq.id
