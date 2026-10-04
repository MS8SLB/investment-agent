import pytest
from sqlalchemy import select

from app.models import Atleta, Teste as TesteModel
Teste = TesteModel
Teste.__test__ = False




def test_pdf_individual_e_coletivo(client, db, cenario):
    a = db.scalar(select(Atleta).where(Atleta.nome == "A1"))
    r = client.get(f"/relatorios/individual.pdf?atleta_id={a.id}")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content[:4] == b"%PDF"
    t = db.scalar(select(Teste).where(Teste.codigo == "velocidade-coordenacao"))
    r = client.get(f"/relatorios/individual.pdf?atleta_id={a.id}&teste_id={t.id}&ini=2025-01-01&fim=2025-12-31")
    assert r.content[:4] == b"%PDF"
    r = client.get(f"/relatorios/coletivo.pdf?escalao_id=2&sexo=F&teste_id={t.id}")
    assert r.status_code == 200 and r.content[:4] == b"%PDF" and len(r.content) > 3000
    r = client.get("/relatorios/coletivo.pdf?equipa_id=99999", follow_redirects=True)
    assert "Nenhum atleta" in r.text
    assert client.get("/relatorios").status_code == 200
