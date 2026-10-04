from datetime import date

import pytest
from sqlalchemy import select

from app.models import Atleta, Teste as TesteModel
Teste = TesteModel
Teste.__test__ = False
from app.services import estatistica as est
from app.services.avaliacao import guardar_resultado, nova_avaliacao


@pytest.fixture(scope="module")
def cenario(client):
    from app.database import SessionLocal
    from app.models import Equipa
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


def test_variacao_respeita_sentido(db, cenario):
    t = db.get(Teste, cenario[0])
    assert est.variacao(t, 13.0, 12.5)["melhorou"] is True     # tempo desceu = melhorou
    assert est.variacao(t, 12.5, 13.0)["melhorou"] is False
    ll = db.get(Teste, cenario[1])
    assert est.variacao(ll, 10, 12)["melhorou"] is True        # mais cestos = melhorou
    assert est.variacao(ll, 12, 10)["melhorou"] is False
    assert est.variacao(ll, 10, 10)["melhorou"] is None


def test_medias_por_data(db, cenario):
    t = db.get(Teste, cenario[0])
    ids = [a.id for a in db.scalars(select(Atleta).where(Atleta.equipa_id == cenario[2]))]
    m = est.medias_por_data(est.resultados(db, t.id, ids))
    assert [x["data"] for x in m] == [date(2025, 1, 10), date(2025, 6, 10)]
    assert m[0]["n"] == 6 and round(m[0]["media"], 2) == 12.5   # melhor tempo (13 e 12), média 12,5
    assert m[1]["media"] < m[0]["media"]


def test_comparar_datas_e_escaloes(db, cenario):
    t = db.get(Teste, cenario[0])
    rows = est.resultados(db, t.id)
    c = est.comparar_datas(t, rows, date(2025, 1, 10), date(2025, 6, 10))
    assert c and all(l["melhorou"] for l in c)
    esc = est.comparar_escaloes(rows)
    assert {e["escalao"] for e in esc} >= {"Sub-10", "Sub-12"}


def test_dados_analise_regras_escaloes(db, cenario):
    t = db.get(Teste, cenario[0])
    d = est.dados_analise(db, t, sexo="F")
    assert d["escaloes"]["permitido"] is True and "aviso" in d["escaloes"]
    assert est.dados_analise(db, t)["escaloes"]["permitido"] is False             # sem sexo
    assert est.dados_analise(db, t, escalao_id=2, sexo="F")["escaloes"]["permitido"] is False


def test_histograma_e_resumo():
    assert est.histograma([1, 2, 3, 4, 5, 6], 3)["counts"] == [2, 2, 2]
    assert est.histograma([5, 5])["counts"] == [2]
    assert est.resumo([])["n"] == 0 and est.resumo([2, 4])["media"] == 3


def test_api(client, cenario):
    r = client.get(f"/analise/dados?teste_id={cenario[0]}&equipa_id={cenario[2]}&d1=2025-01-10&d2=2025-06-10")
    j = r.json()
    assert j["n_atletas"] == 6 and j["comparacao"]["melhoraram"] == 6 and j["teste"]["unidade"] == "s"
    assert client.get("/analise").status_code == 200
