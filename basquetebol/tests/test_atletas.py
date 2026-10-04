def test_paginas_base(client):
    for u in ["/atletas", "/atletas/novo", "/equipas"]:
        assert client.get(u).status_code == 200


def test_catalogo_carregado(db):
    from sqlalchemy import select
    from app.models import Teste, Escalao
    assert len(list(db.scalars(select(Teste)))) == 7
    sub8 = db.scalar(select(Escalao).where(Escalao.codigo == "sub-8"))
    assert sub8.testes == []          # sem testes definidos no documento para Sub-8


def test_criar_equipa_e_atleta(client):
    r = client.post("/equipas", data={"nome": "Equipa A", "clube": "Clube X", "escalao_id": 2}, follow_redirects=True)
    assert "Equipa criada" in r.text
    r = client.post("/equipas", data={"nome": "Equipa A", "clube": "Clube X", "escalao_id": 2}, follow_redirects=True)
    assert "já existe" in r.text
    r = client.post("/atletas/novo", data={"nome": "Ana Silva", "data_nascimento": "2014-05-01", "sexo": "F",
                                           "escalao_id": 2, "equipa_id": 1}, follow_redirects=True)
    assert r.status_code == 200 and "Ana Silva" in r.text and "Sem referências" in r.text
    assert "Clube X" in r.text  # clube herdado da equipa


def test_validacao_atleta(client):
    r = client.post("/atletas/novo", data={"nome": " ", "data_nascimento": "2014-05-01", "sexo": "F",
                                           "escalao_id": 2}, follow_redirects=True)
    assert "nome é obrigatório" in r.text
    r = client.post("/atletas/novo", data={"nome": "X", "data_nascimento": "2999-01-01", "sexo": "F",
                                           "escalao_id": 2}, follow_redirects=True)
    assert "Data de nascimento inválida" in r.text


def test_filtros(client):
    client.post("/atletas/novo", data={"nome": "Rui Costa", "data_nascimento": "2012-02-02", "sexo": "M",
                                       "escalao_id": 3}, follow_redirects=True)
    r = client.get("/atletas?escalao_id=3")
    assert "Rui Costa" in r.text and "Ana Silva" not in r.text
    r = client.get("/atletas?q=ana")
    assert "Ana Silva" in r.text and "Rui Costa" not in r.text
    r = client.get("/atletas?sexo=M&equipa_id=1")
    assert "Rui Costa" not in r.text
