import pytest
from sqlalchemy import select

from app.models import Atleta, Avaliacao, Teste as TesteModel
Teste = TesteModel
Teste.__test__ = False
from app.services import avaliacao as sv


def test_parse_valor():
    assert sv.parse_valor("12,45") == 12.45 and sv.parse_valor(" 3.1 ") == 3.1
    assert sv.parse_valor("") is None and sv.parse_valor(None) is None
    with pytest.raises(ValueError):
        sv.parse_valor("abc")
    with pytest.raises(ValueError):
        sv.parse_valor("nan")


def test_melhor_respeita_sentido():
    assert sv.melhor([12.5, 11.9, 12.0], "menor") == 11.9     # tempo: menor é melhor
    assert sv.melhor([10, 14, 12], "maior") == 14             # pontos: maior é melhor
    assert sv.melhor([], "menor") is None
    assert sv.e_melhor(11, 12, "menor") and not sv.e_melhor(11, 12, "maior")


def _teste(db, codigo):
    return db.scalar(select(Teste).where(Teste.codigo == codigo))


def test_protocolos_do_documento(db):
    n = {t.codigo: (t.n_tentativas, t.direcao, t.unidade) for t in db.scalars(select(Teste))}
    assert n["movimentos-defensivos"] == (3, "menor", "s")
    assert n["drible"] == (3, "menor", "s")
    assert n["illinois-drible"] == (3, "menor", "s")
    assert n["one-minute-shooting"] == (2, "maior", "pontos")
    assert n["velocidade-coordenacao"] == (2, "menor", "s")
    assert n["lancamentos-livres"] == (1, "maior", "convertidos")
    assert _teste(db, "pontos-marcados").pendencias   # lacunas do documento assinaladas


def test_validacao(db):
    ll = _teste(db, "lancamentos-livres")
    assert sv.validar_valor(ll, 31) and sv.validar_valor(ll, -1) and sv.validar_valor(ll, 12.5)
    assert sv.validar_valor(ll, 30) is None
    assert sv.validar_valor(_teste(db, "drible"), 0)


def _atleta_id(db):
    return db.scalar(select(Atleta).where(Atleta.nome == "Ana Silva")).id


def test_guardar_avaliacao_e_melhor(client, db):
    aid = _atleta_id(db)
    t = _teste(db, "movimentos-defensivos")
    r = client.post("/avaliacoes/nova", data={
        "atleta_id": aid, "data": "2025-10-01", "observacoes": "Início de época", "tid": t.id,
        f"t{t.id}_1": "12,50", f"t{t.id}_2": "11,80", f"t{t.id}_3": "12,10", f"t{t.id}_1_obs": "escorregou"},
        follow_redirects=True)
    assert r.status_code == 200 and "Avaliação guardada" in r.text and "11,80" in r.text
    av = db.scalar(select(Avaliacao).order_by(Avaliacao.id.desc()))
    assert av.resultados[0].melhor_valor == 11.8 and len(av.resultados[0].tentativas) == 3
    assert av.resultados[0].tentativas[0].observacoes == "escorregou"


def test_erros_de_validacao_mantem_formulario(client, db):
    aid = _atleta_id(db)
    t = _teste(db, "lancamentos-livres")
    antes = len(list(db.scalars(select(Avaliacao))))
    r = client.post("/avaliacoes/nova", data={"atleta_id": aid, "data": "2025-10-02", "tid": t.id,
                                              f"t{t.id}_1": "45"})
    assert "acima do máximo" in r.text and 'value="45"' in r.text
    r = client.post("/avaliacoes/nova", data={"atleta_id": aid, "data": "2025-10-02", "tid": t.id})
    assert "pelo menos um resultado" in r.text
    r = client.post("/avaliacoes/nova", data={"atleta_id": aid, "data": "2999-10-02", "tid": t.id, f"t{t.id}_1": "5"})
    assert "não pode ser futura" in r.text
    db.expire_all()
    assert len(list(db.scalars(select(Avaliacao)))) == antes


def test_editar_e_apagar(client, db):
    av = db.scalar(select(Avaliacao).order_by(Avaliacao.id.desc()))
    t = av.resultados[0].teste
    r = client.post(f"/avaliacoes/{av.id}/editar", data={
        "atleta_id": av.atleta_id, "data": "2025-10-01", "tid": t.id, f"t{t.id}_1": "11,00"}, follow_redirects=True)
    assert "Avaliação guardada" in r.text
    db.expire_all()
    av = db.get(Avaliacao, av.id)
    assert av.resultados[0].melhor_valor == 11.0 and len(av.resultados[0].tentativas) == 1
    assert client.get(f"/avaliacoes/{av.id}/editar").status_code == 200
    avid = av.id
    client.post(f"/avaliacoes/{avid}/apagar")
    db.expunge_all()
    assert db.get(Avaliacao, avid) is None


def test_avaliacao_coletiva(client, db):
    t = _teste(db, "one-minute-shooting")
    ids = [a.id for a in db.scalars(select(Atleta))]
    data = {"teste_id": t.id, "data": "2025-11-05", "escalao_id": 2}
    for i in ids:
        data.setdefault("aid", [])
        data["aid"].append(i)
    data[f"t{ids[0]}_1"] = "14"
    data[f"t{ids[0]}_2"] = "17"
    r = client.post("/avaliacoes/coletiva", data=data, follow_redirects=True)
    assert "1 resultado(s) guardado(s)" in r.text
    assert client.get(f"/avaliacoes/coletiva?teste_id={t.id}&escalao_id=2&data=2025-11-05").status_code == 200
    db.expire_all()
    av = db.scalar(select(Avaliacao).where(Avaliacao.atleta_id == ids[0], Avaliacao.data.__eq__(__import__("datetime").date(2025, 11, 5))))
    assert av.resultados[0].melhor_valor == 17     # pontos: maior é melhor


def test_paginas_testes(client, db):
    assert client.get("/testes").status_code == 200
    for t in db.scalars(select(Teste)):
        assert client.get(f"/testes/{t.id}").status_code == 200
    assert client.get("/avaliacoes").status_code == 200
    assert client.get("/avaliacoes/nova").status_code == 200
    assert client.get(f"/avaliacoes/nova?atleta_id={_atleta_id(db)}").status_code == 200
    assert client.get("/avaliacoes/coletiva").status_code == 200


def test_teste_personalizado(client, db):
    r = client.post("/testes/novo", data={"nome": "Teste X", "unidade": "reps", "direcao": "maior",
                                          "n_tentativas": 2}, follow_redirects=True)
    assert "Teste criado" in r.text
    assert _teste(db, "p-teste-x").personalizado
