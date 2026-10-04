from sqlalchemy import select

from app.models import Atleta, Escalao, ReferenciaTabela, Teste


def test_testes_por_escalao(client, db):
    ts = list(db.scalars(select(Teste)))
    sel = [f"1:{ts[0].id}"]  # só um teste para o Sub-8
    r = client.post("/configuracao/escaloes", data={"sel": sel}, follow_redirects=True)
    assert "guardados" in r.text
    db.expunge_all()
    assert [t.codigo for t in db.scalar(select(Escalao).where(Escalao.id == 1)).testes] == [ts[0].codigo]
    # repõe: Sub-8 vazio, restantes com todos
    todos = [f"{e}:{t.id}" for e in (2, 3, 4) for t in ts]
    client.post("/configuracao/escaloes", data={"sel": todos})
    db.expunge_all()
    assert db.scalar(select(Escalao).where(Escalao.id == 1)).testes == []


def test_referencias_csv(client, db):
    t = db.scalar(select(Teste).where(Teste.codigo == "lancamentos-livres"))
    csv_ok = "rotulo;minimo;maximo\nBaixo;0;9\nMédio;10;19\nAlto;20;\n".encode()
    r = client.post("/configuracao/referencias", data={"teste_id": t.id, "nome": "Tab X", "fonte": "Estudo Y", "validada": "1", "escalao_id": "2"},
                    files={"ficheiro": ("t.csv", csv_ok, "text/csv")}, follow_redirects=True)
    assert "importada" in r.text
    # sem confirmação de validação / fonte → recusa
    r = client.post("/configuracao/referencias", data={"teste_id": t.id, "nome": "Z", "fonte": "F"},
                    files={"ficheiro": ("t.csv", csv_ok, "text/csv")}, follow_redirects=True)
    assert "validada" in r.text
    r = client.post("/configuracao/referencias", data={"teste_id": t.id, "nome": "Z", "fonte": "F", "validada": "1"},
                    files={"ficheiro": ("t.csv", b"a;b\n1;2", "text/csv")}, follow_redirects=True)
    assert "cabeçalho" in r.text
    from app.services.referencias import classificar, tabela_aplicavel
    db.expunge_all()
    ana = db.scalar(select(Atleta).where(Atleta.nome == "Ana Silva"))   # Sub-10
    tab = tabela_aplicavel(db, t.id, ana.escalao_id, ana.sexo)
    assert tab and [classificar(tab, v) for v in (5, 10, 25)] == ["Baixo", "Médio", "Alto"]
    rui = db.scalar(select(Atleta).where(Atleta.nome == "Rui Costa"))   # Sub-12: sem tabela
    assert tabela_aplicavel(db, t.id, rui.escalao_id, rui.sexo) is None
    assert client.get("/configuracao").status_code == 200
    client.post(f"/configuracao/referencias/{tab.id}/apagar")
    db.expunge_all()
    assert db.scalar(select(ReferenciaTabela)) is None
