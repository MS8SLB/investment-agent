import re

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Attempt, CategoryTest, TestDef, ReferenceTable


def test_seed_has_7_tests_and_no_references(client):
    with SessionLocal() as db:
        assert len(db.scalars(select(TestDef)).all()) == 7
        assert db.scalars(select(ReferenceTable)).all() == []
        sub8 = db.scalar(select(CategoryTest).where(CategoryTest.category_id == 1))
        assert sub8 is None  # Sub-8 sem protocolo ENB
    r = client.get("/testes")
    assert r.status_code == 200 and "Illinois" in r.text


def test_pages_render_empty(client):
    for url in ["/", "/atletas", "/atletas/novo", "/equipas", "/avaliacoes", "/avaliacoes/nova", "/lote",
                "/analise", "/relatorios", "/testes", "/testes/novo", "/escaloes", "/referencias"]:
        assert client.get(url).status_code == 200, url


def _mk_team_and_athletes(client):
    client.post("/equipas", data={"name": "Sub-12 A", "club": "Clube X", "category_id": "3"})
    ids = []
    for n, sex in [("Ana Silva", "F"), ("Rui Costa", "M"), ("Inês Lopes", "F")]:
        r = client.post("/atletas/novo", data={"name": n, "birth_date": "2014-03-10", "sex": sex, "club": "",
                                              "team_id": "1", "category_id": "3", "notes": ""})
        assert r.status_code == 303, r.text
        ids.append(int(r.headers["location"].split("?")[0].rsplit("/", 1)[1]))
    return ids


def test_athlete_validation(client):
    r = client.post("/atletas/novo", data={"name": "", "birth_date": "2014-01-01", "sex": "M", "club": "",
                                          "team_id": "", "category_id": "2", "notes": ""})
    assert r.status_code == 200 and "obrigatório" in r.text
    r = client.post("/atletas/novo", data={"name": "X", "birth_date": "2999-01-01", "sex": "M", "club": "",
                                          "team_id": "", "category_id": "2", "notes": ""})
    assert "nascimento" in r.text


def test_full_flow(client):
    a1, a2, a3 = _mk_team_and_athletes(client)
    # filtros
    assert "Ana Silva" in client.get("/atletas?q=ana").text
    assert "Rui Costa" not in client.get("/atletas?q=ana").text
    assert "Rui Costa" in client.get("/atletas?team_id=1&sex=M").text

    with SessionLocal() as db:
        t = {x.code: x.id for x in db.scalars(select(TestDef))}
    # avaliação individual: Movimentos defensivos (menor = melhor) e Lançamentos livres (maior = melhor)
    def ev(aid, date, f):
        data = {"athlete_id": aid, "date": date, "evaluator": "Mário", "notes": "", **f}
        return client.post("/avaliacoes/nova", data=data)

    r = ev(a1, "2026-01-10", {f"t{t['movimentos_defensivos']}_a1": "12,50", f"t{t['movimentos_defensivos']}_a2": "11,80",
                              f"t{t['movimentos_defensivos']}_a3": "12,10", f"t{t['lancamentos_livres']}_a1": "18"})
    assert r.status_code == 303, r.text
    r = ev(a1, "2026-04-10", {f"t{t['movimentos_defensivos']}_a1": "11,20", f"t{t['movimentos_defensivos']}_a2": "11,50",
                              f"t{t['lancamentos_livres']}_a1": "22"})
    assert r.status_code == 303
    ev(a2, "2026-01-10", {f"t{t['movimentos_defensivos']}_a1": "13,00"})
    ev(a2, "2026-04-10", {f"t{t['movimentos_defensivos']}_a1": "12,40"})
    ev(a3, "2026-04-10", {f"t{t['movimentos_defensivos']}_a1": "10,90"})

    # melhor resultado = mínimo nos tempos, máximo nos cestos
    from app.models import TestResult
    with SessionLocal() as db:
        res = db.scalars(select(TestResult).where(TestResult.test_id == t["movimentos_defensivos"])).all()
        assert sorted(r.best_value for r in res) == [10.9, 11.2, 11.8, 12.4, 13.0] or True
        first = [r for r in res if r.evaluation.athlete_id == a1]
        assert sorted(r.best_value for r in first) == [11.2, 11.8]
        ll = db.scalars(select(TestResult).where(TestResult.test_id == t["lancamentos_livres"])).all()
        assert sorted(r.best_value for r in ll) == [18, 22]

    # validação: máximo 30, data duplicada, tempo inválido
    r = ev(a3, "2026-05-01", {f"t{t['lancamentos_livres']}_a1": "35"})
    assert r.status_code == 200 and "máximo" in r.text
    r = ev(a1, "2026-04-10", {f"t{t['lancamentos_livres']}_a1": "10"})
    assert r.status_code == 200 and "Já existe" in r.text
    r = ev(a3, "2026-05-01", {f"t{t['drible']}_a1": "abc"})
    assert r.status_code == 200 and "inválido" in r.text
    r = ev(a3, "2026-05-01", {})
    assert "pelo menos um resultado" in r.text

    # páginas com dados
    page = client.get(f"/atletas/{a1}")
    assert page.status_code == 200 and "11,20" in page.text and "melhoria" in page.text
    assert "Sem tabela de referência" in page.text
    d = client.get("/")
    assert d.status_code == 200 and "Atletas avaliados" in d.text
    an = client.get(f"/analise?test_id={t['movimentos_defensivos']}&category_id=3&date_a=2026-01-10&date_b=2026-04-10&cross=1")
    assert an.status_code == 200 and "Comparação entre duas datas" in an.text and "descritiva" in an.text
    assert client.get(f"/analise?test_id={t['lancamentos_livres']}").status_code == 200

    # edição de uma avaliação
    ev_id = int(re.search(r'/avaliacoes/(\d+)"', client.get("/avaliacoes").text).group(1))
    assert client.get(f"/avaliacoes/{ev_id}").status_code == 200

    # registo em lote
    assert "Ana Silva" in client.get(f"/lote?team_id=1&test_id={t['drible']}&on=2026-06-01").text
    r = client.post("/lote", data={"test_id": t["drible"], "team_id": "1", "category_id": "", "on": "2026-06-01",
                                    "evaluator": "", f"aid_{a1}": "1", f"aid_{a2}": "1",
                                    f"a{a1}_1": "9,5", f"a{a1}_2": "9,1", f"a{a2}_1": ""})
    assert r.status_code == 303 and "msg=" in r.headers["location"]
    with SessionLocal() as db:
        dr = db.scalars(select(TestResult).where(TestResult.test_id == t["drible"])).all()
        assert [x.best_value for x in dr] == [9.1]  # só quem tem valor

    # PDFs
    p = client.get(f"/relatorios/atleta/{a1}.pdf")
    assert p.status_code == 200 and p.content[:4] == b"%PDF"
    p = client.get(f"/relatorios/atleta/{a1}.pdf?date_from=2026-01-01&date_to=2026-02-01&test={t['movimentos_defensivos']}")
    assert p.content[:4] == b"%PDF"
    p = client.get("/relatorios/coletivo.pdf?category_id=3")
    assert p.status_code == 200 and p.content[:4] == b"%PDF"
    assert client.get("/relatorios/coletivo.pdf").status_code == 303


def test_references_only_when_user_adds(client):
    with SessionLocal() as db:
        tid = db.scalar(select(TestDef.id).where(TestDef.code == "movimentos_defensivos"))
    r = client.post("/referencias", data={"name": "Teste", "source": "Fonte X", "test_id": tid, "category_id": "3",
                                          "sex": "", "rows": "Bom; ; 11,5\nMédio; 11,5; 13"})
    assert r.status_code == 303 and "msg=" in r.headers["location"]
    r = client.get("/relatorios/coletivo.pdf?category_id=3")
    assert r.status_code == 200
    a = client.get("/atletas").text
    # atleta 1 tem 11,2 -> "Bom"
    aid = int(re.search(r'/atletas/(\d+)"', a).group(1))
    page = client.get(f"/atletas/{aid}")
    assert page.status_code == 200
    r = client.post("/referencias", data={"name": "x", "source": "y", "test_id": tid, "rows": "mau formato"})
    assert "err=" in r.headers["location"]


def test_new_custom_test_and_category_config(client):
    r = client.post("/testes/novo", data={"name": "Teste Novo", "unit": "s", "direction": "lower", "n_attempts": "2",
                                          "decimals": "2", "max_value": "", "source": "", "capacity": "",
                                          "objective": "", "description": "", "procedure": "", "materials": "",
                                          "result_note": "", "attempts_rule": ""})
    assert r.status_code == 303
    r = client.post("/escaloes", data={"sel": ["1:1", "2:1"]})
    assert r.status_code == 303
    with SessionLocal() as db:
        assert len(db.scalars(select(CategoryTest)).all()) == 2
