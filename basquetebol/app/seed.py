"""Carrega o catálogo (escalões e testes do documento ENB). Idempotente."""
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import CATALOGO_PATH
from .models import Escalao, EscalaoTeste, Teste

CAMPOS_TESTE = ["nome", "referencia", "dominio", "objetivo", "descricao", "protocolo", "material", "unidade",
                "unidade_nome", "direcao", "n_tentativas", "tentativas_min", "valor_min", "valor_max",
                "casas", "resultado_descricao", "ordem"]


def seed(db: Session, path=CATALOGO_PATH) -> None:
    cat = json.loads(path.read_text(encoding="utf-8"))
    esc = {e.codigo: e for e in db.scalars(select(Escalao))}
    for e in cat["escaloes"]:
        if e["codigo"] not in esc:
            esc[e["codigo"]] = Escalao(**e)
            db.add(esc[e["codigo"]])
    novos_testes = {}
    existentes = {t.codigo: t for t in db.scalars(select(Teste))}
    for t in cat["testes"]:
        if t["codigo"] in existentes:   # não sobrescreve edições feitas pelo utilizador
            continue
        obj = Teste(codigo=t["codigo"], a_confirmar="\n".join(t.get("a_confirmar", [])),
                    **{k: t[k] for k in CAMPOS_TESTE if k in t})
        db.add(obj)
        novos_testes[t["codigo"]] = obj
    db.flush()
    todos = {**existentes, **novos_testes}
    if novos_testes and not db.scalar(select(EscalaoTeste).limit(1)):  # só na 1.ª instalação
        for cod_esc, lista in cat["escaloes_por_defeito"].items():
            for cod_t in lista:
                db.add(EscalaoTeste(escalao_id=esc[cod_esc].id, teste_id=todos[cod_t].id))
    db.commit()
