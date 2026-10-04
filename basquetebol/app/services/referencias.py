"""Tabelas de referência validadas. A app não traz nenhuma; só usa as que o responsável importar."""
from __future__ import annotations

import csv
import io

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Atleta, ReferenciaFaixa, ReferenciaTabela, Teste
from .avaliacao import parse_valor


def tabela_aplicavel(db: Session, teste_id: int, escalao_id: int, sexo: str) -> ReferenciaTabela | None:
    """A tabela mais específica (escalão e sexo > escalão > sexo > geral)."""
    melhor_t, melhor_score = None, -1
    for t in db.scalars(select(ReferenciaTabela).where(ReferenciaTabela.teste_id == teste_id)):
        if t.escalao_id not in (None, escalao_id) or t.sexo not in (None, sexo):
            continue
        score = (2 if t.escalao_id else 0) + (1 if t.sexo else 0)
        if score > melhor_score:
            melhor_t, melhor_score = t, score
    return melhor_t


def classificar(tabela: ReferenciaTabela | None, valor: float | None) -> str | None:
    if tabela is None or valor is None:
        return None
    for f in tabela.faixas:
        if (f.minimo is None or valor >= f.minimo) and (f.maximo is None or valor <= f.maximo):
            return f.rotulo
    return "Fora das faixas da tabela"


def referencia_para(db: Session, teste: Teste, atleta: Atleta, valor: float | None) -> dict:
    """{'disponivel': bool, 'classe': str|None, 'tabela': str|None}"""
    t = tabela_aplicavel(db, teste.id, atleta.escalao_id, atleta.sexo)
    if t is None:
        return {"disponivel": False, "classe": None, "tabela": None}
    return {"disponivel": True, "classe": classificar(t, valor), "tabela": t.nome}


def importar_csv(db: Session, teste: Teste, nome: str, fonte: str, escalao_id: int | None,
                 sexo: str | None, conteudo: str) -> ReferenciaTabela:
    """CSV com cabeçalho: rotulo;minimo;maximo  (minimo/maximo inclusivos; vazio = sem limite)."""
    amostra = conteudo[:2000]
    delim = ";" if amostra.count(";") >= amostra.count(",") else ","
    rd = csv.DictReader(io.StringIO(conteudo.lstrip("﻿")), delimiter=delim)
    campos = {(c or "").strip().lower() for c in (rd.fieldnames or [])}
    if not {"rotulo", "minimo", "maximo"} <= campos:
        raise ValueError("O CSV tem de ter o cabeçalho: rotulo;minimo;maximo")
    faixas = []
    for i, row in enumerate(rd, start=1):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        if not row.get("rotulo"):
            continue
        try:
            mn, mx = parse_valor(row["minimo"]), parse_valor(row["maximo"])
        except ValueError:
            raise ValueError(f"Linha {i}: valor numérico inválido")
        if mn is not None and mx is not None and mn > mx:
            raise ValueError(f"Linha {i}: mínimo maior do que o máximo")
        faixas.append(ReferenciaFaixa(rotulo=row["rotulo"], minimo=mn, maximo=mx, ordem=i))
    if not faixas:
        raise ValueError("O CSV não tem linhas de dados")
    tab = ReferenciaTabela(teste_id=teste.id, escalao_id=escalao_id, sexo=sexo or None,
                           nome=nome.strip(), fonte=fonte.strip(), faixas=faixas)
    db.add(tab)
    return tab
