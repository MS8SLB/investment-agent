"""Consultas e estatística por teste. Nunca mistura testes/unidades diferentes."""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Atleta, Avaliacao, ResultadoTeste, Teste
from .avaliacao import e_melhor


def atletas_filtrados(db: Session, escalao_id=None, equipa_id=None, sexo=None, q=None) -> list[Atleta]:
    stmt = select(Atleta).order_by(Atleta.nome)
    if escalao_id:
        stmt = stmt.where(Atleta.escalao_id == escalao_id)
    if equipa_id:
        stmt = stmt.where(Atleta.equipa_id == equipa_id)
    if sexo:
        stmt = stmt.where(Atleta.sexo == sexo)
    if q:
        stmt = stmt.where(Atleta.nome.ilike(f"%{q}%"))
    return list(db.scalars(stmt))


def resultados(db: Session, teste_id: int, atleta_ids=None, ini: date | None = None,
               fim: date | None = None) -> list[tuple[Atleta, date, float, int]]:
    """[(atleta, data, melhor_valor, avaliacao_id)] ordenado por data."""
    stmt = (select(Atleta, Avaliacao.data, ResultadoTeste.melhor_valor, Avaliacao.id)
            .join(Avaliacao, Avaliacao.atleta_id == Atleta.id)
            .join(ResultadoTeste, ResultadoTeste.avaliacao_id == Avaliacao.id)
            .where(ResultadoTeste.teste_id == teste_id, ResultadoTeste.melhor_valor.is_not(None))
            .order_by(Avaliacao.data, Avaliacao.id))
    if atleta_ids is not None:
        if not atleta_ids:
            return []
        stmt = stmt.where(Atleta.id.in_(list(atleta_ids)))
    if ini:
        stmt = stmt.where(Avaliacao.data >= ini)
    if fim:
        stmt = stmt.where(Avaliacao.data <= fim)
    return [tuple(r) for r in db.execute(stmt)]


def serie_atleta(db: Session, atleta_id: int, teste_id: int, ini=None, fim=None) -> list[dict]:
    return [{"data": d, "valor": v, "avaliacao_id": aid}
            for _, d, v, aid in resultados(db, teste_id, [atleta_id], ini, fim)]


def variacao(teste: Teste, primeiro: float, ultimo: float) -> dict:
    """Variação respeitando o sentido: 'melhorou' é True quando o último é melhor do que o primeiro."""
    delta = ultimo - primeiro
    return {"delta": delta,
            "pct": (delta / primeiro * 100) if primeiro else None,
            "melhorou": e_melhor(ultimo, primeiro, teste.direcao) if delta else None}


def resumo(valores: list[float]) -> dict:
    if not valores:
        return {"n": 0, "media": None, "mediana": None, "dp": None, "min": None, "max": None}
    return {"n": len(valores), "media": statistics.fmean(valores), "mediana": statistics.median(valores),
            "dp": statistics.stdev(valores) if len(valores) > 1 else None,
            "min": min(valores), "max": max(valores)}


def medias_por_data(rows) -> list[dict]:
    """Média coletiva por data: para cada atleta usa o melhor resultado dessa data."""
    por_data: dict[date, dict[int, float]] = defaultdict(dict)
    for at, d, v, _ in rows:
        por_data[d][at.id] = v  # um atleta pode ter 2 avaliações no mesmo dia: fica a última
    out = []
    for d in sorted(por_data):
        vs = list(por_data[d].values())
        out.append({"data": d, **resumo(vs)})
    return out


def ultimo_por_atleta(rows) -> dict[int, tuple[Atleta, date, float]]:
    ult = {}
    for at, d, v, _ in rows:  # rows já ordenado por data
        ult[at.id] = (at, d, v)
    return ult


def histograma(valores: list[float], bins: int = 6) -> dict:
    if not valores:
        return {"labels": [], "counts": []}
    lo, hi = min(valores), max(valores)
    if lo == hi:
        return {"labels": [f"{lo:g}"], "counts": [len(valores)]}
    w = (hi - lo) / bins
    counts = [0] * bins
    for v in valores:
        counts[min(int((v - lo) / w), bins - 1)] += 1
    labels = [f"{lo + i * w:.1f}–{lo + (i + 1) * w:.1f}" for i in range(bins)]
    return {"labels": labels, "counts": counts}


def comparar_datas(teste: Teste, rows, d1: date, d2: date) -> list[dict]:
    """Atletas com resultado nas duas datas; delta = d2 − d1 e se melhorou."""
    a, b = {}, {}
    for at, d, v, _ in rows:
        if d == d1:
            a[at.id] = (at, v)
        if d == d2:
            b[at.id] = (at, v)
    out = []
    for aid in a.keys() & b.keys():
        at, v1 = a[aid]
        v2 = b[aid][1]
        out.append({"atleta": at, "v1": v1, "v2": v2, **variacao(teste, v1, v2)})
    return sorted(out, key=lambda r: r["atleta"].nome)


def comparar_escaloes(rows, minimo_n: int = 3) -> list[dict]:
    """Média do último resultado de cada atleta, por escalão (para o mesmo teste e sexo)."""
    por_esc: dict[str, list[float]] = defaultdict(list)
    for at, _, v in ultimo_por_atleta(rows).values():
        por_esc[at.escalao.nome].append(v)
    return [{"escalao": k, **resumo(v), "suficiente": len(v) >= minimo_n} for k, v in por_esc.items()]
