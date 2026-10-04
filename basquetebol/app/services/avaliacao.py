"""Regras de registo: leitura de valores, validação e melhor resultado (respeitando o sentido do teste)."""
from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from ..models import Atleta, Avaliacao, ResultadoTeste, Teste, Tentativa


def parse_valor(texto) -> float | None:
    """'12,45' ou '12.45' -> 12.45; vazio -> None; lixo -> ValueError."""
    if texto is None:
        return None
    s = str(texto).strip().replace(" ", "").replace(",", ".")
    if s == "":
        return None
    v = float(s)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("valor inválido")
    return v


def melhor(valores: list[float], direcao: str) -> float | None:
    if not valores:
        return None
    return min(valores) if direcao == "menor" else max(valores)


def e_melhor(a: float, b: float, direcao: str) -> bool:
    """True se a é estritamente melhor do que b."""
    return a < b if direcao == "menor" else a > b


def validar_valor(teste: Teste, v: float) -> str | None:
    if teste.valor_min is not None and v < teste.valor_min:
        return f"{v} abaixo do mínimo permitido ({teste.valor_min:g} {teste.unidade})"
    if teste.valor_max is not None and v > teste.valor_max:
        return f"{v} acima do máximo permitido ({teste.valor_max:g} {teste.unidade})"
    if teste.casas == 0 and abs(v - round(v)) > 1e-9:
        return f"{v} tem de ser um número inteiro"
    return None


def fmt(valor: float | None, teste: Teste) -> str:
    if valor is None:
        return "—"
    return f"{valor:.{teste.casas}f}".replace(".", ",")


def guardar_resultado(db: Session, av: Avaliacao, teste: Teste, tentativas: list[tuple[float, str]],
                      observacoes: str = "") -> ResultadoTeste:
    """Cria/substitui o resultado de um teste numa avaliação. tentativas = [(valor, obs), ...]"""
    if len(tentativas) > teste.n_tentativas:
        raise ValueError(f"{teste.nome}: máximo de {teste.n_tentativas} tentativa(s)")
    for v, _ in tentativas:
        erro = validar_valor(teste, v)
        if erro:
            raise ValueError(f"{teste.nome}: {erro}")
    res = next((r for r in av.resultados if r.teste_id == teste.id), None)
    if res is None:
        res = ResultadoTeste(teste_id=teste.id, avaliacao=av)
        db.add(res)
    res.tentativas.clear()
    for i, (v, obs) in enumerate(tentativas, start=1):
        res.tentativas.append(Tentativa(numero=i, valor=v, observacoes=obs or ""))
    res.melhor_valor = melhor([v for v, _ in tentativas], teste.direcao)
    res.observacoes = observacoes or ""
    return res


def nova_avaliacao(db: Session, atleta: Atleta, data: date, observacoes: str = "") -> Avaliacao:
    av = Avaliacao(atleta=atleta, data=data, escalao_id=atleta.escalao_id, observacoes=observacoes)
    db.add(av)
    return av
