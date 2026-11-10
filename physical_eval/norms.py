"""Cálculo de percentis — reproduz as fórmulas da ficha Excel «Avaliação Física» (ENB).

Regras (iguais às da folha):
* Altura / peso / envergadura / salto: percentil = maior percentil cujo limiar é <= valor
  (maior = melhor). Abaixo do 1.º limiar: 5 (altura/peso/envergadura) ou 10 (salto).
* 20 m corrida / 20 m drible: percentil = melhor percentil cujo limiar é >= tempo
  (menor tempo = melhor). Acima do último limiar: 5.
* Sem referência para a idade/sexo -> None (a folha deixa a célula vazia).
"""

from __future__ import annotations

from typing import Optional

from . import norms_data as nd

# Cada teste: chave -> (rótulo, unidade, melhor="max"|"min")
TESTS = {
    "altura": ("Altura", "cm", "max"),
    "peso": ("Peso", "kg", "max"),
    "envergadura": ("Envergadura", "cm", "max"),
    "salto": ("Salto horizontal", "m", "max"),
    "corrida": ("20 m corrida", "s", "min"),
    "drible": ("20 m drible", "s", "min"),
}

PROVISIONAL_DRIBBLE_AGES = (12, 13, 14, 15)   # projeções provisórias (folha, Normas!BD18)

AGE_RANGES = {
    "altura": (7, 17), "peso": (7, 17), "envergadura": (7, 17),
    "corrida": (7, 17), "drible": (7, 15),
    "salto_F": (7, 15), "salto_M": (13, 15),
}


def _higher_is_better(value: float, limits: tuple, pcts: tuple, floor: int) -> int:
    pct = floor
    for lim, p in zip(limits, pcts):
        if value >= lim:
            pct = p
    return pct


def _lower_is_better(value: float, limits: tuple, pcts: tuple) -> int:
    for lim, p in zip(limits, pcts):
        if value <= lim:
            return p
    return pcts[-1]


def percentile(test: str, value: Optional[float], age: Optional[int], sex: Optional[str] = None) -> Optional[int]:
    """Percentil (5–90) de ``value`` no teste ``test``; None se faltar dado ou referência.

    ``sex``: 'F' | 'M' (só necessário para o salto).
    """
    if value is None or age is None:
        return None
    age = int(age)
    if test in ("altura", "peso", "envergadura"):
        table = {"altura": nd.ALTURA, "peso": nd.PESO, "envergadura": nd.ENVERGADURA}[test]
        row = table.get(age)
        return None if row is None else _higher_is_better(value, row, nd.ALTURA_PCTS, 5)
    if test == "salto":
        if sex not in ("F", "M"):
            return None
        row = (nd.SALTO_F if sex == "F" else nd.SALTO_M).get(age)
        if row is None:
            return None
        if value <= row[0]:           # a folha atribui 10 a quem não excede o P10
            return 10
        return _higher_is_better(value, row, nd.SALTO_PCTS, 10)
    if test in ("corrida", "drible"):
        table = nd.CORRIDA_20M if test == "corrida" else nd.DRIBLE_20M
        row = table.get(age)
        return None if row is None else _lower_is_better(value, row, nd.TEMPO_PCTS)
    raise ValueError(f"Teste desconhecido: {test!r}")


def is_provisional(test: str, age: Optional[int]) -> bool:
    return test == "drible" and age in PROVISIONAL_DRIBBLE_AGES


def reference_table(test: str, sex: Optional[str] = None) -> dict:
    """Tabela {idade: limiares} e respetivos percentis, para consulta na interface."""
    if test == "salto":
        return {"pcts": nd.SALTO_PCTS, "rows": dict(nd.SALTO_F if sex == "F" else nd.SALTO_M)}
    tabs = {"altura": nd.ALTURA, "peso": nd.PESO, "envergadura": nd.ENVERGADURA,
            "corrida": nd.CORRIDA_20M, "drible": nd.DRIBLE_20M}
    pcts = nd.TEMPO_PCTS if test in ("corrida", "drible") else nd.ALTURA_PCTS
    return {"pcts": pcts, "rows": dict(tabs[test])}
