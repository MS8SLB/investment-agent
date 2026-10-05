"""Cálculos puros (sem BD): média global e estado de completude.

A média nunca é introduzida nem guardada: calcula-se sempre a partir dos resultados.
Avaliação incompleta → média das competências avaliadas, assinalada como incompleta.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from typing import Mapping, Optional

from . import competencies as comp


def filled(scores: Mapping[str, Optional[int]]) -> dict[str, int]:
    """Só as competências conhecidas com resultado."""
    return {k: v for k, v in scores.items() if k in comp.BY_KEY and v is not None}


def global_average(scores: Mapping[str, Optional[int]]) -> Optional[float]:
    """Média das competências avaliadas; None se nenhuma foi avaliada."""
    f = filled(scores)
    if not f:
        return None
    return float(Fraction(sum(f.values()), len(f)))


def is_complete(scores: Mapping[str, Optional[int]]) -> bool:
    return len(filled(scores)) == len(comp.COMPETENCIES)


def missing(scores: Mapping[str, Optional[int]]) -> list[str]:
    """Chaves das competências ainda por avaliar, pela ordem da roda."""
    f = filled(scores)
    return [k for k in comp.KEYS if k not in f]


def round2(x: Optional[float]) -> Optional[float]:
    """Arredonda a 2 casas decimais (meio para cima)."""
    if x is None:
        return None
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def fmt(x: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Formato português: vírgula decimal («3,33»); «—» se não houver valor."""
    if x is None:
        return "—"
    q = Decimal(1).scaleb(-decimals)
    s = f"{Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP):{'+' if signed else ''}.{decimals}f}"
    return s.replace(".", ",")


def compare(before: Mapping[str, Optional[int]], after: Mapping[str, Optional[int]]) -> dict:
    """Compara duas avaliações do mesmo jogador, competência a competência.

    delta = depois − antes (positivo = evolução); None se faltar um dos resultados.
    As médias comparam-se só nas competências avaliadas nas duas ocasiões, para que uma
    avaliação incompleta não crie uma evolução aparente.
    """
    rows = []
    for k in comp.KEYS:
        b, a = before.get(k), after.get(k)
        rows.append({"key": k, "before": b, "after": a, "delta": None if b is None or a is None else a - b})
    common = [r for r in rows if r["delta"] is not None]
    avg_b = float(Fraction(sum(r["before"] for r in common), len(common))) if common else None
    avg_a = float(Fraction(sum(r["after"] for r in common), len(common))) if common else None
    return {
        "rows": rows,
        "n_common": len(common),
        "avg_before": avg_b,
        "avg_after": avg_a,
        "avg_delta": None if avg_b is None else float(Fraction(sum(r["delta"] for r in common), len(common))),
    }
