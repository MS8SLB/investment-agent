"""Evolução individual ao longo das avaliações (funções puras).

Recebem a lista cronológica devolvida por `evaluations.list_evaluations` (só versões em
vigor). Comparam sempre o jogador com ele próprio; nunca com outros jogadores.
"""

from __future__ import annotations

from typing import Optional, Sequence

from . import calc
from . import competencies as comp


def evolution_series(evals: Sequence[dict]) -> list[dict]:
    """Um ponto por avaliação: média global e variação face à avaliação anterior.

    A variação compara as competências avaliadas nas duas ocasiões (`n_common`), por isso
    pode diferir da diferença entre as médias quando uma avaliação está incompleta.
    """
    points, prev = [], None
    for e in evals:
        delta, n_common = None, None
        if prev is not None:
            c = calc.compare(prev["scores"], e["scores"])
            delta, n_common = c["avg_delta"], c["n_common"]
        points.append({
            "id": e["id"], "date": e["evaluation_date"], "moment": e["moment"], "category": e["category"],
            "average": e["average"], "complete": e["complete"], "n_rated": len(calc.filled(e["scores"])),
            "delta": delta, "n_common": n_common,
        })
        prev = e
    return points


def competency_series(evals: Sequence[dict], key: str) -> list[dict]:
    """Evolução de uma competência: um ponto por avaliação (score None se não avaliada).

    A variação é face ao último resultado disponível da mesma competência.
    """
    if key not in comp.BY_KEY:
        raise ValueError(f"Competência desconhecida: {key}")
    out, last = [], None
    for e in evals:
        s = e["scores"].get(key)
        out.append({"id": e["id"], "date": e["evaluation_date"], "moment": e["moment"], "score": s,
                    "delta": None if s is None or last is None else s - last})
        if s is not None:
            last = s
    return out


def competency_table(evals: Sequence[dict]) -> list[dict]:
    """Uma linha por competência: resultados por avaliação e variação do primeiro ao último resultado."""
    rows = []
    for c in comp.COMPETENCIES:
        scores = [e["scores"].get(c.key) for e in evals]
        rated = [s for s in scores if s is not None]
        rows.append({"key": c.key, "scores": scores, "n_rated": len(rated),
                     "first": rated[0] if rated else None, "last": rated[-1] if rated else None,
                     "change": rated[-1] - rated[0] if len(rated) >= 2 else None})
    return rows


def overall_change(evals: Sequence[dict]) -> Optional[dict]:
    """Primeira vs última avaliação; None com menos de duas."""
    if len(evals) < 2:
        return None
    c = calc.compare(evals[0]["scores"], evals[-1]["scores"])
    return {"first_date": evals[0]["evaluation_date"], "last_date": evals[-1]["evaluation_date"],
            "avg_first": c["avg_before"], "avg_last": c["avg_after"], "delta": c["avg_delta"],
            "n_common": c["n_common"], "n_evaluations": len(evals)}
