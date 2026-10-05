"""Estatísticas da equipa (funções sobre a BD; não expõem nomes nem criam rankings).

Foto da equipa numa data: para cada jogador que pertencia à equipa nessa data, a avaliação
mais recente (em vigor) feita nessa equipa até essa data. Estatísticas por competência:
média, mediana, melhor e mais baixo resultado, e nº de jogadores avaliados.
"""

from __future__ import annotations

import statistics
from datetime import date
from fractions import Fraction
from typing import Optional

from . import calc
from . import competencies as comp
from . import evaluations as ev
from .db import connect, init_db

# Abaixo deste nº de jogadores avaliados não se compara um jogador com a equipa
# (a «média da equipa» seria quase o resultado de um ou dois colegas).
MIN_TEAM_FOR_COMPARISON = 3


def snapshot(team_id: int, as_of=None, db_path: str | None = None) -> dict:
    """{'team_id', 'as_of', 'members', 'evaluated': [{'player_id','evaluation_id','date','scores'}]}"""
    on = (as_of if isinstance(as_of, str) else as_of.isoformat()) if as_of else date.today().isoformat()
    init_db(db_path)
    with connect(db_path) as c:
        members = [r["player_id"] for r in c.execute(
            """SELECT player_id FROM team_memberships WHERE team_id=?
               AND (joined_on IS NULL OR joined_on<=?) AND (left_on IS NULL OR ?<left_on)""", (team_id, on, on))]
    evaluated = []
    for pid in sorted(members):
        mine = [e for e in ev.list_evaluations(pid, db_path=db_path)
                if e["team_id"] == team_id and e["evaluation_date"] <= on]
        if mine:
            e = mine[-1]
            evaluated.append({"player_id": pid, "evaluation_id": e["id"], "date": e["evaluation_date"],
                              "scores": e["scores"]})
    return {"team_id": team_id, "as_of": on, "members": len(members), "evaluated": evaluated}


def _mean(values: list[int]) -> Optional[float]:
    return float(Fraction(sum(values), len(values))) if values else None


def competency_stats(snap: dict) -> list[dict]:
    """Por competência: média, mediana, melhor, mais baixo e n (jogadores com resultado)."""
    out = []
    for c in comp.COMPETENCIES:
        vals = [e["scores"][c.key] for e in snap["evaluated"] if e["scores"].get(c.key) is not None]
        out.append({"key": c.key, "n": len(vals), "mean": _mean(vals),
                    "median": float(statistics.median(vals)) if vals else None,
                    "best": max(vals) if vals else None, "lowest": min(vals) if vals else None})
    return out


def team_average(snap: dict) -> Optional[float]:
    """Média global da equipa: média das médias globais dos jogadores avaliados (cada jogador conta uma vez)."""
    avgs = [calc.global_average(e["scores"]) for e in snap["evaluated"]]
    avgs = [a for a in avgs if a is not None]
    return float(Fraction(sum(Fraction(a).limit_denominator(10**6) for a in avgs), len(avgs))) if avgs else None


def compare_to_team(player_scores: dict, stats: list[dict]) -> list[dict]:
    """Jogador vs média da equipa. diferença = jogador − média da equipa (informação, não classificação)."""
    by = {s["key"]: s for s in stats}
    rows = []
    for c in comp.COMPETENCIES:
        p, m = player_scores.get(c.key), by[c.key]["mean"]
        rows.append({"key": c.key, "player": p, "team_mean": m, "diff": None if p is None or m is None else p - m,
                     "n": by[c.key]["n"]})
    return rows


def can_compare(snap: dict) -> bool:
    return len(snap["evaluated"]) >= MIN_TEAM_FOR_COMPARISON
