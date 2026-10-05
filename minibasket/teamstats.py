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


def _load(team_id: int, db_path: str | None = None) -> dict:
    """Carrega de uma vez as pertenças e as avaliações em vigor (feitas nesta equipa)."""
    init_db(db_path)
    with connect(db_path) as c:
        members = [dict(r) for r in c.execute(
            "SELECT player_id, joined_on, left_on FROM team_memberships WHERE team_id=?", (team_id,))]
        rows = c.execute(
            """SELECT e.id, e.player_id, e.evaluation_date AS date FROM evaluations e
               WHERE e.team_id=? AND NOT EXISTS (SELECT 1 FROM evaluations x WHERE x.supersedes_id=e.id)
               ORDER BY e.evaluation_date, e.id""", (team_id,)).fetchall()
        scores: dict[int, dict] = {}
        for r in c.execute("""SELECT s.evaluation_id, s.competency_key, s.score FROM evaluation_scores s
                              JOIN evaluations e ON e.id=s.evaluation_id WHERE e.team_id=?""", (team_id,)):
            scores.setdefault(r["evaluation_id"], {})[r["competency_key"]] = r["score"]
    evals = [{"id": r["id"], "player_id": r["player_id"], "date": r["date"],
              "scores": scores.get(r["id"], {})} for r in rows]
    return {"members": members, "evals": evals}


def _snapshot(data: dict, team_id: int, on: str) -> dict:
    ids = sorted({m["player_id"] for m in data["members"]
                  if (m["joined_on"] is None or m["joined_on"] <= on) and (m["left_on"] is None or on < m["left_on"])})
    evaluated = []
    for pid in ids:
        mine = [e for e in data["evals"] if e["player_id"] == pid and e["date"] <= on]
        if mine:
            e = mine[-1]
            evaluated.append({"player_id": pid, "evaluation_id": e["id"], "date": e["date"], "scores": e["scores"]})
    return {"team_id": team_id, "as_of": on, "members": len(ids), "evaluated": evaluated}


def _on(as_of) -> str:
    return (as_of if isinstance(as_of, str) else as_of.isoformat()) if as_of else date.today().isoformat()


def snapshot(team_id: int, as_of=None, db_path: str | None = None) -> dict:
    """{'team_id', 'as_of', 'members', 'evaluated': [{'player_id','evaluation_id','date','scores'}]}"""
    return _snapshot(_load(team_id, db_path), team_id, _on(as_of))


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


def timeline(team_id: int, db_path: str | None = None) -> list[dict]:
    """Um ponto por data de avaliação da equipa: nº de avaliados e média global nessa data.

    Atenção: os jogadores avaliados podem mudar de ponto para ponto; para medir evolução
    com os mesmos jogadores usa-se `snapshot_change`.
    """
    data = _load(team_id, db_path)
    out = []
    for d in sorted({e["date"] for e in data["evals"]}):
        snap = _snapshot(data, team_id, d)
        out.append({"date": d, "n_evaluated": len(snap["evaluated"]), "members": snap["members"],
                    "average": team_average(snap), "snapshot": snap})
    return out


def snapshot_change(before: dict, after: dict) -> dict:
    """Evolução da equipa entre duas fotos, só com jogadores avaliados nas duas ocasiões.

    "Mesmos jogadores" = jogadores com uma avaliação nova entre as duas datas.
    Por competência: média antes/depois e diferença (mesmos jogadores, mesmas competências).
    Média global: média das médias individuais (comparadas só nas competências em comum).
    """
    a = {e["player_id"]: e["scores"] for e in before["evaluated"]}
    b = {e["player_id"]: e["scores"] for e in after["evaluated"]}
    # só jogadores reavaliados: a mesma avaliação "arrastada" nas duas fotos não é evolução
    same = {e["player_id"] for e in before["evaluated"]} & {
        e["player_id"] for e in after["evaluated"]
        if any(x["player_id"] == e["player_id"] and x["evaluation_id"] == e["evaluation_id"] for x in before["evaluated"])}
    common = sorted((set(a) & set(b)) - same)
    rows = []
    for c in comp.COMPETENCIES:
        pairs = [(a[p][c.key], b[p][c.key]) for p in common
                 if a[p].get(c.key) is not None and b[p].get(c.key) is not None]
        mb, ma = _mean([x for x, _ in pairs]), _mean([y for _, y in pairs])
        rows.append({"key": c.key, "n": len(pairs), "before": mb, "after": ma,
                     "delta": None if mb is None else float(Fraction(sum(y - x for x, y in pairs), len(pairs)))})
    per = [calc.compare(a[p], b[p]) for p in common]
    per = [c for c in per if c["avg_delta"] is not None]
    n = len(per)
    avg = lambda k: float(sum(Fraction(c[k]).limit_denominator(10**6) for c in per) / n) if n else None
    return {"rows": rows, "n_common": len(common), "avg_before": avg("avg_before"), "avg_after": avg("avg_after"),
            "avg_delta": avg("avg_delta"), "as_of_before": before["as_of"], "as_of_after": after["as_of"]}


def overview(team_id: int, db_path: str | None = None) -> dict:
    """Resumo da equipa para o Dashboard (foto atual, nº de avaliações, evolução primeira→última data)."""
    data = _load(team_id, db_path)
    dates = sorted({e["date"] for e in data["evals"]})
    snap = _snapshot(data, team_id, _on(None))
    # equipa com avaliações futuras/ainda sem foto atual: usa a última data disponível
    if not snap["evaluated"] and dates:
        snap = _snapshot(data, team_id, dates[-1])
    change = None
    if len(dates) >= 2:
        change = snapshot_change(_snapshot(data, team_id, dates[0]), _snapshot(data, team_id, dates[-1]))
    avgs = [calc.global_average(e["scores"]) for e in snap["evaluated"]]
    return {"members": snap["members"], "n_evaluated": len(snap["evaluated"]), "average": team_average(snap),
            "n_evaluations": len(data["evals"]), "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None, "change": change,
            "player_averages": [a for a in avgs if a is not None]}
