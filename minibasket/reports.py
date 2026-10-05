"""Relatórios do treinador (individual e da equipa) como estruturas de dados.

A vista Streamlit e, mais tarde, o PDF consomem estas estruturas. Linguagem pedagógica:
descreve necessidades de desenvolvimento, nunca "maus" jogadores; sem rankings de jogadores.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from . import calc
from . import competencies as comp
from . import evaluations as ev
from . import teamstats
from .db import connect, init_db, scale_levels
from .service import ValidationError, get_player

MAX_AREAS = 3


def rank_areas(values: dict[str, Optional[float]], k: int = MAX_AREAS) -> dict:
    """Divide competências em «melhores» e «a desenvolver» (até k de cada, sem sobreposição).

    Desempate pela ordem da roda. Com menos de duas competências, ou todas iguais, não há
    distinção a fazer (`balanced`). `tie_top`/`tie_bottom` assinalam empates na fronteira.
    """
    rated = [(key, values[key]) for key in comp.KEYS if values.get(key) is not None]
    out = {"top": [], "bottom": [], "balanced": False, "tie_top": False, "tie_bottom": False, "n": len(rated)}
    if len(rated) < 2:
        return out
    if len({v for _, v in rated}) == 1:
        out["balanced"] = True
        return out
    order = {key: i for i, key in enumerate(comp.KEYS)}
    k = min(k, len(rated) // 2)
    top = sorted(rated, key=lambda x: (-x[1], order[x[0]]))[:k]
    rest = [x for x in rated if x not in top]
    bottom = sorted(rest, key=lambda x: (x[1], order[x[0]]))[:k]
    out["top"], out["bottom"] = top, bottom
    out["tie_top"] = sum(1 for _, v in rated if v == top[-1][1]) > sum(1 for _, v in top if v == top[-1][1])
    out["tie_bottom"] = sum(1 for _, v in rated if v == bottom[-1][1]) > sum(1 for _, v in bottom if v == bottom[-1][1])
    return out


def _level(levels: dict[int, str], v: Optional[float]) -> Optional[str]:
    return levels.get(int(v)) if v is not None and float(v).is_integer() else None


def individual_report(evaluation_id: int, db_path: str | None = None) -> dict:
    """Relatório de avaliação individual para o treinador."""
    e = ev.get_evaluation(evaluation_id, db_path)
    if not e:
        raise ValidationError("Avaliação inexistente.")
    player = get_player(e["player_id"], db_path)
    levels = dict(scale_levels(e["scale_id"], db_path))
    results = [{"key": c.key, "name": c.name, "score": e["scores"].get(c.key), "note": e["notes"].get(c.key),
                "level": _level(levels, e["scores"].get(c.key))} for c in comp.COMPETENCIES]
    areas = rank_areas(e["scores"])

    def entry(key, score):
        return {"key": key, "name": comp.name_of(key), "score": score, "level": _level(levels, score)}

    strengths = [entry(k, v) for k, v in areas["top"]]
    to_develop = [entry(k, v) for k, v in areas["bottom"]]

    # avaliação anterior (versões em vigor, anteriores em data/ordem de registo)
    history = ev.list_evaluations(e["player_id"], db_path=db_path)
    before = [x for x in history if x["id"] != e["id"] and (x["evaluation_date"], x["id"]) < (e["evaluation_date"], e["id"])]
    evolution = None
    if before:
        prev = before[-1]
        cmp_ = calc.compare(prev["scores"], e["scores"])
        rows = [{"key": r["key"], "name": comp.name_of(r["key"]), "before": r["before"], "after": r["after"],
                 "delta": r["delta"]} for r in cmp_["rows"]]
        evolution = {
            "previous_date": prev["evaluation_date"], "previous_moment": prev["moment"],
            "previous_average": prev["average"], "avg_delta": cmp_["avg_delta"], "n_common": cmp_["n_common"],
            "rows": rows,
            "improved": [r["name"] for r in sorted((r for r in rows if r["delta"] and r["delta"] > 0),
                                                   key=lambda r: -r["delta"])],
            "maintained": [r["name"] for r in rows if r["delta"] == 0],
            "to_consolidate": [r["name"] for r in rows if r["delta"] is not None and r["delta"] < 0],
            "previous_objectives": prev["next_objectives"],
        }
    return {
        "kind": "individual", "evaluation_id": e["id"], "generated_on": date.today().isoformat(),
        "player": {"id": player["id"], "name": player["name"], "photo_path": player["photo_path"]},
        "category": e["category"], "team": e["team"], "club": e["club"], "date": e["evaluation_date"],
        "moment": e["moment"], "coach": e["coach"], "is_demo": bool(e["is_demo"]),
        "complete": e["complete"], "missing": [comp.name_of(k) for k in e["missing"]],
        "scale_max": max(levels) if levels else 5, "levels": levels, "scores": e["scores"],
        "results": results, "average": e["average"],
        "evolution": evolution, "strengths": strengths, "to_develop": to_develop,
        "balanced": areas["balanced"], "tie_strengths": areas["tie_top"], "tie_to_develop": areas["tie_bottom"],
        "general_notes": e["general_notes"], "objectives": e["next_objectives"],
        "suggested_focus": [x["name"] for x in to_develop],
    }


def team_report(team_id: int, db_path: str | None = None) -> dict:
    """Relatório coletivo: sem nomes de jogadores e sem comparações entre eles."""
    init_db(db_path)
    with connect(db_path) as c:
        t = c.execute("""SELECT t.id, t.name, t.category, t.season, t.is_demo, cl.name AS club
                         FROM teams t JOIN clubs cl ON cl.id=t.club_id WHERE t.id=?""", (team_id,)).fetchone()
    if not t:
        raise ValidationError("Equipa inexistente.")
    line = teamstats.timeline(team_id, db_path)
    out = {"kind": "team", "generated_on": date.today().isoformat(), "team": dict(t), "has_data": bool(line),
           "members": 0, "n_evaluated": 0, "average": None, "as_of": None, "stats": [], "evolution": None,
           "attention": [], "strong": [], "balanced_means": False}
    if not line:
        return out
    snap = line[-1]["snapshot"]
    stats = teamstats.competency_stats(snap)
    out.update(members=snap["members"], n_evaluated=len(snap["evaluated"]), average=teamstats.team_average(snap),
               as_of=snap["as_of"],
               stats=[{**s, "name": comp.name_of(s["key"])} for s in stats])
    means = rank_areas({s["key"]: s["mean"] for s in stats})
    out["attention"] = [{"key": k, "name": comp.name_of(k), "mean": v} for k, v in means["bottom"]]
    out["strong"] = [{"key": k, "name": comp.name_of(k), "mean": v} for k, v in means["top"]]
    out["balanced_means"] = means["balanced"]
    if len(line) >= 2:
        ch = teamstats.snapshot_change(line[0]["snapshot"], line[-1]["snapshot"])
        rows = [{"key": r["key"], "name": comp.name_of(r["key"]), "n": r["n"], "before": r["before"],
                 "after": r["after"], "delta": r["delta"]} for r in ch["rows"]]
        gains = rank_areas({r["key"]: r["delta"] for r in rows})
        out["evolution"] = {
            "date_before": line[0]["date"], "date_after": line[-1]["date"], "n_common": ch["n_common"],
            "avg_before": ch["avg_before"], "avg_after": ch["avg_after"], "avg_delta": ch["avg_delta"], "rows": rows,
            "most_improved": [{"key": k, "name": comp.name_of(k), "delta": v} for k, v in gains["top"]],
            "least_improved": [{"key": k, "name": comp.name_of(k), "delta": v} for k, v in gains["bottom"]],
            "homogeneous": gains["balanced"],
        }
    return out
