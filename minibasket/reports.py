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


# ── Relatório para os pais ──────────────────────────────────────────────────
# Nomes naturais das competências para texto corrido.
PARENT_NAMES = {
    "shooting": "o lançamento", "dribbling": "o domínio da bola", "passing": "o passe",
    "reception": "a receção da bola", "footwork": "o trabalho de pés", "finishing": "as finalizações",
    "individual_defense": "a defesa individual",
    "individual_tactics": "a tática individual (tomada de decisão)", "fast_break": "o contra-ataque",
}
# Forma sem artigo, para listas («particularmente em …», «destaca-se em …»).
_BARE = {k: v.split(" ", 1)[1] for k, v in PARENT_NAMES.items()}
_IN = {"o": "no", "a": "na", "os": "nos", "as": "nas"}


def _in(key: str) -> str:
    """«no lançamento», «na defesa individual», «nas finalizações»."""
    art, rest = PARENT_NAMES[key].split(" ", 1)
    return f"{_IN[art]} {rest}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " e " + items[-1]


def _subject(first_name: str, sex: Optional[str]) -> str:
    """«O João» / «A Ana»; sem artigo se o sexo não estiver registado (evita adivinhar)."""
    return {"M": f"O {first_name}", "F": f"A {first_name}"}.get(sex or "", first_name)


def parent_report(evaluation_id: int, db_path: str | None = None) -> dict:
    """Relatório simples e positivo para o encarregado de educação.

    Só contém o próprio jogador: sem médias nem comparações com a equipa, sem outros jogadores,
    sem notas internas do treinador (apenas a «mensagem para os pais», escrita para esse fim).
    """
    e = ev.get_evaluation(evaluation_id, db_path)
    if not e:
        raise ValidationError("Avaliação inexistente.")
    player = get_player(e["player_id"], db_path)
    levels = dict(scale_levels(e["scale_id"], db_path))
    top = max(levels) if levels else 5
    first = player["name"].split()[0]
    who = _subject(first, player["sex"])

    ind = individual_report(evaluation_id, db_path)           # reutiliza a lógica de áreas e evolução
    evo = ind["evolution"]

    # Como está a evoluir?
    prev_scores, show_prev = None, False
    if not evo:
        evo_text = (f"Esta é a primeira avaliação de {first}. Serve como ponto de partida para acompanhar, "
                    "ao longo da época, o seu desenvolvimento.")
    else:
        gained = sorted((r for r in evo["rows"] if r["delta"] and r["delta"] > 0),
                        key=lambda r: (-r["delta"], comp.KEYS.index(r["key"])))
        if gained:
            where = _join([_in(r["key"]) for r in gained[:2]])
            evo_text = f"{who} apresentou uma evolução positiva ao longo deste período, particularmente {where}."
        elif evo["avg_delta"] is not None and evo["avg_delta"] < 0:
            evo_text = (f"Neste período, {first} esteve a consolidar algumas competências. Estamos a acompanhar "
                        "este processo com exercícios de treino adequados.")
        else:
            evo_text = (f"{who} manteve o seu nível neste período, o que é uma base sólida "
                        "para continuar a desenvolver.")
        # a roda comparativa só se mostra quando o conjunto não recuou
        show_prev = evo["avg_delta"] is None or evo["avg_delta"] >= 0
        if show_prev:
            prev = ev.list_evaluations(e["player_id"], db_path=db_path)
            prev = [x for x in prev if x["evaluation_date"] == evo["previous_date"] and x["id"] != e["id"]]
            prev_scores = prev[-1]["scores"] if prev else None
            show_prev = prev_scores is not None

    # Os seus pontos fortes / O que estamos a trabalhar
    strengths = [a["key"] for a in ind["strengths"]]
    focus = [a["key"] for a in ind["to_develop"]]
    if strengths:
        strengths_text = f"{who} destaca-se, neste momento, {_join([_in(k) for k in strengths])}."
    elif ind["balanced"]:
        strengths_text = f"{who} apresenta um perfil equilibrado, com todas as competências avaliadas ao mesmo nível."
    else:
        strengths_text = "Ainda não há dados suficientes para destacar competências."
    if focus:
        working_text = f"Estamos a trabalhar principalmente {_join([PARENT_NAMES[k] for k in focus])}."
    elif ind["balanced"]:
        working_text = "Estamos a trabalhar todas as competências de forma equilibrada."
    else:
        working_text = "Estamos a trabalhar as competências de base do Minibasquete."

    # Objetivos: os do treinador, ou um texto geral a partir do foco
    if e["next_objectives"]:
        goals_text, from_coach = e["next_objectives"], True
    elif focus:
        goals_text = ("Para o próximo período, o objetivo será continuar a desenvolver "
                      f"{_join([PARENT_NAMES[k] for k in focus[:2]])}, com exercícios de treino adequados à idade.")
        from_coach = False
    else:
        goals_text = "Para o próximo período, o objetivo será continuar a desenvolver todas as competências, com prazer e confiança."
        from_coach = False

    skills = [{"key": r["key"], "name": r["name"], "score": r["score"], "level": r["level"],
               "dots": "●" * r["score"] + "○" * (top - r["score"])} for r in ind["results"] if r["score"] is not None]
    return {
        "kind": "parent", "evaluation_id": e["id"], "generated_on": date.today().isoformat(),
        "player": {"name": player["name"], "first_name": first, "photo_path": player["photo_path"]},
        "category": e["category"], "team": e["team"], "club": e["club"], "date": e["evaluation_date"],
        "moment": e["moment"], "coach": e["coach"], "is_demo": bool(e["is_demo"]),
        "incomplete_note": None if e["complete"] else "Algumas competências ainda não foram avaliadas neste momento.",
        "sections": {
            "evolution": {"title": "Como está a evoluir?", "text": evo_text},
            "strengths": {"title": "Os seus pontos fortes", "text": strengths_text, "items": strengths},
            "working": {"title": "O que estamos a trabalhar", "text": working_text, "items": focus},
            "goals": {"title": "Objetivos para a próxima etapa", "text": goals_text, "from_coach": from_coach},
        },
        "wheel": {"scores": e["scores"], "previous_scores": prev_scores if show_prev else None,
                  "previous_date": evo["previous_date"] if show_prev else None, "scale_max": top, "levels": levels},
        "skills": skills, "message": e["parent_message"],
    }
