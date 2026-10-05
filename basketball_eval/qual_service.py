"""Operações de gravação/consulta da avaliação qualitativa."""

from __future__ import annotations

import unicodedata
from datetime import date
from typing import Mapping, Optional

from . import competencies, qual_db, qualitative as q
from . import service as base
from .db import connect


def _iso(d) -> str:
    return d.isoformat() if isinstance(d, date) else date.fromisoformat(str(d)).isoformat()


def _sort_key(text: str) -> str:
    """Ordem alfabética sem distinguir acentos nem maiúsculas (o ORDER BY do SQLite é binário)."""
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode().casefold()


def _clean(text: Optional[str]) -> Optional[str]:
    text = (text or "").strip()
    return text or None


# ── Contexto: escalões, equipas, jogadores ──────────────────────────────────

def list_age_groups(db_path: str | None = None) -> list[str]:
    qual_db.init(db_path)
    with connect(db_path) as c:
        return [r["name"] for r in c.execute("SELECT name FROM age_groups ORDER BY sort_order, name")]


def add_age_group(name: str, db_path: str | None = None) -> None:
    """Acrescenta um escalão (ex.: «Sub-14»). Para ter jogadores nesse escalão, vê `players.category`."""
    name = (name or "").strip()
    if not name:
        raise ValueError("O nome do escalão é obrigatório.")
    qual_db.init(db_path)
    with connect(db_path) as c:
        n = c.execute("SELECT COALESCE(MAX(sort_order), -1) + 1 FROM age_groups").fetchone()[0]
        c.execute("INSERT OR IGNORE INTO age_groups(name, sort_order) VALUES (?, ?)", (name, n))


def get_or_create_team(name: str, db_path: str | None = None) -> int:
    name = (name or "").strip()
    if not name:
        raise ValueError("O nome da equipa é obrigatório.")
    qual_db.init(db_path)
    with connect(db_path) as c:
        c.execute("INSERT OR IGNORE INTO teams(name) VALUES (?)", (name,))
        return c.execute("SELECT id FROM teams WHERE name=?", (name,)).fetchone()["id"]


def list_teams(db_path: str | None = None) -> list[str]:
    """Equipas registadas + equipas já indicadas nos jogadores (módulo existente)."""
    qual_db.init(db_path)
    with connect(db_path) as c:
        rows = c.execute("""SELECT name FROM teams
                            UNION SELECT team FROM players WHERE team IS NOT NULL AND team <> ''""")
        return sorted((r[0] for r in rows), key=_sort_key)


def players_for(age_group: str, team: Optional[str] = None, db_path: str | None = None) -> list[dict]:
    """Jogadores do escalão (e da equipa, se indicada). team=None → sem filtro de equipa."""
    qual_db.init(db_path)
    sql, args = "SELECT * FROM players WHERE category=?", [age_group]
    if team:
        sql += " AND team=?"
        args.append(team)
    with connect(db_path) as c:
        rows = [dict(r) for r in c.execute(sql, args)]
    return sorted(rows, key=lambda r: _sort_key(r["name"]))


# ── Gravar ──────────────────────────────────────────────────────────────────

def _write_items(c, evaluation_id: int, comp, scores: Mapping[str, int]) -> None:
    c.execute("DELETE FROM evaluation_items WHERE evaluation_id=?", (evaluation_id,))
    for crit_key, score in scores.items():
        c.execute("INSERT INTO evaluation_items(evaluation_id, category, criterion, score) VALUES (?,?,?,?)",
                  (evaluation_id, comp.dimension_of(crit_key).key, crit_key, score))


def save_evaluation(competency: str, player_id: int, evaluation_date, scores: Mapping[str, Optional[int]],
                    observations: Optional[str] = None, team: Optional[str] = None,
                    age_group: Optional[str] = None, coach: Optional[str] = None,
                    db_path: str | None = None) -> int:
    """Grava uma avaliação nova. Exige pelo menos um critério avaliado."""
    comp = competencies.get(competency)
    clean = q.validate_scores(comp, scores)
    if not clean:
        raise ValueError("Avalie pelo menos um critério antes de guardar.")
    d = _iso(evaluation_date)
    qual_db.init(db_path)
    with connect(db_path) as c:
        p = c.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
    if p is None:
        raise ValueError(f"Jogador {player_id} não existe.")
    age_group = age_group or p["category"]
    if age_group not in list_age_groups(db_path):
        raise ValueError(f"Escalão inválido: {age_group!r}.")
    team = team or p["team"]
    team_id = get_or_create_team(team, db_path) if _clean(team) else None
    coach_id = base.get_or_create_coach(coach, db_path) if _clean(coach) else None
    with connect(db_path) as c:
        cur = c.execute(
            """INSERT INTO evaluations(competency, player_id, team_id, age_group, coach_id,
                                       evaluation_date, observations) VALUES (?,?,?,?,?,?,?)""",
            (competency, player_id, team_id, age_group, coach_id, d, _clean(observations)))
        _write_items(c, cur.lastrowid, comp, clean)
        return cur.lastrowid


def update_evaluation(evaluation_id: int, evaluation_date, scores: Mapping[str, Optional[int]],
                      observations: Optional[str] = None, coach: Optional[str] = None,
                      db_path: str | None = None) -> None:
    """Corrige uma avaliação existente (atualiza `updated_at`; mantém `created_at`)."""
    ev = get_evaluation(evaluation_id, db_path)
    if ev is None:
        raise ValueError(f"Avaliação {evaluation_id} não existe.")
    comp = competencies.get(ev["competency"])
    clean = q.validate_scores(comp, scores)
    if not clean:
        raise ValueError("Avalie pelo menos um critério antes de guardar.")
    coach_id = base.get_or_create_coach(coach, db_path) if _clean(coach) else None
    with connect(db_path) as c:
        c.execute("""UPDATE evaluations SET evaluation_date=?, observations=?, coach_id=?,
                     updated_at=datetime('now') WHERE id=?""",
                  (_iso(evaluation_date), _clean(observations), coach_id, evaluation_id))
        _write_items(c, evaluation_id, comp, clean)


def delete_evaluation(evaluation_id: int, confirm: bool = False, db_path: str | None = None) -> None:
    """Apaga uma avaliação. Só com confirmação explícita."""
    if not confirm:
        raise ValueError("Confirmação necessária para apagar a avaliação.")
    qual_db.init(db_path)
    with connect(db_path) as c:
        c.execute("DELETE FROM evaluations WHERE id=?", (evaluation_id,))


# ── Consultar ───────────────────────────────────────────────────────────────

_SELECT = """SELECT e.*, p.name AS player_name, p.category AS player_category,
                    t.name AS team_name, co.name AS coach_name
             FROM evaluations e
             JOIN players p ON p.id = e.player_id
             LEFT JOIN teams t ON t.id = e.team_id
             LEFT JOIN coaches co ON co.id = e.coach_id"""


def _hydrate(c, row) -> dict:
    ev = dict(row)
    comp = competencies.get(ev["competency"])
    items = c.execute("SELECT criterion, score FROM evaluation_items WHERE evaluation_id=?", (ev["id"],))
    ev["scores"] = {r["criterion"]: r["score"] for r in items}
    s = q.summarize(comp, ev["scores"])
    ev.update(mean=s.mean, mean_display=s.mean_display, level=s.level, label=s.label,
              dimension_means=s.dimension_means)
    return ev


def get_evaluation(evaluation_id: int, db_path: str | None = None) -> Optional[dict]:
    qual_db.init(db_path)
    with connect(db_path) as c:
        row = c.execute(_SELECT + " WHERE e.id=?", (evaluation_id,)).fetchone()
        return _hydrate(c, row) if row else None


def list_evaluations(player_id: int, competency: str, db_path: str | None = None) -> list[dict]:
    """Avaliações do jogador por ordem cronológica (data, depois id)."""
    competencies.get(competency)
    qual_db.init(db_path)
    with connect(db_path) as c:
        rows = c.execute(_SELECT + " WHERE e.player_id=? AND e.competency=? ORDER BY e.evaluation_date, e.id",
                         (player_id, competency)).fetchall()
        return [_hydrate(c, r) for r in rows]


def player_evolution(player_id: int, competency: str, db_path: str | None = None) -> Optional[q.Evolution]:
    evs = [e for e in list_evaluations(player_id, competency, db_path) if e["mean"] is not None]
    return q.evolution([(e["evaluation_date"], e["mean"]) for e in evs])


def compare_evaluations(id_a: int, id_b: int, db_path: str | None = None) -> tuple[dict, dict, q.Comparison]:
    """Compara duas avaliações do MESMO jogador e competência; a mais antiga é a «inicial»."""
    a, b = get_evaluation(id_a, db_path), get_evaluation(id_b, db_path)
    if a is None or b is None:
        raise ValueError("Avaliação inexistente.")
    if a["player_id"] != b["player_id"] or a["competency"] != b["competency"]:
        raise ValueError("Só é possível comparar avaliações do mesmo jogador e da mesma competência.")
    if (b["evaluation_date"], b["id"]) < (a["evaluation_date"], a["id"]):
        a, b = b, a
    comp = competencies.get(a["competency"])
    return a, b, q.compare(comp, a["scores"], b["scores"])
