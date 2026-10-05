"""Avaliações das nove competências.

Regras: nunca se apaga nem se altera o resultado de uma avaliação. Uma correção cria
uma nova versão (supersedes_id); a versão anterior fica guardada mas deixa de contar
na evolução. O escalão fica gravado na avaliação (equipa em vigor à data).
"""

from __future__ import annotations

from datetime import date
from typing import Mapping, Optional

from . import calc
from . import competencies as comp
from .db import MOMENTS, active_scale, connect, init_db
from .service import ValidationError, _iso


def get_or_create_coach(display_name: str, db_path: str | None = None) -> int:
    """Treinador (utilizador com perfil «coach», sem palavra-passe até à Fase 10)."""
    name = (display_name or "").strip()
    if not name:
        raise ValidationError("O nome do treinador é obrigatório.")
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT id FROM users WHERE role='coach' AND display_name=? COLLATE NOCASE", (name,)).fetchone()
        if row:
            return row["id"]
        base = "".join(ch for ch in name.lower() if ch.isalnum()) or "treinador"
        username, n = base, 1
        while c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            n += 1
            username = f"{base}{n}"
        return c.execute("INSERT INTO users(username, display_name, role) VALUES (?,?,'coach')",
                         (username, name)).lastrowid


def list_coaches(db_path: str | None = None) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(
            "SELECT id, display_name FROM users WHERE role='coach' AND active=1 ORDER BY display_name")]


def _team_at(conn, player_id: int, on: str) -> Optional[dict]:
    """Equipa do jogador na data da avaliação; se nenhuma cobrir, a atual."""
    base = """SELECT m.team_id, t.category FROM team_memberships m JOIN teams t ON t.id=m.team_id
              WHERE m.player_id=?"""
    row = conn.execute(base + " AND (m.joined_on IS NULL OR m.joined_on<=?) AND (m.left_on IS NULL OR ?<m.left_on)"
                       " ORDER BY m.id DESC LIMIT 1", (player_id, on, on)).fetchone()
    if not row:
        row = conn.execute(base + " AND m.left_on IS NULL ORDER BY m.id DESC LIMIT 1", (player_id,)).fetchone()
    return dict(row) if row else None


def _check_scores(scores: Mapping[str, Optional[int]], valid_values: set[int]) -> dict[str, Optional[int]]:
    unknown = set(scores) - set(comp.KEYS)
    if unknown:
        raise ValidationError(f"Competência desconhecida: {', '.join(sorted(unknown))}.")
    out: dict[str, Optional[int]] = {}
    for k in comp.KEYS:
        v = scores.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v not in valid_values):
            lo, hi = min(valid_values), max(valid_values)
            raise ValidationError(f"{comp.name_of(k)}: classificação inválida ({v!r}); use valores de {lo} a {hi}.")
        out[k] = v
    if not calc.filled(out):
        raise ValidationError("Classifique pelo menos uma competência.")
    return out


def create_evaluation(player_id: int, evaluation_date, moment: str, scores: Mapping[str, Optional[int]],
                      notes: Mapping[str, str] | None = None, coach_id: int | None = None,
                      general_notes: str | None = None, next_objectives: str | None = None,
                      parent_message: str | None = None, supersedes_id: int | None = None, is_demo: bool = False,
                      db_path: str | None = None) -> int:
    on = _iso(evaluation_date, "Data da avaliação", required=True)
    if on > date.today().isoformat():
        raise ValidationError("A data da avaliação não pode estar no futuro.")
    if moment not in MOMENTS:
        raise ValidationError(f"Momento inválido: {moment!r}.")
    init_db(db_path)
    scale = active_scale(db_path)
    clean = _check_scores(scores, {v for v, _ in scale["levels"]})
    notes = {k: (v or "").strip() for k, v in (notes or {}).items() if k in comp.BY_KEY}
    with connect(db_path) as c:
        if not c.execute("SELECT 1 FROM players WHERE id=?", (player_id,)).fetchone():
            raise ValidationError("Jogador inexistente.")
        team = _team_at(c, player_id, on)
        if not team:
            raise ValidationError("O jogador não tem equipa; associe-o a uma equipa antes de avaliar.")
        if coach_id and not c.execute("SELECT 1 FROM users WHERE id=? AND role='coach'", (coach_id,)).fetchone():
            raise ValidationError("Treinador inexistente.")
        if supersedes_id is not None:
            old = c.execute("SELECT player_id FROM evaluations WHERE id=?", (supersedes_id,)).fetchone()
            if not old:
                raise ValidationError("Avaliação a corrigir inexistente.")
            if old["player_id"] != player_id:
                raise ValidationError("A correção tem de ser do mesmo jogador.")
            if c.execute("SELECT 1 FROM evaluations WHERE supersedes_id=?", (supersedes_id,)).fetchone():
                raise ValidationError("Esta avaliação já foi corrigida; corrija a versão mais recente.")
        eid = c.execute(
            """INSERT INTO evaluations(player_id, team_id, category, scale_id, evaluation_date, moment, coach_id,
                                       general_notes, next_objectives, parent_message, supersedes_id, is_demo)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (player_id, team["team_id"], team["category"], scale["id"], on, moment, coach_id,
             (general_notes or "").strip() or None, (next_objectives or "").strip() or None,
             (parent_message or "").strip() or None, supersedes_id, int(is_demo))).lastrowid
        c.executemany("INSERT INTO evaluation_scores(evaluation_id, competency_key, score, note) VALUES (?,?,?,?)",
                      [(eid, k, clean[k], notes.get(k) or None) for k in comp.KEYS])
    return eid


def correct_evaluation(evaluation_id: int, evaluation_date, moment: str, scores: Mapping[str, Optional[int]],
                       notes: Mapping[str, str] | None = None, coach_id: int | None = None,
                       general_notes: str | None = None, next_objectives: str | None = None,
                       parent_message: str | None = None, db_path: str | None = None) -> int:
    """Cria uma nova versão que substitui (sem apagar) a avaliação indicada."""
    old = get_evaluation(evaluation_id, db_path)
    if not old:
        raise ValidationError("Avaliação a corrigir inexistente.")
    return create_evaluation(old["player_id"], evaluation_date, moment, scores, notes, coach_id, general_notes,
                             next_objectives, parent_message, supersedes_id=evaluation_id, is_demo=bool(old["is_demo"]),
                             db_path=db_path)


def _hydrate(conn, row) -> dict:
    ev = dict(row)
    rs = conn.execute("SELECT competency_key, score, note FROM evaluation_scores WHERE evaluation_id=?",
                      (ev["id"],)).fetchall()
    by = {r["competency_key"]: r for r in rs}
    ev["scores"] = {k: by[k]["score"] for k in comp.KEYS if k in by}
    ev["notes"] = {k: by[k]["note"] for k in comp.KEYS if k in by and by[k]["note"]}
    ev["average"] = calc.global_average(ev["scores"])      # calculada, nunca guardada
    ev["complete"] = calc.is_complete(ev["scores"])
    ev["missing"] = calc.missing(ev["scores"])
    ev["superseded_by"] = (conn.execute("SELECT id FROM evaluations WHERE supersedes_id=?", (ev["id"],))
                           .fetchone() or {"id": None})["id"]
    return ev


_EV_SELECT = """SELECT e.*, u.display_name AS coach, t.name AS team, cl.name AS club
                FROM evaluations e LEFT JOIN users u ON u.id=e.coach_id
                JOIN teams t ON t.id=e.team_id JOIN clubs cl ON cl.id=t.club_id"""


def get_evaluation(evaluation_id: int, db_path: str | None = None) -> Optional[dict]:
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute(_EV_SELECT + " WHERE e.id=?", (evaluation_id,)).fetchone()
        return _hydrate(c, row) if row else None


def list_evaluations(player_id: int, include_superseded: bool = False,
                     db_path: str | None = None) -> list[dict]:
    """Histórico cronológico (mais antiga primeiro). Por omissão só as versões em vigor."""
    init_db(db_path)
    sql = _EV_SELECT + " WHERE e.player_id=?"
    if not include_superseded:
        sql += " AND NOT EXISTS (SELECT 1 FROM evaluations x WHERE x.supersedes_id=e.id)"
    with connect(db_path) as c:
        return [_hydrate(c, r) for r in c.execute(sql + " ORDER BY e.evaluation_date, e.id", (player_id,))]


def latest_evaluation(player_id: int, db_path: str | None = None) -> Optional[dict]:
    evs = list_evaluations(player_id, db_path=db_path)
    return evs[-1] if evs else None
