"""Lógica de negócio: clubes, equipas e jogadores.

Todas as funções aceitam `db_path` (testes usam BD temporária). As permissões por
utilizador são aplicadas na Fase 10; aqui não há filtragem por perfil.
"""

from __future__ import annotations

import os
import re
import unicodedata
import uuid
from datetime import date
from typing import Optional

from .db import CATEGORIES, connect, init_db

PHOTO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "photos")
_SEASON_RE = re.compile(r"^\d{4}/\d{4}$")


class ValidationError(ValueError):
    """Dados inválidos introduzidos pelo utilizador."""


def _iso(d, field: str, required: bool = False) -> Optional[str]:
    if d in (None, ""):
        if required:
            raise ValidationError(f"{field} é obrigatória.")
        return None
    try:
        return (d if isinstance(d, date) else date.fromisoformat(str(d))).isoformat()
    except ValueError:
        raise ValidationError(f"{field} inválida: {d!r} (formato AAAA-MM-DD).")


def _name(value, what: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValidationError(f"O nome {what} é obrigatório.")
    return value


def _category(category: str) -> str:
    if category not in CATEGORIES:
        raise ValidationError(f"Escalão inválido: {category!r}. Escalões: {', '.join(CATEGORIES)}.")
    return category


def _fold(text: str) -> str:
    """Minúsculas e sem acentos, para pesquisa tolerante («joao» encontra «João»)."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def _jersey(n) -> Optional[int]:
    if n in (None, ""):
        return None
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise ValidationError("Número da camisola inválido.")
    if not 0 <= n <= 99:
        raise ValidationError("O número da camisola deve estar entre 0 e 99.")
    return n


# ── Clubes ───────────────────────────────────────────────────────────────────
def create_club(name: str, db_path: str | None = None) -> int:
    name = _name(name, "do clube")
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT id FROM clubs WHERE name=? COLLATE NOCASE", (name,)).fetchone()
        if row:
            raise ValidationError(f"Já existe um clube com o nome «{name}».")
        return c.execute("INSERT INTO clubs(name) VALUES (?)", (name,)).lastrowid


def list_clubs(db_path: str | None = None) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute("SELECT id, name FROM clubs ORDER BY name")]


# ── Equipas ──────────────────────────────────────────────────────────────────
def create_team(club_id: int, name: str, category: str, season: str,
                is_demo: bool = False, db_path: str | None = None) -> int:
    name, category = _name(name, "da equipa"), _category(category)
    season = (season or "").strip()
    if not _SEASON_RE.match(season) or int(season[5:]) != int(season[:4]) + 1:
        raise ValidationError("Época inválida: use o formato 2026/2027.")
    init_db(db_path)
    with connect(db_path) as c:
        if not c.execute("SELECT 1 FROM clubs WHERE id=?", (club_id,)).fetchone():
            raise ValidationError("Clube inexistente.")
        if c.execute("SELECT 1 FROM teams WHERE club_id=? AND name=? COLLATE NOCASE AND season=?",
                     (club_id, name, season)).fetchone():
            raise ValidationError(f"Já existe a equipa «{name}» nesta época.")
        return c.execute("INSERT INTO teams(club_id, name, category, season, is_demo) VALUES (?,?,?,?,?)",
                         (club_id, name, category, season, int(is_demo))).lastrowid


def list_teams(category: str | None = None, club_id: int | None = None, season: str | None = None,
               db_path: str | None = None) -> list[dict]:
    init_db(db_path)
    sql = """SELECT t.id, t.name, t.category, t.season, t.club_id, t.is_demo, cl.name AS club,
                    (SELECT COUNT(*) FROM team_memberships m WHERE m.team_id=t.id AND m.left_on IS NULL) AS n_players
             FROM teams t JOIN clubs cl ON cl.id=t.club_id WHERE 1=1"""
    args: list = []
    for col, val in (("t.category", category), ("t.club_id", club_id), ("t.season", season)):
        if val is not None:
            sql += f" AND {col}=?"
            args.append(val)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(sql + " ORDER BY t.season DESC, t.category, t.name", args)]


# ── Fotografias ──────────────────────────────────────────────────────────────
def save_photo(data: bytes, filename: str) -> str:
    """Guarda a fotografia em data/photos com nome aleatório (não expõe o nome do jogador)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValidationError("Fotografia: use PNG, JPG ou WEBP.")
    if len(data) > 5 * 1024 * 1024:
        raise ValidationError("Fotografia demasiado grande (máximo 5 MB).")
    os.makedirs(PHOTO_DIR, exist_ok=True)
    path = os.path.join(PHOTO_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path


# ── Jogadores ────────────────────────────────────────────────────────────────
def _validated_player_fields(name, birth_date, sex) -> tuple[str, Optional[str], Optional[str]]:
    name = _name(name, "do jogador")
    birth = _iso(birth_date, "Data de nascimento")
    if birth and birth > date.today().isoformat():
        raise ValidationError("A data de nascimento não pode estar no futuro.")
    if sex not in (None, "", "M", "F"):
        raise ValidationError("Sexo inválido (M ou F).")
    return name, birth, sex or None


def create_player(name: str, team_id: int, birth_date=None, sex: str | None = None,
                  jersey_number=None, joined_on=None, notes: str | None = None,
                  photo_path: str | None = None, is_demo: bool = False,
                  db_path: str | None = None) -> int:
    name, birth, sex = _validated_player_fields(name, birth_date, sex)
    jersey, joined = _jersey(jersey_number), _iso(joined_on, "Data de entrada")
    init_db(db_path)
    with connect(db_path) as c:
        team = c.execute("SELECT is_demo FROM teams WHERE id=?", (team_id,)).fetchone()
        if not team:
            raise ValidationError("Equipa inexistente.")
        is_demo = is_demo or bool(team["is_demo"])        # nunca misturar dados reais numa equipa de teste
        pid = c.execute(
            "INSERT INTO players(name, photo_path, birth_date, sex, notes, is_demo) VALUES (?,?,?,?,?,?)",
            (name, photo_path, birth, sex, (notes or "").strip() or None, int(is_demo))).lastrowid
        c.execute("INSERT INTO team_memberships(player_id, team_id, jersey_number, joined_on) VALUES (?,?,?,?)",
                  (pid, team_id, jersey, joined or date.today().isoformat()))
        return pid


def update_player(player_id: int, name: str, birth_date=None, sex: str | None = None,
                  notes: str | None = None, photo_path: str | None = None,
                  jersey_number=None, db_path: str | None = None) -> None:
    """Atualiza a ficha (dados do jogador, não avaliações). photo_path=None mantém a atual."""
    name, birth, sex = _validated_player_fields(name, birth_date, sex)
    jersey = _jersey(jersey_number)
    init_db(db_path)
    with connect(db_path) as c:
        if not c.execute("SELECT 1 FROM players WHERE id=?", (player_id,)).fetchone():
            raise ValidationError("Jogador inexistente.")
        c.execute("UPDATE players SET name=?, birth_date=?, sex=?, notes=? WHERE id=?",
                  (name, birth, sex, (notes or "").strip() or None, player_id))
        if photo_path:
            c.execute("UPDATE players SET photo_path=? WHERE id=?", (photo_path, player_id))
        c.execute("UPDATE team_memberships SET jersey_number=? WHERE player_id=? AND left_on IS NULL",
                  (jersey, player_id))


def change_team(player_id: int, new_team_id: int, on_date=None, jersey_number=None,
                db_path: str | None = None) -> int:
    """Muda o jogador de equipa/escalão. Fecha a pertença atual e abre outra; nada é apagado."""
    on = _iso(on_date, "Data da mudança") or date.today().isoformat()
    jersey = _jersey(jersey_number)
    init_db(db_path)
    with connect(db_path) as c:
        cur = c.execute("SELECT id, team_id, joined_on FROM team_memberships "
                        "WHERE player_id=? AND left_on IS NULL ORDER BY id DESC LIMIT 1", (player_id,)).fetchone()
        if not cur:
            raise ValidationError("Jogador inexistente ou sem equipa atual.")
        if cur["team_id"] == new_team_id:
            raise ValidationError("O jogador já pertence a esta equipa.")
        if not c.execute("SELECT 1 FROM teams WHERE id=?", (new_team_id,)).fetchone():
            raise ValidationError("Equipa inexistente.")
        if cur["joined_on"] and on < cur["joined_on"]:
            raise ValidationError("A data da mudança é anterior à data de entrada na equipa atual.")
        c.execute("UPDATE team_memberships SET left_on=? WHERE id=?", (on, cur["id"]))
        return c.execute("INSERT INTO team_memberships(player_id, team_id, jersey_number, joined_on) VALUES (?,?,?,?)",
                         (player_id, new_team_id, jersey, on)).lastrowid


_PLAYER_SELECT = """
    SELECT p.id, p.name, p.photo_path, p.birth_date, p.sex, p.notes, p.is_demo,
           m.team_id, m.jersey_number, m.joined_on,
           t.name AS team, t.category, t.season, cl.name AS club
    FROM players p
    LEFT JOIN team_memberships m ON m.player_id=p.id AND m.left_on IS NULL
    LEFT JOIN teams t ON t.id=m.team_id
    LEFT JOIN clubs cl ON cl.id=t.club_id"""


def get_player(player_id: int, db_path: str | None = None) -> Optional[dict]:
    """Ficha do jogador com a equipa/escalão atuais e o histórico de equipas (mais recente primeiro)."""
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute(_PLAYER_SELECT + " WHERE p.id=?", (player_id,)).fetchone()
        if not row:
            return None
        hist = c.execute(
            """SELECT m.team_id, t.name AS team, t.category, t.season, cl.name AS club,
                      m.jersey_number, m.joined_on, m.left_on
               FROM team_memberships m JOIN teams t ON t.id=m.team_id JOIN clubs cl ON cl.id=t.club_id
               WHERE m.player_id=? ORDER BY m.id DESC""", (player_id,)).fetchall()
    out = dict(row)
    out["team_history"] = [dict(h) for h in hist]
    return out


def search_players(name: str | None = None, category: str | None = None, team_id: int | None = None,
                   db_path: str | None = None) -> list[dict]:
    """Pesquisa por nome (sem distinguir acentos/maiúsculas), escalão e equipa atuais.

    Jogadores sem equipa atual só aparecem sem filtros de escalão/equipa. Ordem alfabética
    (nunca por desempenho).
    """
    init_db(db_path)
    sql, args = _PLAYER_SELECT + " WHERE 1=1", []
    if category:
        sql += " AND t.category=?"
        args.append(_category(category))
    if team_id:
        sql += " AND m.team_id=?"
        args.append(team_id)
    with connect(db_path) as c:
        rows = [dict(r) for r in c.execute(sql, args)]
    if name and name.strip():
        needle = _fold(name.strip())
        rows = [r for r in rows if needle in _fold(r["name"])]
    return sorted(rows, key=lambda r: _fold(r["name"]))
