"""Controlo de acesso: tudo o que as vistas leem ou escrevem passa por aqui.

Regras:
- ADMINISTRADOR: gere tudo.
- TREINADOR: vê e avalia os jogadores atuais das equipas que acompanha (team_coaches) e os
  jogadores com acesso individual (player_access); vê relatórios e estatísticas dessas equipas.
  Pode criar equipas do seu clube (fica a acompanhá-las) e jogadores nas suas equipas.
  Mudar um jogador para outra equipa exige acompanhar também a equipa de destino (ou ser admin).
- ENCARREGADO DE EDUCAÇÃO: só os educandos associados à sua conta, e só através do relatório
  para pais e da evolução do próprio educando. Nunca vê equipas, outros jogadores, estatísticas
  coletivas nem as notas internas do treinador.

Cada função recebe `user` (dict devolvido por `auth.get_user`) e levanta PermissionDenied
quando o acesso não é permitido. Os cálculos puros (evolution, calc) não precisam de controlo.
"""

from __future__ import annotations

from typing import Optional

from . import evaluations as ev
from . import auth, pdf, reports, seed, service, teamstats
from .db import active_scale, connect, init_db


class PermissionDenied(Exception):
    """O utilizador não tem permissão para esta operação."""


def _role(user: Optional[dict]) -> str:
    if not user or not user.get("active", True):
        raise PermissionDenied("Sessão inválida.")
    return user["role"]


def require_admin(user) -> None:
    if _role(user) != "admin":
        raise PermissionDenied("Apenas o administrador pode fazer isto.")


def require_staff(user) -> None:
    if _role(user) not in ("admin", "coach"):
        raise PermissionDenied("Esta área é reservada a treinadores.")


# ── conjuntos visíveis ──────────────────────────────────────────────────────
def coached_team_ids(user, db_path=None) -> set[int]:
    role = _role(user)
    init_db(db_path)
    with connect(db_path) as c:
        if role == "admin":
            return {r[0] for r in c.execute("SELECT id FROM teams")}
        if role == "coach":
            return {r[0] for r in c.execute("SELECT team_id FROM team_coaches WHERE user_id=?", (user["id"],))}
    return set()


def visible_player_ids(user, db_path=None) -> set[int]:
    role = _role(user)
    init_db(db_path)
    with connect(db_path) as c:
        if role == "admin":
            return {r[0] for r in c.execute("SELECT id FROM players")}
        if role == "guardian":
            return {r[0] for r in c.execute("SELECT player_id FROM guardians_players WHERE user_id=?", (user["id"],))}
        ids = {r[0] for r in c.execute(
            """SELECT m.player_id FROM team_memberships m JOIN team_coaches tc ON tc.team_id=m.team_id
               WHERE m.left_on IS NULL AND tc.user_id=?""", (user["id"],))}
        ids |= {r[0] for r in c.execute("SELECT player_id FROM player_access WHERE user_id=?", (user["id"],))}
        return ids


def can_view_player(user, player_id: int, db_path=None) -> bool:
    return player_id in visible_player_ids(user, db_path)


def _need_player(user, player_id: int, db_path=None) -> None:
    require_staff(user)
    if not can_view_player(user, player_id, db_path):
        raise PermissionDenied("Não tem acesso a este jogador.")


def _need_team(user, team_id: int, db_path=None) -> None:
    require_staff(user)
    if team_id not in coached_team_ids(user, db_path):
        raise PermissionDenied("Não tem acesso a esta equipa.")


def _eval_player(evaluation_id: int, db_path=None) -> int:
    e = ev.get_evaluation(evaluation_id, db_path)
    if not e:
        raise service.ValidationError("Avaliação inexistente.")
    return e["player_id"]


# ── leitura (treinador / administrador) ─────────────────────────────────────
def list_clubs(user, db_path=None) -> list[dict]:
    require_staff(user)
    clubs = service.list_clubs(db_path)
    if user["role"] == "admin":
        return clubs
    mine = {t["club_id"] for t in service.list_teams(db_path=db_path) if t["id"] in coached_team_ids(user, db_path)}
    if user.get("club_id"):
        mine.add(user["club_id"])
    return [c for c in clubs if c["id"] in mine]


def list_teams(user, category=None, club_id=None, season=None, db_path=None) -> list[dict]:
    require_staff(user)
    allowed = coached_team_ids(user, db_path)
    return [t for t in service.list_teams(category, club_id, season, db_path) if t["id"] in allowed]


def search_players(user, name=None, category=None, team_id=None, db_path=None) -> list[dict]:
    require_staff(user)
    allowed = visible_player_ids(user, db_path)
    return [p for p in service.search_players(name, category, team_id, db_path) if p["id"] in allowed]


def get_player(user, player_id: int, db_path=None) -> Optional[dict]:
    _need_player(user, player_id, db_path)
    return service.get_player(player_id, db_path)


def list_evaluations(user, player_id: int, include_superseded: bool = False, db_path=None) -> list[dict]:
    _need_player(user, player_id, db_path)
    return ev.list_evaluations(player_id, include_superseded, db_path)


def individual_report(user, evaluation_id: int, db_path=None) -> dict:
    _need_player(user, _eval_player(evaluation_id, db_path), db_path)
    return reports.individual_report(evaluation_id, db_path)


def team_report(user, team_id: int, db_path=None) -> dict:
    _need_team(user, team_id, db_path)
    return reports.team_report(team_id, db_path)


def team_overview(user, team_id: int, db_path=None) -> dict:
    _need_team(user, team_id, db_path)
    return teamstats.overview(team_id, db_path)


def team_timeline(user, team_id: int, db_path=None) -> list[dict]:
    _need_team(user, team_id, db_path)
    return teamstats.timeline(team_id, db_path)


def team_snapshot(user, team_id: int, as_of=None, db_path=None) -> dict:
    _need_team(user, team_id, db_path)
    return teamstats.snapshot(team_id, as_of, db_path)


# ── escrita (treinador / administrador) ─────────────────────────────────────
def create_club(user, name: str, db_path=None) -> int:
    require_admin(user)
    return service.create_club(name, db_path)


def create_team(user, club_id: int, name: str, category: str, season: str, is_demo=False, db_path=None) -> int:
    role = _role(user)
    if role == "guardian" or (role == "coach" and user.get("club_id") != club_id):
        raise PermissionDenied("Não pode criar equipas neste clube.")
    tid = service.create_team(club_id, name, category, season, is_demo, db_path)
    if role == "coach":                                    # quem cria a equipa passa a acompanhá-la
        with connect(db_path) as c:
            c.execute("INSERT OR IGNORE INTO team_coaches(team_id, user_id) VALUES (?,?)", (tid, user["id"]))
    return tid


def create_player(user, name: str, team_id: int, *args, db_path=None, **kwargs) -> int:
    _need_team(user, team_id, db_path)
    return service.create_player(name, team_id, *args, db_path=db_path, **kwargs)


def update_player(user, player_id: int, *args, db_path=None, **kwargs) -> None:
    _need_player(user, player_id, db_path)
    service.update_player(player_id, *args, db_path=db_path, **kwargs)


def change_team(user, player_id: int, new_team_id: int, *args, db_path=None, **kwargs) -> int:
    _need_player(user, player_id, db_path)
    _need_team(user, new_team_id, db_path)
    return service.change_team(player_id, new_team_id, *args, db_path=db_path, **kwargs)


def create_evaluation(user, player_id: int, evaluation_date, moment: str, scores, notes=None,
                      general_notes=None, next_objectives=None, parent_message=None, db_path=None) -> int:
    """O avaliador é sempre o utilizador autenticado (não se regista em nome de outro treinador)."""
    _need_player(user, player_id, db_path)
    return ev.create_evaluation(player_id, evaluation_date, moment, scores, notes, user["id"], general_notes,
                                next_objectives, parent_message, db_path=db_path)


def correct_evaluation(user, evaluation_id: int, evaluation_date, moment: str, scores, notes=None,
                       general_notes=None, next_objectives=None, parent_message=None, db_path=None) -> int:
    _need_player(user, _eval_player(evaluation_id, db_path), db_path)
    return ev.correct_evaluation(evaluation_id, evaluation_date, moment, scores, notes, user["id"], general_notes,
                                 next_objectives, parent_message, db_path=db_path)


# ── encarregado de educação ─────────────────────────────────────────────────
def _need_guardian(user) -> None:
    if _role(user) != "guardian":
        raise PermissionDenied("Área reservada a encarregados de educação.")


def guardian_children(user, db_path=None) -> list[dict]:
    """Só dados básicos dos educandos associados (sem equipa-colegas, sem estatísticas)."""
    _need_guardian(user)
    out = []
    for pid in sorted(visible_player_ids(user, db_path)):
        p = service.get_player(pid, db_path)
        if p:
            out.append({"id": p["id"], "name": p["name"], "category": p["category"], "team": p["team"],
                        "club": p["club"]})
    return sorted(out, key=lambda x: x["name"])


def _need_child(user, player_id: int, db_path=None) -> None:
    _need_guardian(user)
    if player_id not in visible_player_ids(user, db_path):
        raise PermissionDenied("Este jogador não está associado à sua conta.")


def guardian_evaluations(user, player_id: int, db_path=None) -> list[dict]:
    """Avaliações em vigor do educando, sem notas internas, treinador nem observações."""
    _need_child(user, player_id, db_path)
    keep = ("id", "evaluation_date", "moment", "category", "scores", "average", "complete", "is_demo")
    return [{k: e[k] for k in keep} for e in ev.list_evaluations(player_id, db_path=db_path)]


def parent_report(user, evaluation_id: int, db_path=None) -> dict:
    """Relatório para os pais: o encarregado só acede ao do seu educando; o treinador pré-visualiza."""
    pid = _eval_player(evaluation_id, db_path)
    if _role(user) == "guardian":
        _need_child(user, pid, db_path)
        if ev.get_evaluation(evaluation_id, db_path)["superseded_by"]:
            raise PermissionDenied("Esta avaliação foi substituída por uma versão mais recente.")
    else:
        _need_player(user, pid, db_path)
    return reports.parent_report(evaluation_id, db_path)


# ── administração de contas ─────────────────────────────────────────────────
def list_users(user, db_path=None) -> list[dict]:
    require_admin(user)
    return auth.list_users(db_path)


def admin_create_user(user, *args, db_path=None, **kwargs) -> int:
    require_admin(user)
    return auth.create_user(*args, db_path=db_path, **kwargs)


def admin_set_password(user, target_id: int, password: str, db_path=None) -> None:
    require_admin(user)
    auth.set_password(target_id, password, db_path)


def admin_set_active(user, target_id: int, active: bool, db_path=None) -> None:
    require_admin(user)
    if target_id == user["id"] and not active:
        raise auth.AuthError("Não pode desativar a sua própria conta.")
    auth.set_active(target_id, active, db_path)


def admin_set_club(user, target_id: int, club_id, db_path=None) -> None:
    require_admin(user)
    auth.set_club(target_id, club_id, db_path)


def admin_set_team_assignments(user, target_id: int, team_ids, db_path=None) -> None:
    require_admin(user)
    auth.set_team_assignments(target_id, list(team_ids), db_path)


def admin_set_guardian_links(user, target_id: int, player_ids, db_path=None) -> None:
    require_admin(user)
    auth.set_guardian_links(target_id, list(player_ids), db_path)


def admin_set_player_access(user, target_id: int, player_ids, db_path=None) -> None:
    require_admin(user)
    auth.set_player_access(target_id, list(player_ids), db_path)


def admin_assignments(user, target_id: int, db_path=None) -> dict:
    """Equipas, educandos e acessos individuais atuais de uma conta."""
    require_admin(user)
    init_db(db_path)
    with connect(db_path) as c:
        q = lambda sql: sorted(r[0] for r in c.execute(sql, (target_id,)))
        return {"teams": q("SELECT team_id FROM team_coaches WHERE user_id=?"),
                "children": q("SELECT player_id FROM guardians_players WHERE user_id=?"),
                "player_access": q("SELECT player_id FROM player_access WHERE user_id=?")}


def admin_demo_summary(user, db_path=None) -> dict:
    require_admin(user)
    return seed.summary(db_path)


def admin_load_demo(user, db_path=None) -> dict:
    require_admin(user)
    return seed.load_demo(db_path=db_path)


def admin_remove_demo(user, db_path=None) -> dict:
    require_admin(user)
    return seed.remove_demo(db_path=db_path)


# ── exportação em PDF (mesmas permissões que a consulta) ────────────────────
def export_individual_pdf(user, evaluation_id: int, db_path=None) -> tuple[bytes, str]:
    r = individual_report(user, evaluation_id, db_path)
    return pdf.individual_pdf(r), pdf.filename("relatorio-individual", r["player"]["name"], r["date"])


def export_parent_pdf(user, evaluation_id: int, db_path=None) -> tuple[bytes, str]:
    """O encarregado de educação só exporta o relatório do seu educando; o treinador, dos seus jogadores."""
    r = parent_report(user, evaluation_id, db_path)
    return pdf.parent_pdf(r), pdf.filename("relatorio-pais", r["player"]["name"], r["date"])


def export_team_pdf(user, team_id: int, db_path=None) -> tuple[bytes, str]:
    r = team_report(user, team_id, db_path)
    return pdf.team_pdf(r), pdf.filename("relatorio-equipa", f"{r['team']['name']}-{r['team']['season'].replace('/', '-')}",
                                         r["as_of"])


def export_player_sheet_pdf(user, player_id: int, db_path=None) -> tuple[bytes, str]:
    """Ficha individual (inclui dados pessoais): só treinadores com acesso ao jogador e administrador."""
    p = get_player(user, player_id, db_path)
    if not p:
        raise service.ValidationError("Jogador inexistente.")
    history = [{"evaluation_date": e["evaluation_date"], "moment": e["moment"], "category": e["category"],
                "coach": e["coach"], "average": e["average"], "complete": e["complete"]}
               for e in ev.list_evaluations(player_id, db_path=db_path)]
    top = max(v for v, _ in active_scale(db_path)["levels"])
    return pdf.player_sheet_pdf({"player": p, "history": history, "scale_max": top}), pdf.filename("ficha", p["name"])
