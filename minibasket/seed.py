"""Dados de teste (fictícios), claramente identificados e removíveis.

Tudo o que é criado aqui leva is_demo=1 (clubes, equipas, jogadores, avaliações e contas) e os nomes
terminam em «Teste». Nunca se misturam com dados reais: `remove_demo` só apaga linhas marcadas e recusa-se
a avançar se existir algum dado real ligado a dados de teste.

Linha de comandos:  python -m minibasket.seed load | remove | status
"""

from __future__ import annotations

import random
import sys
from datetime import date
from typing import Optional

from . import competencies as comp
from . import evaluations as ev
from . import service
from .db import CATEGORIES, connect, init_db
from .service import ValidationError

DEMO_CLUB = "Clube de Demonstração (dados de teste)"
PLAYERS_PER_CATEGORY = {"Sub-8": 5, "Sub-10": 8, "Sub-12": 8}
FIRST_NAMES = [("Afonso", "M"), ("Beatriz", "F"), ("Carlos", "M"), ("Diana", "F"), ("Eduardo", "M"), ("Filipa", "F"),
               ("Gonçalo", "M"), ("Helena", "F"), ("Ivo", "M"), ("Joana", "F"), ("Kevin", "M"), ("Laura", "F"),
               ("Miguel", "M"), ("Nuno", "M"), ("Olívia", "F"), ("Pedro", "M"), ("Rita", "F"), ("Sofia", "F"),
               ("Tiago", "M"), ("Vera", "F"), ("Xavier", "M"), ("Yara", "F"), ("Zeca", "M")]
MOMENTS = ("Avaliação Inicial", "1.º Período", "2.º Período", "Avaliação Final")
PASSWORD_NOTE = "As contas de teste não têm palavra-passe; o administrador pode definir uma em «Utilizadores»."


def season_start(today: date) -> int:
    """Ano de início da época concluída mais recente (30/06 do ano seguinte já passou)."""
    return today.year - 1 if today.month >= 7 else today.year - 2


def _clamp(v: int) -> int:
    return max(1, min(5, v))


def _scores(rng: random.Random, step: int, profile: dict) -> dict:
    return {k: _clamp(profile[k]["base"] + round(step * profile[k]["growth"])) for k in comp.KEYS}


def summary(db_path: str | None = None) -> dict:
    """Quantos dados de teste existem."""
    init_db(db_path)
    with connect(db_path) as c:
        q = lambda sql: c.execute(sql).fetchone()[0]
        return {"clubs": q("SELECT COUNT(*) FROM clubs WHERE is_demo=1"), "teams": q("SELECT COUNT(*) FROM teams WHERE is_demo=1"),
                "players": q("SELECT COUNT(*) FROM players WHERE is_demo=1"),
                "evaluations": q("SELECT COUNT(*) FROM evaluations WHERE is_demo=1"),
                "users": q("SELECT COUNT(*) FROM users WHERE is_demo=1")}


def has_demo(db_path: str | None = None) -> bool:
    return any(summary(db_path).values())


def load_demo(seed: int = 2026, today: Optional[date] = None, db_path: str | None = None) -> dict:
    """Cria 5 jogadores Sub-8, 8 Sub-10 e 8 Sub-12 com 3 ou 4 avaliações cada (época concluída mais recente)."""
    today = today or date.today()
    init_db(db_path)
    if has_demo(db_path):
        raise ValidationError("Já existem dados de teste. Remova-os primeiro se quiser voltar a carregá-los.")
    rng = random.Random(seed)
    s0 = season_start(today)
    season = f"{s0}/{s0 + 1}"
    dates = [f"{s0}-09-15", f"{s0}-12-15", f"{s0 + 1}-03-15", f"{s0 + 1}-06-15"]
    with connect(db_path) as c:
        club_id = c.execute("INSERT INTO clubs(name, is_demo) VALUES (?,1)", (DEMO_CLUB,)).lastrowid
    from . import auth
    names = list(FIRST_NAMES)
    n_players = n_evals = 0
    made = {"incomplete": 0, "corrected": 0}
    guardian_target = None
    for cat in CATEGORIES:
        team_id = service.create_team(club_id, f"{cat} A (teste)", cat, season, is_demo=True, db_path=db_path)
        tag = cat.lower().replace("-", "")
        coach = auth.create_user(f"treinador.{tag}.teste", f"Treinador {cat} (teste)", "coach", None, club_id=club_id,
                                 is_demo=True, db_path=db_path)
        auth.set_team_assignments(coach, [team_id], db_path)
        for i in range(PLAYERS_PER_CATEGORY[cat]):
            first, sex = names.pop(rng.randrange(len(names)))
            born = s0 - {"Sub-8": 7, "Sub-10": 9, "Sub-12": 11}[cat]
            pid = service.create_player(f"{first} Teste", team_id, f"{born}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                                        sex, rng.randint(4, 15), f"{s0}-09-01", "Jogador fictício (dados de teste).",
                                        is_demo=True, db_path=db_path)
            n_players += 1
            if cat == "Sub-10" and guardian_target is None:
                guardian_target = pid
            profile = {k: {"base": _clamp(round(rng.gauss(2.0, 0.7))), "growth": rng.uniform(0.2, 0.9)} for k in comp.KEYS}
            profile["footwork"]["growth"] *= 0.5                                  # só para variar o perfil
            n = 3 if i % 3 == 2 else 4                                           # alguns só têm 3 avaliações
            first_eval = None
            for step in range(n):
                scores = _scores(rng, step, profile)
                if cat == "Sub-10" and i == 1 and step == 1:                     # exemplo de avaliação incompleta
                    scores["passing"] = scores["finishing"] = None
                    made["incomplete"] += 1
                last = step == n - 1
                eid = ev.create_evaluation(
                    pid, dates[step], MOMENTS[step], scores,
                    notes={"shooting": "Melhorou a preparação dos pés."} if step else None,
                    coach_id=coach, general_notes="Evolução regular ao longo do período (dados de teste).",
                    next_objectives="Melhorar a qualidade das decisões e a utilização do espaço." if last else None,
                    parent_message="Muito empenho nos treinos!" if last else None, is_demo=True, db_path=db_path)
                first_eval = first_eval or eid
                n_evals += 1
            if cat == "Sub-12" and i == 0:                                       # exemplo de correção (versão nova)
                fixed = _scores(rng, 0, profile)
                fixed["shooting"] = _clamp((fixed["shooting"] or 1) + 1)
                ev.correct_evaluation(first_eval, dates[0], MOMENTS[0], fixed, coach_id=coach, db_path=db_path)
                n_evals += 1
                made["corrected"] += 1
    kid = auth.create_user("encarregado.teste", "Encarregado de educação (teste)", "guardian", None, is_demo=True,
                           db_path=db_path)
    auth.set_guardian_links(kid, [guardian_target], db_path)
    return {"season": season, "players": n_players, "evaluations": n_evals, **made, **summary(db_path)}


def remove_demo(db_path: str | None = None) -> dict:
    """Apaga só os dados de teste. Recusa se houver dados reais ligados a eles (nunca se perdem dados reais)."""
    init_db(db_path)
    before = summary(db_path)
    with connect(db_path) as c:
        ids = lambda sql: [r[0] for r in c.execute(sql)]
        players, teams = ids("SELECT id FROM players WHERE is_demo=1"), ids("SELECT id FROM teams WHERE is_demo=1")
        users, clubs = ids("SELECT id FROM users WHERE is_demo=1"), ids("SELECT id FROM clubs WHERE is_demo=1")
        marks = lambda xs: ",".join("?" * len(xs)) or "NULL"
        checks = [
            ("jogadores reais em equipas de teste",
             f"SELECT 1 FROM team_memberships m JOIN players p ON p.id=m.player_id WHERE p.is_demo=0 AND m.team_id IN ({marks(teams)})", teams),
            ("avaliações reais de jogadores ou equipas de teste",
             f"SELECT 1 FROM evaluations WHERE is_demo=0 AND (player_id IN ({marks(players)}) OR team_id IN ({marks(teams)}))", players + teams),
            ("equipas reais em clubes de teste", f"SELECT 1 FROM teams WHERE is_demo=0 AND club_id IN ({marks(clubs)})", clubs),
            ("contas reais em clubes de teste", f"SELECT 1 FROM users WHERE is_demo=0 AND club_id IN ({marks(clubs)})", clubs),
        ]
        for what, sql, args in checks:
            if args and c.execute(sql, args).fetchone():
                raise ValidationError(f"Não é possível remover os dados de teste: existem {what}. Nada foi apagado.")
        evs = ids("SELECT id FROM evaluations WHERE is_demo=1")
        if evs:
            c.execute(f"DELETE FROM evaluation_scores WHERE evaluation_id IN ({marks(evs)})", evs)
            c.execute(f"DELETE FROM evaluations WHERE id IN ({marks(evs)})", evs)
        if players or teams:
            c.execute(f"DELETE FROM team_memberships WHERE player_id IN ({marks(players)}) OR team_id IN ({marks(teams)})", players + teams)
        for table in ("guardians_players", "player_access", "team_coaches"):
            c.execute(f"DELETE FROM {table} WHERE user_id IN ({marks(users)}) OR "
                      f"{'team_id' if table == 'team_coaches' else 'player_id'} IN ({marks(teams if table == 'team_coaches' else players)})",
                      users + (teams if table == "team_coaches" else players))
        for table, xs in (("players", players), ("teams", teams), ("users", users), ("clubs", clubs)):
            if xs:
                c.execute(f"DELETE FROM {table} WHERE id IN ({marks(xs)})", xs)
    return before


def main(argv: list[str] | None = None) -> int:
    cmd = (argv if argv is not None else sys.argv[1:]) or ["status"]
    try:
        if cmd[0] == "load":
            print("Dados de teste carregados:", load_demo())
            print(PASSWORD_NOTE)
        elif cmd[0] == "remove":
            print("Removido:", remove_demo())
        elif cmd[0] == "status":
            print("Dados de teste existentes:", summary())
        else:
            print(__doc__)
            return 2
    except ValidationError as e:
        print("Erro:", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
