"""Operações sobre jogadores, avaliações do Teste de Movimentos Defensivos e equipa."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import json

from . import defensive_movement as dm
from . import objectives as ob
from . import qualitative as ql
from .db import connect, init_db


def _iso(d) -> str:
    if isinstance(d, date):
        return d.isoformat()
    return date.fromisoformat(str(d)).isoformat()


def _check_category(category: str) -> str:
    if category not in dm.CATEGORIES:
        raise ValueError(f"Escalão inválido: {category!r}. Opções: {', '.join(dm.CATEGORIES)}")
    return category


# ── Jogadores / treinadores ─────────────────────────────────────────────────

def add_player(name: str, category: str, sex: Optional[str] = None, team: Optional[str] = None,
               birth_date=None, db_path: str | None = None) -> int:
    if not name or not name.strip():
        raise ValueError("O nome do jogador é obrigatório.")
    if sex not in (None, "M", "F"):
        raise ValueError("Sexo deve ser 'M' ou 'F'.")
    init_db(db_path)
    with connect(db_path) as c:
        cur = c.execute(
            "INSERT INTO players(name, sex, category, team, birth_date) VALUES (?,?,?,?,?)",
            (name.strip(), sex, _check_category(category), team, _iso(birth_date) if birth_date else None),
        )
        return cur.lastrowid


def list_players(category: str | None = None, team: str | None = None, db_path: str | None = None):
    init_db(db_path)
    q, args = "SELECT * FROM players WHERE 1=1", []
    if category:
        q += " AND category=?"; args.append(category)
    if team:
        q += " AND team=?"; args.append(team)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(q + " ORDER BY name", args)]


def get_or_create_coach(name: str, db_path: str | None = None) -> int:
    init_db(db_path)
    with connect(db_path) as c:
        c.execute("INSERT OR IGNORE INTO coaches(name) VALUES (?)", (name.strip(),))
        return c.execute("SELECT id FROM coaches WHERE name=?", (name.strip(),)).fetchone()["id"]


# ── Cálculo (sem gravar) ────────────────────────────────────────────────────

@dataclass
class Calculation:
    best_time: Optional[float]
    best_trial: Optional[int]
    previous_best_time: Optional[float]
    change_seconds: Optional[float]
    change_percentage: Optional[float]
    status: Optional[str]
    message: Optional[str]
    valid: bool


def previous_best(player_id: int, before_date, db_path: str | None = None) -> Optional[float]:
    """Melhor tempo da avaliação válida mais recente ANTERIOR a before_date."""
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute(
            """SELECT best_time FROM defensive_movement_tests
               WHERE player_id=? AND valid=1 AND best_time IS NOT NULL AND evaluation_date < ?
               ORDER BY evaluation_date DESC, id DESC LIMIT 1""",
            (player_id, _iso(before_date)),
        ).fetchone()
        return row["best_time"] if row else None


def calculate(times: Sequence, valid_flags: Sequence[bool] = (True, True, True),
              first_is_practice: bool = False, previous: Optional[float] = None) -> Calculation:
    """Valida as tentativas e calcula melhor tempo/evolução (botão CALCULAR)."""
    if len(times) != dm.N_TRIALS or len(valid_flags) != dm.N_TRIALS:
        raise ValueError(f"São necessárias {dm.N_TRIALS} tentativas.")
    trials = [dm.Trial(dm.parse_time(t), bool(v)) for t, v in zip(times, valid_flags)]
    res = dm.compute_best(trials, first_is_practice)
    prev = None if previous is None else dm.parse_time(previous)
    evo = dm.compute_evolution(res.best_time, prev)
    return Calculation(
        best_time=dm.r2(res.best_time), best_trial=res.best_trial,
        previous_best_time=dm.r2(prev),
        change_seconds=dm.r2(evo.change_seconds) if evo else None,
        change_percentage=dm.r2(evo.change_percentage) if evo else None,
        status=evo.status if evo else None, message=evo.message if evo else None,
        valid=res.best_time is not None,
    )


# ── Gravar / consultar ──────────────────────────────────────────────────────

def save_test(player_id: int, evaluation_date, times: Sequence, valid_flags: Sequence[bool] = (True, True, True),
              first_is_practice: bool = False, category: str | None = None, location: str | None = None,
              session_label: str | None = None, notes: str | None = None, coach_id: int | None = None,
              db_path: str | None = None) -> int:
    """Grava a avaliação (resultado bruto + derivados). O escalão por omissão é o do jogador."""
    init_db(db_path)
    with connect(db_path) as c:
        p = c.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
    if p is None:
        raise ValueError(f"Jogador {player_id} não existe.")
    category = _check_category(category or p["category"])
    d = _iso(evaluation_date)
    calc = calculate(times, valid_flags, first_is_practice, previous_best(player_id, d, db_path))
    raw = [dm.r2(dm.parse_time(t)) for t in times]
    with connect(db_path) as c:
        cur = c.execute(
            """INSERT INTO defensive_movement_tests
               (player_id, evaluation_date, category, trial_1, trial_2, trial_3,
                trial_1_valid, trial_2_valid, trial_3_valid, first_is_practice,
                best_time, best_trial, previous_best_time, change_seconds, change_percentage,
                valid, location, session_label, notes, coach_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (player_id, d, category, *raw, *[int(bool(v)) for v in valid_flags], int(first_is_practice),
             calc.best_time, calc.best_trial, calc.previous_best_time, calc.change_seconds,
             calc.change_percentage, int(calc.valid), location, session_label, notes, coach_id),
        )
        return cur.lastrowid


def history(player_id: int, db_path: str | None = None) -> list[dict]:
    """Avaliações válidas do jogador, por data crescente (para o gráfico de evolução)."""
    init_db(db_path)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(
            """SELECT * FROM defensive_movement_tests
               WHERE player_id=? AND valid=1 AND best_time IS NOT NULL
               ORDER BY evaluation_date, id""", (player_id,))]


def player_report(player_id: int, db_path: str | None = None) -> Optional[dict]:
    """Ficha individual: última avaliação válida + evolução (derivada dos dados)."""
    h = history(player_id, db_path)
    if not h:
        return None
    last = h[-1]
    evo = None
    if last["previous_best_time"] is not None:
        evo = dm.compute_evolution(dm.parse_time(last["best_time"]), dm.parse_time(last["previous_best_time"]))
    return {"test": last, "evolution": evo, "history": h}


# ── Equipa ──────────────────────────────────────────────────────────────────

def team_snapshot(category: str, date_from, date_to, team: str | None = None,
                  db_path: str | None = None) -> dict[int, float]:
    """{player_id: melhor tempo} no intervalo, para UM escalão (nunca mistura escalões)."""
    _check_category(category)
    init_db(db_path)
    q = """SELECT t.player_id, MIN(t.best_time) AS bt
           FROM defensive_movement_tests t JOIN players p ON p.id = t.player_id
           WHERE t.valid=1 AND t.best_time IS NOT NULL AND t.category=?
             AND t.evaluation_date BETWEEN ? AND ?"""
    args = [category, _iso(date_from), _iso(date_to)]
    if team:
        q += " AND p.team=?"; args.append(team)
    with connect(db_path) as c:
        return {r["player_id"]: r["bt"] for r in c.execute(q + " GROUP BY t.player_id", args)}


def team_comparison(category: str, before: tuple, after: tuple, team: str | None = None,
                    db_path: str | None = None) -> dm.TeamComparison:
    """Compara a equipa entre dois momentos; cada momento é (data_inicio, data_fim)."""
    return dm.compare_groups(
        team_snapshot(category, *before, team=team, db_path=db_path),
        team_snapshot(category, *after, team=team, db_path=db_path),
    )


# ── Relatório do treinador ──────────────────────────────────────────────────

def coach_report_text(player_id: int, db_path: str | None = None) -> str:
    rep = player_report(player_id, db_path)
    if rep is None:
        return "Sem avaliação válida do Teste de Movimentos Defensivos."
    t, evo = rep["test"], rep["evolution"]
    lines = [
        "TESTE:", "Movimentos Defensivos", "",
        "Objetivo:", "Avaliar os movimentos defensivos básicos.", "",
        "Resultado individual:", dm.fmt_seconds(t["best_time"]), "",
        "Melhor tentativa:", f"Tentativa {t['best_trial']}", "",
        "Resultado anterior:", dm.fmt_seconds(t["previous_best_time"]), "",
        "Variação:", dm.fmt_seconds(t["change_seconds"], signed=True), "",
        "Variação percentual:", dm.fmt_pct(t["change_percentage"], signed=True), "",
        "Evolução:", evo.label if evo else "Sem avaliação anterior",
    ]
    if evo:
        lines += ["", evo.message]
    return "\n".join(lines)


# ── Avaliação qualitativa ───────────────────────────────────────────────────

def save_qualitative(player_id: int, evaluation_date, ratings: dict, strengths_note: str | None = None,
                     improve_note: str | None = None, session_label: str | None = None,
                     coach_id: int | None = None, db_path: str | None = None) -> int:
    """Grava a grelha de observação (classificações 1–4 por critério)."""
    init_db(db_path)
    clean = ql.validate_ratings(ratings)
    avg = ql.summarize(clean).average
    with connect(db_path) as c:
        p = c.execute("SELECT category FROM players WHERE id=?", (player_id,)).fetchone()
        if p is None:
            raise ValueError(f"Jogador {player_id} não existe.")
        cur = c.execute(
            """INSERT INTO qualitative_evaluations
               (player_id, evaluation_date, category, ratings, average, strengths_note,
                improve_note, session_label, coach_id) VALUES (?,?,?,?,?,?,?,?,?)""",
            (player_id, _iso(evaluation_date), p["category"], json.dumps(clean), avg,
             strengths_note, improve_note, session_label, coach_id))
        return cur.lastrowid


def qualitative_history(player_id: int, db_path: str | None = None) -> list[dict]:
    """Avaliações qualitativas do jogador por data crescente (ratings já como dict)."""
    init_db(db_path)
    with connect(db_path) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM qualitative_evaluations WHERE player_id=? ORDER BY evaluation_date, id",
            (player_id,))]
    for r in rows:
        r["ratings"] = json.loads(r["ratings"])
    return rows


def qualitative_report_text(player_id: int, db_path: str | None = None) -> str:
    h = qualitative_history(player_id, db_path)
    if not h:
        return "Sem avaliação qualitativa registada."
    with connect(db_path) as c:
        name = c.execute("SELECT name FROM players WHERE id=?", (player_id,)).fetchone()["name"]
    text = ql.feedback_text(name, h[-1]["ratings"], h[-2]["ratings"] if len(h) > 1 else None)
    for label, key in (("Notas — pontos fortes", "strengths_note"), ("Notas — a melhorar", "improve_note")):
        if h[-1][key]:
            text += f"\n\n{label}: {h[-1][key]}"
    return text


# ── Objetivos individuais ───────────────────────────────────────────────────

def _latest_objective_scores(player_id: int, c) -> dict[str, dict[str, int]]:
    """Última avaliação de cada objetivo → {chave: {indicador: valor}}."""
    out: dict[str, dict[str, int]] = {}
    for r in c.execute("SELECT * FROM objective_evaluations WHERE player_id=? ORDER BY evaluation_date, id",
                       (player_id,)):
        out[r["objective_key"]] = {k: r[k] for k in ob.INDICATORS if r[k] is not None}
    return out


def objective_suggestions(player_id: int, limit: int = 4, db_path: str | None = None) -> dict[str, list[dict]]:
    """Objetivos sugeridos {'off': [...], 'def': [...]} a partir dos dados do jogador."""
    init_db(db_path)
    qh = qualitative_history(player_id, db_path)
    h = history(player_id, db_path)
    change = h[-1]["change_percentage"] if h else None
    with connect(db_path) as c:
        last = _latest_objective_scores(player_id, c)
    qual = qh[-1]["ratings"] if qh else None
    return {s: ob.suggest_objectives(s, qual, change, last, limit) for s in ob.SIDES}


def set_objectives(player_id: int, selections: Sequence[tuple[str, str]], db_path: str | None = None) -> None:
    """Define os objetivos ativos do jogador: lista de (chave, origem 'sugerido'|'manual')."""
    for key, src in selections:
        if key not in ob.LIBRARY:
            raise ValueError(f"Objetivo desconhecido: {key!r}")
        if src not in ("sugerido", "manual"):
            raise ValueError(f"Origem inválida: {src!r}")
    init_db(db_path)
    with connect(db_path) as c:
        if c.execute("SELECT 1 FROM players WHERE id=?", (player_id,)).fetchone() is None:
            raise ValueError(f"Jogador {player_id} não existe.")
        c.execute("UPDATE player_objectives SET active=0 WHERE player_id=?", (player_id,))
        for key, src in selections:
            c.execute("""INSERT INTO player_objectives(player_id, objective_key, side, source, active)
                         VALUES (?,?,?,?,1)
                         ON CONFLICT(player_id, objective_key) DO UPDATE SET active=1, source=excluded.source""",
                      (player_id, key, ob.LIBRARY[key]["side"], src))


def get_objectives(player_id: int, db_path: str | None = None) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM player_objectives WHERE player_id=? AND active=1 ORDER BY side DESC, id",
            (player_id,))]
    for r in rows:
        r["text"] = ob.LIBRARY[r["objective_key"]]["text"]
    return rows


def save_objective_evaluation(player_id: int, evaluation_date, scores: dict[str, dict],
                              notes: str | None = None, coach_id: int | None = None,
                              db_path: str | None = None) -> int:
    """Grava a avaliação (1–5 × 4 indicadores) de um ou mais objetivos. Devolve nº de linhas."""
    init_db(db_path)
    clean = {k: ob.validate_scores(v) for k, v in scores.items() if any(x is not None for x in v.values())}
    if not clean:
        raise ValueError("Preencha pelo menos um indicador de um objetivo.")
    for k in clean:
        if k not in ob.LIBRARY:
            raise ValueError(f"Objetivo desconhecido: {k!r}")
    with connect(db_path) as c:
        if c.execute("SELECT 1 FROM players WHERE id=?", (player_id,)).fetchone() is None:
            raise ValueError(f"Jogador {player_id} não existe.")
        for k, s in clean.items():
            c.execute("""INSERT INTO objective_evaluations
                         (player_id, evaluation_date, objective_key, execution, frequency, application,
                          evolution, average, notes, coach_id) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                      (player_id, _iso(evaluation_date), k, s.get("execution"), s.get("frequency"),
                       s.get("application"), s.get("evolution"), ob.objective_average(s), notes, coach_id))
    return len(clean)


def objective_history(player_id: int, db_path: str | None = None) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM objective_evaluations WHERE player_id=? ORDER BY evaluation_date, id", (player_id,))]


# ── Proposta de objetivos com IA ────────────────────────────────────────────

def ai_payload(player_id: int, limit: int = 4, db_path: str | None = None) -> dict:
    """Dados ANONIMIZADOS do jogador a enviar à IA (ver ai_coach)."""
    from . import ai_coach
    init_db(db_path)
    qh = qualitative_history(player_id, db_path)
    h = history(player_id, db_path)
    with connect(db_path) as c:
        p = c.execute("SELECT category FROM players WHERE id=?", (player_id,)).fetchone()
        if p is None:
            raise ValueError(f"Jogador {player_id} não existe.")
        last = _latest_objective_scores(player_id, c)
    return ai_coach.build_payload(p["category"], qh[-1]["ratings"] if qh else None,
                                  h[-1]["change_percentage"] if h else None, last, limit)


def ai_objective_proposal(player_id: int, limit: int = 4, client=None, db_path: str | None = None) -> dict:
    from . import ai_coach
    return ai_coach.propose(ai_payload(player_id, limit, db_path), client)


def latest_objective_scores(player_id: int, db_path: str | None = None) -> dict[str, dict[str, int]]:
    init_db(db_path)
    with connect(db_path) as c:
        return _latest_objective_scores(player_id, c)
