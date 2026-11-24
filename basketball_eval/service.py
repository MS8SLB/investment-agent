"""Operações sobre jogadores, avaliações do Teste de Movimentos Defensivos e equipa."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import json

from . import defensive_movement as dm
from . import norms
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


def team_evolution(category: str, team: str | None = None, db_path: str | None = None) -> list[dict]:
    """Evolução coletiva: por data de avaliação, estatísticas do melhor tempo de cada jogador."""
    _check_category(category)
    init_db(db_path)
    q = """SELECT t.evaluation_date AS d, t.player_id, MIN(t.best_time) AS bt
           FROM defensive_movement_tests t JOIN players p ON p.id = t.player_id
           WHERE t.valid=1 AND t.best_time IS NOT NULL AND t.category=?"""
    args = [category]
    if team:
        q += " AND p.team=?"; args.append(team)
    by_date: dict[str, list[float]] = {}
    with connect(db_path) as c:
        for r in c.execute(q + " GROUP BY t.evaluation_date, t.player_id ORDER BY t.evaluation_date", args):
            by_date.setdefault(r["d"], []).append(r["bt"])
    out = []
    for d, xs in by_date.items():
        g = dm.group_stats(xs)
        out.append({"date": d, "n": g.n, "mean": g.mean, "median": g.median, "best": g.best, "worst": g.worst})
    return out


def team_sheet(category: str, date_from, date_to, team: str | None = None,
               db_path: str | None = None) -> dict:
    """Folha de equipa (Deslizamento Defensivo): linhas por jogador + média / mais rápido / mais lento."""
    snap = team_snapshot(category, date_from, date_to, team=team, db_path=db_path)
    rows = []
    with connect(db_path) as c:
        for pid, bt in snap.items():
            p = c.execute("SELECT * FROM players WHERE id=?", (pid,)).fetchone()
            d = c.execute("""SELECT evaluation_date FROM defensive_movement_tests
                             WHERE player_id=? AND valid=1 AND best_time=? AND category=?
                               AND evaluation_date BETWEEN ? AND ? ORDER BY evaluation_date DESC LIMIT 1""",
                          (pid, bt, category, _iso(date_from), _iso(date_to))).fetchone()
            age = norms.age_at(p["birth_date"], d["evaluation_date"]) if p["birth_date"] and d else None
            ref = norms.percentile_score(bt, age, p["sex"], db_path) if age is not None else None
            rows.append({"name": p["name"], "club": p["team"], "age": age, "best_time": bt,
                         "percentile": None if ref is None else round(ref["score"]),
                         "percentile_bound": None if ref is None else ref["bound"]})
    rows.sort(key=lambda r: r["best_time"])
    g = dm.group_stats([r["best_time"] for r in rows])
    return {"rows": rows, "mean": g.mean, "fastest": g.best, "slowest": g.worst, "n": g.n}


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
