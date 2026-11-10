"""Operações: atletas, avaliações (validação, melhor tentativa, percentis) e exportação."""

from __future__ import annotations

import io
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence

from . import norms
from .db import connect, init_db

# Intervalos plausíveis (rejeitam erros de digitação, ex.: 1500 cm ou 0,15 m).
LIMITS = {
    "height_cm": (80, 230, "Altura (cm)"),
    "weight_kg": (10, 200, "Peso (kg)"),
    "wingspan_cm": (80, 250, "Envergadura (cm)"),
    "jump": (0.3, 3.5, "Salto horizontal (m)"),
    "run": (2.0, 10.0, "20 m corrida (s)"),
    "dribble": (2.5, 15.0, "20 m drible (s)"),
}


def parse_number(value, key: str) -> Optional[float]:
    """Aceita '1,52' / '1.52' / número / vazio. Valida o intervalo plausível de ``key``."""
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            raise ValueError("Valor inválido.")
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value == "":
            return None
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{LIMITS[key][2]}: «{value}» não é um número.")
    if not d.is_finite():
        raise ValueError(f"{LIMITS[key][2]}: valor inválido.")
    x = float(d)
    lo, hi, label = LIMITS[key]
    if not lo <= x <= hi:
        raise ValueError(f"{label}: {x:g} fora do intervalo plausível ({lo:g}–{hi:g}).")
    return x


def age_on(birth_date, on_date) -> int:
    b, d = date.fromisoformat(str(birth_date)), date.fromisoformat(str(on_date))
    return d.year - b.year - ((d.month, d.day) < (b.month, b.day))


def _iso(d) -> str:
    return d.isoformat() if isinstance(d, date) else date.fromisoformat(str(d)).isoformat()


# ── Atletas ─────────────────────────────────────────────────────────────────

def add_athlete(name: str, sex: Optional[str] = None, birth_date=None, team: Optional[str] = None,
                number: Optional[str] = None, db_path: str | None = None) -> int:
    if not name or not name.strip():
        raise ValueError("O nome do atleta é obrigatório.")
    if sex not in (None, "M", "F"):
        raise ValueError("Sexo deve ser 'M' ou 'F'.")
    bd = _iso(birth_date) if birth_date else None
    if bd and bd > date.today().isoformat():
        raise ValueError("A data de nascimento não pode ser futura.")
    init_db(db_path)
    with connect(db_path) as c:
        return c.execute(
            "INSERT INTO athletes(number, name, sex, birth_date, team) VALUES (?,?,?,?,?)",
            ((number or "").strip() or None, name.strip(), sex, bd, (team or "").strip() or None)).lastrowid


def update_athlete(athlete_id: int, name: str, sex, birth_date, team, number, db_path: str | None = None) -> None:
    if not name or not name.strip():
        raise ValueError("O nome do atleta é obrigatório.")
    if sex not in (None, "M", "F"):
        raise ValueError("Sexo deve ser 'M' ou 'F'.")
    with connect(db_path) as c:
        c.execute("UPDATE athletes SET number=?, name=?, sex=?, birth_date=?, team=? WHERE id=?",
                  ((number or "").strip() or None, name.strip(), sex, _iso(birth_date) if birth_date else None,
                   (team or "").strip() or None, athlete_id))


def list_athletes(team: str | None = None, db_path: str | None = None) -> list[dict]:
    init_db(db_path)
    q, args = "SELECT * FROM athletes", []
    if team:
        q += " WHERE team=?"; args.append(team)
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(q + " ORDER BY name COLLATE NOCASE", args)]


def get_athlete(athlete_id: int, db_path: str | None = None) -> Optional[dict]:
    init_db(db_path)
    with connect(db_path) as c:
        r = c.execute("SELECT * FROM athletes WHERE id=?", (athlete_id,)).fetchone()
    return dict(r) if r else None


def delete_athlete(athlete_id: int, db_path: str | None = None) -> None:
    with connect(db_path) as c:
        c.execute("DELETE FROM assessments WHERE athlete_id=?", (athlete_id,))
        c.execute("DELETE FROM athletes WHERE id=?", (athlete_id,))


# ── Avaliações ──────────────────────────────────────────────────────────────

def _best(trials: Sequence[Optional[float]], key: str) -> Optional[float]:
    vals = [t for t in trials if t is not None]
    if not vals:
        return None
    return max(vals) if key == "jump" else min(vals)


def build_assessment(*, eval_date, recorded_by: str, age: Optional[int] = None, birth_date=None,
                     height=None, weight=None, wingspan=None,
                     jumps: Sequence = (), runs: Sequence = (), dribbles: Sequence = (),
                     notes: Optional[str] = None) -> dict:
    """Valida os dados e devolve o registo pronto a gravar (inclui melhores tentativas)."""
    if not recorded_by or not recorded_by.strip():
        raise ValueError("Indique o nome do estagiário que regista os dados.")
    ed = _iso(eval_date)
    if ed > date.today().isoformat():
        raise ValueError("A data da avaliação não pode ser futura.")
    if birth_date:
        age = age_on(birth_date, ed)
        if age < 0:
            raise ValueError("A data de avaliação é anterior ao nascimento.")
    if age is None:
        raise ValueError("Indique a idade ou a data de nascimento do atleta.")
    if not 3 <= int(age) <= 60:
        raise ValueError("Idade fora do intervalo plausível (3–60).")
    rec = {
        "eval_date": ed, "age": int(age), "recorded_by": recorded_by.strip(),
        "height_cm": parse_number(height, "height_cm"),
        "weight_kg": parse_number(weight, "weight_kg"),
        "wingspan_cm": parse_number(wingspan, "wingspan_cm"),
        "notes": (notes or "").strip() or None,
    }
    for prefix, key, trials, n in (("jump", "jump", jumps, 3), ("run", "run", runs, 2),
                                   ("dribble", "dribble", dribbles, 2)):
        vals = [parse_number(trials[i], key) if i < len(trials) else None for i in range(n)]
        for i, v in enumerate(vals, 1):
            rec[f"{prefix}_{i}"] = v
        rec[f"{prefix}_best"] = _best(vals, key)
    measured = [rec[k] for k in ("height_cm", "weight_kg", "wingspan_cm", "jump_best", "run_best", "dribble_best")]
    if all(v is None for v in measured):
        raise ValueError("Introduza pelo menos uma medição.")
    return rec


_COLS = ("eval_date", "age", "height_cm", "weight_kg", "wingspan_cm", "jump_1", "jump_2", "jump_3", "jump_best",
         "run_1", "run_2", "run_best", "dribble_1", "dribble_2", "dribble_best", "recorded_by", "notes")


def save_assessment(athlete_id: int, rec: dict, assessment_id: int | None = None, db_path: str | None = None) -> int:
    init_db(db_path)
    with connect(db_path) as c:
        if assessment_id is None:
            return c.execute(
                f"INSERT INTO assessments(athlete_id, {', '.join(_COLS)}) VALUES (?, {', '.join('?' * len(_COLS))})",
                (athlete_id, *[rec[k] for k in _COLS])).lastrowid
        c.execute(f"UPDATE assessments SET {', '.join(k + '=?' for k in _COLS)}, updated_at=datetime('now') "
                  "WHERE id=? AND athlete_id=?", (*[rec[k] for k in _COLS], assessment_id, athlete_id))
        return assessment_id


def delete_assessment(assessment_id: int, db_path: str | None = None) -> None:
    with connect(db_path) as c:
        c.execute("DELETE FROM assessments WHERE id=?", (assessment_id,))


def with_percentiles(rec: dict, sex: Optional[str]) -> dict:
    """Acrescenta pctl_* (None quando não há referência para a idade/sexo)."""
    a = rec["age"]
    out = dict(rec)
    out["pctl_altura"] = norms.percentile("altura", rec.get("height_cm"), a)
    out["pctl_peso"] = norms.percentile("peso", rec.get("weight_kg"), a)
    out["pctl_envergadura"] = norms.percentile("envergadura", rec.get("wingspan_cm"), a)
    out["pctl_salto"] = norms.percentile("salto", rec.get("jump_best"), a, sex)
    out["pctl_corrida"] = norms.percentile("corrida", rec.get("run_best"), a)
    out["pctl_drible"] = norms.percentile("drible", rec.get("dribble_best"), a)
    return out


def athlete_history(athlete_id: int, db_path: str | None = None) -> list[dict]:
    """Avaliações do atleta (cronológicas) com percentis."""
    init_db(db_path)
    ath = get_athlete(athlete_id, db_path)
    if ath is None:
        return []
    with connect(db_path) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM assessments WHERE athlete_id=? ORDER BY eval_date, id", (athlete_id,))]
    return [with_percentiles(r, ath["sex"]) for r in rows]


def all_records(db_path: str | None = None) -> list[dict]:
    """Todas as avaliações (com dados do atleta e percentis), mais recentes primeiro."""
    init_db(db_path)
    with connect(db_path) as c:
        rows = [dict(r) for r in c.execute(
            """SELECT a.*, t.name AS athlete_name, t.number AS athlete_number, t.sex AS sex, t.team AS team
               FROM assessments a JOIN athletes t ON t.id = a.athlete_id
               ORDER BY a.eval_date DESC, a.id DESC""")]
    return [with_percentiles(r, r["sex"]) for r in rows]


EXPORT_COLUMNS = [
    ("athlete_number", "N.º"), ("athlete_name", "Nome"), ("team", "Equipa"), ("sex", "Sexo"),
    ("eval_date", "Data"), ("age", "Idade"),
    ("height_cm", "Altura (cm)"), ("pctl_altura", "Pctl Altura"),
    ("weight_kg", "Peso (kg)"), ("pctl_peso", "Pctl Peso"),
    ("wingspan_cm", "Envergadura (cm)"), ("pctl_envergadura", "Pctl Envergadura"),
    ("jump_best", "Salto horizontal (m)"), ("pctl_salto", "Pctl Salto"),
    ("run_best", "20 m corrida (s)"), ("pctl_corrida", "Pctl Corrida"),
    ("dribble_best", "20 m drible (s)"), ("pctl_drible", "Pctl Drible"),
    ("recorded_by", "Registado por"), ("notes", "Notas"),
]


def export_rows(db_path: str | None = None) -> list[dict]:
    return [{label: r.get(k) for k, label in EXPORT_COLUMNS} for r in all_records(db_path)]


def export_xlsx(db_path: str | None = None) -> bytes:
    import pandas as pd
    df = pd.DataFrame(export_rows(db_path), columns=[l for _, l in EXPORT_COLUMNS])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Avaliação Física")
    return buf.getvalue()


def export_csv(db_path: str | None = None) -> bytes:
    import pandas as pd
    df = pd.DataFrame(export_rows(db_path), columns=[l for _, l in EXPORT_COLUMNS])
    return df.to_csv(index=False).encode("utf-8-sig")   # BOM: o Excel abre os acentos bem
