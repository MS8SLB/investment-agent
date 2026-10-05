"""Lógica de acesso a dados partilhada pelas páginas, gráficos e relatórios."""
from collections import defaultdict
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from . import scoring
from .models import (Athlete, Attempt, Category, CategoryTest, Evaluation, ReferenceTable, Team, TestDef,
                     TestResult)


class ValidationError(ValueError):
    pass


def parse_number(txt):
    txt = (txt or "").strip().replace(",", ".")
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        raise ValidationError(f"Valor inválido: «{txt}»")


def parse_date(txt):
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        return date.fromisoformat(txt)
    except ValueError:
        raise ValidationError(f"Data inválida: «{txt}»")


def check_value(test: TestDef, v: float):
    if v < 0:
        raise ValidationError(f"{test.name}: os valores não podem ser negativos.")
    if test.unit == "s" and v == 0:
        raise ValidationError(f"{test.name}: o tempo tem de ser maior que zero.")
    if test.max_value is not None and v > test.max_value:
        raise ValidationError(f"{test.name}: o valor máximo possível é {test.max_value:g}.")


def enabled_test_ids(db: Session, category_id: int) -> set[int]:
    return set(db.scalars(select(CategoryTest.test_id).where(CategoryTest.category_id == category_id)))


def save_result(db: Session, evaluation: Evaluation, test: TestDef, values: list, notes: str = ""):
    """Cria/atualiza o resultado de um teste numa avaliação. values = lista (posição = nº da tentativa)."""
    vals = [(i + 1, v) for i, v in enumerate(values) if v is not None]
    for _, v in vals:
        check_value(test, v)
    res = next((r for r in evaluation.results if r.test_id == test.id), None)
    if not vals and not (notes or "").strip():
        if res is not None:
            evaluation.results.remove(res)
        return None
    if res is None:
        res = TestResult(test_id=test.id, test=test)
        evaluation.results.append(res)
    res.attempts.clear()
    db.flush()
    for n, v in vals:
        res.attempts.append(Attempt(number=n, value=v))
    res.best_value = scoring.best_of([v for _, v in vals], test.direction)
    res.notes = (notes or "").strip()
    return res


def get_or_create_evaluation(db: Session, athlete: Athlete, on: date, evaluator="", notes=None):
    ev = db.scalar(select(Evaluation).where(Evaluation.athlete_id == athlete.id, Evaluation.date == on))
    if ev is None:
        ev = Evaluation(athlete=athlete, athlete_id=athlete.id, category_id=athlete.category_id, date=on,
                        evaluator=evaluator or "")
        db.add(ev)
        db.flush()
    elif evaluator:
        ev.evaluator = evaluator
    if notes is not None:
        ev.notes = notes
    return ev


# ---------------------------------------------------------------- consultas para análise

def result_rows(db: Session, *, test_id=None, athlete_id=None, category_id=None, team_id=None, sex=None,
                date_from=None, date_to=None):
    """Linhas (resultado, avaliação, atleta) filtradas, ordenadas por data."""
    q = (select(TestResult).join(Evaluation).join(Athlete, Evaluation.athlete_id == Athlete.id)
         .options(joinedload(TestResult.evaluation).joinedload(Evaluation.athlete),
                  joinedload(TestResult.test), joinedload(TestResult.attempts))
         .where(TestResult.best_value.is_not(None)))
    if test_id:
        q = q.where(TestResult.test_id == test_id)
    if athlete_id:
        q = q.where(Evaluation.athlete_id == athlete_id)
    if category_id:
        q = q.where(Evaluation.category_id == category_id)
    if team_id:
        q = q.where(Athlete.team_id == team_id)
    if sex:
        q = q.where(Athlete.sex == sex)
    if date_from:
        q = q.where(Evaluation.date >= date_from)
    if date_to:
        q = q.where(Evaluation.date <= date_to)
    q = q.order_by(Evaluation.date, Athlete.name)
    return list(db.scalars(q).unique())


def series_by_date(rows, direction):
    """[(data, descrição estatística dos melhores resultados nessa data)]."""
    by = defaultdict(list)
    for r in rows:
        by[r.evaluation.date].append(r.best_value)
    return [(d, scoring.describe(v)) for d, v in sorted(by.items())]


def latest_per_athlete(rows):
    out = {}
    for r in rows:  # ordenadas por data
        out[r.evaluation.athlete_id] = r
    return out


def first_per_athlete(rows):
    out = {}
    for r in rows:
        out.setdefault(r.evaluation.athlete_id, r)
    return out


def reference_rows_for(db: Session, test_id, category_id, sex):
    """Linhas da tabela de referência ativa aplicável; [] se não existir (resultados absolutos)."""
    tables = db.scalars(select(ReferenceTable).where(ReferenceTable.test_id == test_id,
                                                     ReferenceTable.active.is_(True))).all()
    for t in sorted(tables, key=lambda t: ((t.category_id is None) + (t.sex is None))):
        if (t.category_id in (None, category_id)) and (t.sex in (None, sex)):
            return t, list(t.rows)
    return None, []


def has_reference(db: Session, test_id, category_id) -> bool:
    return any(t.category_id in (None, category_id) for t in db.scalars(
        select(ReferenceTable).where(ReferenceTable.test_id == test_id, ReferenceTable.active.is_(True))))


def dashboard_counts(db: Session):
    athletes = db.scalars(select(Athlete)).all()
    evaluated_ids = set(db.scalars(select(Evaluation.athlete_id).join(TestResult)))
    cats = db.scalars(select(Category).order_by(Category.sort)).all()
    dist = []
    for c in cats:
        tot = sum(1 for a in athletes if a.category_id == c.id)
        ev = sum(1 for a in athletes if a.category_id == c.id and a.id in evaluated_ids)
        dist.append({"name": c.name, "athletes": tot, "evaluated": ev})
    n_eval = len(db.scalars(select(Evaluation.id).join(TestResult).distinct()).all())
    n_results = len(db.scalars(select(TestResult.id)).all())
    return {"athletes": len(athletes), "evaluated": len(evaluated_ids), "dist": dist,
            "evaluations": n_eval, "results": n_results}
