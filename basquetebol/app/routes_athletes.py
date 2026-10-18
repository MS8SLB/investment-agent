from datetime import date

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from . import scoring, services
from .db import get_db
from .models import Athlete, Category, Evaluation, Team, TestDef, TestResult
from .web import msg_url, render

router = APIRouter()


def _form_context(db, athlete=None, error=None, form=None):
    return dict(categories=db.scalars(select(Category).order_by(Category.sort)).all(),
                teams=db.scalars(select(Team).order_by(Team.club, Team.name)).all(),
                clubs=sorted({c for c in db.scalars(select(Athlete.club)) if c}
                             | {c for c in db.scalars(select(Team.club)) if c}),
                athlete=athlete, error=error, form=form or {})


@router.get("/atletas")
def athlete_list(request: Request, q: str = "", category_id: str = "", team_id: str = "", sex: str = "",
                 db: Session = Depends(get_db)):
    stmt = select(Athlete).order_by(Athlete.name)
    if q.strip():
        stmt = stmt.where(Athlete.name.ilike(f"%{q.strip()}%"))
    if category_id:
        stmt = stmt.where(Athlete.category_id == int(category_id))
    if team_id:
        stmt = stmt.where(Athlete.team_id == int(team_id))
    if sex in ("M", "F"):
        stmt = stmt.where(Athlete.sex == sex)
    athletes = db.scalars(stmt).all()
    counts = dict(db.execute(select(Evaluation.athlete_id, func.count(Evaluation.id)).group_by(
        Evaluation.athlete_id)).all())
    last = dict(db.execute(select(Evaluation.athlete_id, func.max(Evaluation.date)).group_by(
        Evaluation.athlete_id)).all())
    return render(request, "athletes.html", athletes=athletes, counts=counts, last=last, q=q,
                  category_id=category_id, team_id=team_id, sex=sex,
                  categories=db.scalars(select(Category).order_by(Category.sort)).all(),
                  teams=db.scalars(select(Team).order_by(Team.club, Team.name)).all(), today=date.today())


def _read_form(db, name, birth_date, sex, club, team_id, category_id, notes):
    name = name.strip()
    if not name:
        raise services.ValidationError("O nome é obrigatório.")
    bd = services.parse_date(birth_date)
    if bd is None or bd > date.today():
        raise services.ValidationError("Indique uma data de nascimento válida.")
    if sex not in ("M", "F"):
        raise services.ValidationError("Indique o sexo.")
    if not category_id:
        raise services.ValidationError("Indique o escalão.")
    team = db.get(Team, int(team_id)) if team_id else None
    club = club.strip() or (team.club if team else "")
    return dict(name=name, birth_date=bd, sex=sex, club=club, team_id=team.id if team else None,
                category_id=int(category_id), notes=notes.strip())


@router.get("/atletas/novo")
def athlete_new(request: Request, db: Session = Depends(get_db)):
    return render(request, "athlete_form.html", **_form_context(db))


@router.post("/atletas/novo")
def athlete_create(request: Request, name: str = Form(""), birth_date: str = Form(""), sex: str = Form(""),
                   club: str = Form(""), team_id: str = Form(""), category_id: str = Form(""),
                   notes: str = Form(""), db: Session = Depends(get_db)):
    form = dict(name=name, birth_date=birth_date, sex=sex, club=club, team_id=team_id, category_id=category_id,
                notes=notes)
    try:
        data = _read_form(db, **form)
    except services.ValidationError as e:
        return render(request, "athlete_form.html", **_form_context(db, error=str(e), form=form))
    a = Athlete(**data)
    db.add(a)
    db.commit()
    return RedirectResponse(msg_url(f"/atletas/{a.id}", "Atleta registado."), 303)


@router.get("/atletas/{athlete_id}/editar")
def athlete_edit(athlete_id: int, request: Request, db: Session = Depends(get_db)):
    a = db.get(Athlete, athlete_id)
    if not a:
        return RedirectResponse(msg_url("/atletas", err="Atleta não encontrado."), 303)
    return render(request, "athlete_form.html", **_form_context(db, athlete=a))


@router.post("/atletas/{athlete_id}/editar")
def athlete_update(athlete_id: int, request: Request, name: str = Form(""), birth_date: str = Form(""),
                   sex: str = Form(""), club: str = Form(""), team_id: str = Form(""),
                   category_id: str = Form(""), notes: str = Form(""), db: Session = Depends(get_db)):
    a = db.get(Athlete, athlete_id)
    form = dict(name=name, birth_date=birth_date, sex=sex, club=club, team_id=team_id, category_id=category_id,
                notes=notes)
    try:
        data = _read_form(db, **form)
    except services.ValidationError as e:
        return render(request, "athlete_form.html", **_form_context(db, athlete=a, error=str(e), form=form))
    for k, v in data.items():
        setattr(a, k, v)
    db.commit()
    return RedirectResponse(msg_url(f"/atletas/{a.id}", "Dados guardados."), 303)


@router.post("/atletas/{athlete_id}/apagar")
def athlete_delete(athlete_id: int, db: Session = Depends(get_db)):
    a = db.get(Athlete, athlete_id)
    if a:
        db.delete(a)
        db.commit()
    return RedirectResponse(msg_url("/atletas", "Atleta eliminado (com o respetivo histórico)."), 303)


@router.get("/atletas/{athlete_id}")
def athlete_detail(athlete_id: int, request: Request, db: Session = Depends(get_db)):
    a = db.get(Athlete, athlete_id)
    if not a:
        return RedirectResponse(msg_url("/atletas", err="Atleta não encontrado."), 303)
    rows = services.result_rows(db, athlete_id=a.id)
    by_test: dict[int, list] = {}
    for r in rows:
        by_test.setdefault(r.test_id, []).append(r)
    blocks = []
    for test_id, rs in by_test.items():
        test = rs[0].test
        _, ref_rows = services.reference_rows_for(db, test_id, a.category_id, a.sex)
        history = []
        prev = None
        for r in rs:
            d, imp = scoring.change(prev.best_value if prev else None, r.best_value, test.direction)
            history.append(dict(r=r, delta=d, improved=imp,
                                cls=scoring.classify(r.best_value, ref_rows) if ref_rows else None))
            prev = r
        first, last = rs[0].best_value, rs[-1].best_value
        total, total_imp = scoring.change(first, last, test.direction)
        blocks.append(dict(test=test, history=history, has_ref=bool(ref_rows), total=total, total_imp=total_imp,
                           chart=dict(labels=[r.evaluation.date.strftime("%d/%m/%Y") for r in rs],
                                      vals=[r.best_value for r in rs])))
    blocks.sort(key=lambda b: b["test"].sort)
    evaluations = db.scalars(select(Evaluation).where(Evaluation.athlete_id == a.id)
                             .order_by(Evaluation.date.desc())).all()
    return render(request, "athlete_detail.html", a=a, blocks=blocks, evaluations=evaluations,
                  age=scoring.age_on(a.birth_date, date.today()))


# ---------------------------------------------------------------- equipas

@router.get("/equipas")
def team_list(request: Request, db: Session = Depends(get_db)):
    teams = db.scalars(select(Team).order_by(Team.club, Team.name)).all()
    n = dict(db.execute(select(Athlete.team_id, func.count(Athlete.id)).group_by(Athlete.team_id)).all())
    return render(request, "teams.html", teams=teams, n=n,
                  categories=db.scalars(select(Category).order_by(Category.sort)).all())


@router.post("/equipas")
def team_create(name: str = Form(""), club: str = Form(""), category_id: str = Form(""),
                db: Session = Depends(get_db)):
    name, club = name.strip(), club.strip()
    if not name:
        return RedirectResponse(msg_url("/equipas", err="O nome da equipa é obrigatório."), 303)
    if db.scalar(select(Team).where(Team.name == name, Team.club == club)):
        return RedirectResponse(msg_url("/equipas", err="Essa equipa já existe."), 303)
    db.add(Team(name=name, club=club, category_id=int(category_id) if category_id else None))
    db.commit()
    return RedirectResponse(msg_url("/equipas", "Equipa criada."), 303)


@router.post("/equipas/{team_id}/apagar")
def team_delete(team_id: int, db: Session = Depends(get_db)):
    t = db.get(Team, team_id)
    if t:
        for a in db.scalars(select(Athlete).where(Athlete.team_id == t.id)):
            a.team_id = None
        db.delete(t)
        db.commit()
    return RedirectResponse(msg_url("/equipas", "Equipa eliminada (os atletas mantêm-se, sem equipa)."), 303)
