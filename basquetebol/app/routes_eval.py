from datetime import date

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from . import services
from .db import get_db
from .models import Athlete, Category, Evaluation, Team, TestDef
from .web import msg_url, render

router = APIRouter()


@router.get("/avaliacoes")
def evaluation_list(request: Request, athlete_id: str = "", category_id: str = "", team_id: str = "",
                    db: Session = Depends(get_db)):
    q = (select(Evaluation).join(Athlete, Evaluation.athlete_id == Athlete.id)
         .options(joinedload(Evaluation.athlete), joinedload(Evaluation.results))
         .order_by(Evaluation.date.desc(), Athlete.name))
    if athlete_id:
        q = q.where(Evaluation.athlete_id == int(athlete_id))
    if category_id:
        q = q.where(Evaluation.category_id == int(category_id))
    if team_id:
        q = q.where(Athlete.team_id == int(team_id))
    evs = db.scalars(q).unique().all()
    return render(request, "evaluations.html", evs=evs, athlete_id=athlete_id, category_id=category_id,
                  team_id=team_id, athletes=db.scalars(select(Athlete).order_by(Athlete.name)).all(),
                  categories=db.scalars(select(Category).order_by(Category.sort)).all(),
                  teams=db.scalars(select(Team).order_by(Team.club, Team.name)).all())


# ---------------------------------------------------------------- avaliação individual

def _tests_for(db, category_id):
    on = services.enabled_test_ids(db, category_id)
    tests = db.scalars(select(TestDef).where(TestDef.active.is_(True)).order_by(TestDef.sort)).all()
    return [t for t in tests if t.id in on], [t for t in tests if t.id not in on]


def _eval_form(request, db, athlete, ev=None, form=None, error=None):
    main, others = _tests_for(db, athlete.category_id)
    existing = {}
    if ev:
        for r in ev.results:
            existing[r.test_id] = {"attempts": {a.number: a.value for a in r.attempts}, "notes": r.notes}
    return render(request, "evaluation_form.html", athlete=athlete, ev=ev, main=main, others=others,
                  existing=existing, form=form, error=error, category=athlete.category)


@router.get("/avaliacoes/nova")
def evaluation_new(request: Request, athlete_id: str = "", db: Session = Depends(get_db)):
    if not athlete_id:
        return render(request, "evaluation_pick.html",
                      athletes=db.scalars(select(Athlete).order_by(Athlete.name)).all())
    a = db.get(Athlete, int(athlete_id))
    if not a:
        return RedirectResponse(msg_url("/avaliacoes/nova", err="Atleta não encontrado."), 303)
    return _eval_form(request, db, a)


async def _save_from_form(db, ev_athlete, ev, form):
    on = services.parse_date(form.get("date"))
    if on is None or on > date.today():
        raise services.ValidationError("Indique uma data de avaliação válida (não pode ser futura).")
    if ev is None:
        if db.scalar(select(Evaluation.id).where(Evaluation.athlete_id == ev_athlete.id, Evaluation.date == on)):
            raise services.ValidationError(
                "Já existe uma avaliação deste atleta nessa data. Abra-a a partir do histórico para a editar.")
        ev = Evaluation(athlete=ev_athlete, athlete_id=ev_athlete.id, category_id=ev_athlete.category_id)
        db.add(ev)
    else:
        clash = db.scalar(select(Evaluation.id).where(Evaluation.athlete_id == ev_athlete.id,
                                                      Evaluation.date == on, Evaluation.id != ev.id))
        if clash:
            raise services.ValidationError("Já existe outra avaliação deste atleta nessa data.")
    ev.date = on
    ev.evaluator = (form.get("evaluator") or "").strip()
    ev.notes = (form.get("notes") or "").strip()
    db.flush()
    n_saved = 0
    for t in db.scalars(select(TestDef)).all():
        vals = [services.parse_number(form.get(f"t{t.id}_a{n}")) for n in range(1, max(t.n_attempts, 1) + 1)]
        res = services.save_result(db, ev, t, vals, form.get(f"t{t.id}_obs", ""))
        n_saved += res is not None
    if n_saved == 0 and not ev.results:
        raise services.ValidationError("Introduza pelo menos um resultado.")
    return ev


@router.post("/avaliacoes/nova")
async def evaluation_create(request: Request, db: Session = Depends(get_db)):
    form = dict((await request.form()).items())
    a = db.get(Athlete, int(form["athlete_id"]))
    try:
        ev = await _save_from_form(db, a, None, form)
    except services.ValidationError as e:
        db.rollback()
        return _eval_form(request, db, a, form=form, error=str(e))
    db.commit()
    return RedirectResponse(msg_url(f"/atletas/{a.id}", "Avaliação guardada."), 303)


@router.get("/avaliacoes/{eval_id}")
def evaluation_edit(eval_id: int, request: Request, db: Session = Depends(get_db)):
    ev = db.get(Evaluation, eval_id)
    if not ev:
        return RedirectResponse(msg_url("/avaliacoes", err="Avaliação não encontrada."), 303)
    return _eval_form(request, db, ev.athlete, ev=ev)


@router.post("/avaliacoes/{eval_id}")
async def evaluation_update(eval_id: int, request: Request, db: Session = Depends(get_db)):
    ev = db.get(Evaluation, eval_id)
    form = dict((await request.form()).items())
    try:
        await _save_from_form(db, ev.athlete, ev, form)
    except services.ValidationError as e:
        db.rollback()
        return _eval_form(request, db, ev.athlete, ev=db.get(Evaluation, eval_id), form=form, error=str(e))
    db.commit()
    return RedirectResponse(msg_url(f"/atletas/{ev.athlete_id}", "Avaliação atualizada."), 303)


@router.post("/avaliacoes/{eval_id}/apagar")
def evaluation_delete(eval_id: int, db: Session = Depends(get_db)):
    ev = db.get(Evaluation, eval_id)
    aid = ev.athlete_id
    db.delete(ev)
    db.commit()
    return RedirectResponse(msg_url(f"/atletas/{aid}", "Avaliação eliminada."), 303)


# ---------------------------------------------------------------- registo em lote (uma equipa, um teste)

@router.get("/lote")
def batch_form(request: Request, team_id: str = "", category_id: str = "", test_id: str = "",
               on: str = "", db: Session = Depends(get_db)):
    tests = db.scalars(select(TestDef).where(TestDef.active.is_(True)).order_by(TestDef.sort)).all()
    ctx = dict(teams=db.scalars(select(Team).order_by(Team.club, Team.name)).all(),
               categories=db.scalars(select(Category).order_by(Category.sort)).all(), tests=tests,
               team_id=team_id, category_id=category_id, test_id=test_id, on=on or date.today().isoformat(),
               rows=None, test=None)
    if test_id and (team_id or category_id):
        test = db.get(TestDef, int(test_id))
        q = select(Athlete).order_by(Athlete.name)
        q = q.where(Athlete.team_id == int(team_id)) if team_id else q.where(Athlete.category_id == int(category_id))
        athletes = db.scalars(q).all()
        try:
            d = services.parse_date(on)
        except services.ValidationError:
            d = None
        rows = []
        for a in athletes:
            existing = {}
            notes = ""
            if d:
                ev = db.scalar(select(Evaluation).where(Evaluation.athlete_id == a.id, Evaluation.date == d))
                if ev:
                    r = next((r for r in ev.results if r.test_id == test.id), None)
                    if r:
                        existing = {x.number: x.value for x in r.attempts}
                        notes = r.notes
            rows.append(dict(a=a, existing=existing, notes=notes))
        ctx.update(rows=rows, test=test)
    return render(request, "batch.html", **ctx)


@router.post("/lote")
async def batch_save(request: Request, db: Session = Depends(get_db)):
    form = dict((await request.form()).items())
    test = db.get(TestDef, int(form["test_id"]))
    back = f"/lote?test_id={test.id}&team_id={form.get('team_id', '')}&category_id={form.get('category_id', '')}&on={form.get('on', '')}"
    try:
        d = services.parse_date(form.get("on"))
        if d is None or d > date.today():
            raise services.ValidationError("Indique uma data válida (não pode ser futura).")
        n = 0
        for key in [k for k in form if k.startswith("aid_")]:
            aid = int(key[4:])
            vals = [services.parse_number(form.get(f"a{aid}_{i}")) for i in range(1, max(test.n_attempts, 1) + 1)]
            obs = form.get(f"o{aid}", "")
            ath = db.get(Athlete, aid)
            ev = db.scalar(select(Evaluation).where(Evaluation.athlete_id == aid, Evaluation.date == d))
            if ev is None and not any(v is not None for v in vals):
                continue
            ev = services.get_or_create_evaluation(db, ath, d, form.get("evaluator", "").strip())
            n += services.save_result(db, ev, test, vals, obs) is not None
    except services.ValidationError as e:
        db.rollback()
        return RedirectResponse(msg_url(back, err=str(e)), 303)
    db.commit()
    return RedirectResponse(msg_url(back, f"Resultados guardados ({n} atletas com resultado)."), 303)
