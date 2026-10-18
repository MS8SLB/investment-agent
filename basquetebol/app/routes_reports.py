from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import reports, services
from .db import get_db
from .models import Athlete, Category, Team, TestDef
from .web import msg_url, render

router = APIRouter()


@router.get("/relatorios")
def report_form(request: Request, db: Session = Depends(get_db)):
    return render(request, "reports.html",
                  athletes=db.scalars(select(Athlete).order_by(Athlete.name)).all(),
                  categories=db.scalars(select(Category).order_by(Category.sort)).all(),
                  teams=db.scalars(select(Team).order_by(Team.club, Team.name)).all(),
                  tests=db.scalars(select(TestDef).where(TestDef.active.is_(True)).order_by(TestDef.sort)).all())


def _pdf(content: bytes, name: str):
    return Response(content, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{name}"'})


def _parse(date_from, date_to, tests):
    df, dt = services.parse_date(date_from), services.parse_date(date_to)
    if df and dt and df > dt:
        raise services.ValidationError("A data inicial é posterior à data final.")
    return df, dt, {int(t) for t in tests if t}


@router.get("/relatorios/atleta/{athlete_id}.pdf")
def report_athlete(athlete_id: int, date_from: str = "", date_to: str = "", test: list[str] = (),
                   db: Session = Depends(get_db)):
    a = db.get(Athlete, athlete_id)
    if not a:
        return RedirectResponse(msg_url("/relatorios", err="Atleta não encontrado."), 303)
    try:
        df, dt, ids = _parse(date_from, date_to, test)
    except services.ValidationError as e:
        return RedirectResponse(msg_url("/relatorios", err=str(e)), 303)
    safe = "".join(c if c.isalnum() else "_" for c in a.name)
    return _pdf(reports.individual_report(db, a, df, dt, ids), f"relatorio_{safe}.pdf")


@router.get("/relatorios/coletivo.pdf")
def report_collective(category_id: str = "", team_id: str = "", sex: str = "", date_from: str = "",
                      date_to: str = "", test: list[str] = (), db: Session = Depends(get_db)):
    if not (category_id or team_id):
        return RedirectResponse(msg_url("/relatorios", err="Escolha um escalão ou uma equipa."), 303)
    try:
        df, dt, ids = _parse(date_from, date_to, test)
    except services.ValidationError as e:
        return RedirectResponse(msg_url("/relatorios", err=str(e)), 303)
    return _pdf(reports.collective_report(db, int(category_id) if category_id else None,
                                          int(team_id) if team_id else None, sex or None, df, dt, ids),
                "relatorio_coletivo.pdf")
