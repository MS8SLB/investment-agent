from datetime import date

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import scoring, services
from .db import get_db
from .models import Athlete, Category, Team, TestDef
from .web import render

router = APIRouter()


def indicators(db, tests):
    """Indicadores coletivos por teste (nunca agregados entre testes)."""
    out = []
    for t in tests:
        rows = services.result_rows(db, test_id=t.id)
        if not rows:
            continue
        series = services.series_by_date(rows, t.direction)
        last_d, last = series[-1]
        prev = series[-2][1] if len(series) > 1 else None
        delta, imp = scoring.change(prev["mean"] if prev else None, last["mean"], t.direction)
        out.append(dict(test=t, last_date=last_d, stats=last, delta=delta, improved=imp,
                        n_dates=len(series), best=last["min"] if t.direction == "lower" else last["max"]))
    return out


def athlete_progress(rows, direction):
    """Evolução primeiro→último resultado por atleta (só atletas com ≥2 datas)."""
    first, last = services.first_per_athlete(rows), services.latest_per_athlete(rows)
    out = []
    for aid, f in first.items():
        l = last[aid]
        if l.evaluation.date == f.evaluation.date:
            continue
        d, imp = scoring.change(f.best_value, l.best_value, direction)
        out.append(dict(athlete=f.evaluation.athlete, first=f, last=l, delta=d, improved=imp))
    out.sort(key=lambda x: (x["delta"] if direction == "lower" else -x["delta"]))
    return out


@router.get("/")
def dashboard(request: Request, test_id: str = "", db: Session = Depends(get_db)):
    counts = services.dashboard_counts(db)
    tests = db.scalars(select(TestDef).where(TestDef.active.is_(True)).order_by(TestDef.sort)).all()
    inds = indicators(db, tests)
    with_data = [i["test"] for i in inds]
    sel = next((t for t in with_data if str(t.id) == test_id), with_data[0] if with_data else None)
    evo = prog = None
    if sel:
        rows = services.result_rows(db, test_id=sel.id)
        series = services.series_by_date(rows, sel.direction)
        evo = dict(labels=[d.strftime("%d/%m/%Y") for d, _ in series],
                   mean=[round(s["mean"], 3) for _, s in series], n=[s["n"] for _, s in series])
        prog = athlete_progress(rows, sel.direction)
    return render(request, "dashboard.html", counts=counts, inds=inds, with_data=with_data, sel=sel, evo=evo,
                  prog=prog)


@router.get("/analise")
def analysis(request: Request, test_id: str = "", category_id: str = "", team_id: str = "", sex: str = "",
             date_from: str = "", date_to: str = "", date_a: str = "", date_b: str = "", dist_date: str = "",
             cross: str = "", db: Session = Depends(get_db)):
    tests = db.scalars(select(TestDef).where(TestDef.active.is_(True)).order_by(TestDef.sort)).all()
    ctx = dict(tests=tests, categories=db.scalars(select(Category).order_by(Category.sort)).all(),
               teams=db.scalars(select(Team).order_by(Team.club, Team.name)).all(), test_id=test_id,
               category_id=category_id, team_id=team_id, sex=sex, date_from=date_from, date_to=date_to,
               date_a=date_a, date_b=date_b, dist_date=dist_date, cross=cross, test=None, error=None)
    if not test_id:
        return render(request, "analysis.html", **ctx)
    test = db.get(TestDef, int(test_id))
    ctx["test"] = test
    try:
        df, dt = services.parse_date(date_from), services.parse_date(date_to)
    except services.ValidationError as e:
        ctx["error"] = str(e)
        return render(request, "analysis.html", **ctx)
    filt = dict(test_id=test.id, category_id=int(category_id) if category_id else None,
                team_id=int(team_id) if team_id else None, sex=sex or None, date_from=df, date_to=dt)
    rows = services.result_rows(db, **filt)
    ctx["n_rows"] = len(rows)
    if not rows:
        return render(request, "analysis.html", **ctx)

    series = services.series_by_date(rows, test.direction)
    dates = [d for d, _ in series]
    ctx["series"] = [dict(date=d, **s) for d, s in series]
    ctx["evo"] = dict(labels=[d.strftime("%d/%m/%Y") for d in dates], mean=[round(s["mean"], 3) for _, s in series],
                      best=[s["min"] if test.direction == "lower" else s["max"] for _, s in series],
                      n=[s["n"] for _, s in series])
    ctx["dates"] = dates

    # comparação entre duas datas
    da = scoring_date(date_a) or (dates[0] if dates else None)
    dbb = scoring_date(date_b) or (dates[-1] if dates else None)
    ctx["date_a_v"], ctx["date_b_v"] = da, dbb
    by_a = {r.evaluation.athlete_id: r for r in rows if r.evaluation.date == da}
    by_b = {r.evaluation.athlete_id: r for r in rows if r.evaluation.date == dbb}
    comp = []
    for aid in by_a.keys() & by_b.keys():
        ra, rb = by_a[aid], by_b[aid]
        d, imp = scoring.change(ra.best_value, rb.best_value, test.direction)
        comp.append(dict(athlete=ra.evaluation.athlete, a=ra.best_value, b=rb.best_value, delta=d, improved=imp))
    comp.sort(key=lambda x: x["athlete"].name)
    ctx["comp"] = comp
    ctx["comp_chart"] = dict(labels=[c["athlete"].name for c in comp], a=[c["a"] for c in comp],
                             b=[c["b"] for c in comp])

    # distribuição (melhor resultado por atleta numa data; por defeito a mais recente)
    dd = scoring_date(dist_date) or dates[-1]
    ctx["dist_date_v"] = dd
    vals = [r.best_value for r in rows if r.evaluation.date == dd]
    bins = scoring.histogram(vals)
    dec = test.decimals
    ctx["dist"] = dict(labels=[f"{b[0]:.{dec}f}–{b[1]:.{dec}f}".replace(".", ",") if b[0] != b[1] else f"{b[0]:.{dec}f}".replace(".", ",")
                               for b in bins], counts=[b[2] for b in bins])
    ctx["dist_stats"] = scoring.describe(vals)

    # comparação entre escalões: opcional e apenas descritiva
    if cross:
        per = []
        for c in ctx["categories"]:
            rs = services.result_rows(db, test_id=test.id, category_id=c.id, team_id=filt["team_id"],
                                      sex=filt["sex"], date_from=df, date_to=dt)
            last = services.latest_per_athlete(rs)
            st = scoring.describe([r.best_value for r in last.values()])
            if st:
                per.append(dict(cat=c, **st))
        ctx["cross_rows"] = per
    return render(request, "analysis.html", **ctx)


def scoring_date(txt):
    try:
        return services.parse_date(txt)
    except services.ValidationError:
        return None
