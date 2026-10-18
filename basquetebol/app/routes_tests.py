import re

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from . import services
from .db import get_db
from .models import Category, CategoryTest, ReferenceRow, ReferenceTable, TestDef
from .web import msg_url, render

router = APIRouter()


@router.get("/testes")
def test_list(request: Request, db: Session = Depends(get_db)):
    tests = db.scalars(select(TestDef).order_by(TestDef.sort, TestDef.id)).all()
    return render(request, "tests.html", tests=tests)


@router.get("/testes/novo")
def test_new(request: Request):
    return render(request, "test_form.html", t=None, error=None)


def _fill(t: TestDef | None, f: dict) -> TestDef:
    name = f["name"].strip()
    unit = f["unit"].strip()
    if not name or not unit:
        raise services.ValidationError("Nome e unidade de medida são obrigatórios.")
    if f["direction"] not in ("lower", "higher"):
        raise services.ValidationError("Indique o sentido do resultado (menor ou maior = melhor).")
    n = int(f["n_attempts"] or 1)
    if not 1 <= n <= 10:
        raise services.ValidationError("O número de tentativas deve estar entre 1 e 10.")
    mv = services.parse_number(f["max_value"])
    if t is None:
        base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "teste"
        t = TestDef(code=base, unit=unit, direction=f["direction"], name=name, is_enb=False)
    t.name, t.unit, t.direction, t.n_attempts, t.max_value = name, unit, f["direction"], n, mv
    t.decimals = int(f["decimals"] or 0)
    for k in ("source", "capacity", "objective", "description", "procedure", "materials", "result_note",
              "attempts_rule"):
        setattr(t, k, f[k].strip())
    t.attempts_confirmed = f.get("attempts_confirmed") == "on"
    return t


FIELDS = ["name", "source", "capacity", "objective", "description", "procedure", "materials", "result_note",
          "attempts_rule", "unit", "direction", "n_attempts", "decimals", "max_value"]


@router.post("/testes/novo")
async def test_create(request: Request, db: Session = Depends(get_db)):
    form = dict((await request.form()).items())
    f = {k: form.get(k, "") for k in FIELDS} | {"attempts_confirmed": form.get("attempts_confirmed", "on")}
    try:
        t = _fill(None, f)
    except (services.ValidationError, ValueError) as e:
        return render(request, "test_form.html", t=None, error=str(e), form=f)
    existing = set(db.scalars(select(TestDef.code)))
    base, i = t.code, 2
    while t.code in existing:
        t.code, i = f"{base}_{i}", i + 1
    t.sort = (db.scalar(select(TestDef.sort).order_by(TestDef.sort.desc())) or 0) + 1
    db.add(t)
    db.commit()
    return RedirectResponse(msg_url(f"/testes/{t.id}", "Teste criado. Ative-o nos escalões pretendidos."), 303)


@router.get("/testes/{test_id}")
def test_detail(test_id: int, request: Request, db: Session = Depends(get_db)):
    t = db.get(TestDef, test_id)
    if not t:
        return RedirectResponse(msg_url("/testes", err="Teste não encontrado."), 303)
    cats = db.scalars(select(Category).order_by(Category.sort)).all()
    enabled = services.enabled_test_ids  # noqa
    cat_status = [dict(c=c, enabled=t.id in services.enabled_test_ids(db, c.id),
                       ref=services.has_reference(db, t.id, c.id)) for c in cats]
    return render(request, "test_detail.html", t=t, cat_status=cat_status)


@router.get("/testes/{test_id}/editar")
def test_edit(test_id: int, request: Request, db: Session = Depends(get_db)):
    t = db.get(TestDef, test_id)
    return render(request, "test_form.html", t=t, error=None)


@router.post("/testes/{test_id}/editar")
async def test_update(test_id: int, request: Request, db: Session = Depends(get_db)):
    t = db.get(TestDef, test_id)
    form = dict((await request.form()).items())
    f = {k: form.get(k, "") for k in FIELDS} | {"attempts_confirmed": form.get("attempts_confirmed", "")}
    try:
        _fill(t, f)
    except (services.ValidationError, ValueError) as e:
        return render(request, "test_form.html", t=t, error=str(e), form=f)
    db.commit()
    return RedirectResponse(msg_url(f"/testes/{t.id}", "Teste atualizado."), 303)


@router.post("/testes/{test_id}/ativo")
def test_toggle(test_id: int, db: Session = Depends(get_db)):
    t = db.get(TestDef, test_id)
    t.active = not t.active
    db.commit()
    return RedirectResponse(msg_url("/testes", "Estado do teste atualizado."), 303)


# ---------------------------------------------------------------- testes por escalão

@router.get("/escaloes")
def category_config(request: Request, db: Session = Depends(get_db)):
    cats = db.scalars(select(Category).order_by(Category.sort)).all()
    tests = db.scalars(select(TestDef).where(TestDef.active.is_(True)).order_by(TestDef.sort)).all()
    on = {(c.id, tid) for c in cats for tid in services.enabled_test_ids(db, c.id)}
    ref = {(c.id, t.id): services.has_reference(db, t.id, c.id) for c in cats for t in tests}
    return render(request, "categories.html", cats=cats, tests=tests, on=on, ref=ref)


@router.post("/escaloes")
async def category_save(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    chosen = set(form.getlist("sel"))  # "catid:testid"
    db.execute(delete(CategoryTest))
    for item in chosen:
        c, t = item.split(":")
        db.add(CategoryTest(category_id=int(c), test_id=int(t)))
    db.commit()
    return RedirectResponse(msg_url("/escaloes", "Seleção de testes por escalão guardada."), 303)


# ---------------------------------------------------------------- tabelas de referência

@router.get("/referencias")
def reference_list(request: Request, db: Session = Depends(get_db)):
    return render(request, "references.html",
                  tables=db.scalars(select(ReferenceTable).order_by(ReferenceTable.id)).all(),
                  tests=db.scalars(select(TestDef).order_by(TestDef.sort)).all(),
                  cats=db.scalars(select(Category).order_by(Category.sort)).all())


@router.post("/referencias")
def reference_create(name: str = Form(""), source: str = Form(""), test_id: int = Form(...),
                     category_id: str = Form(""), sex: str = Form(""), rows: str = Form(""),
                     db: Session = Depends(get_db)):
    """Linhas no formato «rótulo; mínimo; máximo» (um por linha; mínimo/máximo podem ficar vazios)."""
    parsed = []
    try:
        for ln in rows.strip().splitlines():
            if not ln.strip():
                continue
            parts = [p.strip() for p in ln.split(";")]
            if len(parts) != 3 or not parts[0]:
                raise services.ValidationError(f"Linha inválida: «{ln}». Use: rótulo; mínimo; máximo")
            parsed.append((parts[0], services.parse_number(parts[1]), services.parse_number(parts[2])))
    except services.ValidationError as e:
        return RedirectResponse(msg_url("/referencias", err=str(e)), 303)
    if not name.strip() or not source.strip() or not parsed:
        return RedirectResponse(msg_url("/referencias", err="Nome, fonte e pelo menos uma linha são obrigatórios."), 303)
    t = ReferenceTable(name=name.strip(), source=source.strip(), test_id=test_id,
                       category_id=int(category_id) if category_id else None, sex=sex or None)
    for i, (lab, lo, hi) in enumerate(parsed):
        t.rows.append(ReferenceRow(label=lab, min_value=lo, max_value=hi, sort=i))
    db.add(t)
    db.commit()
    return RedirectResponse(msg_url("/referencias", "Tabela de referência guardada."), 303)


@router.post("/referencias/{table_id}/ativo")
def reference_toggle(table_id: int, db: Session = Depends(get_db)):
    t = db.get(ReferenceTable, table_id)
    t.active = not t.active
    db.commit()
    return RedirectResponse(msg_url("/referencias", "Estado da tabela atualizado."), 303)


@router.post("/referencias/{table_id}/apagar")
def reference_delete(table_id: int, db: Session = Depends(get_db)):
    t = db.get(ReferenceTable, table_id)
    if t:
        db.delete(t)
        db.commit()
    return RedirectResponse(msg_url("/referencias", "Tabela eliminada."), 303)
