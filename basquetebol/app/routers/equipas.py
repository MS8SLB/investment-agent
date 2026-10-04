from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from urllib.parse import quote

from ..database import get_db
from ..models import Atleta, Equipa, Escalao
from ..web import listas, render

router = APIRouter(prefix="/equipas")


def _volta(msg=None, erro=None):
    q = f"?msg={quote(msg)}" if msg else f"?erro={quote(erro)}"
    return RedirectResponse("/equipas" + q, status_code=303)


@router.get("")
def lista(request: Request, db: Session = Depends(get_db)):
    contagens = dict(db.execute(select(Atleta.equipa_id, func.count()).group_by(Atleta.equipa_id)).all())
    ctx = listas(db)
    ctx["equipas"] = list(db.scalars(select(Equipa).join(Escalao).order_by(Escalao.ordem, Equipa.nome)))
    return render(request, "equipas.html", sec="equipas", contagens=contagens, **ctx)


@router.post("")
def criar(nome: str = Form(...), clube: str = Form(""), escalao_id: int = Form(...),
          db: Session = Depends(get_db)):
    if not nome.strip():
        return _volta(erro="O nome da equipa é obrigatório.")
    db.add(Equipa(nome=nome.strip(), clube=clube.strip(), escalao_id=escalao_id))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return _volta(erro="Essa equipa já existe.")
    return _volta(msg="Equipa criada.")


@router.post("/{eid}/editar")
def editar(eid: int, nome: str = Form(...), clube: str = Form(""), escalao_id: int = Form(...),
           db: Session = Depends(get_db)):
    e = db.get(Equipa, eid)
    if not e or not nome.strip():
        return _volta(erro="Equipa inválida.")
    e.nome, e.clube, e.escalao_id = nome.strip(), clube.strip(), escalao_id
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return _volta(erro="Já existe uma equipa igual.")
    return _volta(msg="Equipa atualizada.")


@router.post("/{eid}/apagar")
def apagar(eid: int, db: Session = Depends(get_db)):
    e = db.get(Equipa, eid)
    if e:
        db.delete(e)  # atletas ficam sem equipa (ON DELETE SET NULL)
        db.commit()
    return _volta(msg="Equipa apagada (os atletas mantêm-se, sem equipa).")
