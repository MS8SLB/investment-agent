from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Teste
from ..services import estatistica as est
from ..web import listas, render, to_date, to_int

router = APIRouter(prefix="/analise")


@router.get("")
def pagina(request: Request, db: Session = Depends(get_db)):
    testes = list(db.scalars(select(Teste).order_by(Teste.ordem, Teste.id)))
    return render(request, "analise.html", sec="analise", testes=testes, atletas=est.atletas_filtrados(db),
                  **listas(db))


@router.get("/dados")
def dados(teste_id: int, escalao_id: str = "", equipa_id: str = "", sexo: str = "", ini: str = "", fim: str = "",
          atleta_id: str = "", d1: str = "", d2: str = "", db: Session = Depends(get_db)):
    t = db.get(Teste, teste_id)
    return est.dados_analise(db, t, to_int(escalao_id), to_int(equipa_id), sexo or None, to_date(ini), to_date(fim),
                             to_int(atleta_id), to_date(d1), to_date(d2))
