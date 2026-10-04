from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Atleta, Equipa, Escalao, Teste
from ..services import estatistica as est
from ..services import pdf
from ..web import listas, render, to_date, to_int

router = APIRouter(prefix="/relatorios")


def _testes(db: Session, ids: list[int]) -> list[Teste]:
    stmt = select(Teste).order_by(Teste.ordem, Teste.id)
    if ids:
        stmt = stmt.where(Teste.id.in_(ids))
    return list(db.scalars(stmt))


def _pdf(conteudo: bytes, nome: str) -> Response:
    return Response(conteudo, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{nome}"'})


@router.get("")
def pagina(request: Request, db: Session = Depends(get_db)):
    testes = list(db.scalars(select(Teste).order_by(Teste.ordem, Teste.id)))
    return render(request, "relatorios.html", sec="relatorios", testes=testes,
                  atletas=est.atletas_filtrados(db), **listas(db))


@router.get("/individual.pdf")
def individual(request: Request, atleta_id: int, ini: str = "", fim: str = "", db: Session = Depends(get_db)):
    a = db.get(Atleta, atleta_id)
    if not a:
        return RedirectResponse("/relatorios?erro=Atleta+não+encontrado", status_code=303)
    ids = [int(x) for x in request.query_params.getlist("teste_id") if x.isdigit()]
    out = pdf.relatorio_individual(db, a, _testes(db, ids), to_date(ini), to_date(fim))
    return _pdf(out, f"relatorio_{a.nome.replace(' ', '_')}.pdf")


@router.get("/coletivo.pdf")
def coletivo(request: Request, escalao_id: str = "", equipa_id: str = "", sexo: str = "", ini: str = "",
             fim: str = "", db: Session = Depends(get_db)):
    atletas = est.atletas_filtrados(db, to_int(escalao_id), to_int(equipa_id), sexo or None)
    if not atletas:
        return RedirectResponse("/relatorios?erro=Nenhum+atleta+nos+filtros+escolhidos", status_code=303)
    partes = []
    if to_int(equipa_id):
        e = db.get(Equipa, to_int(equipa_id))
        partes.append(f"Equipa {e.nome}")
    if to_int(escalao_id):
        partes.append(db.get(Escalao, to_int(escalao_id)).nome)
    if sexo:
        partes.append("Feminino" if sexo == "F" else "Masculino")
    ids = [int(x) for x in request.query_params.getlist("teste_id") if x.isdigit()]
    out = pdf.relatorio_coletivo(db, atletas, _testes(db, ids), " · ".join(partes) or "Todos os atletas",
                                 to_date(ini), to_date(fim))
    return _pdf(out, "relatorio_coletivo.pdf")
