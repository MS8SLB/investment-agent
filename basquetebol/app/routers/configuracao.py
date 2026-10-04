from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Request, UploadFile, File
from fastapi.responses import RedirectResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Escalao, EscalaoTeste, ReferenciaTabela, Teste
from ..services.referencias import importar_csv
from ..web import render, to_int

router = APIRouter(prefix="/configuracao")


@router.get("")
def pagina(request: Request, db: Session = Depends(get_db)):
    escaloes = list(db.scalars(select(Escalao).order_by(Escalao.ordem)))
    testes = list(db.scalars(select(Teste).order_by(Teste.ordem, Teste.id)))
    tabelas = list(db.scalars(select(ReferenciaTabela).order_by(ReferenciaTabela.id)))
    # por teste × escalão: há referência aplicável?
    cobertura = {}
    for t in testes:
        for e in escaloes:
            cobertura[(t.id, e.id)] = [x.nome for x in tabelas
                                       if x.teste_id == t.id and x.escalao_id in (None, e.id)]
    return render(request, "configuracao.html", sec="configuracao", escaloes=escaloes, testes=testes,
                  tabelas=tabelas, cobertura=cobertura)


@router.post("/escaloes")
async def guardar_escaloes(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    marcados = set(form.getlist("sel"))   # "escalaoid:testeid"
    db.execute(delete(EscalaoTeste))
    for m in marcados:
        e, t = m.split(":")
        db.add(EscalaoTeste(escalao_id=int(e), teste_id=int(t)))
    db.commit()
    return RedirectResponse(f"/configuracao?msg={quote('Testes por escalão guardados.')}", status_code=303)


@router.post("/referencias")
async def importar(teste_id: int = Form(...), escalao_id: str = Form(""), sexo: str = Form(""),
                   nome: str = Form(""), fonte: str = Form(""), validada: str = Form(""),
                   ficheiro: UploadFile = File(...), db: Session = Depends(get_db)):
    def volta(erro):
        return RedirectResponse(f"/configuracao?erro={quote(erro)}#referencias", status_code=303)
    if not nome.strip() or not fonte.strip():
        return volta("Indique o nome da tabela e a fonte (publicação/estudo) de onde provém.")
    if not validada:
        return volta("Confirme que a tabela foi validada metodologicamente.")
    teste = db.get(Teste, teste_id)
    try:
        texto = (await ficheiro.read()).decode("utf-8-sig")
        importar_csv(db, teste, nome, fonte, to_int(escalao_id), sexo or None, texto)
        db.commit()
    except (ValueError, UnicodeDecodeError) as e:
        db.rollback()
        return volta(str(e))
    return RedirectResponse(f"/configuracao?msg={quote('Tabela de referência importada.')}#referencias", status_code=303)


@router.post("/referencias/{rid}/apagar")
def apagar_ref(rid: int, db: Session = Depends(get_db)):
    r = db.get(ReferenciaTabela, rid)
    if r:
        db.delete(r)
        db.commit()
    return RedirectResponse(f"/configuracao?msg={quote('Tabela removida.')}#referencias", status_code=303)
