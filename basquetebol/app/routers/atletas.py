from datetime import date
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Atleta, Avaliacao, Equipa, Escalao
from ..services import estatistica as est
from ..services.referencias import referencia_para
from ..web import listas, render, to_date, to_int

router = APIRouter(prefix="/atletas")


def _validar(nome, nasc, sexo):
    if not nome.strip():
        return "O nome é obrigatório."
    d = to_date(nasc)
    if not d or d > date.today() or d.year < 1990:
        return "Data de nascimento inválida."
    if sexo not in ("F", "M"):
        return "Indique o sexo."
    return None


@router.get("")
def lista(request: Request, q: str = "", escalao_id: str = "", equipa_id: str = "", sexo: str = "",
          db: Session = Depends(get_db)):
    atletas = est.atletas_filtrados(db, to_int(escalao_id), to_int(equipa_id), sexo or None, q or None)
    n_aval = {a.id: len(a.avaliacoes) for a in atletas}
    return render(request, "atletas.html", sec="atletas", atletas=atletas, n_aval=n_aval,
                  f={"q": q, "escalao_id": to_int(escalao_id), "equipa_id": to_int(equipa_id), "sexo": sexo},
                  **listas(db))


@router.get("/novo")
def novo(request: Request, db: Session = Depends(get_db)):
    return render(request, "atleta_form.html", sec="atletas", a=None, **listas(db))


@router.post("/novo")
def criar(request: Request, nome: str = Form(""), data_nascimento: str = Form(""), sexo: str = Form(""),
          clube: str = Form(""), equipa_id: str = Form(""), escalao_id: int = Form(...),
          notas: str = Form(""), db: Session = Depends(get_db)):
    erro = _validar(nome, data_nascimento, sexo)
    if erro:
        return RedirectResponse(f"/atletas/novo?erro={quote(erro)}", status_code=303)
    eq = db.get(Equipa, to_int(equipa_id)) if to_int(equipa_id) else None
    a = Atleta(nome=nome.strip(), data_nascimento=to_date(data_nascimento), sexo=sexo,
               clube=clube.strip() or (eq.clube if eq else ""), equipa_id=eq.id if eq else None,
               escalao_id=escalao_id, notas=notas)
    db.add(a)
    db.commit()
    return RedirectResponse(f"/atletas/{a.id}?msg={quote('Atleta registado.')}", status_code=303)


@router.get("/{aid}")
def ficha(aid: int, request: Request, db: Session = Depends(get_db)):
    a = db.get(Atleta, aid)
    if not a:
        return RedirectResponse("/atletas?erro=Atleta+não+encontrado", status_code=303)
    avs = list(db.scalars(select(Avaliacao).where(Avaliacao.atleta_id == aid)
                          .order_by(Avaliacao.data.desc(), Avaliacao.id.desc())))
    # resumo por teste (do escalão do atleta + qualquer teste já realizado)
    ids = {r.teste_id: r.teste for av in avs for r in av.resultados}
    for t in a.escalao.testes:
        ids.setdefault(t.id, t)
    resumo = []
    for t in sorted(ids.values(), key=lambda t: t.ordem):
        s = est.serie_atleta(db, aid, t.id)
        item = {"teste": t, "n": len(s), "ultimo": s[-1]["valor"] if s else None,
                "melhor": (min if t.menor_melhor else max)([p["valor"] for p in s]) if s else None,
                "var": est.variacao(t, s[0]["valor"], s[-1]["valor"]) if len(s) > 1 else None,
                "ref": referencia_para(db, t, a, s[-1]["valor"] if s else None)}
        resumo.append(item)
    testes_js = {r["teste"].id: {"nome": r["teste"].nome, "u": r["teste"].unidade, "m": r["teste"].menor_melhor,
                                 "c": r["teste"].casas} for r in resumo}
    return render(request, "atleta_ficha.html", sec="atletas", a=a, avs=avs, resumo=resumo, testes_js=testes_js)


@router.get("/{aid}/editar")
def editar_form(aid: int, request: Request, db: Session = Depends(get_db)):
    a = db.get(Atleta, aid)
    return render(request, "atleta_form.html", sec="atletas", a=a, **listas(db))


@router.post("/{aid}/editar")
def editar(aid: int, nome: str = Form(""), data_nascimento: str = Form(""), sexo: str = Form(""),
           clube: str = Form(""), equipa_id: str = Form(""), escalao_id: int = Form(...),
           notas: str = Form(""), db: Session = Depends(get_db)):
    a = db.get(Atleta, aid)
    erro = _validar(nome, data_nascimento, sexo)
    if not a or erro:
        return RedirectResponse(f"/atletas/{aid}/editar?erro={quote(erro or 'Atleta inexistente')}", status_code=303)
    a.nome, a.data_nascimento, a.sexo = nome.strip(), to_date(data_nascimento), sexo
    a.clube, a.equipa_id, a.escalao_id, a.notas = clube.strip(), to_int(equipa_id), escalao_id, notas
    db.commit()
    return RedirectResponse(f"/atletas/{aid}?msg={quote('Atleta atualizado.')}", status_code=303)


@router.post("/{aid}/apagar")
def apagar(aid: int, db: Session = Depends(get_db)):
    a = db.get(Atleta, aid)
    if a:
        db.delete(a)
        db.commit()
    return RedirectResponse(f"/atletas?msg={quote('Atleta e respetivas avaliações apagados.')}", status_code=303)


@router.get("/{aid}/serie/{tid}")
def serie_json(aid: int, tid: int, db: Session = Depends(get_db)):
    return [{"x": p["data"].isoformat(), "y": p["valor"]} for p in est.serie_atleta(db, aid, tid)]
