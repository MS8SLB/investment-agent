from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Escalao, Teste
from ..services.avaliacao import parse_valor
from ..web import render

router = APIRouter(prefix="/testes")


@router.get("")
def catalogo(request: Request, db: Session = Depends(get_db)):
    testes = list(db.scalars(select(Teste).order_by(Teste.ordem, Teste.id)))
    escaloes = list(db.scalars(select(Escalao).order_by(Escalao.ordem)))
    return render(request, "testes.html", sec="testes", testes=testes, escaloes=escaloes)


@router.get("/novo")
def novo_form(request: Request):
    return render(request, "teste_form.html", sec="testes", t=None)


def _campos(nome, referencia, objetivo, descricao, protocolo, material, unidade, unidade_nome, direcao,
            n_tentativas, valor_min, valor_max, casas, resultado_descricao, a_confirmar):
    if not nome.strip() or not unidade.strip():
        raise ValueError("Nome e unidade são obrigatórios.")
    if direcao not in ("menor", "maior"):
        raise ValueError("Indique se o melhor resultado é o menor ou o maior valor.")
    if not 1 <= n_tentativas <= 20:
        raise ValueError("O n.º de tentativas tem de estar entre 1 e 20.")
    return dict(nome=nome.strip(), referencia=referencia.strip(), objetivo=objetivo, descricao=descricao,
                protocolo=protocolo, material=material, unidade=unidade.strip(),
                unidade_nome=unidade_nome.strip() or unidade.strip(), direcao=direcao,
                n_tentativas=n_tentativas, valor_min=parse_valor(valor_min), valor_max=parse_valor(valor_max),
                casas=max(0, min(4, casas)), resultado_descricao=resultado_descricao, a_confirmar=a_confirmar)


@router.post("/novo")
def criar(nome: str = Form(""), referencia: str = Form(""), objetivo: str = Form(""), descricao: str = Form(""),
          protocolo: str = Form(""), material: str = Form(""), unidade: str = Form(""),
          unidade_nome: str = Form(""), direcao: str = Form(""), n_tentativas: int = Form(1),
          valor_min: str = Form(""), valor_max: str = Form(""), casas: int = Form(2),
          resultado_descricao: str = Form(""), a_confirmar: str = Form(""), db: Session = Depends(get_db)):
    try:
        c = _campos(nome, referencia, objetivo, descricao, protocolo, material, unidade, unidade_nome, direcao,
                    n_tentativas, valor_min, valor_max, casas, resultado_descricao, a_confirmar)
    except ValueError as e:
        return RedirectResponse(f"/testes/novo?erro={quote(str(e))}", status_code=303)
    base = "personalizado"
    codigo, i = base, 1
    slug = "".join(ch if ch.isalnum() else "-" for ch in c["nome"].lower()).strip("-")[:40] or base
    codigo = f"p-{slug}"
    while db.scalar(select(Teste).where(Teste.codigo == codigo)):
        i += 1
        codigo = f"p-{slug}-{i}"
    t = Teste(codigo=codigo, personalizado=True, ordem=100 + i, **c)
    db.add(t)
    db.commit()
    return RedirectResponse(f"/testes/{t.id}?msg={quote('Teste criado. Associe-o aos escalões em Configuração.')}", status_code=303)


@router.get("/{tid}")
def detalhe(tid: int, request: Request, db: Session = Depends(get_db)):
    t = db.get(Teste, tid)
    if not t:
        return RedirectResponse("/testes?erro=Teste+não+encontrado", status_code=303)
    escaloes = list(db.scalars(select(Escalao).order_by(Escalao.ordem)))
    return render(request, "teste_detalhe.html", sec="testes", t=t, escaloes=escaloes)


@router.get("/{tid}/editar")
def editar_form(tid: int, request: Request, db: Session = Depends(get_db)):
    return render(request, "teste_form.html", sec="testes", t=db.get(Teste, tid))


@router.post("/{tid}/editar")
def editar(tid: int, nome: str = Form(""), referencia: str = Form(""), objetivo: str = Form(""),
           descricao: str = Form(""), protocolo: str = Form(""), material: str = Form(""),
           unidade: str = Form(""), unidade_nome: str = Form(""), direcao: str = Form(""),
           n_tentativas: int = Form(1), valor_min: str = Form(""), valor_max: str = Form(""),
           casas: int = Form(2), resultado_descricao: str = Form(""), a_confirmar: str = Form(""),
           db: Session = Depends(get_db)):
    t = db.get(Teste, tid)
    try:
        c = _campos(nome, referencia, objetivo, descricao, protocolo, material, unidade, unidade_nome, direcao,
                    n_tentativas, valor_min, valor_max, casas, resultado_descricao, a_confirmar)
    except ValueError as e:
        return RedirectResponse(f"/testes/{tid}/editar?erro={quote(str(e))}", status_code=303)
    for k, v in c.items():
        setattr(t, k, v)
    db.commit()
    return RedirectResponse(f"/testes/{tid}?msg={quote('Teste atualizado.')}", status_code=303)
