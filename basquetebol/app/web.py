"""Utilitários partilhados pelas páginas (templates, filtros, mensagens)."""
from __future__ import annotations

from datetime import date

from fastapi import Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import config
from .models import Equipa, Escalao
from .services.avaliacao import fmt

templates = Jinja2Templates(directory=str(config.BASE_DIR / "app" / "templates"))

MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
         "outubro", "novembro", "dezembro"]


def data_pt(d: date | None) -> str:
    return d.strftime("%d/%m/%Y") if d else "—"


def idade(nasc: date, ref: date | None = None) -> int:
    ref = ref or date.today()
    return ref.year - nasc.year - ((ref.month, ref.day) < (nasc.month, nasc.day))


def sexo_pt(s: str) -> str:
    return {"F": "Feminino", "M": "Masculino"}.get(s, s)


def num(v, casas: int = 2) -> str:
    return "—" if v is None else f"{v:.{casas}f}".replace(".", ",")


templates.env.filters.update(data_pt=data_pt, sexo_pt=sexo_pt, num=num)
templates.env.globals.update(config=config, idade=idade, fmt=fmt, hoje=date.today)


def render(request: Request, nome: str, **ctx):
    ctx.setdefault("msg", request.query_params.get("msg"))
    ctx.setdefault("erro", request.query_params.get("erro"))
    return templates.TemplateResponse(request, nome, ctx)


def listas(db: Session) -> dict:
    return {"escaloes": list(db.scalars(select(Escalao).order_by(Escalao.ordem))),
            "equipas": list(db.scalars(select(Equipa).order_by(Equipa.nome)))}


def to_int(v) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def to_date(v) -> date | None:
    try:
        return date.fromisoformat(v) if v else None
    except ValueError:
        return None
