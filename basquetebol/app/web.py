"""Configuração partilhada das páginas (Jinja2, filtros em português de Portugal)."""
from datetime import date
from pathlib import Path
from urllib.parse import quote

from fastapi import Request
from fastapi.templating import Jinja2Templates

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def num(v, decimals=2):
    if v is None:
        return "—"
    s = f"{v:.{decimals}f}".replace(".", ",")
    return s


def dpt(d):
    return d.strftime("%d/%m/%Y") if d else "—"


def unit_label(test):
    return "" if test is None else test.unit


def signed(v, decimals=2):
    if v is None:
        return "—"
    return ("+" if v > 0 else "") + num(v, decimals)


templates.env.filters.update(num=num, dpt=dpt, signed=signed)
templates.env.globals["today"] = date.today


def render(request: Request, name: str, **ctx):
    ctx.setdefault("msg", request.query_params.get("msg"))
    ctx.setdefault("err", request.query_params.get("err"))
    return templates.TemplateResponse(request, name, ctx)


def msg_url(path: str, msg: str | None = None, err: str | None = None):
    sep = "&" if "?" in path else "?"
    if msg:
        return f"{path}{sep}msg={quote(msg)}"
    if err:
        return f"{path}{sep}err={quote(err)}"
    return path
