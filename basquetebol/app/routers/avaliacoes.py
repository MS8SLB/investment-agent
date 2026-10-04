from datetime import date
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Atleta, Avaliacao, Teste
from ..services import estatistica as est
from ..services.avaliacao import guardar_resultado, nova_avaliacao, parse_valor
from ..services.referencias import referencia_para
from ..web import listas, render, to_date, to_int

router = APIRouter(prefix="/avaliacoes")


def _ler_tentativas(form, tid: int, teste: Teste):
    """Lê t{tid}_{n} e t{tid}_{n}_obs do formulário -> ([(valor, obs)], erros, brutos)."""
    tent, erros, brutos = [], [], {}
    for n in range(1, teste.n_tentativas + 1):
        raw = form.get(f"t{tid}_{n}", "")
        obs = form.get(f"t{tid}_{n}_obs", "")
        brutos[n] = (raw, obs)
        try:
            v = parse_valor(raw)
        except ValueError:
            erros.append(f"{teste.nome}: «{raw}» não é um número válido (tentativa {n}).")
            continue
        if v is not None:
            tent.append((v, obs.strip()))
    return tent, erros, brutos


def _testes_do_atleta(db: Session, a: Atleta, todos: bool, extra_ids=()):
    if todos:
        return list(db.scalars(select(Teste).order_by(Teste.ordem, Teste.id)))
    ts = {t.id: t for t in a.escalao.testes}
    for tid in extra_ids:
        t = db.get(Teste, tid)
        if t:
            ts[t.id] = t
    return sorted(ts.values(), key=lambda t: (t.ordem, t.id))


def _form_ctx(db, a, testes, av=None, brutos=None, data=None, obs=None, todos=False):
    vals = {}
    if av:
        for r in av.resultados:
            vals[r.teste_id] = {"obs": r.observacoes, "t": {x.numero: (f"{x.valor:.{r.teste.casas}f}".replace(".", ","), x.observacoes) for x in r.tentativas}}
    if brutos:
        vals = {tid: {"obs": b.get("obs", ""), "t": {n: v for n, v in b["t"].items()}} for tid, b in brutos.items()}
    return dict(a=a, testes=testes, av=av, vals=vals, todos=todos,
                data=data or (av.data if av else date.today()), obs=obs if obs is not None else (av.observacoes if av else ""))


@router.get("/nova")
def nova(request: Request, atleta_id: str = "", todos: str = "", db: Session = Depends(get_db)):
    aid = to_int(atleta_id)
    a = db.get(Atleta, aid) if aid else None
    if not a:
        return render(request, "avaliacao_escolher.html", sec="nova", atletas=est.atletas_filtrados(db), **listas(db))
    testes = _testes_do_atleta(db, a, bool(todos))
    return render(request, "avaliacao_form.html", sec="nova", **_form_ctx(db, a, testes, todos=bool(todos)))


async def _processar(request: Request, db: Session, av: Avaliacao | None):
    form = await request.form()
    a = db.get(Atleta, to_int(form.get("atleta_id")))
    d = to_date(form.get("data"))
    todos = bool(form.get("todos"))
    testes = _testes_do_atleta(db, a, todos, [to_int(x) for x in form.getlist("tid")]) if a else []
    erros, lidos, brutos = [], {}, {}
    if not a:
        erros.append("Atleta inválido.")
    if not d:
        erros.append("Indique a data da avaliação.")
    elif d > date.today():
        erros.append("A data da avaliação não pode ser futura.")
    for t in testes:
        tent, errs, br = _ler_tentativas(form, t.id, t)
        erros += errs
        for v, _ in tent:
            from ..services.avaliacao import validar_valor
            e = validar_valor(t, v)
            if e:
                erros.append(f"{t.nome}: {e}")
        brutos[t.id] = {"obs": form.get(f"t{t.id}_obs", ""), "t": br}
        if tent:
            lidos[t.id] = (t, tent, form.get(f"t{t.id}_obs", "").strip())
    if not lidos and not erros:
        erros.append("Introduza pelo menos um resultado.")
    if erros:
        ctx = _form_ctx(db, a, testes, av, brutos, d, form.get("observacoes", ""), todos)
        return render(request, "avaliacao_form.html", sec="nova", erro=" ".join(erros), **ctx)
    if av is None:
        av = nova_avaliacao(db, a, d, form.get("observacoes", "").strip())
    else:
        av.data, av.observacoes = d, form.get("observacoes", "").strip()
        for r in list(av.resultados):          # um teste deixado em branco é removido
            if r.teste_id in {t.id for t in testes} and r.teste_id not in lidos:
                av.resultados.remove(r)
    for t, tent, obs in lidos.values():
        guardar_resultado(db, av, t, tent, obs)
    db.commit()
    return RedirectResponse(f"/avaliacoes/{av.id}?msg={quote('Avaliação guardada.')}", status_code=303)


@router.post("/nova")
async def criar(request: Request, db: Session = Depends(get_db)):
    return await _processar(request, db, None)


@router.get("")
def historico(request: Request, atleta_id: str = "", escalao_id: str = "", equipa_id: str = "",
              ini: str = "", fim: str = "", db: Session = Depends(get_db)):
    stmt = select(Avaliacao).join(Atleta).order_by(Avaliacao.data.desc(), Avaliacao.id.desc())
    if to_int(atleta_id):
        stmt = stmt.where(Avaliacao.atleta_id == to_int(atleta_id))
    if to_int(escalao_id):
        stmt = stmt.where(Avaliacao.escalao_id == to_int(escalao_id))
    if to_int(equipa_id):
        stmt = stmt.where(Atleta.equipa_id == to_int(equipa_id))
    if to_date(ini):
        stmt = stmt.where(Avaliacao.data >= to_date(ini))
    if to_date(fim):
        stmt = stmt.where(Avaliacao.data <= to_date(fim))
    avs = list(db.scalars(stmt.limit(300)))
    return render(request, "avaliacoes.html", sec="avaliacoes", avs=avs, atletas=est.atletas_filtrados(db),
                  f={"atleta_id": to_int(atleta_id), "escalao_id": to_int(escalao_id),
                     "equipa_id": to_int(equipa_id), "ini": ini, "fim": fim}, **listas(db))


@router.get("/coletiva")
def coletiva(request: Request, teste_id: str = "", equipa_id: str = "", escalao_id: str = "", data: str = "",
             db: Session = Depends(get_db)):
    testes = list(db.scalars(select(Teste).order_by(Teste.ordem, Teste.id)))
    t = db.get(Teste, to_int(teste_id)) if to_int(teste_id) else None
    atletas = []
    d = to_date(data) or date.today()
    if t and (to_int(equipa_id) or to_int(escalao_id)):
        atletas = est.atletas_filtrados(db, to_int(escalao_id), to_int(equipa_id))
    existentes = {}
    if t:
        for a in atletas:
            av = db.scalar(select(Avaliacao).where(Avaliacao.atleta_id == a.id, Avaliacao.data == d))
            r = next((r for r in av.resultados if r.teste_id == t.id), None) if av else None
            if r:
                existentes[a.id] = {x.numero: f"{x.valor:.{t.casas}f}".replace(".", ",") for x in r.tentativas}
    return render(request, "avaliacao_coletiva.html", sec="coletiva", testes=testes, t=t, atletas=atletas,
                  d=d, existentes=existentes, f={"equipa_id": to_int(equipa_id), "escalao_id": to_int(escalao_id)},
                  **listas(db))


@router.post("/coletiva")
async def coletiva_guardar(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    t = db.get(Teste, to_int(form.get("teste_id")))
    d = to_date(form.get("data"))
    voltar = (f"/avaliacoes/coletiva?teste_id={t.id if t else ''}&equipa_id={form.get('equipa_id', '')}"
              f"&escalao_id={form.get('escalao_id', '')}&data={form.get('data', '')}")
    if not t or not d or d > date.today():
        return RedirectResponse(voltar + "&erro=" + quote("Teste ou data inválidos."), status_code=303)
    guardar, erros = [], []
    for aid in form.getlist("aid"):
        a = db.get(Atleta, to_int(aid))
        if not a:
            continue
        tent, errs, _ = _ler_tentativas(form, int(aid), t)
        for v, _o in tent:
            from ..services.avaliacao import validar_valor
            e = validar_valor(t, v)
            if e:
                errs.append(f"{a.nome}: {e}")
        erros += [f"{a.nome}: {e}" if not e.startswith(t.nome) else e for e in errs]
        if tent:
            guardar.append((a, tent))
    if erros:
        return RedirectResponse(voltar + "&erro=" + quote(" ".join(erros)), status_code=303)
    for a, tent in guardar:
        av = db.scalar(select(Avaliacao).where(Avaliacao.atleta_id == a.id, Avaliacao.data == d))
        if av is None:
            av = nova_avaliacao(db, a, d)
        guardar_resultado(db, av, t, tent)
    db.commit()
    return RedirectResponse(voltar + "&msg=" + quote(f"{len(guardar)} resultado(s) guardado(s)."), status_code=303)


@router.get("/{avid}")
def detalhe(avid: int, request: Request, db: Session = Depends(get_db)):
    av = db.get(Avaliacao, avid)
    if not av:
        return RedirectResponse("/avaliacoes?erro=Avaliação+não+encontrada", status_code=303)
    linhas = []
    for r in sorted(av.resultados, key=lambda r: r.teste.ordem):
        linhas.append({"r": r, "ref": referencia_para(db, r.teste, av.atleta, r.melhor_valor)})
    return render(request, "avaliacao_detalhe.html", sec="avaliacoes", av=av, linhas=linhas)


@router.get("/{avid}/editar")
def editar_form(avid: int, request: Request, db: Session = Depends(get_db)):
    av = db.get(Avaliacao, avid)
    ids = [r.teste_id for r in av.resultados]
    testes = _testes_do_atleta(db, av.atleta, False, ids)
    return render(request, "avaliacao_form.html", sec="nova", **_form_ctx(db, av.atleta, testes, av))


@router.post("/{avid}/editar")
async def editar(avid: int, request: Request, db: Session = Depends(get_db)):
    return await _processar(request, db, db.get(Avaliacao, avid))


@router.post("/{avid}/apagar")
def apagar(avid: int, db: Session = Depends(get_db)):
    av = db.get(Avaliacao, avid)
    aid = av.atleta_id if av else None
    if av:
        db.delete(av)
        db.commit()
    return RedirectResponse(f"/atletas/{aid}?msg={quote('Avaliação apagada.')}" if aid else "/avaliacoes", status_code=303)
