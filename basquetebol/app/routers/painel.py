from fastapi import APIRouter, Depends, Request
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Atleta, Avaliacao, Escalao, Teste
from ..services import estatistica as est
from ..web import render

router = APIRouter()


@router.get("/")
def painel(request: Request, db: Session = Depends(get_db)):
    n_atletas = db.scalar(select(func.count(Atleta.id)))
    n_avaliados = db.scalar(select(func.count(distinct(Avaliacao.atleta_id))))
    n_aval = db.scalar(select(func.count(Avaliacao.id)))
    escaloes = list(db.scalars(select(Escalao).order_by(Escalao.ordem)))
    total = dict(db.execute(select(Atleta.escalao_id, func.count()).group_by(Atleta.escalao_id)).all())
    aval = dict(db.execute(select(Atleta.escalao_id, func.count(distinct(Avaliacao.atleta_id)))
                           .join(Avaliacao, Avaliacao.atleta_id == Atleta.id).group_by(Atleta.escalao_id)).all())
    dist = [{"nome": e.nome, "atletas": total.get(e.id, 0), "avaliados": aval.get(e.id, 0)} for e in escaloes]
    # indicadores coletivos por teste (cada um na sua unidade; nunca uma média global)
    indicadores = []
    for t in db.scalars(select(Teste).order_by(Teste.ordem, Teste.id)):
        rows = est.resultados(db, t.id)
        if not rows:
            indicadores.append({"t": t, "n": 0})
            continue
        ult = est.ultimo_por_atleta(rows)
        vs = [v for _, _, v in ult.values()]
        primeiro = {}
        for a, d, v, _ in rows:
            primeiro.setdefault(a.id, v)
        evo = [(est.variacao(t, primeiro[aid], v)["melhorou"]) for aid, (_, _, v) in ult.items()
               if rows and sum(1 for r in rows if r[0].id == aid) > 1]
        resumo = est.resumo(vs)
        indicadores.append({"t": t, "n": len(vs), "media": resumo["media"],
                            "melhor": min(vs) if t.menor_melhor else max(vs),
                            "com_evolucao": len(evo), "melhoraram": sum(1 for x in evo if x)})
    ultimas = list(db.scalars(select(Avaliacao).order_by(Avaliacao.data.desc(), Avaliacao.id.desc()).limit(6)))
    testes = [i["t"] for i in indicadores if i["n"]]
    return render(request, "painel.html", sec="painel", n_atletas=n_atletas, n_avaliados=n_avaliados, n_aval=n_aval,
                  dist=dist, indicadores=indicadores, ultimas=ultimas, testes_com_dados=testes)
