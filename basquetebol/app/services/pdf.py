"""Relatórios PDF (individual e coletivo) com ReportLab + Matplotlib."""
from __future__ import annotations

import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)
from sqlalchemy.orm import Session  # noqa: E402

from .. import config  # noqa: E402
from ..models import Atleta, Teste  # noqa: E402
from . import estatistica as est  # noqa: E402
from .avaliacao import fmt  # noqa: E402
from .referencias import referencia_para  # noqa: E402

NAVY, ORANGE, LIGHT = colors.HexColor("#0B1F3A"), colors.HexColor("#E8600A"), colors.HexColor("#FDEBD8")
GREY = colors.HexColor("#F2F2F2")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], textColor=NAVY, fontSize=18, alignment=0, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=12.5, spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9, leading=12)
SM = ParagraphStyle("sm", parent=P, fontSize=7.8, textColor=colors.HexColor("#5A5A5A"), leading=10)


def _d(d: date | None) -> str:
    return d.strftime("%d/%m/%Y") if d else "—"


def _periodo(ini, fim) -> str:
    if ini and fim:
        return f"{_d(ini)} a {_d(fim)}"
    if ini:
        return f"desde {_d(ini)}"
    if fim:
        return f"até {_d(fim)}"
    return "todo o período"


def _grafico(series: list[tuple[str, list[tuple[date, float]]]], teste: Teste, titulo: str = "") -> Image | None:
    series = [(n, p) for n, p in series if p]
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=150)
    cores = ["#0B1F3A", "#E8600A", "#2a7ab0", "#7a8a2b"]
    for i, (nome, pts) in enumerate(series):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=cores[i % 4], label=nome, linewidth=2)
        if len(series) == 1:
            for x, y in pts:
                ax.annotate(fmt(y, teste), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_ylabel(f"{teste.unidade} ({'menor' if teste.menor_melhor else 'maior'} = melhor)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.25)
    ax.margins(y=0.2)
    fig.autofmt_xdate()
    if titulo:
        ax.set_title(titulo, fontsize=8, color="#0B1F3A", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=17 * cm, height=17 * cm * 2.6 / 7.2)


def _tabela(dados, larguras=None, destaque_col=None) -> Table:
    t = Table(dados, colWidths=larguras, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#E3E6EB")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if destaque_col is not None:
        st.append(("FONTNAME", (destaque_col, 1), (destaque_col, -1), "Helvetica-Bold"))
    t.setStyle(TableStyle(st))
    return t


def _caixa_treinador() -> Table:
    linhas = [[Paragraph("<b>Observações do treinador</b>", P)]] + [[""]] * 5
    t = Table(linhas, colWidths=[17 * cm], rowHeights=[0.6 * cm] + [0.75 * cm] * 5)
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .8, NAVY), ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                           ("LINEBELOW", (0, 1), (-1, -2), .3, colors.HexColor("#B8BEC9"))]))
    return t


def _rodape(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ORANGE)
    canvas.rect(0, A4[1] - 0.5 * cm, A4[0], 0.5 * cm, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#5A5A5A"))
    canvas.drawString(2 * cm, 1 * cm, f"{config.TITULO} — {config.SUBTITULO} · Responsável metodológico: {config.AUTOR}")
    canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Página {doc.page}")
    canvas.restoreState()


def _cabecalho(titulo: str, linhas: list[str]) -> list:
    out = [Paragraph(config.TITULO.upper(), ParagraphStyle("k", parent=SM, textColor=ORANGE, fontName="Helvetica-Bold")),
           Paragraph(titulo, H1)]
    out += [Paragraph(l, P) for l in linhas]
    out.append(Spacer(1, 6))
    return out


def _nota_metodologica() -> Paragraph:
    return Paragraph("Resultados absolutos. Não foram aplicadas normas, percentis nem classificações, exceto quando "
                     "indicado com tabela de referência validada. Tempos: menor = melhor; pontos/cestos: maior = melhor.", SM)


def _build(story) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.8 * cm, title=config.TITULO, author=config.AUTOR)
    doc.build(story, onFirstPage=_rodape, onLaterPages=_rodape)
    return buf.getvalue()


def relatorio_individual(db: Session, a: Atleta, testes: list[Teste], ini=None, fim=None) -> bytes:
    story = _cabecalho("Relatório individual de avaliação", [
        f"<b>Atleta:</b> {a.nome} &nbsp; <b>Escalão:</b> {a.escalao.nome} &nbsp; <b>Sexo:</b> {'Feminino' if a.sexo == 'F' else 'Masculino'}",
        f"<b>Data de nascimento:</b> {_d(a.data_nascimento)} &nbsp; <b>Equipa:</b> {a.equipa.nome if a.equipa else '—'} "
        f"&nbsp; <b>Clube:</b> {a.clube or '—'}",
        f"<b>Período:</b> {_periodo(ini, fim)} &nbsp; <b>Emitido em:</b> {_d(date.today())}"])
    story.append(_nota_metodologica())
    tem = False
    for t in testes:
        rows = est.resultados(db, t.id, [a.id], ini, fim)
        if not rows:
            continue
        tem = True
        avs = {}
        from ..models import Avaliacao, ResultadoTeste
        from sqlalchemy import select
        for _, d, v, aid in rows:
            r = db.scalar(select(ResultadoTeste).where(ResultadoTeste.avaliacao_id == aid, ResultadoTeste.teste_id == t.id))
            avs[aid] = (d, r)
        cab = ["Data"] + [f"Tent. {n}" for n in range(1, t.n_tentativas + 1)] + ["Melhor", "Variação vs. anterior"]
        dados, ant = [cab], None
        for aid, (d, r) in avs.items():
            tv = {x.numero: x.valor for x in r.tentativas}
            var = "—"
            if ant is not None:
                v = est.variacao(t, ant, r.melhor_valor)
                seta = "melhorou" if v["melhorou"] else ("piorou" if v["melhorou"] is False else "igual")
                var = f"{v['delta']:+.{t.casas}f} {t.unidade} ({seta})".replace(".", ",")
            dados.append([_d(d)] + [fmt(tv.get(n), t) for n in range(1, t.n_tentativas + 1)] + [fmt(r.melhor_valor, t), var])
            ant = r.melhor_valor
        vals = [r[2] for r in rows]
        ref = referencia_para(db, t, a, vals[-1])
        bloco = [Paragraph(f"{t.nome}", H2),
                 Paragraph(f"{t.referencia} · Unidade: {t.unidade_nome} · {'menor' if t.menor_melhor else 'maior'} valor = melhor", SM),
                 Spacer(1, 3), _tabela(dados, destaque_col=len(cab) - 2)]
        g = _grafico([("Melhor resultado", [(r[1], r[2]) for r in rows])], t)
        if g:
            bloco += [Spacer(1, 4), g]
        sumario = (f"Melhor de sempre no período: <b>{fmt(min(vals) if t.menor_melhor else max(vals), t)} {t.unidade}</b>. "
                   + (f"Referência ({ref['tabela']}): <b>{ref['classe']}</b>." if ref["disponivel"]
                      else "Sem tabela de referência disponível — resultado absoluto."))
        bloco.append(Paragraph(sumario, P))
        story.append(KeepTogether(bloco[:4]))
        story += bloco[4:]
    if not tem:
        story.append(Paragraph("Sem resultados registados para os testes e período selecionados.", P))
    story += [Spacer(1, 14), _caixa_treinador()]
    return _build(story)


def relatorio_coletivo(db: Session, atletas: list[Atleta], testes: list[Teste], descricao: str, ini=None, fim=None) -> bytes:
    ids = [a.id for a in atletas]
    story = _cabecalho("Relatório coletivo de avaliação", [
        f"<b>Grupo:</b> {descricao} &nbsp; <b>Atletas:</b> {len(atletas)}",
        f"<b>Período:</b> {_periodo(ini, fim)} &nbsp; <b>Emitido em:</b> {_d(date.today())}"])
    story.append(_nota_metodologica())
    tem = False
    for t in testes:
        rows = est.resultados(db, t.id, ids, ini, fim)
        if not rows:
            continue
        tem = True
        medias = est.medias_por_data(rows)
        ult = est.ultimo_por_atleta(rows)
        r = est.resumo([v for _, _, v in ult.values()])
        primeiro = {}
        for a, d, v, _ in rows:
            primeiro.setdefault(a.id, (d, v))
        story.append(KeepTogether([
            Paragraph(t.nome, H2),
            Paragraph(f"{t.referencia} · Unidade: {t.unidade_nome} · {'menor' if t.menor_melhor else 'maior'} valor = melhor", SM),
            Spacer(1, 3),
            _tabela([["Atletas", "Média", "Mediana", "Desvio-padrão", "Mínimo", "Máximo"],
                     [str(r["n"]), fmt(r["media"], t), fmt(r["mediana"], t), fmt(r["dp"], t), fmt(r["min"], t), fmt(r["max"], t)]]),
            Paragraph("Estatística calculada sobre o último resultado de cada atleta no período.", SM)]))
        g = _grafico([("Média coletiva", [(m["data"], m["media"]) for m in medias])], t, "Média coletiva por data")
        if g:
            story += [Spacer(1, 4), g]
        story.append(_tabela([["Data", "Atletas (n)", "Média", "Mín.", "Máx."]] +
                             [[_d(m["data"]), str(m["n"]), fmt(m["media"], t), fmt(m["min"], t), fmt(m["max"], t)] for m in medias]))
        story.append(Spacer(1, 5))
        linhas = [["Atleta", "Escalão", "1.º resultado", "Último resultado", "Variação"]]
        for aid, (at, d, v) in sorted(ult.items(), key=lambda kv: kv[1][0].nome):
            d0, v0 = primeiro[aid]
            if d0 != d:
                var = est.variacao(t, v0, v)
                seta = "▲" if var["melhorou"] else ("▼" if var["melhorou"] is False else "=")
                txt = f"{var['delta']:+.{t.casas}f} {t.unidade} {seta}".replace(".", ",").replace("▲", "(melhorou)").replace("▼", "(piorou)").replace("=", "(igual)")
            else:
                txt = "—"
            linhas.append([at.nome, at.escalao.nome, fmt(v0, t), fmt(v, t), txt])
        story.append(_tabela(linhas, destaque_col=3))
        story.append(Spacer(1, 4))
    if not tem:
        story.append(Paragraph("Sem resultados registados para os testes e período selecionados.", P))
    story += [Spacer(1, 14), _caixa_treinador()]
    return _build(story)
