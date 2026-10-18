"""Geração dos relatórios PDF (individual e coletivo) com ReportLab."""
import io
from datetime import date

from reportlab.graphics.charts.lineplots import LinePlot  # noqa: F401
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from . import scoring, services
from .models import Athlete, Category, Team
from .web import dpt, num, signed

NAVY = colors.HexColor("#0B1F3A")
ORANGE = colors.HexColor("#E8600A")
TINT = colors.HexColor("#FDEBD8")
GREY = colors.HexColor("#F2F2F2")
MID = colors.HexColor("#5A5A5A")
GREEN = colors.HexColor("#1B7F3B")
RED = colors.HexColor("#B3261E")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=NAVY, fontSize=18, leading=22, alignment=0, spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], textColor=MID, fontSize=9.5, leading=12)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=12.5, spaceBefore=10, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9, leading=11.5)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8, textColor=MID, leading=10)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.5, leading=10)

SYS_NAME = "Plataforma de Avaliação Quantitativa Técnica — Basquetebol de Formação"
AUTHOR = "Autor e responsável metodológico: Mário Silva"


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ORANGE)
    canvas.rect(0, A4[1] - 0.5 * cm, A4[0], 0.5 * cm, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID)
    canvas.drawString(1.8 * cm, 1.0 * cm, f"{SYS_NAME}  |  {AUTHOR}")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Página {doc.page}")
    canvas.restoreState()


def _doc(buf, title):
    return SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                             bottomMargin=1.8 * cm, title=title, author="Mário Silva")


def _header(title, subtitle):
    t = Table([[Paragraph(f"<font color='white'>{title}</font>",
                          ParagraphStyle("t", parent=H1, textColor=colors.white)),
                ]], colWidths=[17.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                           ("LINEBELOW", (0, 0), (-1, -1), 3, ORANGE)]))
    return [Paragraph("ESCOLA NACIONAL DE BASQUETEBOL · formação de treinadores", SMALL), Spacer(1, 3), t,
            Spacer(1, 4), Paragraph(subtitle, SUB), Spacer(1, 6)]


def _table(data, widths, header=True, align_right_from=1):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]),
             ("LINEBELOW", (0, -1), (-1, -1), 0.5, MID), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    if align_right_from is not None:
        style.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def _line_chart(labels, series, names, test, width=17 * cm, height=5.2 * cm):
    """series: lista de listas (None permitido). Eixo Y no sentido natural; o sentido é indicado no título."""
    d = Drawing(width, height)
    flat = [v for s in series for v in s if v is not None]
    if not flat:
        return d
    lc = HorizontalLineChart()
    lc.x, lc.y, lc.width, lc.height = 1.6 * cm, 0.9 * cm, width - 2.6 * cm, height - 1.9 * cm
    lc.data = [[v if v is not None else None for v in s] for s in series]
    lc.categoryAxis.categoryNames = labels
    lc.categoryAxis.labels.fontSize = 7
    lc.categoryAxis.labels.angle = 0 if len(labels) <= 6 else 30
    lc.categoryAxis.labels.boxAnchor = "n" if len(labels) <= 6 else "ne"
    lo, hi = min(flat), max(flat)
    pad = (hi - lo) * 0.15 or max(abs(hi) * 0.1, 1)
    lc.valueAxis.valueMin, lc.valueAxis.valueMax = max(0, lo - pad), hi + pad
    lc.valueAxis.labels.fontSize = 7
    lc.valueAxis.labelTextFormat = lambda v: num(v, 1 if test.decimals else 0)
    lc.valueAxis.visibleGrid = True
    lc.valueAxis.gridStrokeColor = colors.HexColor("#DDDDDD")
    palette = [ORANGE, NAVY, colors.HexColor("#7A8CA5")]
    for i in range(len(series)):
        lc.lines[i].strokeColor = palette[i % 3]
        lc.lines[i].strokeWidth = 2
        lc.lines[i].symbol = None
    from reportlab.graphics.widgets.markers import makeMarker
    for i in range(len(series)):
        lc.lines[i].symbol = makeMarker("FilledCircle", size=4, fillColor=palette[i % 3], strokeColor=palette[i % 3])
    d.add(lc)
    sentido = "menor = melhor" if test.direction == "lower" else "maior = melhor"
    d.add(String(1.6 * cm, height - 0.5 * cm, f"{test.name} ({test.unit}; {sentido})", fontSize=8,
                 fillColor=NAVY, fontName="Helvetica-Bold"))
    if len(series) > 1:
        x = width - 6 * cm
        for i, n in enumerate(names):
            d.add(String(x + i * 3 * cm, height - 0.5 * cm, f"— {n}", fontSize=7.5, fillColor=palette[i % 3]))
    return d


def _obs_box(extra_notes=None):
    rows = [[Paragraph("<b>Observações do treinador</b>", BODY)]]
    for n in (extra_notes or []):
        rows.append([Paragraph(n, CELL)])
    rows += [[""]] * (5 if not extra_notes else 3)
    t = Table(rows, colWidths=[17.4 * cm], rowHeights=[0.7 * cm] + [None] * len(extra_notes or []) + [0.8 * cm] * (len(rows) - 1 - len(extra_notes or [])))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, NAVY), ("BACKGROUND", (0, 0), (-1, 0), TINT),
                           ("LINEBELOW", (0, 1), (-1, -2), 0.3, colors.HexColor("#BBBBBB"))]))
    return KeepTogether([Spacer(1, 10), t])


def _var_text(delta, improved, test):
    if delta is None:
        return "—"
    word = "melhoria" if improved else ("agravamento" if improved is False else "sem alteração")
    return f"{signed(delta, test.decimals)} {test.unit} ({word})"


def _ref_note(has_ref):
    return ("Classificação segundo tabela de referência ativa." if has_ref else
            "Sem tabela de referência validada para este teste/escalão: resultados apresentados em valores absolutos.")


# ---------------------------------------------------------------- individual

def individual_report(db, athlete: Athlete, date_from, date_to, test_ids):
    buf = io.BytesIO()
    doc = _doc(buf, f"Relatório individual — {athlete.name}")
    period = (f"{dpt(date_from)} a {dpt(date_to)}" if date_from or date_to else "todo o histórico")
    if date_from and not date_to:
        period = f"desde {dpt(date_from)}"
    if date_to and not date_from:
        period = f"até {dpt(date_to)}"
    story = _header("Relatório Individual de Avaliação Técnica",
                    f"Gerado em {dpt(date.today())} · Período: {period}")
    info = [["Atleta", athlete.name, "Escalão", athlete.category.name],
            ["Nascimento", f"{dpt(athlete.birth_date)} ({scoring.age_on(athlete.birth_date, date.today())} anos)",
             "Sexo", "Masculino" if athlete.sex == "M" else "Feminino"],
            ["Clube", athlete.club or "—", "Equipa", athlete.team.name if athlete.team else "—"]]
    t = Table(info, colWidths=[2.4 * cm, 6.6 * cm, 2.2 * cm, 6.2 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("BACKGROUND", (0, 0), (0, -1), GREY),
                           ("BACKGROUND", (2, 0), (2, -1), GREY), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                           ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.3, MID)]))
    story += [t, Spacer(1, 6)]

    rows = services.result_rows(db, athlete_id=athlete.id, date_from=date_from, date_to=date_to)
    by_test = {}
    for r in rows:
        if not test_ids or r.test_id in test_ids:
            by_test.setdefault(r.test_id, []).append(r)
    notes = []
    if not by_test:
        story.append(Paragraph("Não existem resultados registados para os critérios selecionados.", BODY))
    for tid, rs in sorted(by_test.items(), key=lambda kv: kv[1][0].test.sort):
        test = rs[0].test
        _, ref_rows = services.reference_rows_for(db, tid, athlete.category_id, athlete.sex)
        head = ["Data", "Tentativas", f"Melhor ({test.unit})", "Variação vs. anterior"] + (["Classificação"] if ref_rows else [])
        data = [head]
        prev = None
        for r in rs:
            d, imp = scoring.change(prev.best_value if prev else None, r.best_value, test.direction)
            atts = "; ".join(num(a.value, test.decimals) for a in r.attempts)
            row = [dpt(r.evaluation.date), atts, num(r.best_value, test.decimals),
                   "—" if d is None else _var_text(d, imp, test)]
            if ref_rows:
                row.append(scoring.classify(r.best_value, ref_rows) or "—")
            data.append(row)
            prev = r
            if r.notes:
                notes.append(f"<b>{test.name} ({dpt(r.evaluation.date)}):</b> {r.notes}")
        first, last = rs[0].best_value, rs[-1].best_value
        total, imp = scoring.change(first, last, test.direction)
        block = [Paragraph(f"{test.numeral}. {test.name}", H2),
                 Paragraph(f"{test.capacity} · Unidade: {test.unit} · {test.better_label} · "
                           f"Protocolo: {test.attempts_rule or 'ver documento ENB'}", SMALL),
                 Spacer(1, 3), _table(data, [2.4 * cm, 4.6 * cm, 3 * cm, 4.6 * cm] + ([2.8 * cm] if ref_rows else []), align_right_from=2)]
        if len(rs) > 1:
            block += [Spacer(1, 3), _line_chart([r.evaluation.date.strftime("%d/%m/%y") for r in rs],
                                                [[r.best_value for r in rs]], ["Melhor"], test),
                      Paragraph(f"Evolução entre a primeira e a última avaliação: <b>{_var_text(total, imp, test)}</b>.", BODY)]
        else:
            block.append(Paragraph("Apenas uma avaliação no período: não é possível apresentar evolução.", SMALL))
        block.append(Paragraph(_ref_note(bool(ref_rows)), SMALL))
        story.append(KeepTogether(block))
    for ev in {r.evaluation for rs in by_test.values() for r in rs}:
        if ev.notes:
            notes.append(f"<b>Avaliação de {dpt(ev.date)}:</b> {ev.notes}")
    story.append(_obs_box(notes))
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ---------------------------------------------------------------- coletivo

def collective_report(db, category_id, team_id, sex, date_from, date_to, test_ids):
    buf = io.BytesIO()
    doc = _doc(buf, "Relatório coletivo")
    cat = db.get(Category, category_id) if category_id else None
    team = db.get(Team, team_id) if team_id else None
    group = " · ".join(x for x in [f"Escalão: {cat.name}" if cat else "Todos os escalões",
                                   f"Equipa: {team.name}" + (f" ({team.club})" if team.club else "") if team else "",
                                   {"M": "Masculino", "F": "Feminino"}.get(sex or "", "")] if x)
    period = (f"{dpt(date_from)} a {dpt(date_to)}" if date_from and date_to else
              f"desde {dpt(date_from)}" if date_from else f"até {dpt(date_to)}" if date_to else "todo o histórico")
    story = _header("Relatório Coletivo de Avaliação Técnica",
                    f"{group} · Período: {period} · Gerado em {dpt(date.today())}")
    all_rows = services.result_rows(db, category_id=category_id, team_id=team_id, sex=sex or None,
                                    date_from=date_from, date_to=date_to)
    by_test = {}
    for r in all_rows:
        if not test_ids or r.test_id in test_ids:
            by_test.setdefault(r.test_id, []).append(r)
    if not by_test:
        story.append(Paragraph("Não existem resultados registados para os critérios selecionados.", BODY))
    cross_cat = len({r.evaluation.category_id for rs in by_test.values() for r in rs}) > 1
    if cross_cat:
        story.append(Paragraph("Atenção: o grupo inclui mais do que um escalão. As estatísticas são descritivas e não "
                               "devem ser usadas para comparar escalões.", SMALL))
    for tid, rs in sorted(by_test.items(), key=lambda kv: kv[1][0].test.sort):
        test = rs[0].test
        series = services.series_by_date(rs, test.direction)
        data = [["Data", "N", "Média", "Mediana", "Mínimo", "Máximo", "Desvio-padrão"]]
        d_ = test.decimals
        for d, s in series:
            data.append([dpt(d), str(s["n"]), num(s["mean"], d_), num(s["median"], d_), num(s["min"], d_),
                         num(s["max"], d_), num(s["sd"], d_) if s["sd"] is not None else "—"])
        first, last = services.first_per_athlete(rs), services.latest_per_athlete(rs)
        ath = [["Atleta", "Primeira", "Última", "Variação"]]
        for aid in sorted(first, key=lambda i: first[i].evaluation.athlete.name):
            f, l = first[aid], last[aid]
            if f is l or f.evaluation.date == l.evaluation.date:
                ath.append([f.evaluation.athlete.name, num(f.best_value, d_), "—", "—"])
            else:
                dl, imp = scoring.change(f.best_value, l.best_value, test.direction)
                ath.append([f.evaluation.athlete.name, num(f.best_value, d_), num(l.best_value, d_), _var_text(dl, imp, test)])
        block = [Paragraph(f"{test.numeral}. {test.name}", H2),
                 Paragraph(f"Unidade: {test.unit} · {test.better_label}. Valores = melhor resultado de cada atleta.", SMALL),
                 Spacer(1, 3), _table(data, [2.6 * cm, 1.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 3.2 * cm])]
        if len(series) > 1:
            block += [Spacer(1, 3), _line_chart([d.strftime("%d/%m/%y") for d, _ in series],
                                                [[s["mean"] for _, s in series]], ["Média"], test)]
        story.append(KeepTogether(block))
        story += [Spacer(1, 3), _table(ath, [7 * cm, 3 * cm, 3 * cm, 4.4 * cm], align_right_from=1),
                  Paragraph("Sem tabela de referência validada: valores absolutos." if not any(
                      services.has_reference(db, tid, c) for c in {r.evaluation.category_id for r in rs}) else
                      "Existem tabelas de referência ativas; ver relatórios individuais para a classificação.", SMALL)]
    story.append(_obs_box())
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
