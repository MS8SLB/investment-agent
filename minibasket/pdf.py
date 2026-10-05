"""Exportação em PDF dos relatórios (reportlab; gráficos desenhados em vetor, sem dependências externas).

Cada função recebe a estrutura de dados do relatório (módulo `reports`) e devolve os bytes do PDF.
A verificação de permissões é feita em `access.export_*`, que é o único ponto de entrada das vistas.
"""

from __future__ import annotations

import io
import math
import os
import re
import unicodedata
from datetime import date
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import calc
from . import competencies as comp

AUTHOR = "Plataforma de Avaliação do Minibasquete"
BLUE, ORANGE = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834")
INK, MUTED = colors.HexColor("#1b1b1a"), colors.HexColor("#52514e")
GRID, SOFT = colors.HexColor("#c9c9c4"), colors.HexColor("#efefec")
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

_ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14, textColor=INK)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9, leading=11.5)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=BLUE,
                    spaceBefore=12, spaceAfter=4)
BIG = ParagraphStyle("big", parent=BODY, fontName="Helvetica-Bold", fontSize=15, leading=19)


# ── texto ───────────────────────────────────────────────────────────────────
def _safe(text) -> str:
    """Texto compatível com as fontes standard do PDF (WinAnsi); o que não couber vira «?»."""
    return str(text).encode("cp1252", "replace").decode("cp1252")


def _x(text) -> str:
    return escape(_safe(text if text is not None else ""))


def P(text, style=BODY) -> Paragraph:
    """Parágrafo com texto de utilizador escapado (nunca interpreta marcação)."""
    return Paragraph(_x(text).replace("\n", "<br/>"), style)


def PB(label: str, value, style=BODY) -> Paragraph:
    return Paragraph(f"<b>{_x(label)}</b> {_x(value)}", style)


def fmt_date(iso: str | None) -> str:
    return date.fromisoformat(iso).strftime("%d/%m/%Y") if iso else "—"


def slug(text: str) -> str:
    s = unicodedata.normalize("NFD", text or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "relatorio"


def filename(kind: str, label: str, iso_date: str | None = None) -> str:
    return f"{kind}-{slug(label)}-{iso_date or date.today().isoformat()}.pdf"


# ── gráficos vetoriais ──────────────────────────────────────────────────────
def radar_drawing(series: list[dict], scale_max: int = 5, w: float = 430, h: float = 290) -> Drawing:
    """Roda das Competências. series: [{"name","scores","dash": bool,"fill": bool,"color"}]; a 1.ª é a principal."""
    d = Drawing(w, h)
    legend_h = 20 if len(series) > 1 else 0
    cx, cy = w / 2, (h + legend_h) / 2
    R = min(w / 2 - 78, (h - legend_h) / 2 - 24)
    n = len(comp.COMPETENCIES)

    def pt(i: int, v: float) -> tuple[float, float]:
        a = math.radians(90 - i * 360 / n)
        return cx + R * v / scale_max * math.cos(a), cy + R * v / scale_max * math.sin(a)

    for ring in range(1, scale_max + 1):
        pts = [c for i in range(n) for c in pt(i, ring)]
        d.add(Polygon(pts, strokeColor=GRID, strokeWidth=0.5, fillColor=None))
    for i, c in enumerate(comp.COMPETENCIES):
        x, y = pt(i, scale_max)
        d.add(Line(cx, cy, x, y, strokeColor=GRID, strokeWidth=0.5))
        lx, ly = pt(i, scale_max * 1.0)
        a = math.radians(90 - i * 360 / n)
        ca = math.cos(a)
        anchor = "start" if ca > 0.3 else "end" if ca < -0.3 else "middle"
        d.add(String(lx + 8 * ca, ly + 8 * math.sin(a) - 3, _safe(c.short), fontSize=8.5, fontName="Helvetica",
                     textAnchor=anchor, fillColor=INK))
    for v in range(0, scale_max + 1):
        x, y = pt(0, v)
        d.add(String(x + 3, y - 2.5, str(v), fontSize=6.5, fillColor=MUTED, fontName="Helvetica"))

    for idx, s in sorted(enumerate(series), key=lambda t: -t[0]):          # referência por baixo
        color = s.get("color") or (BLUE if idx == 0 else ORANGE)
        vals = [s["scores"].get(k) for k in comp.KEYS]
        pts = [pt(i, v) if v is not None else None for i, v in enumerate(vals)]
        dash = (4, 3) if s.get("dash") else None
        if all(p is not None for p in pts):
            fill = colors.Color(color.red, color.green, color.blue, alpha=0.18) if s.get("fill") else None
            d.add(Polygon([c for p in pts for c in p], strokeColor=color, strokeWidth=1.8, strokeDashArray=dash,
                          fillColor=fill))
        else:                                                         # competências por avaliar: só segmentos reais
            for i in range(n):
                a, b = pts[i], pts[(i + 1) % n]
                if a and b:
                    d.add(Line(a[0], a[1], b[0], b[1], strokeColor=color, strokeWidth=1.8, strokeDashArray=dash))
        for p in pts:
            if p:
                d.add(Circle(p[0], p[1], 3, fillColor=color, strokeColor=color))
    if legend_h:
        total = sum(len(_safe(s["name"])) * 4.6 + 38 for s in series)
        x = (w - total) / 2
        for idx, s in enumerate(series):
            color = s.get("color") or (BLUE if idx == 0 else ORANGE)
            d.add(Line(x, 8, x + 16, 8, strokeColor=color, strokeWidth=1.8, strokeDashArray=(4, 3) if s.get("dash") else None))
            d.add(Circle(x + 8, 8, 3, fillColor=color, strokeColor=color))
            d.add(String(x + 22, 5, _safe(s["name"]), fontSize=8, fontName="Helvetica", fillColor=INK))
            x += len(_safe(s["name"])) * 4.6 + 38
    return d


def line_drawing(points: list[dict], scale_max: int = 5, w: float = CONTENT_W, h: float = 190,
                 decimals: int = 2) -> Drawing:
    """Evolução no tempo. points: [{"date": ISO, "value": float|None}] (datas com escala real)."""
    d = Drawing(w, h)
    L, Rr, B, T = 30, 22, 30, 16
    pw, ph = w - L - Rr, h - B - T
    d.add(Rect(L, B, pw, ph, strokeColor=None, fillColor=None))
    for v in range(0, scale_max + 1):
        y = B + ph * v / scale_max
        d.add(Line(L, y, L + pw, y, strokeColor=GRID, strokeWidth=0.5))
        d.add(String(L - 6, y - 3, str(v), fontSize=7.5, textAnchor="end", fillColor=MUTED, fontName="Helvetica"))
    ords = [date.fromisoformat(p["date"]).toordinal() for p in points]
    lo, hi = (min(ords), max(ords)) if ords else (0, 1)
    xs = [L + pw / 2 if hi == lo else L + 12 + (pw - 24) * (o - lo) / (hi - lo) for o in ords]
    ys = [None if p["value"] is None else B + ph * p["value"] / scale_max for p in points]
    for i in range(len(points) - 1):
        if ys[i] is not None and ys[i + 1] is not None:
            d.add(Line(xs[i], ys[i], xs[i + 1], ys[i + 1], strokeColor=BLUE, strokeWidth=2))
    rated = [i for i, y in enumerate(ys) if y is not None]
    label_at = {rated[0], rated[-1]} if rated else set()
    step = max(1, math.ceil(len(points) / 8))
    for i, p in enumerate(points):
        if i % step == 0 or i == len(points) - 1:
            iso = p["date"]
            d.add(String(xs[i], B - 14, f"{iso[8:10]}/{iso[5:7]}/{iso[2:4]}", fontSize=7.5, textAnchor="middle",
                         fillColor=MUTED, fontName="Helvetica"))
        if ys[i] is not None:
            d.add(Circle(xs[i], ys[i], 3.2, fillColor=BLUE, strokeColor=BLUE))
            if i in label_at:
                d.add(String(xs[i], ys[i] + 7, calc.fmt(p["value"], decimals), fontSize=8.5, textAnchor="middle",
                             fillColor=INK, fontName="Helvetica-Bold"))
    return d


def bars_drawing(stats: list[dict], scale_max: int = 5, w: float = CONTENT_W, row_h: float = 17) -> Drawing:
    """Média da equipa por competência (barras horizontais, pela ordem da roda)."""
    n = len(stats)
    h = n * row_h + 24
    d = Drawing(w, h)
    L, Rr = 120, 40
    pw = w - L - Rr
    for v in range(0, scale_max + 1):
        x = L + pw * v / scale_max
        d.add(Line(x, 16, x, h - 4, strokeColor=GRID, strokeWidth=0.5))
        d.add(String(x, 4, str(v), fontSize=7.5, textAnchor="middle", fillColor=MUTED, fontName="Helvetica"))
    for i, s in enumerate(stats):
        y = h - 8 - (i + 1) * row_h + 4
        d.add(String(L - 8, y + 2, _safe(comp.short_of(s["key"])), fontSize=8.5, textAnchor="end", fillColor=INK,
                     fontName="Helvetica"))
        if s["mean"] is not None:
            d.add(Rect(L, y, pw * s["mean"] / scale_max, 9, fillColor=BLUE, strokeColor=None))
            d.add(String(L + pw * s["mean"] / scale_max + 5, y + 1, calc.fmt(s["mean"]), fontSize=8.5, fillColor=INK,
                         fontName="Helvetica-Bold"))
    return d


def dots_drawing(score: int, scale_max: int = 5, size: float = 4.2) -> Drawing:
    """●●●○○ desenhado (a fonte standard não tem estes símbolos)."""
    d = Drawing(scale_max * (size * 2 + 3), size * 2 + 2)
    for i in range(scale_max):
        cx = size + i * (size * 2 + 3)
        d.add(Circle(cx, size + 1, size, strokeColor=BLUE, strokeWidth=1,
                     fillColor=BLUE if i < score else colors.white))
    return d


# ── estrutura do documento ──────────────────────────────────────────────────
def _table(data, widths, header=True, extra=None) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, GRID),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def _info_table(rows: list[tuple[str, str]], label_w: float = 38 * mm) -> Table:
    data = [[Paragraph(f"<b>{_x(k)}</b>", CELL), P(v, CELL)] for k, v in rows]
    return _table(data, [label_w, CONTENT_W - label_w], header=False,
                  extra=[("BACKGROUND", (0, 0), (0, -1), SOFT)])


def _bullets(items: list[str]) -> list:
    return [P(f"{i}. {t}", BODY) for i, t in enumerate(items, 1)]


def _build(story: list, title: str, footer: str, is_demo: bool = False) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=20 * mm,
                            bottomMargin=18 * mm, title=_safe(title), author=AUTHOR, subject=_safe(title),
                            creator=AUTHOR)

    def decorate(canvas, d):
        canvas.saveState()
        canvas.setFillColor(BLUE)
        canvas.rect(0, PAGE_H - 7 * mm, PAGE_W, 7 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.drawString(MARGIN, PAGE_H - 4.8 * mm, _safe(AUTHOR))
        if is_demo:
            canvas.setFont("Helvetica-Bold", 8.5)
            canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 4.8 * mm, "DADOS DE TESTE")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, 10 * mm, _safe(footer))
        canvas.drawRightString(PAGE_W - MARGIN, 10 * mm,
                               f"Gerado em {date.today().strftime('%d/%m/%Y')} · Página {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buf.getvalue()


CONFIDENTIAL = "Documento confidencial: contém dados de menores. Uso restrito ao clube e à família."


# ── relatório individual (treinador) ────────────────────────────────────────
def individual_pdf(r: dict) -> bytes:
    top = r["scale_max"]
    s = [P("Relatório de Avaliação Individual", TITLE), Spacer(1, 4)]
    s.append(_info_table([
        ("Jogador", r["player"]["name"]), ("Escalão", r["category"]), ("Equipa", f"{r['team']} ({r['club']})"),
        ("Data", f"{fmt_date(r['date'])} · {r['moment']}"), ("Treinador", r["coach"] or "—")]))
    if not r["complete"]:
        s += [Spacer(1, 4), P("Avaliação incompleta. Por avaliar: " + ", ".join(r["missing"]) + ".", SMALL)]
    s += [P("Roda das Competências", H2),
          radar_drawing([{"name": fmt_date(r["date"]), "scores": r["scores"], "fill": True}], top)]
    s.append(P("Resultados", H2))
    rows = [[P("Competência", CELL_B), P("Classificação", CELL_B), P("Nível", CELL_B), P("Observação", CELL_B)]]
    for x in r["results"]:
        rows.append([P(x["name"], CELL), P("—" if x["score"] is None else f"{x['score']}/{top}", CELL),
                     P(x["level"] or "—", CELL), P(x["note"] or "", CELL)])
    s.append(_table(rows, [46 * mm, 28 * mm, 32 * mm, CONTENT_W - 106 * mm]))
    s += [Spacer(1, 6), P(f"MÉDIA GLOBAL: {calc.fmt(r['average'])}/{top}", BIG)]

    s.append(P("Evolução", H2))
    e = r["evolution"]
    if not e:
        s.append(P("Primeira avaliação: ainda não há avaliação anterior para comparar.", BODY))
    else:
        delta = f" ({calc.fmt(e['avg_delta'], signed=True)})" if e["avg_delta"] is not None else ""
        s.append(P(f"Comparação com a avaliação anterior ({fmt_date(e['previous_date'])} · {e['previous_moment']}): "
                   f"média {calc.fmt(e['previous_average'])} para {calc.fmt(r['average'])}{delta}.", BODY))
        for label, key in (("Evolução em", "improved"), ("Mantido em", "maintained"), ("A consolidar", "to_consolidate")):
            if e[key]:
                s.append(PB(f"{label}:", ", ".join(e[key]) + "."))
        if e["previous_objectives"]:
            s.append(PB("Objetivos definidos na avaliação anterior:", e["previous_objectives"]))

    s.append(P("Áreas fortes", H2))
    if r["balanced"]:
        s.append(P("Perfil equilibrado: todas as competências avaliadas têm a mesma classificação.", BODY))
    s += _bullets([f"{a['name']}: {a['score']}" + (f" ({a['level']})" if a["level"] else "") for a in r["strengths"]])
    if r["tie_strengths"]:
        s.append(P("Há outras competências com a mesma classificação.", SMALL))
    s.append(P("Áreas a desenvolver", H2))
    s += _bullets([f"{a['name']}: {a['score']}" + (f" ({a['level']})" if a["level"] else "") for a in r["to_develop"]])
    if r["tie_to_develop"]:
        s.append(P("Há outras competências com a mesma classificação.", SMALL))
    s += [P("Observações do treinador", H2), P(r["general_notes"] or "Sem observações gerais registadas.", BODY),
          P("Objetivos para o próximo período", H2)]
    if r["objectives"]:
        s.append(P(r["objectives"], BODY))
    else:
        s.append(P("Ainda não definidos." + (" Sugestão de foco: " + ", ".join(r["suggested_focus"]) + "."
                                             if r["suggested_focus"] else ""), BODY))
    return _build(s, f"Relatório individual — {r['player']['name']}", CONFIDENTIAL, r["is_demo"])


# ── ficha do jogador ────────────────────────────────────────────────────────
def player_sheet_pdf(sheet: dict) -> bytes:
    p = sheet["player"]
    s = [P("Ficha do Jogador", TITLE), Spacer(1, 4)]
    rows = [("Nome", p["name"]), ("Data de nascimento", fmt_date(p["birth_date"])),
            ("Sexo", {"M": "Masculino", "F": "Feminino"}.get(p["sex"], "—")), ("Escalão", p["category"] or "—"),
            ("Equipa", p["team"] or "—"), ("Clube", p["club"] or "—"), ("Época", p["season"] or "—"),
            ("Nº da camisola", "—" if p["jersey_number"] is None else str(p["jersey_number"])),
            ("Entrada na equipa", fmt_date(p["joined_on"])), ("Observações", p["notes"] or "—")]
    info = _info_table(rows)
    photo = p.get("photo_path")
    if photo and os.path.exists(photo):
        try:
            img = Image(photo)
            ratio = min(34 * mm / img.imageWidth, 42 * mm / img.imageHeight)
            img.drawWidth, img.drawHeight = img.imageWidth * ratio, img.imageHeight * ratio
            head = Table([[img, info]], colWidths=[40 * mm, CONTENT_W - 40 * mm])
            head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            s.append(head)
        except Exception:
            s.append(info)
    else:
        s.append(info)
    s.append(P("Percurso nas equipas", H2))
    teams = [[P(x, CELL_B) for x in ("Escalão", "Equipa", "Época", "Entrada", "Saída")]]
    for h in p["team_history"]:
        teams.append([P(h["category"], CELL), P(f"{h['team']} ({h['club']})", CELL), P(h["season"], CELL),
                      P(fmt_date(h["joined_on"]), CELL), P(fmt_date(h["left_on"]) if h["left_on"] else "atual", CELL)])
    s.append(_table(teams, [22 * mm, 62 * mm, 24 * mm, 30 * mm, CONTENT_W - 138 * mm]))
    s.append(P("Histórico de avaliações", H2))
    hist = sheet["history"]
    if not hist:
        s.append(P("Ainda sem avaliações.", BODY))
    else:
        rows = [[P(x, CELL_B) for x in ("Data", "Momento", "Escalão", "Treinador", "Média global")]]
        for e in hist:
            rows.append([P(fmt_date(e["evaluation_date"]), CELL), P(e["moment"], CELL), P(e["category"], CELL),
                         P(e["coach"] or "—", CELL),
                         P(calc.fmt(e["average"]) + ("" if e["complete"] else " (incompleta)"), CELL)])
        s.append(_table(rows, [26 * mm, 42 * mm, 22 * mm, 40 * mm, CONTENT_W - 130 * mm]))
        if len(hist) >= 2:
            s += [P("Evolução da média global", H2),
                  line_drawing([{"date": e["evaluation_date"], "value": e["average"]} for e in hist], sheet["scale_max"])]
    return _build(s, f"Ficha do jogador — {p['name']}", CONFIDENTIAL, bool(p.get("is_demo")))


# ── relatório para os pais ──────────────────────────────────────────────────
def parent_pdf(r: dict) -> bytes:
    w, sec = r["wheel"], r["sections"]
    first = r["player"]["first_name"]
    s = [P(f"{first}: como está a correr", TITLE),
         P(f"{r['player']['name']} · {r['category']} · {r['team']} ({r['club']}) · {fmt_date(r['date'])} · {r['moment']}"
           + (f" · Treinador: {r['coach']}" if r["coach"] else ""), SMALL)]
    if r["incomplete_note"]:
        s.append(P(r["incomplete_note"], SMALL))
    s += [P(sec["evolution"]["title"], H2), P(sec["evolution"]["text"], BODY)]
    series = [{"name": "Esta avaliação", "scores": w["scores"], "fill": True}]
    if w["previous_scores"]:
        series.append({"name": f"Avaliação anterior ({fmt_date(w['previous_date'])})", "scores": w["previous_scores"],
                       "dash": True})
    s.append(radar_drawing(series, w["scale_max"]))
    rows = []
    for k in r["skills"]:
        rows.append([P(k["name"], CELL), dots_drawing(k["score"], w["scale_max"]), P(k["level"] or "", CELL)])
    if rows:
        s.append(KeepTogether([_table(rows, [70 * mm, 36 * mm, CONTENT_W - 106 * mm], header=False)]))
    for key in ("strengths", "working", "goals"):
        s += [P(sec[key]["title"], H2), P(sec[key]["text"], BODY)]
    if r["message"]:
        box = Table([[P("Mensagem do treinador: " + r["message"], BODY)]], colWidths=[CONTENT_W])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f0fb")),
                                 ("BOX", (0, 0), (-1, -1), 0.6, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                 ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        s += [Spacer(1, 10), box]
    s += [Spacer(1, 10), P("Cada jogador evolui ao seu ritmo. Obrigado pelo acompanhamento e apoio.", SMALL)]
    return _build(s, f"Relatório para os pais — {r['player']['name']}",
                  f"Relatório pessoal de {first}. Documento confidencial; não partilhar.", r["is_demo"])


# ── relatório da equipa ─────────────────────────────────────────────────────
def team_pdf(r: dict) -> bytes:
    t = r["team"]
    s = [P("Relatório da Equipa", TITLE),
         P(f"{t['name']} · {t['category']} · {t['club']} · {t['season']}", BODY)]
    if not r["has_data"]:
        s.append(P("Esta equipa ainda não tem avaliações.", BODY))
        return _build(s, f"Relatório da equipa — {t['name']}", CONFIDENTIAL, bool(t["is_demo"]))
    top = max(5, *(x["best"] or 0 for x in r["stats"]))
    e = r["evolution"]
    ind = [("Jogadores avaliados", f"{r['n_evaluated']} de {r['members']}"),
           ("Média global da equipa", f"{calc.fmt(r['average'])} / 5"),
           ("Avaliação considerada", f"mais recente de cada jogador em {fmt_date(r['as_of'])}")]
    if e and e["avg_delta"] is not None:
        ind.append(("Evolução da média", f"{calc.fmt(e['avg_before'])} para {calc.fmt(e['avg_after'])} "
                                         f"({calc.fmt(e['avg_delta'], signed=True)}), com {e['n_common']} jogadores reavaliados"))
    s += [Spacer(1, 4), _info_table(ind, 55 * mm), P("Média de cada competência", H2), bars_drawing(r["stats"], top)]
    rows = [[P(x, CELL_B) for x in ("Competência", "Média", "Mediana", "Melhor", "Mais baixo", "Avaliados")]]
    for x in r["stats"]:
        rows.append([P(x["name"], CELL), P(calc.fmt(x["mean"]), CELL), P(calc.fmt(x["median"]), CELL),
                     P("—" if x["best"] is None else str(x["best"]), CELL),
                     P("—" if x["lowest"] is None else str(x["lowest"]), CELL), P(str(x["n"]), CELL)])
    s.append(_table(rows, [52 * mm, 24 * mm, 24 * mm, 24 * mm, 26 * mm, CONTENT_W - 150 * mm]))

    s.append(P("Evolução da equipa", H2))
    if not e:
        s.append(P("A evolução surge quando a equipa tiver duas datas de avaliação.", BODY))
    elif e["n_common"] == 0:
        s.append(P("Nenhum jogador foi reavaliado entre as duas datas.", BODY))
    else:
        s.append(P(f"De {fmt_date(e['date_before'])} a {fmt_date(e['date_after'])}, com os {e['n_common']} jogadores "
                   f"reavaliados: média {calc.fmt(e['avg_before'])} para {calc.fmt(e['avg_after'])} "
                   f"({calc.fmt(e['avg_delta'], signed=True)}).", BODY))
        if e["homogeneous"]:
            s.append(P("A evolução foi semelhante em todas as competências.", BODY))
        else:
            s.append(PB("Maior evolução:", "; ".join(f"{a['name']} {calc.fmt(a['delta'], signed=True)}"
                                                    for a in e["most_improved"]) + "."))
            s.append(PB("Menor evolução:", "; ".join(f"{a['name']} {calc.fmt(a['delta'], signed=True)}"
                                                    for a in e["least_improved"]) + "."))
        rows = [[P(x, CELL_B) for x in ("Competência", "Inicial", "Atual", "Evolução")]]
        for x in e["rows"]:
            rows.append([P(x["name"], CELL), P(calc.fmt(x["before"]), CELL), P(calc.fmt(x["after"]), CELL),
                         P(calc.fmt(x["delta"], signed=True), CELL)])
        s += [Spacer(1, 4), _table(rows, [60 * mm, 30 * mm, 30 * mm, CONTENT_W - 120 * mm])]

    s.append(P("Competências que necessitam de maior atenção", H2))
    if r["balanced_means"]:
        s.append(P("As médias são idênticas em todas as competências.", BODY))
    s += _bullets([f"{a['name']}: média {calc.fmt(a['mean'])}" for a in r["attention"]])
    if r["strong"]:
        s.append(P("Competências em que a equipa está mais forte", H2))
        s += _bullets([f"{a['name']}: média {calc.fmt(a['mean'])}" for a in r["strong"]])
    s += [Spacer(1, 8), P("Este relatório não identifica nem compara jogadores.", SMALL)]
    return _build(s, f"Relatório da equipa — {t['name']}", CONFIDENTIAL, bool(t["is_demo"]))
