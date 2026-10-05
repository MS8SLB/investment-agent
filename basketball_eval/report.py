"""Relatório individual de avaliação qualitativa.

Duas camadas separadas, para permitir novos formatos sem tocar nos dados:
  * `build_report()`  → `ReportData`: estrutura neutra (sem HTML) com tudo o que o relatório mostra.
  * `render_html()`   → HTML autónomo (CSS e gráficos SVG inline), pronto a imprimir/guardar como PDF.

Exportação PDF (futura): consumir `ReportData` num novo renderizador (ex.: `render_pdf(report)` com
WeasyPrint sobre `render_html`, ou ReportLab) — não é preciso alterar mais nada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from html import escape
from typing import Optional

from . import competencies, qual_service as svc, qualitative as q
from .competencies import LEVELS


@dataclass
class CriterionLine:
    label: str
    score: Optional[int]
    level: Optional[str]            # «4 – Bom»; None se não avaliado


@dataclass
class DimensionBlock:
    name: str
    title: str
    mean: Optional[float]
    criteria: list[CriterionLine]


@dataclass
class ReportData:
    title: str
    subtitle: str
    player: str
    team: str
    age_group: str
    date: str                       # ISO
    coach: str
    competency: str
    mean: Optional[float]
    mean_display: str
    label: str
    dimensions: list[DimensionBlock]
    strengths: list[str]
    improvements: list[str]
    observations: str
    history: list[tuple[str, float]] = field(default_factory=list)   # (data ISO, média) até esta avaliação
    evolution: Optional[q.Evolution] = None


def fmt_date(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%d/%m/%Y")


def build_report(evaluation_id: int, db_path: str | None = None) -> ReportData:
    ev = svc.get_evaluation(evaluation_id, db_path)
    if ev is None:
        raise ValueError(f"Avaliação {evaluation_id} não existe.")
    comp = competencies.get(ev["competency"])
    scores = ev["scores"]
    blocks = [
        DimensionBlock(
            d.name, d.title, ev["dimension_means"][d.key],
            [CriterionLine(c.label, scores.get(c.key), q.level_label(scores[c.key]) if c.key in scores else None)
             for c in d.criteria])
        for d in comp.dimensions
    ]
    ins = q.insights(comp, scores)
    upto = (ev["evaluation_date"], ev["id"])
    history = [(e["evaluation_date"], e["mean"]) for e in svc.list_evaluations(ev["player_id"], ev["competency"], db_path)
               if e["mean"] is not None and (e["evaluation_date"], e["id"]) <= upto]
    return ReportData(
        title="RELATÓRIO DE AVALIAÇÃO",
        subtitle=f"{comp.domain.upper()} – {comp.name.upper()}",
        player=ev["player_name"], team=ev["team_name"] or "—", age_group=ev["age_group"],
        date=ev["evaluation_date"], coach=ev["coach_name"] or "—", competency=comp.name,
        mean=ev["mean"], mean_display=q.fmt_score(ev["mean"]), label=(ev["label"] or "—"),
        dimensions=blocks,
        strengths=[i.text for i in ins.strengths], improvements=[i.text for i in ins.improvements],
        observations=ev["observations"] or "", history=history, evolution=q.evolution(history),
    )


# ── Gráficos SVG (sem dependências) ─────────────────────────────────────────

ACCENT, GRID, INK, MUTED = "#e8590c", "#d9dee5", "#1b2733", "#667085"


def radar_svg(labels: list[str], values: list[Optional[float]], size: int = 360) -> str:
    """Radar 1–5 (centro = 0). Eixos sem avaliação ficam sem ponto."""
    width = int(size * 1.5)                      # margem lateral para os rótulos dos eixos
    n, cx, cy, r = len(labels), width / 2, size / 2, size * 0.34

    def pt(i, v):
        a = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * v / 5 * math.cos(a), cy + r * v / 5 * math.sin(a)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {size}" width="{width}" style="max-width:100%;height:auto" role="img" '
           f'aria-label="Radar das competências do lançamento">']
    for lvl in range(1, 6):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, lvl) for i in range(n)))
        out.append(f'<polygon points="{pts}" fill="none" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{cx + 3:.1f}" y="{cy - r * lvl / 5 + 10:.1f}" font-size="9" fill="{MUTED}">{lvl}</text>')
    for i, lab in enumerate(labels):
        x, y = pt(i, 5)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        lx, ly = pt(i, 5.9)
        anchor = "middle" if abs(lx - cx) < 8 else ("start" if lx > cx else "end")
        out.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" font-size="12" font-weight="600" fill="{INK}" '
                   f'text-anchor="{anchor}">{escape(lab)}</text>')
    drawn = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(drawn) >= 3:
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, v) for i, v in drawn))
        out.append(f'<polygon points="{poly}" fill="{ACCENT}" fill-opacity=".22" stroke="{ACCENT}" stroke-width="2.5"/>')
    elif len(drawn) == 2:
        (i, v), (j, w) = drawn
        (x1, y1), (x2, y2) = pt(i, v), pt(j, w)
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ACCENT}" stroke-width="2.5"/>')
    for i, v in drawn:
        x, y = pt(i, v)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{ACCENT}"/>')
    out.append("</svg>")
    return "".join(out)


def line_svg(points: list[tuple[str, float]], w: int = 520, h: int = 240) -> str:
    """Evolução da média ao longo do tempo (eixo vertical 1–5, pontos equidistantes)."""
    if not points:
        return ""
    left, right, top, bottom = 34, 16, 18, 40
    iw, ih = w - left - right, h - top - bottom

    def X(i):
        return left + (iw / 2 if len(points) == 1 else iw * i / (len(points) - 1))

    def Y(v):
        return top + ih * (5 - v) / 4          # 1 em baixo, 5 em cima

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" role="img" '
           f'aria-label="Evolução da média técnica">']
    for lvl in range(1, 6):
        out.append(f'<line x1="{left}" y1="{Y(lvl):.1f}" x2="{w - right}" y2="{Y(lvl):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left - 8}" y="{Y(lvl) + 4:.1f}" font-size="10" fill="{MUTED}" text-anchor="end">{lvl}</text>')
    if len(points) > 1:
        path = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, (_, v) in enumerate(points))
        out.append(f'<polyline points="{path}" fill="none" stroke="{ACCENT}" stroke-width="2.5"/>')
    for i, (d, v) in enumerate(points):
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{ACCENT}"/>')
        out.append(f'<text x="{X(i):.1f}" y="{Y(v) - 10:.1f}" font-size="11" font-weight="600" fill="{INK}" '
                   f'text-anchor="middle">{q.fmt_score(v)}</text>')
        out.append(f'<text x="{X(i):.1f}" y="{h - 14}" font-size="10" fill="{MUTED}" text-anchor="middle">'
                   f'{fmt_date(d)[:5]}</text>')
    out.append("</svg>")
    return "".join(out)


# ── HTML ────────────────────────────────────────────────────────────────────

CSS = f"""
*{{box-sizing:border-box}}body{{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:{INK};
margin:0;background:#f4f6f8}}.page{{max-width:860px;margin:24px auto;background:#fff;padding:36px 40px;
border-radius:12px;box-shadow:0 1px 6px rgba(0,0,0,.08)}}h1{{margin:0;font-size:22px;letter-spacing:.04em}}
h2{{margin:28px 0 10px;font-size:15px;letter-spacing:.05em;text-transform:uppercase;color:{ACCENT}}}
.sub{{color:{MUTED};font-weight:600;letter-spacing:.04em;margin-top:2px}}
.meta{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:22px 0}}
.meta div{{background:#f4f6f8;border-radius:8px;padding:8px 12px;font-size:13px}}
.meta b{{display:block;font-size:11px;color:{MUTED};text-transform:uppercase;letter-spacing:.05em}}
.score{{display:flex;align-items:center;gap:22px;background:#fff4ec;border-radius:12px;padding:16px 24px}}
.score .n{{font-size:40px;font-weight:800;color:{ACCENT}}}.score .l{{font-size:20px;font-weight:700}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:center}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:6px 8px;border-bottom:1px solid #e6eaef;text-align:left}}
th{{color:{MUTED};font-size:11px;text-transform:uppercase}}.dim td{{background:#f4f6f8;font-weight:700}}
.na{{color:#98a2b3}}ul{{margin:4px 0 0 18px;padding:0}}li{{margin:3px 0}}
.obs{{white-space:pre-wrap;background:#f4f6f8;border-radius:8px;padding:12px 14px;font-size:14px}}
.foot{{margin-top:28px;font-size:11px;color:{MUTED};border-top:1px solid #e6eaef;padding-top:10px}}
@media print{{body{{background:#fff}}.page{{box-shadow:none;margin:0;padding:0;max-width:none}}}}
@media (max-width:640px){{.cols,.meta{{grid-template-columns:1fr}}.page{{padding:20px}}}}
"""


def _list(items: list[str], empty: str) -> str:
    if not items:
        return f'<p class="na">{escape(empty)}</p>'
    return "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def render_html(r: ReportData) -> str:
    rows = []
    for d in r.dimensions:
        rows.append(f'<tr class="dim"><td>{escape(d.title)}</td><td>{q.fmt_score(d.mean, compact=True)}</td></tr>')
        for c in d.criteria:
            cell = escape(c.level) if c.level else '<span class="na">não avaliado</span>'
            rows.append(f"<tr><td>{escape(c.label)}</td><td>{cell}</td></tr>")
    evo = ""
    if r.evolution and r.evolution.n >= 2:
        e = r.evolution
        pct = f" ({q.fmt_score(e.change_pct, signed=True)}%)" if e.change_pct is not None else ""
        evo = (f"<p><b>{q.fmt_score(e.first)} → {q.fmt_score(e.last)}</b> · {e.trend_label.capitalize()}: "
               f"{q.fmt_score(e.change, signed=True)} pontos{pct}. "
               f"Primeira avaliação: {fmt_date(e.first_date)} · Esta avaliação: {fmt_date(e.last_date)}.</p>")
    elif r.history:
        evo = '<p class="na">Primeira avaliação — ainda sem evolução para apresentar.</p>'
    scale = " · ".join(f"{n} – {t}" for n, t in LEVELS.items())
    meta = [("Jogador", r.player), ("Equipa", r.team), ("Escalão", r.age_group),
            ("Data", fmt_date(r.date)), ("Treinador", r.coach), ("Competência", r.competency)]
    return f"""<!doctype html><html lang="pt-PT"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(r.title)} – {escape(r.player)}</title>
<style>{CSS}</style></head><body><div class="page">
<h1>{escape(r.title)}</h1><div class="sub">{escape(r.subtitle)}</div>
<div class="meta">{"".join(f"<div><b>{k}</b>{escape(v)}</div>" for k, v in meta)}</div>
<div class="score"><div class="n">{r.mean_display} / 5</div><div><div class="l">{escape(r.label.upper())}</div>
<div class="na">Média técnica do {escape(r.competency.lower())}</div></div></div>
<h2>Competências do {escape(r.competency.lower())}</h2>
<div class="cols">{radar_svg([d.name for d in r.dimensions], [d.mean for d in r.dimensions], 300)}
<div><table><tr><th>Dimensão</th><th>Média</th></tr>{"".join(
    f"<tr><td>{escape(d.name)}</td><td>{q.fmt_score(d.mean, compact=True)}</td></tr>" for d in r.dimensions)}</table></div></div>
<h2>Resultados por critério</h2><table><tr><th>Critério</th><th>Avaliação</th></tr>{"".join(rows)}</table>
<h2>Evolução do {escape(r.competency.lower())}</h2>{line_svg(r.history)}{evo}
<div class="cols" style="align-items:start"><div><h2>Pontos fortes</h2>{_list(r.strengths, "Nenhum critério avaliado com 4 ou 5.")}</div>
<div><h2>Áreas de melhoria</h2>{_list(r.improvements, "Nenhum critério avaliado com 1 ou 2.")}</div></div>
<h2>Observações do treinador</h2>{f'<div class="obs">{escape(r.observations)}</div>' if r.observations
                                  else '<p class="na">Sem observações.</p>'}
<div class="foot">Escala qualitativa estruturada: {escape(scale)}. Síntese baseada exclusivamente nas pontuações
introduzidas pelo treinador; sem comparação com outros jogadores. Avaliar → identificar → acompanhar → intervir → reavaliar.</div>
</div></body></html>"""
