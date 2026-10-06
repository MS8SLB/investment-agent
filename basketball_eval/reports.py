"""Relatórios automáticos (texto e HTML imprimível) para treinadores e pais."""

from __future__ import annotations

import html

from . import defensive_movement as dm
from . import qualitative as ql
from . import service
from .db import connect, init_db


def _player(player_id: int, db_path: str | None = None) -> dict:
    init_db(db_path)
    with connect(db_path) as c:
        row = c.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
    if row is None:
        raise ValueError(f"Jogador {player_id} não existe.")
    return dict(row)


def coach_report(player_id: int, db_path: str | None = None) -> str:
    """Relatório técnico: tempo, evolução e grelha qualitativa."""
    p = _player(player_id, db_path)
    return "\n".join([
        f"RELATÓRIO DO TREINADOR — {p['name']} ({p['category']})", "=" * 50, "",
        service.coach_report_text(player_id, db_path), "", "-" * 50, "",
        service.qualitative_report_text(player_id, db_path),
    ])


def parent_report(player_id: int, db_path: str | None = None) -> str:
    """Relatório para pais: linguagem simples, sem comparação com outros atletas."""
    p = _player(player_id, db_path)
    lines = [f"RELATÓRIO PARA OS PAIS — {p['name']} ({p['category']})", "",
             "A defesa é uma competência fundamental no basquetebol. Avaliámos a rapidez e a "
             "técnica dos deslocamentos defensivos do(a) atleta.", ""]
    rep = service.player_report(player_id, db_path)
    if rep is None:
        lines.append("Ainda não há resultado do teste de deslocamento defensivo.")
    else:
        t, evo = rep["test"], rep["evolution"]
        lines.append(f"Tempo no percurso defensivo: {dm.fmt_seconds(t['best_time'])} (quanto menor, melhor).")
        if evo:
            lines.append(f"Face à avaliação anterior ({dm.fmt_seconds(t['previous_best_time'])}): {evo.message}")
        else:
            lines.append("Este é o primeiro registo; nas próximas avaliações mostraremos a evolução.")
    qh = service.qualitative_history(player_id, db_path)
    if qh:
        last = qh[-1]["ratings"]
        sm = ql.summarize(last)
        lines += ["", "Técnica observada pelo treinador:"]
        lines += [f"  + {ql.CRITERIA[k]['name']}" for k in sm.strengths] or ["  + (a desenvolver)"]
        if sm.to_improve:
            lines += ["", "Em que vamos continuar a trabalhar:"]
            lines += [f"  - {ql.CRITERIA[k]['name']}" for k in sm.to_improve[:3]]
        if qh[-1]["strengths_note"]:
            lines += ["", f"Nota do treinador: {qh[-1]['strengths_note']}"]
    lines += ["", "O progresso de cada criança é individual: o importante é a evolução face a si própria."]
    return "\n".join(lines)


def team_report(category: str, date_from, date_to, team: str | None = None,
                db_path: str | None = None) -> str:
    """Relatório coletivo do escalão para o treinador."""
    sheet = service.team_sheet(category, date_from, date_to, team, db_path)
    lines = [f"RELATÓRIO DE EQUIPA — {category}" + (f" · {team}" if team else ""),
             f"Período: {date_from} a {date_to}", ""]
    if not sheet["n"]:
        return "\n".join(lines + ["Sem avaliações válidas no período."])
    lines += [f"Atletas avaliados: {sheet['n']}", f"Média: {dm.fmt_seconds(sheet['mean'])}",
              f"Mais rápido: {dm.fmt_seconds(sheet['fastest'])}", f"Mais lento: {dm.fmt_seconds(sheet['slowest'])}", ""]
    for r in sheet["rows"]:
        pct = "—" if r["percentile"] is None else f"{r['percentile']}"
        lines.append(f"  {r['name']}: {dm.fmt_seconds(r['best_time'])} · nota percentil {pct}")
    evo = service.team_evolution(category, team, db_path)
    if len(evo) > 1:
        lines += ["", f"Evolução da média: {dm.fmt_seconds(evo[0]['mean'])} ({evo[0]['date']}) → "
                      f"{dm.fmt_seconds(evo[-1]['mean'])} ({evo[-1]['date']})"]
    return "\n".join(lines)


def to_html(title: str, text: str) -> str:
    """Documento HTML autónomo, pronto a imprimir/guardar em PDF."""
    return (f"<!doctype html><html lang='pt'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            "<style>body{font-family:sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem}"
            "pre{white-space:pre-wrap;font:inherit;line-height:1.5}</style></head><body>"
            f"<pre>{html.escape(text)}</pre></body></html>")
