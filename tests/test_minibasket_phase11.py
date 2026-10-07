"""Fase 11 — exportação em PDF."""

import io
import os
import re
import sys

import pytest
from pypdf import PdfReader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from ui_helper import logged_in_app  # noqa: E402

from minibasket import access, auth, db, evaluations as ev, pdf, reports, service  # noqa: E402
from minibasket import competencies as comp  # noqa: E402
from minibasket.access import PermissionDenied  # noqa: E402

BANNED = re.compile(r"mau jogador|fraco|fraca|sem capacidade|incapaz|péssim|inferior|ranking|pior|dificuldade", re.I)
EXAMPLE = dict(zip(comp.KEYS, (3, 4, 3, 4, 2, 3, 4, 3, 4)))


def flat(n, **over):
    return {**{k: n for k in comp.KEYS}, **over}


def text_of(data: bytes) -> str:
    return " ".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages)


def pages(data: bytes) -> int:
    return len(PdfReader(io.BytesIO(data)).pages)


@pytest.fixture
def w(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    club = service.create_club("Clube", db_path=p)
    ta = service.create_team(club, "Sub-10 A", "Sub-10", "2024/2025", db_path=p)
    tb = service.create_team(club, "Sub-10 B", "Sub-10", "2024/2025", db_path=p)
    ids = {}
    for name, team in (("João Silva", ta), ("Rui Pires", ta), ("Inês Costa", ta), ("Eva Reis", tb)):
        pid = service.create_player(name, team, sex="M" if name.startswith(("João", "Rui")) else "F",
                                    birth_date="2016-05-03", jersey_number=7, joined_on="2024-01-01",
                                    notes="Canhoto", db_path=p)
        ids[name] = pid
        ev.create_evaluation(pid, "2024-09-15", "Avaliação Inicial", flat(2), general_notes=f"INTERNA-{name}",
                             notes={"shooting": f"NOTA-{name}"}, db_path=p)
        ev.create_evaluation(pid, "2024-12-15", "2.º Período", EXAMPLE, next_objectives="Melhorar equilíbrio",
                             parent_message=f"MSG-{name}", db_path=p)
    mk = lambda n, role, **k: auth.get_user(auth.create_user(n, n.title(), role, "palavra-passe-1", db_path=p, **k), p)
    admin, coach, other = mk("admin", "admin"), mk("treinadora", "coach"), mk("outra", "coach")
    pai = mk("pai1", "guardian")
    auth.set_team_assignments(coach["id"], [ta], p)
    auth.set_team_assignments(other["id"], [tb], p)
    auth.set_guardian_links(pai["id"], [ids["João Silva"]], p)
    evals = {n: [e["id"] for e in ev.list_evaluations(i, db_path=p)] for n, i in ids.items()}
    return dict(p=p, ta=ta, tb=tb, ids=ids, evals=evals, admin=admin, coach=coach, other=other, pai=pai)


# ── PDFs válidos e conteúdo ─────────────────────────────────────────────────
def test_individual_pdf_content(w):
    data, name = access.export_individual_pdf(w["coach"], w["evals"]["João Silva"][1], w["p"])
    assert data.startswith(b"%PDF-") and name == "relatorio-individual-joao-silva-2024-12-15.pdf"
    t = text_of(data)
    for needle in ("Relatório de Avaliação Individual", "João Silva", "Sub-10", "Sub-10 A", "15/12/2024", "3,33",
                   "MÉDIA GLOBAL: 3,33/5", "Roda das Competências", "Áreas fortes", "Áreas a desenvolver",
                   "Objetivos para o próximo período", "Melhorar equilíbrio", "Receção da Bola", "Contra-Ataque",
                   "Comparação com a avaliação anterior", "Documento confidencial"):
        assert needle in t, needle
    meta = PdfReader(io.BytesIO(data)).metadata
    assert "João Silva" in meta.title and meta.author == pdf.AUTHOR


def test_individual_pdf_first_evaluation_and_incomplete(w):
    first = text_of(access.export_individual_pdf(w["coach"], w["evals"]["João Silva"][0], w["p"])[0])
    assert "Primeira avaliação" in first
    pid = w["ids"]["Rui Pires"]
    eid = ev.create_evaluation(pid, "2025-01-10", "3.º Período", {"shooting": 4, "passing": 4}, db_path=w["p"])
    t = text_of(access.export_individual_pdf(w["coach"], eid, w["p"])[0])
    assert "Avaliação incompleta" in t and "Perfil equilibrado" in t


def test_user_text_is_escaped_not_interpreted_and_unsupported_chars_survive(w):
    pid = w["ids"]["João Silva"]
    eid = ev.create_evaluation(pid, "2025-01-10", "3.º Período", EXAMPLE, notes={"shooting": "<b>negrito</b> & <i>x</i> → ✓ 😀"},
                               general_notes="<para>quebra</para> " + "texto longo " * 400, db_path=w["p"])
    data, _ = access.export_individual_pdf(w["admin"], eid, w["p"])
    t = text_of(data)
    assert "<b>negrito</b>" in t and "<i>x</i>" in t and "<para>" in t              # literal, não interpretado
    assert pages(data) >= 2                                                      # texto longo → várias páginas, sem erro


def test_demo_label_only_for_demo_data(w):
    assert "DADOS DE TESTE" not in text_of(access.export_individual_pdf(w["admin"], w["evals"]["João Silva"][1], w["p"])[0])
    with db.connect(w["p"]) as c:
        c.execute("UPDATE evaluations SET is_demo=1")
    assert "DADOS DE TESTE" in text_of(access.export_individual_pdf(w["admin"], w["evals"]["João Silva"][1], w["p"])[0])


def test_parent_pdf_content_and_privacy(w):
    data, name = access.export_parent_pdf(w["pai"], w["evals"]["João Silva"][1], w["p"])
    assert name == "relatorio-pais-joao-silva-2024-12-15.pdf"
    t = text_of(data)
    for needle in ("João: como está a correr", "Como está a evoluir?", "Os seus pontos fortes", "O que estamos a trabalhar",
                   "Objetivos para a próxima etapa", "O João apresentou uma evolução positiva", "MSG-João Silva",
                   "Melhorar equilíbrio", "Cada jogador evolui ao seu ritmo"):
        assert needle in t, needle
    for forbidden in ("INTERNA-", "NOTA-", "Rui", "Inês", "Eva", "média da equipa", "Média global"):
        assert forbidden not in t, forbidden
    assert not BANNED.search(t)


def test_team_pdf_content_and_privacy(w):
    data, name = access.export_team_pdf(w["coach"], w["ta"], w["p"])
    assert name == "relatorio-equipa-sub-10-a-2024-2025-2024-12-15.pdf"
    t = text_of(data)
    for needle in ("Relatório da Equipa", "Sub-10 A", "Jogadores avaliados", "3 de 3", "Média global da equipa",
                   "Média de cada competência", "Evolução da equipa", "Maior evolução", "Menor evolução",
                   "necessitam de maior atenção", "Mediana", "Este relatório não identifica nem compara jogadores"):
        assert needle in t, needle
    for forbidden in ("João", "Rui", "Inês", "Eva", "INTERNA-"):
        assert forbidden not in t, forbidden
    assert not BANNED.search(t)


def test_team_pdf_without_evaluations_and_single_date(w):
    empty = service.create_team(1, "Vazia", "Sub-8", "2024/2025", db_path=w["p"])
    t = text_of(pdf.team_pdf(reports.team_report(empty, w["p"])))
    assert "ainda não tem avaliações" in t
    one = service.create_team(1, "Uma data", "Sub-12", "2024/2025", db_path=w["p"])
    pid = service.create_player("Zé", one, joined_on="2024-01-01", db_path=w["p"])
    ev.create_evaluation(pid, "2024-10-01", "1.º Período", EXAMPLE, db_path=w["p"])
    t = text_of(pdf.team_pdf(reports.team_report(one, w["p"])))
    assert "duas datas de avaliação" in t and "Zé" not in t


def test_player_sheet_pdf_with_and_without_photo(w, tmp_path):
    from PIL import Image
    png = tmp_path / "foto.png"
    Image.new("RGB", (60, 80), (200, 120, 60)).save(png)
    pid = w["ids"]["João Silva"]
    service.update_player(pid, "João Silva", "2016-05-03", "M", "Canhoto", str(png), 7, db_path=w["p"])
    data, name = access.export_player_sheet_pdf(w["coach"], pid, w["p"])
    assert name.startswith("ficha-joao-silva-") and pages(data) >= 1
    t = text_of(data)
    for needle in ("Ficha do Jogador", "João Silva", "03/05/2016", "Masculino", "Sub-10 A", "Clube", "2024/2025", "Canhoto",
                   "Percurso nas equipas", "Histórico de avaliações", "3,33", "Evolução da média global"):
        assert needle in t, needle
    with db.connect(w["p"]) as c:                                                  # foto em falta / ficheiro inválido
        c.execute("UPDATE players SET photo_path='/nao/existe.png' WHERE id=?", (pid,))
    assert text_of(access.export_player_sheet_pdf(w["coach"], pid, w["p"])[0]).count("Ficha do Jogador") == 1
    bad = tmp_path / "x.png"
    bad.write_bytes(b"nao e imagem")
    with db.connect(w["p"]) as c:
        c.execute("UPDATE players SET photo_path=? WHERE id=?", (str(bad), pid))
    assert "Ficha do Jogador" in text_of(access.export_player_sheet_pdf(w["coach"], pid, w["p"])[0])


def test_player_sheet_after_category_change_and_no_evaluations(w):
    new = service.create_player("Novo", w["ta"], joined_on="2024-01-01", db_path=w["p"])
    t = text_of(access.export_player_sheet_pdf(w["admin"], new, w["p"])[0])
    assert "Ainda sem avaliações" in t
    service.change_team(w["ids"]["Rui Pires"], w["tb"], "2025-01-05", db_path=w["p"])
    t = text_of(access.export_player_sheet_pdf(w["admin"], w["ids"]["Rui Pires"], w["p"])[0])
    assert t.count("Sub-10 A") >= 1 and t.count("Sub-10 B") >= 1 and "atual" in t


# ── permissões na exportação ────────────────────────────────────────────────
def test_export_permissions(w):
    p, ev_joao, ev_eva = w["p"], w["evals"]["João Silva"][1], w["evals"]["Eva Reis"][1]
    for fn, target in ((access.export_individual_pdf, ev_eva), (access.export_parent_pdf, ev_eva),
                       (access.export_team_pdf, w["tb"]), (access.export_player_sheet_pdf, w["ids"]["Eva Reis"])):
        with pytest.raises(PermissionDenied):
            fn(w["coach"], target, p)                                         # treinador de outra equipa
    for fn, target in ((access.export_individual_pdf, ev_joao), (access.export_team_pdf, w["ta"]),
                       (access.export_player_sheet_pdf, w["ids"]["João Silva"])):
        with pytest.raises(PermissionDenied):
            fn(w["pai"], target, p)                                           # encarregado: sem relatórios internos nem ficha
    with pytest.raises(PermissionDenied):
        access.export_parent_pdf(w["pai"], ev_eva, p)                          # filho de outro
    with pytest.raises(PermissionDenied):
        access.export_parent_pdf(w["pai"], w["evals"]["Rui Pires"][1], p)      # colega
    assert access.export_parent_pdf(w["pai"], ev_joao, p)[0].startswith(b"%PDF")
    for fn, target in ((access.export_individual_pdf, ev_eva), (access.export_parent_pdf, ev_eva),
                       (access.export_team_pdf, w["tb"]), (access.export_player_sheet_pdf, w["ids"]["Eva Reis"])):
        assert fn(w["admin"], target, p)[0].startswith(b"%PDF")               # administrador
    with pytest.raises(PermissionDenied):
        access.export_team_pdf(None, w["ta"], p)
    with pytest.raises(service.ValidationError):
        access.export_individual_pdf(w["admin"], 9999, p)


def test_guardian_cannot_export_superseded_version(w):
    old = w["evals"]["João Silva"][1]
    ev.correct_evaluation(old, "2024-12-15", "2.º Período", flat(4), db_path=w["p"])
    with pytest.raises(PermissionDenied):
        access.export_parent_pdf(w["pai"], old, w["p"])


# ── funções auxiliares e gráficos ───────────────────────────────────────────
def test_filename_and_safe_text():
    assert pdf.filename("ficha", "João  Ç. d'Ávila", "2025-01-02") == "ficha-joao-c-d-avila-2025-01-02.pdf"
    assert pdf.slug("") == "relatorio" and pdf.slug("***") == "relatorio"
    assert pdf._safe("ação → ✓") == "ação ? ?" and pdf._safe("Receção · 1.º") == "Receção · 1.º"
    assert pdf._x("<b>&") == "&lt;b&gt;&amp;"


def test_all_generated_text_fits_the_pdf_font(w):
    import json
    r = reports.parent_report(w["evals"]["João Silva"][1], w["p"])
    for k in r["skills"]:
        k.pop("dots")                                          # só texto de ecrã; no PDF as bolinhas são desenhadas
    json.dumps(r, ensure_ascii=False).encode("cp1252")        # os textos gerados pela app cabem na fonte (sem «?»)


def test_drawings_handle_edge_cases():
    full = {"name": "A", "scores": EXAMPLE, "fill": True}
    part = {"name": "B", "scores": {"shooting": 3, "passing": 4}, "dash": True}
    empty = {"name": "C", "scores": {}}
    for series in ([full], [full, part], [part], [empty], [full, empty]):
        d = pdf.radar_drawing(series)
        assert d.width > 0 and len(d.contents) > 20
    assert pdf.line_drawing([]).height > 0
    assert pdf.line_drawing([{"date": "2024-09-15", "value": 3.0}]).height > 0
    pts = [{"date": f"2024-{m:02d}-15", "value": v} for m, v in zip(range(1, 11), (2, None, 3, 3.5, None, None, 4, 4, 4.5, 5))]
    assert pdf.line_drawing(pts).height > 0
    stats = [{"key": k, "mean": None if i % 2 else 3.2} for i, k in enumerate(comp.KEYS)]
    assert pdf.bars_drawing(stats).height > 0
    assert pdf.dots_drawing(0).width > 0 and pdf.dots_drawing(5).width > 0


def test_pdf_text_extraction_has_no_replacement_chars(w):
    t = text_of(access.export_individual_pdf(w["admin"], w["evals"]["João Silva"][1], w["p"])[0])
    assert "�" not in t and "Receção" in t and "Finalizações" in t


# ── UI ──────────────────────────────────────────────────────────────────────
def test_ui_download_buttons(w, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", w["p"])
    at = logged_in_app("admin", "admin")
    at.sidebar.radio[0].set_value("Relatórios").run()
    at.radio(key="rep_cat").set_value("Sub-10").run()
    at.radio(key="rep_t_cat").set_value("Sub-10").run()
    at.radio(key="rep_p_cat").set_value("Sub-10").run()
    assert not at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels.count("⬇️ Descarregar PDF") == 3                                 # individual, equipa, pais
    at.sidebar.radio[0].set_value("Jogadores").run()
    assert not at.exception and any(b.proto.label == "⬇️ Ficha em PDF" for b in at.get("download_button"))


def test_ui_guardian_has_only_parent_pdf(w, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", w["p"])
    at = logged_in_app("guardian", "pai1")
    assert not at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["⬇️ Descarregar PDF"]
