"""Fase 12 — ciclo completo (os 16 pontos do «resultado final» do enunciado), com permissões reais."""

import io
import os
import sys

from pypdf import PdfReader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minibasket import access, auth, calc, charts, db, evolution, service, teamstats  # noqa: E402
from minibasket import competencies as comp  # noqa: E402

EXAMPLE = dict(zip(comp.KEYS, (3, 4, 3, 4, 2, 3, 4, 3, 4)))      # média 3,33


def flat(n, **over):
    return {**{k: n for k in comp.KEYS}, **over}


def test_full_coach_workflow(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    admin = auth.get_user(auth.create_first_admin("admin", "Administrador", "palavra-passe-1", p), p)
    club = access.create_club(admin, "Clube Exemplo", db_path=p)
    coach_id = auth.create_user("treinador", "Marta Treinadora", "coach", "palavra-passe-1", club_id=club, db_path=p)
    coach = auth.get_user(coach_id, p)

    # 1. Criar uma equipa
    t10 = access.create_team(coach, club, "Sub-10 A", "Sub-10", "2025/2026", db_path=p)
    assert [t["name"] for t in access.list_teams(coach, db_path=p)] == ["Sub-10 A"]

    # 2. Criar jogadores Sub-8, Sub-10 e Sub-12
    t8 = access.create_team(coach, club, "Sub-8 A", "Sub-8", "2025/2026", db_path=p)
    t12 = access.create_team(coach, club, "Sub-12 A", "Sub-12", "2025/2026", db_path=p)
    ids = {n: access.create_player(coach, n, t, sex="M", joined_on="2025-09-01", jersey_number=i, db_path=p)
           for i, (n, t) in enumerate([("João", t10), ("Rui", t10), ("Tomás", t10), ("Leo", t8), ("Gil", t12)], 1)}
    assert {x["category"] for x in access.search_players(coach, db_path=p)} == {"Sub-8", "Sub-10", "Sub-12"}

    # 3. Abrir a ficha de cada jogador
    ficha = access.get_player(coach, ids["João"], p)
    assert (ficha["name"], ficha["category"], ficha["team"], ficha["jersey_number"]) == ("João", "Sub-10", "Sub-10 A", 1)

    # 4–5. Realizar uma avaliação e classificar as nove competências de 1 a 5
    first = access.create_evaluation(coach, ids["João"], "2025-09-15", "Avaliação Inicial",
                                     {k: v - 1 if v > 1 else v for k, v in EXAMPLE.items()}, db_path=p)
    for name in ("Rui", "Tomás"):
        access.create_evaluation(coach, ids[name], "2025-09-15", "Avaliação Inicial", flat(2), db_path=p)
    second = access.create_evaluation(coach, ids["João"], "2025-12-15", "1.º Período", EXAMPLE,
                                      {"shooting": "Melhorou a preparação dos pés."},
                                      general_notes="Boa evolução.", next_objectives="Melhorar o equilíbrio.",
                                      parent_message="Muito empenho!", db_path=p)
    for name in ("Rui", "Tomás"):
        access.create_evaluation(coach, ids[name], "2025-12-15", "1.º Período", flat(3), db_path=p)
    ev = access.list_evaluations(coach, ids["João"], db_path=p)
    assert len(ev) == 2 and all(e["coach"] == "Marta Treinadora" for e in ev)

    # 6. Roda das Competências automática
    fig = charts.radar_figure([{"name": "Atual", "scores": ev[1]["scores"]}, {"name": "Inicial", "scores": ev[0]["scores"]}])
    assert len(fig.data[0].theta) == 10 and tuple(fig.layout.polar.radialaxis.range) == (0, 5)

    # 7. Média global (3,33)
    assert calc.fmt(ev[1]["average"]) == "3,33" and ev[1]["complete"]

    # 8. Comparar avaliações
    cmp_ = calc.compare(ev[0]["scores"], ev[1]["scores"])
    assert cmp_["avg_delta"] > 0 and cmp_["n_common"] == 9

    # 9. Evolução individual
    series = evolution.evolution_series(ev)
    assert [calc.fmt(x["average"]) for x in series] == ["2,33", "3,33"]
    assert evolution.competency_series(ev, "shooting")[0]["score"] == 2

    # 10. Evolução da equipa
    tl = access.team_timeline(coach, t10, p)
    assert [x["n_evaluated"] for x in tl] == [3, 3]
    stats = teamstats.competency_stats(tl[-1]["snapshot"])
    assert stats[0]["n"] == 3 and stats[0]["median"] is not None
    assert access.team_overview(coach, t10, p)["change"]["avg_delta"] > 0

    # 11–12. Pontos fortes e competências a desenvolver
    rep = access.individual_report(coach, second, p)
    assert [a["name"] for a in rep["strengths"]][0] == "Drible / Domínio da Bola"
    assert rep["to_develop"][0]["name"] == "Trabalho de Pés"

    # 13. Relatório para o treinador (individual e da equipa)
    assert rep["evolution"]["previous_date"] == "2025-09-15" and rep["objectives"] == "Melhorar o equilíbrio."
    team_rep = access.team_report(coach, t10, p)
    assert team_rep["n_evaluated"] == 3 and team_rep["evolution"]["n_common"] == 3

    # 14. Relatório simples para os pais
    parent = access.parent_report(coach, second, p)
    assert parent["sections"]["evolution"]["title"] == "Como está a evoluir?"
    assert "evolução positiva" in parent["sections"]["evolution"]["text"]

    # 15. Exportar os relatórios
    for data, name in (access.export_individual_pdf(coach, second, p), access.export_parent_pdf(coach, second, p),
                       access.export_team_pdf(coach, t10, p), access.export_player_sheet_pdf(coach, ids["João"], p)):
        assert data.startswith(b"%PDF") and name.endswith(".pdf") and len(PdfReader(io.BytesIO(data)).pages) >= 1

    # 16. Manter todo o histórico (também depois de uma correção e de mudar de escalão)
    fixed = access.correct_evaluation(coach, first, "2025-09-15", "Avaliação Inicial", flat(2), db_path=p)
    assert [e["id"] for e in access.list_evaluations(coach, ids["João"], db_path=p)] == [fixed, second]
    assert len(access.list_evaluations(coach, ids["João"], include_superseded=True, db_path=p)) == 3
    service.change_team(ids["João"], t12, "2026-09-01", db_path=p)
    hist = service.get_player(ids["João"], p)["team_history"]
    assert [h["category"] for h in hist] == ["Sub-12", "Sub-10"]
    admin_view = access.list_evaluations(admin, ids["João"], include_superseded=True, db_path=p)
    assert len(admin_view) == 3 and {e["category"] for e in admin_view} == {"Sub-10"}      # histórico intacto no escalão antigo
    assert access.individual_report(admin, second, p)["category"] == "Sub-10"


def test_parent_workflow_end_to_end(tmp_path):
    p = str(tmp_path / "mb.db")
    db.init_db(p)
    club = service.create_club("Clube", db_path=p)
    team = service.create_team(club, "Sub-10 A", "Sub-10", "2025/2026", db_path=p)
    joao = service.create_player("João", team, sex="M", joined_on="2025-09-01", db_path=p)
    rui = service.create_player("Rui", team, sex="M", joined_on="2025-09-01", db_path=p)
    coach = auth.get_user(auth.create_user("trein", "Treinador", "coach", "palavra-passe-1", db_path=p), p)
    auth.set_team_assignments(coach["id"], [team], p)
    pai = auth.create_user("pai.joao", "Pai do João", "guardian", "palavra-passe-1", db_path=p)
    auth.set_guardian_links(pai, [joao], p)
    for pid in (joao, rui):
        access.create_evaluation(coach, pid, "2025-09-15", "Avaliação Inicial", flat(2), general_notes="INTERNA", db_path=p)
        access.create_evaluation(coach, pid, "2025-12-15", "1.º Período", flat(3, dribbling=4), db_path=p)

    user = auth.authenticate("pai.joao", "palavra-passe-1", db_path=p)                   # início de sessão
    assert [k["name"] for k in access.guardian_children(user, p)] == ["João"]
    evals = access.guardian_evaluations(user, joao, p)
    assert len(evals) == 2 and "general_notes" not in evals[0]
    report = access.parent_report(user, evals[-1]["id"], p)
    assert "O João apresentou uma evolução positiva" in report["sections"]["evolution"]["text"]
    assert access.export_parent_pdf(user, evals[-1]["id"], p)[0].startswith(b"%PDF")
    for forbidden in (lambda: access.guardian_evaluations(user, rui, p), lambda: access.team_report(user, team, p),
                      lambda: access.search_players(user, db_path=p), lambda: access.individual_report(user, evals[-1]["id"], p)):
        try:
            forbidden()
        except access.PermissionDenied:
            continue
        raise AssertionError("o encarregado de educação não devia ter acesso")
