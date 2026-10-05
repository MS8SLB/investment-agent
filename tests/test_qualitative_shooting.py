"""Testes da avaliação qualitativa — Técnica Individual / Lançamento."""

import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from basketball_eval import competencies as comps
from basketball_eval import qual_db, qual_service as svc
from basketball_eval import qualitative as q
from basketball_eval import service as base
from basketball_eval.db import connect

L = comps.LANCAMENTO


def scores_for(**dim_scores):
    """Pontua todos os critérios de cada dimensão com o mesmo valor: preparacao=4, ..."""
    return {c.key: dim_scores[d.key] for d in L.dimensions if d.key in dim_scores for c in d.criteria}


SPEC = scores_for(preparacao=4, execucao=3, finalizacao=4, consistencia=3, aplicacao_jogo=3)


PG_URL = os.environ.get("TEST_DATABASE_URL")     # ex.: postgresql://user@localhost:5432/teste (BD descartável!)


@pytest.fixture(params=["sqlite"] + (["postgres"] if PG_URL else []))
def db(request, tmp_path):
    """Corre os testes de armazenamento em SQLite e, se TEST_DATABASE_URL existir, também em Postgres."""
    if request.param == "postgres":
        from basketball_eval import db as dbmod
        with connect(PG_URL) as c:                  # esquema limpo em cada teste
            c.execute("DROP SCHEMA public CASCADE")
            c.execute("CREATE SCHEMA public")
        dbmod._PG_READY.clear()
        path = PG_URL
    else:
        path = str(tmp_path / "q.db")
    qual_db.init(path)
    return path


@pytest.fixture
def player(db):
    return base.add_player("Ana", "Sub-10", team="Águias", db_path=db)


# ── Estrutura / escala ──────────────────────────────────────────────────────
def test_criteria_cover_spec():
    assert [d.name for d in L.dimensions] == ["Preparação", "Execução", "Finalização", "Consistência", "Aplicação no jogo"]
    assert [len(d.criteria) for d in L.dimensions] == [5, 6, 5, 3, 6]
    keys = [c.key for c in L.criteria]
    assert len(keys) == len(set(keys))


def test_level_labels_always_number_and_text():
    assert [q.level_label(i) for i in range(1, 6)] == [
        "1 – Inicial", "2 – Em desenvolvimento", "3 – Adequado", "4 – Bom", "5 – Muito bom"]
    with pytest.raises(ValueError):
        q.level_label(6)


# ── Média e classificação ───────────────────────────────────────────────────
def test_spec_example_mean_3_4_adequado():
    s = q.summarize(L, SPEC)
    assert s.mean == pytest.approx(3.4)
    assert q.fmt_score(s.mean) == "3,4"
    assert s.level == 3 and s.label == "Adequado"
    assert s.dimension_means["preparacao"] == 4 and s.dimension_means["consistencia"] == 3


def test_dimension_mean_uses_only_assessed_criteria():
    s = {"equilibrio_corporal": 5, "posicao_pes": 3}      # restantes não avaliados
    assert q.dimension_means(L, s)["preparacao"] == 4
    assert q.dimension_means(L, s)["execucao"] is None
    assert q.overall_mean(L, s) == 4                        # só entra a dimensão avaliada


def test_nothing_assessed_gives_none():
    s = q.summarize(L, {})
    assert s.mean is None and s.label is None and s.level is None


@pytest.mark.parametrize("mean,level", [(1.0, 1), (1.4, 1), (1.5, 2), (2.4, 2), (2.5, 3), (3.4, 3),
                                         (3.5, 4), (3.8, 4), (4.5, 5), (5.0, 5)])
def test_classification_boundaries(mean, level):
    assert q.level_for_mean(mean) == level
    assert q.classify(mean) == comps.LEVELS[level]


def test_rounding_is_half_up_not_bankers():
    assert q.r1(2.25) == Decimal("2.3") and q.r1(0.35) == Decimal("0.4")
    # 3,46 apresenta-se «3,5» → classificação coerente com o que se vê (Bom)
    assert q.fmt_score(3.46) == "3,5" and q.level_for_mean(3.46) == 4


def test_fmt_score_portuguese_formats():
    assert q.fmt_score(3.4) == "3,4" and q.fmt_score(4, compact=True) == "4" and q.fmt_score(4) == "4,0"
    assert q.fmt_score(1.3, signed=True) == "+1,3" and q.fmt_score(-0.2, signed=True) == "-0,2"
    assert q.fmt_score(None) == "—"


@pytest.mark.parametrize("bad", [0, 6, -1, 3.5, "3", True])
def test_invalid_scores_rejected(bad):
    with pytest.raises(ValueError):
        q.validate_scores(L, {"posicao_pes": bad})
    with pytest.raises(ValueError):
        q.validate_scores(L, {"nao_existe": 3})


# ── Pontos fortes / melhoria ────────────────────────────────────────────────
def test_insights_only_from_scores():
    s = {"equilibrio_corporal": 5, "posicao_pes": 4, "estabilidade_corporal": 3,
         "repetir_gesto": 2, "sob_oposicao": 1}
    ins = q.insights(L, s)
    assert [i.criterion for i in ins.strengths] == ["Equilíbrio corporal", "Posição dos pés"]
    assert [i.criterion for i in ins.improvements] == ["Execução sob oposição", "Capacidade de repetir o gesto"]
    assert ins.strengths[0].text == "Equilíbrio corporal (5 – Muito bom)"


def test_insights_empty_when_all_adequate():
    ins = q.insights(L, scores_for(preparacao=3, execucao=3))
    assert not ins.strengths and not ins.improvements


# ── Evolução ────────────────────────────────────────────────────────────────
def test_evolution_spec_example():
    ev = q.evolution([("2025-09-10", 2.6), ("2025-10-15", 3.1), ("2025-11-20", 3.5), ("2026-01-15", 3.9)])
    assert (ev.first, ev.last, ev.change) == (Decimal("2.6"), Decimal("3.9"), Decimal("1.3"))
    assert ev.change_pct == Decimal("50.0") and ev.trend == q.POSITIVE and ev.n == 4
    assert ev.trend_label == "EVOLUÇÃO POSITIVA"


def test_evolution_single_point_has_no_percentage_or_trend():
    ev = q.evolution([("2025-09-10", 3.0)])
    assert ev.change == 0 and ev.change_pct is None and ev.trend == q.STABLE
    assert q.evolution([]) is None


def test_evolution_negative_and_stable():
    assert q.evolution([("a", 3.5), ("b", 3.0)]).trend == q.NEGATIVE
    assert q.evolution([("a", 3.0), ("b", 3.0)]).trend == q.STABLE


# ── Comparação ──────────────────────────────────────────────────────────────
def test_compare_spec_example():
    before = scores_for(preparacao=2, execucao=3, finalizacao=2, consistencia=3, aplicacao_jogo=2)
    after = scores_for(preparacao=4, execucao=4, finalizacao=4, consistencia=3, aplicacao_jogo=4)
    cmp = q.compare(L, before, after)
    assert [(r.label, r.before, r.after) for r in cmp.dimensions] == [
        ("Preparação", 2, 4), ("Execução", 3, 4), ("Finalização", 2, 4), ("Consistência", 3, 3),
        ("Aplicação no jogo", 2, 4)]
    assert (q.fmt_score(cmp.overall.before), q.fmt_score(cmp.overall.after)) == ("2,4", "3,8")
    assert cmp.trend == q.POSITIVE and cmp.trend_label == "EVOLUÇÃO POSITIVA"
    assert cmp.overall.delta == Decimal("1.4")


def test_compare_criteria_only_when_assessed_in_both():
    cmp = q.compare(L, {"posicao_pes": 2, "flexao_pulso": 3}, {"posicao_pes": 4})
    assert [(r.key, r.before, r.after) for r in cmp.criteria] == [("posicao_pes", 2, 4)]
    assert cmp.dimensions[2].delta is None            # Finalização só numa das avaliações


# ── Armazenamento / recuperação ─────────────────────────────────────────────
def test_tables_created_and_age_groups_seeded(db):
    assert svc.list_age_groups(db) == ["Sub-8", "Sub-10", "Sub-12"]
    svc.add_age_group("Sub-14", db)
    assert svc.list_age_groups(db)[-1] == "Sub-14"
    with connect(db) as c:
        names = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'" if db.startswith("postgres")
            else "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"players", "teams", "coaches", "evaluations", "evaluation_items"} <= names


def test_save_and_retrieve_roundtrip(db, player):
    eid = svc.save_evaluation("lancamento", player, "2025-10-15", SPEC, "  Boa evolução  ", coach="Rui", db_path=db)
    ev = svc.get_evaluation(eid, db)
    assert ev["scores"] == SPEC and len(ev["scores"]) == 25      # dados originais intactos
    assert ev["player_name"] == "Ana" and ev["age_group"] == "Sub-10" and ev["team_name"] == "Águias"
    assert ev["coach_name"] == "Rui" and ev["observations"] == "Boa evolução"
    assert ev["evaluation_date"] == "2025-10-15" and ev["created_at"] and ev["updated_at"]
    assert ev["mean"] == pytest.approx(3.4) and ev["label"] == "Adequado"
    with connect(db) as c:
        cats = {r[0] for r in c.execute("SELECT DISTINCT category FROM evaluation_items")}
    assert cats == {d.key for d in L.dimensions}


def test_partial_assessment_stores_only_assessed(db, player):
    eid = svc.save_evaluation("lancamento", player, "2025-10-15", {"posicao_pes": 4, "flexao_pulso": None}, db_path=db)
    assert svc.get_evaluation(eid, db)["scores"] == {"posicao_pes": 4}


def test_save_validation(db, player):
    with pytest.raises(ValueError, match="pelo menos um"):
        svc.save_evaluation("lancamento", player, "2025-10-15", {}, db_path=db)
    with pytest.raises(ValueError):
        svc.save_evaluation("lancamento", player, "2025-10-15", {"posicao_pes": 9}, db_path=db)
    with pytest.raises(ValueError):
        svc.save_evaluation("lancamento", 999, "2025-10-15", SPEC, db_path=db)
    with pytest.raises(ValueError):
        svc.save_evaluation("drible", player, "2025-10-15", SPEC, db_path=db)
    with pytest.raises(ValueError):
        svc.save_evaluation("lancamento", player, "2025-10-15", SPEC, age_group="Sub-99", db_path=db)
    assert svc.list_evaluations(player, "lancamento", db) == []


def test_update_keeps_created_at_and_replaces_items(db, player):
    eid = svc.save_evaluation("lancamento", player, "2025-10-15", SPEC, "a", db_path=db)
    with connect(db) as c:
        c.execute("UPDATE evaluations SET updated_at='2000-01-01 00:00:00' WHERE id=?", (eid,))
    svc.update_evaluation(eid, "2025-10-16", {"posicao_pes": 5}, "b", db_path=db)
    ev = svc.get_evaluation(eid, db)
    assert ev["scores"] == {"posicao_pes": 5} and ev["observations"] == "b" and ev["evaluation_date"] == "2025-10-16"
    assert ev["updated_at"] != "2000-01-01 00:00:00"


def test_delete_requires_confirmation(db, player):
    eid = svc.save_evaluation("lancamento", player, "2025-10-15", SPEC, db_path=db)
    with pytest.raises(ValueError, match="Confirmação"):
        svc.delete_evaluation(eid, db_path=db)
    assert svc.get_evaluation(eid, db) is not None
    svc.delete_evaluation(eid, confirm=True, db_path=db)
    assert svc.get_evaluation(eid, db) is None
    with connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM evaluation_items").fetchone()[0] == 0


def test_history_chronological_and_isolated_per_player(db, player):
    other = base.add_player("Rui", "Sub-10", db_path=db)
    for d, v in [("2025-11-20", 4), ("2025-09-10", 2), ("2025-10-15", 3)]:
        svc.save_evaluation("lancamento", player, d, scores_for(preparacao=v), db_path=db)
    svc.save_evaluation("lancamento", other, "2025-10-01", scores_for(preparacao=5), db_path=db)
    evs = svc.list_evaluations(player, "lancamento", db)
    assert [e["evaluation_date"] for e in evs] == ["2025-09-10", "2025-10-15", "2025-11-20"]
    assert [e["mean"] for e in evs] == [2, 3, 4]
    ev = svc.player_evolution(player, "lancamento", db)
    assert (ev.first, ev.last, ev.change, ev.n) == (Decimal("2.0"), Decimal("4.0"), Decimal("2.0"), 3)


def test_compare_evaluations_orders_by_date_and_blocks_other_players(db, player):
    other = base.add_player("Rui", "Sub-10", db_path=db)
    old = svc.save_evaluation("lancamento", player, "2025-09-10", scores_for(preparacao=2), db_path=db)
    new = svc.save_evaluation("lancamento", player, "2026-01-15", scores_for(preparacao=4), db_path=db)
    foreign = svc.save_evaluation("lancamento", other, "2026-01-15", scores_for(preparacao=5), db_path=db)
    a, b, cmp = svc.compare_evaluations(new, old, db)           # ordem trocada de propósito
    assert (a["id"], b["id"]) == (old, new) and cmp.trend == q.POSITIVE
    with pytest.raises(ValueError, match="mesmo jogador"):
        svc.compare_evaluations(old, foreign, db)


def test_context_helpers(db):
    base.add_player("Ana", "Sub-10", team="Águias", db_path=db)
    base.add_player("Bia", "Sub-10", team="Lobos", db_path=db)
    base.add_player("Cris", "Sub-12", team="Águias", db_path=db)
    assert svc.list_teams(db) == ["Águias", "Lobos"]
    assert [p["name"] for p in svc.players_for("Sub-10", "Águias", db)] == ["Ana"]
    assert [p["name"] for p in svc.players_for("Sub-10", db_path=db)] == ["Ana", "Bia"]
    assert svc.get_or_create_team("Águias", db) == svc.get_or_create_team(" Águias ", db)


def test_existing_defensive_module_untouched(db, player):
    """O módulo quantitativo continua a funcionar na mesma base de dados."""
    tid = base.save_test(player, "2025-10-15", ["12.8", "12.3", "12.5"], db_path=db)
    assert base.history(player, db)[0]["id"] == tid


# ── Relatório ───────────────────────────────────────────────────────────────
def test_report_contains_everything_and_escapes_html(db, player):
    from basketball_eval import report as rep
    svc.save_evaluation("lancamento", player, "2025-09-10", scores_for(preparacao=2, execucao=2), db_path=db)
    eid = svc.save_evaluation("lancamento", player, "2026-01-15", SPEC, "Bom <b>trabalho</b>", coach="Rui", db_path=db)
    data = rep.build_report(eid, db)
    assert (data.player, data.team, data.age_group, data.coach) == ("Ana", "Águias", "Sub-10", "Rui")
    assert data.mean_display == "3,4" and data.label == "Adequado" and len(data.history) == 2
    assert data.evolution.trend == q.POSITIVE
    html = rep.render_html(data)
    for needle in ["RELATÓRIO DE AVALIAÇÃO", "TÉCNICA INDIVIDUAL – LANÇAMENTO", "3,4 / 5", "ADEQUADO", "<svg",
                   "Pontos fortes", "Áreas de melhoria", "4 – Bom", "15/01/2026", "2,0 → 3,4"]:
        assert needle in html, needle
    assert "<b>trabalho</b>" not in html and "&lt;b&gt;trabalho" in html


def test_report_only_includes_history_up_to_that_evaluation(db, player):
    from basketball_eval import report as rep
    first = svc.save_evaluation("lancamento", player, "2025-09-10", scores_for(preparacao=2), db_path=db)
    svc.save_evaluation("lancamento", player, "2026-01-15", scores_for(preparacao=4), db_path=db)
    data = rep.build_report(first, db)
    assert len(data.history) == 1 and "ainda sem evolução" in rep.render_html(data)


def test_radar_handles_missing_axes():
    from basketball_eval import report as rep
    assert rep.radar_svg(["A", "B", "C", "D", "E"], [4, None, 3, None, 5]).count("<circle") == 3
    assert "<svg" in rep.radar_svg(["A", "B", "C", "D", "E"], [None] * 5)
