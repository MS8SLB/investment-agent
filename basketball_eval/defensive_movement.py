"""Teste de Movimentos Defensivos (Defensive Movement Test — Johnson & Nelson, 1986).

Teste TEMPORAL: resultado bruto em segundos, MENOR TEMPO = MELHOR DESEMPENHO.
Este módulo é lógica pura (sem I/O): validação, melhor tempo, evolução e
mensagens. Nenhuma norma ou classificação é assumida.

Convenção de sinais (conforme especificação):
  change_seconds    = atual - anterior          (negativo = melhoria)
  change_percentage = (anterior - atual)/anterior * 100   (positivo = melhoria)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence

TEST_KEY = "defensive_movement"
TEST_NAME_PT = "Teste de Movimentos Defensivos"
TEST_NAME_EN = "Defensive Movement Test"
REFERENCE = "Johnson & Nelson (1986)"
UNIT = "s"
LOWER_IS_BETTER = True
CATEGORIES = ("Sub-8", "Sub-10", "Sub-12", "Sub-14")
N_TRIALS = 3

IMPROVED, UNCHANGED, WORSENED = "melhorou", "manteve", "piorou"
MESSAGES = {
    IMPROVED: "Melhoria do desempenho relativamente à avaliação anterior.",
    UNCHANGED: "Desempenho sem alteração relativamente à avaliação anterior.",
    WORSENED: "Pioria do desempenho relativamente à avaliação anterior.",
}
LABELS = {IMPROVED: "Melhorou", UNCHANGED: "Manteve", WORSENED: "Piorou"}


class InvalidTimeError(ValueError):
    pass


def parse_time(value) -> Optional[Decimal]:
    """Converte um tempo em segundos para Decimal.

    Aceita número ou texto numérico ('12.35' ou '12,35'). Vazio/None → None
    (tentativa não realizada). Texto não numérico, negativo, zero, NaN ou
    infinito → InvalidTimeError.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidTimeError("Tempo inválido.")
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if text == "":
            return None
        if text.lower().endswith("s"):
            text = text[:-1].strip()
        try:
            d = Decimal(text)
        except InvalidOperation:
            raise InvalidTimeError(f"O tempo deve ser numérico (recebido: {value!r}).")
    else:
        try:
            d = Decimal(str(value))
        except InvalidOperation:
            raise InvalidTimeError(f"O tempo deve ser numérico (recebido: {value!r}).")
    if not d.is_finite():
        raise InvalidTimeError("O tempo deve ser um número finito.")
    if d < 0:
        raise InvalidTimeError("O tempo não pode ser negativo.")
    if d == 0:
        raise InvalidTimeError("O tempo deve ser superior a zero.")
    return d


@dataclass
class Trial:
    time: Optional[Decimal] = None   # None = não realizada
    valid: bool = True               # False = protocolo invalidado


@dataclass
class TrialResult:
    best_time: Optional[Decimal]
    best_trial: Optional[int]        # 1..3
    counted_trials: list[int] = field(default_factory=list)


def compute_best(trials: Sequence[Trial], first_is_practice: bool = False) -> TrialResult:
    """Melhor tempo = MENOR tempo entre tentativas válidas (e contabilizadas).

    Tentativas inválidas ou vazias não entram. Se a 1.ª for de familiarização,
    é guardada mas não conta. Empate → a tentativa mais cedo.
    """
    counted = [
        (i + 1, t.time)
        for i, t in enumerate(trials)
        if t.time is not None and t.valid and not (first_is_practice and i == 0)
    ]
    if not counted:
        return TrialResult(None, None, [])
    best_trial, best_time = min(counted, key=lambda x: (x[1], x[0]))
    return TrialResult(best_time, best_trial, [n for n, _ in counted])


@dataclass
class Evolution:
    status: str                       # IMPROVED / UNCHANGED / WORSENED
    change_seconds: Decimal           # atual - anterior
    change_percentage: Decimal        # (anterior - atual)/anterior*100

    @property
    def label(self) -> str:
        return LABELS[self.status]

    @property
    def message(self) -> str:
        return MESSAGES[self.status]


def compute_evolution(current: Optional[Decimal], previous: Optional[Decimal]) -> Optional[Evolution]:
    """Compara com a avaliação anterior. None se faltar um dos tempos."""
    if current is None or previous is None:
        return None
    if previous <= 0:
        raise InvalidTimeError("O tempo anterior deve ser superior a zero.")
    change = current - previous
    pct = (previous - current) / previous * 100
    status = IMPROVED if change < 0 else WORSENED if change > 0 else UNCHANGED
    return Evolution(status, change, pct)


def r2(x: Optional[Decimal | float]) -> Optional[float]:
    """Arredonda a 2 casas decimais só para apresentação/armazenamento."""
    if x is None:
        return None
    return float(round(Decimal(str(x)), 2))


def fmt_seconds(x, signed: bool = False) -> str:
    """12.31 → '12,31 s' (vírgula decimal, formato PT)."""
    if x is None:
        return "—"
    s = f"{float(round(Decimal(str(x)), 2)):+.2f}" if signed else f"{float(round(Decimal(str(x)), 2)):.2f}"
    return s.replace(".", ",") + " s"


def fmt_pct(x, signed: bool = False) -> str:
    if x is None:
        return "—"
    s = f"{float(round(Decimal(str(x)), 2)):+.2f}" if signed else f"{float(round(Decimal(str(x)), 2)):.2f}"
    return s.replace(".", ",") + " %"


# ── Evolução da equipa ──────────────────────────────────────────────────────

@dataclass
class GroupStats:
    n: int
    mean: Optional[float]
    median: Optional[float]
    best: Optional[float]    # menor tempo
    worst: Optional[float]   # maior tempo


def group_stats(best_times: Sequence[float]) -> GroupStats:
    """Estatísticas de um grupo (1 melhor tempo por jogador). Melhor = mínimo."""
    xs = [float(x) for x in best_times if x is not None and math.isfinite(float(x))]
    if not xs:
        return GroupStats(0, None, None, None, None)
    return GroupStats(len(xs), statistics.fmean(xs), statistics.median(xs), min(xs), max(xs))


@dataclass
class TeamComparison:
    before: GroupStats
    after: GroupStats
    mean_change_seconds: Optional[float]      # depois - antes (negativo = melhoria)
    median_change_seconds: Optional[float]
    paired_n: int                             # jogadores avaliados nas duas datas
    paired_mean_change_seconds: Optional[float]
    paired_improved: int
    paired_unchanged: int
    paired_worsened: int


def compare_groups(before: dict[int, float], after: dict[int, float]) -> TeamComparison:
    """Compara dois momentos {player_id: melhor_tempo}. Os deltas por jogador só
    usam jogadores presentes nos dois momentos."""
    b, a = group_stats(list(before.values())), group_stats(list(after.values()))
    both = sorted(set(before) & set(after))
    deltas = [after[p] - before[p] for p in both]
    return TeamComparison(
        before=b,
        after=a,
        mean_change_seconds=None if a.mean is None or b.mean is None else a.mean - b.mean,
        median_change_seconds=None if a.median is None or b.median is None else a.median - b.median,
        paired_n=len(both),
        paired_mean_change_seconds=statistics.fmean(deltas) if deltas else None,
        paired_improved=sum(d < 0 for d in deltas),
        paired_unchanged=sum(d == 0 for d in deltas),
        paired_worsened=sum(d > 0 for d in deltas),
    )


# ── Referências normativas (opcionais, configuráveis) ───────────────────────

def percentile_band(time_s: float, rows: Sequence[tuple[float, float]]) -> Optional[float]:
    """Percentil a partir de uma tabela de referência fornecida pelo projeto.

    rows = [(percentil, tempo_limite_s)], com a lógica 'menor tempo = melhor':
    devolve o MAIOR percentil cujo limite é >= tempo. Sem tabela → None (nenhuma
    classificação é inventada). Tempo pior que todos os limites → None.
    """
    if not rows:
        return None
    ok = [p for p, limit in rows if time_s <= limit]
    return max(ok) if ok else None
