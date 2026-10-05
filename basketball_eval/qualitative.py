"""Lógica pura da avaliação qualitativa 1–5 (sem I/O).

Princípios:
  * Só se usam as pontuações introduzidas pelo treinador; nada é inventado.
  * Um critério sem pontuação é «não avaliado» e fica fora de qualquer média.
  * A média por dimensão é a média dos critérios avaliados dessa dimensão;
    a média técnica é a média das dimensões avaliadas (cada dimensão pesa o mesmo).
  * A classificação é o nível 1–5 mais próximo da média apresentada (1 casa
    decimal, arredondamento «metade para cima»): 3,4 → Adequado; 3,5 → Bom.
  * Sem percentis, rankings ou normas: é uma síntese descritiva do treinador.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from typing import Mapping, Optional, Sequence

from .competencies import LEVELS, SCALE_MAX, SCALE_MIN, Competency

STRENGTH_MIN = 4      # critérios com 4–5 → ponto forte
IMPROVEMENT_MAX = 2   # critérios com 1–2 → área de melhoria

POSITIVE, STABLE, NEGATIVE = "positiva", "sem alteração", "negativa"
TREND_LABELS = {POSITIVE: "EVOLUÇÃO POSITIVA", STABLE: "SEM ALTERAÇÃO", NEGATIVE: "VARIAÇÃO NEGATIVA"}


# ── Escala ──────────────────────────────────────────────────────────────────

def level_label(score: int) -> str:
    """«3 – Adequado» — número e descrição sempre juntos."""
    if score not in LEVELS:
        raise ValueError(f"Pontuação fora da escala {SCALE_MIN}–{SCALE_MAX}: {score!r}")
    return f"{score} – {LEVELS[score]}"


def validate_scores(comp: Competency, scores: Mapping[str, Optional[int]]) -> dict[str, int]:
    """Devolve só os critérios avaliados; rejeita chaves desconhecidas e valores inválidos."""
    known = {c.key for c in comp.criteria}
    clean: dict[str, int] = {}
    for key, val in scores.items():
        if key not in known:
            raise ValueError(f"Critério desconhecido: {key!r}")
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, int) or not SCALE_MIN <= val <= SCALE_MAX:
            raise ValueError(f"Pontuação inválida para {key!r}: {val!r} (esperado {SCALE_MIN}–{SCALE_MAX})")
        clean[key] = val
    return clean


# ── Arredondamento / formatação ─────────────────────────────────────────────

def r1(x: float | Fraction) -> Decimal:
    """1 casa decimal, metade para cima (evita o arredondamento «ao par» do Python)."""
    d = Decimal(x.numerator) / Decimal(x.denominator) if isinstance(x, Fraction) else Decimal(repr(round(x, 9)))
    return d.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def fmt_score(x: Optional[float | Decimal | Fraction], compact: bool = False, signed: bool = False) -> str:
    """Vírgula decimal portuguesa. compact: inteiros sem casa decimal («4»)."""
    if x is None:
        return "—"
    d = x if isinstance(x, Decimal) else r1(x)
    if compact and d == d.to_integral_value():
        s = f"{int(d)}"
    else:
        s = f"{d:.1f}"
    if signed and d > 0:
        s = "+" + s
    return s.replace(".", ",")


def level_for_mean(mean: float | Fraction) -> int:
    """Nível 1–5 mais próximo da média (a partir do valor apresentado, 1 casa decimal)."""
    lvl = int(Decimal(r1(mean)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return min(SCALE_MAX, max(SCALE_MIN, lvl))


def classify(mean: Optional[float | Fraction]) -> Optional[str]:
    return None if mean is None else LEVELS[level_for_mean(mean)]


# ── Médias ──────────────────────────────────────────────────────────────────

def _frac_dimension_means(comp: Competency, scores: Mapping[str, int]) -> dict[str, Optional[Fraction]]:
    out: dict[str, Optional[Fraction]] = {}
    for d in comp.dimensions:
        vals = [scores[c.key] for c in d.criteria if c.key in scores]
        out[d.key] = Fraction(sum(vals), len(vals)) if vals else None
    return out


def dimension_means(comp: Competency, scores: Mapping[str, int]) -> dict[str, Optional[float]]:
    return {k: (None if v is None else float(v)) for k, v in _frac_dimension_means(comp, scores).items()}


def overall_mean(comp: Competency, scores: Mapping[str, int]) -> Optional[float]:
    """Média Técnica: média das dimensões avaliadas. None se nada foi avaliado."""
    dims = [v for v in _frac_dimension_means(comp, scores).values() if v is not None]
    return float(sum(dims) / len(dims)) if dims else None


@dataclass
class Summary:
    dimension_means: dict[str, Optional[float]]
    assessed: dict[str, tuple[int, int]]       # dimensão → (avaliados, total)
    mean: Optional[float]
    mean_display: Optional[Decimal]            # 1 casa decimal
    level: Optional[int]
    label: Optional[str]                       # «Adequado»


def summarize(comp: Competency, scores: Mapping[str, int]) -> Summary:
    scores = validate_scores(comp, scores)
    mean = overall_mean(comp, scores)
    return Summary(
        dimension_means=dimension_means(comp, scores),
        assessed={d.key: (sum(c.key in scores for c in d.criteria), len(d.criteria)) for d in comp.dimensions},
        mean=mean,
        mean_display=None if mean is None else r1(mean),
        level=None if mean is None else level_for_mean(mean),
        label=classify(mean),
    )


# ── Pontos fortes e áreas de melhoria ───────────────────────────────────────

@dataclass
class Insight:
    dimension: str
    criterion: str
    score: int

    @property
    def text(self) -> str:
        return f"{self.criterion} ({level_label(self.score)})"


@dataclass
class Insights:
    strengths: list[Insight] = field(default_factory=list)
    improvements: list[Insight] = field(default_factory=list)


def insights(comp: Competency, scores: Mapping[str, int]) -> Insights:
    """Derivado só das pontuações: 4–5 → ponto forte; 1–2 → área de melhoria; 3 → neutro."""
    scores = validate_scores(comp, scores)
    res = Insights()
    for d in comp.dimensions:
        for c in d.criteria:
            s = scores.get(c.key)
            if s is None:
                continue
            item = Insight(d.name, c.label, s)
            if s >= STRENGTH_MIN:
                res.strengths.append(item)
            elif s <= IMPROVEMENT_MAX:
                res.improvements.append(item)
    res.strengths.sort(key=lambda i: -i.score)
    res.improvements.sort(key=lambda i: i.score)
    return res


# ── Evolução ────────────────────────────────────────────────────────────────

@dataclass
class Evolution:
    n: int
    first_date: str
    last_date: str
    first: Decimal
    last: Decimal
    change: Decimal                  # pontos (valores apresentados: último − primeiro)
    change_pct: Optional[Decimal]    # só com ≥2 avaliações e primeira > 0
    trend: str

    @property
    def trend_label(self) -> str:
        return TREND_LABELS[self.trend]


def _trend(change: Decimal) -> str:
    return POSITIVE if change > 0 else NEGATIVE if change < 0 else STABLE


def evolution(points: Sequence[tuple[str, float]]) -> Optional[Evolution]:
    """points = [(data ISO, média)], por ordem cronológica. None se não há pontos."""
    if not points:
        return None
    (d0, m0), (d1, m1) = points[0], points[-1]
    first, last = r1(m0), r1(m1)
    change = last - first
    pct = None
    if len(points) >= 2 and first > 0:
        pct = (change / first * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return Evolution(len(points), d0, d1, first, last, change, pct, _trend(change) if len(points) >= 2 else STABLE)


# ── Comparação entre duas avaliações ────────────────────────────────────────

@dataclass
class Row:
    key: str
    label: str
    before: Optional[float]
    after: Optional[float]

    @property
    def delta(self) -> Optional[Decimal]:
        if self.before is None or self.after is None:
            return None
        return r1(self.after) - r1(self.before)


@dataclass
class Comparison:
    dimensions: list[Row]
    criteria: list[Row]              # só critérios avaliados nas duas avaliações
    overall: Row
    trend: Optional[str]

    @property
    def trend_label(self) -> Optional[str]:
        return None if self.trend is None else TREND_LABELS[self.trend]


def compare(comp: Competency, before: Mapping[str, int], after: Mapping[str, int]) -> Comparison:
    """Compara duas avaliações (dados originais; sem interpretação causal)."""
    before, after = validate_scores(comp, before), validate_scores(comp, after)
    dm_b, dm_a = dimension_means(comp, before), dimension_means(comp, after)
    dims = [Row(d.key, d.name, dm_b[d.key], dm_a[d.key]) for d in comp.dimensions]
    crit = [Row(c.key, c.label, before[c.key], after[c.key])
            for c in comp.criteria if c.key in before and c.key in after]
    overall = Row("media", "Média técnica", overall_mean(comp, before), overall_mean(comp, after))
    return Comparison(dims, crit, overall, None if overall.delta is None else _trend(overall.delta))
