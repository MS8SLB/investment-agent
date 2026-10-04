"""Funções puras de cálculo. Respeitam o sentido de cada teste (menor/maior = melhor).

Nunca se calcula média global entre testes: todas as estatísticas são por teste.
"""
import math
import statistics
from datetime import date


def best_of(values, direction):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return min(vals) if direction == "lower" else max(vals)


def change(first, last, direction):
    """Variação last-first e se representa melhoria (True), piora (False) ou igual (None)."""
    if first is None or last is None:
        return None, None
    delta = last - first
    if abs(delta) < 1e-9:
        return delta, None
    improved = delta < 0 if direction == "lower" else delta > 0
    return delta, improved


def describe(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else None,
    }


def histogram(values, max_bins=8):
    """Distribuição em classes de igual amplitude. Devolve [(rótulo_min, rótulo_max, contagem)]."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return []
    lo, hi = vals[0], vals[-1]
    if hi - lo < 1e-9:
        return [(lo, hi, len(vals))]
    k = max(2, min(max_bins, math.ceil(math.log2(len(vals)) + 1)))
    width = (hi - lo) / k
    bins = [[lo + i * width, lo + (i + 1) * width, 0] for i in range(k)]
    for v in vals:
        idx = min(int((v - lo) / width), k - 1)
        bins[idx][2] += 1
    return [tuple(b) for b in bins]


def age_on(birth: date, on: date) -> int:
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


def classify(value, rows):
    """Classifica contra linhas de uma tabela de referência (min/max inclusivos). None se não encaixa."""
    if value is None:
        return None
    for r in rows:
        lo_ok = r.min_value is None or value >= r.min_value
        hi_ok = r.max_value is None or value <= r.max_value
        if lo_ok and hi_ok:
            return r.label
    return None
