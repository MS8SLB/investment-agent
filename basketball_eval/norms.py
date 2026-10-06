"""Valores de referência (normas) — carregamento explícito, nunca automático.

Tabela transcrita do documento ENB «Avaliação Quantitativa», pág. 6:
«Tabela comparativa em segundos (Matulaitis, K. et al., 2019)», Defensive Movement Test,
idades 8–17. O documento não indica o sexo da amostra → sex = NULL. As colunas
'90 >' e '< 10' são guardadas como percentis 90 e 10 (limites tal como reportados).

Lógica: menor tempo = melhor. Só se devolve a posição face à tabela (percentil);
nenhum rótulo («Bom», «Excelente»…) é atribuído.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from .db import connect, init_db
from .defensive_movement import TEST_KEY

SOURCE = "Matulaitis et al., 2019 (tabela do documento ENB)"
AGES = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
PERCENTILES = (90, 80, 70, 60, 50, 40, 30, 20, 10)   # 10 = linha «< 10»
TABLE = {
    90: (9.67, 9.59, 8.88, 9.0, 8.3, 7.81, 7.72, 7.27, 7.3, 7.4),
    80: (9.69, 9.66, 9.04, 9.2, 8.7, 8.04, 8.09, 7.7, 7.42, 7.7),
    70: (10.3, 9.94, 9.32, 9.4, 8.9, 8.25, 8.2, 7.54, 7.53, 7.8),
    60: (10.32, 10.15, 9.63, 9.6, 9.03, 8.4, 8.28, 7.73, 7.7, 8.0),
    50: (10.43, 10.36, 9.84, 9.7, 9.26, 8.66, 8.4, 7.9, 7.83, 8.11),
    40: (10.5, 10.57, 9.94, 9.8, 9.4, 8.97, 8.67, 8.09, 7.9, 8.2),
    30: (10.79, 10.69, 10.05, 10.03, 9.5, 9.3, 8.9, 8.16, 8.01, 8.3),
    20: (11.08, 10.78, 10.39, 10.3, 9.79, 9.6, 9.17, 8.4, 8.2, 8.61),
    10: (11.36, 11.09, 10.92, 10.6, 10.3, 10.2, 9.6, 8.78, 8.49, 8.7),
}


def load_matulaitis_2019(db_path: str | None = None, replace: bool = True) -> int:
    """Insere a tabela em reference_norms (substitui a mesma fonte). Devolve nº de linhas."""
    init_db(db_path)
    rows = [(TEST_KEY, SOURCE, None, age, None, pct, TABLE[pct][i], None)
            for pct in PERCENTILES for i, age in enumerate(AGES)]
    with connect(db_path) as c:
        if replace:
            c.execute("DELETE FROM reference_norms WHERE test_key=? AND source=?", (TEST_KEY, SOURCE))
        c.executemany(
            "INSERT INTO reference_norms(test_key, source, category, age, sex, percentile, threshold, level)"
            " VALUES (?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def age_at(birth_date, on_date) -> int:
    b, d = date.fromisoformat(str(birth_date)), date.fromisoformat(str(on_date))
    return d.year - b.year - ((d.month, d.day) < (b.month, b.day))


def reference_position(time_s: float, age: int, sex: Optional[str] = None,
                       db_path: str | None = None) -> Optional[dict]:
    """Posição face à tabela de referência, ou None se não houver norma para essa idade/sexo.

    Devolve o maior percentil da tabela cujo limite é >= tempo (menor tempo = melhor).
    Linhas com sex NULL aplicam-se a ambos os sexos.
    """
    init_db(db_path)
    with connect(db_path) as c:
        rows = c.execute(
            """SELECT percentile, threshold, source FROM reference_norms
               WHERE test_key=? AND age=? AND (sex IS NULL OR sex=?)
               ORDER BY percentile DESC""", (TEST_KEY, age, sex)).fetchall()
    if not rows:
        return None
    ok = [r for r in rows if time_s <= r["threshold"]]
    if not ok:
        return {"percentile": None, "note": f"Tempo acima do limite mais lento da tabela (<{rows[-1]['percentile']})",
                "source": rows[0]["source"], "age": age}
    top = ok[0]
    return {"percentile": top["percentile"], "note": f"Tempo ≤ limite do percentil {top['percentile']} ({top['threshold']:.2f} s)",
            "source": top["source"], "age": age}


def percentile_score(time_s: float, age: int, sex: Optional[str] = None,
                     db_path: str | None = None) -> Optional[dict]:
    """Nota 0–100 por interpolação linear entre os limites da tabela (menor tempo = melhor).

    É uma ESTIMATIVA: a tabela só dá os percentis 10…90. Fora desse intervalo o valor fica
    em 90 (``bound='above'``, tempo mais rápido que o limite P90) ou 10 (``bound='below'``,
    mais lento que o limite «<10»); nada é extrapolado. None se não houver norma.
    """
    init_db(db_path)
    with connect(db_path) as c:
        rows = c.execute(
            """SELECT percentile, MIN(threshold) AS threshold FROM reference_norms
               WHERE test_key=? AND age=? AND (sex IS NULL OR sex=?) GROUP BY percentile""",
            (TEST_KEY, age, sex)).fetchall()
    pts = sorted((r["threshold"], r["percentile"]) for r in rows)   # tempo crescente, percentil decrescente
    if not pts:
        return None
    if time_s <= pts[0][0]:
        return {"score": pts[0][1], "bound": "above", "age": age}
    if time_s >= pts[-1][0]:
        return {"score": pts[-1][1], "bound": "below" if time_s > pts[-1][0] else None, "age": age}
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t0 <= time_s <= t1:
            frac = 0.0 if t1 == t0 else (time_s - t0) / (t1 - t0)
            return {"score": p0 + frac * (p1 - p0), "bound": None, "age": age}


def main() -> None:
    n = load_matulaitis_2019()
    print(f"{n} valores de referência carregados ({SOURCE}).")


if __name__ == "__main__":
    main()
