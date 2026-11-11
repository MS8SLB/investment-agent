"""Objetivos individuais do jogador, sugeridos a partir dos dados de avaliação.

Lógica pura (sem I/O). Liga:
  • Avaliação qualitativa da defesa (escala 1–4 por critério técnico) e
  • Teste de Movimentos Defensivos (variação do melhor tempo)
a objetivos defensivos concretos. Os objetivos ofensivos não têm teste associado
neste módulo: ficam pela biblioteca-modelo e passam a ser priorizados pela
avaliação de objetivos (1–5) feita pelo treinador.

A ligação critério → objetivo é uma PROPOSTA de trabalho (como os descritores
de qualitative.py), ajustável em LIBRARY. É um apoio à decisão do treinador,
não um diagnóstico validado.
"""

from __future__ import annotations

import statistics
from typing import Mapping, Optional, Sequence

SIDES = ("off", "def")
SIDE_LABEL = {"off": "Ofensivo", "def": "Defensivo"}
INDICATORS = ("execution", "frequency", "application", "evolution")
INDICATOR_LABEL = {"execution": "Execução", "frequency": "Frequência no treino",
                   "application": "Aplicação no jogo", "evolution": "Evolução"}
SCALE = (1, 2, 3, 4, 5)

# Peso do tempo do teste de movimentos quando o jogador não melhorou/piorou.
MOVEMENT_NEED_STALLED, MOVEMENT_NEED_WORSE = 0.4, 0.7

LIBRARY: dict[str, dict] = {
    "off_scan": {"side": "off", "text": "Olhar antes de receber e identificar vantagens.",
                 "drills": ["3x3 com regra: só pode receber após 'scan'", "Rondo com 1 passe e leitura de ombros"]},
    "off_decision": {"side": "off", "text": "Melhorar a qualidade e velocidade da decisão após receção.",
                     "drills": ["3x2 / 2x1 com contagem de decisões corretas", "Ameaça tripla com leitura do defensor"]},
    "off_gaps": {"side": "off", "text": "Atacar os gaps e criar vantagens através do drible.",
                 "drills": ["1x1 com gap obrigatório", "4x4 só com penetração + kick-out"]},
    "off_pnr": {"side": "off", "text": "Melhorar a tomada de decisão no Pick and Roll.",
                "drills": ["2x2 Pick and Roll com defesa condicionada", "3x3 só com bloqueio direto"]},
    "def_position": {"side": "def", "text": "Manter posição defensiva e controlar o adversário direto.",
                     "criteria": ["posicao_base", "maos_bracos", "cabeca_olhar"],
                     "drills": ["Shell drill 4x4", "1x1 em meia-quadra com ênfase no deslizamento"]},
    "def_closeout": {"side": "def", "text": "Melhorar o Close Out e a contenção do drible.",
                     "criteria": ["deslize_lateral", "drop_step", "velocidade_ritmo"], "movement": True,
                     "drills": ["Close out + 1x1 de 3 segundos", "Contenção de drible em corredor (zig-zag)"]},
    "def_help": {"side": "def", "text": "Identificar ajudas defensivas e recuperar a posição.",
                 "criteria": ["cabeca_olhar", "toque_marcadores", "atitude"],
                 "drills": ["Shell com 'ajuda e recuperação'", "3x3 com pontos extra para recuperações"]},
    "def_rebound": {"side": "def", "text": "Participar ativamente no ressalto defensivo.",
                    "drills": ["Box out 1x1/2x2 até ao ressalto", "4x4 com ressalto defensivo = ponto"]},
}
DEFAULT_ORDER = {s: [k for k, v in LIBRARY.items() if v["side"] == s] for s in SIDES}


def validate_scores(scores: Mapping[str, Optional[int]]) -> dict[str, int]:
    """Valida os 4 indicadores (1–5) de um objetivo; ignora os não preenchidos."""
    out: dict[str, int] = {}
    for k, v in scores.items():
        if k not in INDICATORS:
            raise ValueError(f"Indicador desconhecido: {k!r}")
        if v is None:
            continue
        if isinstance(v, bool) or v not in SCALE:
            raise ValueError(f"Valor inválido para {k!r}: {v!r} (use 1 a 5).")
        out[k] = int(v)
    if not out:
        raise ValueError("Preencha pelo menos um indicador.")
    return out


def objective_average(scores: Mapping[str, Optional[int]]) -> float:
    return round(statistics.fmean(validate_scores(scores).values()), 2)


def application_gap(scores: Mapping[str, int]) -> bool:
    """Treina bem mas não aplica em jogo (execução − aplicação ≥ 2)."""
    return "execution" in scores and "application" in scores and scores["execution"] - scores["application"] >= 2


def suggest_objectives(side: str, qualitative: Optional[Mapping[str, int]] = None,
                       movement_change_pct: Optional[float] = None,
                       last_scores: Optional[Mapping[str, Mapping[str, int]]] = None,
                       limit: int = 4) -> list[dict]:
    """Ordena os objetivos de `side` por necessidade (0–1; maior = prioritário).

    Sinais (cada um normalizado a 0–1, depois média dos disponíveis):
      • qualitativa: (4 − classificação)/3 média dos critérios ligados ao objetivo;
      • movimentos defensivos: `movement_change_pct` (+ = melhoria): piorou →
        MOVEMENT_NEED_WORSE; sem melhoria → MOVEMENT_NEED_STALLED; melhorou → 0;
      • avaliação de objetivos: (5 − média)/4 da última avaliação do objetivo.
    Sem qualquer dado, mantém a ordem da biblioteca (`reasons` vazio).
    """
    if side not in SIDES:
        raise ValueError(f"Lado inválido: {side!r}")
    qualitative = qualitative or {}
    last_scores = last_scores or {}
    out = []
    for order, key in enumerate(DEFAULT_ORDER[side]):
        obj, signals, reasons = LIBRARY[key], [], []
        rated = [(c, qualitative[c]) for c in obj.get("criteria", []) if c in qualitative]
        if rated:
            signals.append(statistics.fmean((4 - r) / 3 for _, r in rated))
            low = [c for c, r in rated if r <= 2]
            if low:
                reasons.append("Critérios técnicos a melhorar: " + ", ".join(low))
        if obj.get("movement") and movement_change_pct is not None:
            need = (MOVEMENT_NEED_WORSE if movement_change_pct < 0
                    else MOVEMENT_NEED_STALLED if movement_change_pct == 0 else 0.0)
            signals.append(need)
            if need:
                reasons.append("Tempo no teste de movimentos defensivos sem melhoria")
        if key in last_scores and last_scores[key]:
            signals.append((5 - objective_average(last_scores[key])) / 4)
            if application_gap(last_scores[key]):
                reasons.append("Executa bem no treino mas aplica pouco em jogo")
        need = round(statistics.fmean(signals), 3) if signals else 0.0
        out.append({"key": key, "side": side, "text": obj["text"], "need": need,
                    "reasons": reasons, "drills": obj["drills"], "_order": order})
    out.sort(key=lambda o: (-o["need"], o["_order"]))
    for o in out:
        del o["_order"]
    return out[:limit]


def library_for(keys: Sequence[str]) -> list[dict]:
    return [{"key": k, **LIBRARY[k]} for k in keys if k in LIBRARY]
