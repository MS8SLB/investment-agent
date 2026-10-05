"""Avaliação QUALITATIVA da técnica defensiva (grelha de observação do treinador).

Complementa o Teste de Movimentos Defensivos (quantitativo, em segundos): o
treinador observa a execução do percurso A-B-C-D-E-F-A e classifica cada
critério técnico numa escala de 1 a 4 com descritores observáveis.

Lógica pura (sem I/O). Os critérios e descritores são uma PROPOSTA de trabalho
derivada da descrição do teste (deslocamento lateral sem cruzar os pés, drop
step, deslize, toque no chão fora da área); não constituem norma validada e
podem ser ajustados em CRITERIA. Mais alto = melhor (contrário ao tempo).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Optional

TEST_KEY = "defensive_qualitative"
TEST_NAME_PT = "Avaliação Qualitativa da Defesa"
SCALE = (1, 2, 3, 4)
SCALE_LABELS = {1: "Não observado / a iniciar", 2: "Em desenvolvimento",
                3: "Conseguido", 4: "Consolidado"}

# key → (nome, dimensão, {nível: descritor observável})
CRITERIA: dict[str, dict] = {
    "posicao_base": {
        "name": "Posição defensiva base", "dimension": "Técnica",
        "descriptors": {
            1: "Corpo alto, joelhos quase estendidos, peso nos calcanhares.",
            2: "Flete os joelhos mas perde a posição ao deslocar-se.",
            3: "Mantém joelhos fletidos, tronco direito e base larga na maior parte do percurso.",
            4: "Posição baixa, equilibrada e estável durante todo o percurso.",
        }},
    "deslize_lateral": {
        "name": "Deslize lateral sem cruzar os pés", "dimension": "Técnica",
        "descriptors": {
            1: "Cruza os pés ou salta frequentemente.",
            2: "Cruza os pés em alguns momentos ou junta os pés.",
            3: "Desliza sem cruzar os pés, com pequenas falhas de ritmo.",
            4: "Passos curtos e rápidos, pés sempre afastados, sem saltar.",
        }},
    "drop_step": {
        "name": "Drop step e mudança de direção", "dimension": "Técnica",
        "descriptors": {
            1: "Não executa o drop step; roda o corpo ou para para mudar.",
            2: "Executa o drop step com perda de equilíbrio ou de velocidade.",
            3: "Drop step correto, com ligeira paragem na transição.",
            4: "Drop step fluido, sem paragem, mantém a posição após a mudança.",
        }},
    "toque_marcadores": {
        "name": "Toques nos pontos e cumprimento do percurso", "dimension": "Execução",
        "descriptors": {
            1: "Falha toques ou omite pontos do percurso.",
            2: "Toca nos pontos mas com a mão errada ou sem chegar ao chão.",
            3: "Toca corretamente, perdendo posição ao baixar.",
            4: "Toca corretamente fora da área sem perder a posição defensiva.",
        }},
    "cabeca_olhar": {
        "name": "Cabeça e olhar", "dimension": "Perceção",
        "descriptors": {
            1: "Olha para o chão ou para os pés durante o percurso.",
            2: "Levanta a cabeça apenas em parte do percurso.",
            3: "Cabeça levantada na maior parte do tempo.",
            4: "Cabeça levantada e olhar à frente de forma constante.",
        }},
    "maos_bracos": {
        "name": "Posição das mãos e braços", "dimension": "Técnica",
        "descriptors": {
            1: "Braços junto ao corpo ou descoordenados.",
            2: "Braços ativos mas sem posição definida.",
            3: "Braços afastados e ativos, com momentos de descuido.",
            4: "Braços sempre ativos e em posição de defesa ao longo do percurso.",
        }},
    "velocidade_ritmo": {
        "name": "Velocidade e ritmo", "dimension": "Execução",
        "descriptors": {
            1: "Ritmo lento ou irregular.",
            2: "Acelera em alguns segmentos e abranda noutros.",
            3: "Ritmo razoavelmente constante.",
            4: "Ritmo rápido e constante, com acelerações nas mudanças.",
        }},
    "atitude": {
        "name": "Atitude e empenho", "dimension": "Atitude",
        "descriptors": {
            1: "Pouco empenho ou desatenção às instruções.",
            2: "Empenho irregular; precisa de incentivo.",
            3: "Empenhado e atento.",
            4: "Muito empenhado, concentrado e procura corrigir-se.",
        }},
}

DIMENSIONS = tuple(dict.fromkeys(c["dimension"] for c in CRITERIA.values()))
STRENGTH_MIN, IMPROVE_MAX = 3, 2   # ≥3 ponto forte; ≤2 a melhorar

# Sugestões de treino por critério (aparecem para critérios ≤ IMPROVE_MAX).
SUGGESTIONS = {
    "posicao_base": "Deslizes mantendo a posição baixa; manter 10 s em posição defensiva.",
    "deslize_lateral": "Deslizes em linha com cones, passos curtos, sem juntar nem cruzar os pés.",
    "drop_step": "Drop step em parado, depois em movimento, com sinal do treinador.",
    "toque_marcadores": "Percurso lento com toque no chão sem levantar o tronco.",
    "cabeca_olhar": "Deslizes a seguir um colega ou bola, sem olhar para os pés.",
    "maos_bracos": "Deslizes com os braços abertos; sombra defensiva a pares.",
    "velocidade_ritmo": "Percursos curtos a ritmo crescente, com pausa de recuperação.",
    "atitude": "Objetivos pessoais simples e reforço positivo após cada tentativa.",
}


def validate_ratings(ratings: Mapping[str, Optional[int]]) -> dict[str, int]:
    """Devolve só os critérios classificados. Rejeita chaves/valores inválidos."""
    out: dict[str, int] = {}
    for k, v in ratings.items():
        if k not in CRITERIA:
            raise ValueError(f"Critério desconhecido: {k!r}")
        if v is None:
            continue
        if isinstance(v, bool) or v not in SCALE:
            raise ValueError(f"Classificação inválida para {k!r}: {v!r} (use 1 a 4).")
        out[k] = int(v)
    if not out:
        raise ValueError("Classifique pelo menos um critério.")
    return out


@dataclass
class Summary:
    average: float
    by_dimension: dict[str, float]
    strengths: list[str]
    to_improve: list[str]
    n_rated: int
    n_total: int
    complete: bool


def summarize(ratings: Mapping[str, Optional[int]]) -> Summary:
    r = validate_ratings(ratings)
    dims: dict[str, list[int]] = {}
    for k, v in r.items():
        dims.setdefault(CRITERIA[k]["dimension"], []).append(v)
    return Summary(
        average=round(statistics.fmean(r.values()), 2),
        by_dimension={d: round(statistics.fmean(v), 2) for d, v in dims.items()},
        strengths=[k for k, v in r.items() if v >= STRENGTH_MIN],
        to_improve=[k for k, v in r.items() if v <= IMPROVE_MAX],
        n_rated=len(r), n_total=len(CRITERIA), complete=len(r) == len(CRITERIA),
    )


def compare(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    """Variação por critério (depois - antes; positivo = melhoria), só critérios comuns."""
    return {k: after[k] - before[k] for k in CRITERIA if k in before and k in after}


def feedback_text(player_name: str, ratings: Mapping[str, Optional[int]],
                  previous: Optional[Mapping[str, int]] = None) -> str:
    """Relatório descritivo para o treinador/atleta."""
    s = summarize(ratings)
    r = validate_ratings(ratings)
    lines = [f"AVALIAÇÃO QUALITATIVA DA DEFESA — {player_name}", "",
             f"Critérios classificados: {s.n_rated}/{s.n_total}  ·  Média: {s.average:.2f}/4", ""]
    for d, v in s.by_dimension.items():
        lines.append(f"  {d}: {v:.2f}/4")
    lines += ["", "Pontos fortes:"]
    lines += [f"  + {CRITERIA[k]['name']}: {CRITERIA[k]['descriptors'][r[k]]}" for k in s.strengths] or ["  —"]
    lines += ["", "A melhorar:"]
    lines += [f"  - {CRITERIA[k]['name']}: {CRITERIA[k]['descriptors'][r[k]]}\n"
              f"    Sugestão: {SUGGESTIONS[k]}" for k in s.to_improve] or ["  —"]
    if previous:
        delta = compare(previous, r)
        if delta:
            up = [CRITERIA[k]["name"] for k, v in delta.items() if v > 0]
            down = [CRITERIA[k]["name"] for k, v in delta.items() if v < 0]
            lines += ["", "Evolução face à avaliação anterior:",
                      f"  Melhorou: {', '.join(up) or '—'}",
                      f"  Piorou: {', '.join(down) or '—'}"]
    return "\n".join(lines)
