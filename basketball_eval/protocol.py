"""Protocolo configurável do Teste de Movimentos Defensivos (pontos A–F).

Fonte: ENB — Avaliação Quantitativa (Johnson & Nelson, 1986). Trajetória
A-B-C-D-E-F-A. Só se preenche o que o documento descreve; o resto fica None
para o treinador/organização completar. As coordenadas (m) vêm da figura
(5,80 m de comprimento, 2,90 m a meio); a largura (x) não é indicada.
"""

import json

DEFAULT_PROTOCOL = {
    "test": "defensive_movement",
    "reference": "Johnson & Nelson (1986)",
    "area": {"length_m": 5.80, "half_length_m": 2.90, "width_m": None},
    "sequence": ["A", "B", "C", "D", "E", "F", "A"],
    "points": {
        "A": {"x_m": None, "y_m": 0.0, "note": "Início/chegada; de costas para o cesto"},
        "B": {"x_m": None, "y_m": 0.0, "note": "Toca no chão fora da área com a mão esquerda"},
        "C": {"x_m": None, "y_m": 2.90, "note": "Marcador central; toca no chão fora da área com a mão direita"},
        "D": {"x_m": None, "y_m": 5.80, "note": None},
        "E": {"x_m": None, "y_m": 5.80, "note": None},
        "F": {"x_m": None, "y_m": 2.90, "note": "Marcador central"},
    },
    "legs": [
        {"from": "A", "to": "B", "movement": "deslocamento lateral, sem cruzar os pés"},
        {"from": "B", "to": "C", "movement": "drop step e deslize"},
    ],
    "start_signal": "Pronto, vai!",
    "end_condition": "Os dois pés ultrapassam a linha de chegada",
    "n_trials": 3,
    "recovery_minutes": 5,
    "best_of": "melhor (menor) tempo",
}


def dumps(protocol: dict) -> str:
    return json.dumps(protocol, ensure_ascii=False)


def validate(protocol: dict) -> None:
    """Garante coerência interna: a sequência só usa pontos definidos."""
    pts = set(protocol.get("points", {}))
    bad = [p for p in protocol.get("sequence", []) if p not in pts]
    if bad:
        raise ValueError(f"Sequência usa pontos não definidos: {bad}")
    for leg in protocol.get("legs", []):
        if leg["from"] not in pts or leg["to"] not in pts:
            raise ValueError(f"Segmento inválido: {leg}")
