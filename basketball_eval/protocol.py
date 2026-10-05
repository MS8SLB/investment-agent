"""Protocolo configurável do Teste de Movimentos Defensivos (pontos A–F).

Fonte: ENB — Avaliação Quantitativa (Johnson & Nelson, 1986). Trajetória
A-B-C-D-E-F-A. Só se preenche o que o documento descreve; o resto fica None
para o treinador/organização completar. As coordenadas (m) vêm da figura
(5,80 m de comprimento, 2,90 m a meio); a largura (x) não é indicada.
"""

import json

DEFAULT_PROTOCOL = {
    "version": 2,
    "test": "defensive_movement",
    "reference": "Johnson & Nelson (1986)",
    "area": {"length_m": 5.80, "half_length_m": 2.90, "width_m": None},
    # Trajetória da figura do documento ENB: A-B-C-D-E-F-A.
    "sequence": ["A", "B", "C", "D", "E", "F", "A"],
    # side = lado visto na figura (esquerda/direita da página); y_m medido a partir da linha A-B.
    "points": {
        "A": {"side": "left", "y_m": 0.0, "role": "start_finish",
              "note": "Início/chegada; de costas para o cesto"},
        "B": {"side": "right", "y_m": 0.0, "role": "turn",
              "note": "Toca no chão fora da área restritiva (mão esquerda, segundo o texto)"},
        "C": {"side": "left", "y_m": 2.90, "role": "target",
              "note": "Marcador central da área; toca no chão fora da área (mão direita)"},
        "D": {"side": "right", "y_m": 5.80, "role": "turn", "note": None},
        "E": {"side": "left", "y_m": 5.80, "role": "turn", "note": None},
        "F": {"side": "right", "y_m": 2.90, "role": "target", "note": "Marcador central da área"},
    },
    "legs": [
        {"from": "A", "to": "B", "path": "straight", "movement": "deslocamento lateral, sem cruzar os pés"},
        {"from": "B", "to": "C", "path": "diagonal", "movement": "drop step e deslize"},
        {"from": "C", "to": "D", "path": "diagonal", "movement": None},
        {"from": "D", "to": "E", "path": "straight", "movement": None},
        {"from": "E", "to": "F", "path": "diagonal", "movement": None},
        {"from": "F", "to": "A", "path": "diagonal", "movement": None},
    ],
    # Itens que o documento não esclarece — a organização deve definir.
    "open_items": [
        "Tipo de movimento nos segmentos C-D, D-E, E-F e F-A (o texto só descreve A-B e B-C).",
        "Largura da área (x) não indicada no documento.",
        "Lado/mão em B: o texto diz deslocamento para a esquerda, mas B surge à direita de A na figura.",
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


def load(conn) -> dict:
    """Protocolo guardado na base de dados (configurável)."""
    row = conn.execute("SELECT config FROM test_protocols WHERE test_key='defensive_movement'").fetchone()
    cfg = json.loads(row["config"])
    validate(cfg)
    return cfg
