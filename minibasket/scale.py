"""Escala de avaliação pedagógica (configurável).

Não é uma norma científica nem um percentil: apenas descreve o nível de
desenvolvimento observado pelo treinador. Os níveis ficam na tabela
`scale_levels`, para poderem ser substituídos no futuro; esta é a escala inicial.
"""

DEFAULT_SCALE_NAME = "Escala pedagógica 1–5"

DEFAULT_LEVELS = (
    (1, "Inicial"),
    (2, "Em desenvolvimento"),
    (3, "Adequado"),
    (4, "Bom"),
    (5, "Muito bom"),
)


def bounds(levels=DEFAULT_LEVELS) -> tuple[int, int]:
    values = [v for v, _ in levels]
    return min(values), max(values)


def label(value: int, levels=DEFAULT_LEVELS) -> str:
    for v, text in levels:
        if v == value:
            return text
    raise ValueError(f"Nível inexistente na escala: {value}")
