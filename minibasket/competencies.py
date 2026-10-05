"""As nove competências da Roda das Competências do Minibasquete."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Competency:
    key: str          # chave estável (BD)
    name: str         # nome completo (PT-PT)
    short: str        # nome curto para gráficos
    position: int     # ordem na roda


COMPETENCIES = (
    Competency("shooting", "Lançamento", "Lançamento", 1),
    Competency("dribbling", "Drible / Domínio da Bola", "Drible", 2),
    Competency("passing", "Passe", "Passe", 3),
    Competency("reception", "Receção da Bola", "Receção", 4),
    Competency("footwork", "Trabalho de Pés", "Trabalho de Pés", 5),
    Competency("finishing", "Finalizações", "Finalizações", 6),
    Competency("individual_defense", "Defesa Individual", "Defesa Individual", 7),
    Competency("individual_tactics", "Tática Individual", "Tática Individual", 8),
    Competency("fast_break", "Contra-Ataque", "Contra-Ataque", 9),
)

KEYS = tuple(c.key for c in COMPETENCIES)
BY_KEY = {c.key: c for c in COMPETENCIES}


def name_of(key: str) -> str:
    return BY_KEY[key].name


def short_of(key: str) -> str:
    return BY_KEY[key].short
