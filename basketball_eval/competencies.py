"""Registo de competências da avaliação QUALITATIVA (Técnica Individual).

Cada competência (ex.: Lançamento) é uma definição de dados: dimensões e
critérios. Acrescentar Drible, Passe, etc. = acrescentar uma definição aqui
(e uma página); o cálculo, a base de dados e o relatório são genéricos.

Escala qualitativa estruturada 1–5 (não é uma medição quantitativa).
"""

from __future__ import annotations

from dataclasses import dataclass

SCALE_MIN, SCALE_MAX = 1, 5

# Níveis da escala — mostrados sempre com número + descrição.
LEVELS: dict[int, str] = {
    1: "Inicial",
    2: "Em desenvolvimento",
    3: "Adequado",
    4: "Bom",
    5: "Muito bom",
}

# Escalões iniciais. Para acrescentar outros: inserir em `age_groups` (ver qual_db).
DEFAULT_AGE_GROUPS = ("Sub-8", "Sub-10", "Sub-12")


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str


@dataclass(frozen=True)
class Dimension:
    key: str
    name: str            # nome curto (eixo do radar)
    title: str           # nome completo da secção
    criteria: tuple[Criterion, ...]


@dataclass(frozen=True)
class Competency:
    key: str
    name: str            # ex.: "Lançamento"
    domain: str          # ex.: "Técnica Individual"
    dimensions: tuple[Dimension, ...]

    @property
    def criteria(self) -> tuple[Criterion, ...]:
        return tuple(c for d in self.dimensions for c in d.criteria)

    def dimension_of(self, criterion_key: str) -> Dimension:
        for d in self.dimensions:
            if any(c.key == criterion_key for c in d.criteria):
                return d
        raise KeyError(criterion_key)

    def criterion(self, criterion_key: str) -> Criterion:
        for c in self.criteria:
            if c.key == criterion_key:
                return c
        raise KeyError(criterion_key)


def _dim(key, name, title, *crit):
    return Dimension(key, name, title, tuple(Criterion(k, l) for k, l in crit))


LANCAMENTO = Competency(
    key="lancamento",
    name="Lançamento",
    domain="Técnica Individual",
    dimensions=(
        _dim("preparacao", "Preparação", "Preparação para o lançamento",
             ("equilibrio_corporal", "Equilíbrio corporal"),
             ("posicao_pes", "Posição dos pés"),
             ("flexao_inferiores", "Flexão dos membros inferiores"),
             ("estabilidade_corporal", "Estabilidade corporal"),
             ("preparacao_maos_bola", "Preparação das mãos e da bola")),
        _dim("execucao", "Execução", "Execução do lançamento",
             ("coordenacao", "Coordenação entre membros inferiores e superiores"),
             ("posicao_cotovelo", "Posição do cotovelo"),
             ("mao_lancamento", "Utilização adequada da mão de lançamento"),
             ("mao_apoio", "Ação da mão de apoio"),
             ("extensao", "Extensão dos membros inferiores e superiores"),
             ("continuidade", "Continuidade do movimento")),
        _dim("finalizacao", "Finalização", "Finalização",
             ("extensao_braco", "Extensão do braço"),
             ("flexao_pulso", "Flexão do pulso"),
             ("direcao_mao_cesto", "Direção da mão para o cesto"),
             ("equilibrio_final", "Manutenção do equilíbrio"),
             ("follow_through", "Follow-through")),
        _dim("consistencia", "Consistência", "Consistência",
             ("repetir_gesto", "Capacidade de repetir o gesto"),
             ("estabilidade_tecnica", "Estabilidade técnica"),
             ("situacoes_diferentes", "Execução do lançamento em diferentes situações")),
        _dim("aplicacao_jogo", "Aplicação no jogo", "Aplicação no jogo",
             ("reconhecer_quando", "Capacidade de reconhecer quando lançar"),
             ("preparacao_jogo", "Preparação para o lançamento em contexto de jogo"),
             ("apos_rececao", "Lançamento após receção"),
             ("apos_drible", "Lançamento após drible"),
             ("sob_oposicao", "Execução sob oposição"),
             ("adaptacao_contexto", "Adaptação ao contexto")),
    ),
)

# Competências implementadas (chave → definição).
COMPETENCIES: dict[str, Competency] = {LANCAMENTO.key: LANCAMENTO}

# Roda das Competências (futura): ordem prevista e estado de implementação.
WHEEL: tuple[tuple[str, str], ...] = (
    ("lancamento", "Lançamento"),
    ("drible", "Drible / Domínio da Bola"),
    ("passe", "Passe"),
    ("rececao", "Receção da Bola"),
    ("trabalho_pes", "Trabalho de Pés"),
    ("finalizacoes", "Finalizações"),
    ("defesa_individual", "Defesa Individual"),
    ("tatica_individual", "Tática Individual"),
    ("contra_ataque", "Contra-Ataque"),
)


def get(key: str) -> Competency:
    try:
        return COMPETENCIES[key]
    except KeyError:
        raise ValueError(f"Competência não implementada: {key!r}") from None


def implemented_wheel() -> list[tuple[str, str, bool]]:
    return [(k, n, k in COMPETENCIES) for k, n in WHEEL]
