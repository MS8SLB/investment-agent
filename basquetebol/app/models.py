"""Modelo de dados (ver ARQUITETURA.md)."""
from datetime import date, datetime

from sqlalchemy import (Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Category(Base):  # Escalão
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(20), unique=True)  # "Sub-10"
    sort: Mapped[int] = mapped_column(Integer, default=0)
    has_enb_protocol: Mapped[bool] = mapped_column(Boolean, default=True)  # coberto pelo documento ENB?


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    club: Mapped[str] = mapped_column(String(120), default="")
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))
    category: Mapped[Category | None] = relationship()
    __table_args__ = (UniqueConstraint("name", "club"),)


class Athlete(Base):
    __tablename__ = "athletes"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    birth_date: Mapped[date] = mapped_column(Date)
    sex: Mapped[str] = mapped_column(String(1))  # "M" | "F"
    club: Mapped[str] = mapped_column(String(120), default="")
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    team: Mapped[Team | None] = relationship()
    category: Mapped[Category] = relationship()
    evaluations: Mapped[list["Evaluation"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan", order_by="Evaluation.date")


class TestDef(Base):
    """Definição de um teste (protocolo). Novos testes = novas linhas, sem alterar código."""
    __tablename__ = "tests"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(60), unique=True)
    numeral: Mapped[str] = mapped_column(String(8), default="")
    name: Mapped[str] = mapped_column(String(160))
    source: Mapped[str] = mapped_column(String(200), default="")      # citação do protocolo
    capacity: Mapped[str] = mapped_column(String(200), default="")    # o que avalia (subtítulo)
    objective: Mapped[str] = mapped_column(Text, default="")
    description: Mapped[str] = mapped_column(Text, default="")
    procedure: Mapped[str] = mapped_column(Text, default="")
    materials: Mapped[str] = mapped_column(Text, default="")
    result_note: Mapped[str] = mapped_column(Text, default="")
    unit: Mapped[str] = mapped_column(String(40))                     # "s", "pontos", ...
    direction: Mapped[str] = mapped_column(String(6))                 # "lower" | "higher"
    n_attempts: Mapped[int] = mapped_column(Integer, default=1)
    attempts_rule: Mapped[str] = mapped_column(Text, default="")
    attempts_confirmed: Mapped[bool] = mapped_column(Boolean, default=True)
    decimals: Mapped[int] = mapped_column(Integer, default=2)
    max_value: Mapped[float | None] = mapped_column(Float)
    is_enb: Mapped[bool] = mapped_column(Boolean, default=True)       # False = teste acrescentado pelo utilizador
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    sort: Mapped[int] = mapped_column(Integer, default=0)

    @property
    def better_label(self) -> str:
        return "menor valor = melhor" if self.direction == "lower" else "maior valor = melhor"


class CategoryTest(Base):
    """Que testes são usados em cada escalão."""
    __tablename__ = "category_tests"
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"), primary_key=True)
    test_id: Mapped[int] = mapped_column(ForeignKey("tests.id"), primary_key=True)


class Evaluation(Base):
    """Uma sessão de avaliação de um atleta numa data."""
    __tablename__ = "evaluations"
    id: Mapped[int] = mapped_column(primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))  # escalão à data da avaliação
    date: Mapped[date] = mapped_column(Date)
    evaluator: Mapped[str] = mapped_column(String(120), default="")      # treinador (futuro: FK para utilizadores)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    athlete: Mapped[Athlete] = relationship(back_populates="evaluations")
    category: Mapped[Category] = relationship()
    results: Mapped[list["TestResult"]] = relationship(
        back_populates="evaluation", cascade="all, delete-orphan", order_by="TestResult.test_id")


class TestResult(Base):
    __tablename__ = "test_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluations.id"))
    test_id: Mapped[int] = mapped_column(ForeignKey("tests.id"))
    best_value: Mapped[float | None] = mapped_column(Float)  # calculado a partir das tentativas
    notes: Mapped[str] = mapped_column(Text, default="")
    evaluation: Mapped[Evaluation] = relationship(back_populates="results")
    test: Mapped[TestDef] = relationship()
    attempts: Mapped[list["Attempt"]] = relationship(
        back_populates="result", cascade="all, delete-orphan", order_by="Attempt.number")
    __table_args__ = (UniqueConstraint("evaluation_id", "test_id"),)


class Attempt(Base):
    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("test_results.id"))
    number: Mapped[int] = mapped_column(Integer)
    value: Mapped[float] = mapped_column(Float)
    result: Mapped[TestResult] = relationship(back_populates="attempts")


class ReferenceTable(Base):
    """Tabelas de referência validadas (introduzidas pelo utilizador). Vazio por defeito."""
    __tablename__ = "reference_tables"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    source: Mapped[str] = mapped_column(String(300), default="")
    test_id: Mapped[int] = mapped_column(ForeignKey("tests.id"))
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))  # None = todos
    sex: Mapped[str | None] = mapped_column(String(1))                            # None = ambos
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    test: Mapped[TestDef] = relationship()
    category: Mapped[Category | None] = relationship()
    rows: Mapped[list["ReferenceRow"]] = relationship(
        back_populates="table", cascade="all, delete-orphan", order_by="ReferenceRow.sort")


class ReferenceRow(Base):
    __tablename__ = "reference_rows"
    id: Mapped[int] = mapped_column(primary_key=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("reference_tables.id"))
    label: Mapped[str] = mapped_column(String(80))       # ex.: "Percentil 50", "Bom"
    min_value: Mapped[float | None] = mapped_column(Float)  # inclusivo
    max_value: Mapped[float | None] = mapped_column(Float)  # inclusivo
    sort: Mapped[int] = mapped_column(Integer, default=0)
    table: Mapped[ReferenceTable] = relationship(back_populates="rows")
