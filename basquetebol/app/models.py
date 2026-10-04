from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Escalao(Base):
    __tablename__ = "escalao"
    id: Mapped[int] = mapped_column(primary_key=True)
    codigo: Mapped[str] = mapped_column(String(20), unique=True)
    nome: Mapped[str] = mapped_column(String(40))
    ordem: Mapped[int] = mapped_column(Integer, default=0)
    testes: Mapped[list["Teste"]] = relationship(secondary="escalao_teste", order_by="Teste.ordem")


class EscalaoTeste(Base):
    __tablename__ = "escalao_teste"
    escalao_id: Mapped[int] = mapped_column(ForeignKey("escalao.id", ondelete="CASCADE"), primary_key=True)
    teste_id: Mapped[int] = mapped_column(ForeignKey("teste.id", ondelete="CASCADE"), primary_key=True)


class Equipa(Base):
    __tablename__ = "equipa"
    id: Mapped[int] = mapped_column(primary_key=True)
    nome: Mapped[str] = mapped_column(String(120))
    clube: Mapped[str] = mapped_column(String(120), default="")
    escalao_id: Mapped[int] = mapped_column(ForeignKey("escalao.id"))
    escalao: Mapped[Escalao] = relationship()
    atletas: Mapped[list["Atleta"]] = relationship(back_populates="equipa")
    __table_args__ = (UniqueConstraint("nome", "clube", "escalao_id"),)


class Atleta(Base):
    __tablename__ = "atleta"
    id: Mapped[int] = mapped_column(primary_key=True)
    nome: Mapped[str] = mapped_column(String(160))
    data_nascimento: Mapped[date] = mapped_column(Date)
    sexo: Mapped[str] = mapped_column(String(1))  # 'F' ou 'M'
    clube: Mapped[str] = mapped_column(String(120), default="")
    equipa_id: Mapped[int | None] = mapped_column(ForeignKey("equipa.id", ondelete="SET NULL"), nullable=True)
    escalao_id: Mapped[int] = mapped_column(ForeignKey("escalao.id"))
    notas: Mapped[str] = mapped_column(Text, default="")
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    equipa: Mapped[Equipa | None] = relationship(back_populates="atletas")
    escalao: Mapped[Escalao] = relationship()
    avaliacoes: Mapped[list["Avaliacao"]] = relationship(
        back_populates="atleta", cascade="all, delete-orphan", order_by="Avaliacao.data")


class Teste(Base):
    __tablename__ = "teste"
    id: Mapped[int] = mapped_column(primary_key=True)
    codigo: Mapped[str] = mapped_column(String(60), unique=True)
    nome: Mapped[str] = mapped_column(String(200))
    referencia: Mapped[str] = mapped_column(String(200), default="")
    dominio: Mapped[str] = mapped_column(String(300), default="")
    objetivo: Mapped[str] = mapped_column(Text, default="")
    descricao: Mapped[str] = mapped_column(Text, default="")
    protocolo: Mapped[str] = mapped_column(Text, default="")
    material: Mapped[str] = mapped_column(Text, default="")
    unidade: Mapped[str] = mapped_column(String(30))            # s, pontos, cestos...
    unidade_nome: Mapped[str] = mapped_column(String(80), default="")
    direcao: Mapped[str] = mapped_column(String(5))             # 'menor' ou 'maior' = melhor
    n_tentativas: Mapped[int] = mapped_column(Integer, default=1)
    tentativas_min: Mapped[int] = mapped_column(Integer, default=1)
    valor_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    valor_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    casas: Mapped[int] = mapped_column(Integer, default=2)
    resultado_descricao: Mapped[str] = mapped_column(Text, default="")
    a_confirmar: Mapped[str] = mapped_column(Text, default="")  # uma pendência por linha
    personalizado: Mapped[bool] = mapped_column(Boolean, default=False)
    ordem: Mapped[int] = mapped_column(Integer, default=100)

    @property
    def pendencias(self) -> list[str]:
        return [l for l in self.a_confirmar.split("\n") if l.strip()]

    @property
    def menor_melhor(self) -> bool:
        return self.direcao == "menor"


class Avaliacao(Base):
    __tablename__ = "avaliacao"
    id: Mapped[int] = mapped_column(primary_key=True)
    atleta_id: Mapped[int] = mapped_column(ForeignKey("atleta.id", ondelete="CASCADE"))
    data: Mapped[date] = mapped_column(Date)
    escalao_id: Mapped[int] = mapped_column(ForeignKey("escalao.id"))  # escalão no momento da avaliação
    observacoes: Mapped[str] = mapped_column(Text, default="")
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    atleta: Mapped[Atleta] = relationship(back_populates="avaliacoes")
    escalao: Mapped[Escalao] = relationship()
    resultados: Mapped[list["ResultadoTeste"]] = relationship(
        back_populates="avaliacao", cascade="all, delete-orphan")


class ResultadoTeste(Base):
    __tablename__ = "resultado_teste"
    id: Mapped[int] = mapped_column(primary_key=True)
    avaliacao_id: Mapped[int] = mapped_column(ForeignKey("avaliacao.id", ondelete="CASCADE"))
    teste_id: Mapped[int] = mapped_column(ForeignKey("teste.id"))
    melhor_valor: Mapped[float | None] = mapped_column(Float, nullable=True)
    observacoes: Mapped[str] = mapped_column(Text, default="")
    avaliacao: Mapped[Avaliacao] = relationship(back_populates="resultados")
    teste: Mapped[Teste] = relationship()
    tentativas: Mapped[list["Tentativa"]] = relationship(
        back_populates="resultado", cascade="all, delete-orphan", order_by="Tentativa.numero")
    __table_args__ = (UniqueConstraint("avaliacao_id", "teste_id"),)


class Tentativa(Base):
    __tablename__ = "tentativa"
    id: Mapped[int] = mapped_column(primary_key=True)
    resultado_id: Mapped[int] = mapped_column(ForeignKey("resultado_teste.id", ondelete="CASCADE"))
    numero: Mapped[int] = mapped_column(Integer)
    valor: Mapped[float] = mapped_column(Float)
    observacoes: Mapped[str] = mapped_column(Text, default="")
    resultado: Mapped[ResultadoTeste] = relationship(back_populates="tentativas")


class ReferenciaTabela(Base):
    """Tabela de referência validada (importada por CSV). Vazia por defeito."""
    __tablename__ = "referencia_tabela"
    id: Mapped[int] = mapped_column(primary_key=True)
    teste_id: Mapped[int] = mapped_column(ForeignKey("teste.id", ondelete="CASCADE"))
    escalao_id: Mapped[int | None] = mapped_column(ForeignKey("escalao.id"), nullable=True)
    sexo: Mapped[str | None] = mapped_column(String(1), nullable=True)
    nome: Mapped[str] = mapped_column(String(200))
    fonte: Mapped[str] = mapped_column(String(300), default="")
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    teste: Mapped[Teste] = relationship()
    escalao: Mapped[Escalao | None] = relationship()
    faixas: Mapped[list["ReferenciaFaixa"]] = relationship(
        back_populates="tabela", cascade="all, delete-orphan", order_by="ReferenciaFaixa.ordem")


class ReferenciaFaixa(Base):
    __tablename__ = "referencia_faixa"
    id: Mapped[int] = mapped_column(primary_key=True)
    tabela_id: Mapped[int] = mapped_column(ForeignKey("referencia_tabela.id", ondelete="CASCADE"))
    rotulo: Mapped[str] = mapped_column(String(80))
    minimo: Mapped[float | None] = mapped_column(Float, nullable=True)  # inclusivo
    maximo: Mapped[float | None] = mapped_column(Float, nullable=True)  # inclusivo
    ordem: Mapped[int] = mapped_column(Integer, default=0)
    tabela: Mapped[ReferenciaTabela] = relationship(back_populates="faixas")
