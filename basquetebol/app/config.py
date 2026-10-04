"""Configuração. Tudo pode ser alterado por variáveis de ambiente (para publicação online)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'avaliacao.db'}")
CATALOGO_PATH = DATA_DIR / "catalogo_testes.json"

TITULO = "Plataforma de Avaliação Quantitativa Técnica"
SUBTITULO = "Basquetebol de Formação"
AUTOR = "Mário Silva"
