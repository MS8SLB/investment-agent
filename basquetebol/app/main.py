"""Plataforma de Avaliação Quantitativa Técnica — Basquetebol de Formação.

Autor e responsável metodológico: Mário Silva.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import models  # noqa: F401  (regista as tabelas)
from .db import Base, SessionLocal, engine
from .routes_analysis import router as analysis_router
from .routes_athletes import router as athletes_router
from .routes_eval import router as eval_router
from .routes_reports import router as reports_router
from .routes_tests import router as tests_router
from .seed import seed
from .web import APP_DIR


@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed(db)
    yield


app = FastAPI(title="Plataforma de Avaliação Quantitativa Técnica — Basquetebol de Formação", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
for r in (analysis_router, athletes_router, eval_router, tests_router, reports_router):
    app.include_router(r)
