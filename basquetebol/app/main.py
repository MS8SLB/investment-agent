from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config, models  # noqa: F401
from .database import Base, SessionLocal, engine
from .routers import analise, atletas, avaliacoes, configuracao, equipas, painel, relatorios, testes
from .seed import seed


def init_db(eng=engine, session_factory=SessionLocal):
    config.DATA_DIR.mkdir(exist_ok=True)
    Base.metadata.create_all(eng)
    with session_factory() as db:
        seed(db)


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=config.TITULO, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "app" / "static")), name="static")
    for r in (painel, atletas, equipas, testes, avaliacoes, analise, relatorios, configuracao):
        app.include_router(r.router)
    return app


app = create_app()
