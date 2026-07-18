from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.services.training_data_seed import seed_training_data_once


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    seed_training_data_once()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        root_path=settings.root_path,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()
