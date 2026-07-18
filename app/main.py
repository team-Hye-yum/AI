from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Thread

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.services.training_data_seed import seed_training_data_once


def _api_base_path() -> str:
    base_path = settings.root_path.strip()
    if not base_path or base_path == "/":
        return ""
    return f"/{base_path.strip('/')}"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    Thread(target=seed_training_data_once, daemon=True).start()
    yield


def create_app() -> FastAPI:
    api_base_path = _api_base_path()
    app = FastAPI(
        title=settings.app_name,
        docs_url=f"{api_base_path}/docs",
        redoc_url=f"{api_base_path}/redoc",
        openapi_url=f"{api_base_path}/openapi.json",
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix=api_base_path)
    return app


app = create_app()
