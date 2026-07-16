from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        root_path=settings.root_path,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(api_router)
    return app


app = create_app()
