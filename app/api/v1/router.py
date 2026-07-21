from fastapi import APIRouter

from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.review import router as review_router
from app.api.v1.routes.uploads import router as uploads_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(review_router, tags=["review"])
api_router.include_router(uploads_router, tags=["uploads"])
