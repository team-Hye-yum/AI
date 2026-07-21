from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from app.services.ai_review_cache import AiReviewCache, cache_key
from app.services.ai_review_opinions import (
    MODEL_VERSION,
    generate_review_opinions,
    knowledge_base_version,
)

router = APIRouter(prefix="/review")


class PatentSummary(BaseModel):
    activeRegisteredPatentCount: int | None = None
    latestRegistrationYear: int | None = None


class CompanyProfile(BaseModel):
    companyName: str | None = None
    industryName: str | None = None
    industryDescription: str | None = None
    industryBrief: str | None = None
    ksicCode: str | None = None
    mainProduct: str | None = None
    regionName: str | None = None
    supportedSalesGrowthRate: float | None = None
    companyActivityText: str | None = None
    businessPurposes: list[str] = Field(default_factory=list)
    ntisProjectNames: list[str] = Field(default_factory=list)
    patentSummary: PatentSummary | None = None


class SupportSummary(BaseModel):
    totalSupportCount: int | None = None
    marketExpansionSupportCount: int | None = None
    techRnDSupportCount: int | None = None
    jobCreationSupportSelected: bool | None = None
    recentSupportTexts: list[str] = Field(default_factory=list)


class EmploymentSummary(BaseModel):
    observationYear: int | None = None
    employeeCountPreviousYear: int | None = None
    employeeCountObservationYear: int | None = None
    pensionSubscriberCount: int | None = None
    pensionNewHireCount: int | None = None
    pensionRetireeCount: int | None = None
    employeeTurnoverRate: float | None = None
    turnoverRatePeriod: str | None = None
    turnoverBenchmarkRate: float | None = None
    benchmarkPeriod: str | None = None
    benchmarkSource: str | None = None


class ReviewOptions(BaseModel):
    maxEvidenceCount: int | None = None


class AiReviewOpinionRequest(BaseModel):
    companyId: int
    profile: CompanyProfile | None = None
    supportSummary: SupportSummary | None = None
    employmentSummary: EmploymentSummary | None = None
    options: ReviewOptions | None = None


class AiReviewOpinionMeta(BaseModel):
    cached: bool
    modelVersion: str


class AiReviewOpinionResponse(BaseModel):
    display: bool
    budgetMismatchLine: str | None
    employmentCarouselLine: str | None
    meta: AiReviewOpinionMeta


@router.post("/opinions", response_model=AiReviewOpinionResponse)
def review_opinions(request_body: dict[str, Any]) -> dict[str, Any]:
    try:
        opinion_request = AiReviewOpinionRequest.model_validate(request_body)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    kb_version = knowledge_base_version()
    key = cache_key(
        request_body=request_body,
        model_version=MODEL_VERSION,
        knowledge_base_version=kb_version,
    )
    cache = AiReviewCache()
    cached = cache.get(key)
    if cached is not None:
        return mark_cached_response(cached)

    response = generate_review_opinions(opinion_request.model_dump(mode="json"))
    cache.set(key, response)
    return response


def mark_cached_response(response: dict[str, Any]) -> dict[str, Any]:
    meta = response.setdefault("meta", {})
    meta["cached"] = True
    meta.setdefault("modelVersion", MODEL_VERSION)
    return response
