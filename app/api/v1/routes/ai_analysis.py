from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from app.services.ai_analysis import MODEL_VERSION, generate_ai_analysis
from app.services.ai_analysis_cache import AiAnalysisCache, cache_key
from app.services.ai_review_opinions import knowledge_base_version

router = APIRouter(prefix="/companies")


class Profile(BaseModel):
    industryName: str | None = None
    industryBrief: str | None = None
    ksicCode: str | None = None
    regionName: str | None = None
    establishedDate: str | None = None
    companySize: str | None = None
    mainProduct: str | None = None


class PatentSummary(BaseModel):
    activeRegisteredPatentCount: int | None = None
    latestRegistrationYear: int | None = None


class ResearchOrganizations(BaseModel):
    hasResearchLab: bool | None = None
    hasRndDepartment: bool | None = None
    researcherCount: int | None = None


class Capabilities(BaseModel):
    businessPurposes: list[str] = Field(default_factory=list)
    ntisProjectNames: list[str] = Field(default_factory=list)
    ntisProjectCount: int | None = None
    patentSummary: PatentSummary | None = None
    researchOrganizations: ResearchOrganizations | None = None


class Financials(BaseModel):
    latestYear: int | None = None
    latestSalesAmount: int | None = None
    salesGrowthRate: float | None = None
    supportedSalesGrowthRate: float | None = None
    debtRatio: float | None = None
    governmentRndDependency: float | None = None
    latestRndExpense: int | None = None


class Employment(BaseModel):
    observationYear: int | None = None
    employeeCountPreviousYear: int | None = None
    employeeCountObservationYear: int | None = None
    pensionSubscriberCount: int | None = None
    pensionNewHireCount: int | None = None
    pensionRetireeCount: int | None = None
    employeeTurnoverRate: float | None = None


class SupportHistory(BaseModel):
    totalSupportCount: int | None = None
    marketExpansionSupportCount: int | None = None
    techRnDSupportCount: int | None = None
    jobCreationSupportSelected: bool | None = None
    recentSupportTexts: list[str] = Field(default_factory=list)


class Options(BaseModel):
    lineCount: int | None = 3


class AiAnalysisRequest(BaseModel):
    companyId: int
    profile: Profile | None = None
    capabilities: Capabilities | None = None
    financials: Financials | None = None
    employment: Employment | None = None
    supportHistory: SupportHistory | None = None
    options: Options | None = None


class AnalysisLine(BaseModel):
    type: Literal["IDENTITY", "PERFORMANCE", "EMPLOYMENT_SUPPORT"]
    line: str


class AiAnalysisMeta(BaseModel):
    cached: bool
    modelVersion: str


class AiAnalysisResponse(BaseModel):
    analysisLines: list[AnalysisLine]
    meta: AiAnalysisMeta


@router.post("/analysis", response_model=AiAnalysisResponse)
def ai_analysis(request_body: dict[str, Any]) -> dict[str, Any]:
    try:
        analysis_request = AiAnalysisRequest.model_validate(request_body)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    kb_version = knowledge_base_version()
    key = cache_key(
        request_body=request_body,
        model_version=MODEL_VERSION,
        knowledge_base_version=kb_version,
    )
    cache = AiAnalysisCache()
    cached = cache.get(key)
    if cached is not None:
        return mark_cached_response(cached)

    response = generate_ai_analysis(analysis_request.model_dump(mode="json"))
    cache.set(key, response)
    return response


def mark_cached_response(response: dict[str, Any]) -> dict[str, Any]:
    meta = response.setdefault("meta", {})
    meta["cached"] = True
    meta.setdefault("modelVersion", MODEL_VERSION)
    return response
