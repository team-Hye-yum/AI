from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from typing import Any

from app.core.config import settings


MODEL_VERSION = "company-ai-analysis-v1"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "ai_analysis.txt"
ANALYSIS_TYPES = ["IDENTITY", "PERFORMANCE", "EMPLOYMENT_SUPPORT"]
FORBIDDEN_TERMS = [
    "검토 필요",
    "경고",
    "부적합",
    "탈락",
    "제재",
    "위험 기업",
    "부실 기업",
    "회전문 기업",
    "예산 낭비 기업",
]


def generate_ai_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    llm_response = _generate_ai_analysis_with_llm(payload)
    if llm_response is not None:
        return llm_response
    return _fallback_response(payload)


def _generate_ai_analysis_with_llm(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds)
        response = client.responses.create(
            model=settings.openai_model,
            instructions=prompt,
            input=dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        return _validated_response(response.output_text)
    except Exception:
        return None


def _validated_response(output_text: str) -> dict[str, Any] | None:
    try:
        parsed = loads(output_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    raw_lines = parsed.get("analysisLines")
    if not isinstance(raw_lines, list) or len(raw_lines) != 3:
        return None

    analysis_lines: list[dict[str, str]] = []
    for expected_type, item in zip(ANALYSIS_TYPES, raw_lines, strict=True):
        if not isinstance(item, dict) or item.get("type") != expected_type:
            return None
        line = item.get("line")
        if not isinstance(line, str):
            return None
        line = _clean_line(line)
        if not line or any(term in line for term in FORBIDDEN_TERMS):
            return None
        analysis_lines.append({"type": expected_type, "line": line})

    return _response(analysis_lines)


def _fallback_response(payload: dict[str, Any]) -> dict[str, Any]:
    profile = payload.get("profile") or {}
    capabilities = payload.get("capabilities") or {}
    financials = payload.get("financials") or {}
    employment = payload.get("employment") or {}
    support = payload.get("supportHistory") or {}

    industry_name = _short(profile.get("industryName")) or "업종 정보가 제한적인 기업"
    business_purpose = _first_text(capabilities.get("businessPurposes"))
    ntis_count = _as_int(capabilities.get("ntisProjectCount")) or 0
    patent_summary = capabilities.get("patentSummary") or {}
    patent_count = _as_int(patent_summary.get("activeRegisteredPatentCount")) or 0
    recent_support_count = _as_int(support.get("totalSupportCount")) or 0
    sales_growth_rate = _as_float(financials.get("salesGrowthRate"))
    latest_sales_amount = _as_int(financials.get("latestSalesAmount"))
    retiree_count = _as_int(employment.get("pensionRetireeCount"))
    turnover_rate = _as_float(employment.get("employeeTurnoverRate"))

    identity_evidence = []
    if business_purpose:
        identity_evidence.append(f"사업목적 '{_short(business_purpose, 24)}'")
    if ntis_count:
        identity_evidence.append(f"NTIS {ntis_count}건")
    if patent_count:
        identity_evidence.append(f"등록특허 {patent_count}건")
    identity_suffix = "와 ".join(identity_evidence[:2]) if identity_evidence else "사업목적과 기술 이력"

    performance_parts = []
    if recent_support_count:
        performance_parts.append(f"지원 이력 {recent_support_count}건")
    if sales_growth_rate is not None:
        performance_parts.append(f"매출 성장률 {sales_growth_rate:.1f}%")
    elif latest_sales_amount is not None:
        performance_parts.append("최근 매출 규모")
    if ntis_count or patent_count:
        performance_parts.append("R&D·특허 흐름")
    performance_suffix = ", ".join(performance_parts[:3]) or "지원 이력과 성과 흐름"

    employment_parts = []
    if retiree_count is not None:
        employment_parts.append(f"퇴직자 {retiree_count}명")
    if turnover_rate is not None:
        employment_parts.append(f"회전율 {turnover_rate:.1f}%")
    employment_suffix = ", ".join(employment_parts) if employment_parts else "연도별 고용 규모"

    return _response(
        [
            {
                "type": "IDENTITY",
                "line": f"{industry_name} 기반 기업으로, {identity_suffix}을 함께 보며 실제 활동 영역을 먼저 파악하는 것이 좋습니다.",
            },
            {
                "type": "PERFORMANCE",
                "line": f"{performance_suffix}을 나란히 비교해 지원 이후 성과 흐름을 확인하는 것이 적절합니다.",
            },
            {
                "type": "EMPLOYMENT_SUPPORT",
                "line": f"{employment_suffix}은 단독 해석보다 지원 시점과 함께 확인해 고용 흐름을 보는 것이 좋습니다.",
            },
        ]
    )


def _response(analysis_lines: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "analysisLines": analysis_lines,
        "meta": {
            "cached": False,
            "modelVersion": MODEL_VERSION,
        },
    }


def _clean_line(value: str, max_length: int = 180) -> str:
    return _short(" ".join(value.split()), max_length)


def _first_text(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _short(value: Any, max_length: int = 60) -> str:
    text = str(value).strip() if value is not None else ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
