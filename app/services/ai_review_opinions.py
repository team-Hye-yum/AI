from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from typing import Any

from app.core.config import settings


MODEL_VERSION = "ai-review-opinions-v3"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "ai_review_opinions.txt"
MARKET_KEYWORDS = ["전시", "수출", "마케팅", "판로", "해외", "브랜드", "홍보", "시장", "바이어"]
TECH_KEYWORDS = ["소프트웨어", "시스템", "플랫폼", "앱", "정보", "데이터", "AI", "ICT", "기술", "개발", "연구", "R&D"]
FORBIDDEN_TERMS = [
    "경고",
    "부적합",
    "탈락",
    "제재",
    "위험 기업",
    "부실 기업",
    "예산 낭비 기업",
    "회전문 기업",
    "지원금 목적 해고",
    "고의적 해고",
]


def knowledge_base_version() -> str:
    state_path = Path(settings.seed_training_data_dir) / settings.seed_training_data_state_file
    if not state_path.exists():
        return "unknown"
    try:
        state = loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    fragments: list[str] = []
    files = state.get("files", {})
    items = files.values() if isinstance(files, dict) else files
    for item in items:
        if not isinstance(item, dict):
            continue
        dataset_type = item.get("dataset_type") or item.get("datasetType")
        sha256 = item.get("sha256")
        if dataset_type and sha256:
            fragments.append(f"{dataset_type}:{sha256}")
    return "|".join(sorted(fragments)) or "unknown"


def generate_review_opinions(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = _candidate_review_opinions(payload)
    if not candidate["display"]:
        return candidate

    llm_response = _generate_review_opinions_with_llm(payload, candidate)
    if llm_response is not None:
        return llm_response
    return candidate


def _candidate_review_opinions(payload: dict[str, Any]) -> dict[str, Any]:
    budget_line = _budget_mismatch_candidate_line(payload)
    employment_line = _employment_candidate_line(payload)
    return {
        "display": budget_line is not None or employment_line is not None,
        "budgetMismatchLine": budget_line,
        "employmentCarouselLine": employment_line,
        "meta": {
            "cached": False,
            "modelVersion": MODEL_VERSION,
        },
    }


def _budget_mismatch_candidate_line(payload: dict[str, Any]) -> str | None:
    profile = payload.get("profile") or {}
    support = payload.get("supportSummary") or {}
    activity_text = _joined(
        [
            profile.get("industryName"),
            profile.get("industryDescription"),
            profile.get("industryBrief"),
            profile.get("mainProduct"),
            profile.get("companyActivityText"),
            " ".join(profile.get("businessPurposes") or []),
            " ".join(profile.get("ntisProjectNames") or []),
        ]
    )
    market_count = _as_int(support.get("marketExpansionSupportCount")) or 0
    tech_count = _as_int(support.get("techRnDSupportCount")) or 0
    supported_sales_growth_rate = _as_float(profile.get("supportedSalesGrowthRate"))

    # v1.1부터는 화면 노출을 줄이기 위해 반복 수혜와 음수 성과가 함께 있을 때만 후보로 둔다.
    repeated_support_observed = market_count >= 2
    low_outcome_observed = supported_sales_growth_rate is not None and supported_sales_growth_rate < 0
    if not (repeated_support_observed and low_outcome_observed):
        return None

    activity_relation_weak = _activity_relation_weak(activity_text, market_count, tech_count)
    if not activity_relation_weak:
        return None

    industry_name = _short(profile.get("industryName")) or "해당 기업"
    outcome = ""
    if supported_sales_growth_rate is not None:
        outcome = f" 지원 이후 매출 변화율 {supported_sales_growth_rate:.1f}%가 함께 관찰되어"
    return f"예산 미스매칭 검토 필요: {industry_name}의 주된 활동과 시장진출성 지원 이력의 연결성이 낮아 보이며,{outcome} 지원 목적 적합성 확인이 필요합니다."


def _employment_candidate_line(payload: dict[str, Any]) -> str | None:
    support = payload.get("supportSummary") or {}
    employment = payload.get("employmentSummary") or {}
    job_creation_support_selected = support.get("jobCreationSupportSelected") is True
    retiree_count = _as_int(employment.get("pensionRetireeCount"))
    average_workforce = _average_workforce(employment)
    turnover_rate = None
    if average_workforce and retiree_count is not None:
        turnover_rate = retiree_count * 100.0 / average_workforce
    if turnover_rate is None:
        turnover_rate = _as_float(employment.get("employeeTurnoverRate"))

    if retiree_count is None or retiree_count < 2 or not average_workforce or turnover_rate is None:
        return None
    if turnover_rate < 30.0:
        return None

    observation_year = employment.get("observationYear") or "관측연도"
    if job_creation_support_selected:
        return f"고용 유지 검토 필요: {observation_year}년 일자리창출 지원 이력 이후 퇴직자 {retiree_count}명, 회전율 {turnover_rate:.1f}%가 관찰되어 고용 유지 실적 확인이 필요합니다."
    return f"고용 변동 검토 필요: {observation_year}년 퇴직자 {retiree_count}명, 회전율 {turnover_rate:.1f}%가 관찰되어 고용 변동 원인 확인이 필요합니다."


def _generate_review_opinions_with_llm(
    payload: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
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
            input=dumps(
                {
                    "ruleCandidate": candidate,
                    "companyPayload": payload,
                    "strictInstruction": (
                        "ruleCandidate에서 null인 문구는 절대 생성하지 말고, "
                        "ruleCandidate에 있는 문구는 null로 낮추지 말고 문장만 다듬어라."
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        return _validated_llm_response(response.output_text, candidate)
    except Exception:
        return None


def _validated_llm_response(output_text: str, candidate: dict[str, Any]) -> dict[str, Any] | None:
    try:
        parsed = loads(output_text)
    except (TypeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    budget_line = _validated_line(
        parsed.get("budgetMismatchLine"),
        candidate.get("budgetMismatchLine"),
        "예산 미스매칭 검토 필요:",
    )
    employment_prefix = _line_prefix(candidate.get("employmentCarouselLine"))
    employment_line = _validated_line(
        parsed.get("employmentCarouselLine"),
        candidate.get("employmentCarouselLine"),
        employment_prefix,
    )
    budget_line = budget_line or candidate.get("budgetMismatchLine")
    employment_line = employment_line or candidate.get("employmentCarouselLine")
    return {
        "display": budget_line is not None or employment_line is not None,
        "budgetMismatchLine": budget_line,
        "employmentCarouselLine": employment_line,
        "meta": {
            "cached": False,
            "modelVersion": MODEL_VERSION,
        },
    }


def _validated_line(value: Any, candidate_value: Any, required_prefix: str | None) -> str | None:
    if candidate_value is None or value is None:
        return None
    if not isinstance(value, str):
        return None
    line = value.strip()
    if not line or any(term in line for term in FORBIDDEN_TERMS):
        return None
    if required_prefix and not line.startswith(required_prefix):
        return None
    return _short(line, 180)


def _line_prefix(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if value.startswith("고용 유지 검토 필요:"):
        return "고용 유지 검토 필요:"
    if value.startswith("고용 변동 검토 필요:"):
        return "고용 변동 검토 필요:"
    return None


def _activity_relation_weak(activity_text: str, market_count: int, tech_count: int) -> bool:
    activity_has_market = any(keyword in activity_text for keyword in MARKET_KEYWORDS)
    activity_has_tech = any(keyword.lower() in activity_text.lower() for keyword in TECH_KEYWORDS)
    if activity_has_market:
        return False
    if activity_has_tech and market_count >= tech_count:
        return True
    return market_count > tech_count and not activity_has_market


def _average_workforce(employment: dict[str, Any]) -> float | None:
    previous_count = _as_int(employment.get("employeeCountPreviousYear"))
    current_count = _as_int(employment.get("employeeCountObservationYear"))
    if previous_count is not None and current_count is not None:
        return (previous_count + current_count) / 2.0
    subscriber_count = _as_int(employment.get("pensionSubscriberCount"))
    if subscriber_count is not None:
        return float(subscriber_count)
    return float(current_count) if current_count is not None else None


def _joined(values: list[Any]) -> str:
    return " ".join(str(value).strip() for value in values if value is not None and str(value).strip())


def _short(value: Any, max_length: int = 34) -> str:
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
