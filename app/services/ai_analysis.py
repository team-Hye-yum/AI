from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from typing import Any

from app.core.config import settings


MODEL_VERSION = "company-ai-analysis-v4"
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
    analysis_markdown = parsed.get("analysisMarkdown")
    if not isinstance(analysis_markdown, str) or not analysis_markdown.strip():
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

    analysis_markdown = _clean_markdown(analysis_markdown)
    if not analysis_markdown or any(term in analysis_markdown for term in FORBIDDEN_TERMS):
        return None

    return _response(analysis_lines, analysis_markdown)


def _fallback_response(payload: dict[str, Any]) -> dict[str, Any]:
    profile = payload.get("profile") or {}
    capabilities = payload.get("capabilities") or {}
    financials = payload.get("financials") or {}
    industry_comparison = payload.get("industryComparison") or {}
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
    supported_sales_growth_rate = _as_float(financials.get("supportedSalesGrowthRate"))
    debt_ratio = _as_float(financials.get("debtRatio"))
    government_rnd_dependency = _as_float(financials.get("governmentRndDependency"))
    latest_rnd_expense = _as_int(financials.get("latestRndExpense"))
    previous_employee_count = _as_int(employment.get("employeeCountPreviousYear"))
    observation_employee_count = _as_int(employment.get("employeeCountObservationYear"))
    industry_summary = _industry_comparison_summary(industry_comparison)
    industry_gap_rate = _as_float(industry_comparison.get("gapRate"))
    company_industry_change_rate = _as_float(industry_comparison.get("companyChangeRate"))
    industry_change_rate = _as_float(industry_comparison.get("industryChangeRate"))

    identity_evidence = []
    if business_purpose:
        identity_evidence.append(f"사업목적 '{_short(business_purpose, 24)}'")
    if ntis_count:
        identity_evidence.append(f"NTIS {ntis_count}건")
    if patent_count:
        identity_evidence.append(f"등록특허 {patent_count}건")
    identity_suffix = "와 ".join(identity_evidence[:2]) if identity_evidence else "사업목적과 기술 이력"

    priority_metrics = _priority_metrics(
        sales_growth_rate=sales_growth_rate,
        supported_sales_growth_rate=supported_sales_growth_rate,
        debt_ratio=debt_ratio,
        government_rnd_dependency=government_rnd_dependency,
        latest_rnd_expense=latest_rnd_expense,
        previous_employee_count=previous_employee_count,
        observation_employee_count=observation_employee_count,
        turnover_rate=turnover_rate,
        ntis_count=ntis_count,
        patent_count=patent_count,
        recent_support_count=recent_support_count,
        industry_gap_rate=industry_gap_rate,
        company_industry_change_rate=company_industry_change_rate,
        industry_change_rate=industry_change_rate,
    )

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
    first_metric = priority_metrics[0] if priority_metrics else "지원 이력과 성과 흐름"
    second_metric = priority_metrics[1] if len(priority_metrics) > 1 else performance_suffix
    third_metric = priority_metrics[2] if len(priority_metrics) > 2 else employment_suffix

    analysis_lines = [
        {
            "type": "IDENTITY",
            "line": f"{industry_name} 기반 기업으로, {identity_suffix}을 함께 보며 실제 활동 영역을 먼저 파악하는 것이 좋습니다.",
        },
        {
            "type": "PERFORMANCE",
            "line": f"{industry_summary} {first_metric}을 함께 보며 지원 이력과 성과 흐름을 연결해 보는 것이 적절합니다.",
        },
        {
            "type": "EMPLOYMENT_SUPPORT",
            "line": f"{second_metric}은 단독 해석보다 지원 시점과 함께 확인해 흐름을 보는 것이 좋습니다.",
        },
    ]
    analysis_markdown = "\n\n".join(
        [
            "## 기업 개요",
            f"- **{industry_name}** 기반 기업으로, {identity_suffix}을 함께 보며 실제 활동 영역을 먼저 파악하는 것이 좋습니다.",
            "## 산업 대비 흐름",
            f"- {industry_summary}",
            "## 성과 흐름",
            f"- **{first_metric}**이 가장 먼저 볼 지표로 보이며, 지원 이력과 같은 시점에 놓고 흐름을 확인하는 것이 적절합니다.",
            f"- 다음으로는 **{second_metric}**을 함께 보면 단순 규모보다 변화의 방향을 더 분명하게 읽을 수 있습니다.",
            "## 우선 확인할 지표",
            f"- **{first_metric}**은 이 기업 대시보드에서 우선 확인할 값으로 정리됩니다.",
            f"- **{third_metric}**은 앞선 지표를 보완해 활동 역량 또는 사업화 흐름을 확인하는 보조 항목으로 볼 수 있습니다.",
            "## 참고 사항",
            "- 본 리포트는 입력된 대시보드 데이터를 해석하기 위한 참고자료이며, 평가 결과나 지원 여부를 제시하지 않습니다.",
        ]
    )
    return _response(analysis_lines, analysis_markdown)


def _response(analysis_lines: list[dict[str, str]], analysis_markdown: str) -> dict[str, Any]:
    return {
        "analysisMarkdown": analysis_markdown,
        "analysisLines": analysis_lines,
        "meta": {
            "cached": False,
            "modelVersion": MODEL_VERSION,
        },
    }


def _clean_line(value: str, max_length: int = 180) -> str:
    return _short(" ".join(value.split()), max_length)


def _clean_markdown(value: str, max_length: int = 2200) -> str:
    text = value.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip()


def _priority_metrics(
    *,
    sales_growth_rate: float | None,
    supported_sales_growth_rate: float | None,
    debt_ratio: float | None,
    government_rnd_dependency: float | None,
    latest_rnd_expense: int | None,
    previous_employee_count: int | None,
    observation_employee_count: int | None,
    turnover_rate: float | None,
    ntis_count: int,
    patent_count: int,
    recent_support_count: int,
    industry_gap_rate: float | None,
    company_industry_change_rate: float | None,
    industry_change_rate: float | None,
) -> list[str]:
    candidates: list[tuple[float, str]] = []
    if industry_gap_rate is not None:
        candidates.append((abs(industry_gap_rate) + 90, f"산업 대비 매출 격차 {industry_gap_rate:+.1f}%p"))
    if company_industry_change_rate is not None and industry_change_rate is not None:
        candidates.append((
            abs(company_industry_change_rate - industry_change_rate) + 60,
            f"기업 {company_industry_change_rate:+.1f}%·산업 {industry_change_rate:+.1f}% 흐름",
        ))
    if debt_ratio is not None:
        candidates.append((abs(debt_ratio) + (80 if debt_ratio >= 100 else 0), f"부채비율 {debt_ratio:.1f}%"))
    if sales_growth_rate is not None:
        candidates.append((abs(sales_growth_rate) + 40, f"매출 성장률 {sales_growth_rate:.1f}%"))
    if supported_sales_growth_rate is not None:
        candidates.append((abs(supported_sales_growth_rate) + 35, f"지원 후 매출 성장률 {supported_sales_growth_rate:.1f}%"))
    if government_rnd_dependency is not None:
        candidates.append((abs(government_rnd_dependency) + 20, f"정부 R&D 의존도 {government_rnd_dependency:.1f}%"))
    if turnover_rate is not None:
        candidates.append((abs(turnover_rate) + 15, f"고용 회전율 {turnover_rate:.1f}%"))
    employee_change = _employee_change_rate(previous_employee_count, observation_employee_count)
    if employee_change is not None:
        candidates.append((abs(employee_change) + 25, f"종사자수 변화율 {employee_change:.1f}%"))
    if latest_rnd_expense is not None and latest_rnd_expense > 0:
        candidates.append((30, "최근 R&D 비용"))
    if recent_support_count:
        candidates.append((recent_support_count * 8, f"지원 이력 {recent_support_count}건"))
    if ntis_count:
        candidates.append((ntis_count * 6, f"NTIS 과제 {ntis_count}건"))
    if patent_count:
        candidates.append((patent_count * 5, f"등록특허 {patent_count}건"))

    return [label for _, label in sorted(candidates, key=lambda item: item[0], reverse=True)[:3]]


def _industry_comparison_summary(industry_comparison: dict[str, Any]) -> str:
    provided_summary = industry_comparison.get("summary")
    if isinstance(provided_summary, str) and provided_summary.strip():
        return provided_summary.strip()

    company_change = _as_float(industry_comparison.get("companyChangeRate"))
    industry_change = _as_float(industry_comparison.get("industryChangeRate"))
    gap = _as_float(industry_comparison.get("gapRate"))
    base_year = _as_int(industry_comparison.get("baseYear")) or 2021
    latest_year = _as_int(industry_comparison.get("latestYear")) or 2024
    if company_change is None or industry_change is None or gap is None:
        return "산업 대비 매출 흐름은 비교 데이터가 제한적이므로 기업 내부 지표 중심으로 확인하는 것이 좋습니다."
    period = f"{base_year}~{latest_year}년"
    if industry_change <= -15 and gap >= 10:
        return f"{period} 산업은 {industry_change:+.1f}% 흐름이었지만 기업은 {company_change:+.1f}%로 상대적으로 방어한 구간입니다."
    if industry_change >= 15 and gap <= -10:
        return f"{period} 산업은 {industry_change:+.1f}% 확장됐지만 기업은 {company_change:+.1f}%에 그쳐 시장 흐름을 따라갔는지 확인할 수 있습니다."
    if gap >= 10:
        return f"{period} 기업 흐름이 산업보다 {gap:+.1f}%p 높아 기업 고유 성장 요인을 먼저 볼 수 있습니다."
    if gap <= -10:
        return f"{period} 기업 흐름이 산업보다 {abs(gap):.1f}%p 낮아 업종 환경과 내부 성과를 나누어 볼 수 있습니다."
    return f"{period} 기업과 산업 흐름의 격차가 {gap:+.1f}%p로 크지 않아 재무·고용·기술 지표를 함께 봐야 합니다."


def _employee_change_rate(previous: int | None, current: int | None) -> float | None:
    if previous is None or current is None or previous == 0:
        return None
    return (current - previous) * 100 / previous


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
