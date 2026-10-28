"""[모듈 2] 쿼리 파싱 및 라우팅 (담당: 공지혁)
LLM을 '자연어 답변 생성기'가 아니라 '구조화 파서'로 사용 (02_쿼리_파싱_및_라우팅_모듈.pdf)
출력: 실행 계획 JSON (question_type / answer_policy / entities / 라우팅 정보)
"""
from __future__ import annotations
import re
from typing import Optional, Literal
from pydantic import BaseModel, Field

# ============================================================================
# [모듈 2] 쿼리 파싱 및 라우팅 (FN-002 질문 유형 분류, FN-003 조건 추출)
#   - LLM을 '자연어 답변 생성기'가 아니라 '구조화 파서'로 사용 (모듈 2 문서 §3)
#   - 출력: 실행 계획 JSON (question_type / answer_policy / entities / 라우팅)
# ============================================================================
QUESTION_TYPES = ["rent_lookup", "condition_compare", "region_compare",
                  "region_recommendation", "transport_access", "rent_with_transport",
                  "policy_info", "restricted", "unknown_or_incomplete"]
ANSWER_POLICIES = ["allowed", "conditional_allowed", "denied", "need_clarification"]


class ParsedEntities(BaseModel):
    regions: list[str] = Field(default_factory=list, description="자치구명 또는 법정동명 목록 (예: ['성북구 안암동'])")
    deposit: Optional[int] = Field(default=None, description="보증금 (만원 단위). 없으면 null")
    monthly_rent: Optional[int] = Field(default=None, description="월세 (만원 단위). 없으면 null")
    period: str = Field(default="recent_1_year", description="recent_1_year 또는 recent_3_years")
    property_type: Optional[Literal["아파트", "연립다세대", "오피스텔", "단독다가구"]] = Field(
        default=None, description="주택 유형. 빌라/다세대→연립다세대, 원룸→단독다가구. 언급 없으면 null")
    rent_type: Optional[Literal["전세", "월세"]] = Field(default=None, description="전월세 구분. 언급 없으면 null")
    traffic_required: bool = Field(default=False, description="교통 접근성 분석 필요 여부")
    comparison_targets: list[str] = Field(default_factory=list, description="비교 대상 지역 목록")
    budget_limit: Optional[int] = Field(default=None, description="추천 조건의 월세 상한 (만원)")


class ExecutionPlanLLM(BaseModel):
    """LLM이 채우는 부분만 정의. 라우팅/필수값 검증은 코드(결정론)가 수행."""
    question_type: Literal[tuple(QUESTION_TYPES)] = Field(description="질문 유형 (선택지 중 하나만)")
    restriction_reason: Optional[str] = Field(default=None, description="제한 질문일 경우 제한 사유 (영문 스네이크케이스)")
    entities: ParsedEntities


ROUTER_SYSTEM_PROMPT = """너는 서울시 전월세 시세 분석 서비스의 부동산 질문 분류기이다.
사용자 질문을 분석하여 반드시 정해진 JSON 스키마로만 출력하라.

[질문 유형 — 반드시 아래 선택지 중 하나]
- rent_lookup: 특정 지역의 전월세 시세 조회 ("안암동 월세 수준 알려줘", 과거 추세 포함)
- condition_compare: 사용자의 보증금·월세 조건이 비싼지 비교 ("보증금 1000, 월세 60이면 비싸?")
- region_compare: 여러 지역의 전월세 수준 비교 ("성북구랑 마포구 중 어디가 비싸?")
- region_recommendation: 조건에 맞는 지역 후보 추천 ("월세 60 이하 지역 추천해줘")
- transport_access: 교통 접근성 질문 ("안암동 주변에 역 많아?")
- rent_with_transport: 전월세와 교통을 함께 질문
- policy_info: 데이터 출처/한계/통계 의미 질문 ("중앙값이 왜 중요해?")
- restricted: 서비스 범위 밖 질문
- unknown_or_incomplete: 의도 불명확

[제한(restricted) 판단 기준 — 키워드가 아니라 의도 기준]
- 실시간 매물 탐색 (지금 계약 가능한 방) → restriction_reason: realtime_listing_not_supported
- 전세사기·계약 위험 판정 (특정 매물/계약 판단) → fraud_judgement_not_supported
- 미래 가격 예측 ("내년에 오를까?") → future_price_prediction_not_supported
- 투자 판단 → investment_judgement_not_supported
- 법률 판단 → legal_judgement_not_supported
- 정확한 통근시간/경로 계산 → commute_time_not_supported
- 민간 플랫폼(네이버부동산 등) 기준 요청 → private_platform_not_supported
주의: "최근 3년 동안 월세가 올랐는지"는 과거 데이터 분석이므로 rent_lookup(허용)이다.
주의: 전세사기 '일반 예방 정보' 질문은 policy_info로 허용한다.

[파라미터 추출 규칙]
- 지역명: 자치구/법정동을 그대로 추출 (예: "성북구 안암동"). 없으면 빈 리스트.
- 보증금/월세/예산: 만원 단위 정수 (예: "보증금 1000만 원"→1000, "월세 60"→60).
- "월세 N 이하 지역 추천" 류는 budget_limit=N.
- 빌라/다세대→연립다세대, 원룸→단독다가구 로 매핑.
- 기간 언급 없으면 recent_1_year, "최근 3년" 언급 시 recent_3_years.
- 교통(역/정류장/교통) 언급 시 traffic_required=true.
"""

# 질문 유형 → (라우팅 모듈, 필요 계산, RAG 대상, 기본 답변 정책)  — 모듈 2 문서 §6, §9
ROUTING_TABLE = {
    "rent_lookup": (
        ["rent_stats_engine", "policy_rag_engine"],
        ["transaction_count", "average_monthly_rent", "median_monthly_rent"],
        ["data_source_policy", "rent_data_limitations", "statistics_explanation"],
        "allowed"),
    "condition_compare": (
        ["rent_condition_compare_engine", "policy_rag_engine"],
        ["transaction_count", "average_monthly_rent", "median_monthly_rent",
         "p25_monthly_rent", "p75_monthly_rent", "user_rent_percentile"],
        ["data_source_policy", "rent_data_limitations", "statistics_explanation"],
        "allowed"),
    "region_compare": (
        ["region_compare_engine", "policy_rag_engine"],
        ["region_transaction_count", "region_average_monthly_rent", "region_median_monthly_rent"],
        ["data_source_policy", "rent_data_limitations", "statistics_explanation"],
        "allowed"),
    "region_recommendation": (
        ["region_recommendation_engine", "policy_rag_engine"],
        ["region_median_monthly_rent", "region_transaction_count"],
        ["data_source_policy", "rent_data_limitations"],
        "conditional_allowed"),
    "transport_access": (
        ["transport_access_engine", "policy_rag_engine"],
        ["subway_station_count_500m", "bus_stop_count_300m",
         "nearest_subway_station_distance", "nearest_bus_stop_distance"],
        ["data_source_policy", "transport_limitations"],
        "conditional_allowed"),
    "rent_with_transport": (
        ["rent_condition_compare_engine", "transport_access_engine", "policy_rag_engine"],
        ["transaction_count", "average_monthly_rent", "median_monthly_rent", "user_rent_percentile",
         "subway_station_count_500m", "bus_stop_count_300m",
         "nearest_subway_station_distance", "nearest_bus_stop_distance"],
        ["data_source_policy", "rent_data_limitations", "transport_limitations"],
        "conditional_allowed"),
    "policy_info": (
        ["policy_rag_engine"], [],
        ["data_source_policy", "statistics_explanation", "rent_data_limitations"],
        "allowed"),
    "restricted": (["refusal_handler"], [], ["unsupported_question_policy"], "denied"),
    "unknown_or_incomplete": (["clarification_handler"], [], [], "need_clarification"),
}

# 질문 유형별 필수값 (모듈 2 문서 §11)
REQUIRED_FIELDS = {
    "rent_lookup": ["region"],
    "condition_compare": ["region", "monthly_rent"],
    "region_compare": ["two_regions"],
    "region_recommendation": ["budget"],
    "transport_access": ["region"],
    "rent_with_transport": ["region"],
}


class QueryParserRouter:
    """사용자 자연어 질문 → 실행 계획 JSON 변환 (모듈 2 전체 흐름 구현)."""

    def __init__(self, llm_client):
        self.llm = llm_client

    # --- 전처리: 공백 정리, 숫자 단위 정규화 ---------------------------------
    @staticmethod
    def normalize_query(q: str) -> str:
        q = re.sub(r"\s+", " ", q).strip()
        # "1,000만 원" → "1000만원" 식 정규화
        q = re.sub(r"(\d),(\d{3})", r"\1\2", q)
        q = q.replace("만 원", "만원")
        return q

    def parse_and_route(self, user_query: str) -> dict:
        nq = self.normalize_query(user_query)

        # 1) LLM 기반 의도 분류 + 파라미터 추출 (구조화 출력 강제)
        parsed: ExecutionPlanLLM = self.llm.parse_query(ROUTER_SYSTEM_PROMPT, nq, ExecutionPlanLLM)
        ents = parsed.entities

        # 2) 후처리 보정 — 숫자/지역 정규식 보정 (모듈 2 문서 §3 '후처리 로직')
        ents = self._postprocess_entities(nq, parsed.question_type, ents)

        qtype = parsed.question_type
        modules, calcs, rag_targets, policy = ROUTING_TABLE[qtype]

        # 3) 제한 질문 → 거절 계획
        if qtype == "restricted":
            return self._plan(user_query, qtype, "denied", ents, ["refusal_handler"], [],
                              ["unsupported_question_policy"], [],
                              parsed.restriction_reason or "out_of_scope")

        # 4) 필수값 검증 (코드 결정론 — LLM에 맡기지 않음)
        missing = self._validate_required_fields(qtype, ents)
        if missing:
            return self._plan(user_query, qtype, "need_clarification", ents,
                              ["clarification_handler"], [], [], missing, None)

        # 5) 정상 실행 계획
        return self._plan(user_query, qtype, policy, ents, modules, calcs, rag_targets, [], None)

    # --- 후처리 보정 ----------------------------------------------------------
    @staticmethod
    def _postprocess_entities(nq: str, qtype: str, ents: ParsedEntities) -> ParsedEntities:
        # 보증금/월세 정규식 백업 추출 (LLM 누락 대비)
        if ents.deposit is None:
            m = re.search(r"보증금\s*(\d+)\s*만?원?", nq)
            if m:
                ents.deposit = int(m.group(1))
        if ents.monthly_rent is None and qtype != "region_recommendation":
            m = re.search(r"월세\s*(\d+)\s*만?원?", nq)
            if m:
                ents.monthly_rent = int(m.group(1))
        if qtype == "region_recommendation" and ents.budget_limit is None:
            m = re.search(r"(\d+)\s*만?원?\s*이하", nq)
            if m:
                ents.budget_limit = int(m.group(1))
        # 기간 보정
        if "3년" in nq:
            ents.period = "recent_3_years"
        # 교통 키워드 보정
        if any(k in nq for k in ["교통", "역 ", "지하철", "정류장", "버스"]):
            ents.traffic_required = True
        # 지역 비교 대상 동기화
        if qtype == "region_compare" and not ents.comparison_targets:
            ents.comparison_targets = list(ents.regions)
        return ents

    @staticmethod
    def _validate_required_fields(qtype: str, ents: ParsedEntities) -> list[str]:
        missing = []
        req = REQUIRED_FIELDS.get(qtype, [])
        if "region" in req and not ents.regions:
            missing.append("region")
        if "two_regions" in req and len(set(ents.regions) | set(ents.comparison_targets)) < 2:
            missing.append("comparison_regions")
        if "monthly_rent" in req and ents.monthly_rent is None:
            missing.append("monthly_rent")
        if "budget" in req and ents.budget_limit is None and ents.monthly_rent is None:
            missing.append("budget_limit")
        return missing

    @staticmethod
    def _plan(original_query, qtype, policy, ents, modules, calcs, rag_targets, missing, reason) -> dict:
        return {
            "original_query": original_query,
            "question_type": qtype,
            "answer_policy": policy,
            "entities": ents.model_dump(),
            "required_modules": modules,
            "required_calculations": calcs,
            "rag_targets": rag_targets,
            "missing_fields": missing,
            "restriction_reason": reason,
        }
