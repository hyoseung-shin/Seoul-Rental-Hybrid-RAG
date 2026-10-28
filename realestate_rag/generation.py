"""[모듈 4] 답변 생성 및 검증 (담당: 신효승)"""
# ============================================================================
# [모듈 4] 답변 생성 및 검증 (FN-010, FN-011, FN-012)
#   - 모듈 4 문서의 의사코드를 그대로 따르되, '# have to rewrite' 스텁 전부 구현
#   - 흐름: 제한 게이트 → LLM 초안 생성 → 검증(4종) → 1회 자가수정 → 폴백 조립
# ============================================================================
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .engine import (EvidencePackage, PolicyDoc, QueryParams, RegionStat,
                     TransportInfo)
from .utils import format_number

# ---- 설정값 ------------------------------------------------------------------
MAX_RETRY = 1
STAT_TOLERANCE = 0.5   # 시세 통계 허용오차: 만원 단위 반올림 표시 정책 반영 ([결함] 해결)


@dataclass
class StatClaim:
    region: str
    label: str   # "avg" | "median" | "p25" | "p75"
    value: float


@dataclass
class VerifyReport:
    passed: bool
    errors: list[str] = field(default_factory=list)


# ============================================================================
# 메인 흐름
# ============================================================================
class GenerationVerificationModule:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.last_debug = {}

    def generate_and_verify(self, pkg: EvidencePackage, user_query: str) -> str:
        self.last_debug = {"attempts": 0, "fallback": False, "errors_history": []}

        # 0. 제한 게이트 (FN-011) + 추가질문 게이트 — LLM 자유생성 호출하지 않음
        if pkg.query_type == "제한형" or pkg.out_of_scope \
                or str(pkg.out_of_scope_reason).startswith("missing:"):
            return build_restricted_response(pkg)

        # 1. LLM 초안 생성 (FN-010) — 설명자 역할만, 계산 금지
        draft = self.run_generation(pkg, user_query)

        # 2. 검증 + 재시도 루프
        for attempt in range(MAX_RETRY + 1):
            self.last_debug["attempts"] = attempt + 1
            report = verify(draft, pkg, user_query)
            self.last_debug["errors_history"].append(list(report.errors))
            if report.passed:
                return finalize(draft, pkg)  # 한계·출처는 여기서 기계적으로 강제 부착
            if attempt < MAX_RETRY:
                draft = self.run_generation(pkg, user_query, feedback=report.errors)

        # 3. 끝까지 실패 → LLM 자유 텍스트 폐기, 검증된 수치만 기계 조립 (폴백)
        self.last_debug["fallback"] = True
        return build_safe_template_answer(pkg)

    def run_generation(self, pkg: EvidencePackage, user_query: str,
                       feedback: Optional[list[str]] = None) -> str:
        system_prompt = (
            "당신은 서울시 공공 부동산 데이터 기반 시세 분석 어시스턴트입니다.\n"
            "아래 [근거 패키지]의 내용만 사용해 사용자 질문에 답하세요.\n"
            "[엄격한 생성 규칙]\n"
            "1. 패키지에 없는 수치·지역·판단을 새로 만들지 마세요. 평균·거리·접근성을 직접 계산하거나 추정하지 마세요.\n"
            "2. 수치는 제공된 값 그대로 사용하고, 금액(만원)은 천 단위 콤마로 표기하세요 (예: 35000 → 35,000만 원).\n"
            "3. 미래 예측, 투자·법률 판단, 전세사기 판정, 실시간 매물 언급을 하지 마세요.\n"
            "4. 통계를 언급할 때는 반드시 해당 지역명을 같은 문장에 함께 쓰세요.\n"
            "5. 사용자 조건 비교 시 user_rent_percentile(백분위)과 p25/p75를 활용해 '비싼 편/보통/낮은 편'을 설명하세요.\n"
            "6. 데이터 출처·한계 문구는 시스템이 자동으로 덧붙이므로 본문에 따로 쓰지 마세요.\n"
            "7. 한국어로 3~6문장 분량으로 간결하게 답하세요."
        )
        llm_input = render_llm_input(pkg, user_query)
        if feedback:
            llm_input += "\n[수정요구] 이전 답안에서 다음 오류가 발견되었습니다. 패키지 수치만 사용해 다시 작성하세요:\n- " \
                         + "\n- ".join(feedback)
        return self.llm.generate(system_prompt, llm_input)


# ============================================================================
# 검증부 — 4종 검증 (a) 통계 정합성 (b) 교통 정합성 (c) 미근거 숫자 (d) 범위 이탈
# ============================================================================
def verify(draft: str, pkg: EvidencePackage, user_query: str) -> VerifyReport:
    errors: list[str] = []

    # 화이트리스트: 파라미터 + 사용자 인용값 (기간·반경·조건이 환각으로 오판되지 않도록)
    allowed_params: set[float] = set()
    for s in pkg.params.period + pkg.params.radius + pkg.params.user_conditions:
        allowed_params.update(_numbers_in(s))
    allowed_params |= echoed_from(user_query)
    allowed_params |= {1, 3, 25, 75, 100, 500, 300}  # 기간/분위 라벨/반경 상수

    grounded = flatten_values(pkg)

    # (a) 시세 통계 정합성 — avg/median/p25/p75, 지역명 키 매핑
    known_regions = [s.region for s in pkg.stats]
    for claim in extract_stat_claims(draft, known_regions):
        truth = lookup_stat(pkg.stats, claim.region, claim.label)
        if truth is None:
            # 추천 후보 중앙값 등 stats 외 산출값에 근거한 주장은 허용
            if float(claim.value) not in grounded:
                errors.append(f"UNGROUNDED_STAT: {claim.region}/{claim.label}={claim.value}")
        elif not within_tolerance(claim.value, truth, STAT_TOLERANCE):
            errors.append(f"STAT_MISMATCH: {claim.region}/{claim.label} "
                          f"(답변 {claim.value} ≠ 산출 {truth})")

    # (b) 교통 개수·거리 정합성 — tolerance = 0 (정확 일치). 단, 언급된 경우에만 대조.
    if pkg.transport is not None:
        t = pkg.transport
        checks = [(extract_count(draft, unit="역"), t.subway_count, "지하철역 수"),
                  (extract_count(draft, unit="정류장"), t.bus_count, "버스정류장 수"),
                  (extract_distance(draft, target="역"), t.nearest_subway_m, "최단 역 거리"),
                  (extract_distance(draft, target="정류장"), t.nearest_bus_m, "최단 정류장 거리")]
        for found, truth, name in checks:
            if found is not None and float(found) != float(truth):
                errors.append(f"TRANSPORT_MISMATCH: {name} (답변 {found} ≠ 산출 {truth})")

    # (c) 남은 미근거 숫자 = 화이트리스트에도 산출값에도 없는 수치
    for n in extract_numbers(draft):
        if n not in allowed_params and n not in grounded:
            errors.append(f"UNGROUNDED_NUMBER: {format_number(n)}")

    # (d) 범위 이탈 주장 탐지 (예측/투자/법률 판단, 실시간 매물 단정)
    if contains_prediction_or_judgement(draft):
        errors.append("OUT_OF_SCOPE_CLAIM")
    if mentions_realtime_listing(draft):
        errors.append("REALTIME_CLAIM")

    return VerifyReport(passed=(len(errors) == 0), errors=errors)


# ============================================================================
# 출력 조립 (FN-012) — 한계·출처를 의미 매칭이 아니라 '기계적으로' 부착
# ============================================================================
def finalize(draft: str, pkg: EvidencePackage) -> str:
    limits = required_limits_for(pkg.query_type)
    sources = list(dict.fromkeys(d.source for d in pkg.policy_docs))  # 중복 제거
    extra = ("\n".join("ℹ️ " + n for n in pkg.notes) + "\n") if pkg.notes else ""
    return draft.strip() + "\n\n" + extra + render_limit_block(limits) + render_source_block(sources)


def build_restricted_response(pkg: EvidencePackage) -> str:
    reason_ko = {
        "future_price_prediction_not_supported": "미래 가격 예측",
        "fraud_judgement_not_supported": "전세사기·계약 위험 판정",
        "investment_judgement_not_supported": "투자 판단",
        "legal_judgement_not_supported": "법률 판단",
        "realtime_listing_not_supported": "실시간 매물 탐색",
        "commute_time_not_supported": "정확한 통근시간 계산",
        "private_platform_not_supported": "민간 플랫폼 기준 답변",
    }.get(pkg.out_of_scope_reason, pkg.out_of_scope_reason or "서비스 범위 밖 질문")

    if str(pkg.out_of_scope_reason).startswith("missing:"):
        missing = pkg.out_of_scope_reason.split(":", 1)[1]
        ask = {"region": "분석할 지역명(자치구 또는 동)", "comparison_regions": "비교할 지역 2곳",
               "monthly_rent": "월세 금액", "budget_limit": "월세 예산 상한"}
        wants = " / ".join(ask.get(m, m) for m in missing.split(","))
        return (f"질문에 필요한 정보가 부족합니다. {wants}을(를) 알려주시면 "
                f"과거 거래 데이터 기준으로 분석해 드리겠습니다.")

    return render_template(
        message="해당 질문은 답변 범위를 벗어나 제공이 어렵습니다.",
        reason=reason_ko,
        scope_note="본 서비스는 서울시 과거 거래 데이터 기반 설명만 제공합니다.",
    )


def build_safe_template_answer(pkg: EvidencePackage) -> str:
    # 폴백: LLM 자유 문장 없이 검증된 수치 + 한계만 조립
    body = render_stats_table(pkg.stats) + render_recommendations(pkg.recommendations) \
           + render_transport_summary(pkg.transport)
    header = "⚠️ AI 응답 검증에 실패하여, 시스템이 산출한 검증된 수치만 표 형태로 제공합니다.\n\n"
    return finalize(header + body, pkg)


# ============================================================================
# 보조 함수 (의사코드의 '# have to rewrite' 스텁 전체 구현)
# ============================================================================
def lookup_stat(stats: list[RegionStat], region: str, label: str) -> Optional[float]:
    for s in stats:
        if s.region == region or region in s.region or s.region in region:
            return getattr(s, label, None)
    return None


def within_tolerance(value: float, truth: float, tol: float) -> bool:
    return abs(value - truth) <= tol


def flatten_values(pkg: EvidencePackage) -> set:
    vals: set = set()
    for s in pkg.stats:
        vals.update([s.avg, s.median, s.p25, s.p75, s.count, s.avg_deposit, s.median_deposit])
        if s.user_rent_percentile is not None:
            vals.update([s.user_rent_percentile, round(100 - s.user_rent_percentile)])
    if pkg.transport is not None:
        t = pkg.transport
        vals.update([t.subway_count, t.bus_count, t.nearest_subway_m, t.nearest_bus_m,
                     t.subway_radius_m, t.bus_radius_m])
    for r in pkg.recommendations:
        vals.update([r["median_rent"], r["count"]])
    return {float(v) for v in vals}


_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _numbers_in(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUM_RE.findall(str(text))}


def extract_numbers(draft: str) -> set[float]:
    """답변 텍스트의 모든 숫자 토큰 추출 (천 단위 콤마 정규화 포함)."""
    return _numbers_in(draft)


def echoed_from(user_query: str) -> set[float]:
    """사용자 질문에 인용된 수치 추출 (질문 속 숫자는 환각으로 보지 않음)."""
    return _numbers_in(user_query)


_LABEL_PATTERNS = [
    ("median", r"중앙값|중간값|중위값"),
    ("p25", r"25\s*분위|하위\s*25"),
    ("p75", r"75\s*분위|상위\s*25"),
    ("avg", r"평균"),
]


def extract_stat_claims(draft: str, known_regions: list[str]) -> list[StatClaim]:
    """답변에서 '(지역) ... 라벨 ... 숫자만' 패턴의 통계 주장을 추출."""
    claims = []
    sentences = re.split(r"(?<=[.!?다요음됨임])\s+|\n", draft)
    last_region = known_regions[0] if len(known_regions) == 1 else None
    for sent in sentences:
        # 문장 내 지역명 탐지 (가장 마지막 언급 지역으로 귀속)
        region = last_region
        for kr in known_regions:
            if kr in sent or any(tok in sent for tok in kr.split()):
                region = kr
        if region:
            last_region = region
        for label, pat in _LABEL_PATTERNS:
            for m in re.finditer(r"(?:" + pat + r")[^0-9%]{0,20}?([\d,]+(?:\.\d+)?)\s*만\s*원?", sent):
                claims.append(StatClaim(region=region or "?", label=label,
                                        value=float(m.group(1).replace(",", ""))))
    return claims


def extract_count(draft: str, unit: str) -> Optional[int]:
    """'N개 역' / '역 N개' / '지하철역이 3개' 패턴 추출. 단순 글자수 카운트 금지."""
    pats = [rf"(\d+)\s*개(?:의)?[^.。\n]{{0,8}}{unit}", rf"{unit}[^0-9\n]{{0,10}}?(\d+)\s*개"]
    for p in pats:
        m = re.search(p, draft)
        if m:
            return int(m.group(1))
    return None


def extract_distance(draft: str, target: str) -> Optional[float]:
    """'역까지 350m' / '최단거리 350m' 패턴 추출 (target 주변 거리)."""
    m = re.search(rf"{target}[^0-9\n]{{0,15}}?약?\s*([\d,]+(?:\.\d+)?)\s*(?:m|미터)", draft)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


_PREDICTION_PATTERNS = [
    r"오를\s*(것|전망|가능성)", r"상승할\s*(것|전망)", r"하락할\s*(것|전망)",
    r"예측(됩니다|할 수|하면)", r"투자\s*(가치|수익|를 추천)", r"매수.{0,6}추천",
    r"법적으로\s*(문제|유효)", r"사기\s*위험이?\s*(없|있)습니다",
]


def contains_prediction_or_judgement(draft: str) -> bool:
    return any(re.search(p, draft) for p in _PREDICTION_PATTERNS)


_REALTIME_PATTERNS = [r"계약\s*가능한\s*매물", r"현재\s*매물이\s*있", r"지금\s*나와\s*있는\s*매물"]


def mentions_realtime_listing(draft: str) -> bool:
    """실시간 매물 '단정' 탐지. '~매물 추천이 아닙니다' 같은 부정문(한계 고지)은 제외."""
    for p in _REALTIME_PATTERNS:
        for m in re.finditer(p, draft):
            tail = draft[m.end(): m.end() + 14]
            if not re.search(r"아닙|아니|않", tail):
                return True
    return False


# ---- 렌더링 ------------------------------------------------------------------
def required_limits_for(query_type: str) -> list[str]:
    common = []
    if query_type == "교통형":
        common.append("교통 지표는 반경 기반 보조 지표이며, 실제 통근시간·환승·배차간격은 반영하지 않습니다.")
        common.append("전월세 통계는 과거 거래 데이터 기준이며, 현재 매물 정보가 아닙니다.")
    elif query_type in ("조회형", "비교형", "추천형"):
        common.append("과거 거래 데이터 기준이며, 현재 매물 정보가 아닙니다.")
        if query_type == "추천형":
            common.append("지역 후보 제시일 뿐, 현재 계약 가능한 매물 추천이 아닙니다.")
    return common


def render_limit_block(limits: list[str]) -> str:
    if not limits:
        return ""
    return "📌 [한계 안내]\n" + "\n".join(f"- {x}" for x in limits) + "\n"


def render_source_block(sources: list[str]) -> str:
    if not sources:
        return ""
    return "📚 [출처]\n" + "\n".join(f"- {s}" for s in sources)


def render_template(message: str, reason: str, scope_note: str) -> str:
    return f"🚫 {message}\n- 제한 사유: {reason}\n- {scope_note}"


def render_stats_table(stats: list[RegionStat]) -> str:
    if not stats:
        return ""
    lines = ["[지역별 전월세 통계 (월세 기준, 만원)]"]
    for s in stats:
        lines.append(f"- {s.region} ({s.period}, {format_number(s.count)}건): "
                     f"평균 {format_number(s.avg)} / 중앙값 {format_number(s.median)} / "
                     f"p25 {format_number(s.p25)} / p75 {format_number(s.p75)}")
        if s.user_rent_percentile is not None:
            lines.append(f"  · 사용자 조건은 해당 지역 분포의 약 {s.user_rent_percentile}% 거래보다 높습니다.")
    return "\n".join(lines) + "\n"


def render_recommendations(recs: list[dict]) -> str:
    if not recs:
        return ""
    lines = ["[과거 거래 기준 지역 후보]"]
    for r in recs:
        lines.append(f"- {r['region']}: 월세 중앙값 {format_number(r['median_rent'])}만원 "
                     f"(거래 {format_number(r['count'])}건)")
    return "\n".join(lines) + "\n"


def render_transport_summary(transport: Optional[TransportInfo]) -> str:
    if transport is None:
        return ""
    t = transport
    return (f"[교통 접근성 ({t.region})]\n"
            f"- 반경 {t.subway_radius_m}m 내 지하철역 {t.subway_count}개 "
            f"(최단 {t.nearest_subway_name} {format_number(t.nearest_subway_m)}m)\n"
            f"- 반경 {t.bus_radius_m}m 내 버스정류장 {t.bus_count}개 "
            f"(최단 {t.nearest_bus_name} {format_number(t.nearest_bus_m)}m)\n")


def render_llm_input(pkg: EvidencePackage, user_query: str) -> str:
    """근거 패키지 → LLM 프롬프트 직렬화."""
    payload = {
        "query_type": pkg.query_type,
        "stats": [vars(s) for s in pkg.stats],
        "transport": vars(pkg.transport) if pkg.transport else None,
        "recommendations": pkg.recommendations,
        "notes": pkg.notes,
        "policy_docs": [{"doc_type": d.doc_type, "source": d.source, "내용요약": d.limit_text}
                        for d in pkg.policy_docs],
    }
    return (f"[사용자 질문]\n{user_query}\n\n[근거 패키지]\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=1)}")
