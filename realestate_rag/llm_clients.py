"""LLM 클라이언트: GeminiClient(운영) / RuleBasedMockClient(무비용 테스트)."""
from __future__ import annotations
import json
import re

from .router import ExecutionPlanLLM, ParsedEntities
from .utils import format_number

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def parse_query(self, system_prompt: str, user_query: str, schema) -> ExecutionPlanLLM:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=system_prompt + f"\n\n사용자 질문: {user_query}",
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,  # 의도 추출은 결정론적으로
            ),
        )
        return schema.model_validate_json(resp.text)

    def generate(self, system_prompt: str, llm_input: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=system_prompt + "\n\n" + llm_input,
            config=self._types.GenerateContentConfig(temperature=0.1),
        )
        return resp.text

    def generate_free(self, prompt: str) -> str:
        """베이스라인용: 시스템 제약 없는 일반 LLM 응답."""
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=self._types.GenerateContentConfig(temperature=0.1),
        )
        return resp.text


class RuleBasedMockClient:
    """API 키 없이 전체 파이프라인을 검증하기 위한 규칙 기반 대체 클라이언트."""

    # ---- 모듈 2 대체: 규칙 기반 의도 분류 -------------------------------------
    def parse_query(self, system_prompt: str, q: str, schema) -> ExecutionPlanLLM:
        ents = ParsedEntities()
        # 지역 추출
        regions = re.findall(r"([가-힣]+구)?\s*([가-힣]+동)", q)
        flat = []
        for gu, dong in regions:
            flat.append((gu + " " + dong).strip())
        gus_only = [g for g in re.findall(r"([가-힣]+구)(?!\s*[가-힣]+동)", q)]
        for g in gus_only:
            if not any(g in r for r in flat):
                flat.append(g)
        ents.regions = flat
        # 숫자 조건
        m = re.search(r"보증금\s*([\d,]+)", q)
        if m: ents.deposit = int(m.group(1).replace(",", ""))
        m = re.search(r"월세\s*([\d,]+)", q)
        if m: ents.monthly_rent = int(m.group(1).replace(",", ""))
        m = re.search(r"([\d,]+)\s*만?원?\s*이하", q)
        if m: ents.budget_limit = int(m.group(1).replace(",", ""))
        if "3년" in q: ents.period = "recent_3_years"
        for k, v in [("아파트", "아파트"), ("빌라", "연립다세대"), ("다세대", "연립다세대"),
                     ("오피스텔", "오피스텔"), ("원룸", "단독다가구")]:
            if k in q: ents.property_type = v
        if "전세" in q and "전세사기" not in q: ents.rent_type = "전세"
        traffic = any(k in q for k in ["교통", "지하철", "정류장", "버스", "역 많", "역이나"])
        ents.traffic_required = traffic

        # 유형/제한 판정
        reason = None
        if re.search(r"내년|오를까|떨어질까|전망", q): qt, reason = "restricted", "future_price_prediction_not_supported"
        elif "전세사기" in q and ("위험" in q or "없어" in q): qt, reason = "restricted", "fraud_judgement_not_supported"
        elif "투자" in q: qt, reason = "restricted", "investment_judgement_not_supported"
        elif re.search(r"계약서|법적", q): qt, reason = "restricted", "legal_judgement_not_supported"
        elif re.search(r"몇\s*분|소요시간", q): qt, reason = "restricted", "commute_time_not_supported"
        elif re.search(r"네이버부동산|직방|다방", q): qt, reason = "restricted", "private_platform_not_supported"
        elif re.search(r"지금\s*계약\s*가능|제일\s*싼\s*방", q): qt, reason = "restricted", "realtime_listing_not_supported"
        elif re.search(r"중앙값|평균이 뭐|출처|한계", q): qt = "policy_info"
        elif "추천" in q: qt = "region_recommendation"
        elif ("중 어디" in q or "어디가" in q) and len(ents.regions) >= 1: qt = "region_compare"
        elif traffic and ents.monthly_rent is not None: qt = "rent_with_transport"
        elif traffic: qt = "transport_access"
        elif ents.monthly_rent is not None and re.search(r"비싸|비싼|괜찮", q): qt = "condition_compare"
        elif re.search(r"시세|수준|얼마|올랐는지", q): qt = "rent_lookup"
        else: qt = "unknown_or_incomplete"
        if qt == "region_compare":
            ents.comparison_targets = list(ents.regions)
        return ExecutionPlanLLM(question_type=qt, restriction_reason=reason, entities=ents)

    # ---- 모듈 4 대체: 근거 패키지를 그대로 서술하는 템플릿 생성 -----------------
    def generate(self, system_prompt: str, llm_input: str) -> str:
        body = llm_input.split("[근거 패키지]\n", 1)[1].split("\n[수정요구]", 1)[0]
        pkg_json = json.loads(body)
        parts = []
        for s in pkg_json["stats"]:
            line = (f"{s['region']}의 {s['period']} 월세 거래 {format_number(s['count'])}건 기준으로 "
                    f"평균 {format_number(s['avg'])}만원, 중앙값 {format_number(s['median'])}만원입니다.")
            if s.get("user_rent_percentile") is not None:
                p = s["user_rent_percentile"]
                level = "비싼 편" if p >= 75 else ("보통 수준" if p >= 25 else "낮은 편")
                line += (f" 말씀하신 조건은 분포의 약 {p}% 거래보다 높아 {level}이며, "
                         f"75분위는 {format_number(s['p75'])}만원입니다.")
            parts.append(line)
        for r in pkg_json.get("recommendations", []):
            parts.append(f"{r['region']}은 월세 중앙값 {format_number(r['median_rent'])}만원"
                         f"(거래 {format_number(r['count'])}건)으로 조건에 부합하는 후보입니다.")
        t = pkg_json.get("transport")
        if t:
            parts.append(f"{t['region']} 기준 반경 500m 내 지하철역 {t['subway_count']}개, "
                         f"반경 300m 내 버스정류장 {t['bus_count']}개가 있으며, "
                         f"가장 가까운 역까지 {format_number(t['nearest_subway_m'])}m, "
                         f"가장 가까운 정류장까지 {format_number(t['nearest_bus_m'])}m입니다.")
        if not parts:
            docs = pkg_json.get("policy_docs", [])
            parts.append(docs[0]["내용요약"] if docs else "관련 정보를 찾지 못했습니다.")
        return " ".join(parts)

    def generate_free(self, prompt: str) -> str:
        # 베이스라인 흉내: 근거 없는 그럴듯한 수치를 생성 (환각 시뮬레이션)
        return ("해당 지역 월세는 평균 약 65만원, 중앙값 58만원 수준으로 알려져 있으며 "
                "내년에는 5% 정도 오를 것으로 예상됩니다. 역까지는 보통 400m 정도입니다.")


