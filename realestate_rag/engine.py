"""[모듈 3] 하이브리드 검색 및 계산 엔진 (담당: 김채연)"""
# ============================================================================
# [모듈 3] 하이브리드 검색 및 계산 엔진 (FN-004~FN-009)
#   원칙: 모든 숫자와 비교는 DB(Pandas)와 코드가 계산하고, LLM은 설명만 수행
#   출력: 근거 패키지 (EvidencePackage) — 모듈 4 문서의 STRUCT와 동일 구조
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .ingestion import REGION_COORDS
from .utils import haversine_m

# ----------------------------------------------------------------------------
# 근거 패키지 데이터 구조 (모듈 4 문서 §2의 dataclass 정의를 그대로 채택)
# ----------------------------------------------------------------------------
QueryType = str  # "조회형" | "비교형" | "추천형" | "교통형" | "정책형" | "제한형"

QTYPE_KO = {  # 모듈2 영문 유형 → 기능명세서/모듈4 한글 유형 매핑
    "rent_lookup": "조회형", "condition_compare": "비교형", "region_compare": "비교형",
    "region_recommendation": "추천형", "transport_access": "교통형",
    "rent_with_transport": "교통형", "policy_info": "정책형",
    "restricted": "제한형", "unknown_or_incomplete": "제한형",
}


@dataclass
class RegionStat:
    region: str
    avg: float
    median: float
    p25: float
    p75: float
    count: int
    period: str
    avg_deposit: float = 0.0
    median_deposit: float = 0.0
    user_rent_percentile: Optional[float] = None  # 사용자 월세가 분포에서 차지하는 백분위


@dataclass
class TransportInfo:
    region: str
    subway_count: int
    bus_count: int
    nearest_subway_m: float
    nearest_bus_m: float
    nearest_subway_name: str = ""
    nearest_bus_name: str = ""
    subway_radius_m: int = 500
    bus_radius_m: int = 300


@dataclass
class PolicyDoc:
    doc_type: str
    source: str
    limit_text: str


@dataclass
class QueryParams:
    period: list[str] = field(default_factory=list)          # 예: ["1년"]
    radius: list[str] = field(default_factory=list)          # 예: ["500", "300"]
    user_conditions: list[str] = field(default_factory=list)  # 예: ["60", "1000"]


@dataclass
class EvidencePackage:
    query_type: QueryType
    params: QueryParams
    stats: list[RegionStat] = field(default_factory=list)
    transport: Optional[TransportInfo] = None
    policy_docs: list[PolicyDoc] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)  # 추천형 결과
    out_of_scope: bool = False
    out_of_scope_reason: str = ""
    notes: list[str] = field(default_factory=list)  # 기간 확장 등 데이터 품질 메모


# ----------------------------------------------------------------------------
# 문서형 RAG: 정책/한계/통계 해석 문서 (명세서 policy_docs 테이블)
#   - 정형 수치는 임베딩하지 않음 (전처리 정책 '임베딩 대상 제한')
#   - TF-IDF 벡터 검색 (오프라인·무비용). Gemini 임베딩으로 교체 가능.
# ----------------------------------------------------------------------------
POLICY_DOCUMENTS = [
    {"doc_id": "data_source_policy", "doc_type": "출처",
     "source": "서울시 열린데이터광장 전월세 실거래 데이터",
     "content": "본 서비스의 전월세 통계는 서울시 열린데이터광장에 공개된 전월세 실거래 신고 데이터를 기반으로 계산됩니다. "
                "데이터 출처는 서울시 공공데이터이며, 자치구명과 법정동명 기준으로 집계합니다."},
    {"doc_id": "rent_data_limitations", "doc_type": "한계",
     "source": "서비스 데이터 한계 안내 문서",
     "content": "전월세 통계는 과거 거래 신고 데이터 기준이며, 현재 계약 가능한 매물 정보가 아닙니다. "
                "신고되지 않은 거래는 포함되지 않으며, 실제 시장 호가와 차이가 있을 수 있습니다. "
                "거래 건수가 적은 지역은 통계의 대표성이 낮을 수 있습니다."},
    {"doc_id": "statistics_explanation", "doc_type": "통계해석",
     "source": "통계 해석 기준 문서",
     "content": "평균은 극단값(고가 거래)에 민감하므로, 시세 수준 판단에는 중앙값을 함께 보는 것이 중요합니다. "
                "중앙값은 전체 거래를 금액순으로 정렬했을 때 한가운데 값입니다. "
                "25분위(p25)와 75분위(p75)는 분포의 하위·상위 경계를 보여주며, "
                "사용자 조건이 75분위보다 높으면 해당 지역에서 비싼 편으로 해석합니다."},
    {"doc_id": "transport_limitations", "doc_type": "한계",
     "source": "교통 접근성 해석 기준 문서",
     "content": "교통 접근성 지표는 지역 대표 좌표 기준 반경 내 지하철역 수(500m), 버스정류장 수(300m), "
                "최단 직선거리로 계산한 보조 지표입니다. 실제 통근시간, 환승, 배차간격, 노선은 반영하지 않습니다."},
    {"doc_id": "unsupported_question_policy", "doc_type": "제한정책",
     "source": "답변 제한 정책 문서",
     "content": "본 서비스는 미래 가격 예측, 투자 판단, 전세사기 위험 판정, 법률 판단, 실시간 매물 탐색, "
                "정확한 통근시간 계산, 민간 플랫폼 기준 답변을 제공하지 않습니다. "
                "서울시 과거 거래 데이터 기반 설명만 제공합니다."},
    {"doc_id": "service_scope", "doc_type": "범위",
     "source": "서비스 사용 범위 문서",
     "content": "본 서비스는 서울특별시 전월세 데이터만 지원합니다. 서울 외 지역 질문은 답변 범위를 벗어납니다. "
                "지역 추천은 과거 거래 데이터 기준 후보 제시이며 현재 매물 추천이 아닙니다."},
]


class PolicyRagEngine:
    """문서형 정책/한계 문서에 대한 TF-IDF 기반 벡터 검색 (FN-009 보조)."""

    def __init__(self, documents: list[dict] = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = documents or POLICY_DOCUMENTS
        # 한국어: 형태소 분석기 없이 문자 n-gram TF-IDF 사용 (조사 변형에 강건)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        self.doc_matrix = self.vectorizer.fit_transform([d["content"] for d in self.docs])

    def search(self, query: str, rag_targets: list[str], top_k: int = 3) -> list[PolicyDoc]:
        from sklearn.metrics.pairwise import cosine_similarity
        # 1) 라우팅 모듈이 지정한 rag_targets 문서는 항상 포함 (정책 일관성)
        selected, seen = [], set()
        for d in self.docs:
            if d["doc_id"] in rag_targets:
                selected.append(d)
                seen.add(d["doc_id"])
        # 2) 잔여 슬롯은 질의-문서 유사도 상위 문서로 채움 (의미 검색)
        if len(selected) < top_k and query:
            sims = cosine_similarity(self.vectorizer.transform([query]), self.doc_matrix)[0]
            for idx in np.argsort(-sims):
                d = self.docs[idx]
                if d["doc_id"] not in seen and sims[idx] > 0.05:
                    selected.append(d)
                    seen.add(d["doc_id"])
                if len(selected) >= max(top_k, len(rag_targets)):
                    break
        # 3) 질의-문서 유사도 순으로 정렬 (가장 관련 높은 문서가 먼저 오도록)
        if query and selected:
            qv = self.vectorizer.transform([query])
            order = {d["doc_id"]: float(cosine_similarity(
                qv, self.vectorizer.transform([d["content"]]))[0][0]) for d in selected}
            selected.sort(key=lambda d: -order[d["doc_id"]])
        return [PolicyDoc(doc_type=d["doc_type"], source=d["source"], limit_text=d["content"])
                for d in selected]


# ----------------------------------------------------------------------------
# 메인 엔진
# ----------------------------------------------------------------------------
MIN_TRANSACTIONS = 10  # 기간 확장 기준 (모듈 2 문서 §7: 건수 부족 시 3년으로 확장)


class HybridRetrievalAnalyticsEngine:
    def __init__(self, rent_df: pd.DataFrame,
                 subway_df: Optional[pd.DataFrame] = None,
                 bus_df: Optional[pd.DataFrame] = None,
                 rag_engine: Optional[PolicyRagEngine] = None):
        self.rent_df = rent_df
        self.subway_df = subway_df
        self.bus_df = bus_df
        self.rag = rag_engine or PolicyRagEngine()

    # ==== 진입점: 실행 계획 JSON → 근거 패키지 =================================
    def run(self, plan: dict) -> EvidencePackage:
        qtype_en = plan["question_type"]
        ents = plan["entities"]
        qtype_ko = QTYPE_KO.get(qtype_en, "제한형")

        params = QueryParams(
            period=["1년"] if ents.get("period") == "recent_1_year" else ["3년"],
            radius=["500", "300"],
            user_conditions=[str(v) for v in [ents.get("deposit"), ents.get("monthly_rent"),
                                              ents.get("budget_limit")] if v is not None],
        )

        # 제한/추가질문 → 계산 없이 패키지 구성
        if plan["answer_policy"] in ("denied", "need_clarification"):
            return EvidencePackage(
                query_type="제한형" if plan["answer_policy"] == "denied" else qtype_ko,
                params=params,
                policy_docs=self.rag.search(plan["original_query"], plan["rag_targets"]),
                out_of_scope=(plan["answer_policy"] == "denied"),
                out_of_scope_reason=plan.get("restriction_reason") or
                                    ("missing:" + ",".join(plan.get("missing_fields", []))),
            )

        pkg = EvidencePackage(query_type=qtype_ko, params=params)
        notes = []

        # ---- 전월세 통계 (FN-004, FN-005, FN-006) ------------------------------
        if any("rent" in m or "region" in m for m in plan["required_modules"]):
            if qtype_en == "region_recommendation":
                budget = ents.get("budget_limit") or ents.get("monthly_rent")
                pkg.recommendations = self.recommend_regions(budget, ents.get("property_type"))
            else:
                regions = ents.get("regions") or []
                if qtype_en == "region_compare":
                    regions = list(dict.fromkeys(regions + ents.get("comparison_targets", [])))
                for region in regions:
                    stat, note = self.calc_rent_statistics(
                        region, ents.get("period", "recent_1_year"),
                        ents.get("property_type"), ents.get("rent_type"),
                        user_rent=ents.get("monthly_rent"))
                    if stat:
                        pkg.stats.append(stat)
                    if note:
                        notes.append(note)

        # ---- 교통 접근성 (FN-008, 선택 기능) -----------------------------------
        if ents.get("traffic_required") and ents.get("regions"):
            t = self.calc_transport_access(ents["regions"][0])
            if t:
                pkg.transport = t
            else:
                notes.append(f"'{ents['regions'][0]}' 지역의 대표 좌표가 없어 교통 접근성을 계산하지 못했습니다.")

        # ---- 문서형 RAG (FN-009) ----------------------------------------------
        pkg.policy_docs = self.rag.search(plan["original_query"], plan["rag_targets"])
        pkg.notes = notes
        return pkg

    # ==== 전월세 통계 계산 =====================================================
    def _filter(self, region: str, period: str, property_type, rent_type) -> pd.DataFrame:
        df = self.rent_df
        mask = pd.Series(True, index=df.index)
        # 지역: "성북구 안암동" 처럼 복합 입력 지원 — 토큰별 자치구/법정동 매칭
        tokens = region.split()
        for tok in tokens:
            if tok.endswith("구"):
                mask &= (df["자치구명"] == tok)
            else:
                mask &= (df["법정동명"] == tok)
        if property_type:
            mask &= (df["건물용도"] == property_type)
        # 시세 비교는 월세 거래 기준이 기본 (rent_type 미지정 시 월세)
        mask &= (df["전월세구분"] == (rent_type or "월세"))
        sub = df[mask]
        # 기간 필터 (계약년월 존재 시)
        if "계약년월" in df.columns and len(sub):
            months = 12 if period == "recent_1_year" else 36
            ym = pd.to_numeric(sub["계약년월"], errors="coerce").fillna(0)
            cutoff = self._cutoff_ym(months)
            sub = sub[ym >= cutoff]
        return sub

    @staticmethod
    def _cutoff_ym(months: int) -> int:
        today = pd.Timestamp.today()
        cut = today - pd.DateOffset(months=months)
        return cut.year * 100 + cut.month

    def calc_rent_statistics(self, region, period, property_type, rent_type, user_rent=None):
        """평균/중앙값/분위수/거래건수 + 사용자 월세 백분위. 건수 부족 시 3년으로 확장."""
        note = None
        sub = self._filter(region, period, property_type, rent_type)
        if len(sub) < MIN_TRANSACTIONS and period == "recent_1_year":
            sub3 = self._filter(region, "recent_3_years", property_type, rent_type)
            if len(sub3) > len(sub):
                sub, period = sub3, "recent_3_years"
                note = f"'{region}' 최근 1년 거래 건수가 적어 최근 3년 데이터로 확장했습니다."
        if sub.empty:
            return None, f"'{region}' 조건의 실거래 데이터가 없습니다."

        rent_col, dep_col = sub["월세(만원)"], sub["보증금(만원)"]
        stat = RegionStat(
            region=region,
            avg=round(float(rent_col.mean())),
            median=round(float(rent_col.median())),
            p25=round(float(rent_col.quantile(0.25))),
            p75=round(float(rent_col.quantile(0.75))),
            count=int(len(sub)),
            period="최근 1년" if period == "recent_1_year" else "최근 3년",
            avg_deposit=round(float(dep_col.mean())),
            median_deposit=round(float(dep_col.median())),
        )
        if user_rent is not None:
            stat.user_rent_percentile = round(float((rent_col < user_rent).mean() * 100))
        return stat, note

    # ==== 지역 후보 추천 (FN-007) — 현재 매물 아님, 과거 거래 기준 ==============
    def recommend_regions(self, budget: Optional[int], property_type=None, top_n: int = 5) -> list[dict]:
        df = self.rent_df[self.rent_df["전월세구분"] == "월세"]
        if property_type:
            df = df[df["건물용도"] == property_type]
        g = (df.groupby(["자치구명", "법정동명"])["월세(만원)"]
               .agg(median="median", count="count").reset_index())
        g = g[g["count"] >= MIN_TRANSACTIONS]
        if budget:
            g = g[g["median"] <= budget]
        g = g.sort_values(["median", "count"], ascending=[True, False]).head(top_n)
        return [{"region": f"{r.자치구명} {r.법정동명}", "median_rent": round(float(r.median)),
                 "count": int(r.count)} for r in g.itertuples(index=False)]

    # ==== 교통 접근성 계산 (FN-008) ===========================================
    def calc_transport_access(self, region: str) -> Optional[TransportInfo]:
        # 대표 좌표 매핑 ("성북구 안암동" → 안암동 우선)
        coord = None
        for tok in reversed(region.split()):
            if tok in REGION_COORDS:
                coord = REGION_COORDS[tok]
                break
        if coord is None or self.subway_df is None or self.bus_df is None:
            return None
        lat, lng = coord

        def nearest_and_count(points: pd.DataFrame, radius_m: int):
            d = points.apply(lambda r: haversine_m(lat, lng, r["lat"], r["lng"]), axis=1)
            idx = d.idxmin()
            return int((d <= radius_m).sum()), round(float(d.min())), str(points.loc[idx, "name"])

        s_cnt, s_min, s_name = nearest_and_count(self.subway_df, 500)
        b_cnt, b_min, b_name = nearest_and_count(self.bus_df, 300)
        return TransportInfo(region=region, subway_count=s_cnt, bus_count=b_cnt,
                             nearest_subway_m=s_min, nearest_bus_m=b_min,
                             nearest_subway_name=s_name, nearest_bus_name=b_name)
