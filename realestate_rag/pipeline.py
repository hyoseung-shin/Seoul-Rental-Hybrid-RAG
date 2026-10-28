"""전체 서비스 파이프라인: 모듈 1 → 2 → 3 → 4 오케스트레이션."""
from __future__ import annotations
import time

from .ingestion import DataIngestionPipeline
from .router import QueryParserRouter
from .engine import HybridRetrievalAnalyticsEngine, PolicyRagEngine
from .generation import GenerationVerificationModule

# 전체 서비스 파이프라인 (모듈 1 → 2 → 3 → 4)
# ----------------------------------------------------------------------------
class RealEstateRagService:
    def __init__(self, llm_client, data_dir: str = "./",
                 rent_file: str = "seoul_rent_data.csv",
                 subway_file: str = "seoul_subway_stations.csv",
                 bus_file: str = "seoul_bus_stops.csv"):
        # [모듈 1]
        self.ingest = DataIngestionPipeline(data_dir)
        rent_df = self.ingest.load_and_clean_rent_data(rent_file)
        subway_df, bus_df = self.ingest.load_transport_data(subway_file, bus_file)
        # [모듈 2]
        self.router = QueryParserRouter(llm_client)
        # [모듈 3]
        self.engine = HybridRetrievalAnalyticsEngine(rent_df, subway_df, bus_df, PolicyRagEngine())
        # [모듈 4]
        self.generator = GenerationVerificationModule(llm_client)
        self.llm = llm_client

    def answer(self, user_query: str, verbose: bool = True) -> dict:
        if verbose: print(f"\n{'='*70}\n💬 질문: {user_query}")
        t0 = time.time()
        # [모듈 2] 실행 계획
        plan = self.router.parse_and_route(user_query)
        if verbose:
            print(f"[1/3] 🔍 라우팅 → 유형: {plan['question_type']} | 정책: {plan['answer_policy']}"
                  f" | 지역: {plan['entities']['regions']}")
        # [모듈 3] 근거 패키지
        pkg = self.engine.run(plan)
        if verbose:
            print(f"[2/3] 📊 근거 패키지 → 통계 {len(pkg.stats)}건, 교통 {'O' if pkg.transport else 'X'},"
                  f" 문서 {len(pkg.policy_docs)}건, 추천 {len(pkg.recommendations)}건")
        # [모듈 4] 생성 + 검증
        final = self.generator.generate_and_verify(pkg, user_query)
        dbg = self.generator.last_debug
        if verbose:
            print(f"[3/3] 🛡️ 검증 → 시도 {dbg.get('attempts', 0)}회, 폴백 {dbg.get('fallback')}")
            print(f"\n🤖 최종 답변 ({time.time()-t0:.1f}s)\n{'-'*70}\n{final}")
        return {"plan": plan, "package": pkg, "answer": final, "debug": dbg}


