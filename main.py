#!/usr/bin/env python3
"""서울시 전월세 시세 분석 하이브리드 RAG 서비스 — CLI 진입점.

사용 예:
  python main.py --demo                          # 대표 시나리오 7종 일괄 실행
  python main.py --query "안암동 월세 수준 알려줘"   # 단일 질문
  python main.py --interactive                   # 대화형 모드
  python main.py --mock --demo                   # API 키 없이 규칙 기반 Mock으로 실행

LLM 선택:
  환경변수 GEMINI_API_KEY 가 있으면 Gemini(gemini-2.5-flash), 없거나 --mock 이면 Mock.
  ⚠️ API 키는 절대 코드에 하드코딩하지 말 것 (export GEMINI_API_KEY=... 사용).
"""
from __future__ import annotations

import argparse
import os
import sys

from realestate_rag import GeminiClient, RealEstateRagService, RuleBasedMockClient

# 기능명세서 '질문정책' 시트의 6개 유형을 모두 커버하는 데모 질문
DEMO_QUESTIONS = [
    "안암동에서 보증금 1000만원에 월세 60만원이면 비싼 편이야?",  # 비교형 (FN-005)
    "성북구랑 마포구 중 어디가 월세가 더 비싸?",                  # 비교형 (FN-006)
    "월세 50만원 이하 지역 추천해줘",                             # 추천형 (FN-007)
    "안암동에서 월세 60이면 괜찮고 교통도 좋은 편이야?",          # 교통형 (FN-008)
    "중앙값이 왜 중요해?",                                        # 정책형 (문서 RAG)
    "내년에 안암동 월세 오를까?",                                 # 제한형 (FN-011)
    "월세 60이면 비싼 편이야?",                                   # 필수값 부족 → 추가 질문
]


def build_service(args) -> RealEstateRagService:
    api_key = None if args.mock else os.environ.get("GEMINI_API_KEY")
    if api_key:
        print("🔑 Gemini API 모드 (gemini-2.5-flash)")
        llm = GeminiClient(api_key)
    else:
        print("🧪 Mock 모드 (규칙 기반 — API 비용 없음). 실제 LLM은 export GEMINI_API_KEY=... 후 실행")
        llm = RuleBasedMockClient()
    return RealEstateRagService(
        llm, data_dir=args.data_dir,
        rent_file=args.rent_file, subway_file=args.subway_file, bus_file=args.bus_file)


def main():
    ap = argparse.ArgumentParser(description="서울시 전월세 시세 분석 하이브리드 RAG 서비스")
    ap.add_argument("--query", "-q", help="단일 질문 실행")
    ap.add_argument("--demo", action="store_true", help="대표 시나리오 일괄 실행")
    ap.add_argument("--interactive", "-i", action="store_true", help="대화형 모드")
    ap.add_argument("--mock", action="store_true", help="API 키가 있어도 Mock LLM 강제 사용")
    ap.add_argument("--data-dir", default="./data", help="CSV 데이터 디렉토리 (기본 ./data)")
    ap.add_argument("--rent-file", default="seoul_rent_data.csv")
    ap.add_argument("--subway-file", default="seoul_subway_stations.csv")
    ap.add_argument("--bus-file", default="seoul_bus_stops.csv")
    args = ap.parse_args()

    if not (args.query or args.demo or args.interactive):
        ap.print_help()
        sys.exit(0)

    service = build_service(args)

    if args.demo:
        for q in DEMO_QUESTIONS:
            service.answer(q)
    if args.query:
        service.answer(args.query)
    if args.interactive:
        print("\n💬 대화형 모드 (종료: exit / quit / 빈 입력)")
        while True:
            try:
                q = input("\n질문 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            service.answer(q)
        print("종료합니다.")


if __name__ == "__main__":
    main()
