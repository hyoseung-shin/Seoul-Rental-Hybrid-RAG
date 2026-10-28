#!/usr/bin/env python3
"""평가 스크립트 — 텀프로젝트 '실험/평가 및 개선' 단계 (리포트 Result 섹션 산출용).

실행:
  python evaluate.py                  # GEMINI_API_KEY 있으면 Gemini, 없으면 Mock
  python evaluate.py --mock           # Mock 강제
  python evaluate.py --judge          # (선택) LLM-as-a-Judge 비교 평가 포함 (Gemini 필요)

산출물 (results/):
  routing_eval.csv        질문별 유형/정책 분류 결과
  metrics_summary.json    정량 지표 요약 (베이스라인 vs 제안 방법론)
  qualitative_samples.md  정성 비교 (입력 — 베이스라인 — 제안 방법론)
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from realestate_rag import (EVAL_DATASET, GeminiClient, RealEstateRagService,
                            RuleBasedMockClient, evaluate_groundedness,
                            evaluate_routing, llm_as_judge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="Mock LLM 강제 사용")
    ap.add_argument("--judge", action="store_true", help="LLM-as-a-Judge 평가 포함 (Gemini 필요)")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./results")
    args = ap.parse_args()

    api_key = None if args.mock else os.environ.get("GEMINI_API_KEY")
    llm = GeminiClient(api_key) if api_key else RuleBasedMockClient()
    mode = "Gemini(gemini-2.5-flash)" if api_key else "Mock(규칙 기반)"
    print(f"평가 모드: {mode} | 평가 데이터셋: {len(EVAL_DATASET)}문항")

    service = RealEstateRagService(llm, data_dir=args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 정량 평가 ① 라우팅 정확도 (모듈 2) ----------------------------------
    print("\n[1/3] 라우팅 정확도 평가 중...")
    routing = evaluate_routing(service)
    df = pd.DataFrame(routing["rows"])
    df.to_csv(os.path.join(args.out_dir, "routing_eval.csv"), index=False, encoding="utf-8-sig")
    print(f"  질문 유형 분류 정확도: {routing['type_accuracy']:.2%}")
    print(f"  답변 정책 분류 정확도: {routing['policy_accuracy']:.2%}")
    wrong = df[~(df["유형정답"] & df["정책정답"])]
    if len(wrong):
        print("  오분류 사례:")
        for _, r in wrong.iterrows():
            print(f"    ✗ {r['질문']} → {r['예측유형']}/{r['예측정책']} (정답 {r['정답유형']}/{r['정답정책']})")

    # ---- 정량 평가 ②③ 근거율 / 안전 거절률 (베이스라인 vs 제안) ---------------
    print("\n[2/3] 근거율·안전 거절률 평가 중 (베이스라인 vs 제안 방법론)...")
    g = evaluate_groundedness(service)
    print(f"  수치 근거율   : 베이스라인 {g['baseline_grounded_rate']:.2%} → "
          f"제안 {g['proposed_grounded_rate']:.2%}")
    print(f"  안전 거절률   : 베이스라인 {g['baseline_safe_refusal_rate']:.2%} → "
          f"제안 {g['proposed_safe_refusal_rate']:.2%}")

    metrics = {
        "llm_mode": mode,
        "n_eval": len(EVAL_DATASET),
        "routing_type_accuracy": routing["type_accuracy"],
        "routing_policy_accuracy": routing["policy_accuracy"],
        "baseline_grounded_rate": g["baseline_grounded_rate"],
        "proposed_grounded_rate": g["proposed_grounded_rate"],
        "baseline_safe_refusal_rate": g["baseline_safe_refusal_rate"],
        "proposed_safe_refusal_rate": g["proposed_safe_refusal_rate"],
    }

    # ---- 정성 평가 샘플 저장 --------------------------------------------------
    md = ["# 정성 평가: 입력 — 베이스라인 — 제안 방법론 비교\n"]
    for s in g["qualitative_samples"]:
        md += [f"## {s['질문']}\n", "**[베이스라인 (일반 LLM)]**\n", s["베이스라인"] + "\n",
               "**[제안 방법론 (하이브리드 RAG + 검증)]**\n", s["제안방법"] + "\n"]
    with open(os.path.join(args.out_dir, "qualitative_samples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---- (선택) LLM-as-a-Judge ------------------------------------------------
    if args.judge and api_key:
        print("\n[3/3] LLM-as-a-Judge 평가 중...")
        wins = {"A": 0, "B": 0, "tie": 0}
        for q, _, policy in EVAL_DATASET[:10]:
            base = llm.generate_free(f"서울 부동산 질문에 답해줘: {q}")
            prop = service.answer(q, verbose=False)["answer"]
            verdict = llm_as_judge(llm, q, base, prop)  # A=베이스라인, B=제안
            wins[verdict.get("winner", "tie")] = wins.get(verdict.get("winner", "tie"), 0) + 1
        metrics["judge_wins_baseline"] = wins["A"]
        metrics["judge_wins_proposed"] = wins["B"]
        metrics["judge_ties"] = wins["tie"]
        print(f"  Judge 결과 — 베이스라인 {wins['A']}승 / 제안 {wins['B']}승 / 무승부 {wins['tie']}")
    else:
        print("\n[3/3] LLM-as-a-Judge 생략 (--judge + GEMINI_API_KEY 필요)")

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 결과 저장 완료 → {args.out_dir}/ (routing_eval.csv, metrics_summary.json, qualitative_samples.md)")


if __name__ == "__main__":
    main()
