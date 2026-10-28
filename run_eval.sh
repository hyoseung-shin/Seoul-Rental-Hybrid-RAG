#!/usr/bin/env bash
# 평가 실행: 라우팅 정확도 + 근거율/안전 거절률 (베이스라인 vs 제안 방법론)
# 사용법: bash run_eval.sh             — 기본 평가
#         bash run_eval.sh --judge     — LLM-as-a-Judge 포함 (GEMINI_API_KEY 필요)
#         bash run_eval.sh --mock      — Mock LLM 강제
# 결과: results/ (routing_eval.csv, metrics_summary.json, qualitative_samples.md)
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

python evaluate.py --data-dir ./data --out-dir ./results "$@"
