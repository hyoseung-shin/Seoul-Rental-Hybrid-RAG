#!/usr/bin/env bash
# 대표 시나리오 데모 실행 (질문정책 6개 유형 전부 커버)
# 사용법: bash run_demo.sh            — 데모 7문항
#         bash run_demo.sh -i         — 대화형 모드
#         bash run_demo.sh -q "질문"  — 단일 질문
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

if [ $# -eq 0 ]; then
    python main.py --demo --data-dir ./data
else
    python main.py "$@" --data-dir ./data
fi
