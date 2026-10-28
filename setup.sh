#!/usr/bin/env bash
# ============================================================================
# 환경 설정: 가상환경 생성 + 의존성 설치
# 사용법: bash setup.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}

echo "[1/3] 가상환경 생성 (.venv)"
$PYTHON -m venv .venv
source .venv/bin/activate

echo "[2/3] 의존성 설치"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/3] 데이터 디렉토리 준비"
mkdir -p data results
cat <<'MSG'

✅ 설치 완료.

다음 단계:
  1) (선택) 서울시 열린데이터광장 CSV를 data/ 에 배치
       data/seoul_rent_data.csv          전월세 실거래
       data/seoul_subway_stations.csv    역사마스터 (선택)
       data/seoul_bus_stops.csv          버스정류소 위치 (선택)
     → 없으면 데모용 합성 데이터로 자동 동작합니다.

  2) (선택) Gemini API 키 등록 — 코드에 하드코딩 금지!
       export GEMINI_API_KEY="발급받은_키"
     → 없으면 규칙 기반 Mock으로 전체 파이프라인이 무비용 동작합니다.

  3) 실행
       bash run_demo.sh        # 대표 시나리오 데모
       bash run_eval.sh        # 평가 (리포트 Result 산출)
MSG
