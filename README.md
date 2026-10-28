# 🏠 서울시 전월세 시세 분석 하이브리드 RAG 서비스

**기계학습개론 (26-1) 텀프로젝트 — 팀 8 (오유덕, 공지혁, 신효승, 김채연)**

LLM이 전월세 가격·통계·거리를 직접 추측하면 **숫자 환각**이 발생한다. 본 서비스는
*"계산은 코드가, 해석은 LLM이"* 원칙으로 서울시 공공 전월세 데이터, 교통 데이터,
문서형 RAG를 결합해 **근거 기반 답변**을 생성하고, 생성된 답변의 모든 수치를 **사후 검증**한다.

## 아키텍처 (모듈 1 → 2 → 3 → 4)

```
사용자 질문
   ↓
[모듈 2] 쿼리 파싱·라우팅 (router.py)        LLM = 구조화 파서
   질문 유형 9종 분류 · 조건 추출 · 답변 정책 4종 · 실행 계획 JSON
   ↓
[모듈 3] 하이브리드 검색·계산 (engine.py)     숫자는 전부 코드가 계산
   전월세 통계(평균/중앙값/p25/p75/백분위) + 교통 반경 계산(Haversine)
   + 정책 문서 TF-IDF RAG  →  근거 패키지(EvidencePackage)
   ↓
[모듈 4] 답변 생성·검증 (generation.py)       LLM = 설명자
   제한 게이트 → LLM 초안 → 4종 검증(통계/교통/미근거 숫자/범위 이탈)
   → 1회 자가수정 → 실패 시 검증된 수치만 기계 조립(폴백)
   → 한계·출처 기계적 강제 부착
```

[모듈 1] (`ingestion.py`)은 앱 구동 시 1회 실행되어 CSV를 Pandas In-Memory로
적재·정제한다 (DB 적재 생략 설계 결정).

| 파일 | 모듈 | 담당 | 기능 ID |
|---|---|---|---|
| `realestate_rag/ingestion.py` | 1. 데이터 수집·정제 | 오유덕 | (기반 계층) |
| `realestate_rag/router.py` | 2. 쿼리 파싱·라우팅 | 공지혁 | FN-002, FN-003 |
| `realestate_rag/engine.py` | 3. 하이브리드 검색·계산 | 김채연 | FN-004~FN-009 |
| `realestate_rag/generation.py` | 4. 답변 생성·검증 | 신효승 | FN-010~FN-012 |
| `realestate_rag/llm_clients.py` | Gemini / Mock 클라이언트 | 공통 | — |
| `realestate_rag/pipeline.py` | 전체 오케스트레이션 | 공통 | — |
| `realestate_rag/evaluation.py` | 평가 데이터셋·지표 | 공통 | — |

## 빠른 시작

```bash
bash setup.sh                       # 가상환경 + 의존성 설치
export GEMINI_API_KEY="..."         # (선택) 없으면 Mock 모드로 무비용 동작
bash run_demo.sh                    # 대표 시나리오 7문항 데모
bash run_demo.sh -q "안암동 월세 수준 알려줘"   # 단일 질문
bash run_demo.sh -i                 # 대화형 모드
bash run_eval.sh                    # 평가 → results/ 에 리포트용 산출물 저장
bash run_eval.sh --judge            # LLM-as-a-Judge 포함 (Gemini 필요)
```

> ⚠️ **보안**: API 키는 반드시 환경변수로만 전달한다. 코드·노트북에 키를 하드코딩하지
> 말 것. 과거에 키가 평문으로 커밋/공유된 적이 있다면 즉시 폐기(revoke) 후 재발급한다.

## 데이터

`data/` 에 서울시 열린데이터광장 CSV를 배치한다. **파일이 없으면 합성 데모 데이터로
자동 대체**되어 전체 파이프라인을 즉시 검증할 수 있다.

| 파일 | 내용 | 필수 |
|---|---|---|
| `data/seoul_rent_data.csv` | 전월세 실거래 (자치구명/법정동명/건물용도/전월세구분/보증금(만원)/월세(만원)) | 권장 |
| `data/seoul_subway_stations.csv` | 역사마스터 (역명/위도/경도) | 선택 |
| `data/seoul_bus_stops.csv` | 버스정류소 위치 (정류소명/위도/경도) | 선택 |

지역 대표 좌표는 `ingestion.py`의 `REGION_COORDS` 사전에 정의한다 (데모용 10개 지역
포함 — 평가에 사용할 지역을 추가할 것).

## 평가 (텀프로젝트 §4~6)

- **평가 데이터셋**: 24문항 (질문, 정답 유형, 정답 정책) — `evaluation.py`의 `EVAL_DATASET`
- **베이스라인**: 동일 LLM에 질문만 그대로 전달 (RAG·검증 없음)
- **정량 지표**
  1. 라우팅 정확도 — 질문 유형/답변 정책 분류 Accuracy
  2. 수치 근거율(groundedness) — 답변 내 모든 수치가 시스템 산출값·질문 인용값에
     근거하는 비율 (검증기를 평가 지표로 재사용)
  3. 안전 거절률 — 제한 질문(denied)에 수치·예측 없이 거절한 비율
  4. (선택) LLM-as-a-Judge 비교 평가
- **정성 평가**: 입력 — 베이스라인 출력 — 제안 방법론 출력 비교 (`results/qualitative_samples.md`)

Mock 모드 측정값(합성 데이터 기준): 라우팅 유형/정책 정확도 100%/100%,
근거율 0% → 100%, 안전 거절률 0% → 100%. 실제 Gemini + 실데이터 수치를 리포트
Result 섹션에 사용한다.

## 리포트(ACL 2p) 매핑 가이드

| 섹션 | 가져갈 내용 |
|---|---|
| Introduction | LLM 숫자 환각 문제 + 주거 도메인 필요성 |
| Proposed Method | 4모듈 파이프라인 그림 + 검증기 4종 설명 |
| Experiment | 평가 데이터셋 24문항, 모델(gemini-2.5-flash), 지표 정의 |
| Result | `results/metrics_summary.json` 표 + `qualitative_samples.md` 사례 |
| Conclusion | 계산-해석 분리 + 사후 검증의 효과, 한계(형태소 분석 미사용, 좌표 수동 매핑) |

**Trial & error 기록 (발표용)**
1. 검증기가 한계 고지 문구("매물 추천이 아닙니다")를 실시간 매물 단정으로 오탐 → 부정문 인식 추가
2. 추천형 답변의 후보 중앙값이 stats 외 산출값이라 UNGROUNDED 오탐 → grounded 집합 확장
3. 필수값 부족 질문이 RAG 문서로 응답됨 → 모듈 4 게이트에 clarification 분기 추가
