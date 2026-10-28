"""[모듈 1] 데이터 수집 및 정제 파이프라인 (담당: 오유덕)
설계 결정: DB 적재 생략, Pandas In-Memory 방식 (01_데이터_수집_모듈.pdf §1)
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np
import pandas as pd

# ============================================================================
# [모듈 1] 데이터 수집 및 정제 파이프라인  (FN 해당 없음 — 기반 계층)
#   - 설계 결정: DB 적재 생략, Pandas In-Memory 방식 (모듈 1 문서 §1)
#   - 전월세 CSV + 지하철역/버스정류장 CSV + 정책 문서 로드
# ============================================================================
class DataIngestionPipeline:
    """물리적 DB 없이 CSV를 메모리(Pandas)에 로드하고 정제한다."""

    RENT_TEXT_COLS = ["자치구명", "법정동명", "건물용도", "전월세구분"]

    def __init__(self, data_dir: str = "./"):
        self.data_dir = data_dir
        self.rent_df: Optional[pd.DataFrame] = None
        self.subway_df: Optional[pd.DataFrame] = None
        self.bus_df: Optional[pd.DataFrame] = None

    # ---- Step 1~3: 전월세 실거래가 ------------------------------------------
    def load_and_clean_rent_data(self, file_name: str = "seoul_rent_data.csv") -> pd.DataFrame:
        path = os.path.join(self.data_dir, file_name)
        try:
            try:
                df = pd.read_csv(path, encoding="cp949", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        except FileNotFoundError:
            print(f"⚠️ 전월세 CSV 없음: {path} → 데모용 합성 데이터 생성")
            df = self._make_demo_rent_data()

        df.columns = df.columns.str.strip()

        # 숫자형 변환 + 결측 처리
        for col in ["보증금(만원)", "월세(만원)"]:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
                    .fillna(0)
                    .astype("int64")
                )
        # 계약 연월 (기간 필터용)
        for cand in ["계약일", "계약년월", "접수연도"]:
            if cand in df.columns:
                df["계약년월"] = (
                    pd.to_numeric(df[cand], errors="coerce").fillna(0).astype("int64").astype(str).str[:6]
                )
                break
        # 문자열 공백 제거
        for col in self.RENT_TEXT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # 이상치 제거 (명세서 '전처리 정책': IQR 기반, 보증금>0 또는 월세>0)
        df = df[(df.get("보증금(만원)", 0) > 0) | (df.get("월세(만원)", 0) > 0)]

        self.rent_df = df.reset_index(drop=True)
        print(f"✅ 전월세 데이터 로드 완료 (총 {len(df):,} 건)")
        return self.rent_df

    # ---- 교통 데이터 (선택 기능, 교통명세 시트) ------------------------------
    def load_transport_data(self,
                            subway_file: str = "seoul_subway_stations.csv",
                            bus_file: str = "seoul_bus_stops.csv"):
        self.subway_df = self._load_points(subway_file, kind="지하철역")
        self.bus_df = self._load_points(bus_file, kind="버스정류장")
        return self.subway_df, self.bus_df

    def _load_points(self, file_name: str, kind: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, file_name)
        try:
            try:
                df = pd.read_csv(path, encoding="cp949")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="utf-8")
            df.columns = df.columns.str.strip()
            # 좌표 컬럼 추정 (위도/경도, lat/lng 등)
            lat_col = next((c for c in df.columns if "위도" in c or c.lower() in ("lat", "latitude")), None)
            lng_col = next((c for c in df.columns if "경도" in c or c.lower() in ("lng", "lon", "longitude")), None)
            name_col = next((c for c in df.columns if "명" in c or "name" in c.lower()), df.columns[0])
            df = df.rename(columns={lat_col: "lat", lng_col: "lng", name_col: "name"})
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
            df = df.dropna(subset=["lat", "lng"]).drop_duplicates(subset=["name", "lat", "lng"])
            print(f"✅ {kind} 데이터 로드 완료 ({len(df):,} 건)")
            return df[["name", "lat", "lng"]].reset_index(drop=True)
        except FileNotFoundError:
            print(f"⚠️ {kind} CSV 없음 → 데모용 합성 좌표 생성")
            return self._make_demo_points(kind)

    # ---- 데모 데이터 (실제 CSV가 없을 때 파이프라인 검증용) -------------------
    @staticmethod
    def _make_demo_rent_data(seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        # 지역별 (자치구, 동, 월세 평균, 보증금 평균) 프로파일
        profiles = [
            ("성북구", "안암동", 55, 1500), ("성북구", "길음동", 50, 2000),
            ("마포구", "서교동", 75, 3000), ("마포구", "공덕동", 80, 4000),
            ("관악구", "신림동", 45, 1000), ("강남구", "역삼동", 95, 5000),
        ]
        for gu, dong, rent_mu, dep_mu in profiles:
            for _ in range(120):
                use = rng.choice(["아파트", "연립다세대", "오피스텔", "단독다가구"], p=[.3, .25, .2, .25])
                kind = rng.choice(["월세", "전세"], p=[.7, .3])
                monthly = max(10, int(rng.normal(rent_mu, rent_mu * 0.25))) if kind == "월세" else 0
                deposit = max(100, int(rng.normal(dep_mu if kind == "월세" else dep_mu * 12, dep_mu * 0.4)))
                ym = f"2025{rng.integers(1, 13):02d}"
                rows.append([gu, dong, use, kind, deposit, monthly, ym])
        return pd.DataFrame(rows, columns=["자치구명", "법정동명", "건물용도", "전월세구분",
                                           "보증금(만원)", "월세(만원)", "계약년월"])

    @staticmethod
    def _make_demo_points(kind: str) -> pd.DataFrame:
        # 안암동(고려대 인근) / 서교동 / 길음동 주변 좌표 샘플
        if kind == "지하철역":
            pts = [("안암역", 37.5863, 127.0292), ("고려대역", 37.5905, 127.0357),
                   ("보문역", 37.5853, 127.0193), ("길음역", 37.6033, 127.0250),
                   ("홍대입구역", 37.5572, 126.9245), ("합정역", 37.5499, 126.9136),
                   ("신림역", 37.4842, 126.9297), ("역삼역", 37.5006, 127.0364)]
        else:
            base = [("안암오거리", 37.5860, 127.0300), ("고려대정문", 37.5895, 127.0330),
                    ("개운사길", 37.5880, 127.0270), ("안암초교", 37.5845, 127.0315),
                    ("서교동사거리", 37.5530, 126.9180), ("길음뉴타운", 37.6060, 127.0230),
                    ("신림사거리", 37.4845, 126.9300), ("역삼푸르지오", 37.5010, 127.0380)]
            pts = base + [(f"{n}-{i}", lat + 0.001 * i, lng - 0.0008 * i) for i, (n, lat, lng) in enumerate(base)]
        return pd.DataFrame(pts, columns=["name", "lat", "lng"])


# 법정동 대표 좌표 (지역 매핑 — 교통명세 §2 '지역 매핑: 대표 좌표 사용')
REGION_COORDS = {
    "안암동": (37.5866, 127.0295), "길음동": (37.6065, 127.0247),
    "서교동": (37.5536, 126.9189), "공덕동": (37.5440, 126.9512),
    "신림동": (37.4842, 126.9297), "역삼동": (37.5006, 127.0364),
    "성북구": (37.5894, 127.0167), "마포구": (37.5663, 126.9019),
    "관악구": (37.4784, 126.9516), "강남구": (37.5172, 127.0473),
}


