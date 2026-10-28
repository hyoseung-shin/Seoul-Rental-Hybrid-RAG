"""공용 유틸리티: 숫자 포맷팅, 좌표 거리 계산."""
from __future__ import annotations
import math


def format_number(num) -> str:
    """숫자를 천 단위 콤마 문자열로 변환 (예: 35000 -> '35,000')"""
    if isinstance(num, float) and num.is_integer():
        num = int(num)
    return f"{num:,}"


def haversine_m(lat1, lng1, lat2, lng2):
    """두 좌표 간 거리(m). 교통 접근성 계산용 (WGS84 직선거리)."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


