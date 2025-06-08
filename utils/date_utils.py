from datetime import datetime, timedelta
from typing import List

# 2025년 한국 공휴일 목록
KOREAN_HOLIDAYS_2025 = [
    datetime(2025, 1, 1),   # 신정
    datetime(2025, 2, 9),   # 설날
    datetime(2025, 2, 10),  # 설날
    datetime(2025, 2, 11),  # 설날
    datetime(2025, 3, 1),   # 삼일절
    datetime(2025, 5, 5),   # 어린이날
    datetime(2025, 6, 3),   # 선거
    datetime(2025, 6, 6),   # 현충일
    datetime(2025, 8, 15),  # 광복절
    datetime(2025, 9, 28),  # 추석
    datetime(2025, 9, 29),  # 추석
    datetime(2025, 9, 30),  # 추석
    datetime(2025, 10, 3),  # 개천절
    datetime(2025, 10, 9),  # 한글날
    datetime(2025, 12, 25), # 크리스마스
]

def is_holiday(date: datetime) -> bool:
    """주말과 공휴일 체크"""
    return date.weekday() >= 5 or date in KOREAN_HOLIDAYS_2025

def get_next_business_day(date: datetime) -> datetime:
    """다음 영업일 계산"""
    next_day = date + timedelta(days=1)
    while is_holiday(next_day):
        next_day += timedelta(days=1)
    return next_day

def get_next_five_business_days(start_date: datetime) -> List[datetime]:
    """다음 5개 영업일 계산"""
    business_days = []
    current_date = start_date
    
    while len(business_days) < 5:
        current_date = get_next_business_day(current_date)
        business_days.append(current_date)
    
    return business_days 