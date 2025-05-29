import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models.stocks.lg_electronics import LGElectronicsModel
from database.database import DatabaseManager

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

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

def get_previous_predictions(stock_name: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """이전 예측값 조회"""
    db = DatabaseManager()
    try:
        query = """
        SELECT target_date, predicted_price
        FROM predicted_stock_prices
        WHERE stock_name = %s
        AND target_date BETWEEN %s AND %s
        ORDER BY target_date
        """
        params = (stock_name, start_date, end_date)
        results = db.execute_query(query, params)
        return pd.DataFrame(results, columns=['target_date', 'predicted_price'])
    except Exception as e:
        logger.error(f"이전 예측값 조회 중 오류 발생: {str(e)}")
        return pd.DataFrame()
    finally:
        db.close()

def get_latest_stock_price(stock_name):
    """최근 주가 조회"""
    db = DatabaseManager()
    try:
        query = """
        SELECT close_price
        FROM stock_prices
        WHERE stock_name = %s
        ORDER BY time DESC
        LIMIT 1
        """
        params = (stock_name,)
        result = db.execute_query(query, params)
        if result:
            return float(result[0]['close_price'])
        return None
    except Exception as e:
        logger.error(f"최근 주가 조회 중 오류 발생: {str(e)}")
        return None
    finally:
        db.close()

def calculate_prediction_adjustment(actual_price: float, predicted_price: float, next_predicted_price: float) -> float:
    """예측값 조정 계산"""
    if predicted_price is None:
        return 0
    
    # 이전 예측의 오차 계산
    error = actual_price - predicted_price
    
    # 오차의 일부를 다음 예측에 반영 (점진적 조정)
    adjustment = error * 0.3
    
    # 다음 예측값이 너무 크게 변하지 않도록 제한
    max_adjustment = actual_price * 0.05  # 최대 5% 조정
    adjustment = max(min(adjustment, max_adjustment), -max_adjustment)
    
    return adjustment

def save_prediction(stock_code: str, stock_name: str, prediction_date: datetime, target_date: datetime, predicted_price: float):
    """예측 결과를 데이터베이스에 저장"""
    db = DatabaseManager()
    try:
        query = """
        INSERT INTO predicted_stock_prices (
            stock_code, stock_name, prediction_date, target_date,
            predicted_price
        ) VALUES (%s, %s, %s, %s, %s)
        """
        params = (
            stock_code,
            stock_name,
            prediction_date,
            target_date,
            float(predicted_price)
        )
        db.execute_query(query, params)
        logger.info(f"예측 결과가 데이터베이스에 저장되었습니다: {target_date}")
    except Exception as e:
        logger.error(f"데이터베이스 저장 중 오류 발생: {str(e)}")
        raise
    finally:
        db.close()

def main():
    logger = setup_logging()
    logger.info("LG전자 주가 예측을 시작합니다.")
    
    try:
        # 시작일 설정 (2025년 3월 27일)
        start_date = datetime(2025, 3, 27)
        
        # LG전자 모델 초기화
        model = LGElectronicsModel()
        
        # 예측 수행
        predictions = model.predict_next_five_days(start_date)
        
        # 결과 출력
        logger.info("\n=== LG전자 주가 예측 결과 ===")
        logger.info(f"예측 시작일: {start_date.strftime('%Y-%m-%d')}\n")
        logger.info("예측 결과:")
        for pred in predictions:
            logger.info(f"{pred['date'].strftime('%Y-%m-%d')}: {pred['price']:,.0f}원")
        
        # 예측 결과 저장
        for pred in predictions:
            save_prediction(
                stock_code='066570',  # LG전자 종목코드
                stock_name='LG전자',
                prediction_date=datetime.now(),
                target_date=pred['date'],
                predicted_price=pred['price']
            )
        
        logger.info("\n✅ 예측 결과가 데이터베이스에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 