import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.stocks.lg_electronics import LGElectronicsModel
from utils.logger import Logger
from utils.config import Config
from database.database import DatabaseManager
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 전역 logger 설정
logger = Logger("predict_script")

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

def is_holiday(date):
    """주말과 공휴일 체크"""
    # 주말 체크 (5: 토요일, 6: 일요일)
    if date.weekday() >= 5:
        return True
    
    # 공휴일 체크
    if date in KOREAN_HOLIDAYS_2025:
        return True
    
    return False

def get_next_business_day(date):
    """다음 영업일 계산 (주말과 공휴일 제외)"""
    next_day = date + timedelta(days=1)
    while is_holiday(next_day):
        next_day += timedelta(days=1)
    return next_day

def get_next_five_business_days(start_date):
    """다음 5개 영업일 계산"""
    business_days = []
    current_date = start_date
    
    while len(business_days) < 5:
        current_date = get_next_business_day(current_date)
        business_days.append(current_date)
    
    return business_days

def get_previous_predictions(stock_name, start_date, end_date):
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

def calculate_prediction_adjustment(actual_price, predicted_price, next_predicted_price):
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

def save_prediction(stock_code, stock_name, prediction_date, target_date, predicted_price):
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
    logger.info("예측 시작")
    
    try:
        # LG전자 모델 로드
        lg_model = LGElectronicsModel()
        
        # 예측 시작일 설정
        start_date = datetime(2025, 3, 27)
        
        # 최근 주가 조회
        last_actual_price = get_latest_stock_price('LG전자')
        if last_actual_price is None:
            raise ValueError("최근 주가를 조회할 수 없습니다.")
        
        logger.info(f"예측 기준 주가: {last_actual_price:,.0f}원")
        
        # 다음 5개 영업일 계산
        business_days = get_next_five_business_days(start_date)
        
        # 이전 예측값 조회
        end_date = business_days[-1]
        previous_predictions = get_previous_predictions('LG전자', start_date, end_date)
        
        # 예측 수행
        predictions = lg_model.predict_next_five_days(start_date)
        
        # 예측값 조정
        adjusted_predictions = []
        for i, pred in enumerate(predictions):
            target_date = pred['date']
            if i == 0:
                # 첫날은 실제값 사용
                predicted_price = last_actual_price
            else:
                # 이전 예측값이 있는 경우 조정
                if not previous_predictions.empty and i < len(previous_predictions):
                    prev_pred = previous_predictions.iloc[i-1]['predicted_price']
                    next_pred = pred['price']
                    adjustment = calculate_prediction_adjustment(
                        last_actual_price, prev_pred, next_pred
                    )
                    predicted_price = next_pred + adjustment
                else:
                    predicted_price = pred['price']
            
            # 100원 단위로 반올림
            predicted_price = round(predicted_price / 100) * 100
            
            adjusted_predictions.append({
                'date': target_date,
                'price': predicted_price
            })
        
        # 예측 결과를 데이터베이스에 저장
        for pred in adjusted_predictions:
            save_prediction(
                stock_code='066570',
                stock_name='LG전자',
                prediction_date=start_date,
                target_date=pred['date'],
                predicted_price=pred['price']
            )
        
        # 예측 결과 출력
        print("\n[LG전자 주가 예측 결과]")
        print(f"예측 기준일: {start_date.strftime('%Y-%m-%d')}")
        print(f"기준 주가: {last_actual_price:,.0f}원")
        print(f"{'날짜':<12} {'요일':<8} {'예측 가격':>10}")
        print("-" * 35)
        
        for pred in adjusted_predictions:
            date = pred['date']
            price = pred['price']
            weekday = date.strftime('%a')  # 요일 약자 (Mon, Tue, etc.)
            print(f"{date.strftime('%Y-%m-%d'):<12} {weekday:<8} {price:>10,.0f}")
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 