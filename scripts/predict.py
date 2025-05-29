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

def get_next_business_day(date):
    """다음 영업일 계산 (주말 제외)"""
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5: 토요일, 6: 일요일
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
        db.execute(query, params)
        db.commit()
        logger.info(f"예측 결과가 데이터베이스에 저장되었습니다: {target_date}")
    except Exception as e:
        db.rollback()
        logger.error(f"데이터베이스 저장 중 오류 발생: {str(e)}")
        raise
    finally:
        db.close()

def main():
    logger.info("예측 시작")
    
    try:
        # LG전자 모델 로드
        lg_model = LGElectronicsModel()
        
        # 예측 시작일 설정 (오늘 날짜)
        start_date = datetime(2025, 3, 27)
        
        # 다음 5개 영업일 계산
        business_days = get_next_five_business_days(start_date)
        
        # 예측 수행
        predictions = lg_model.predict_next_five_days(start_date)
        logger.info(f"예측 결과: {predictions}")
        
        # 예측 결과를 데이터베이스에 저장
        for pred in predictions:
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
        print(f"{'날짜':<12} {'요일':<8} {'예측 가격':>10}")
        print("-" * 35)
        
        for pred in predictions:
            date = pred['date']
            price = pred['price']
            weekday = date.strftime('%a')  # 요일 약자 (Mon, Tue, etc.)
            print(f"{date.strftime('%Y-%m-%d'):<12} {weekday:<8} {price:>10,.0f}")
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 