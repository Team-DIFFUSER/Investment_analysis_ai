import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models.stocks.lg_electronics import LGElectronicsModel
from database.database import DatabaseManager
from utils.config import Config
from utils.logger import setup_logger

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_sentiment_data(db_manager: DatabaseManager, stock_code: str, start_date: str, end_date: str, logger: logging.Logger) -> pd.DataFrame:
    """TimescaleDB에서 감성 분석 결과 조회"""
    try:
        query = """
        SELECT 
            pub_date as date,
            stock_code,
            finbert_positive,
            finbert_negative,
            finbert_neutral,
            finbert_sentiment
        FROM news_sentiment
        WHERE stock_code = %s
        AND pub_date BETWEEN %s AND %s
        ORDER BY pub_date;
        """
        
        results = db_manager.execute_query(query, (stock_code, start_date, end_date))
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)
            
    except Exception as e:
        logger.error(f"감성 데이터 조회 중 오류 발생: {str(e)}")
        raise

def load_training_data(db_manager: DatabaseManager, start_date: str, end_date: str, logger: logging.Logger) -> tuple:
    """학습 데이터 로드"""
    try:
        # 주가 데이터 로드
        stock_data = db_manager.get_stock_data(
            stock_code='066570',
            start_date=start_date,
            end_date=end_date
        )
        
        # 감성 분석 결과 로드
        sentiment_data = get_sentiment_data(
            db_manager=db_manager,
            stock_code='066570',
            start_date=start_date,
            end_date=end_date,
            logger=logger
        )
        
        # 경제 데이터 로드
        economic_data = db_manager.get_economic_data(
            start_date=start_date,
            end_date=end_date
        )
        
        return stock_data, sentiment_data, economic_data
        
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

def main():
    """메인 실행 함수"""
    try:
        # 로거 설정
        logger = setup_logger()
        logger.info("LG전자 주가 예측 모델 학습을 시작합니다.")
        
        # 데이터베이스 연결
        db_manager = DatabaseManager()
        logger.info("데이터베이스 연결 성공")
        
        # 학습 기간 설정 (가장 오래된 데이터부터 현재까지)
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 가장 오래된 데이터 날짜 조회
        query = """
        SELECT MIN(time) as start_date
        FROM stock_prices
        WHERE stock_code = '066570';
        """
        result = db_manager.execute_query(query)
        start_date = result[0][0].strftime('%Y-%m-%d')
        
        logger.info(f"학습 기간: {start_date} ~ {end_date}")
        
        # 모델 초기화
        model = LGElectronicsModel()
        
        # 학습 데이터 로드
        logger.info("학습 데이터 로드 중...")
        stock_data, sentiment_data, economic_data = load_training_data(db_manager, start_date, end_date, logger)
        
        # 데이터 전처리
        logger.info("데이터 전처리 중...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = model.prepare_training_data()
        
        # 모델 학습
        logger.info("모델 학습 시작...")
        model.train(X_train, y_train, X_val, y_val)
        
        # 모델 평가
        logger.info("모델 평가 중...")
        metrics = model.evaluate(X_test, y_test)
        
        # 결과 출력
        logger.info(f"학습 완료! 평가 지표: {metrics}")
        
    except Exception as e:
        logger.error(f"학습 중 오류가 발생했습니다: {str(e)}")
        raise
    finally:
        if 'db_manager' in locals():
            db_manager.close()
            logger.info("데이터베이스 연결 종료")

if __name__ == "__main__":
    main() 