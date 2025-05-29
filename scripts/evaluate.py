import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
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

def load_test_data(db_manager, start_date, end_date):
    """테스트 데이터 로드"""
    try:
        # 주가 데이터 로드
        stock_data = db_manager.get_stock_data(
            stock_code='066570',
            start_date=start_date,
            end_date=end_date
        )
        
        # 감성 데이터 로드
        sentiment_data = db_manager.get_sentiment_data(
            stock_code='066570',
            start_date=start_date,
            end_date=end_date
        )
        
        # 경제 데이터 로드
        economic_data = db_manager.get_economic_data(
            start_date=start_date,
            end_date=end_date
        )
        
        return stock_data, sentiment_data, economic_data
        
    except Exception as e:
        logger.error(f"테스트 데이터 로드 중 오류 발생: {str(e)}")
        raise

def main():
    logger = setup_logging()
    logger.info("모델 평가를 시작합니다.")
    
    try:
        # 결과 디렉토리 생성
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # 데이터베이스 연결
        db_manager = DatabaseManager()
        
        # 테스트 기간 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1년치 데이터
        
        # 테스트 데이터 로드
        logger.info("테스트 데이터 로드 중...")
        stock_data, sentiment_data, economic_data = load_test_data(
            db_manager, start_date, end_date
        )
        
        # 모델 초기화
        logger.info("모델 초기화 중...")
        model = LGElectronicsModel()
        
        # 데이터 전처리
        logger.info("데이터 전처리 중...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = model.prepare_training_data()
        
        # 모델 평가
        logger.info("모델 평가 중...")
        metrics = model.evaluate(X_test, y_test)
        
        # 결과 출력
        logger.info("\n=== 평가 결과 ===")
        logger.info(f"MSE: {metrics['mse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"방향성 정확도: {metrics['direction_accuracy']:.2%}")
        
        # 결과 저장
        results_file = results_dir / 'evaluation_results.txt'
        with open(results_file, 'w') as f:
            f.write("=== LG전자 주가 예측 모델 평가 결과 ===\n\n")
            f.write(f"평가 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}\n\n")
            f.write(f"MSE: {metrics['mse']:.4f}\n")
            f.write(f"MAE: {metrics['mae']:.4f}\n")
            f.write(f"방향성 정확도: {metrics['direction_accuracy']:.2%}\n")
        
        logger.info(f"평가 결과가 {results_file}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {str(e)}")
        raise
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    main() 