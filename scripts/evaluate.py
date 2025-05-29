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
from models.evaluation import ModelEvaluator
from utils.database import DatabaseManager
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger('evaluate')

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
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = model.data_processor.prepare_data(
            stock_data, sentiment_data, economic_data
        )
        
        # 모델 평가
        logger.info("모델 평가 중...")
        evaluator = ModelEvaluator(model, scaler)
        results = evaluator.evaluate_predictions(X_test, y_test)
        
        # 결과 시각화
        logger.info("결과 시각화 중...")
        evaluator.plot_predictions(
            results,
            save_path=results_dir / 'predictions.png'
        )
        evaluator.plot_error_distribution(
            results,
            save_path=results_dir / 'error_distribution.png'
        )
        evaluator.plot_direction_accuracy(
            results,
            save_path=results_dir / 'direction_accuracy.png'
        )
        
        # 평가 보고서 생성
        logger.info("평가 보고서 생성 중...")
        report = evaluator.generate_evaluation_report(
            results,
            save_path=results_dir / 'evaluation_report.txt'
        )
        
        logger.info("평가 완료!")
        logger.info("\n" + report)
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {str(e)}")
        raise
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    main() 