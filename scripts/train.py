import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.stocks.lg_electronics import LGElectronicsModel
from database.database import DatabaseManager
from utils.config import Config
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger('train')

def load_training_data(db_manager: DatabaseManager, start_date: str, end_date: str) -> tuple:
    """학습 데이터 로드"""
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
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

def main():
    try:
        # 결과 디렉토리 생성
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # 설정 로드
        config = Config()
        
        # 데이터베이스 연결
        db_manager = DatabaseManager()
        
        # 학습 기간 설정 (3년)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        
        # 데이터 로드
        logger.info("학습 데이터 로드 중...")
        stock_data, sentiment_data, economic_data = load_training_data(db_manager, start_date, end_date)
        
        # 모델 초기화
        model = LGElectronicsModel(config)
        
        # 데이터 전처리
        logger.info("데이터 전처리 중...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = model.data_processor.prepare_data(
            stock_data, sentiment_data, economic_data
        )
        
        # 모델 학습
        logger.info("모델 학습 시작...")
        training_results = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=32,
            epochs=100,
            early_stopping_patience=10
        )
        
        # 학습 결과 시각화
        plt.figure(figsize=(12, 4))
        
        # 손실 그래프
        plt.subplot(1, 2, 1)
        plt.plot(training_results['history']['loss'], label='Training Loss')
        plt.plot(training_results['history']['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 학습률 그래프
        plt.subplot(1, 2, 2)
        plt.plot(training_results['history']['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / 'training_curves.png')
        plt.close()
        
        logger.info(f"최적의 에포크: {training_results['best_epoch']}")
        logger.info(f"최적의 검증 손실: {training_results['best_val_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        raise
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    main() 