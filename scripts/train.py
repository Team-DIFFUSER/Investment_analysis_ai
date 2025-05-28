import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from models.stocks.lg_electronics import LGElectronicsModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 모델 초기화
        lg_model = LGElectronicsModel()
        
        # 데이터 로드
        logger.info("데이터 로드 중...")
        stock_data = lg_model.load_stock_data()
        sentiment_data = lg_model.load_sentiment_data()
        economic_data = lg_model.load_economic_data()
        
        # 데이터 전처리
        logger.info("데이터 전처리 중...")
        X_train, y_train, X_val, y_val, scaler = lg_model.data_processor.prepare_data(
            stock_data, sentiment_data, economic_data
        )
        
        # 모델 학습
        logger.info("모델 학습 시작...")
        history = lg_model.train(X_train, y_train, X_val, y_val)
        
        # 최근 30일 데이터로 평가
        logger.info("최근 30일 데이터로 모델 평가 중...")
        recent_data = stock_data.tail(30)
        recent_sentiment = sentiment_data.tail(30)
        recent_economic = economic_data.tail(30)
        
        X_test, y_test, _, _, _ = lg_model.data_processor.prepare_data(
            recent_data, recent_sentiment, recent_economic
        )
        
        evaluation_results = lg_model.evaluate(X_test, y_test)
        
        # 평가 결과 출력
        logger.info("\n모델 평가 결과:")
        logger.info(f"Test Loss: {evaluation_results['test_loss']:.4f}")
        logger.info(f"MSE: {evaluation_results['mse']:.4f}")
        logger.info(f"MAE: {evaluation_results['mae']:.4f}")
        logger.info(f"Direction Accuracy: {evaluation_results['direction_accuracy']:.4f}")
        
        # 학습 곡선 저장
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('models/history/lg_electronics_training_history.csv')
        
        logger.info("학습 완료!")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 