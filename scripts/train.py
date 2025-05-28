import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.stocks.lg_electronics import LGElectronicsModel
from utils.logger import Logger
from utils.config import Config
import logging
from datetime import datetime, timedelta

# 전역 logger 설정
logger = Logger("train_script")

def main():
    logger.info("학습 시작")
    
    try:
        # LG전자 모델 로드
        lg_model = LGElectronicsModel()
        
        # 데이터 준비
        X_train, y_train, X_val, y_val = lg_model.prepare_data()
        
        # 모델 학습
        lg_model.train(X_train, y_train, X_val, y_val)
        
        # 모델 평가
        evaluation_period = 30  # 30일 동안의 예측 평가
        end_date = datetime.now()
        start_date = end_date - timedelta(days=evaluation_period)
        
        evaluation_results = lg_model.evaluate(start_date, end_date)
        logger.info(f"평가 결과: {evaluation_results}")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 