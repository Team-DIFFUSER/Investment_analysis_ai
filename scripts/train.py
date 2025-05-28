import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.stocks.lg_electronics import LGElectronicsModel
from utils.logger import Logger
from utils.config import Config
import logging

def main():
    # 로거 설정
    logger = Logger("train_script")
    logger.info("학습 시작")
    
    try:
        # LG전자 모델 학습
        lg_model = LGElectronicsModel()
        lg_model.train()
        logger.info("LG전자 모델 학습 완료")
        
        # 학습된 모델로 예측 수행
        start_date = "2025-03-24"
        end_date = "2025-03-28"
        
        # 모델 평가
        metrics = lg_model.evaluate(start_date, end_date)
        logger.info(f"모델 평가 결과: {metrics}")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 