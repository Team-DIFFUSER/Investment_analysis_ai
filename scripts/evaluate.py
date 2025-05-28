import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.stocks.lg_electronics import LGElectronicsModel
from utils.logger import Logger
from utils.config import Config
import logging
from datetime import datetime, timedelta

def main():
    # 로거 설정
    logger = Logger("evaluate_script")
    logger.info("모델 평가 시작")
    
    try:
        # LG전자 모델 로드
        lg_model = LGElectronicsModel()
        
        # 평가 기간 설정 (최근 30일)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # 모델 평가
        metrics = lg_model.evaluate(start_date, end_date)
        
        # 평가 결과 출력
        print("\n[LG전자 모델 평가 결과]")
        print(f"평가 기간: {start_date} ~ {end_date}")
        print("\n성능 지표:")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"Direction Accuracy: {metrics['Direction_Accuracy']:.2f}%")
        print(f"Trend Accuracy: {metrics['Trend_Accuracy']:.2f}%")
        
        logger.info("모델 평가 완료")
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 