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
    logger = Logger("predict_script")
    logger.info("예측 시작")
    
    try:
        # LG전자 모델 로드
        lg_model = LGElectronicsModel()
        
        # 예측 시작일 설정 (오늘 날짜)
        start_date = datetime.now().strftime("%Y-%m-%d")
        
        # 예측 수행
        predictions = lg_model.predict(start_date)
        logger.info(f"예측 결과: {predictions}")
        
        # 예측 결과 출력
        print("\n[LG전자 주가 예측 결과]")
        print(f"{'날짜':<12} {'예측 가격':>10}")
        print("-" * 25)
        for _, row in predictions.iterrows():
            print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['predicted_price']:>10,.0f}")
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 