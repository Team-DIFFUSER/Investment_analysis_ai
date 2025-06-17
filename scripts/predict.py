import os
import sys
import logging
from datetime import datetime
from models.stocks.base.prediction_manager import PredictionManager
from models.stocks.lg_electronics import LGElectronicsModel
from models.stocks.samsung_electronics import SamsungElectronicsModel

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
        samsung_model = SamsungElectronicsModel()
        
        prediction_manager = PredictionManager()
        prediction_manager.add_model('LG전자', lg_model)
        prediction_manager.add_model('삼성전자', samsung_model)
        
        # 일일 예측 실행
        logger.info("일일 예측 시작")
        prediction_manager.run_daily_prediction()
        
        # 예측 결과 출력
        for stock_name in prediction_manager.models.keys():
            logger.info(f"\n{stock_name} 예측 결과:")
            logger.info(f"예측 이력: {prediction_manager.predictions_data['stock_predictions'].get(stock_name, {})}")
            logger.info(f"오차 통계: {prediction_manager.predictions_data['error_metrics'].get(stock_name, {})}")
            
    except Exception as e:
        logger.error(f"예측 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 