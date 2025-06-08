import os
import sys
import logging
from datetime import datetime
from models.stocks.base.prediction_manager import PredictionManager
from models.stocks.lg_electronics import LGElectronicsModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 예측 관리자 초기화
        manager = PredictionManager()
        
        # LG전자 모델 추가
        lg_model = LGElectronicsModel()
        manager.add_model(lg_model)
        
        # 일일 예측 실행
        logger.info("일일 예측 시작")
        manager.run_daily_predictions()
        
        # 예측 결과 출력
        for stock_name in manager.models.keys():
            history = manager.get_prediction_history(stock_name)
            error_stats = manager.get_error_statistics(stock_name)
            
            logger.info(f"\n{stock_name} 예측 결과:")
            logger.info(f"예측 이력: {history}")
            logger.info(f"오차 통계: {error_stats}")
            
    except Exception as e:
        logger.error(f"예측 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 