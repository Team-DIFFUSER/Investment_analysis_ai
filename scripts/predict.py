import os
import sys
import logging
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ALL_STOCK_MODELS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_and_predict(model, stock_name):
    """모델 초기화 및 예측 수행"""
    try:
        logger.info(f"{stock_name} 모델 초기화 시작")
        
        # 모델 초기화
        if hasattr(model, 'initialize'):
            model.initialize()
        else:
            # 모델이 초기화되지 않은 경우 학습 수행
            logger.info(f"{stock_name} 모델 학습 시작")
            model.train()
        
        # 예측 수행
        logger.info(f"{stock_name} 예측 시작")
        predictions = model.predict_next_five_days()
        
        if predictions:
            logger.info(f"{stock_name} 예측 완료: {predictions}")
            return True
        else:
            logger.error(f"{stock_name} 예측 실패")
            return False
            
    except Exception as e:
        logger.error(f"{stock_name} 처리 중 오류 발생: {str(e)}")
        return False

def main():
    try:
        # 모델 인스턴스 생성
        models = {name: model_class() for name, model_class in ALL_STOCK_MODELS.items()}
        
        # 각 모델별로 초기화 및 예측 수행
        success_count = 0
        total_count = len(models)
        
        for stock_name, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"{stock_name} 처리 시작")
            logger.info(f"{'='*50}")
            
            if initialize_and_predict(model, stock_name):
                success_count += 1
            else:
                logger.error(f"{stock_name} 처리 실패")
        
        # 결과 요약
        logger.info(f"\n{'='*50}")
        logger.info(f"예측 완료: {success_count}/{total_count} 성공")
        logger.info(f"{'='*50}")
        
        if success_count == 0:
            logger.error("모든 모델 예측이 실패했습니다.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"예측 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 