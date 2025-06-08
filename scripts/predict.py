import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stocks.lg_electronics import LGElectronicsModel
from utils.date_utils import get_next_five_business_days
from database.database import Database

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        """주식 예측기 초기화"""
        self.db = Database()
        self.models = {}
        self.max_workers = 4  # 동시에 처리할 최대 종목 수
        
    def initialize_models(self) -> None:
        """모든 모델 초기화"""
        try:
            # 여기에 새로운 종목 모델들을 추가
            self.models = {
                'LG전자': LGElectronicsModel(),
                # '삼성전자': SamsungElectronicsModel(),
                # 'SK하이닉스': SKHynixModel(),
                # ... 다른 종목들 추가
            }
            logger.info(f"초기화된 모델 수: {len(self.models)}")
        except Exception as e:
            logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
            raise

    def predict_single_stock(self, stock_name: str, model: Any, start_date: datetime) -> Dict[str, Any]:
        """단일 종목 예측 수행"""
        try:
            logger.info(f"{stock_name} 예측 시작")
            predictions = model.predict_next_five_days(start_date)
            
            # 예측 결과 저장
            self.save_predictions(stock_name, predictions, start_date)
            
            return {
                'stock_name': stock_name,
                'status': 'success',
                'predictions': predictions
            }
        except Exception as e:
            logger.error(f"{stock_name} 예측 중 오류 발생: {str(e)}")
            return {
                'stock_name': stock_name,
                'status': 'error',
                'error': str(e)
            }

    def save_predictions(self, stock_name: str, predictions: List[Dict[str, Any]], start_date: datetime) -> None:
        """예측 결과를 데이터베이스에 저장"""
        try:
            for pred in predictions:
                self.db.insert_prediction(
                    stock_name=stock_name,
                    date=pred['date'],
                    predicted_price=pred['predicted_price'],
                    confidence=pred['confidence']
                )
            logger.info(f"{stock_name} 예측 결과 저장 완료")
        except Exception as e:
            logger.error(f"{stock_name} 예측 결과 저장 중 오류 발생: {str(e)}")
            raise

    def predict_all_stocks(self, start_date: datetime = None) -> Dict[str, Any]:
        """모든 종목에 대한 예측 수행"""
        if start_date is None:
            start_date = datetime.now()

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 종목에 대한 예측 작업 제출
            future_to_stock = {
                executor.submit(self.predict_single_stock, stock_name, model, start_date): stock_name
                for stock_name, model in self.models.items()
            }
            
            # 진행 상황 표시
            for future in tqdm(as_completed(future_to_stock), total=len(future_to_stock), desc="종목 예측 진행률"):
                stock_name = future_to_stock[future]
                try:
                    result = future.result()
                    results[stock_name] = result
                except Exception as e:
                    logger.error(f"{stock_name} 처리 중 오류 발생: {str(e)}")
                    results[stock_name] = {
                        'status': 'error',
                        'error': str(e)
                    }

        return results

    def generate_report(self, results: Dict[str, Any]) -> None:
        """예측 결과 리포트 생성"""
        try:
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            error_count = len(results) - success_count
            
            report = f"""
            예측 결과 리포트
            ==============
            총 종목 수: {len(results)}
            성공: {success_count}
            실패: {error_count}
            
            실패한 종목:
            {chr(10).join(f"- {stock}: {result['error']}" for stock, result in results.items() if result['status'] == 'error')}
            """
            
            logger.info(report)
            
            # 리포트를 파일로 저장
            report_path = os.path.join('data', 'reports', f'prediction_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
            logger.info(f"리포트가 저장되었습니다: {report_path}")
            
        except Exception as e:
            logger.error(f"리포트 생성 중 오류 발생: {str(e)}")
            raise

def main():
    """메인 실행 함수"""
    try:
        logger.info("주가 예측을 시작합니다.")
        
        # 데이터베이스 연결
        db = Database()
        
        # 예측기 초기화
        predictor = StockPredictor()
        predictor.initialize_models()
        
        # 모든 종목 예측
        results = predictor.predict_all_stocks()
        
        # 결과 리포트 생성
        predictor.generate_report(results)
        
        logger.info("모든 종목의 예측이 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}")
        raise
    finally:
        # 데이터베이스 연결 종료
        db.close()
        logger.info("데이터베이스 연결이 종료되었습니다.")

if __name__ == "__main__":
    main() 