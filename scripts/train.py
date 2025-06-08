import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stocks.lg_electronics import LGElectronicsModel
from database.database import DatabaseManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockTrainer:
    def __init__(self):
        """주식 학습기 초기화"""
        self.db_manager = DatabaseManager()
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

    def train_single_stock(self, stock_name: str, model: Any) -> Dict[str, Any]:
        """단일 종목 학습 수행"""
        try:
            logger.info(f"{stock_name} 학습 시작")
            
            # 학습 데이터 로드
            training_data = self.load_training_data(stock_name)
            if training_data is None or training_data.empty:
                raise ValueError(f"{stock_name}의 학습 데이터가 없습니다.")
            
            # 모델 학습
            history = model.train_model()
            
            # 학습 결과가 None이 아닌 경우에만 저장
            if history is not None:
                self.save_training_results(stock_name, history)
            
            return {
                'stock_name': stock_name,
                'status': 'success',
                'history': history
            }
        except Exception as e:
            logger.error(f"{stock_name} 학습 중 오류 발생: {str(e)}")
            return {
                'stock_name': stock_name,
                'status': 'error',
                'error': str(e)
            }

    def load_training_data(self, stock_name: str) -> pd.DataFrame:
        """학습 데이터 로드"""
        try:
            query = """
            SELECT time, open_price, high_price, low_price, close_price, volume
            FROM stock_prices
            WHERE stock_name = %s
            ORDER BY time
            """
            params = (stock_name,)
            results = self.db_manager.execute_query(query, params)
            return pd.DataFrame(results)
        except Exception as e:
            logger.error(f"학습 데이터 로드 중 오류 발생: {str(e)}")
            return None

    def save_training_results(self, stock_name: str, history: Dict[str, List[float]]) -> None:
        """학습 결과 저장"""
        try:
            # history가 None인 경우 처리
            if history is None:
                logger.warning(f"{stock_name}의 학습 히스토리가 없습니다.")
                return

            # history가 dict가 아닌 경우 처리
            if not isinstance(history, dict):
                logger.warning(f"{stock_name}의 학습 히스토리가 올바른 형식이 아닙니다.")
                return

            # 필요한 키가 있는지 확인
            required_keys = ['loss', 'val_loss', 'mae', 'val_mae']
            if not all(key in history for key in required_keys):
                logger.warning(f"{stock_name}의 학습 히스토리에 필요한 키가 없습니다.")
                return

            # 학습 결과를 데이터베이스에 저장
            query = """
            INSERT INTO model_training_history (
                stock_name, training_date, loss, val_loss, mae, val_mae
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (
                stock_name,
                datetime.now(),
                float(history['loss'][-1]) if history['loss'] else None,
                float(history['val_loss'][-1]) if history['val_loss'] else None,
                float(history['mae'][-1]) if history['mae'] else None,
                float(history['val_mae'][-1]) if history['val_mae'] else None
            )
            self.db_manager.execute_query(query, params)
            
            # 모델 저장
            model_path = os.path.join('models', 'checkpoints', f'{stock_name}_model.h5')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.models[stock_name].model.save(model_path)
            
            logger.info(f"{stock_name} 학습 결과 저장 완료")
        except Exception as e:
            logger.error(f"학습 결과 저장 중 오류 발생: {str(e)}")
            raise

    def train_all_stocks(self) -> Dict[str, Any]:
        """모든 종목에 대한 학습 수행"""
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 종목에 대한 학습 작업 제출
            future_to_stock = {
                executor.submit(self.train_single_stock, stock_name, model): stock_name
                for stock_name, model in self.models.items()
            }
            
            # 진행 상황 표시
            for future in tqdm(as_completed(future_to_stock), total=len(future_to_stock), desc="종목 학습 진행률"):
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
        """학습 결과 리포트 생성"""
        try:
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            error_count = len(results) - success_count
            
            # 성공한 종목의 학습 결과 문자열 생성
            success_results = []
            for stock, result in results.items():
                if result['status'] == 'success':
                    history = result.get('history', {})
                    if history and isinstance(history, dict):
                        loss = history.get('loss', [None])[-1]
                        val_loss = history.get('val_loss', [None])[-1]
                        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                        success_results.append(f"- {stock}: loss={loss_str}, val_loss={val_loss_str}")
                    else:
                        success_results.append(f"- {stock}: 학습 히스토리 없음")
            
            # 실패한 종목 목록 생성
            failed_stocks = [f"- {stock}: {result['error']}" 
                           for stock, result in results.items() 
                           if result['status'] == 'error']
            
            report = f"""
            학습 결과 리포트
            ==============
            총 종목 수: {len(results)}
            성공: {success_count}
            실패: {error_count}
            
            실패한 종목:
            {chr(10).join(failed_stocks) if failed_stocks else "없음"}
            
            성공한 종목의 학습 결과:
            {chr(10).join(success_results) if success_results else "없음"}
            """
            
            logger.info(report)
            
            # 리포트를 파일로 저장
            report_path = os.path.join('data', 'reports', f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
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
        logger.info("모델 학습을 시작합니다.")
        
        # 데이터베이스 연결
        db_manager = DatabaseManager()
        
        # 학습기 초기화
        trainer = StockTrainer()
        trainer.initialize_models()
        
        # 모든 종목 학습
        results = trainer.train_all_stocks()
        
        # 결과 리포트 생성
        trainer.generate_report(results)
        
        logger.info("모든 종목의 학습이 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        raise
    finally:
        # 데이터베이스 연결 종료
        db_manager.close()
        logger.info("데이터베이스 연결이 종료되었습니다.")

if __name__ == "__main__":
    main() 