import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .base_model import BaseStockModel

class PredictionManager:
    def __init__(self):
        """예측 관리자 초기화"""
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.predictions = {}
        self.predictions_data = {
            'stock_predictions': {},
            'error_metrics': {}
        }
        self._load_predictions_data()
        
    def _load_predictions_data(self):
        """예측 데이터 로드"""
        try:
            if os.path.exists(self.PREDICTIONS_FILE):
                with open(self.PREDICTIONS_FILE, 'r') as f:
                    data = json.load(f)
                    self.predictions_data = {
                        'stock_predictions': data.get('stock_predictions', {}),
                        'error_metrics': data.get('error_metrics', {})
                    }
            else:
                self.predictions_data = {
                    'stock_predictions': {},
                    'error_metrics': {}
                }
                
        except Exception as e:
            self.logger.error(f"예측 데이터 로드 중 오류 발생: {str(e)}")
            self.predictions_data = {
                'stock_predictions': {},
                'error_metrics': {}
            }
            
    def _save_predictions_data(self):
        """예측 데이터 저장"""
        try:
            data_path = os.path.join('models', 'predictions', 'predictions.json')
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w') as f:
                json.dump(self.predictions_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"예측 데이터 저장 중 오류 발생: {str(e)}")
            
    def add_model(self, stock_name: str, model: BaseStockModel):
        """모델 추가"""
        try:
            if not isinstance(model, BaseStockModel):
                raise ValueError(f"모델은 BaseStockModel을 상속받아야 합니다: {type(model)}")
                
            # 모델 초기화 확인
            if not model.is_initialized():
                self.logger.warning(f"{stock_name} 모델이 초기화되지 않았습니다. 초기화를 시도합니다.")
                model.initialize()
                
            self.models[stock_name] = model
            if stock_name not in self.predictions_data['stock_predictions']:
                self.predictions_data['stock_predictions'][stock_name] = {}
            if stock_name not in self.predictions_data['error_metrics']:
                self.predictions_data['error_metrics'][stock_name] = {}
            self.logger.info(f"모델 추가 완료: {stock_name}")
            
        except Exception as e:
            self.logger.error(f"모델 추가 중 오류 발생: {str(e)}")
            
    def run_daily_prediction(self):
        """일일 예측 실행"""
        try:
            for stock_name, model in self.models.items():
                self.logger.info(f"{stock_name} 예측 시작")
                
                # 모델 초기화 확인
                if not model.is_initialized():
                    self.logger.warning(f"{stock_name} 모델이 초기화되지 않았습니다. 초기화를 시도합니다.")
                    model.initialize()
                
                # 예측 실행
                predictions = model.predict_next_five_days()
                if not predictions:
                    self.logger.error(f"{stock_name} 예측 실패")
                    continue
                
                # 예측 결과 저장
                self.predictions_data['stock_predictions'][stock_name] = predictions
                
                # 예측 결과 로깅
                self.logger.info(f"{stock_name} 예측 결과: {predictions}")
                
        except Exception as e:
            self.logger.error(f"일일 예측 실행 중 오류 발생: {str(e)}")
            
    def _update_previous_predictions(self, stock_name: str):
        """이전 예측값 업데이트"""
        try:
            if stock_name not in self.predictions_data['stock_predictions']:
                return
                
            today = datetime.now().strftime('%Y-%m-%d')
            model = self.models[stock_name]
            
            # 실제 데이터 로드
            data = model.load_data()
            if data.empty:
                return
                
            # 가장 최근 예측 데이터 찾기
            prediction_dates = sorted(self.predictions_data['stock_predictions'][stock_name].keys())
            if not prediction_dates:
                return
                
            last_prediction_date = prediction_dates[-1]
            predictions = self.predictions_data['stock_predictions'][stock_name][last_prediction_date]
            
            # 예측값과 실제값 비교
            for date, pred_data in predictions['predictions'].items():
                if date in data.index:
                    actual_price = data.loc[date, 'Close']
                    model.update_predictions(date, actual_price)
                    
        except Exception as e:
            self.logger.error(f"이전 예측값 업데이트 중 오류 발생: {str(e)}")
            
    def get_prediction_history(self, stock_name: str) -> Dict:
        """종목별 예측 이력 조회"""
        try:
            if stock_name in self.predictions_data['stock_predictions']:
                return self.predictions_data['stock_predictions'][stock_name]
            return {}
            
        except Exception as e:
            self.logger.error(f"예측 이력 조회 중 오류 발생: {str(e)}")
            return {}
            
    def get_error_statistics(self, stock_name: str) -> Dict:
        """종목별 예측 오차 통계"""
        try:
            if stock_name not in self.models:
                return {}
                
            model = self.models[stock_name]
            errors = list(model.error_history.values())
            
            if not errors:
                return {}
                
            return {
                'mean_error': sum(errors) / len(errors),
                'max_error': max(errors),
                'min_error': min(errors),
                'error_count': len(errors)
            }
            
        except Exception as e:
            self.logger.error(f"오차 통계 조회 중 오류 발생: {str(e)}")
            return {} 