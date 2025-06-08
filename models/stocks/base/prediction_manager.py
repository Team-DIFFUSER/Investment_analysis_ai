import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .base_model import BaseStockModel

class PredictionManager:
    def __init__(self):
        """예측 관리자 초기화"""
        self.logger = logging.getLogger('models.stocks.prediction_manager')
        self.models: Dict[str, BaseStockModel] = {}
        self.predictions_data = {}
        self._load_predictions_data()
        
    def _load_predictions_data(self):
        """예측 데이터 로드"""
        try:
            data_path = os.path.join('models', 'predictions', 'predictions.json')
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    self.predictions_data = json.load(f)
            else:
                self.predictions_data = {'stock_predictions': {}}
                
        except Exception as e:
            self.logger.error(f"예측 데이터 로드 중 오류 발생: {str(e)}")
            self.predictions_data = {'stock_predictions': {}}
            
    def _save_predictions_data(self):
        """예측 데이터 저장"""
        try:
            data_path = os.path.join('models', 'predictions', 'predictions.json')
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w') as f:
                json.dump(self.predictions_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"예측 데이터 저장 중 오류 발생: {str(e)}")
            
    def add_model(self, model: BaseStockModel):
        """새로운 종목 모델 추가"""
        try:
            self.models[model.stock_name] = model
            if model.stock_name not in self.predictions_data['stock_predictions']:
                self.predictions_data['stock_predictions'][model.stock_name] = {}
            self.logger.info(f"모델 추가 완료: {model.stock_name}")
            
        except Exception as e:
            self.logger.error(f"모델 추가 중 오류 발생: {str(e)}")
            
    def run_daily_predictions(self):
        """모든 종목에 대한 일일 예측 실행"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            for stock_name, model in self.models.items():
                self.logger.info(f"{stock_name} 예측 시작")
                
                # 이전 예측값 업데이트
                self._update_previous_predictions(stock_name)
                
                # 새로운 예측 수행
                predictions = model.predict_next_days()
                if predictions:
                    self.predictions_data['stock_predictions'][stock_name][today] = predictions
                    self.logger.info(f"{stock_name} 예측 완료: {predictions}")
                    
            # 예측 데이터 저장
            self._save_predictions_data()
            
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