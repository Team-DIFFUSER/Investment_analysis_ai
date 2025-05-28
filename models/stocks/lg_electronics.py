import sys
import os
import tensorflow as tf
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from datetime import datetime, timedelta

from models.base_model import build_base_model, setup_gpu, enhanced_weighted_time_mse, EnsembleModel
from models.data_processor import DataProcessor
from models.evaluation import ModelEvaluator, evaluate_predictions
from database.database import DatabaseManager

# 환경 변수 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LGElectronicsModel:
    def __init__(self):
        self.stock_name = "LG전자"
        self.stock_code = "066570"
        self.sequence_length = 20
        self.data_processor = DataProcessor()
        self.evaluator = ModelEvaluator()
        self.db_manager = DatabaseManager()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        try:
            # 주가 데이터 로드
            query = """
            SELECT 
                time as 기준일자,
                stock_code as 종목코드,
                stock_name as 종목명,
                open_price as 시가,
                high_price as 고가,
                low_price as 저가,
                close_price as 현재가,
                volume as 거래량,
                market_cap as 시가총액,
                foreign_holding as 외국인보유,
                foreign_holding_ratio as 외국인비율
            FROM stock_prices
            WHERE stock_name = %s
            ORDER BY time;
            """
            stock_data = pd.DataFrame(self.db_manager.execute_query(query, (self.stock_name,)), columns=[
                '기준일자', '종목코드', '종목명', '시가', '고가', '저가', 
                '현재가', '거래량', '시가총액', '외국인보유', '외국인비율'
            ])
            
            # 감성 데이터 로드
            query = """
            SELECT 
                pub_date, title,
                finbert_positive, finbert_negative, finbert_neutral,
                finbert_sentiment
            FROM news_sentiment
            ORDER BY pub_date;
            """
            sentiment_data = pd.DataFrame(self.db_manager.execute_query(query), columns=[
                'PubDate', 'Title', 'finbert_positive', 'finbert_negative', 
                'finbert_neutral', 'finbert_sentiment'
            ])
            
            # 경제지표 데이터 로드
            query = """
            SELECT 
                time,
                treasury_10y,
                dollar_index,
                usd_krw,
                korean_bond_10y
            FROM economic_indicators
            ORDER BY time;
            """
            economic_data = pd.DataFrame(self.db_manager.execute_query(query), columns=[
                'time', 'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
            ])
            
            # 숫자형 컬럼 변환
            numeric_columns = ['시가', '고가', '저가', '현재가', '거래량', '시가총액', '외국인보유', '외국인비율']
            for col in numeric_columns:
                stock_data[col] = stock_data[col].astype(float)
            
            # 감성 점수 변환
            sentiment_columns = ['finbert_positive', 'finbert_negative', 'finbert_neutral']
            for col in sentiment_columns:
                sentiment_data[col] = pd.to_numeric(sentiment_data[col], errors='coerce')
            
            # 경제지표 변환
            economic_columns = ['treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y']
            for col in economic_columns:
                economic_data[col] = pd.to_numeric(economic_data[col], errors='coerce')
            
            return stock_data, sentiment_data, economic_data
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise
    
    def prepare_training_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame, economic_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataProcessor]:
        """학습 데이터 준비"""
        try:
            # 데이터 전처리
            X_train, y_train, X_val, y_val, scaler = self.data_processor.prepare_data(
                stock_data, sentiment_data, economic_data, self.sequence_length
            )
            
            return X_train, y_train, X_val, y_val, scaler
            
        except Exception as e:
            logger.error(f"학습 데이터 준비 중 오류 발생: {str(e)}")
            raise
    
    def train(self) -> None:
        """모델 학습"""
        try:
            # GPU 설정
            setup_gpu()
            
            # 데이터 로드 및 준비
            stock_data, sentiment_data, economic_data = self.load_data()
            X_train, y_train, X_val, y_val, scaler = self.prepare_training_data(
                stock_data, sentiment_data, economic_data
            )
            
            # 앙상블 모델 초기화 및 학습
            ensemble = EnsembleModel(input_shape=(X_train.shape[1], X_train.shape[2]))
            ensemble.build_models()
            histories = ensemble.train(X_train, y_train, X_val, y_val, scaler)
            
            logger.info("모델 학습 완료")
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise
    
    def get_latest_price(self) -> float:
        """가장 최근 주가 조회"""
        try:
            query = """
            SELECT close_price, time
            FROM stock_prices
            WHERE stock_name = %s
            ORDER BY time DESC
            LIMIT 1
            """
            result = self.db_manager.execute_query(query, (self.stock_name,))
            
            if result:
                price, time = result[0]
                logger.info(f"조회된 주가: {price:,.0f}원 (기준일: {time})")
                return float(price)
            else:
                logger.warning(f"경고: {self.stock_name} 종목의 주가 데이터를 찾을 수 없습니다. 기본값을 사용합니다.")
                return 81800.0
                
        except Exception as e:
            logger.error(f"최근 주가 조회 중 오류 발생: {e}")
            return 81800.0
    
    def predict(self, start_date: datetime) -> List[Dict]:
        """주가 예측"""
        try:
            # 현재 가격 가져오기
            last_price = self.get_latest_price()
            
            # 데이터 로드
            stock_data, sentiment_data, economic_data = self.load_data()
            
            # 예측 데이터 준비
            X = self.data_processor.prepare_prediction_data(
                stock_data, sentiment_data, economic_data, self.sequence_length
            )
            
            # 앙상블 모델 로드 및 예측
            ensemble = EnsembleModel(input_shape=(X.shape[1], X.shape[2]))
            ensemble.build_models()
            predictions = ensemble.predict(X)
            
            # 예측 결과를 실제 가격으로 변환
            predicted_prices = []
            current_price = last_price
            
            for pred in predictions[0]:
                next_price = current_price * (1 + pred)
                next_price = round(next_price / 100) * 100
                predicted_prices.append(next_price)
                current_price = next_price
            
            # 예측 결과 생성
            results = []
            for i, price in enumerate(predicted_prices):
                target_date = start_date + timedelta(days=i+1)
                results.append({
                    'date': target_date,
                    'price': price
                })
            
            return results
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise
    
    def evaluate(self, start_date: datetime, end_date: datetime) -> Dict:
        """모델 평가"""
        try:
            # 예측 수행
            predictions = self.predict(start_date)
            
            # 실제 가격 데이터 조회
            query = """
            SELECT time, close_price
            FROM stock_prices
            WHERE stock_name = %s
            AND time BETWEEN %s AND %s
            ORDER BY time
            """
            actual_data = pd.DataFrame(
                self.db_manager.execute_query(query, (self.stock_name, start_date, end_date)),
                columns=['date', 'actual_price']
            )
            
            # 예측 결과와 실제 가격 비교
            pred_dates = [p['date'] for p in predictions]
            pred_prices = [p['price'] for p in predictions]
            actual_prices = actual_data['actual_price'].values
            
            # 평가 수행
            evaluation = evaluate_predictions(pred_prices, pred_dates, actual_prices)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {str(e)}")
            raise

if __name__ == "__main__":
    model = LGElectronicsModel()
    model.train() 