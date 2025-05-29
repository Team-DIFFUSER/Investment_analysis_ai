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

from models.base_model import BaseModel, setup_gpu, enhanced_weighted_time_mse
from models.data_processor import DataProcessor
from models.evaluation import ModelEvaluator, evaluate_predictions
from database.database import DatabaseManager

# 환경 변수 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LGElectronicsModel(BaseModel):
    def __init__(self, use_cloud=False):
        super().__init__(use_cloud)
        self.stock_name = "LG전자"
        self.stock_code = "066570"
        self.sequence_length = 20
        self.data_processor = DataProcessor()
        self.evaluator = ModelEvaluator()
        self.db_manager = DatabaseManager()
        self.n_features = None  # 데이터 로드 시 설정
        self.model = None  # build_model은 데이터 로드 후에 호출
        self.load_model()  # 모델 로드
    
    def load_model(self):
        """저장된 모델 로드"""
        try:
            model_path = 'models/checkpoints/lg_electronics_best.h5'
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path, 
                    custom_objects={'enhanced_weighted_time_mse': enhanced_weighted_time_mse})
                logger.info("모델 로드 완료")
            else:
                logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def train(self, X_train, y_train, X_val, y_val):
        """모델 학습"""
        try:
            # 특성 수 설정 및 모델 빌드
            if self.n_features is None:
                self.n_features = X_train.shape[2]
                logger.info(f"특성 수 설정: {self.n_features}")
            
            # 모델 빌드
            self.build_model()
            
            # 콜백 설정
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/checkpoints/lg_electronics_best.h5',
                    monitor='val_loss',
                    save_best_only=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # 모델 학습
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise
    
    def load_stock_data(self) -> pd.DataFrame:
        """주가 데이터 로드"""
        try:
            query = """
            SELECT 
                time as date,
                stock_code as stock_code,
                stock_name as stock_name,
                open_price as open,
                high_price as high,
                low_price as low,
                close_price as close,
                volume as volume,
                market_cap as market_cap,
                foreign_holding as foreign_holding,
                foreign_holding_ratio as foreign_ratio
            FROM stock_prices
            WHERE stock_name = %s
            ORDER BY time;
            """
            stock_data = pd.DataFrame(self.db_manager.execute_query(query, (self.stock_name,)), columns=[
                'date', 'stock_code', 'stock_name', 'open', 'high', 'low', 
                'close', 'volume', 'market_cap', 'foreign_holding', 'foreign_ratio'
            ])
            
            # 숫자형 컬럼 변환
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap', 'foreign_holding', 'foreign_ratio']
            for col in numeric_columns:
                stock_data[col] = stock_data[col].astype(float)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"주가 데이터 로드 중 오류 발생: {str(e)}")
            raise
    
    def load_sentiment_data(self) -> pd.DataFrame:
        """감성 데이터 로드"""
        try:
            query = """
            SELECT 
                pub_date as date, title,
                finbert_positive, finbert_negative, finbert_neutral,
                finbert_sentiment
            FROM news_sentiment
            ORDER BY pub_date;
            """
            sentiment_data = pd.DataFrame(self.db_manager.execute_query(query), columns=[
                'date', 'title', 'finbert_positive', 'finbert_negative', 
                'finbert_neutral', 'finbert_sentiment'
            ])
            
            # 감성 점수 변환
            sentiment_columns = ['finbert_positive', 'finbert_negative', 'finbert_neutral']
            for col in sentiment_columns:
                sentiment_data[col] = pd.to_numeric(sentiment_data[col], errors='coerce')
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"감성 데이터 로드 중 오류 발생: {str(e)}")
            raise
    
    def load_economic_data(self) -> pd.DataFrame:
        """경제지표 데이터 로드"""
        try:
            query = """
            SELECT 
                time as date,
                treasury_10y,
                dollar_index,
                usd_krw,
                korean_bond_10y
            FROM economic_indicators
            ORDER BY time;
            """
            economic_data = pd.DataFrame(self.db_manager.execute_query(query), columns=[
                'date', 'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
            ])
            
            # 경제지표 변환
            economic_columns = ['treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y']
            for col in economic_columns:
                economic_data[col] = pd.to_numeric(economic_data[col], errors='coerce')
            
            return economic_data
            
        except Exception as e:
            logger.error(f"경제지표 데이터 로드 중 오류 발생: {str(e)}")
            raise
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """모든 데이터 로드"""
        try:
            # 주가 데이터 로드
            stock_query = """
            SELECT 
                time as date,
                stock_code,
                stock_name,
                open_price as open,
                high_price as high,
                low_price as low,
                close_price as close,
                volume,
                market_cap,
                foreign_holding,
                foreign_holding_ratio as foreign_ratio
            FROM stock_prices
            WHERE stock_name = %s
            ORDER BY time;
            """
            stock_data = pd.DataFrame(self.db_manager.execute_query(stock_query, (self.stock_name,)))
            
            # 감성 데이터 로드
            sentiment_query = """
            SELECT 
                pub_date as date,
                finbert_positive,
                finbert_negative,
                finbert_neutral,
                finbert_sentiment
            FROM news_sentiment
            WHERE pub_date >= %s
            ORDER BY pub_date;
            """
            sentiment_data = pd.DataFrame(self.db_manager.execute_query(
                sentiment_query, (stock_data['date'].min(),)
            ))
            
            # 경제지표 데이터 로드
            economic_query = """
            SELECT 
                time as date,
                treasury_10y,
                dollar_index,
                usd_krw,
                korean_bond_10y
            FROM economic_indicators
            WHERE time >= %s
            ORDER BY time;
            """
            economic_data = pd.DataFrame(self.db_manager.execute_query(
                economic_query, (stock_data['date'].min(),)
            ))
            
            # 데이터 전처리
            # 1. 날짜 형식 통일
            for df in [stock_data, sentiment_data, economic_data]:
                df['date'] = pd.to_datetime(df['date'])
            
            # 2. 숫자형 컬럼 변환
            numeric_columns = {
                'stock_data': ['open', 'high', 'low', 'close', 'volume', 'market_cap', 'foreign_holding', 'foreign_ratio'],
                'sentiment_data': ['finbert_positive', 'finbert_negative', 'finbert_neutral', 'finbert_sentiment'],
                'economic_data': ['treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y']
            }
            
            for col in numeric_columns['stock_data']:
                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
            for col in numeric_columns['sentiment_data']:
                sentiment_data[col] = pd.to_numeric(sentiment_data[col], errors='coerce')
            
            for col in numeric_columns['economic_data']:
                economic_data[col] = pd.to_numeric(economic_data[col], errors='coerce')
            
            # 3. 결측치 처리
            stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
            sentiment_data = sentiment_data.fillna(method='ffill').fillna(method='bfill')
            economic_data = economic_data.fillna(method='ffill').fillna(method='bfill')
            
            # 4. 데이터 검증
            if stock_data.empty or sentiment_data.empty or economic_data.empty:
                raise ValueError("데이터 로드 실패: 일부 데이터가 비어있습니다.")
            
            logger.info(f"주가 데이터: {len(stock_data)}행")
            logger.info(f"감성 데이터: {len(sentiment_data)}행")
            logger.info(f"경제 데이터: {len(economic_data)}행")
            
            return stock_data, sentiment_data, economic_data
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise
    
    def build_model(self):
        """LG전자 특화 모델 구조 정의"""
        if self.n_features is None:
            logger.warning("n_features가 설정되지 않았습니다. 기본값 30을 사용합니다.")
            self.n_features = 30
            
        # 입력 레이어
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM 레이어
        x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Attention 메커니즘
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=128
        )(x, x)
        
        # Residual connection
        x = tf.keras.layers.Add()([x, attention_output])
        
        # LSTM 레이어
        x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Dense 레이어
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # 출력 레이어 (5일 예측)
        outputs = tf.keras.layers.Dense(5)(x)
        
        # 모델 생성
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # 컴파일
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss=enhanced_weighted_time_mse,
            metrics=['mae']
        )
        
        logger.info(f"모델 구조 생성 완료 - 입력: {self.sequence_length}x{self.n_features}, 출력: 5")
    
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
    
    def predict_next_five_days(self, start_date: datetime) -> List[Dict]:
        """다음 5일 예측"""
        try:
            if self.model is None:
                raise ValueError("모델이 로드되지 않았습니다. 먼저 모델을 로드해주세요.")
            
            # 데이터 로드
            stock_data = self.load_stock_data()
            sentiment_data = self.load_sentiment_data()
            economic_data = self.load_economic_data()
            
            logger.info(f"주가 데이터: {len(stock_data)}행")
            logger.info(f"감성 데이터: {len(sentiment_data)}행")
            logger.info(f"경제 데이터: {len(economic_data)}행")
            
            # 예측 데이터 준비
            X = self.data_processor.prepare_prediction_data(
                stock_data, sentiment_data, economic_data, self.sequence_length
            )
            
            # 예측 수행
            predictions = self.model.predict(X)
            
            # 예측 결과를 날짜와 함께 반환
            result = []
            current_date = start_date
            
            for i in range(5):
                # 다음 영업일 계산
                current_date = get_next_business_day(current_date)
                result.append({
                    'date': current_date,
                    'price': float(predictions[0][i])
                })
            
            return result
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test) -> Dict:
        """모델 평가"""
        try:
            if self.model is None:
                raise ValueError("모델이 학습되지 않았습니다.")
            
            # 모델 평가
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            
            # 예측 수행
            predictions = self.model.predict(X_test)
            
            # 평가 지표 계산
            mse = np.mean((y_test - predictions) ** 2)
            mae = np.mean(np.abs(y_test - predictions))
            
            # 방향성 정확도 계산
            direction_true = np.sign(y_test)
            direction_pred = np.sign(predictions)
            direction_accuracy = np.mean(direction_true == direction_pred)
            
            return {
                'test_loss': float(test_loss[0]),  # 첫 번째 값만 반환
                'mse': float(mse),
                'mae': float(mae),
                'direction_accuracy': float(direction_accuracy)
            }
            
        except Exception as e:
            logger.error(f"모델 평가 중 오류 발생: {str(e)}")
            raise

if __name__ == "__main__":
    model = LGElectronicsModel(use_cloud=True)
    model.train() 