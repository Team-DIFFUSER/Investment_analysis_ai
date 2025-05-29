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
        self.models = []  # 앙상블 모델 리스트
        self.num_models = 3  # 앙상블 모델 수
        self.load_models()  # 모델 로드
    
    def load_models(self):
        """저장된 앙상블 모델 로드"""
        try:
            for i in range(self.num_models):
                model_path = f'models/checkpoints/lg_electronics_model_{i+1}.h5'
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path, 
                        custom_objects={'enhanced_weighted_time_mse': enhanced_weighted_time_mse})
                    self.models.append(model)
                    logger.info(f"모델 {i+1} 로드 완료")
                else:
                    logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
            if not self.models:
                raise ValueError("모든 모델 로드 실패")
                
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def train(self, X_train, y_train, X_val, y_val):
        """앙상블 모델 학습"""
        try:
            # 특성 수 설정
            if self.n_features is None:
                self.n_features = X_train.shape[2]
                logger.info(f"특성 수 설정: {self.n_features}")
            
            # 각 모델 학습
            histories = []
            for i in range(self.num_models):
                logger.info(f"\n모델 {i+1}/{self.num_models} 학습 시작")
                
                # 모델 빌드
                model = self.build_model()
                
                # 콜백 설정
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=120,
                        restore_best_weights=True,
                        min_delta=0.0001
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.15,
                        patience=25,
                        min_lr=1e-7,
                        min_delta=0.0001
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        f'models/checkpoints/lg_electronics_model_{i+1}.h5',
                        monitor='val_loss',
                        save_best_only=True
                    )
                ]
                
                # 데이터셋 최적화
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_dataset = train_dataset.cache()
                train_dataset = train_dataset.shuffle(buffer_size=50000)
                train_dataset = train_dataset.batch(40)
                train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
                
                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                val_dataset = val_dataset.cache()
                val_dataset = val_dataset.batch(40)
                val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
                
                # 모델 학습
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=450,
                    callbacks=callbacks,
                    verbose=1
                )
                
                histories.append(history)
                self.models.append(model)
            
            return histories
            
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
        
        # Multi-scale Convolutional Input (MCI)
        conv_outputs = []
        kernel_sizes = [2, 3, 5, 7, 11]  # 다양한 시간 스케일
        for kernel_size in kernel_sizes:
            conv = tf.keras.layers.Conv1D(128, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv_outputs.append(conv)
        
        # 컨볼루션 출력 결합
        x = tf.keras.layers.Concatenate()(conv_outputs)
        
        # Attention 메커니즘
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=128
        )(x, x)
        
        # Residual connection
        x = tf.keras.layers.Add()([x, attention_output])
        
        # GRU 레이어
        gru_output = tf.keras.layers.GRU(256, return_sequences=True)(x)
        gru_output = tf.keras.layers.Dropout(0.3)(gru_output)
        
        # 추가 Attention 레이어
        attention_output2 = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=256
        )(gru_output, gru_output)
        
        # Residual connection
        x = tf.keras.layers.Add()([gru_output, attention_output2])
        
        # 시퀀스의 마지막 타임스텝만 선택
        x = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(x)
        
        # Dense 레이어
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # 출력 레이어 (5% 제한을 위한 tanh 활성화 함수 사용)
        outputs = tf.keras.layers.Dense(5, activation='tanh')(x) * 0.05  # tanh의 출력 범위를 -0.05에서 0.05로 조정
        
        # 모델 생성
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # 컴파일
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)  # 더 안정적인 학습을 위해 학습률 감소
        model.compile(
            optimizer=optimizer,
            loss=enhanced_weighted_time_mse,
            metrics=['mae'],
            jit_compile=True
        )
        
        logger.info(f"모델 구조 생성 완료 - 입력: {self.sequence_length}x{self.n_features}, 출력: 5")
        return model
    
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
            if not self.models:
                raise ValueError("모델이 로드되지 않았습니다. 먼저 모델을 로드해주세요.")
            
            # 1. 이전 예측값 조회
            end_date = start_date + timedelta(days=4)
            previous_predictions = self.get_previous_predictions(start_date, end_date)
            
            # 2. 새로운 데이터로 예측
            # 최근 20일 데이터를 가져와서 예측에 사용
            query_start_date = start_date - timedelta(days=20)
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
            
            # 앙상블 예측 수행
            predictions = []
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            new_predictions = np.mean(predictions, axis=0)[0]  # 앙상블 평균
            
            # 3. 예측값 조정
            adjusted_predictions = []
            last_actual_price = float(stock_data['close'].iloc[-1])
            
            for i in range(5):
                target_date = start_date + timedelta(days=i)
                
                if i == 0:
                    # 첫날은 실제값 사용
                    predicted_price = last_actual_price
                else:
                    # 이전 예측값이 있는 경우 조정
                    if not previous_predictions.empty and i < len(previous_predictions):
                        prev_pred = previous_predictions.iloc[i-1]['predicted_price']
                        next_pred = last_actual_price * (1 + float(new_predictions[i]))
                        adjustment = self.calculate_prediction_adjustment(
                            last_actual_price, prev_pred, next_pred
                        )
                        predicted_price = next_pred + adjustment
                    else:
                        # 이전 예측값이 없는 경우 현재 예측값 사용
                        predicted_price = last_actual_price * (1 + float(new_predictions[i]))
                
                # 100원 단위로 반올림
                predicted_price = round(predicted_price / 100) * 100
                
                adjusted_predictions.append({
                    'date': target_date,
                    'price': predicted_price
                })
            
            return adjusted_predictions
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test) -> Dict:
        """모델 평가"""
        try:
            if not self.models:
                raise ValueError("모델이 학습되지 않았습니다.")
            
            # 모델 평가
            test_loss = []
            for model in self.models:
                test_loss.append(model.evaluate(X_test, y_test, verbose=0))
            
            # 예측 수행
            predictions = []
            for model in self.models:
                pred = model.predict(X_test)
                predictions.append(pred)
            
            # 평가 지표 계산
            mse = []
            mae = []
            for i in range(len(self.models)):
                mse.append(np.mean((y_test - predictions[i]) ** 2))
                mae.append(np.mean(np.abs(y_test - predictions[i])))
            
            # 방향성 정확도 계산
            direction_true = np.sign(y_test)
            direction_pred = np.sign(predictions[0])
            direction_accuracy = np.mean(direction_true == direction_pred)
            
            return {
                'test_loss': test_loss,
                'mse': mse,
                'mae': mae,
                'direction_accuracy': direction_accuracy
            }
            
        except Exception as e:
            logger.error(f"모델 평가 중 오류 발생: {str(e)}")
            raise
    
    def get_previous_predictions(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """이전 예측 결과를 데이터베이스에서 가져오는 함수"""
        try:
            query = """
            SELECT prediction_date, target_date, predicted_price
            FROM stock_predictions
            WHERE stock_code = %s
            AND prediction_date BETWEEN %s AND %s
            ORDER BY prediction_date, target_date
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (self.stock_code, start_date, end_date))
                    results = cur.fetchall()
            
            if not results:
                return pd.DataFrame(columns=['prediction_date', 'target_date', 'predicted_price'])
            
            df = pd.DataFrame(results, columns=['prediction_date', 'target_date', 'predicted_price'])
            return df
            
        except Exception as e:
            logger.error(f"이전 예측 결과 조회 중 오류 발생: {str(e)}")
            raise
    
    def calculate_prediction_adjustment(self, actual_price: float, prev_predicted_price: float, next_predicted_price: float) -> float:
        """예측값 조정을 계산하는 함수
        
        Args:
            actual_price (float): 실제 가격
            prev_predicted_price (float): 이전 예측 가격
            next_predicted_price (float): 다음 예측 가격
            
        Returns:
            float: 조정값
        """
        try:
            # 이전 예측의 오차 계산
            prev_error = actual_price - prev_predicted_price
            
            # 오차의 절대값이 너무 크면 조정값을 제한
            max_adjustment = actual_price * 0.02  # 최대 2% 조정
            
            # 오차에 따른 조정값 계산
            adjustment = prev_error * 0.3  # 오차의 30%만큼 조정
            
            # 조정값 제한
            adjustment = max(min(adjustment, max_adjustment), -max_adjustment)
            
            # 다음 예측값이 실제 가격과 너무 다르면 조정값 증가
            price_diff_ratio = abs(next_predicted_price - actual_price) / actual_price
            if price_diff_ratio > 0.05:  # 5% 이상 차이나면
                adjustment *= 1.5
            
            return adjustment
            
        except Exception as e:
            logger.error(f"예측값 조정 계산 중 오류 발생: {str(e)}")
            return 0.0  # 오류 발생 시 조정하지 않음

if __name__ == "__main__":
    model = LGElectronicsModel(use_cloud=True)
    model.train() 