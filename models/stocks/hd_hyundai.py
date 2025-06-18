import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Tuple, List
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from database.database import DatabaseManager
from utils.date_utils import get_next_five_business_days
from .base.base_model import BaseStockModel
from tensorflow.keras import layers

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU 설정 단순화
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print(f"GPU 사용 가능: {gpus[0]}")
        # Mixed Precision 설정
        tf.keras.mixed_precision.set_global_policy('float32')
        print("Mixed Precision 비활성화됨")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")
else:
    print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")

# 기존 세션 정리 및 메모리 해제
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

# 기본 환경 변수 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

# TensorFlow 최적화 설정
tf.config.optimizer.set_jit(False)  # XLA JIT 컴파일 비활성화
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_model_pruning": False,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": True,
    "implementation_selector": True,
    "auto_mixed_precision": False
})

logger.info(f"TensorFlow 버전: {tf.__version__}")

# 시드 고정
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

class HDHyundaiModel(BaseStockModel):
    def __init__(self):
        """HD현대중공업 주가 예측 모델 초기화"""
        super().__init__('HD현대중공업', '329180.KS')
        self.db_manager = DatabaseManager()
        self.n_features = None  # 특성 수 초기화
        self.models = []  # 앙상블 모델 리스트
        self.num_models = 3  # 앙상블 모델 수
        self.logger = logging.getLogger(__name__)
        
        # GPU 사용 가능 여부 확인
        self.device = tf.config.list_physical_devices('GPU')[0] if tf.config.list_physical_devices('GPU') else 'CPU'
        self.logger.info(f"모델이 {self.device}에서 실행됩니다.")
        self.model = None
        self._initialized = False

    def __del__(self):
        """소멸자: 데이터베이스 연결 종료"""
        try:
            if hasattr(self, 'db_manager'):
                self.db_manager.close()
                self.logger.info("데이터베이스 연결이 종료되었습니다.")
        except Exception as e:
            self.logger.error(f"데이터베이스 연결 종료 중 오류 발생: {str(e)}")

    def load_data(self) -> pd.DataFrame:
        """HD현대중공업 주가 데이터 로드"""
        try:
            # 데이터베이스에서 주가 데이터 가져오기
            data = self.db_manager.get_stock_data('A329180')  # HD현대중공업 종목코드
            
            if data.empty:
                self.logger.error("데이터베이스에서 데이터를 찾을 수 없습니다.")
                return pd.DataFrame()
            
            self.logger.info(f"데이터 로드 완료: {len(data)} 행")
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """학습 데이터 준비"""
        try:
            # 데이터 로드
            data = self.load_data()
            
            if data.empty:
                raise ValueError("데이터가 비어있습니다.")
            
            # 데이터 전처리
            processed_data = self.enhanced_preprocessing(data)
            
            # 학습 데이터 준비
            X, y = self.prepare_data(processed_data)
            
            # 학습/검증/테스트 데이터 분할 (80/10/10)
            train_size = int(len(X) * 0.8)
            val_size = int(len(X) * 0.1)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            self.logger.info(f"학습 데이터 준비 완료: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            return X_train, y_train, X_val, y_val, X_test, y_test, self.scaler
            
        except Exception as e:
            self.logger.error(f"학습 데이터 준비 중 오류 발생: {str(e)}")
            raise

    def train_model(self) -> Dict[str, List[float]]:
        """모델 학습 (최적화된 버전)"""
        try:
            # 데이터 로드
            data = self.load_data()
            self.logger.info(f"데이터 로드 완료: {len(data)} 행")
            
            # 데이터 전처리
            processed_data = self.enhanced_preprocessing(data)
            
            # 학습 데이터 준비
            X, y = self.prepare_data(processed_data)
            self.logger.info(f"데이터 준비 완료: X shape={X.shape}, y shape={y.shape}")
            
            # 데이터 분할 (60-20-20)
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.2)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            self.logger.info(f"학습 데이터 준비 완료: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            # 모델 생성
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # 최적화된 콜백 설정
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,  # patience 증가
                    restore_best_weights=True,
                    min_delta=0.001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,  # patience 증가
                    min_lr=0.000001,  # 최소 학습률 감소
                    verbose=0
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join('models', 'checkpoints', f'{self.stock_name}_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0
                )
            ]
            
            # 최적화된 모델 학습
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,  # 에포크 수 증가
                batch_size=128,  # 배치 사이즈 감소
                callbacks=callbacks,
                verbose=0
            )
            
            # 간단한 모델 평가
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            self.logger.info(f"평가 메트릭: {dict(zip(self.model.metrics_names, test_loss))}")
            
            # 예측 및 R2 계산
            y_pred = self.model.predict(X_test, verbose=0)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            self.logger.info(f"모델 평가 결과: {metrics}")
            
            # 모델 저장
            model_path = os.path.join('models', 'checkpoints', f'{self.stock_name}_model.h5')
            self.model.save(model_path)
            self.logger.info(f"모델이 저장되었습니다: {model_path}")
            
            # 백업 저장
            backup_path = os.path.join('models', 'backup', f'{self.stock_name}_model.h5')
            self.model.save(backup_path)
            self.logger.info(f"모델 백업이 저장되었습니다: {backup_path}")
            
            self.logger.info("모델 학습이 완료되었습니다.")
            
            return history.history
            
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise

    def predict_next_day(self) -> float:
        """다음 날 주가 예측"""
        try:
            # 최근 데이터 로드
            data = self.load_data()
            recent_data = data.tail(self.sequence_length)
            
            if len(recent_data) < self.sequence_length:
                raise ValueError(f"충분한 데이터가 없습니다. 필요: {self.sequence_length}, 현재: {len(recent_data)}")
            
            # 데이터 전처리
            processed_data = self.enhanced_preprocessing(recent_data)
            
            # 예측 데이터 준비
            X, _ = self.prepare_data(processed_data)
            
            # 예측 수행
            prediction = self.predict(X[-1:])
            
            # 예측값 역변환
            prediction = self.scaler.inverse_transform(
                np.concatenate([np.zeros((1, 3)), prediction.reshape(-1, 1), np.zeros((1, 16))], axis=1)
            )[0, 3]
            
            # 예측 결과 저장
            self.db_manager.save_prediction(
                stock_code='A329180',
                stock_name='HD현대중공업',
                prediction_date=datetime.now(),
                target_date=datetime.now() + timedelta(days=1),
                predicted_price=float(prediction)
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            raise

    def load_model(self):
        """모델 로드"""
        try:
            # 프로젝트 루트 디렉토리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            
            # 모델 파일 경로
            model_path = os.path.join(project_root, 'models', 'checkpoints', f'{self.stock_name}_model.h5')
            self.logger.info(f"모델 파일 검색: {model_path}")
            
            if os.path.exists(model_path):
                self.logger.info(f"모델 파일 발견: {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info("모델 로드 성공")
                return self.model
            else:
                self.logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
                self.model = None
                return None
                
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            self.model = None
            return None
    
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
    
    def build_model(self, input_shape: tuple) -> tf.keras.Model:
        """HD현대중공업 전용 모델 구축"""
        try:
            # 입력 레이어
            inputs = layers.Input(shape=input_shape)
            
            # 첫 번째 LSTM 레이어 (recurrent_dropout 제거)
            x = layers.LSTM(256, input_shape=input_shape, return_sequences=True,
                          kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal',
                          )(inputs)  # recurrent_dropout 추가
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)  # dropout 비율 증가
            
            # Attention 메커니즘
            attention = layers.MultiHeadAttention(
                num_heads=8,  # attention head 수 증가
                key_dim=32
            )(x, x)
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
            
            # 두 번째 LSTM 레이어 (recurrent_dropout 제거)
            x = layers.LSTM(128, return_sequences=True,
                          kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal',
                          )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            # 세 번째 LSTM 레이어 (recurrent_dropout 제거)
            x = layers.LSTM(64, return_sequences=False,
                          kernel_initializer='glorot_uniform',
                          recurrent_initializer='orthogonal',
                          )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            # Dense 레이어
            x = layers.Dense(128, activation='relu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # L2 정규화 추가
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            x = layers.Dense(64, activation='relu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            # 출력 레이어
            outputs = layers.Dense(1, dtype='float32')(x)
            
            # 모델 생성
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # 옵티마이저 설정 (Mixed Precision 호환)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                clipnorm=1.0
            )
            
            # 모델 컴파일
            model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"모델 구축 중 오류 발생: {str(e)}")
            return None
            
    def is_holiday(self, date: datetime) -> bool:
        """공휴일 체크"""
        # 주말 체크
        if date.weekday() >= 5:  # 5: 토요일, 6: 일요일
            return True
            
        # 공휴일 체크 (2025년 기준)
        holidays_2025 = [
            '2025-01-01',  # 신정
            '2025-02-09',  # 설날
            '2025-02-10',  # 설날
            '2025-02-11',  # 설날
            '2025-03-01',  # 삼일절
            '2025-05-05',  # 어린이날
            '2025-05-15',  # 부처님오신날
            '2025-06-06',  # 현충일
            '2025-08-15',  # 광복절
            '2025-09-28',  # 추석
            '2025-09-29',  # 추석
            '2025-09-30',  # 추석
            '2025-10-03',  # 개천절
            '2025-10-09',  # 한글날
            '2025-12-25',  # 크리스마스
        ]
        
        return date.strftime('%Y-%m-%d') in holidays_2025

    def get_next_trading_day(self, date: datetime) -> datetime:
        """다음 거래일 구하기"""
        next_day = date + timedelta(days=1)
        while self.is_holiday(next_day):
            next_day += timedelta(days=1)
        return next_day

    def predict_next_five_days(self) -> List[float]:
        """다음 5일 예측"""
        try:
            # 데이터 로드
            data = self.load_data()
            if data.empty:
                raise ValueError("데이터가 비어있습니다.")
            
            # 데이터 전처리
            processed_data = self.enhanced_preprocessing(data)
            if processed_data.empty:
                raise ValueError("데이터 전처리 실패")
            
            # 학습 데이터로 스케일러 fit
            features = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'MA5', 'MA20', 'MA60', 'MA120',
                'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
                'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
                'Stoch_K', 'Stoch_D', 'ATR',
                'Volume_MA5', 'Volume_MA20', 'Volume_Ratio',
                'Price_Change', 'Price_Change_MA5', 'Price_Change_MA20',
                'Volatility', 'Volatility_MA5',
                'ROC', 'Momentum', 'ADX']
            
            # 스케일러 적용 전 NaN 검증
            feature_data = processed_data[features]
            if feature_data.isnull().any().any():
                self.logger.warning("스케일러 적용 전 NaN 값 발견. 추가 처리 중...")
                feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
                processed_data[features] = feature_data
            
            # 무한대 값 처리
            processed_data[features] = processed_data[features].replace([np.inf, -np.inf], 0)
            
            # 전체 데이터로 스케일러 학습
            self.scaler.fit(processed_data[features])
            
            # 마지막 sequence_length일의 데이터만 사용
            last_sequence = processed_data[features].iloc[-self.sequence_length:].values
            
            # 스케일링 전 최종 검증
            if np.isnan(last_sequence).any():
                self.logger.error("스케일링 전 입력 데이터에 NaN 값이 있습니다.")
                return []
            
            last_sequence = self.scaler.transform(last_sequence)
            
            # 예측을 위한 입력 데이터 준비
            X = last_sequence.reshape(1, self.sequence_length, len(features))
            self.logger.info(f"예측 입력 데이터 shape: {X.shape}")
            
            # 예측
            predictions = []
            current_sequence = X.copy()
            
            # 모델이 제대로 로드되었는지 확인
            if self.model is None:
                self.logger.error("모델이 로드되지 않았습니다.")
                return []
            
            # 예측 날짜 계산 (마지막 데이터 다음날부터 5거래일)
            last_date = data.index[-1].date()
            prediction_dates = []
            current_date = self.get_next_trading_day(last_date)
            
            # 5개의 거래일 찾기
            while len(prediction_dates) < 5:
                if not self.is_holiday(current_date):
                    prediction_dates.append(current_date)
                current_date = self.get_next_trading_day(current_date)
            
            for i in range(5):
                # 다음 날 예측
                next_day_pred = self.model.predict(current_sequence, verbose=0)[0][0]
                
                # 예측값이 NaN인지 확인
                if np.isnan(next_day_pred):
                    self.logger.error(f"예측값이 NaN입니다. 인덱스: {i}")
                    return []
                
                predictions.append(next_day_pred)
                
                # 예측값을 현재 시퀀스에 추가
                new_sequence = current_sequence[0, 1:, :]
                new_row = np.zeros((1, len(features)))
                new_row[0, 3] = next_day_pred  # close_price 위치에 예측값 저장
                new_sequence = np.vstack([new_sequence, new_row])
                current_sequence = new_sequence.reshape(1, self.sequence_length, len(features))
            
            # 예측값 역스케일링
            predictions_array = np.zeros((len(predictions), len(features)))
            predictions_array[:, 3] = predictions  # close_price 컬럼에 예측값 저장
            predictions_array = self.scaler.inverse_transform(predictions_array)
            predictions = predictions_array[:, 3]  # close_price 컬럼만 추출
            
            # 100원 단위로 반올림
            predictions = np.round(predictions / 100) * 100
            
            # 예측 결과를 데이터베이스에 저장
            for pred_date, pred_price in zip(prediction_dates, predictions):
                self.db_manager.save_prediction(
                    stock_code='A329180',
                    stock_name='HD현대중공업',
                    prediction_date=datetime.now(),
                    target_date=pred_date,
                    predicted_price=float(pred_price)
                )
                self.logger.info(f"예측 결과 저장: {pred_date.strftime('%Y-%m-%d')} - {pred_price:,.0f}원")
            
            self.logger.info("예측 완료")
            return predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            return []

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        try:
            # 예측 수행
            y_pred = self.model.predict(X_test, verbose=0)
            
            # Shape 조정
            y_pred = y_pred.reshape(-1)
            y_test = y_test.reshape(-1)
            
            # 메트릭 계산
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            self.logger.info(f"평가 메트릭: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"모델 평가 중 오류 발생: {str(e)}")
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

    def save_prediction(self, prediction: float, target_date: datetime) -> None:
        """예측 결과 저장"""
        try:
            self.db_manager.save_prediction(
                stock_code=self.symbol,
                stock_name=self.stock_name,
                prediction_date=datetime.now(),
                target_date=target_date,
                predicted_price=float(prediction)
            )
        except Exception as e:
            self.logger.error(f"예측 결과 저장 중 오류 발생: {str(e)}")
            raise

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """학습 데이터 준비"""
        try:
            if data.empty:
                self.logger.error("입력 데이터가 비어있습니다.")
                return np.array([]), np.array([])

            # 필요한 특성 목록
            features = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'MA5', 'MA20', 'MA60', 'MA120',
                'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
                'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
                'Stoch_K', 'Stoch_D', 'ATR',
                'Volume_MA5', 'Volume_MA20', 'Volume_Ratio',
                'Price_Change', 'Price_Change_MA5', 'Price_Change_MA20',
                'Volatility', 'Volatility_MA5',
                'ROC', 'Momentum', 'ADX'
            ]

            # 누락된 특성 확인
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                self.logger.error(f"누락된 특성들: {missing_features}")
                return np.array([]), np.array([])

            # 데이터 스케일링
            feature_data = data[features].copy()
            self.logger.info(f"스케일링 전 데이터 형태: {feature_data.shape}")
            
            # 결측치 확인 및 처리
            if feature_data.isnull().any().any():
                self.logger.warning("스케일링 전 결측치가 있습니다. 전방향 채우기를 수행합니다.")
                feature_data = feature_data.fillna(method='ffill')
                feature_data = feature_data.fillna(method='bfill')
            
            # 무한대 값 처리
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            feature_data = feature_data.fillna(method='ffill')
            feature_data = feature_data.fillna(method='bfill')
            
            # 각 특성별 스케일링
            scaled_data = np.zeros_like(feature_data.values)
            for i in range(feature_data.shape[1]):
                col_data = feature_data.iloc[:, i].values.reshape(-1, 1)
                if not np.all(np.isnan(col_data)):
                    scaled_data[:, i] = self.scaler.fit_transform(col_data).ravel()
            
            self.logger.info(f"스케일링 후 데이터 형태: {scaled_data.shape}")

            # 시퀀스 데이터 생성
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length, 3])  # close_price의 인덱스

            X = np.array(X)
            y = np.array(y)
            
            self.logger.info(f"최종 데이터 형태: X={X.shape}, y={y.shape}")
            
            if len(X) == 0 or len(y) == 0:
                self.logger.error("생성된 시퀀스 데이터가 없습니다.")
                return np.array([]), np.array([])

            return X, y
            
        except Exception as e:
            self.logger.error(f"데이터 준비 중 오류 발생: {str(e)}")
            return np.array([]), np.array([])

    def train(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """모델 학습 (최적화된 버전)"""
        try:
            self.logger.info(f"모델이 {self.device}에서 실행됩니다.")
            
            # X, y가 제공되지 않으면 데이터 로드
            if X is None or y is None:
                # 데이터 로드
                data = self.load_data()
                self.logger.info(f"데이터 로드 완료: {len(data)} 행")
                
                # 데이터 전처리
                processed_data = self.enhanced_preprocessing(data)
                
                # 학습 데이터 준비
                X, y = self.prepare_data(processed_data)
                self.logger.info(f"데이터 준비 완료: X shape={X.shape}, y shape={y.shape}")
            
            # 모델이 없으면 새로 생성
            if self.model is None:
                self.model = self.build_model((self.sequence_length, X.shape[2]))
            
            # 학습 시작
            self.logger.info("학습 시작...")
            history = self.model.fit(
                X, y,
                epochs=100,
                batch_size=128,
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            # 모델 저장
            self.save_model()
            self.logger.info("학습 완료 및 모델 저장")
            
        except Exception as e:
            self.logger.error(f"학습 중 오류 발생: {str(e)}")
            raise

    def save_model(self):
        """모델 저장"""
        try:
            # 프로젝트 루트 디렉토리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            
            # 모델 저장 디렉토리 설정
            model_dir = os.path.join(project_root, 'models', 'checkpoints')
            backup_dir = os.path.join(project_root, 'models', 'backup')
            
            # 디렉토리 생성
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(backup_dir, exist_ok=True)
            
            # 모델 파일 경로
            model_path = os.path.join(model_dir, f'{self.stock_name}_model.h5')
            backup_path = os.path.join(backup_dir, f'{self.stock_name}_model.h5')
            
            # 모델 저장
            self.model.save(model_path)
            self.model.save(backup_path)
            
            self.logger.info(f"모델 저장 완료: {model_path}")
            self.logger.info(f"모델 백업 저장 완료: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            raise

    def enhanced_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 강화"""
        try:
            # Decimal 타입을 float로 변환
            numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = data[col].astype(float)
            
            # 기본 기술적 지표
            data['MA5'] = data['close_price'].rolling(window=5, min_periods=1).mean()
            data['MA20'] = data['close_price'].rolling(window=20, min_periods=1).mean()
            data['MA60'] = data['close_price'].rolling(window=60, min_periods=1).mean()
            data['MA120'] = data['close_price'].rolling(window=120, min_periods=1).mean()
            
            # 볼린저 밴드
            data['BB_middle'] = data['close_price'].rolling(window=20, min_periods=1).mean()
            data['BB_std'] = data['close_price'].rolling(window=20, min_periods=1).std()
            data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
            data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
            
            # RSI
            delta = data['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close_price'].ewm(span=12, adjust=False).mean()
            exp2 = data['close_price'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
            
            # 스토캐스틱
            low_min = data['low_price'].rolling(window=14, min_periods=1).min()
            high_max = data['high_price'].rolling(window=14, min_periods=1).max()
            data['Stoch_K'] = 100 * ((data['close_price'] - low_min) / (high_max - low_min))
            data['Stoch_D'] = data['Stoch_K'].rolling(window=3, min_periods=1).mean()
            
            # ATR (Average True Range)
            tr1 = data['high_price'] - data['low_price']
            tr2 = abs(data['high_price'] - data['close_price'].shift())
            tr3 = abs(data['low_price'] - data['close_price'].shift())
            data['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            data['ATR'] = data['TR'].rolling(window=14, min_periods=1).mean()
            
            # 거래량 지표
            data['Volume_MA5'] = data['volume'].rolling(window=5, min_periods=1).mean()
            data['Volume_MA20'] = data['volume'].rolling(window=20, min_periods=1).mean()
            data['Volume_Ratio'] = data['volume'] / data['Volume_MA20']
            
            # 가격 변화율
            data['Price_Change'] = data['close_price'].pct_change()
            data['Price_Change_MA5'] = data['Price_Change'].rolling(window=5, min_periods=1).mean()
            data['Price_Change_MA20'] = data['Price_Change'].rolling(window=20, min_periods=1).mean()
            
            # 변동성
            data['Volatility'] = data['close_price'].rolling(window=20, min_periods=1).std()
            data['Volatility_MA5'] = data['Volatility'].rolling(window=5, min_periods=1).mean()
            
            # 모멘텀 지표
            data['ROC'] = data['close_price'].pct_change(periods=10) * 100
            data['Momentum'] = data['close_price'] - data['close_price'].shift(10)
            
            # 추세 강도
            data['ADX'] = self._calculate_adx(data)
            
            # 무한대 값 처리
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # 결측치 처리 - 더 강력한 방법 사용
            # 먼저 전방향 채우기
            data = data.fillna(method='ffill')
            # 후방향 채우기
            data = data.fillna(method='bfill')
            # 남은 결측치는 0으로 채우기
            data = data.fillna(0)
            
            # 최종 검증 - 여전히 NaN이 있는지 확인
            if data.isnull().any().any():
                self.logger.warning("전처리 후에도 NaN 값이 남아있습니다. 추가 처리 중...")
                # 각 컬럼별로 개별 처리
                for col in data.columns:
                    if data[col].isnull().any():
                        if col in ['open_price', 'high_price', 'low_price', 'close_price']:
                            # 가격 데이터는 이전 값으로 채우기
                            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                        else:
                            # 기술적 지표는 0으로 채우기
                            data[col] = data[col].fillna(0)
            
            self.logger.info(f"전처리 완료: 데이터 형태 {data.shape}, NaN 개수: {data.isnull().sum().sum()}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            return pd.DataFrame()
            
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX (Average Directional Index) 계산"""
        try:
            # True Range
            tr1 = data['high_price'] - data['low_price']
            tr2 = abs(data['high_price'] - data['close_price'].shift())
            tr3 = abs(data['low_price'] - data['close_price'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=1).mean()
            
            # Directional Movement
            up_move = data['high_price'] - data['high_price'].shift()
            down_move = data['low_price'].shift() - data['low_price']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period, min_periods=1).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period, min_periods=1).mean() / atr)
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period, min_periods=1).mean()
            
            # NaN 값 처리
            adx = adx.fillna(0)
            
            return adx
            
        except Exception as e:
            self.logger.error(f"ADX 계산 중 오류 발생: {str(e)}")
            return pd.Series([0] * len(data))

if __name__ == "__main__":
    model = HDHyundaiModel()
    model.train_model() 