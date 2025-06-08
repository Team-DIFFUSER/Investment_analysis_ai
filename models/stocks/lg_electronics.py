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

from models.base.price_predict_model import BasePricePredictModel, setup_gpu, enhanced_weighted_time_mse
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

class LGElectronicsModel(BaseStockModel):
    def __init__(self):
        """LG전자 주가 예측 모델 초기화"""
        super().__init__('LG전자', '066570.KS')
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
        """LG전자 주가 데이터 로드"""
        try:
            # 데이터베이스에서 주가 데이터 가져오기
            data = self.db_manager.get_stock_data('A066570')  # LG전자 종목코드
            
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
        """모델 학습"""
        try:
            # 데이터 로드
            data = self.load_data()
            self.logger.info(f"데이터 로드 완료: {len(data)} 행")
            
            # 데이터 전처리
            processed_data = self.enhanced_preprocessing(data)
            
            # 학습 데이터 준비
            X, y = self.prepare_data(processed_data)
            self.logger.info(f"데이터 준비 완료: X shape={X.shape}, y shape={y.shape}")
            
            # 데이터 분할
            train_size = int(len(X) * 0.8)
            val_size = int(len(X) * 0.1)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            self.logger.info(f"학습 데이터 준비 완료: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            # 모델 생성
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # 콜백 설정
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    min_delta=0.001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.0001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join('models', 'checkpoints', f'{self.stock_name}_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            # 모델 학습
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # 모델 평가
            test_loss = self.model.evaluate(X_test, y_test, verbose=1)
            self.logger.info(f"평가 메트릭: {dict(zip(self.model.metrics_names, test_loss))}")
            
            # 예측 및 R2 계산
            y_pred = self.model.predict(X_test)
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
                stock_code=self.stock_code,
                stock_name=self.stock_name,
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
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            
            # 모델 파일 경로
            model_path = os.path.join(project_root, 'models', 'checkpoints', f'{self.stock_name}_model.h5')
            self.logger.info(f"모델 파일 검색: {model_path}")
            
            if os.path.exists(model_path):
                self.logger.info(f"모델 파일 발견: {model_path}")
                model = tf.keras.models.load_model(model_path)
                self.logger.info("모델 로드 성공")
                return model
            else:
                self.logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
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
        """LG전자 전용 모델 구축"""
        try:
            model = tf.keras.Sequential([
                # 첫 번째 LSTM 레이어
                layers.LSTM(128, input_shape=input_shape, return_sequences=True),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # 두 번째 LSTM 레이어
                layers.LSTM(64, return_sequences=True),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # 세 번째 LSTM 레이어
                layers.LSTM(32, return_sequences=False),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # Dense 레이어
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # 출력 레이어
                layers.Dense(1)
            ])
            
            # 모델 컴파일
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"모델 구축 중 오류 발생: {str(e)}")
            return None
            
    def predict_next_five_days(self) -> List[float]:
        """다음 5일 예측"""
        try:
            # 데이터 로드
            data = self.load_data()
            if data.empty:
                self.logger.error("데이터 로드 실패")
                return []
            
            # 데이터 전처리
            processed_data = self.enhanced_preprocessing(data)
            if processed_data.empty:
                self.logger.error("데이터 전처리 실패")
                return []
            
            # 예측 데이터 준비
            X, _ = self.prepare_data(processed_data)
            if len(X) == 0:
                self.logger.error("예측 데이터 준비 실패")
                return []
            
            # 마지막 시퀀스만 사용
            last_sequence = X[-1:]
            self.logger.info(f"예측 입력 데이터 shape: {last_sequence.shape}")
            
            # 예측
            predictions = self.model.predict(last_sequence, verbose=0)
            
            # 예측값 역정규화
            predictions = self.scaler.inverse_transform(
                np.concatenate([np.zeros((len(predictions), 3)), predictions.reshape(-1, 1), np.zeros((len(predictions), 15))], axis=1)
            )[:, 3]
            
            self.logger.info(f"예측 완료: {predictions}")
            return predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            return []

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        try:
            # 예측 수행
            y_pred = self.model.predict(X_test)
            
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
                stock_code=self.stock_code,
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
            # 특성 선택
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # 기술적 지표 계산
            df = data.copy()
            
            # 가격 변화율
            df['price_change'] = df['Close'].pct_change()
            df['price_change_5d'] = df['Close'].pct_change(periods=5)
            df['price_change_20d'] = df['Close'].pct_change(periods=20)
            
            # 거래량 변화율
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_change_5d'] = df['Volume'].pct_change(periods=5)
            
            # 이동평균선
            df['sma_5'] = df['Close'].rolling(window=5).mean()
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_60'] = df['Close'].rolling(window=60).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_mid'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_high'] = df['bb_mid'] + (bb_std * 2)
            df['bb_low'] = df['bb_mid'] - (bb_std * 2)
            
            # Rate of Change
            df['roc'] = df['Close'].pct_change(periods=10) * 100
            
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 모든 특성 추가
            features.extend([
                'price_change', 'price_change_5d', 'price_change_20d',
                'volume_change', 'volume_change_5d',
                'sma_5', 'sma_20', 'sma_60',
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'bb_high', 'bb_low', 'bb_mid', 'roc'
            ])
            
            # 데이터 스케일링
            scaled_data = self.scaler.fit_transform(df[features])
            
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length, 3])  # Close 가격
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"데이터 준비 중 오류 발생: {str(e)}")
            raise

    def train(self):
        """모델 학습"""
        try:
            # 데이터 로드
            data = self.load_stock_data()
            if data.empty:
                raise ValueError("학습 데이터가 비어있습니다.")
                
            # 데이터 전처리
            X, y = self.prepare_data(data)
            if X is None or y is None:
                raise ValueError("데이터 전처리 실패")
                
            # 모델 생성
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # 모델 학습
            self.model.fit(
                X, y,
                epochs=50,
                batch_size=32,
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
            self._save_model()
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise
            
    def _save_model(self):
        """모델 저장"""
        try:
            # 프로젝트 루트 디렉토리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            
            # 모델 저장 경로
            model_dir = os.path.join(project_root, 'models', 'checkpoints')
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f'{self.stock_name}_model.h5')
            self.model.save(model_path)
            self.logger.info(f"모델 저장 완료: {model_path}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            raise

    def is_initialized(self) -> bool:
        """모델 초기화 상태 확인"""
        return self._initialized and self.model is not None
        
    def initialize(self):
        """모델 초기화"""
        try:
            # GPU 메모리 설정
            self._setup_gpu()
            
            # 모델 로드
            self.model = self.load_model()
            if self.model is None:
                self.logger.warning("저장된 모델이 없습니다. 모델 학습을 시작합니다...")
                self.train()
                self.model = self.load_model()
                if self.model is None:
                    raise ValueError("모델 학습 후에도 모델을 로드할 수 없습니다.")
            
            self.logger.info(f"모델이 {self.device}에서 실행됩니다.")
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
            self._initialized = False
            raise

if __name__ == "__main__":
    model = LGElectronicsModel()
    model.train_model() 