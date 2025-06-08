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

from models.base.price_predict_model import BasePricePredictModel, setup_gpu, enhanced_weighted_time_mse
from database.database import DatabaseManager
from utils.date_utils import get_next_five_business_days

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

class LGElectronicsModel(BasePricePredictModel):
    def __init__(self):
        super().__init__(
            stock_code='A066570',
            stock_name='LG전자',
            sequence_length=20,
            batch_size=64  # 배치 크기 조정
        )
        self.db_manager = DatabaseManager()
        self.n_features = None  # 특성 수 초기화
        self.models = []  # 앙상블 모델 리스트
        self.num_models = 3  # 앙상블 모델 수
        self.logger = logging.getLogger(__name__)
        
        # GPU 사용 가능 여부 확인
        self.device = tf.config.list_physical_devices('GPU')[0] if tf.config.list_physical_devices('GPU') else 'CPU'
        self.logger.info(f"모델이 {self.device}에서 실행됩니다.")

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
            query = """
                SELECT time, stock_code, stock_name, open_price, high_price, low_price, 
                       close_price, volume, market_cap, foreign_holding, foreign_holding_ratio
                FROM stock_prices
                WHERE stock_code = 'A066570'
                ORDER BY time
            """
            result = self.db_manager.execute_query(query)
            
            if not result:
                raise ValueError("데이터가 비어있습니다.")
            
            # DataFrame 생성 및 컬럼명 설정
            df = pd.DataFrame(result, columns=[
                'time', 'stock_code', 'stock_name', 'open_price', 'high_price', 'low_price',
                'close_price', 'volume', 'market_cap', 'foreign_holding', 'foreign_holding_ratio'
            ])
            
            # 날짜를 인덱스로 설정
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # 컬럼명 변경
            df = df.rename(columns={
                'open_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'close_price': 'close'
            })
            
            self.logger.info(f"데이터 로드 완료: {len(df)} 행")
            return df
            
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

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
        """모델 학습 및 평가"""
        try:
            # 학습 데이터 준비
            X_train, y_train, X_val, y_val, X_test, y_test, _ = self.prepare_training_data()
            
            # 데이터 shape 확인 및 조정
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            
            # 모델 구축
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # 모델 학습
            history = self.train(X_train, y_train, X_val, y_val)
            
            # 모델 평가
            try:
                metrics = self.evaluate(X_test, y_test)
                self.logger.info(f"모델 평가 결과: {metrics}")
            except Exception as e:
                self.logger.warning(f"모델 평가 중 오류 발생: {str(e)}")
                metrics = {}
            
            # 모델 저장
            try:
                save_dir = os.path.join('models', 'checkpoints')
                if os.path.exists(save_dir) and not os.path.isdir(save_dir):
                    os.remove(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                
                model_path = os.path.join(save_dir, f'{self.stock_name}_model.h5')
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    import shutil
                    shutil.rmtree(model_path)
                
                self.model.save(model_path)
                self.logger.info(f"모델이 저장되었습니다: {model_path}")
            except Exception as e:
                self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
                # 백업 저장 시도
                backup_dir = os.path.join('models', 'backup')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, f'{self.stock_name}_model.h5')
                self.model.save(backup_path)
                self.logger.info(f"모델이 백업 위치에 저장되었습니다: {backup_path}")
            
            return history.history
            
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise

    def train(self, X_train, y_train, X_val, y_val) -> tf.keras.callbacks.History:
        """모델 학습"""
        try:
            # 콜백 설정
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,  # patience 증가
                    restore_best_weights=True,
                    min_delta=0.0001  # 최소 개선 기준 추가
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,  # patience 증가
                    min_lr=1e-6
                )
            ]
            
            # 모델 학습
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,  # 최대 에포크 수 증가
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
            
        except Exception as e:
            self.logger.error(f"학습 중 오류 발생: {e}")
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

    def load_models(self):
        """저장된 모델 로드"""
        try:
            # 기본 경로와 백업 경로 모두 확인
            model_paths = [
                os.path.join('models', 'checkpoints', f'{self.stock_name}_model.h5'),
                os.path.join('models', 'backup', f'{self.stock_name}_model.h5')
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    self.model = tf.keras.models.load_model(model_path)
                    self.logger.info(f"모델이 로드되었습니다: {model_path}")
                    return True
            
            self.logger.info("저장된 모델이 없습니다. 모델 학습을 시작합니다...")
            return False
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False
    
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
    
    def build_model(self, input_shape=None):
        """모델 구축"""
        # 입력 shape 설정
        if input_shape is None:
            input_shape = (self.sequence_length, self.n_features)
        
        # 입력 레이어
        inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
        
        # Multi-scale Convolutional Input (MCI)
        conv_outputs = []
        kernel_sizes = [2, 3, 5]  # 커널 크기 감소
        for kernel_size in kernel_sizes:
            conv = tf.keras.layers.Conv1D(64, kernel_size=kernel_size, padding='same', activation='relu', dtype=tf.float32)(inputs)
            conv = tf.keras.layers.BatchNormalization(dtype=tf.float32)(conv)
            conv_outputs.append(conv)
        
        # 컨볼루션 출력 결합
        x = tf.keras.layers.Concatenate()(conv_outputs)
        
        # Attention 메커니즘
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4,  # 헤드 수 감소
            key_dim=64,   # 차원 감소
            dtype=tf.float32
        )(x, x)
        
        # Residual connection
        x = tf.keras.layers.Add()([x, attention_output])
        
        # GRU 레이어
        gru_output = tf.keras.layers.GRU(128, return_sequences=True, dtype=tf.float32)(x)  # 유닛 수 감소
        gru_output = tf.keras.layers.Dropout(0.3)(gru_output)
        
        # 추가 Attention 레이어
        attention_output2 = tf.keras.layers.MultiHeadAttention(
            num_heads=4,  # 헤드 수 감소
            key_dim=128,  # 차원 감소
            dtype=tf.float32
        )(gru_output, gru_output)
        
        # Residual connection
        x = tf.keras.layers.Add()([gru_output, attention_output2])
        
        # 시퀀스의 마지막 타임스텝만 선택
        x = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(x)
        
        # Dense 레이어
        x = tf.keras.layers.Dense(128, activation='relu', dtype=tf.float32)(x)  # 유닛 수 감소
        x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu', dtype=tf.float32)(x)  # 유닛 수 감소
        x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # 출력 레이어 (5% 제한을 위한 tanh 활성화 함수 사용)
        outputs = tf.keras.layers.Dense(1, activation='tanh', dtype=tf.float32)(x) * 0.05
        
        # 모델 생성
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # 컴파일
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        model.compile(
            optimizer=optimizer,
            loss=enhanced_weighted_time_mse,
            metrics=['mae']
        )
        
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
        """다음 5일 주가 예측"""
        try:
            # 모델이 없으면 학습 수행
            if not self.models:
                logger.info("저장된 모델이 없습니다. 모델 학습을 시작합니다...")
                self.train_model()
                logger.info("모델 학습이 완료되었습니다.")
            
            # 다음 5개 영업일 계산
            business_days = get_next_five_business_days(start_date)
            
            # 최근 주가 데이터 로드
            data = self.load_data()
            recent_data = data.tail(self.sequence_length)
            
            # 데이터 전처리
            processed_data = self.enhanced_preprocessing(recent_data)
            
            # 예측 데이터 준비
            X, _ = self.prepare_data(processed_data)
            
            predictions = []
            current_data = X[-1:]
            
            # 각 영업일에 대해 예측 수행
            for target_date in business_days:
                # 앙상블 예측
                ensemble_predictions = []
                for model in self.models:
                    pred = model.predict(current_data, verbose=0)
                    ensemble_predictions.append(pred[0][0])
                
                # 예측값 평균 계산
                predicted_price = np.mean(ensemble_predictions)
                
                # 예측값 역변환
                predicted_price = self.scaler.inverse_transform(
                    np.concatenate([np.zeros((1, 3)), np.array([[predicted_price]]), np.zeros((1, 16))], axis=1)
                )[0, 3]
                
                # 이전 예측값 조회
                prev_predictions = self.get_previous_predictions(start_date, target_date)
                
                # 예측값 조정
                if not prev_predictions.empty:
                    actual_price = self.get_latest_price()
                    prev_predicted_price = prev_predictions.iloc[-1]['predicted_price']
                    adjustment = self.calculate_prediction_adjustment(
                        actual_price, prev_predicted_price, predicted_price
                    )
                    predicted_price += adjustment
                
                predictions.append({
                    'date': target_date,
                    'price': predicted_price
                })
                
                # 다음 예측을 위한 데이터 업데이트
                current_data = np.roll(current_data, -1, axis=1)
                current_data[0, -1] = predicted_price
            
            return predictions
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise
    
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

if __name__ == "__main__":
    model = LGElectronicsModel()
    model.train_model() 