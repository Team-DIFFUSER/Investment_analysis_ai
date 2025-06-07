import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization, Multiply, MultiHeadAttention, Layer, TimeDistributed, Lambda, Conv1D, GRU, RNN
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import random
import math
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
import os
import pickle
import json
import logging
from typing import List, Dict, Tuple, Optional
import time

def setup_gpu():
    """GPU 설정 및 최적화"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            print(f"GPU 사용 가능: {gpus[0]}")
            # Mixed Precision 활성화 (FP16)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed Precision 활성화됨")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    else:
        print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")

def enhanced_weighted_time_mse(y_true, y_pred):
    """시간에 따른 가중치가 적용된 MSE 손실 함수"""
    # 시간에 따른 가중치 계산 (최근 데이터에 더 높은 가중치)
    time_weights = tf.exp(tf.linspace(0., 1., tf.shape(y_true)[1]))
    time_weights = time_weights / tf.reduce_sum(time_weights)
    
    # MSE 계산
    squared_diff = tf.square(y_true - y_pred)
    
    # 가중치 적용
    weighted_squared_diff = squared_diff * time_weights
    
    return tf.reduce_mean(weighted_squared_diff)

# TensorFlow 세션 초기화
import tensorflow as tf

# 기존 세션 정리 및 메모리 해제
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

# 기본 환경 변수 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# TensorFlow 최적화 설정
tf.config.optimizer.set_jit(True)
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
    "auto_mixed_precision": True
})

# 한글 폰트 설정
try:
    font_list = [f.name for f in fm.fontManager.ttflist]
    for font in ['NanumBarunGothic', 'NanumGothic', 'Malgun Gothic', 'Gulim']:
        if font in font_list:
            plt.rcParams['font.family'] = font
            print(f"한글 폰트 '{font}' 사용")
            break
    else:
        print("한글 폰트를 찾을 수 없어 기본 폰트 사용")

    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"폰트 설정 오류: {e}")

# 재현성 설정 강화
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# 모든 랜덤 시드 설정
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

class BasePricePredictModel:
    def __init__(self, stock_code: str, stock_name: str, sequence_length: int = 20, batch_size: int = 128):
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(__name__)

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 추가"""
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()

        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()

        # ROC (Rate of Change)
        roc = ROCIndicator(close=df['close'])
        df['roc'] = roc.roc()

        return df

    def enhanced_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 강화"""
        # 가격 변동률 계산
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        df['price_change_20d'] = df['close'].pct_change(periods=20)

        # 거래량 변동률
        df['volume_change'] = df['volume'].pct_change()
        df['volume_change_5d'] = df['volume'].pct_change(periods=5)

        # 이동평균
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_60'] = df['close'].rolling(window=60).mean()

        # 기술적 지표 추가
        df = self.add_technical_indicators(df)

        # 결측치 처리
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 준비"""
        # 특성 선택
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_high', 'bb_low', 'bb_mid', 'roc',
            'price_change', 'price_change_5d', 'price_change_20d',
            'volume_change', 'volume_change_5d',
            'sma_5', 'sma_20', 'sma_60'
        ]

        # 데이터 정규화
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data[feature_columns])

        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 3])  # close price index

        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """모델 구축"""
        # 입력 레이어
        inputs = Input(shape=input_shape)
        
        # LSTM 레이어
        x = LSTM(128, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = LSTM(64, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # 출력 레이어
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray) -> None:
        """모델 학습"""
        # 콜백 정의
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            ),
            ModelCheckpoint(
                f'models/{self.stock_code}_best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # 모델 학습
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        # 모델 저장
        self.model.save(f"{path}/model.h5")
        
        # 스케일러 저장
        with open(f"{path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self, path: str) -> None:
        """모델 로드"""
        # 모델 로드
        self.model = load_model(f"{path}/model.h5")
        
        # 스케일러 로드
        with open(f"{path}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        if self.model is None:
            raise ValueError("평가할 모델이 없습니다.")
        
        # 예측
        y_pred = self.predict(X_test)
        
        # 평가 지표 계산
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }