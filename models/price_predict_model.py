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
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, Nadam
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
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import logging
import psycopg2
from psycopg2.extras import execute_values
from kaggle_secrets import UserSecretsClient
import networkx as nx
from scipy import stats
from typing import List, Dict, Tuple, Optional
import time
import sys
import argparse

# TensorFlow 세션 초기화
import tensorflow as tf

# GPU 설정 단순화
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

print("TensorFlow 버전:", tf.__version__)

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

# 배치 크기 증가 (GPU 메모리에 맞게 조정)
BATCH_SIZE = 128  # 32에서 128로 증가

# 데이터베이스 연결 설정
user_secrets = UserSecretsClient()
DB_HOST = user_secrets.get_secret("DB_HOST")
DB_PORT = user_secrets.get_secret("DB_PORT")
DB_NAME = user_secrets.get_secret("DB_NAME")
DB_USER = user_secrets.get_secret("DB_USER")
DB_PASSWORD = user_secrets.get_secret("DB_PASSWORD")

def get_db_connection():
    """데이터베이스 연결 함수"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        return None

def execute_query(query, params=None, fetch=True):
    """쿼리 실행 함수"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("데이터베이스 연결 실패")
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            conn.commit()
    except Exception as e:
        print(f"쿼리 실행 오류: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def execute_values_query(query, data):
    """여러 행의 데이터를 한 번에 삽입하는 함수"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("데이터베이스 연결 실패")
        
        with conn.cursor() as cur:
            execute_values(cur, query, data)
            conn.commit()
    except Exception as e:
        print(f"데이터 삽입 오류: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def execute_transaction(queries):
    """트랜잭션 실행 함수"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("데이터베이스 연결 실패")
        
        with conn.cursor() as cur:
            for query, params in queries:
                if params is None:
                    cur.execute(query)
                else:
                    cur.execute(query, params)
            conn.commit()
    except Exception as e:
        print(f"트랜잭션 실행 오류: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def create_predictions_table():
    """예측 결과를 저장할 테이블 생성"""
    queries = [
        ("""
        CREATE TABLE IF NOT EXISTS predicted_stock_prices (
            id SERIAL PRIMARY KEY,
            stock_code VARCHAR(10) NOT NULL,
            stock_name VARCHAR(50) NOT NULL,
            prediction_date TIMESTAMPTZ NOT NULL,
            target_date TIMESTAMPTZ NOT NULL,
            predicted_price DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """, None),
        ("CREATE INDEX IF NOT EXISTS idx_predicted_prices_date ON predicted_stock_prices (prediction_date, target_date);", None),
        ("CREATE INDEX IF NOT EXISTS idx_predicted_prices_stock ON predicted_stock_prices (stock_code);", None)
    ]
    execute_transaction(queries)
    print("Predicted stock prices table created successfully!")

def create_economic_indicators_table():
    """경제지표 테이블 생성"""
    queries = [
        ("""
        CREATE TABLE IF NOT EXISTS economic_indicators (
            id SERIAL PRIMARY KEY,
            time TIMESTAMPTZ NOT NULL,
            treasury_10y DECIMAL(10,2),
            dollar_index DECIMAL(10,2),
            usd_krw DECIMAL(10,2),
            korean_bond_10y DECIMAL(10,2),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """, None),
        ("CREATE INDEX IF NOT EXISTS idx_economic_indicators_time ON economic_indicators (time);", None)
    ]
    execute_transaction(queries)
    print("Economic indicators table created successfully!")

def save_prediction(stock_code, stock_name, prediction_date, target_date, predicted_price):
    """예측 결과를 데이터베이스에 저장"""
    # NumPy 타입을 Python 네이티브 타입으로 변환
    predicted_price = float(predicted_price)
    
    query = """
    INSERT INTO predicted_stock_prices (
        stock_code, stock_name, prediction_date, target_date,
        predicted_price
    ) VALUES (%s, %s, %s, %s, %s)
    """
    params = (
        stock_code, stock_name, prediction_date, target_date,
        predicted_price
    )
    execute_query(query, params, fetch=False)

def load_data_from_db():
    """데이터베이스에서 데이터 로드"""
    print("Loading stock data...")
    try:
        # 테이블 생성 확인
        create_predictions_table()
        create_economic_indicators_table()
        
        # 데이터베이스에서 주가 데이터 로드
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
        WHERE stock_name = 'LG전자'
        ORDER BY time;
        """
        stock_data = pd.DataFrame(execute_query(query), columns=[
            '기준일자', '종목코드', '종목명', '시가', '고가', '저가', 
            '현재가', '거래량', '시가총액', '외국인보유', '외국인비율'
        ])
        
        # 숫자형 컬럼을 float로 변환
        numeric_columns = ['시가', '고가', '저가', '현재가', '거래량', '시가총액', '외국인보유', '외국인비율']
        for col in numeric_columns:
            stock_data[col] = stock_data[col].astype(float)
        
        print("Stock data columns:", stock_data.columns.tolist())
        print("Stock data shape:", stock_data.shape)
        print("Stock data head:\n", stock_data.head())
        
        # 감성 데이터 로드
        query = """
        SELECT 
            pub_date, title,
            finbert_positive, finbert_negative, finbert_neutral,
            finbert_sentiment
        FROM news_sentiment
        ORDER BY pub_date;
        """
        sentiment_data = pd.DataFrame(execute_query(query), columns=[
            'PubDate', 'Title', 'finbert_positive', 'finbert_negative', 
            'finbert_neutral', 'finbert_sentiment'
        ])
        
        # 감성 점수를 float로 변환 (finbert_sentiment 제외)
        sentiment_columns = ['finbert_positive', 'finbert_negative', 'finbert_neutral']
        for col in sentiment_columns:
            sentiment_data[col] = pd.to_numeric(sentiment_data[col], errors='coerce')
        
        # finbert_sentiment를 숫자로 매핑
        sentiment_mapping = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
        sentiment_data['finbert_sentiment'] = sentiment_data['finbert_sentiment'].map(sentiment_mapping)
        
        print("\nSentiment data columns:", sentiment_data.columns.tolist())
        print("Sentiment data shape:", sentiment_data.shape)
        print("Sentiment data head:\n", sentiment_data.head())
        
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
        economic_data = pd.DataFrame(execute_query(query), columns=[
            'time', 'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
        ])
        
        # 경제지표를 float로 변환
        economic_columns = ['treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y']
        for col in economic_columns:
            economic_data[col] = pd.to_numeric(economic_data[col], errors='coerce')
            
        economic_data.set_index('time', inplace=True)
        
        print("\nEconomic data shape:", economic_data.shape)
        print("Economic data head:\n", economic_data.head())
        
        return stock_data, sentiment_data, economic_data
        
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        raise

# 1. 데이터 로드 및 전처리
print("Loading stock data...")
try:
    # load_data_from_db() 함수를 사용하여 모든 데이터를 한 번에 로드
    stock_data, sentiment_data, economic_data = load_data_from_db()
    
    print("Stock data columns:", stock_data.columns.tolist())
    print("Stock data shape:", stock_data.shape)
    print("Stock data head:\n", stock_data.head())
    
    print("\nSentiment data columns:", sentiment_data.columns.tolist())
    print("Sentiment data shape:", sentiment_data.shape)
    print("Sentiment data head:\n", sentiment_data.head())
    
    print("\nEconomic indicators data columns:", economic_data.columns.tolist())
    print("Economic indicators data shape:", economic_data.shape)
    print("Economic indicators data head:\n", economic_data.head())
    
except Exception as e:
    print(f"데이터 로드 실패: {e}")
    raise

# LG전자 데이터만 필터링
lg_data = stock_data[stock_data['종목명'] == 'LG전자'].copy()
print("\nLG data shape:", lg_data.shape)
print("LG data head:\n", lg_data.head())

# 날짜 형식 변환
lg_data['기준일자'] = pd.to_datetime(lg_data['기준일자'])
sentiment_data['PubDate'] = pd.to_datetime(sentiment_data['PubDate'])
economic_data.index = pd.to_datetime(economic_data.index)

# 데이터 병합
merged_data = pd.merge(lg_data, sentiment_data, left_on='기준일자', right_on='PubDate', how='left')
merged_data = pd.merge(merged_data, economic_data, left_on='기준일자', right_index=True, how='left')
print("\nMerged data shape:", merged_data.shape)

# 기술적 지표 추가
def add_technical_indicators(df):
    # RSI
    rsi = RSIIndicator(close=df['현재가'], window=14)
    df['RSI'] = rsi.rsi()

    # MACD
    macd = MACD(close=df['현재가'])
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()
    df['MACD_HIST'] = macd.macd_diff()

    # 볼린저 밴드
    bbands = BollingerBands(close=df['현재가'], window=20)
    df['BB_UPPER'] = bbands.bollinger_hband()
    df['BB_MIDDLE'] = bbands.bollinger_mavg()
    df['BB_LOWER'] = bbands.bollinger_lband()
    df['BB_PERCENT'] = (df['현재가'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])

    # 이동평균
    df['MA5'] = SMAIndicator(close=df['현재가'], window=5).sma_indicator()
    df['MA20'] = SMAIndicator(close=df['현재가'], window=20).sma_indicator()
    df['MA60'] = SMAIndicator(close=df['현재가'], window=60).sma_indicator()

    # 거래량 지표
    df['VOLUME_MA5'] = SMAIndicator(close=df['거래량'], window=5).sma_indicator()
    df['VOLUME_MA20'] = SMAIndicator(close=df['거래량'], window=20).sma_indicator()
    df['VOLUME_RATIO'] = df['거래량'] / df['VOLUME_MA20']

    # 모멘텀 지표
    df['MOM'] = df['현재가'].diff(10)
    df['ROC'] = ROCIndicator(close=df['현재가'], window=10).roc()

    return df

# 기술적 지표 추가
merged_data = add_technical_indicators(merged_data)

# 결측치 처리
merged_data = merged_data.ffill().bfill().fillna(0)

# 데이터 전처리 개선
def enhanced_preprocessing(df):
    # 가격 변동률 계산 (더 긴 기간의 추세 반영)
    df['price_change_5d'] = df['현재가'].pct_change(5)
    df['price_change_10d'] = df['현재가'].pct_change(10)
    
    # 이동평균 기반 변동성
    df['volatility_5d'] = df['현재가'].rolling(window=5).std() / df['현재가'].rolling(window=5).mean()
    df['volatility_10d'] = df['현재가'].rolling(window=10).std() / df['현재가'].rolling(window=10).mean()
    
    # 거래량 가중 가격
    df['vwap'] = (df['현재가'] * df['거래량']).rolling(window=5).sum() / df['거래량'].rolling(window=5).sum()
    
    # 가격 모멘텀 (여러 기간)
    for window in [5, 10, 20]:
        df[f'momentum_{window}d'] = df['현재가'] / df['현재가'].rolling(window=window).mean() - 1
    
    # 이상치 처리 (IQR 방법 강화)
    for col in ['현재가', '거래량', 'price_change_5d', 'price_change_10d']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2.0 * iqr  # 더 엄격한 기준
        upper_bound = q3 + 2.0 * iqr
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 결측치 처리 (최신 pandas 방식)
    df = df.ffill().bfill()
    
    # 감성 데이터 보간
    sentiment_cols = ['finbert_positive', 'finbert_negative', 'finbert_neutral']
    for col in sentiment_cols:
        if col in df.columns:
            # 감성 데이터가 있는 경우에만 보간
            mask = df[col] != 0
            if mask.any():
                df[col] = df[col].interpolate(method='linear')
    
    # 경제 지표 보간
    economic_cols = ['treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y']
    for col in economic_cols:
        if col in df.columns:
            # 경제 지표가 있는 경우에만 보간
            mask = df[col] != 0
            if mask.any():
                df[col] = df[col].interpolate(method='linear')
    
    return df

# 데이터 전처리 적용
merged_data = enhanced_preprocessing(merged_data)

# 스케일링 클래스 개선
class EnhancedPriceScaler:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.price_min = None
        self.price_max = None
        self.feature_names = []  # 빈 리스트로 초기화

    def fit_transform(self, data, price_cols):
        data_copy = data.copy()
        
        # 문자열 컬럼과 날짜 컬럼 제외
        exclude_cols = ['기준일자', '종목코드', '종목명', 'Title', 'PubDate', 'finbert_sentiment']
        for col in exclude_cols:
            if col in data_copy.columns:
                data_copy = data_copy.drop(columns=[col])
        
        # 가격 데이터와 다른 특성 분리
        price_data = data_copy[price_cols]
        other_data = data_copy.drop(columns=price_cols)
        
        # 특성 이름 저장
        self.feature_names = other_data.columns.tolist()
        
        # 가격 데이터의 최소/최대값 저장
        self.price_min = price_data.min().values
        self.price_max = price_data.max().values
        
        # 각각 스케일링
        scaled_price = self.price_scaler.fit_transform(price_data)
        scaled_other = self.feature_scaler.fit_transform(other_data)
        
        # 스케일링된 데이터 결합
        scaled_data = np.concatenate([scaled_price, scaled_other], axis=1)
        scaled_df = pd.DataFrame(scaled_data, columns=price_cols + self.feature_names)
        
        # 원래 컬럼 순서 복원
        scaled_df = scaled_df[data_copy.columns]
        
        return scaled_df

    def inverse_transform_price(self, scaled_price):
        if len(scaled_price.shape) == 1:
            scaled_price = scaled_price.reshape(-1, 1)
        
        # MinMaxScaler 역변환
        unscaled = self.price_scaler.inverse_transform(scaled_price)
        return unscaled.flatten()

    def __getstate__(self):
        """직렬화할 때 호출되는 메서드"""
        state = self.__dict__.copy()
        # MinMaxScaler 객체를 직렬화 가능한 형태로 변환
        state['price_scaler'] = {
            'scale_': self.price_scaler.scale_,
            'min_': self.price_scaler.min_,
            'data_min_': self.price_scaler.data_min_,
            'data_max_': self.price_scaler.data_max_,
            'feature_names_in_': self.price_scaler.feature_names_in_
        }
        state['feature_scaler'] = {
            'scale_': self.feature_scaler.scale_,
            'min_': self.feature_scaler.min_,
            'data_min_': self.feature_scaler.data_min_,
            'data_max_': self.feature_scaler.data_max_,
            'feature_names_in_': self.feature_scaler.feature_names_in_
        }
        return state
    
    def __setstate__(self, state):
        """역직렬화할 때 호출되는 메서드"""
        self.__dict__.update(state)
        # MinMaxScaler 객체 복원
        self.price_scaler = MinMaxScaler()
        self.price_scaler.scale_ = state['price_scaler']['scale_']
        self.price_scaler.min_ = state['price_scaler']['min_']
        self.price_scaler.data_min_ = state['price_scaler']['data_min_']
        self.price_scaler.data_max_ = state['price_scaler']['data_max_']
        self.price_scaler.feature_names_in_ = state['price_scaler']['feature_names_in_']
        
        self.feature_scaler = MinMaxScaler()
        self.feature_scaler.scale_ = state['feature_scaler']['scale_']
        self.feature_scaler.min_ = state['feature_scaler']['min_']
        self.feature_scaler.data_min_ = state['feature_scaler']['data_min_']
        self.feature_scaler.data_max_ = state['feature_scaler']['data_max_']
        self.feature_scaler.feature_names_in_ = state['feature_scaler']['feature_names_in_']

def save_model_and_scaler(model, scaler, model_index):
    """모델과 스케일러를 저장하는 함수"""
    try:
        # 모델 저장
        model_path = f'stock_prediction_model_{model_index}.keras'
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # NumPy 배열을 Python 기본 타입으로 변환하는 함수
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # 스케일러 상태 저장
        scaler_state = {
            'price_scaler': {
                'scale_': convert_to_serializable(scaler.price_scaler.scale_),
                'min_': convert_to_serializable(scaler.price_scaler.min_),
                'data_min_': convert_to_serializable(scaler.price_scaler.data_min_),
                'data_max_': convert_to_serializable(scaler.price_scaler.data_max_),
                'feature_names_in_': convert_to_serializable(scaler.price_scaler.feature_names_in_)
            },
            'feature_scaler': {
                'scale_': convert_to_serializable(scaler.feature_scaler.scale_),
                'min_': convert_to_serializable(scaler.feature_scaler.min_),
                'data_min_': convert_to_serializable(scaler.feature_scaler.data_min_),
                'data_max_': convert_to_serializable(scaler.feature_scaler.data_max_),
                'feature_names_in_': convert_to_serializable(scaler.feature_scaler.feature_names_in_)
            },
            'price_min': convert_to_serializable(scaler.price_min),
            'price_max': convert_to_serializable(scaler.price_max),
            'feature_names': convert_to_serializable(scaler.feature_names if hasattr(scaler, 'feature_names') else [])
        }
        
        scaler_path = f'stock_prediction_scaler_{model_index}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_state, f)
        print(f"Scaler state saved to {scaler_path}")
        
        # 모델 메타데이터 저장
        metadata = {
            'input_shape': convert_to_serializable(model.input_shape[1:]),
            'output_shape': convert_to_serializable(model.output_shape[1:]),
            'scaler_params': scaler_state
        }
        
        metadata_path = f'model_metadata_{model_index}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"Error saving model and scaler: {e}")
        raise

# 데이터 전처리 개선
def prepare_data(merged_data, sequence_length=20):
    # 특성 선택 최적화
    feature_columns = [
        '현재가', '거래량', '시가총액', '외국인보유', '외국인비율',
        'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
        'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_PERCENT',
        'MA5', 'MA20', 'MA60', 'VOLUME_MA5', 'VOLUME_MA20',
        'VOLUME_RATIO', 'MOM', 'ROC',
        'finbert_positive', 'finbert_negative', 'finbert_neutral',
        'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
    ]
    
    # 데이터 정규화
    scaler = EnhancedPriceScaler()
    price_cols = ['현재가']
    scaled_data = scaler.fit_transform(merged_data[feature_columns], price_cols)
    
    # DataFrame을 numpy array로 변환
    scaled_data = scaled_data.values
    
    # 시퀀스 데이터 생성 (슬라이딩 윈도우 방식)
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - 4):
        # 입력 시퀀스
        X.append(scaled_data[i:(i + sequence_length)])
        
        # 타겟 시퀀스 (마지막 가격을 기준으로 상대적 변화율로 변환)
        last_price = scaled_data[i + sequence_length - 1, 0]  # 마지막 가격
        target_prices = scaled_data[i + sequence_length:i + sequence_length + 5, 0]
        relative_changes = (target_prices - last_price) / last_price
        y.append(relative_changes)
    
    X = np.array(X)
    y = np.array(y)
    
    # 학습/검증/테스트 데이터 분할 (80/10/10)
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print("\nData shapes after preparation:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# 데이터 증강 함수 정의
def augment_data(X, y, noise_level=0.0001):
    """
    학습 데이터에 노이즈를 추가하여 데이터 증강을 수행하는 함수
    
    Args:
        X: 입력 데이터 (numpy array)
        y: 타겟 데이터 (numpy array)
        noise_level: 추가할 노이즈의 수준 (기본값: 0.0001)
    
    Returns:
        X_aug: 증강된 입력 데이터
        y_aug: 증강된 타겟 데이터
    """
    # 원본 데이터 복사
    X_aug = X.copy()
    y_aug = y.copy()
    
    # 가우시안 노이즈 생성
    noise = np.random.normal(0, noise_level, X.shape)
    
    # 노이즈 적용 (첫 번째 특성(가격)에는 더 작은 노이즈 적용)
    noise[:, :, 0] *= 0.1  # 가격 특성에 대한 노이즈 감소
    
    # 증강된 데이터 생성
    X_aug = X_aug + noise
    
    # 타겟 데이터에도 작은 노이즈 추가
    y_noise = np.random.normal(0, noise_level * 0.1, y.shape)
    y_aug = y_aug + y_noise
    
    # 음수 값 방지
    X_aug = np.maximum(X_aug, 0)
    y_aug = np.maximum(y_aug, 0)
    
    return X_aug, y_aug

# 손실 함수 완전 재구성
@tf.keras.utils.register_keras_serializable()
def enhanced_weighted_time_mse(y_true, y_pred):
    # 수치적 안정성을 위한 작은 값 추가
    epsilon = 1e-7
    
    # 입력 shape 확인 및 조정
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 5% 변동성 제한 적용
    y_pred = tf.clip_by_value(y_pred, -0.05, 0.05)
    
    # 시간별 가중치 조정 (장기 예측의 정확도 향상)
    time_weights = tf.constant([0.25, 0.2, 0.2, 0.2, 0.15], dtype=tf.float32)
    
    # 기본 MSE 손실
    base_loss = tf.reduce_mean(tf.square(y_true - y_pred) * time_weights)
    
    # 첫날 예측에 대한 중간 패널티
    first_day_penalty = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])) * 35.0
    
    # 과대 예측에 대한 패널티 (상승 예측에 더 큰 패널티)
    overprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_pred - y_true) * tf.constant([250.0, 220.0, 200.0, 180.0, 160.0], dtype=tf.float32)
    )
    
    # 과소 예측에 대한 패널티 (하락 예측에 더 큰 패널티)
    underprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_true - y_pred) * tf.constant([250.0, 220.0, 200.0, 180.0, 160.0], dtype=tf.float32)
    )
    
    # 추세 손실 (하락 추세에 더 민감하게)
    y_true_diff = y_true[:, 1:] - y_true[:, :-1]
    y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    trend_weights = tf.constant([0.3, 0.25, 0.25, 0.2], dtype=tf.float32)
    trend_loss = tf.reduce_mean(tf.square(y_true_diff - y_pred_diff) * trend_weights * 30.0 + epsilon)
    
    # 방향성 손실 (하락 방향에 더 민감하게)
    direction_weights = tf.constant([0.3, 0.25, 0.25, 0.2], dtype=tf.float32)
    direction_loss = tf.reduce_mean(
        tf.square(tf.sign(y_true_diff) - tf.sign(y_pred_diff)) * direction_weights * 40.0 + epsilon
    )
    
    # 연속성 손실 (부드러운 변화에 중점)
    continuity_weights = tf.constant([0.3, 0.25, 0.25, 0.2], dtype=tf.float32)
    continuity_loss = tf.reduce_mean(
        tf.square(y_pred[:, 1:] - y_pred[:, :-1] - (y_true[:, 1:] - y_true[:, :-1])) * continuity_weights * 25.0
    )
    
    # 장기 예측 정확도 향상을 위한 추가 손실
    long_term_loss = tf.reduce_mean(tf.square(y_true[:, -1] - y_pred[:, -1])) * 45.0
    
    # 가중치 적용
    weighted_loss = (
        base_loss +
        0.8 * first_day_penalty +
        1.1 * overprediction_penalty +
        1.1 * underprediction_penalty +
        0.8 * trend_loss +
        0.9 * direction_loss +
        0.7 * continuity_loss +
        1.2 * long_term_loss  # 장기 예측 정확도 향상
    )
    return weighted_loss

# 마지막 가격 정보를 추출하는 커스텀 레이어
class LastPriceExtractor(Layer):
    def __init__(self, **kwargs):
        super(LastPriceExtractor, self).__init__(**kwargs)
        
    def call(self, inputs):
        # 입력의 마지막 시점의 가격 정보만 추출
        return inputs[:, -1, 0:1]  # 첫 번째 특성(가격)만 선택
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

# Market Context Layer 정의
class MarketContextLayer(Layer):
    def __init__(self, feature_dim, **kwargs):
        super(MarketContextLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.attention = MultiHeadAttention(
            num_heads=4,
            key_dim=feature_dim
        )
        
    def call(self, inputs):
        # 시장 맥락을 반영하는 어텐션 메커니즘 적용
        attention_output = self.attention(inputs, inputs)
        return attention_output
    
    def compute_output_shape(self, input_shape):
        return input_shape

# 개선된 GRU with Attention
class AttentionGRU(Layer):
    def __init__(self, units, **kwargs):
        super(AttentionGRU, self).__init__(**kwargs)
        self.units = units
        self.gru = GRU(units, return_sequences=True)
        self.attention = MultiHeadAttention(num_heads=4, key_dim=units)
        self.add = tf.keras.layers.Add()
        
    def build(self, input_shape):
        self.gru.build(input_shape)
        self.attention.build(
            query_shape=(input_shape[0], input_shape[1], self.units),
            key_shape=(input_shape[0], input_shape[1], self.units),
            value_shape=(input_shape[0], input_shape[1], self.units)
        )
        super(AttentionGRU, self).build(input_shape)
        
    def call(self, inputs, training=None):
        # GRU 처리
        gru_output = self.gru(inputs, training=training)
        
        # 주의 메커니즘 적용
        attention_output = self.attention(
            query=gru_output,
            key=gru_output,
            value=gru_output,
            training=training
        )
        
        # 잔차 연결
        output = self.add([gru_output, attention_output])
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

# MCI-GRU 모델 구조
def build_enhanced_model(input_shape, output_days=5):
    # 입력 레이어
    inputs = Input(shape=input_shape)
    
    # Multi-scale Convolutional Input (MCI)
    conv_outputs = []
    kernel_sizes = [2, 3, 5, 7, 11]  # 더 다양한 시간 스케일
    for kernel_size in kernel_sizes:
        conv = Conv1D(128, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
        conv = BatchNormalization()(conv)
        conv_outputs.append(conv)
    
    # 컨볼루션 출력 결합
    x = Concatenate()(conv_outputs)
    
    # Attention 메커니즘 강화
    attention_output = MultiHeadAttention(
        num_heads=8,
        key_dim=128
    )(x, x)
    
    # Residual connection
    x = tf.keras.layers.Add()([x, attention_output])
    
    # GRU 레이어 강화
    gru_output = GRU(256, return_sequences=True)(x)
    gru_output = Dropout(0.3)(gru_output)
    
    # 추가 Attention 레이어
    attention_output2 = MultiHeadAttention(
        num_heads=8,
        key_dim=256
    )(gru_output, gru_output)
    
    # Residual connection
    x = tf.keras.layers.Add()([gru_output, attention_output2])
    
    # 시퀀스의 마지막 타임스텝만 선택
    x = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(x)
    
    # Dense 레이어
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 출력 레이어 (5% 제한을 위한 tanh 활성화 함수 사용)
    outputs = Dense(output_days, activation='tanh')(x) * 0.05  # tanh의 출력 범위를 -0.05에서 0.05로 조정
    
    # 모델 생성
    model = Model(inputs=inputs, outputs=outputs)
    
    # 컴파일
    optimizer = Adam(learning_rate=0.0003)  # 더 안정적인 학습을 위해 학습률 감소
    model.compile(
        optimizer=optimizer,
        loss=enhanced_weighted_time_mse,
        metrics=['mae'],
        jit_compile=True
    )
    
    return model

# 예측 결과를 원래 가격으로 변환하는 함수
def convert_predictions_to_prices(predictions, last_price):
    """상대적 변화율 예측을 실제 가격으로 변환"""
    # 예측값이 이미 -0.05에서 0.05 사이로 제한되어 있음
    predicted_prices = []
    current_price = last_price
    
    for pred in predictions:
        # 예측된 변화율을 적용
        next_price = current_price * (1 + pred)
        # 100원 단위로 반올림
        next_price = round(next_price / 100) * 100
        predicted_prices.append(next_price)
        current_price = next_price
    
    return np.array(predicted_prices)

# 학습 과정 개선
def train_enhanced_model(model, X_train, y_train, X_val, y_val):
    # 콜백 정의
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=120,  # 인내심 조정
            restore_best_weights=True,
            min_delta=0.0001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.15,
            patience=25,  # 인내심 조정
            min_lr=1e-7,
            min_delta=0.0001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # 데이터 증강 적용 (노이즈 레벨 조정)
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_level=0.00003)
    
    # 데이터셋 최적화
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_aug, y_train_aug))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=50000)
    train_dataset = train_dataset.batch(40)  # 배치 크기 조정
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(40)  # 배치 크기 조정
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # 학습
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=450,  # 에포크 수 조정
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# 앙상블 모델 클래스 정의
class EnsembleModel:
    def __init__(self, input_shape, num_models=3):
        self.input_shape = input_shape
        self.num_models = num_models
        self.models = []
        
    def build_models(self):
        """여러 개의 모델을 생성하고 컴파일"""
        for i in range(self.num_models):
            model = build_enhanced_model(self.input_shape)
            self.models.append(model)
    
    def train(self, X_train, y_train, X_val, y_val, scaler):
        """각 모델을 독립적으로 학습"""
        histories = []
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.num_models}")
            history = train_enhanced_model(model, X_train, y_train, X_val, y_val)
            histories.append(history)
            
            # 모델과 스케일러 저장
            save_model_and_scaler(model, scaler, i+1)
            print(f"Model {i+1} and scaler saved successfully")
        
        return histories
    
    def predict(self, X):
        """모든 모델의 예측을 평균하여 최종 예측값 생성"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.mean(predictions, axis=0)

# 데이터 전처리 및 준비
print("Preparing data for training...")
X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(merged_data)

print("\nData shapes after preparation:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

# 앙상블 모델 사용
ensemble = EnsembleModel(input_shape=(X_train.shape[1], X_train.shape[2]))
ensemble.build_models()
histories = ensemble.train(X_train, y_train, X_val, y_val, scaler)

# 예측 수행
predictions = ensemble.predict(X_test)

# 학습 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(histories[0].history['loss'], label='Training Loss')
plt.plot(histories[0].history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(histories[0].history['mae'], label='Training MAE')
plt.plot(histories[0].history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

def get_latest_stock_price(stock_name):
    """데이터베이스에서 가장 최근 주가를 가져오는 함수"""
    query = """
    SELECT close_price, time
    FROM stock_prices
    WHERE stock_name = %s
    ORDER BY time DESC
    LIMIT 1
    """
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("데이터베이스 연결 실패")
        
        with conn.cursor() as cur:
            cur.execute(query, (stock_name,))
            result = cur.fetchone()
            
            if result:
                price, time = result
                print(f"조회된 주가: {price:,.0f}원 (기준일: {time})")
                return float(price)
            else:
                raise Exception(f"{stock_name} 종목의 주가 데이터를 찾을 수 없습니다.")
                
    except Exception as e:
        print(f"최근 주가 조회 중 오류 발생: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_previous_predictions(stock_name: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """이전 예측값 조회"""
    query = """
    SELECT target_date, predicted_price
    FROM predicted_stock_prices
    WHERE stock_name = %s
    AND target_date BETWEEN %s AND %s
    ORDER BY target_date
    """
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("데이터베이스 연결 실패")
        
        with conn.cursor() as cur:
            cur.execute(query, (stock_name, start_date, end_date))
            results = cur.fetchall()
            
            if results:
                return pd.DataFrame(results, columns=['target_date', 'predicted_price'])
            return pd.DataFrame(columns=['target_date', 'predicted_price'])
                
    except Exception as e:
        print(f"이전 예측값 조회 중 오류 발생: {e}")
        return pd.DataFrame(columns=['target_date', 'predicted_price'])
    finally:
        if conn:
            conn.close()

def calculate_prediction_adjustment(actual_price: float, predicted_price: float, next_predicted_price: float) -> float:
    """예측값 조정 계산"""
    # 실제값과 예측값의 차이
    price_diff = actual_price - predicted_price
    
    # 다음 예측값에 대한 조정
    # 차이의 30%만 반영하여 급격한 변화 방지
    adjustment = price_diff * 0.3
    
    # 조정된 예측값이 이전 예측값과 너무 크게 차이나지 않도록 제한
    max_change = next_predicted_price * 0.02  # 최대 2% 변화 허용
    adjustment = np.clip(adjustment, -max_change, max_change)
    
    return adjustment

def predict_next_five_days(stock_name: str, current_date: datetime, last_actual_price: float) -> List[Dict]:
    """다음 5일 예측"""
    try:
        # 1. 이전 예측값 조회
        end_date = current_date + timedelta(days=4)
        previous_predictions = get_previous_predictions(stock_name, current_date, end_date)
        
        # 2. 새로운 데이터로 예측
        # 최근 20일 데이터를 가져와서 예측에 사용
        start_date = current_date - timedelta(days=20)
        query = """
        SELECT time, close_price, volume, market_cap, foreign_holding, foreign_holding_ratio
        FROM stock_prices
        WHERE stock_name = %s
        AND time BETWEEN %s AND %s
        ORDER BY time
        """
        recent_data = pd.DataFrame(execute_query(query, (stock_name, start_date, current_date)),
                                 columns=['time', 'close_price', 'volume', 'market_cap', 
                                         'foreign_holding', 'foreign_holding_ratio'])
        
        if len(recent_data) < 20:
            raise Exception("예측을 위한 충분한 데이터가 없습니다. 최소 20일의 데이터가 필요합니다.")
        
        # 데이터 전처리
        recent_data = add_technical_indicators(recent_data)
        recent_data = enhanced_preprocessing(recent_data)
        
        # 스케일링
        scaled_data = scaler.fit_transform(recent_data, ['close_price'])
        
        # 시퀀스 데이터 생성
        X = []
        for i in range(len(scaled_data) - 20):
            X.append(scaled_data[i:(i + 20)])
        X = np.array(X)
        
        # 예측 수행
        predictions = ensemble.predict(X)
        new_predictions = predictions[-1]  # 마지막 예측값
        
        # 3. 예측값 조정
        adjusted_predictions = []
        for i in range(5):
            target_date = current_date + timedelta(days=i)
            
            if i == 0:
                # 첫날은 실제값 사용
                predicted_price = last_actual_price
            else:
                # 이전 예측값이 있는 경우 조정
                if not previous_predictions.empty and i < len(previous_predictions):
                    prev_pred = previous_predictions.iloc[i-1]['predicted_price']
                    next_pred = new_predictions[i]
                    adjustment = calculate_prediction_adjustment(
                        last_actual_price, prev_pred, next_pred
                    )
                    predicted_price = next_pred + adjustment
                else:
                    predicted_price = new_predictions[i]
            
            # 100원 단위로 반올림
            predicted_price = round(predicted_price / 100) * 100
            
            adjusted_predictions.append({
                'date': target_date,
                'price': predicted_price
            })
        
        return adjusted_predictions
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        raise

def evaluate_predictions(stock_name: str, start_date: datetime, end_date: datetime) -> Dict:
    """실제 가격과 예측값 비교 평가"""
    try:
        # 실제 가격 데이터 조회
        query = """
        SELECT time, close_price
        FROM stock_prices
        WHERE stock_name = %s
        AND time BETWEEN %s AND %s
        ORDER BY time
        """
        actual_data = pd.DataFrame(execute_query(query, (stock_name, start_date, end_date)), 
                                 columns=['date', 'actual_price'])
        
        # 예측값 데이터 조회
        query = """
        SELECT target_date, predicted_price
        FROM predicted_stock_prices
        WHERE stock_name = %s
        AND target_date BETWEEN %s AND %s
        ORDER BY target_date
        """
        predicted_data = pd.DataFrame(execute_query(query, (stock_name, start_date, end_date)),
                                   columns=['date', 'predicted_price'])
        
        if actual_data.empty or predicted_data.empty:
            print("비교할 데이터가 없습니다.")
            return None
            
        # 데이터 병합
        comparison = pd.merge(actual_data, predicted_data, on='date', how='inner')
        
        # 평가 지표 계산
        mae = mean_absolute_error(comparison['actual_price'], comparison['predicted_price'])
        rmse = np.sqrt(mean_squared_error(comparison['actual_price'], comparison['predicted_price']))
        mape = np.mean(np.abs((comparison['actual_price'] - comparison['predicted_price']) / comparison['actual_price'])) * 100
        
        # 방향성 정확도 계산
        actual_direction = np.sign(comparison['actual_price'].diff())
        predicted_direction = np.sign(comparison['predicted_price'].diff())
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # 결과 출력
        print("\n[예측 성능 평가]")
        print(f"평가 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"평균 절대 오차 (MAE): {mae:,.0f}원")
        print(f"평균 제곱근 오차 (RMSE): {rmse:,.0f}원")
        print(f"평균 절대 백분율 오차 (MAPE): {mape:.2f}%")
        print(f"방향성 정확도: {direction_accuracy:.2f}%")
        
        # 상세 비교 결과 출력
        print("\n[상세 비교]")
        print(f"{'날짜':<12} {'실제가격':>10} {'예측가격':>10} {'오차':>10} {'오차율':>8}")
        print("-" * 55)
        
        for _, row in comparison.iterrows():
            error = row['actual_price'] - row['predicted_price']
            error_rate = (error / row['actual_price']) * 100
            print(f"{row['date'].strftime('%Y-%m-%d'):<12} "
                  f"{row['actual_price']:>10,.0f} "
                  f"{row['predicted_price']:>10,.0f} "
                  f"{error:>10,.0f} "
                  f"{error_rate:>8.2f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'comparison_data': comparison
        }
        
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        return None

# 메인 실행 부분 수정
if __name__ == "__main__":
    print("📢 주가 예측 모델 학습을 시작합니다...")
    create_predictions_table()
    
    # 데이터 로드
    stock_data, sentiment_data, economic_data = load_data_from_db()
    
    # 앙상블 모델 초기화
    ensemble = EnsembleModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    ensemble.build_models()
    
    try:
        # 테스트 데이터에 대한 예측 수행
        predictions = ensemble.predict(X_test)
        
        # 마지막 예측 결과 가져오기
        last_prediction = predictions[-1]
        
        # 데이터베이스에서 가장 최근 주가 가져오기
        last_actual_price = get_latest_stock_price('LG전자')
        print(f"\n예측 기준 주가: {last_actual_price:,.0f}원")
        
        # 상대적 변화율을 실제 가격으로 변환
        predicted_prices = convert_predictions_to_prices(last_prediction, last_actual_price)
        
        # 실제 값과 예측 값 비교
        target_dates = ['2024-03-25', '2024-03-26', '2024-03-27', '2024-03-28', '2024-03-31']
        target_prices = [82800, 82700, 82400, 80000, 77200]  # 실제 가격 입력
        
        # 예측 결과 시각화
        plt.figure(figsize=(12, 6))
        
        # 날짜를 datetime 객체로 변환
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in target_dates]
        
        # 실제 가격과 예측 가격 플롯
        plt.plot(dates, target_prices, 'b-', label='실제 가격', marker='o')
        plt.plot(dates, predicted_prices, 'r--', label='예측 가격', marker='s')
        
        # 그래프 스타일링
        plt.title('LG전자 주가 예측 결과 (2024년 3월)', fontsize=14)
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('주가 (원)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # x축 날짜 포맷 설정
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        
        # 오차율 계산 및 표시
        error_rates = [(pred - actual) / actual * 100 for pred, actual in zip(predicted_prices, target_prices)]
        for i, (date, error) in enumerate(zip(dates, error_rates)):
            plt.annotate(f'{error:.2f}%',
                        xy=(date, max(predicted_prices[i], target_prices[i])),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # 예측 결과 상세 분석
        print("\n[예측 결과 분석]")
        print(f"{'날짜':<12} {'실제 가격':>10} {'예측 가격':>10} {'오차율':>8}")
        print("-" * 45)
        for date, actual, pred, error in zip(target_dates, target_prices, predicted_prices, error_rates):
            print(f"{date:<12} {actual:>10,d} {pred:>10.0f} {error:>7.2f}%")
        
        # 전체 예측 성능 지표
        mae = mean_absolute_error(target_prices, predicted_prices)
        mse = mean_squared_error(target_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(error_rates))
        
        print("\n[전체 예측 성능]")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        # 예측 결과 저장
        for i, (date, pred, actual) in enumerate(zip(target_dates, predicted_prices, target_prices)):
            save_prediction(
                stock_code='066570',  # LG전자 종목코드
                stock_name='LG전자',
                prediction_date=datetime.now(),
                target_date=datetime.strptime(date, '%Y-%m-%d'),
                predicted_price=pred
            )
        
        print("\n✅ 예측 결과가 데이터베이스에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)