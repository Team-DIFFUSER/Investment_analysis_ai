import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
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

# TensorFlow 세션 초기화
import tensorflow as tf

# 기존 세션 정리 및 메모리 해제
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

# GPU 설정 단순화
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '4'
os.environ['TF_USE_CUDNN'] = '0'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU 사용 가능 여부 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU 사용 가능: {gpus[0]}")
else:
    print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")

# TensorFlow 최적화 설정
tf.config.optimizer.set_jit(False)
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
        CREATE TABLE IF NOT EXISTS price_predictions (
            id SERIAL PRIMARY KEY,
            stock_code VARCHAR(10) NOT NULL,
            stock_name VARCHAR(50) NOT NULL,
            prediction_date TIMESTAMPTZ NOT NULL,
            target_date TIMESTAMPTZ NOT NULL,
            predicted_price DECIMAL(10,2) NOT NULL,
            actual_price DECIMAL(10,2),
            prediction_error DECIMAL(10,2),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """, None),
        ("CREATE INDEX IF NOT EXISTS idx_price_predictions_date ON price_predictions (prediction_date, target_date);", None),
        ("CREATE INDEX IF NOT EXISTS idx_price_predictions_stock ON price_predictions (stock_code);", None)
    ]
    execute_transaction(queries)
    print("Price predictions table created successfully!")

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

def create_predicted_prices_table():
    """예측 가격을 저장할 테이블 생성"""
    queries = [
        # 기존 테이블 삭제
        ("DROP TABLE IF EXISTS predicted_stock_prices;", None),
        
        # 새 테이블 생성
        ("""
        CREATE TABLE predicted_stock_prices (
            id SERIAL PRIMARY KEY,
            time TIMESTAMPTZ NOT NULL,
            stock_code VARCHAR(10) NOT NULL,
            stock_name VARCHAR(50) NOT NULL,
            open_price DECIMAL(10,2),
            high_price DECIMAL(10,2),
            low_price DECIMAL(10,2),
            close_price DECIMAL(10,2),
            volume BIGINT,
            market_cap BIGINT,
            foreign_holding BIGINT,
            foreign_holding_ratio DECIMAL(5,2),
            prediction_date TIMESTAMPTZ NOT NULL,
            prediction_confidence DECIMAL(5,2),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_prediction UNIQUE (time, stock_code)
        );
        """, None),
        
        # 인덱스 생성
        ("CREATE INDEX idx_predicted_prices_time ON predicted_stock_prices (time);", None),
        ("CREATE INDEX idx_predicted_prices_stock ON predicted_stock_prices (stock_code);", None),
        ("CREATE INDEX idx_predicted_prices_pred_date ON predicted_stock_prices (prediction_date);", None)
    ]
    
    try:
        execute_transaction(queries)
        print("Predicted stock prices table created successfully!")
    except Exception as e:
        print(f"Error creating table: {e}")
        raise

def save_or_update_predicted_price(prediction_data):
    """예측 가격을 저장하거나 업데이트"""
    query = """
    INSERT INTO predicted_stock_prices (
        time, stock_code, stock_name, open_price, high_price, low_price,
        close_price, volume, market_cap, foreign_holding, foreign_holding_ratio,
        prediction_date, prediction_confidence
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (time, stock_code) 
    DO UPDATE SET
        open_price = EXCLUDED.open_price,
        high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price,
        close_price = EXCLUDED.close_price,
        volume = EXCLUDED.volume,
        market_cap = EXCLUDED.market_cap,
        foreign_holding = EXCLUDED.foreign_holding,
        foreign_holding_ratio = EXCLUDED.foreign_holding_ratio,
        prediction_date = EXCLUDED.prediction_date,
        prediction_confidence = EXCLUDED.prediction_confidence,
        updated_at = CURRENT_TIMESTAMP;
    """
    
    try:
        execute_query(query, prediction_data, fetch=False)
        print(f"✅ 예측 가격이 저장/업데이트되었습니다. (날짜: {prediction_data[0]}, 종목: {prediction_data[2]})")
    except Exception as e:
        print(f"❌ 예측 가격 저장 중 오류 발생: {e}")
        raise

def save_predicted_prices(predictions, dates, stock_code, stock_name, confidence=0.95):
    """예측된 가격들을 데이터베이스에 저장"""
    try:
        # 테이블 생성 확인
        create_predicted_prices_table()
        
        # 각 예측 날짜에 대해 데이터 저장
        for i, (date, pred_price) in enumerate(zip(dates, predictions)):
            # NumPy 타입을 Python 네이티브 타입으로 변환
            pred_price = float(pred_price)
            
            # 예측 가격을 기반으로 OHLCV 데이터 생성
            # 실제 데이터의 변동성을 고려하여 가격 범위 설정
            price_volatility = 0.02  # 2% 변동성 가정
            volume_volatility = 0.1  # 10% 변동성 가정
            
            # 가격 데이터 생성
            close_price = pred_price
            high_price = close_price * (1 + price_volatility)
            low_price = close_price * (1 - price_volatility)
            open_price = (high_price + low_price) / 2
            
            # 거래량 데이터 생성 (이전 거래량의 평균을 기반으로)
            volume = int(np.random.normal(1000000, 1000000 * volume_volatility))
            market_cap = int(close_price * volume * 0.1)  # 시가총액 추정
            foreign_holding = int(market_cap * 0.3)  # 외국인 보유량 추정
            foreign_holding_ratio = 30.0  # 외국인 보유 비율 추정
            
            # 예측 데이터 저장
            prediction_data = (
                date,  # time
                stock_code,  # stock_code
                stock_name,  # stock_name
                float(open_price),  # open_price
                float(high_price),  # high_price
                float(low_price),  # low_price
                float(close_price),  # close_price
                int(volume),  # volume
                int(market_cap),  # market_cap
                int(foreign_holding),  # foreign_holding
                float(foreign_holding_ratio),  # foreign_holding_ratio
                datetime.now(),  # prediction_date
                float(confidence)  # prediction_confidence
            )
            
            save_or_update_predicted_price(prediction_data)
            
        print("✅ 모든 예측 가격이 데이터베이스에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 예측 가격 저장 중 오류 발생: {e}")
        raise

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

def save_prediction(stock_code, stock_name, prediction_date, target_date, predicted_price, actual_price=None):
    """예측 결과를 데이터베이스에 저장"""
    # NumPy 타입을 Python 네이티브 타입으로 변환
    predicted_price = float(predicted_price)
    if actual_price is not None:
        actual_price = float(actual_price)
        prediction_error = float(predicted_price - actual_price)
    else:
        prediction_error = None
    
    query = """
    INSERT INTO price_predictions (
        stock_code, stock_name, prediction_date, target_date,
        predicted_price, actual_price, prediction_error
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        stock_code, stock_name, prediction_date, target_date,
        predicted_price, actual_price, prediction_error
    )
    execute_query(query, params, fetch=False)

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
    # 가격 변동률 계산
    df['price_change'] = df['현재가'].pct_change()
    df['price_volatility'] = df['price_change'].rolling(window=5).std()
    
    # 거래량 변동률
    df['volume_change'] = df['거래량'].pct_change()
    df['volume_volatility'] = df['volume_change'].rolling(window=5).std()
    
    # 가격 모멘텀
    df['price_momentum'] = df['현재가'] / df['현재가'].rolling(window=5).mean() - 1
    
    # 거래량 모멘텀
    df['volume_momentum'] = df['거래량'] / df['거래량'].rolling(window=5).mean() - 1
    
    # 가격 변동 추세
    df['price_trend'] = df['현재가'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # 이상치 처리 (IQR 방법)
    for col in ['현재가', '거래량', 'price_change', 'volume_change']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
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
        self.price_scaler = MinMaxScaler()  # RobustScaler 대신 MinMaxScaler 사용
        self.feature_scaler = MinMaxScaler()
        self.price_min = None
        self.price_max = None

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
        
        # 가격 데이터의 최소/최대값 저장
        self.price_min = price_data.min().values
        self.price_max = price_data.max().values
        
        # 각각 스케일링
        scaled_price = self.price_scaler.fit_transform(price_data)
        scaled_other = self.feature_scaler.fit_transform(other_data)
        
        # 스케일링된 데이터 결합
        scaled_data = np.concatenate([scaled_price, scaled_other], axis=1)
        scaled_df = pd.DataFrame(scaled_data, columns=price_cols + other_data.columns.tolist())
        
        # 원래 컬럼 순서 복원
        scaled_df = scaled_df[data_copy.columns]
        
        return scaled_df

    def inverse_transform_price(self, scaled_price):
        if len(scaled_price.shape) == 1:
            scaled_price = scaled_price.reshape(-1, 1)
        
        # MinMaxScaler 역변환
        unscaled = self.price_scaler.inverse_transform(scaled_price)
        return unscaled.flatten()

# 손실 함수 개선
@tf.keras.utils.register_keras_serializable()
def enhanced_weighted_time_mse(y_true, y_pred):
    # 수치적 안정성을 위한 작은 값 추가
    epsilon = 1e-7
    
    # 시간 가중치 조정 (첫날 가중치 강화)
    time_weights = tf.constant([0.4, 0.3, 0.2, 0.07, 0.03], dtype=tf.float32)
    
    # 기본 MSE
    mse_per_step = tf.reduce_mean(tf.square(y_true - y_pred) + epsilon, axis=0)
    
    # 과대 예측 패널티 (첫날 강화)
    overprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_pred - y_true) * tf.constant([25.0, 20.0, 15.0, 10.0, 8.0], dtype=tf.float32)
    )
    
    # 과소 예측 패널티 (첫날 강화)
    underprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_true - y_pred) * tf.constant([10.0, 8.0, 6.0, 5.0, 4.0], dtype=tf.float32)
    )
    
    # 추세 손실 (첫날 강화)
    y_true_diff = y_true[:, 1:] - y_true[:, :-1]
    y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    trend_weights = tf.constant([0.4, 0.3, 0.2, 0.1], dtype=tf.float32)
    trend_loss = tf.reduce_mean(tf.square(y_true_diff - y_pred_diff) * trend_weights + epsilon)
    
    # 방향성 손실 (첫날 강화)
    direction_weights = tf.constant([0.4, 0.3, 0.2, 0.1], dtype=tf.float32)
    direction_loss = tf.reduce_mean(
        tf.square(tf.sign(y_true_diff) - tf.sign(y_pred_diff)) * direction_weights + epsilon
    )
    
    # 연속성 손실 추가
    continuity_loss = tf.reduce_mean(
        tf.square(y_pred[:, 1:] - y_pred[:, :-1] - (y_true[:, 1:] - y_true[:, :-1]))
    )
    
    # 가중치 적용
    weighted_loss = (
        tf.reduce_sum(mse_per_step * time_weights) +
        1.0 * overprediction_penalty +  # 과대 예측 패널티 더욱 강화
        0.3 * underprediction_penalty +  # 과소 예측 패널티 감소
        0.5 * trend_loss +
        0.4 * direction_loss +
        0.3 * continuity_loss  # 연속성 손실 추가
    )
    return weighted_loss

# 데이터 증강 함수 개선
def augment_data(X, y, noise_level=0.01):
    """데이터에 노이즈를 추가하여 증강"""
    X_aug = X.copy()
    y_aug = y.copy()
    
    # 가격 데이터 추출 (첫 번째 컬럼이 가격)
    price_data = X[:, :, 0]
    
    # 추세 계산
    price_trend = np.diff(price_data, axis=1)
    avg_trend = np.mean(price_trend, axis=1)
    
    # 추세 기반 노이즈 생성
    trend_noise = np.random.normal(0, noise_level * 0.5, X.shape)
    trend_noise[:, :, 0] *= np.sign(avg_trend)[:, np.newaxis]  # 추세 방향으로 노이즈 조정
    
    # 기본 가우시안 노이즈
    base_noise = np.random.normal(0, noise_level, X.shape)
    
    # 노이즈 결합
    combined_noise = base_noise + trend_noise
    
    # 노이즈 적용
    X_aug = X_aug + combined_noise
    
    # 시계열 특성 보존을 위한 노이즈 제한
    X_aug = np.clip(X_aug, X.min(), X.max())
    
    # 가격 연속성 보장
    price_diff = np.diff(X_aug[:, :, 0], axis=1)
    max_allowed_diff = np.std(price_data) * 0.1  # 최대 허용 변동폭
    price_diff = np.clip(price_diff, -max_allowed_diff, max_allowed_diff)
    
    # 수정된 가격 데이터 재구성 (브로드캐스팅 수정)
    initial_prices = X_aug[:, 0, 0][:, np.newaxis]  # (batch_size, 1)
    cumulative_diff = np.cumsum(price_diff, axis=1)  # (batch_size, seq_len-1)
    X_aug[:, 1:, 0] = initial_prices + cumulative_diff
    
    return X_aug, y_aug

# MCI-GRU 셀 구현 (cuDNN 사용하지 않는 버전)
@tf.keras.utils.register_keras_serializable()
class MCI_GRU_Cell(Layer):
    def __init__(self, units, return_sequences=False, go_backwards=False, **kwargs):
        super(MCI_GRU_Cell, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.state_size = [units]
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Market Context Integration weights
        self.W_m = self.add_weight(
            shape=(input_dim, self.units),
            name='W_m',
            initializer='glorot_uniform'
        )
        self.U_m = self.add_weight(
            shape=(self.units, self.units),
            name='U_m',
            initializer='glorot_uniform'
        )
        self.b_m = self.add_weight(
            shape=(self.units,),
            name='b_m',
            initializer='zeros'
        )
        
        # GRU gates
        self.W_z = self.add_weight(
            shape=(input_dim, self.units),
            name='W_z',
            initializer='glorot_uniform'
        )
        self.U_z = self.add_weight(
            shape=(self.units, self.units),
            name='U_z',
            initializer='glorot_uniform'
        )
        self.b_z = self.add_weight(
            shape=(self.units,),
            name='b_z',
            initializer='zeros'
        )
        
        self.W_r = self.add_weight(
            shape=(input_dim, self.units),
            name='W_r',
            initializer='glorot_uniform'
        )
        self.U_r = self.add_weight(
            shape=(self.units, self.units),
            name='U_r',
            initializer='glorot_uniform'
        )
        self.b_r = self.add_weight(
            shape=(self.units,),
            name='b_r',
            initializer='zeros'
        )
        
        self.W_h = self.add_weight(
            shape=(input_dim, self.units),
            name='W_h',
            initializer='glorot_uniform'
        )
        self.U_h = self.add_weight(
            shape=(self.units, self.units),
            name='U_h',
            initializer='glorot_uniform'
        )
        self.b_h = self.add_weight(
            shape=(self.units,),
            name='b_h',
            initializer='zeros'
        )
        
        self.built = True

    def call(self, inputs, states=None):
        # Handle states properly
        if states is None:
            states = [tf.zeros((tf.shape(inputs)[0], self.units))]
        else:
            states = list(states)
        
        # Ensure inputs are 3D
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
        
        # Handle bidirectional processing
        if self.go_backwards:
            inputs = tf.reverse(inputs, axis=[1])
        
        # Get sequence length
        sequence_length = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]
        
        # Initialize outputs tensor
        if self.return_sequences:
            outputs = tf.TensorArray(tf.float32, size=sequence_length)
        else:
            outputs = tf.zeros((batch_size, self.units))
        
        # Initialize loop variables
        i = tf.constant(0)
        h = states[0]
        
        # Define loop body
        def body(i, h, outputs):
            # Get current input
            x_t = inputs[:, i, :]
            
            # Market context integration
            m = tf.tanh(
                tf.matmul(x_t, self.W_m) +
                tf.matmul(h, self.U_m) +
                self.b_m
            )
            
            # GRU gates with market context
            z = tf.sigmoid(
                tf.matmul(x_t, self.W_z) +
                tf.matmul(h, self.U_z) +
                self.b_z
            )
            
            r = tf.sigmoid(
                tf.matmul(x_t, self.W_r) +
                tf.matmul(h, self.U_r) +
                self.b_r
            )
            
            h_candidate = tf.tanh(
                tf.matmul(x_t, self.W_h) +
                tf.matmul(r * h, self.U_h) +
                self.b_h
            )
            
            # Update state with market context
            h = z * h + (1 - z) * h_candidate + m
            
            # Store output if return_sequences is True
            if self.return_sequences:
                outputs = outputs.write(i, h)
            
            return i + 1, h, outputs
        
        # Define loop condition
        def cond(i, h, outputs):
            return i < sequence_length
        
        # Run the loop
        _, final_h, final_outputs = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[i, h, outputs],
            maximum_iterations=sequence_length
        )
        
        if self.return_sequences:
            # Convert TensorArray to tensor
            outputs = final_outputs.stack()
            # Reshape to [batch, seq_len, units]
            outputs = tf.transpose(outputs, [1, 0, 2])
            if self.go_backwards:
                outputs = tf.reverse(outputs, axis=[1])
            return outputs
        else:
            return final_h

    def get_config(self):
        config = super(MCI_GRU_Cell, self).get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'go_backwards': self.go_backwards
        })
        return config

# 데이터 전처리 개선
def prepare_data(merged_data, sequence_length=30):
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
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length:i + sequence_length + 5, 0])
    
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# 모델 구조 개선
def build_enhanced_model(input_shape, output_days=5):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Market Context Layer with attention
    x = MarketContextLayer(input_shape[-1])(inputs)
    
    # Ensure 3D tensor for GRU while preserving sequence length
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[-1]))(x)
    
    # Trend Detection Layer
    trend_conv = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    trend_conv = BatchNormalization()(trend_conv)
    trend_conv = Dropout(0.2)(trend_conv)
    
    # Bidirectional GRU Layer with trend information
    x = tf.keras.layers.Bidirectional(
        GRU(input_shape[-1], return_sequences=True)
    )(x)
    x = Dropout(0.2)(x)
    
    # Combine trend and GRU outputs
    x = tf.keras.layers.Concatenate()([x, trend_conv])
    
    # Temporal Attention Layer
    attention_output = MultiHeadAttention(
        num_heads=4, 
        key_dim=input_shape[-1]
    )(x, x)
    x = tf.keras.layers.Add()([x, attention_output])
    x = BatchNormalization()(x)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers with residual connections
    dense1 = Dense(64, activation='relu')(x)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.2)(dense1)
    
    dense2 = Dense(32, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.2)(dense2)
    
    # Residual connection
    residual = Dense(32)(x)
    x = tf.keras.layers.Add()([dense2, residual])
    
    # Trend-specific output layers
    outputs = []
    for i in range(output_days):
        # Trend-specific dense layer
        trend_dense = Dense(16, activation='relu')(x)
        trend_dense = BatchNormalization()(trend_dense)
        
        # Price prediction layer
        day_output = Dense(8, activation='relu')(trend_dense)
        day_output = BatchNormalization()(day_output)
        day_output = Dense(1, name=f'day_{i+1}_output')(day_output)
        outputs.append(day_output)
    
    final_output = tf.keras.layers.Concatenate()(outputs)
    
    # Create model
    model = Model(inputs=inputs, outputs=final_output)
    
    # Compile model with adjusted learning rate and loss function
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss=enhanced_weighted_time_mse,
        metrics=['mae', 'mse']
    )
    
    return model

@tf.keras.utils.register_keras_serializable()
class MarketContextLayer(Layer):
    def __init__(self, units, **kwargs):
        super(MarketContextLayer, self).__init__(**kwargs)
        self.units = units
        self.attention = MultiHeadAttention(num_heads=4, key_dim=units)
        self.dense = Dense(units)
        self.batch_norm = BatchNormalization()
        
    def call(self, inputs):
        # Ensure inputs are 3D for attention
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            
        # Self-attention for market context
        attention_output = self.attention(inputs, inputs)
        
        # Dense transformation
        x = self.dense(attention_output)
        x = self.batch_norm(x)
        
        return x
    
    def get_config(self):
        config = super(MarketContextLayer, self).get_config()
        config.update({'units': self.units})
        return config

# 학습 과정 개선
def train_enhanced_model(model, X_train, y_train, X_val, y_val):
    # 콜백 정의
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            min_delta=0.0001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            '/kaggle/working/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # 데이터 증강 적용 (reduced noise)
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_level=0.00005)
    
    # 데이터셋 최적화
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_aug, y_train_aug))
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(16)  # Smaller batch size
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(16)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # 학습
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=250,  # Increased epochs
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# 앙상블 모델 클래스 개선
class EnsembleModel:
    def __init__(self, input_shape, n_models=5):  # 모델 수 증가
        self.models = []
        self.input_shape = input_shape
        self.n_models = n_models
        self.model_weights = None
        
    def build_models(self):
        for i in range(self.n_models):
            model = build_enhanced_model(self.input_shape)
            self.models.append(model)
    
    def train(self, X_train, y_train, X_val, y_val):
        histories = []
        val_losses = []
        
        # 각 모델 학습
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            
            # 데이터 증강 강화
            X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_level=0.0005 * (i + 1))
            
            history = train_enhanced_model(model, X_train_aug, y_train_aug, X_val, y_val)
            histories.append(history)
            
            # 검증 손실 저장
            val_loss = min(history.history['val_loss'])
            val_losses.append(val_loss)
            
            # GPU 메모리 정리
            tf.keras.backend.clear_session()
            
            # 모델 저장 및 메모리 해제
            model.save(f'/kaggle/working/model_{i+1}.keras')
            del model
            tf.keras.backend.clear_session()
        
        # 모델 가중치 계산
        val_losses = np.array(val_losses)
        self.model_weights = 1.0 / (val_losses + 1e-7)
        self.model_weights = self.model_weights / np.sum(self.model_weights)
        
        return histories
    
    def predict(self, X):
        predictions = []
        for i in range(self.n_models):
            # 모델 로드
            model = tf.keras.models.load_model(
                f'/kaggle/working/model_{i+1}.keras',
                custom_objects={
                    'MCI_GRU_Cell': MCI_GRU_Cell
                }
            )
            
            # 예측
            pred = model.predict(X)
            predictions.append(pred * self.model_weights[i])
            
            # 메모리 정리
            del model
            tf.keras.backend.clear_session()
            
        return np.sum(predictions, axis=0)

# 데이터 전처리 실행
print("데이터 전처리 시작...")
X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(merged_data)
print("데이터 전처리 완료")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 앙상블 모델 사용
ensemble = EnsembleModel(input_shape=(X_train.shape[1], X_train.shape[2]))
ensemble.build_models()
histories = ensemble.train(X_train, y_train, X_val, y_val)

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

# 모델 저장
for i, model in enumerate(ensemble.models):
    model.save(f'/kaggle/working/stock_prediction_model_{i+1}.keras')  # .h5 대신 .keras 사용
    
# 스케일러 저장
with open(f'/kaggle/working/stock_prediction_scaler_{i+1}.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
# 모델 메타데이터 저장
model_metadata = {
    'input_shape': X_train.shape[1:],
    'output_days': 5,
    'price_scaler_params': {
        'scale_': scaler.price_scaler.scale_.tolist(),
        'min_': scaler.price_scaler.min_.tolist(),
        'data_min_': scaler.price_scaler.data_min_.tolist(),
        'data_max_': scaler.price_scaler.data_max_.tolist(),
        'feature_names_in_': scaler.price_scaler.feature_names_in_.tolist()
    }
}
    
with open(f'/kaggle/working/model_metadata_{i+1}.json', 'w') as f:
    json.dump(model_metadata, f)
        
print(f"모델 {i+1} 저장 완료")
print(f"스케일러 {i+1} 저장 완료")
print(f"메타데이터 {i+1} 저장 완료")

# 예측 결과 분석 및 시각화
try:
    # 테스트 데이터에 대한 예측 수행
    predictions = ensemble.predict(X_test)

    # 마지막 예측 결과 가져오기 (가장 최근 예측)
    last_prediction = predictions[-1]

    # 예측 결과를 원래 스케일로 변환
    last_prediction = scaler.inverse_transform_price(last_prediction.reshape(-1, 1)).flatten()

    # 실제 값과 예측 값 비교
    target_dates = ['2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28']
    target_prices = [69700, 67500, 67200, 66800, 65700]

    # 예측 결과 시각화
    plt.figure(figsize=(12, 6))

    # 날짜를 datetime 객체로 변환
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in target_dates]

    # 실제 가격과 예측 가격 플롯
    plt.plot(dates, target_prices, 'b-', label='실제 가격', marker='o')
    plt.plot(dates, last_prediction, 'r--', label='예측 가격', marker='s')

    # 그래프 스타일링
    plt.title('LG전자 주가 예측 결과 (2025년 3월)', fontsize=14)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('주가 (원)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # x축 날짜 포맷 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)

    # 오차율 계산 및 표시
    error_rates = [(pred - actual) / actual * 100 for pred, actual in zip(last_prediction, target_prices)]
    for i, (date, error) in enumerate(zip(dates, error_rates)):
        plt.annotate(f'{error:.2f}%',
                    xy=(date, max(last_prediction[i], target_prices[i])),
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
    for date, actual, pred, error in zip(target_dates, target_prices, last_prediction, error_rates):
        print(f"{date:<12} {actual:>10,d} {pred:>10.0f} {error:>7.2f}%")

    # 전체 예측 성능 지표
    mae = mean_absolute_error(target_prices, last_prediction)
    mse = mean_squared_error(target_prices, last_prediction)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(error_rates))

    print("\n[전체 예측 성능]")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # 예측 결과 저장
    for i, (date, pred, actual) in enumerate(zip(target_dates, last_prediction, target_prices)):
        save_prediction(
            stock_code='066570',  # LG전자 종목코드
            stock_name='LG전자',
            prediction_date=datetime.now(),
            target_date=datetime.strptime(date, '%Y-%m-%d'),
            predicted_price=pred,
            actual_price=actual
        )
    
    print("✅ 예측 결과가 데이터베이스에 저장되었습니다.")

    # 예측 결과 저장 (predicted_stock_prices 테이블)
    try:
        # 예측 날짜 생성 (target_dates 사용)
        prediction_dates = [datetime.strptime(date, '%Y-%m-%d') for date in target_dates]
        
        # 예측 결과 저장
        save_predicted_prices(
            predictions=last_prediction,
            dates=prediction_dates,
            stock_code='066570',  # LG전자 종목코드
            stock_name='LG전자',
            confidence=0.95
        )
        
    except Exception as e:
        print(f"예측 결과 저장 중 오류 발생: {e}")
        raise

except Exception as e:
    print(f"예측 결과 분석 중 오류 발생: {e}")
    raise

if __name__ == "__main__":
    print("📢 주가 예측 모델 학습을 시작합니다...")
    create_predictions_table()
    
    # 데이터 로드
    stock_data, sentiment_data, economic_data = load_data_from_db()
    
    # 예측 결과 저장
    for i, (date, pred, actual) in enumerate(zip(target_dates, last_prediction, target_prices)):
        save_prediction(
            stock_code='066570',  # LG전자 종목코드
            stock_name='LG전자',
            prediction_date=datetime.now(),
            target_date=datetime.strptime(date, '%Y-%m-%d'),
            predicted_price=pred,
            actual_price=actual
        )
    
    print("✅ 예측 결과가 데이터베이스에 저장되었습니다.")