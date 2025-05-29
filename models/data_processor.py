import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict
import logging
from datetime import datetime, timedelta
import tensorflow as tf
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPriceScaler:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.price_min = None
        self.price_max = None
        self.feature_names = []
        
    def fit_transform(self, data, price_cols):
        """데이터 스케일링"""
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
        """스케일된 가격을 원래 가격으로 변환"""
        if len(scaled_price.shape) == 1:
            scaled_price = scaled_price.reshape(-1, 1)
        return self.price_scaler.inverse_transform(scaled_price).flatten()

class DataProcessor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler = EnhancedPriceScaler()
        
    def add_financial_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """재무 지표 추가"""
        try:
            # ROE (Return on Equity)
            df['ROE'] = (df['net_income'] / df['total_equity']) * 100
            
            # ROA (Return on Assets)
            df['ROA'] = (df['net_income'] / df['total_assets']) * 100
            
            # 영업이익률
            df['operating_profit_margin'] = (df['operating_income'] / df['revenue']) * 100
            
            # 매출성장률
            df['revenue_growth'] = df['revenue'].pct_change() * 100
            
            # 부채비율
            df['debt_ratio'] = (df['total_liabilities'] / df['total_equity']) * 100
            
            # 유동비율
            df['current_ratio'] = df['current_assets'] / df['current_liabilities']
            
            # 이자보상배율
            df['interest_coverage'] = df['operating_income'] / df['interest_expense']
            
            # PER (Price to Earnings Ratio)
            df['PER'] = df['close'] / (df['net_income'] / df['shares_outstanding'])
            
            # PBR (Price to Book Ratio)
            df['PBR'] = df['close'] / (df['total_equity'] / df['shares_outstanding'])
            
            # EV/EBITDA
            df['EV_EBITDA'] = (df['market_cap'] + df['total_liabilities']) / df['EBITDA']
            
            # 영업현금흐름비율
            df['operating_cash_flow_ratio'] = df['operating_cash_flow'] / df['total_liabilities']
            
            # 잉여현금흐름
            df['free_cash_flow'] = df['operating_cash_flow'] - df['investing_cash_flow']
            
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"재무지표 추가 중 오류 발생: {str(e)}")
            raise
    
    def add_technical_indicators(self, df):
        """기술적 지표 추가"""
        # RSI
        df['RSI'] = df.ta.rsi(close='close_price', length=14)
        
        # MACD
        macd = df.ta.macd(close='close_price')
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
        df['MACD_HIST'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = df.ta.bbands(close='close_price')
        df['BB_UPPER'] = bb['BBU_20_2.0']
        df['BB_MIDDLE'] = bb['BBM_20_2.0']
        df['BB_LOWER'] = bb['BBL_20_2.0']
        df['BB_PERCENT'] = bb['BBP_20_2.0']
        
        # Moving Averages
        df['MA5'] = df.ta.sma(close='close_price', length=5)
        df['MA20'] = df.ta.sma(close='close_price', length=20)
        df['MA60'] = df.ta.sma(close='close_price', length=60)
        
        # Volume Moving Averages
        df['VOLUME_MA5'] = df.ta.sma(close='volume', length=5)
        df['VOLUME_MA20'] = df.ta.sma(close='volume', length=20)
        
        # Volume Ratio
        df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_MA20']
        
        # Momentum Indicators
        df['MOM'] = df.ta.roc(close='close_price', length=10)
        df['ROC'] = df.ta.roc(close='close_price', length=20)
        
        return df
    
    def enhanced_preprocessing(self, df):
        """데이터 전처리 강화"""
        # 가격 변동률 계산
        df['price_change_5d'] = df['close_price'].pct_change(5)
        
        # 거래량 변동률
        df['volume_change'] = df['volume'].pct_change()
        
        # 시가총액 변동률
        df['market_cap_change'] = df['market_cap'].pct_change()
        
        # 외국인 보유 비율 변동
        df['foreign_holding_change'] = df['foreign_holding_ratio'].diff()
        
        # 결측치 처리
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        
        # 이상치 처리
        for col in ['close_price', 'volume', 'market_cap']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        
        return df
    
    def prepare_data(self, stock_data, sentiment_data, economic_data):
        """데이터 준비"""
        # 데이터 병합
        merged_data = pd.merge(stock_data, sentiment_data, left_on='time', right_on='pub_date', how='left')
        merged_data = pd.merge(merged_data, economic_data, on='time', how='left')
        
        # 기술적 지표 추가
        merged_data = self.add_technical_indicators(merged_data)
        
        # 전처리 적용
        merged_data = self.enhanced_preprocessing(merged_data)
        
        # 특성 선택
        feature_columns = [
            'close_price', 'volume', 'market_cap', 'foreign_holding', 'foreign_holding_ratio',
            'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
            'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_PERCENT',
            'MA5', 'MA20', 'MA60', 'VOLUME_MA5', 'VOLUME_MA20',
            'VOLUME_RATIO', 'MOM', 'ROC',
            'finbert_positive', 'finbert_negative', 'finbert_neutral',
            'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
        ]
        
        # 데이터 정규화
        price_cols = ['close_price']
        scaled_data = self.scaler.fit_transform(merged_data[feature_columns], price_cols)
        
        # DataFrame을 numpy array로 변환
        scaled_data = scaled_data.values
        
        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - 4):
            # 입력 시퀀스
            X.append(scaled_data[i:(i + self.sequence_length)])
            
            # 타겟 시퀀스 (마지막 가격을 기준으로 상대적 변화율로 변환)
            last_price = scaled_data[i + self.sequence_length - 1, 0]  # 마지막 가격
            target_prices = scaled_data[i + self.sequence_length:i + self.sequence_length + 5, 0]
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
        
        return X_train, y_train, X_val, y_val, X_test, y_test, self.scaler
    
    def prepare_prediction_data(self, stock_data, sentiment_data, economic_data, sequence_length):
        """예측을 위한 데이터 준비"""
        # 데이터 병합
        merged_data = pd.merge(stock_data, sentiment_data, left_on='time', right_on='pub_date', how='left')
        merged_data = pd.merge(merged_data, economic_data, on='time', how='left')
        
        # 기술적 지표 추가
        merged_data = self.add_technical_indicators(merged_data)
        
        # 전처리 적용
        merged_data = self.enhanced_preprocessing(merged_data)
        
        # 특성 선택
        feature_columns = [
            'close_price', 'volume', 'market_cap', 'foreign_holding', 'foreign_holding_ratio',
            'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
            'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_PERCENT',
            'MA5', 'MA20', 'MA60', 'VOLUME_MA5', 'VOLUME_MA20',
            'VOLUME_RATIO', 'MOM', 'ROC',
            'finbert_positive', 'finbert_negative', 'finbert_neutral',
            'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
        ]
        
        # 데이터 정규화
        price_cols = ['close_price']
        scaled_data = self.scaler.fit_transform(merged_data[feature_columns], price_cols)
        
        # 시퀀스 데이터 생성
        X = []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
        
        return np.array(X) 