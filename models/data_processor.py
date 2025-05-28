import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.price_min = None
        self.price_max = None
        self.feature_names = None
        
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
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 추가"""
        try:
            # 이동평균선
            for window in [5, 20, 60, 120]:
                df[f'MA_{window}'] = df['close'].rolling(window=window).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_std'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
            
            # 거래량 지표
            df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
            df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
            
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 중 오류 발생: {str(e)}")
            raise
    
    def prepare_data(self, stock_data, sentiment_data, economic_data, sequence_length=20):
        """
        데이터 전처리 및 시퀀스 생성
        
        Args:
            stock_data (pd.DataFrame): 주가 데이터
            sentiment_data (pd.DataFrame): 감성 데이터
            economic_data (pd.DataFrame): 경제 데이터
            sequence_length (int): 시퀀스 길이
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, scaler)
        """
        try:
            # 데이터 병합
            merged_data = self._merge_data(stock_data, sentiment_data, economic_data)
            
            # 특성과 타겟 분리
            features = merged_data.drop(['close'], axis=1)
            target = merged_data['close']
            
            # 스케일링
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.price_scaler.fit_transform(target.values.reshape(-1, 1))
            
            # 가격 범위 저장
            self.price_min = self.price_scaler.data_min_[0]
            self.price_max = self.price_scaler.data_max_[0]
            self.feature_names = features.columns.tolist()
            
            # 시퀀스 생성
            X, y = self._create_sequences(scaled_features, scaled_target, sequence_length)
            
            # 학습/검증 데이터 분할
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            return X_train, y_train, X_val, y_val, self
            
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {e}")
            raise
    
    def _merge_data(self, stock_data, sentiment_data, economic_data):
        """데이터 병합"""
        try:
            # 날짜 인덱스 설정
            stock_data = stock_data.set_index('date')
            sentiment_data = sentiment_data.set_index('date')
            economic_data = economic_data.set_index('date')
            
            # 데이터 병합
            merged = pd.merge(stock_data, sentiment_data, left_index=True, right_index=True, how='left')
            merged = pd.merge(merged, economic_data, left_index=True, right_index=True, how='left')
            
            # 결측치 처리
            merged = merged.fillna(method='ffill')
            merged = merged.fillna(method='bfill')
            
            return merged
            
        except Exception as e:
            print(f"데이터 병합 중 오류 발생: {e}")
            raise
    
    def _create_sequences(self, features, target, sequence_length):
        """시퀀스 데이터 생성"""
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
    
    def inverse_transform_price(self, scaled_price):
        """스케일된 가격을 원래 가격으로 변환"""
        return self.price_scaler.inverse_transform(scaled_price.reshape(-1, 1)).flatten()
    
    def prepare_prediction_data(self, stock_data, sentiment_data, economic_data, sequence_length=20):
        """
        예측을 위한 데이터 준비
        
        Args:
            stock_data (pd.DataFrame): 주가 데이터
            sentiment_data (pd.DataFrame): 감성 데이터
            economic_data (pd.DataFrame): 경제 데이터
            sequence_length (int): 시퀀스 길이
            
        Returns:
            np.array: 예측용 데이터
        """
        try:
            # 데이터 병합
            merged_data = self._merge_data(stock_data, sentiment_data, economic_data)
            
            # 특성 선택
            features = merged_data.drop(['close'], axis=1)
            
            # 스케일링
            scaled_features = self.feature_scaler.transform(features)
            
            # 마지막 시퀀스만 선택
            last_sequence = scaled_features[-sequence_length:]
            
            return np.array([last_sequence])
            
        except Exception as e:
            print(f"예측 데이터 준비 중 오류 발생: {e}")
            raise 