import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict
import logging
from datetime import datetime, timedelta
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPriceScaler:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.price_min = None
        self.price_max = None
        self.feature_names = None
        
    def fit_transform(self, data, price_cols):
        """데이터 스케일링"""
        # 가격 데이터 스케일링
        price_data = data[price_cols].values.reshape(-1, len(price_cols))
        scaled_prices = self.price_scaler.fit_transform(price_data)
        
        # 나머지 특성 스케일링
        feature_cols = [col for col in data.columns if col not in price_cols]
        feature_data = data[feature_cols].values
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        # 스케일링된 데이터 결합
        result = np.hstack([scaled_prices, scaled_features])
        
        # 범위 저장
        self.price_min = self.price_scaler.data_min_[0]
        self.price_max = self.price_scaler.data_max_[0]
        self.feature_names = data.columns.tolist()
        
        return result
    
    def inverse_transform_price(self, scaled_price):
        """스케일된 가격을 원래 가격으로 변환"""
        return self.price_scaler.inverse_transform(scaled_price.reshape(-1, 1)).flatten()

class DataProcessor:
    def __init__(self):
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
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 추가"""
        try:
            # 이동평균선
            for window in [5, 20, 60]:
                df[f'MA{window}'] = df['close'].rolling(window=window).mean()
            
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
            df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
            
            # Bollinger Bands
            df['BB_MIDDLE'] = df['close'].rolling(window=20).mean()
            df['BB_STD'] = df['close'].rolling(window=20).std()
            df['BB_UPPER'] = df['BB_MIDDLE'] + (df['BB_STD'] * 2)
            df['BB_LOWER'] = df['BB_MIDDLE'] - (df['BB_STD'] * 2)
            df['BB_PERCENT'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
            
            # 거래량 지표
            df['VOLUME_MA5'] = df['volume'].rolling(window=5).mean()
            df['VOLUME_MA20'] = df['volume'].rolling(window=20).mean()
            df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_MA20']
            
            # 모멘텀 지표
            df['MOM'] = df['close'].pct_change(periods=10)
            df['ROC'] = df['close'].pct_change(periods=20)
            
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 중 오류 발생: {str(e)}")
            raise
    
    def prepare_data(self, stock_data, sentiment_data, economic_data, sequence_length=20):
        """데이터 전처리 및 시퀀스 생성"""
        try:
            # 데이터 검증
            if stock_data.empty or sentiment_data.empty or economic_data.empty:
                raise ValueError("입력 데이터가 비어있습니다.")
            
            # 데이터 병합
            merged_data = self._merge_data(stock_data, sentiment_data, economic_data)
            
            # 결측치 처리
            merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
            if merged_data.isnull().any().any():
                raise ValueError("결측치가 남아있습니다.")
            
            # 기술적 지표 추가
            merged_data = self.add_technical_indicators(merged_data)
            
            # 특성 선택
            feature_columns = [
                'close', 'volume', 'market_cap', 'foreign_holding', 'foreign_ratio',
                'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
                'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_PERCENT',
                'MA5', 'MA20', 'MA60', 'VOLUME_MA5', 'VOLUME_MA20',
                'VOLUME_RATIO', 'MOM', 'ROC',
                'finbert_positive', 'finbert_negative', 'finbert_neutral',
                'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
            ]
            
            # 컬럼 존재 여부 확인
            missing_columns = [col for col in feature_columns if col not in merged_data.columns]
            if missing_columns:
                raise ValueError(f"다음 컬럼이 데이터에 없습니다: {missing_columns}")
            
            # 데이터 정규화
            price_cols = ['close']
            scaled_data = self.scaler.fit_transform(merged_data[feature_columns], price_cols)
            
            # 시퀀스 데이터 생성
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
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("시퀀스 데이터가 생성되지 않았습니다.")
            
            # 학습/검증/테스트 데이터 분할 (80/10/10)
            train_size = int(len(X) * 0.8)
            val_size = int(len(X) * 0.1)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            logger.info("\nData shapes after preparation:")
            logger.info(f"X_train: {X_train.shape}")
            logger.info(f"y_train: {y_train.shape}")
            logger.info(f"X_val: {X_val.shape}")
            logger.info(f"y_val: {y_val.shape}")
            logger.info(f"X_test: {X_test.shape}")
            logger.info(f"y_test: {y_test.shape}")
            
            return X_train, y_train, X_val, y_val, self.scaler
            
        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
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
            logger.error(f"데이터 병합 중 오류 발생: {str(e)}")
            raise
    
    def prepare_prediction_data(self, stock_data, sentiment_data, economic_data, sequence_length=20):
        """예측을 위한 데이터 준비"""
        try:
            # 데이터 병합
            merged_data = self._merge_data(stock_data, sentiment_data, economic_data)
            
            # 기술적 지표 추가
            merged_data = self.add_technical_indicators(merged_data)
            
            # 특성 선택
            feature_columns = [
                'close', 'volume', 'market_cap', 'foreign_holding', 'foreign_ratio',
                'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
                'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_PERCENT',
                'MA5', 'MA20', 'MA60', 'VOLUME_MA5', 'VOLUME_MA20',
                'VOLUME_RATIO', 'MOM', 'ROC',
                'finbert_positive', 'finbert_negative', 'finbert_neutral',
                'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
            ]
            
            # 데이터 정규화
            price_cols = ['close']
            scaled_data = self.scaler.fit_transform(merged_data[feature_columns], price_cols)
            
            # 마지막 시퀀스만 선택
            last_sequence = scaled_data[-sequence_length:]
            
            return np.array([last_sequence])
            
        except Exception as e:
            logger.error(f"예측 데이터 준비 중 오류 발생: {str(e)}")
            raise 