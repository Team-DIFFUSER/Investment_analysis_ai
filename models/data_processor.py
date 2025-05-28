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
            
            logger.info("데이터 전처리 시작")
            logger.info(f"주가 데이터 크기: {stock_data.shape}")
            logger.info(f"감성 데이터 크기: {sentiment_data.shape}")
            logger.info(f"경제 데이터 크기: {economic_data.shape}")
            
            # 데이터 병합
            merged_data = self._merge_data(stock_data, sentiment_data, economic_data)
            
            # 기술적 지표 추가
            logger.info("기술적 지표 추가 중...")
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
                logger.error(f"누락된 컬럼: {missing_columns}")
                raise ValueError(f"다음 컬럼이 데이터에 없습니다: {missing_columns}")
            
            # 데이터 정규화 전 결측치 확인
            logger.info("데이터 정규화 전 결측치 확인:")
            null_counts = merged_data[feature_columns].isnull().sum()
            for col in feature_columns:
                if null_counts[col] > 0:
                    logger.warning(f"{col}: {null_counts[col]}개의 결측치")
            
            # 데이터 정규화
            logger.info("데이터 정규화 중...")
            price_cols = ['close']
            scaled_data = self.scaler.fit_transform(merged_data[feature_columns], price_cols)
            
            # 시퀀스 데이터 생성
            logger.info("시퀀스 데이터 생성 중...")
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
            
            logger.info("\n데이터 준비 완료:")
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
            
            # 데이터 병합 전 결측치 확인
            logger.info("데이터 병합 전 결측치 확인:")
            logger.info(f"주가 데이터 결측치: {stock_data.isnull().sum().sum()}")
            logger.info(f"감성 데이터 결측치: {sentiment_data.isnull().sum().sum()}")
            logger.info(f"경제 데이터 결측치: {economic_data.isnull().sum().sum()}")
            
            # 데이터 병합
            merged = pd.merge(stock_data, sentiment_data, left_index=True, right_index=True, how='left')
            merged = pd.merge(merged, economic_data, left_index=True, right_index=True, how='left')
            
            # 결측치 처리
            # 1. 숫자형 컬럼은 0으로 채우기
            numeric_columns = merged.select_dtypes(include=[np.number]).columns
            merged[numeric_columns] = merged[numeric_columns].fillna(0)
            
            # 2. 시계열 특성을 고려한 전진 채우기
            merged = merged.fillna(method='ffill')
            
            # 3. 남은 결측치는 이전 값으로 채우기
            merged = merged.fillna(method='bfill')
            
            # 4. 그래도 남은 결측치는 0으로 채우기
            merged = merged.fillna(0)
            
            # 결측치 처리 후 확인
            remaining_nulls = merged.isnull().sum()
            if remaining_nulls.any():
                logger.warning("남아있는 결측치:")
                for col in remaining_nulls[remaining_nulls > 0].index:
                    logger.warning(f"{col}: {remaining_nulls[col]}개")
                raise ValueError("결측치 처리 후에도 결측치가 남아있습니다.")
            
            # 데이터 정렬
            merged = merged.sort_index()
            
            # 중복 제거
            merged = merged[~merged.index.duplicated(keep='first')]
            
            logger.info("데이터 병합 완료")
            logger.info(f"최종 데이터 크기: {merged.shape}")
            
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