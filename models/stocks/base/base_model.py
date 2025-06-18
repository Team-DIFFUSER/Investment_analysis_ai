import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
from database.database import DatabaseManager

class BaseStockModel:
    def __init__(self, stock_name: str, symbol: str):
        """기본 주식 모델 초기화"""
        self.stock_name = stock_name
        self.symbol = symbol
        self.db_manager = DatabaseManager()
        self.logger = logging.getLogger(f'models.stocks.{self.stock_name.lower()}')
        
        # 모델 하이퍼파라미터
        self.sequence_length = 20
        self.batch_size = 32
        self.learning_rate = 0.001
        
        # 데이터 스케일러
        self.scaler = MinMaxScaler()
        
        # 예측 이력 저장
        self.predictions_history = {}
        self.error_history = {}
        
        # 초기화 상태
        self._initialized = False
        
        # GPU 메모리 설정
        self._setup_gpu()
        
        # 모델 초기화
        self.model = None
        
    def _setup_gpu(self):
        """GPU 메모리 설정"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info("GPU 메모리 설정 완료")
            except RuntimeError as e:
                self.logger.warning(f"GPU 메모리 설정 실패: {str(e)}")
                
    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        try:
            # 데이터베이스에서 데이터 로드
            data = self.db_manager.get_stock_data(self.symbol)
            if data.empty:
                self.logger.warning("데이터베이스에서 데이터를 찾을 수 없습니다. Yahoo Finance에서 다운로드합니다.")
                data = yf.download(self.symbol, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
                if not data.empty:
                    self.db_manager.save_stock_data(self.symbol, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return pd.DataFrame()
            
    def enhanced_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """향상된 데이터 전처리"""
        try:
            # 기본 전처리
            df = data.copy()
            df = df.dropna()
            
            # 기술적 지표 추가
            df['MA5'] = df['close_price'].rolling(window=5).mean()
            df['MA20'] = df['close_price'].rolling(window=20).mean()
            df['MA60'] = df['close_price'].rolling(window=60).mean()
            
            # 변동성 지표
            df['Volatility'] = df['close_price'].pct_change().rolling(window=20).std()
            
            # 거래량 관련 지표
            df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
            df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
            
            # 가격 변화율
            df['Price_Change'] = df['close_price'].pct_change()
            df['Price_Change_MA5'] = df['Price_Change'].rolling(window=5).mean()
            
            # RSI 계산
            delta = df['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD 계산
            exp1 = df['close_price'].ewm(span=12, adjust=False).mean()
            exp2 = df['close_price'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            return pd.DataFrame()
            
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """학습 데이터 준비"""
        try:
            # 특성 선택
            features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 
                       'MA5', 'MA20', 'MA60', 'Volatility',
                       'Volume_MA5', 'Volume_MA20', 'Price_Change',
                       'Price_Change_MA5', 'RSI', 'MACD', 'Signal_Line']
            
            # 데이터 스케일링
            scaled_data = self.scaler.fit_transform(data[features])
            
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length, 3])  # close_price
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"데이터 준비 중 오류 발생: {str(e)}")
            return np.array([]), np.array([])
            
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """모델 구축"""
        try:
            model = models.Sequential([
                layers.LSTM(128, input_shape=input_shape, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(64, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"모델 구축 중 오류 발생: {str(e)}")
            return None
            
    def save_model(self):
        """모델 저장"""
        try:
            # 메인 모델 저장
            model_path = os.path.join('models', 'checkpoints', f'{self.stock_name}_model.h5')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            self.logger.info(f"모델 저장 완료: {model_path}")
            
            # 백업 저장
            backup_path = os.path.join('models', 'backup', f'{self.stock_name}_model.h5')
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            self.model.save(backup_path)
            self.logger.info(f"모델 백업 저장 완료: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            
    def load_model(self) -> bool:
        """모델 로드"""
        try:
            # 프로젝트 루트 디렉토리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            
            # 모델 파일 경로
            model_path = os.path.join(project_root, 'models', 'checkpoints', f'{self.stock_name}_model.h5')
            backup_path = os.path.join(project_root, 'models', 'backup', f'{self.stock_name}_model.h5')
            
            self.logger.info(f"모델 파일 검색: {model_path}")
            
            # 메인 모델 로드 시도
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info("모델 로드 성공")
                return True
                
            # 백업 모델 로드 시도
            if os.path.exists(backup_path):
                self.model = tf.keras.models.load_model(backup_path)
                self.logger.info("백업 모델 로드 성공")
                return True
                
            self.logger.warning("저장된 모델을 찾을 수 없습니다.")
            return False
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False
            
    def update_predictions(self, date: str, actual_price: float):
        """예측값 업데이트"""
        try:
            if date in self.predictions_history:
                prediction = self.predictions_history[date]
                error = abs(prediction - actual_price) / actual_price
                self.error_history[date] = error
                self.logger.info(f"예측값 업데이트 완료: {date}, 오차율: {error:.2%}")
                
        except Exception as e:
            self.logger.error(f"예측값 업데이트 중 오류 발생: {str(e)}")
            
    def get_trading_dates(self, start_date: str, days: int = 5) -> List[str]:
        """거래일 목록 조회"""
        try:
            dates = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            count = 0
            
            while len(dates) < days:
                current_date += timedelta(days=1)
                if current_date.weekday() < 5:  # 주말 제외
                    dates.append(current_date.strftime('%Y-%m-%d'))
                    
            return dates
            
        except Exception as e:
            self.logger.error(f"거래일 목록 조회 중 오류 발생: {str(e)}")
            return []
            
    def is_initialized(self) -> bool:
        """모델 초기화 상태 확인"""
        return self._initialized and self.model is not None
        
    def initialize(self):
        """모델 초기화"""
        try:
            # 모델 로드 시도
            if self.load_model():
                self._initialized = True
                self.logger.info(f"{self.stock_name} 모델 초기화 완료")
                return
            
            # 모델이 없으면 학습 수행
            self.logger.warning(f"{self.stock_name} 저장된 모델이 없습니다. 학습을 시작합니다.")
            self.train()
            
            # 학습 후 모델 로드
            if self.load_model():
                self._initialized = True
                self.logger.info(f"{self.stock_name} 모델 학습 및 초기화 완료")
            else:
                raise ValueError(f"{self.stock_name} 모델 초기화 실패")
                
        except Exception as e:
            self.logger.error(f"{self.stock_name} 모델 초기화 중 오류 발생: {str(e)}")
            self._initialized = False
            raise
            
    def train(self):
        """모델 학습"""
        try:
            # 데이터 로드
            data = self.load_data()
            if data.empty:
                raise ValueError("학습 데이터가 없습니다.")
            
            # 데이터 전처리
            processed_data = self.enhanced_preprocessing(data)
            if processed_data.empty:
                raise ValueError("데이터 전처리 실패")
            
            # 학습 데이터 준비
            X, y = self.prepare_data(processed_data)
            if len(X) == 0 or len(y) == 0:
                raise ValueError("학습 데이터 준비 실패")
            
            # 모델 구축
            self.model = self.build_model((self.sequence_length, X.shape[2]))
            if self.model is None:
                raise ValueError("모델 구축 실패")
            
            # 모델 학습
            self.model.fit(
                X, y,
                epochs=50,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            # 모델 저장
            self.save_model()
            self.logger.info(f"{self.stock_name} 모델 학습 완료")
            
        except Exception as e:
            self.logger.error(f"{self.stock_name} 모델 학습 중 오류 발생: {str(e)}")
            raise 