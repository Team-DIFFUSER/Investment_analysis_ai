import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import openai
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

# --- DB 연결 함수들 ---

def get_mongo_user_investment_type(username):
    """MongoDB에서 투자성향 정보 가져오기"""
    mongo_uri = os.environ["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client[os.environ["MONGO_DB_NAME"]]
    user = db[os.environ["MONGO_COLLECTION_NAME"]].find_one({'username': username})
    if user and 'investmentType' in user:
        return user['investmentType']
    return None

def get_mongo_user_holdings(username):
    """MongoDB에서 보유자산 정보 가져오기"""
    mongo_uri = os.environ["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client[os.environ["MONGO_DB_NAME"]]
    user = db[os.environ["MONGO_COLLECTION_NAME"]].find_one({'username': username})
    if user and 'evltData' in user:
        # evltData는 리스트, 각 항목은 dict
        return pd.DataFrame(user['evltData'])
    return pd.DataFrame(columns=['stockCode', 'name', 'quantity', 'avgPrice', 'currentPrice', 'evalAmount', 'plAmount', 'plRate'])

def get_timescale_conn():
    """TimescaleDB 연결"""
    conn = psycopg2.connect(
        host=os.environ["TS_HOST"],
        port=os.environ["TS_PORT"],
        dbname=os.environ["TS_DB"],
        user=os.environ["TS_USER"],
        password=os.environ["TS_PASSWORD"],
        sslmode=os.environ["TS_SSL_MODE"]
    )
    return conn

def load_stock_meta():
    """TimescaleDB에서 종목 메타데이터 로드"""
    with get_timescale_conn() as conn:
        df = pd.read_sql("SELECT * FROM stock_items", conn)
    return df

def load_stock_prices():
    """TimescaleDB에서 종목 가격 데이터 로드"""
    with get_timescale_conn() as conn:
        df = pd.read_sql("SELECT * FROM stock_prices", conn)
    df['time'] = pd.to_datetime(df['time'])
    return df

def load_news_sentiment():
    """TimescaleDB에서 뉴스 감성분석 데이터 로드"""
    with get_timescale_conn() as conn:
        df = pd.read_sql("SELECT * FROM news_sentiment", conn)
    return df

def load_price_predictions():
    """TimescaleDB에서 시계열 예측 데이터 로드"""
    with get_timescale_conn() as conn:
        df = pd.read_sql("SELECT * FROM predicted_stock_prices", conn)
    return df

def load_financial_data():
    """TimescaleDB에서 재무데이터 로드"""
    with get_timescale_conn() as conn:
        df = pd.read_sql("SELECT * FROM financial_data", conn)
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """주요 기술적 지표 추가"""
    try:
        # RSI (과매수/과매도 판단)
        rsi = RSIIndicator(close=df['close_price'], window=14)
        df['RSI'] = rsi.rsi()

        # 이동평균 (추세 판단)
        df['MA5'] = SMAIndicator(close=df['close_price'], window=5).sma_indicator()
        df['MA20'] = SMAIndicator(close=df['close_price'], window=20).sma_indicator()
        
        # 거래량 지표 (시장 관심도)
        df['VOLUME_MA5'] = SMAIndicator(close=df['volume'], window=5).sma_indicator()
        df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_MA5']

        return df
    except Exception as e:
        logger.error(f"기술적 지표 추가 중 오류 발생: {e}")
        raise

# --- MLP 신경망 ---

class StockMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 동적 레이어 생성
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 출력 레이어
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class StockRecommendationModelV3:
    def __init__(self, username):
        self.username = username
        self.investment_type = get_mongo_user_investment_type(username)
        logger.info(f"{username}의 투자성향: {self.investment_type}")

        # 데이터 로드
        self.stock_meta = load_stock_meta()
        self.stock_prices = load_stock_prices()
        self.news_sentiment = load_news_sentiment()
        self.price_predictions = load_price_predictions()
        self.financial_data = load_financial_data()
        self.user_holdings = get_mongo_user_holdings(username)
        
        # MLP 모델 초기화
        self.mlp_model = None
        self.mlp_feature_cols = None
        
        # 데이터 전처리
        self.preprocess_data()

    def preprocess_data(self):
        """데이터 전처리 파이프라인"""
        try:
            # 기술적 지표 추가
            self.stock_prices = add_technical_indicators(self.stock_prices)
            
            # 수익률 계산
            self.calculate_returns()
            
            # 변동성 계산
            self.calculate_volatility()
            
            # 감성분석 데이터 통합
            self.aggregate_sentiment()
            
            # 가격 예측 데이터 통합
            self.aggregate_price_predictions()
            
            # 재무데이터 통합
            self.aggregate_financial_data()
            
            # 특징 데이터셋 구축
            self.build_feature_dataset()
            
            logger.info("데이터 전처리 완료")
        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {e}")
            raise

    def calculate_returns(self):
        """수익률 계산"""
        try:
            returns = []
            for code in self.stock_meta['stock_code'].unique():
                stock_data = self.stock_prices[self.stock_prices['stock_code'] == code].sort_values('time')
                if len(stock_data) >= 21:
                    # 단기 수익률 (1개월)
                    recent_return = (stock_data['close_price'].iloc[-1] / stock_data['close_price'].iloc[-21] - 1) * 100
                    # 중기 수익률 (3개월)
                    mid_return = (stock_data['close_price'].iloc[-1] / stock_data['close_price'].iloc[-63] - 1) * 100
                    
                    returns.append({
                        'stock_code': code,
                        '1개월수익률': recent_return,
                        '3개월수익률': mid_return
                    })
            self.returns_df = pd.DataFrame(returns)
        except Exception as e:
            logger.error(f"수익률 계산 중 오류 발생: {e}")
            raise

    def calculate_volatility(self):
        """변동성 계산"""
        try:
            volatility = []
            for code in self.stock_meta['stock_code'].unique():
                stock_data = self.stock_prices[self.stock_prices['stock_code'] == code].sort_values('time')
                if len(stock_data) >= 21:
                    # 일간 수익률
                    daily_returns = stock_data['close_price'].pct_change().dropna()
                    
                    # 단기 변동성 (1개월)
                    vol_1m = daily_returns.tail(21).std() * np.sqrt(252)
                    # 중기 변동성 (3개월)
                    vol_3m = daily_returns.tail(63).std() * np.sqrt(252)
                    
                    volatility.append({
                        'stock_code': code,
                        '변동성_1개월': vol_1m,
                        '변동성_3개월': vol_3m
                    })
            self.volatility_df = pd.DataFrame(volatility)
        except Exception as e:
            logger.error(f"변동성 계산 중 오류 발생: {e}")
            raise

    def aggregate_sentiment(self):
        """감성분석 데이터 통합"""
        try:
            # 감성 점수 계산
            sentiment_scores = self.news_sentiment.groupby('stock_code').agg({
                'finbert_positive': 'mean',
                'finbert_negative': 'mean',
                'finbert_neutral': 'mean'
            }).reset_index()
            
            # 최종 감성 점수 계산 (가중치 적용)
            sentiment_scores['sentiment_score'] = (
                sentiment_scores['finbert_positive'] * 0.5 +
                sentiment_scores['finbert_neutral'] * 0.3 -
                sentiment_scores['finbert_negative'] * 0.2
            )
            
            self.sentiment_df = self.stock_meta[['stock_code']].merge(
                sentiment_scores,
                on='stock_code',
                how='left'
            ).fillna({
                'finbert_positive': 0.5,
                'finbert_negative': 0.5,
                'finbert_neutral': 0.5,
                'sentiment_score': 0.5
            })
        except Exception as e:
            logger.error(f"감성분석 데이터 통합 중 오류 발생: {e}")
            raise

    def aggregate_price_predictions(self):
        """가격 예측 데이터 통합"""
        try:
            pred_df = self.price_predictions.copy()
            
            # 예측 수익률 계산
            pred_df['예측수익률'] = (pred_df['predicted_price'] - pred_df['close_price']) / pred_df['close_price'] * 100
            
            # 예측 신뢰도 계산 (예측 기간에 따른 가중치)
            pred_df['예측신뢰도'] = 1.0 - (pred_df['target_date'] - pred_df['prediction_date']).dt.days / 30
            
            # 가중 평균 예측 수익률 계산
            pred_score = pred_df.groupby('stock_code').apply(
                lambda x: np.average(x['예측수익률'], weights=x['예측신뢰도'])
            ).reset_index(name='예측수익률')
            
            self.pred_score_df = self.stock_meta[['stock_code']].merge(
                pred_score, on='stock_code', how='left'
            ).fillna({'예측수익률': 0})
        except Exception as e:
            logger.error(f"가격 예측 데이터 통합 중 오류 발생: {e}")
            raise

    def aggregate_financial_data(self):
        """재무제표 데이터 통합"""
        try:
            # 재무제표 데이터 전처리
            financial_df = self.financial_data.copy()
            
            # 결측치 처리
            financial_df = financial_df.fillna({
                'per': financial_df['per'].median(),
                'roe': financial_df['roe'].median(),
                'pbr': financial_df['pbr'].median(),
                'ev': financial_df['ev'].median(),
                'bps': financial_df['bps'].median(),
                'sale_amt': financial_df['sale_amt'].median(),
                'bus_pro': financial_df['bus_pro'].median(),
                'cup_nga': financial_df['cup_nga'].median(),
                'cap': financial_df['cap'].median()
            })
            
            # 이상치 처리 (IQR 방식)
            for col in ['per', 'roe', 'pbr', 'ev', 'bps', 'sale_amt', 'bus_pro', 'cup_nga', 'cap']:
                Q1 = financial_df[col].quantile(0.25)
                Q3 = financial_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                financial_df[col] = financial_df[col].clip(lower_bound, upper_bound)
            
            # 재무비율 계산
            financial_df['profit_margin'] = financial_df['bus_pro'] / financial_df['sale_amt']
            financial_df['asset_turnover'] = financial_df['sale_amt'] / financial_df['cap']
            financial_df['financial_leverage'] = financial_df['cap'] / financial_df['bps']
            
            # 최신 데이터만 사용
            financial_df = financial_df.sort_values('created_at').groupby('stock_code').last().reset_index()
            
            # 종목 코드 형식 통일
            financial_df['stock_code'] = financial_df['stock_code'].astype(str).str.zfill(6)
            
            self.financial_data = financial_df
            logger.info("재무제표 데이터 통합 완료")
            
        except Exception as e:
            logger.error(f"재무제표 데이터 통합 중 오류 발생: {e}")
            raise

    def build_feature_dataset(self):
        """특징 데이터셋 구축"""
        try:
            # 기본 특징
            feature_cols = [
                'RSI', 'MA5', 'MA20', 'VOLUME_MA5', 'VOLUME_RATIO',
                'return_1d', 'return_5d', 'return_20d',
                'volatility_5d', 'volatility_20d',
                'sentiment_score', 'sentiment_volume',
                'price_prediction_1d', 'price_prediction_5d', 'price_prediction_20d'
            ]
            
            # 재무제표 특징 추가
            financial_features = [
                'per', 'roe', 'pbr', 'ev', 'bps',
                'sale_amt', 'bus_pro', 'cup_nga', 'cap',
                'profit_margin', 'asset_turnover', 'financial_leverage'
            ]
            feature_cols.extend(financial_features)
            
            # 데이터 통합
            df = self.stock_prices.merge(
                self.news_sentiment[['stock_code', 'sentiment_score', 'sentiment_volume']],
                on='stock_code',
                how='left'
            ).merge(
                self.price_predictions[['stock_code', 'price_prediction_1d', 'price_prediction_5d', 'price_prediction_20d']],
                on='stock_code',
                how='left'
            ).merge(
                self.financial_data[['stock_code'] + financial_features],
                on='stock_code',
                how='left'
            )
            
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            self.mlp_feature_cols = feature_cols
            return df[feature_cols]
            
        except Exception as e:
            logger.error(f"특징 데이터셋 구축 중 오류 발생: {e}")
            raise

    def normalize_features(self):
        """특징 정규화"""
        for feature in ['1개월수익률', '변동성_1개월', 'sentiment_score', '예측수익률', 
                       '보유평가손익률', 'per', 'pbr', 'roe', 'debt_ratio']:
            min_val = self.features[feature].min()
            max_val = self.features[feature].max()
            if max_val - min_val > 0:
                self.features[f'{feature}_norm'] = (self.features[feature] - min_val) / (max_val - min_val)
            else:
                self.features[f'{feature}_norm'] = 0

    def get_risk_weight(self):
        """투자 성향별 가중치 설정"""
        if self.investment_type == "적극투자형":
            return {
                "수익률": 0.35, "변동성": 0.1, "감성": 0.15, "예측": 0.15,
                "보유손익": 0.1, "재무": 0.15
            }
        elif self.investment_type == "위험중립형":
            return {
                "수익률": 0.25, "변동성": 0.2, "감성": 0.15, "예측": 0.1,
                "보유손익": 0.1, "재무": 0.2
            }
        elif self.investment_type == "안정추구형":
            return {
                "수익률": 0.2, "변동성": 0.25, "감성": 0.15, "예측": 0.1,
                "보유손익": 0.1, "재무": 0.2
            }
        else:
            return {
                "수익률": 0.25, "변동성": 0.2, "감성": 0.15, "예측": 0.1,
                "보유손익": 0.1, "재무": 0.2
            }

    def train_mlp(self, target_col='1개월수익률', feature_cols=None, epochs=1000, lr=0.001, batch_size=32):
        """MLP 모델 학습"""
        try:
            if feature_cols is None:
                feature_cols = [
                    '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm', '예측수익률_norm',
                    '보유평가손익률_norm', 'per_norm', 'pbr_norm', 'roe_norm', 'debt_ratio_norm'
                ]
            
            # 데이터 준비
            X = self.features[feature_cols].values.astype(np.float32)
            y = self.features[target_col].values.astype(np.float32)
            
            # 데이터 분할 (80% 학습, 20% 검증)
            train_size = int(0.8 * len(X))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # 데이터셋 생성
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train),
                torch.tensor(y_train).unsqueeze(1)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val),
                torch.tensor(y_val).unsqueeze(1)
            )
            
            # 데이터 로더 생성
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )
            
            # 모델 초기화
            self.mlp_model = StockMLP(input_dim=X.shape[1])
            self.mlp_feature_cols = feature_cols
            
            # 옵티마이저와 학습률 스케줄러 설정
            optimizer = optim.Adam(self.mlp_model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # 손실 함수
            loss_fn = nn.MSELoss()
            
            # 조기 종료 설정
            early_stopping = EarlyStopping(patience=10)
            
            # 학습 루프
            best_val_loss = float('inf')
            for epoch in range(epochs):
                # 학습
                self.mlp_model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    pred = self.mlp_model(X_batch)
                    loss = loss_fn(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 검증
                self.mlp_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        pred = self.mlp_model(X_batch)
                        val_loss += loss_fn(pred, y_batch).item()
                
                # 평균 손실 계산
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                # 학습률 조정
                scheduler.step(val_loss)
                
                # 조기 종료 확인
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    logger.info(f"조기 종료: {epoch} 에포크")
                    break
                
                # 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.mlp_model.state_dict()
                
                if epoch % 100 == 0:
                    logger.info(f"에포크 {epoch}: 학습 손실 = {train_loss:.4f}, 검증 손실 = {val_loss:.4f}")
            
            # 최적의 모델 상태 복원
            self.mlp_model.load_state_dict(best_model_state)
            logger.info("MLP 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"MLP 모델 학습 중 오류 발생: {e}")
            raise

    def evaluate_mlp(self):
        """MLP 모델 평가"""
        try:
            if self.mlp_model is None:
                raise ValueError("MLP 모델이 학습되지 않았습니다.")
            
            X = self.features[self.mlp_feature_cols].values.astype(np.float32)
            y = self.features['1개월수익률'].values.astype(np.float32)
            
            # 예측
            self.mlp_model.eval()
            with torch.no_grad():
                pred = self.mlp_model(torch.tensor(X)).squeeze().numpy()
            
            # 평가 지표 계산
            mae = np.mean(np.abs(pred - y))
            rmse = np.sqrt(np.mean((pred - y) ** 2))
            mape = np.mean(np.abs((y - pred) / y)) * 100
            
            # 방향 정확도 계산
            direction_accuracy = np.mean((pred * y) > 0)
            
            # 신뢰도 점수 계산
            confidence_scores = 1 / (1 + np.abs(pred - y))
            
            # 결과 저장
            self.features['MLP_예측'] = pred
            self.features['MLP_신뢰도'] = confidence_scores
            
            evaluation_metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                '방향정확도': direction_accuracy
            }
            
            logger.info("MLP 모델 평가 완료")
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"MLP 모델 평가 중 오류 발생: {e}")
            raise

    def get_recommendations(self, top_n=7):
        """종목 추천"""
        try:
            # MLP 예측 점수 계산
            if self.mlp_model is not None:
                self.evaluate_mlp()
                mlp_weight = 0.4  # MLP 모델의 가중치
            else:
                mlp_weight = 0
            
            weights = self.get_risk_weight()
            
            # 최종 점수 계산
            self.features['최종점수'] = (
                (1 - mlp_weight) * (
                    weights['수익률'] * self.features['1개월수익률_norm'] +
                    weights['변동성'] * (1 - self.features['변동성_1개월_norm']) +
                    weights['감성'] * self.features['sentiment_score_norm'] +
                    weights['예측'] * self.features['예측수익률_norm'] +
                    weights['보유손익'] * self.features['보유평가손익률_norm'] +
                    weights['재무'] * (
                        self.features['per_norm'] +
                        self.features['pbr_norm'] +
                        self.features['roe_norm'] +
                        (1 - self.features['debt_ratio_norm'])
                    ) / 4
                ) +
                mlp_weight * self.features['MLP_예측']
            ) * 100
            
            # 상위 종목 선정
            top_stocks = self.features.sort_values('최종점수', ascending=False).head(top_n)
            
            # 추천 결과 생성
            recommendations = []
            for _, row in top_stocks.iterrows():
                recommendation = {
                    '종목코드': row['stock_code'],
                    '종목명': row['stock_name'],
                    '최종점수': row['최종점수'],
                    '주요팩터': {
                        '1개월수익률': row['1개월수익률'],
                        '변동성': row['변동성_1개월'],
                        '감성점수': row['sentiment_score'],
                        '예측수익률': row['예측수익률'],
                        '보유평가손익률': row['보유평가손익률'],
                        'PER': row['per'],
                        'PBR': row['pbr'],
                        'ROE': row['roe'],
                        '부채비율': row['debt_ratio'],
                        '보유여부': row['보유여부']
                    }
                }
                
                # MLP 모델이 있는 경우 신뢰도 점수 추가
                if self.mlp_model is not None:
                    recommendation['MLP_신뢰도'] = row['MLP_신뢰도']
                
                recommendation['추천이유'] = self.generate_explanation(row)
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"종목 추천 중 오류 발생: {e}")
            raise

    def generate_explanation(self, row):
        """추천 이유 생성"""
        prompt = f"""
        [종목 정보]
        - 종목명: {row['stock_name']}
        - 1개월 수익률: {row['1개월수익률']:.2f}%
        - 변동성: {row['변동성_1개월']:.2f}
        - 뉴스 감성점수: {row['sentiment_score']:.2f}
        - 평가손익률: {row['보유평가손익률']:.2f}
        - PER: {row['per']:.2f}
        - PBR: {row['pbr']:.2f}
        - ROE: {row['roe']:.2f}%
        - 부채비율: {row['debt_ratio']:.2f}%
        [투자자 성향] {self.investment_type}
        [요청]
        위 정보를 참고해 이 종목의 투자 매력과 추천 전략을 2~3문장으로 설명해 주세요.
        """
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {"role": "system", "content": "당신은 금융 전문가입니다. 데이터를 바탕으로 설득력 있는 투자 추천 이유를 작성하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT API 오류: {e}")
            return (
                f"추천 이유: 감성점수({row['sentiment_score']:.2f}), "
                f"변동성({row['변동성_1개월']:.2f}), "
                f"1개월수익률({row['1개월수익률']:.2f}%), "
                f"재무지표(PER:{row['per']:.2f}, PBR:{row['pbr']:.2f}, ROE:{row['roe']:.2f}%) 등 종합 고려"
            )

# --- 사용 예시 ---

if __name__ == "__main__":
    username = "JunOh"  # 실제 로그인 사용자명 사용
    model = StockRecommendationModelV3(username)
    
    # MLP 모델 학습
    model.train_mlp(
        target_col='1개월수익률',
        feature_cols=[
            '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm', '예측수익률_norm',
            '보유평가손익률_norm', 'per_norm', 'pbr_norm', 'roe_norm', 'debt_ratio_norm'
        ]
    )
    
    # 모델 평가
    evaluation_metrics = model.evaluate_mlp()
    print("\n모델 평가 지표:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 종목 추천
    recommendations = model.get_recommendations(top_n=7)
    print("\n추천 종목:")
    for rec in recommendations:
        print(rec)
