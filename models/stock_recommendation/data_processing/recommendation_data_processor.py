import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class RecommendationDataProcessor:
    def __init__(self):
        self.features = None
        self.returns_df = None
        self.volatility_df = None
        self.sentiment_df = None
        self.pred_score_df = None
        self.financial_df = None

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def calculate_returns(self, stock_prices: pd.DataFrame, stock_meta: pd.DataFrame):
        """수익률 계산"""
        try:
            returns = []
            for code in stock_meta['stock_code'].unique():
                stock_data = stock_prices[stock_prices['stock_code'] == code].sort_values('time')
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

    def calculate_volatility(self, stock_prices: pd.DataFrame, stock_meta: pd.DataFrame):
        """변동성 계산"""
        try:
            volatility = []
            for code in stock_meta['stock_code'].unique():
                stock_data = stock_prices[stock_prices['stock_code'] == code].sort_values('time')
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

    def aggregate_sentiment(self, news_sentiment: pd.DataFrame, stock_meta: pd.DataFrame):
        """감성분석 데이터 통합"""
        try:
            # 뉴스 제목에서 종목명 추출 및 매칭
            def extract_stock_name(title):
                # 종목명이 포함된 경우 해당 종목명 반환
                for stock_name in stock_meta['stock_name'].unique():
                    if stock_name in title:
                        return stock_name
                return None

            # 뉴스 제목에서 종목명 추출
            news_sentiment['stock_name'] = news_sentiment['title'].apply(extract_stock_name)
            
            # 종목명이 있는 뉴스만 필터링
            valid_news = news_sentiment[news_sentiment['stock_name'].notna()]
            
            # 감성 점수 계산
            sentiment_scores = valid_news.groupby('stock_name').agg({
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
            
            # stock_meta와 매칭
            self.sentiment_df = stock_meta[['stock_code', 'stock_name']].merge(
                sentiment_scores,
                on='stock_name',
                how='left'
            ).fillna({
                'finbert_positive': 0.5,
                'finbert_negative': 0.5,
                'finbert_neutral': 0.5,
                'sentiment_score': 0.5
            })
            
            # stock_name 컬럼 제거
            self.sentiment_df = self.sentiment_df.drop('stock_name', axis=1)
            
        except Exception as e:
            logger.error(f"감성분석 데이터 통합 중 오류 발생: {e}")
            raise

    def aggregate_price_predictions(self, price_predictions: pd.DataFrame, stock_meta: pd.DataFrame):
        """가격 예측 데이터 통합"""
        try:
            pred_df = price_predictions.copy()
            
            # 예측 수익률 계산
            pred_df['예측수익률'] = (pred_df['predicted_price'] - pred_df['actual_price']) / pred_df['actual_price'] * 100
            
            # 예측 신뢰도 계산 (예측 기간에 따른 가중치)
            pred_df['예측신뢰도'] = 1.0 - (pred_df['target_date'] - pred_df['prediction_date']).dt.days / 30
            
            # 가중 평균 예측 수익률 계산
            pred_score = pred_df.groupby('stock_code').apply(
                lambda x: np.average(x['예측수익률'], weights=x['예측신뢰도'])
            ).reset_index(name='예측수익률')
            
            self.pred_score_df = stock_meta[['stock_code']].merge(
                pred_score, on='stock_code', how='left'
            ).fillna({'예측수익률': 0})
        except Exception as e:
            logger.error(f"가격 예측 데이터 통합 중 오류 발생: {e}")
            raise

    def aggregate_financial_data(self, economic_indicators: pd.DataFrame, stock_meta: pd.DataFrame):
        """경제지표 데이터 통합"""
        try:
            # 최신 경제지표 데이터만 사용
            latest_indicators = economic_indicators.sort_values('time').groupby('time').last().reset_index()
            
            # 경제지표 정규화
            economic_metrics = ['treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y']
            for metric in economic_metrics:
                if metric in latest_indicators.columns:
                    # 이상치 제거
                    q1 = latest_indicators[metric].quantile(0.25)
                    q3 = latest_indicators[metric].quantile(0.75)
                    iqr = q3 - q1
                    latest_indicators[metric] = latest_indicators[metric].clip(
                        q1 - 1.5 * iqr,
                        q3 + 1.5 * iqr
                    )
            
            # 모든 종목에 대해 동일한 경제지표 적용
            self.financial_df = pd.DataFrame()
            self.financial_df['stock_code'] = stock_meta['stock_code']
            
            # 경제지표 매핑
            for metric in economic_metrics:
                if metric in latest_indicators.columns:
                    self.financial_df[metric] = latest_indicators[metric].iloc[-1]
            
            # 결측치 처리
            self.financial_df = self.financial_df.fillna({
                'treasury_10y': latest_indicators['treasury_10y'].median(),
                'dollar_index': latest_indicators['dollar_index'].median(),
                'usd_krw': latest_indicators['usd_krw'].median(),
                'korean_bond_10y': latest_indicators['korean_bond_10y'].median()
            })
            
        except Exception as e:
            logger.error(f"경제지표 데이터 통합 중 오류 발생: {e}")
            raise

    def process_financial_data(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """재무제표 데이터 전처리"""
        try:
            # 결측치 처리
            financial_data = financial_data.fillna({
                'per': financial_data['per'].median(),
                'roe': financial_data['roe'].median(),
                'pbr': financial_data['pbr'].median(),
                'ev': financial_data['ev'].median(),
                'bps': financial_data['bps'].median(),
                'sale_amt': financial_data['sale_amt'].median(),
                'bus_pro': financial_data['bus_pro'].median(),
                'cup_nga': financial_data['cup_nga'].median(),
                'cap': financial_data['cap'].median()
            })
            
            # 이상치 처리 (IQR 방식)
            for col in ['per', 'roe', 'pbr', 'ev', 'bps', 'sale_amt', 'bus_pro', 'cup_nga', 'cap']:
                Q1 = financial_data[col].quantile(0.25)
                Q3 = financial_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                financial_data[col] = financial_data[col].clip(lower_bound, upper_bound)
            
            # 재무비율 계산
            financial_data['profit_margin'] = financial_data['bus_pro'] / financial_data['sale_amt']
            financial_data['asset_turnover'] = financial_data['sale_amt'] / financial_data['cap']
            financial_data['financial_leverage'] = financial_data['cap'] / financial_data['bps']
            
            # 최신 데이터만 사용
            financial_data = financial_data.sort_values('created_at').groupby('stock_code').last().reset_index()
            
            # 종목 코드 형식 통일
            financial_data['stock_code'] = financial_data['stock_code'].astype(str).str.zfill(6)
            
            return financial_data
            
        except Exception as e:
            logger.error(f"재무제표 데이터 처리 중 오류 발생: {e}")
            raise

    def build_feature_dataset(self, stock_meta: pd.DataFrame, user_holdings: pd.DataFrame):
        """특징 데이터셋 구축"""
        try:
            # 기본 데이터 통합
            self.features = stock_meta.merge(
                self.returns_df, on='stock_code', how='left'
            ).merge(
                self.volatility_df, on='stock_code', how='left'
            ).merge(
                self.sentiment_df, on='stock_code', how='left'
            ).merge(
                self.pred_score_df, on='stock_code', how='left'
            )
            
            # 결측치 처리
            self.features = self.features.fillna({
                '1개월수익률': 0, '3개월수익률': 0,
                '변동성_1개월': 0, '변동성_3개월': 0,
                'sentiment_score': 0.5,
                '예측수익률': 0
            })
            
            # 보유자산 정보 추가
            self.features['보유여부'] = self.features['stock_code'].apply(
                lambda x: 1 if x in user_holdings['stockCode'].values else 0
            )
            
            # 평가손익률 매핑
            pl_rate_map = dict(zip(user_holdings['stockCode'], user_holdings['plRate'].astype(float)))
            self.features['보유평가손익률'] = self.features['stock_code'].map(pl_rate_map).fillna(0)
            
            # 재무제표 특징 추가
            financial_features = [
                'per', 'roe', 'pbr', 'ev', 'bps',
                'sale_amt', 'bus_pro', 'cup_nga', 'cap',
                'profit_margin', 'asset_turnover', 'financial_leverage'
            ]
            
            self.features = self.features.merge(
                self.financial_df[['stock_code'] + financial_features],
                on='stock_code',
                how='left'
            )
            
            # 재무제표 결측치 처리
            self.features = self.features.fillna({
                'per': self.features['per'].median(),
                'roe': self.features['roe'].median(),
                'pbr': self.features['pbr'].median(),
                'ev': self.features['ev'].median(),
                'bps': self.features['bps'].median(),
                'sale_amt': self.features['sale_amt'].median(),
                'bus_pro': self.features['bus_pro'].median(),
                'cup_nga': self.features['cup_nga'].median(),
                'cap': self.features['cap'].median(),
                'profit_margin': self.features['profit_margin'].median(),
                'asset_turnover': self.features['asset_turnover'].median(),
                'financial_leverage': self.features['financial_leverage'].median()
            })
            
            # 특징 정규화
            self.normalize_features()
            
            return self.features
            
        except Exception as e:
            logger.error(f"특징 데이터셋 구축 중 오류 발생: {e}")
            raise

    def normalize_features(self):
        """특징 정규화"""
        try:
            # 기존 특징 정규화
            for feature in ['1개월수익률', '변동성_1개월', 'sentiment_score', '예측수익률', '보유평가손익률']:
                min_val = self.features[feature].min()
                max_val = self.features[feature].max()
                if max_val - min_val > 0:
                    self.features[f'{feature}_norm'] = (self.features[feature] - min_val) / (max_val - min_val)
                else:
                    self.features[f'{feature}_norm'] = 0
            
            # 재무제표 특징 정규화
            financial_features = [
                'per', 'roe', 'pbr', 'ev', 'bps',
                'sale_amt', 'bus_pro', 'cup_nga', 'cap',
                'profit_margin', 'asset_turnover', 'financial_leverage'
            ]
            
            for feature in financial_features:
                min_val = self.features[feature].min()
                max_val = self.features[feature].max()
                if max_val - min_val > 0:
                    self.features[f'{feature}_norm'] = (self.features[feature] - min_val) / (max_val - min_val)
                else:
                    self.features[f'{feature}_norm'] = 0
                
        except Exception as e:
            logger.error(f"특징 정규화 중 오류 발생: {e}")
            raise

    def process(self, data_dict):
        """전체 데이터 처리 파이프라인 및 데이터 분할"""
        try:
            # 기술적 지표 추가
            stock_prices = self.add_technical_indicators(data_dict['stock_prices'])
            
            # 수익률 계산
            self.calculate_returns(stock_prices, data_dict['stock_meta'])
            
            # 변동성 계산
            self.calculate_volatility(stock_prices, data_dict['stock_meta'])
            
            # 감성분석 데이터 통합
            self.aggregate_sentiment(data_dict['news_sentiment'], data_dict['stock_meta'])
            
            # 가격 예측 데이터 통합
            self.aggregate_price_predictions(data_dict['price_predictions'], data_dict['stock_meta'])
            
            # 재무제표 데이터 처리
            self.financial_df = self.process_financial_data(data_dict['financial_data'])
            
            # 특징 데이터셋 구축
            features = self.build_feature_dataset(data_dict['stock_meta'], data_dict['user_holdings'])

            # feature/label 분리
            feature_cols = [
                '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm', '예측수익률_norm', '보유평가손익률_norm',
                'per_norm', 'roe_norm', 'pbr_norm', 'ev_norm', 'bps_norm',
                'sale_amt_norm', 'bus_pro_norm', 'cup_nga_norm', 'cap_norm',
                'profit_margin_norm', 'asset_turnover_norm', 'financial_leverage_norm'
            ]
            label_col = '1개월수익률'  # 예측 타깃
            X = features[feature_cols].values
            y = features[label_col].values

            # 데이터 분할 (train/val/test = 7:2:1)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'features': features  # features DataFrame도 반환
            }
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {e}")
            raise 