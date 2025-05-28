import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import openai
import os
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리의 .env 파일 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
load_dotenv(os.path.join(project_root, '.env'))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class StockMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class StockRecommendationModelV2:
    def __init__(self):
        # 데이터 로드
        self.load_data()
        # 데이터 전처리 및 feature 생성
        self.preprocess_data()

    def load_data(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
        # 종목 메타/가격/뉴스 데이터 로드
        self.stock_meta = pd.read_csv(os.path.join(data_dir, 'kospi200_and_related.csv'))
        self.stock_prices = pd.read_csv(os.path.join(data_dir, 'kospi200_stock_prices.csv'))
        self.stock_prices['기준일자'] = pd.to_datetime(self.stock_prices['기준일자'])
        try:
            self.news_sentiment = pd.read_excel(
                os.path.join(data_dir, 'lg_news_finbert_sentiment.xlsx'),
                engine='openpyxl'
            )
        except Exception as e:
            print(f"뉴스 데이터 로드 오류: {e}")
            self.news_sentiment = pd.DataFrame()

    def preprocess_data(self):
        self.calculate_returns()
        self.calculate_volatility()
        self.aggregate_sentiment()
        self.build_feature_dataset()

    def calculate_returns(self):
        returns = []
        for code in self.stock_meta['종목코드'].unique():
            stock_data = self.stock_prices[self.stock_prices['종목코드'] == code].sort_values('기준일자')
            if len(stock_data) >= 21:
                recent_return = (stock_data['현재가'].iloc[-1] / stock_data['현재가'].iloc[-21] - 1) * 100
                returns.append({'종목코드': code, '1개월수익률': recent_return})
        self.returns_df = pd.DataFrame(returns)

    def calculate_volatility(self):
        volatility = []
        for code in self.stock_meta['종목코드'].unique():
            stock_data = self.stock_prices[self.stock_prices['종목코드'] == code].sort_values('기준일자')
            if len(stock_data) >= 21:
                daily_returns = stock_data['현재가'].pct_change().dropna()
                vol = daily_returns.std() * np.sqrt(252)
                volatility.append({'종목코드': code, '변동성': vol})
        self.volatility_df = pd.DataFrame(volatility)

    def aggregate_sentiment(self):
        if not self.news_sentiment.empty:
            stock_mapping = dict(zip(
                self.stock_meta['종목명'],
                self.stock_meta['종목코드']
            ))
            def extract_stock_code(title):
                for name, code in stock_mapping.items():
                    if name in title:
                        return code
                return None
            self.news_sentiment['종목코드'] = self.news_sentiment['Title'].apply(extract_stock_code)
            sentiment_df = self.news_sentiment.groupby('종목코드')['finbert_positive'].mean().reset_index()
            sentiment_df.rename(columns={'finbert_positive':'감성점수'}, inplace=True)
            self.sentiment_df = self.stock_meta[['종목코드']].merge(
                sentiment_df,
                on='종목코드',
                how='left'
            ).fillna({'감성점수': 0.5})
        else:
            self.sentiment_df = pd.DataFrame({
                '종목코드': self.stock_meta['종목코드'],
                '감성점수': 0.5
            })

    def build_feature_dataset(self):
        self.features = self.stock_meta.merge(
            self.returns_df, on='종목코드', how='left'
        ).merge(
            self.volatility_df, on='종목코드', how='left'
        ).merge(
            self.sentiment_df, on='종목코드', how='left'
        )
        # 예시: 수급/펀더멘털 데이터가 있다면 추가 merge
        # self.features = self.features.merge(self.supply_demand_df, on='종목코드', how='left')
        # self.features = self.features.merge(self.fundamental_df, on='종목코드', how='left')

        self.features = self.features.fillna({
            '1개월수익률': 0,
            '변동성': 0,
            '감성점수': 0.5,
            # '외국인순매수': 0,
            # 'ROE': 0,
            # 'PBR': 0,
        })
        self.normalize_features()


    def normalize_features(self):
        for feature in ['1개월수익률', '변동성']:
            min_val = self.features[feature].min()
            max_val = self.features[feature].max()
            if max_val - min_val > 0:
                self.features[f'{feature}_norm'] = (self.features[feature] - min_val) / (max_val - min_val)
            else:
                self.features[f'{feature}_norm'] = 0
    #자산 정보 추가시 확장
        # for feature in ['1개월수익률', '변동성', '외국인순매수', 'ROE', 'PBR']:
        #     if feature in self.features.columns:
        #         min_val = self.features[feature].min()
        #         max_val = self.features[feature].max()
        #         if max_val - min_val > 0:
        #             self.features[f'{feature}_norm'] = (self.features[feature] - min_val) / (max_val - min_val)
        #         else:
        #             self.features[f'{feature}_norm'] = 0

    # ---------------- MLP 신경망 ----------------

    def train_mlp(self, target_col='1개월수익률', feature_cols=None, epochs=3000, lr=0.01):
        if feature_cols is None:
            feature_cols = ['1개월수익률_norm', '변동성_norm', '감성점수']
        X = self.features[feature_cols].values.astype(np.float32)
        y = self.features[target_col].values.astype(np.float32)
        model = StockMLP(input_dim=X.shape[1], hidden_dim=16)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y).unsqueeze(1)
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0:
                print(f"epoch {epoch}, loss: {loss.item():.4f}")
        self.mlp_model = model
        self.mlp_feature_cols = feature_cols

    def calculate_scores_mlp(self):
        X = self.features[self.mlp_feature_cols].values.astype(np.float32)
        with torch.no_grad():
            scores = self.mlp_model(torch.tensor(X)).squeeze().numpy()
        self.features['MLP_점수'] = scores
        # 0~100 정규화
        min_score = self.features['MLP_점수'].min()
        max_score = self.features['MLP_점수'].max()
        if max_score > min_score:
            self.features['MLP_점수_100'] = 100 * (self.features['MLP_점수'] - min_score) / (max_score - min_score)
        else:
            self.features['MLP_점수_100'] = 0
        self.features = self.features.sort_values('MLP_점수_100', ascending=False)

    def get_recommendations_mlp(self, top_n=7, investment_type=None):
        self.calculate_scores_mlp()
        top_stocks = self.features.head(top_n)
        recommendations = []
        for _, row in top_stocks.iterrows():
            recommendations.append({
                '종목코드': row['종목코드'],
                '종목명': row['종목명'],
                'MLP_점수': row['MLP_점수'],
                'MLP_점수_100': row['MLP_점수_100'],
                '주요팩터': {
                    '1개월수익률': row['1개월수익률'],
                    '변동성': row['변동성'],
                    '감성점수': row['감성점수'],
                    # '외국인순매수': row.get('외국인순매수', None),
                    # 'ROE': row.get('ROE', None),
                    # 'PBR': row.get('PBR', None),
                },
                '추천이유': self.generate_explanation(row, investment_type)
            })
        return recommendations

    def generate_explanation(self, row, investment_type=None):
        client = openai.OpenAI()  # .env 파일의 OPENAI_API_KEY를 자동으로 사용합니다
        # 주요 팩터별로 설명 강조
        prompt = f"""
        [종목 정보]
        - 종목명: {row['종목명']}
        - 1개월 수익률: {row['1개월수익률']:.2f}%
        - 변동성: {row['변동성']:.2f}
        - 뉴스 감성점수: {row['감성점수']:.2f}
        """
        if investment_type:
            prompt += f"\n[투자자 성향] {investment_type} 투자자에게 추천하는 이유를 포함해 주세요.\n"
        prompt += """
        [요청]
        위 정보를 참고해 이 종목의 투자 매력과 추천 전략을 2~3문장으로 설명해 주세요.
        """
        try:
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
            print(f"GPT API 오류: {e}")
            # Fallback: rule 기반 설명
            return (
                f"MLP 기반 추천: 감성점수({row['감성점수']:.2f}), "
                f"변동성({row['변동성_norm']:.2f}), "
                f"1개월수익률({row['1개월수익률_norm']:.2f}) 등 종합 고려"
            )

    #모델 성능 평가
    def evaluate_model(self):
            X = self.features[self.mlp_feature_cols].values.astype(np.float32)
            y_true = self.features['1개월수익률'].values.astype(np.float32)
            with torch.no_grad():
                y_pred = self.mlp_model(torch.tensor(X)).squeeze().numpy()
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            print(f"MLP 예측 RMSE: {rmse:.4f}, 상관계수: {corr:.4f}")

# 사용 예시
if __name__ == "__main__":
    model = StockRecommendationModelV2()
    model.train_mlp(
        target_col='1개월수익률',
        feature_cols=['1개월수익률_norm', '변동성_norm', '감성점수']
    )
    model.evaluate_model()  # 성능 평가
    recommendations = model.get_recommendations_mlp(top_n=7, investment_type="적극투자형")
    for rec in recommendations:
        print(rec)
