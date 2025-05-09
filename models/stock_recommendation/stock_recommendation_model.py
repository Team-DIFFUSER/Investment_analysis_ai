import pandas as pd
import numpy as np
import requests
import openai
import os


class StockRecommendationModel:
    def __init__(self, user_id=None, backend_url=None, jwt_token=None):
        """
        user_id: 사용자 고유 ID (MongoDB _id 또는 username)
        backend_url: 백엔드 API 주소
        jwt_token: 인증 토큰 (필요시)
        """
        self.user_id = user_id
        self.backend_url = backend_url
        self.jwt_token = jwt_token

        # 5단계 투자성향별 가중치 설정 예시
        self.weights = {
            "안정형": {
                "financial_score": 0.45,
                "news_sentiment": 0.05,
                "price_momentum": 0.05,
                "volatility": -0.35,
                "dividend": 0.10
            },
            "안정추구형": {
                "financial_score": 0.35,
                "news_sentiment": 0.10,
                "price_momentum": 0.10,
                "volatility": -0.25,
                "dividend": 0.20
            },
            "위험중립형": {
                "financial_score": 0.25,
                "news_sentiment": 0.20,
                "price_momentum": 0.20,
                "volatility": -0.10,
                "dividend": 0.25
            },
            "적극투자형": {
                "financial_score": 0.15,
                "news_sentiment": 0.25,
                "price_momentum": 0.35,
                "volatility": 0.05,
                "dividend": 0.20
            },
            "공격투자형": {
                "financial_score": 0.05,
                "news_sentiment": 0.30,
                "price_momentum": 0.50,
                "volatility": 0.10,
                "dividend": 0.05
            }
        }

        # 데이터 로드
        self.load_data()

    def load_data(self):
        # 현재 파일의 위치를 기준으로 data 폴더의 절대경로를 생성
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))

        self.stock_meta = pd.read_csv(os.path.join(data_dir, 'kospi200_and_related.csv'))
        self.stock_prices = pd.read_csv(os.path.join(data_dir, 'kospi200_stock_prices.csv'))
        self.stock_prices['기준일자'] = pd.to_datetime(self.stock_prices['기준일자'])
        # 뉴스 감성분석 결과 로드
        try:
            self.news_sentiment = pd.read_excel(
                os.path.join(data_dir, 'lg_news_finbert_sentiment.xlsx'),
                engine='openpyxl'  # openpyxl 엔진 명시적 지정
            )
        except Exception as e:
            print(f"뉴스 데이터 로드 오류: {e}")
            self.news_sentiment = pd.DataFrame()

        self.preprocess_data()

    #데이터 전처리 및 종목별 특성 계산산
    def preprocess_data(self):
        self.calculate_returns() # 최근 1개월 수익률 계산
        self.calculate_volatility() # 변동성 계산
        self.aggregate_sentiment() # 뉴스 감성 점수 종목별 집계
        self.build_feature_dataset()  # 종목별 종합 데이터셋 구성

    #종목별 최근 수익률 계산
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
        """개선된 감성점수 집계 함수"""
        if not self.news_sentiment.empty:
            # 종목명-코드 매핑 딕셔너리 생성
            stock_mapping = dict(zip(
                self.stock_meta['종목명'], 
                self.stock_meta['종목코드']
            ))
        # 실제 뉴스-종목 매핑 필요, 여기서는 임시 랜덤값
        # sentiment_scores = []
        # for code in self.stock_meta['종목코드'].unique():
        #     sentiment = np.random.uniform(0, 1)
        #     sentiment_scores.append({'종목코드': code, '감성점수': sentiment})
        # self.sentiment_df = pd.DataFrame(sentiment_scores)

        # 뉴스 제목에서 종목코드 추출
            def extract_stock_code(title):
                for name, code in stock_mapping.items():
                    if name in title:
                        return code
                return None
                
            self.news_sentiment['종목코드'] = self.news_sentiment['Title'].apply(extract_stock_code)
            
            # 종목별 평균 긍정 점수 계산
            sentiment_df = self.news_sentiment.groupby('종목코드')['finbert_positive'].mean().reset_index()
            sentiment_df.rename(columns={'finbert_positive':'감성점수'}, inplace=True)
            
            # 모든 종목에 대한 감성점수 병합
            self.sentiment_df = self.stock_meta[['종목코드']].merge(
                sentiment_df, 
                on='종목코드', 
                how='left'
            ).fillna({'감성점수': 0.5})
        else:
            # 뉴스 데이터 없을 경우 기본값
            self.sentiment_df = pd.DataFrame({
                '종목코드': self.stock_meta['종목코드'],
                '감성점수': 0.5
            })

    #최종 특성 데이터셋 구축
    def build_feature_dataset(self):
        # 종목코드 기준으로 모든 데이터 통합
        self.features = self.stock_meta.merge(
            self.returns_df, on='종목코드', how='left'
        ).merge(
            self.volatility_df, on='종목코드', how='left'
        ).merge(
            self.sentiment_df, on='종목코드', how='left'
        )
        # 결측치 처리
        self.features = self.features.fillna({
            '1개월수익률': 0,
            '변동성': 0,
            '감성점수': 0.5
        })
        self.normalize_features() # 점수 정규화 (0~1 범위로)

    #특성 정규화
    def normalize_features(self):
         # Min-Max 스케일링으로 0~1 범위로 정규화
        for feature in ['1개월수익률', '변동성']:
            min_val = self.features[feature].min()
            max_val = self.features[feature].max()
            if max_val - min_val > 0:
                self.features[f'{feature}_norm'] = (self.features[feature] - min_val) / (max_val - min_val)
            else:
                self.features[f'{feature}_norm'] = 0

    def map_score_to_investment_type(self, score):
        if score <= 20:
            return "안정형"
        elif score <= 40:
            return "안정추구형"
        elif score <= 60:
            return "위험중립형"
        elif score <= 80:
            return "적극투자형"
        else:
            return "공격투자형"

    def get_user_investment_profile(self):
        # 백엔드 연동 부분은 주석 처리
        # if not self.user_id or not self.backend_url:
        #     return {'investmentType': '위험중립형', 'investmentScore': 50}
        # url = f"{self.backend_url}/api/users/profile/{self.user_id}"
        # headers = {}
        # if self.jwt_token:
        #     headers['Authorization'] = f"Bearer {self.jwt_token}"
        # try:
        #     response = requests.get(url, headers=headers)
        #     if response.status_code == 200:
        #         data = response.json()
        #         investment_score = data.get('investmentScore', 50)
        #         investment_type = data.get('investmentType')
        #         if not investment_type:
        #             investment_type = self.map_score_to_investment_type(investment_score)
        #         return {
        #             'investmentType': investment_type,
        #             'investmentScore': investment_score
        #         }
        # except Exception as e:
        #     print(f"API 호출 오류: {e}")
        # return {'investmentType': '위험중립형', 'investmentScore': 50}

        #테스트용
        return {'investmentType': '위험중립형', 'investmentScore': 50}
    
    #임의로 투자성향 지정해서 테스트
    def get_recommendations(self, investment_type="위험중립형", top_n=5):
        self.calculate_scores(investment_type)
        top_stocks = self.features.head(top_n)
        recommendations = []
        for _, row in top_stocks.iterrows():
            recommendations.append({
                '종목코드': row['종목코드'],
                '종목명': row['종목명'],
                '종합점수': row['종합점수'],
                '추천이유': self.generate_explanation(row, investment_type)
        })
        return recommendations


    #종목별 종합 점수 계산
    def calculate_scores(self, investment_type="위험중립형"):
        weights = self.weights[investment_type]
        self.features['종합점수'] = (
            weights['price_momentum'] * self.features['1개월수익률_norm'] +
            weights['volatility'] * self.features['변동성_norm'] +
            weights['news_sentiment'] * self.features['감성점수']
        )
        # 0~1로 정규화 (최소~최대)
        min_score = self.features['종합점수'].min()
        max_score = self.features['종합점수'].max()
        if max_score > min_score:
            self.features['종합점수_norm'] = (self.features['종합점수'] - min_score) / (max_score - min_score)
        else:
            self.features['종합점수_norm'] = 0
        self.features = self.features.sort_values('종합점수', ascending=False)

    def get_recommendations_for_user(self, top_n=5):
        profile = self.get_user_investment_profile()
        investment_type = profile['investmentType']
        self.calculate_scores(investment_type)
        top_stocks = self.features.head(top_n)
        recommendations = []
        for _, row in top_stocks.iterrows():
            # 0~100점 변환
            score_100 = int(row['종합점수_norm'] * 100)
            recommendations.append({
                '종목코드': row['종목코드'],
                '종목명': row['종목명'],
                '종합점수': score_100,
                '추천이유': self.generate_explanation(row, investment_type)
            })
        return recommendations

    def generate_explanation(self, stock_data, investment_type):
        """
        GPT-4o를 사용해 종목 추천 이유를 생성
        """
 # 최신 방식
        client = openai.OpenAI(api_key='OPENAI_API_KEY')  # 또는 환경변수 사용시 client = openai.OpenAI()

        # 프롬프트 구성 
        prompt = f"""
        [투자자 정보]
        - 투자성향: {investment_type}

        [종목 정보]
        - 종목명: {stock_data['종목명']}
        - 1개월 수익률: {stock_data['1개월수익률']:.2f}%
        - 변동성: {stock_data['변동성']:.2f}
        - 뉴스 감성점수: {stock_data['감성점수']:.2f}

        [요청]
        위 정보를 참고해 {investment_type} 투자자에게 이 종목을 추천하는 이유와 투자 전략을 2~3문장으로 설명해 주세요.
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

            # GPT 호출 실패시 기존 룰베이스 방식으로 fallback
            reasons = []
            if stock_data['1개월수익률'] > 0:
                reasons.append(f"최근 1개월 수익률이 {stock_data['1개월수익률']:.2f}%로 양호합니다.")
            if investment_type == "안정형" and stock_data['변동성'] < 0.2:
                reasons.append(f"변동성이 {stock_data['변동성']:.2f}로 낮아 안정적입니다.")
            elif investment_type == "적극투자형" and stock_data['변동성'] > 0.3:
                reasons.append(f"변동성이 {stock_data['변동성']:.2f}로 높아 수익 기회가 있습니다.")
            if stock_data['감성점수'] > 0.6:
                reasons.append("최근 긍정적인 뉴스가 많아 투자심리가 좋습니다.")
            if investment_type == "안정형":
                reasons.append(f"{stock_data['종목명']}은(는) 안정적인 투자를 선호하는 고객님께 적합합니다.")
            elif investment_type == "위험중립형":
                reasons.append(f"{stock_data['종목명']}은(는) 적절한 위험과 수익의 균형을 추구하는 고객님께 적합합니다.")
            elif investment_type == "적극투자형":
                reasons.append(f"{stock_data['종목명']}은(는) 높은 수익을 추구하는 고객님께 적합합니다.")
            elif investment_type == "안정추구형":
                reasons.append(f"{stock_data['종목명']}은(는) 안정적인 수익을 추구하는 고객님께 적합합니다.")
            elif investment_type == "공격투자형":
                reasons.append(f"{stock_data['종목명']}은(는) 높은 위험을 감수하는 공격적인 투자자에게 적합합니다.")
            return " ".join(reasons)

