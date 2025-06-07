import os
from dotenv import load_dotenv
from typing import Dict, Any

# .env 파일 로드
load_dotenv()

class RecommendationConfig:
    def __init__(self):
        # 데이터베이스 설정
        self.mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        self.timescale_uri = os.getenv('TIMESCALE_URI', 'postgresql://postgres:postgres@localhost:5432/timescale')
        
        # 모델 설정
        self.model_config = {
            'input_dim': 17,  # 특징 개수: 5(기존) + 12(재무제표)
            'hidden_dims': [128, 64, 32],  # 은닉층 크기 증가
            'dropout_rate': 0.3,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 10,
            'min_delta': 0.001
        }
        
        # 평가 설정
        self.eval_config = {
            'test_size': 0.2,
            'random_state': 42,
            'metrics': ['MAE', 'RMSE', 'MAPE', '방향정확도', '상관계수', '평균신뢰도']
        }
        
        # 추천 설정
        self.recommendation_config = {
            'top_n': 10,
            'min_confidence': 0.6,
            'max_volatility': 0.3,
            'min_market_cap': 1000000000,
            'min_roe': 0.1,  # 최소 ROE 10%
            'max_per': 30,   # 최대 PER 30
            'min_profit_margin': 0.05  # 최소 순이익률 5%
        }
        
        # 투자 유형별 가중치 설정
        self.investment_weights = {
            '공격투자형': {
                '수익률': 0.4,
                '변동성': 0.1,
                '감성': 0.3,
                '재무': 0.2  # 재무제표 가중치 추가
            },
            '적극투자형': {
                '수익률': 0.35,
                '변동성': 0.15,
                '감성': 0.3,
                '재무': 0.2
            },
            '위험중립형': {
                '수익률': 0.3,
                '변동성': 0.3,
                '감성': 0.2,
                '재무': 0.2
            },
            '안정추구형': {
                '수익률': 0.25,
                '변동성': 0.35,
                '감성': 0.2,
                '재무': 0.2
            },
            '안정형': {
                '수익률': 0.2,
                '변동성': 0.4,
                '감성': 0.2,
                '재무': 0.2
            }
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return self.model_config
    
    def get_eval_config(self) -> Dict[str, Any]:
        """평가 설정 반환"""
        return self.eval_config
    
    def get_recommendation_config(self) -> Dict[str, Any]:
        """추천 설정 반환"""
        return self.recommendation_config
    
    def get_investment_weights(self, investment_type: str) -> Dict[str, float]:
        """투자 유형별 가중치 반환"""
        return self.investment_weights.get(investment_type, self.investment_weights['위험중립형'])
    
    def update_model_config(self, **kwargs):
        """모델 설정 업데이트"""
        self.model_config.update(kwargs)
    
    def update_eval_config(self, **kwargs):
        """평가 설정 업데이트"""
        self.eval_config.update(kwargs)
    
    def update_recommendation_config(self, **kwargs):
        """추천 설정 업데이트"""
        self.recommendation_config.update(kwargs) 