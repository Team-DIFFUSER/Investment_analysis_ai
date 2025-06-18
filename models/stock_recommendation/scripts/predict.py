import os
import sys
import logging
import torch
import pandas as pd
import openai
import psycopg2
import pymongo
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from dotenv import load_dotenv
import numpy as np

# .env 파일 로드
load_dotenv()

# API 키 확인
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# TimescaleDB 연결 정보
TIMESCALE_URI = os.getenv('TIMESCALE_URI', 'postgresql://postgres:postgres@localhost:5432/timescale')

# MongoDB 연결 정보
MONGO_URI = os.environ["MONGO_URI"]
MONGO_DB = os.environ["MONGO_DB_NAME"]
MONGO_USER_ACCOUNTS = os.environ["MONGO_USER_ACCOUNTS"]

# TimescaleDB 연결 설정
TS_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'sslmode': os.getenv('DB_SSL_MODE', 'require'),
    'options': '-c client_encoding=utf8 -c timezone=Asia/Seoul'
}

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.recommendation_data_loader import RecommendationDataLoader
from data_processing.recommendation_data_processor import RecommendationDataProcessor
from mlp_model.recommendation_mlp_model import RecommendationMLP, RecommendationModelTrainer
from evaluation_utils.recommendation_config import RecommendationConfig
from evaluation_utils.recommendation_evaluation import RecommendationEvaluator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_user_investment_type(user_id: str) -> str:
    """MongoDB에서 사용자의 투자 유형을 가져옴"""
    try:
        # MongoDB 연결
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[MONGO_USER_ACCOUNTS]
        
        # 사용자 정보 조회 (username 필드 사용)
        user = collection.find_one({'username': user_id})
        
        if not user:
            logger.warning(f"사용자 {user_id}를 찾을 수 없습니다. 기본값 '위험중립형'을 사용합니다.")
            return '위험중립형'
        
        # investmentType 필드 사용
        investment_type = user.get('investmentType', '위험중립형')
        logger.info(f"사용자 {user_id}의 투자 유형: {investment_type}")
        
        return investment_type
        
    except Exception as e:
        logger.error(f"MongoDB 조회 중 오류 발생: {e}")
        return '위험중립형'
    finally:
        if 'client' in locals():
            client.close()

def load_latest_model() -> RecommendationMLP:
    """가장 최근 공통 모델 로드"""
    try:
        # saved 폴더에서 가장 최근 모델 찾기 (model_latest.pt)
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved')
        model_path = os.path.join(model_dir, 'model_latest.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"공통 모델(model_latest.pt)을 찾을 수 없습니다.")
        # 모델 초기화
        config = RecommendationConfig()
        model = RecommendationMLP(
            input_dim=config.get_model_config()['input_dim'],
            hidden_dims=config.get_model_config()['hidden_dims'],
            dropout_rate=config.get_model_config()['dropout_rate']
        )
        # 모델 로드
        trainer = RecommendationModelTrainer(model)
        trainer.load_model(model_path)
        logger.info(f"공통 모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {e}")
        raise

def generate_explanation(row: pd.Series, investment_type: str) -> str:
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
    - EV: {row['ev']:.2f}
    - BPS: {row['bps']:.2f}
    - 순이익률: {row['profit_margin']:.2f}%
    - 자산회전율: {row['asset_turnover']:.2f}
    - 재무레버리지: {row['financial_leverage']:.2f}
    [투자자 성향] {investment_type}
    [요청]
    위 정보를 참고해 이 종목의 투자 매력과 추천 전략을 2~3문장으로 설명해 주세요.
    """
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",  
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
            f"1개월수익률({row['1개월수익률']:.2f}%) 등 종합 고려"
        )

def save_recommendations_to_db(recommendations: List[Dict[str, Any]], user_id: str, investment_type: str):
    """추천 결과를 TimescaleDB에 저장"""
    try:
        conn = psycopg2.connect(**TS_CONFIG)
        cur = conn.cursor()
        
        # 문자셋 및 시간대 설정
        cur.execute("SET client_encoding TO 'UTF8';")
        cur.execute("SET names 'utf8';")
        cur.execute("SET timezone TO 'Asia/Seoul';")
        
        # 현재 시간을 KST로 가져오기
        current_time = datetime.now(timezone(timedelta(hours=9)))
        
        for rec in recommendations:
            cur.execute("""
                INSERT INTO stock_recommendations (
                    user_id, stock_code, stock_name, final_score, predicted_return,
                    recommendation_reason, investment_type, monthly_return, volatility,
                    sentiment_score, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s AT TIME ZONE 'Asia/Seoul'
                )
            """, (
                user_id,
                rec['종목코드'],
                rec['종목명'],
                rec['최종점수'],
                rec['예측수익률'],
                rec['추천이유'],
                investment_type,
                rec['주요팩터']['1개월수익률'],
                rec['주요팩터']['변동성'],
                rec['주요팩터']['감성점수'],
                current_time  # KST 시간 저장
            ))
        
        conn.commit()
        logger.info(f"추천 결과 {len(recommendations)}개 저장 완료")
        
    except Exception as e:
        logger.error(f"DB 저장 중 오류 발생: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def recommend_stocks(user_id: str, investment_type: str = '위험중립형') -> List[Dict[str, Any]]:
    """
    주식 추천 예측 실행
    
    Args:
        user_id (str): 사용자 ID
        investment_type (str): 투자 유형 ('공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형')
    
    Returns:
        List[Dict[str, Any]]: 추천 종목 목록
    """
    try:
        # 설정 로드
        config = RecommendationConfig()
        
        # 데이터 로드 및 전처리
        data_loader = RecommendationDataLoader()
        data = data_loader.load_all_data(user_id)
        
        processor = RecommendationDataProcessor()
        processed_data = processor.process(data)
        
        features_df = processed_data['features']
        print(features_df.columns)

        # stock_code가 리스트인 경우 첫 번째 값만 사용
        if features_df['stock_code'].apply(lambda x: isinstance(x, list)).any():
            features_df['stock_code'] = features_df['stock_code'].apply(lambda x: x[0] if isinstance(x, list) else x)

        # 모델 로드
        model = load_latest_model()
        
        # 예측 실행
        model.eval()
        with torch.no_grad():
            feature_cols = [
                '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm', '예측수익률_norm',
                '보유평가손익률_norm', 'per_norm', 'pbr_norm', 'roe_norm', 'ev_norm', 'bps_norm',
                'sale_amt_norm', 'bus_pro_norm', 'cup_nga_norm', 'cap_norm',
                'profit_margin_norm', 'asset_turnover_norm', 'financial_leverage_norm'
            ]
            X = features_df[feature_cols].values
            
            # nan/inf를 0으로 대체
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            X = torch.tensor(X, dtype=torch.float32)
            pred_array = model(X).squeeze().numpy()

        # 예측 결과를 딕셔너리로 변환
        predictions = {}
        for i, code in enumerate(features_df['stock_code']):
            predictions[code] = float(pred_array[i])

        # 예측 결과에 종목 정보 추가
        features_df['예측수익률'] = features_df['stock_code'].map(predictions)
        
        # 투자 유형별 가중치 적용
        weights = config.get_investment_weights(investment_type)
        
        # 최종 점수 계산에 사용할 정규화된 특징들에 nan/inf 처리
        norm_features = [
            '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm',
            'per_norm', 'pbr_norm', 'roe_norm', 'ev_norm', 'bps_norm',
            'profit_margin_norm', 'asset_turnover_norm', 'financial_leverage_norm'
        ]
        
        for feature in norm_features:
            if feature in features_df.columns:
                features_df[feature] = features_df[feature].fillna(0.0)
                features_df[feature] = features_df[feature].replace([np.inf, -np.inf], 0.0)
        
        # 최종 점수 계산
        features_df['최종점수'] = (
            weights['수익률'] * features_df['1개월수익률_norm'] +
            weights['변동성'] * (1 - features_df['변동성_1개월_norm']) +
            weights['감성'] * features_df['sentiment_score_norm'] +
            weights['재무'] * (
                features_df['per_norm'] +
                features_df['pbr_norm'] +
                features_df['roe_norm'] +
                features_df['ev_norm'] +
                features_df['bps_norm'] +
                features_df['profit_margin_norm'] +
                features_df['asset_turnover_norm'] +
                (1 - features_df['financial_leverage_norm'])
            ) / 8
        ) * 100
        
        # 최종점수에도 nan/inf 처리
        features_df['최종점수'] = features_df['최종점수'].fillna(0.0)
        features_df['최종점수'] = features_df['최종점수'].replace([np.inf, -np.inf], 0.0)
        
        # 상위 종목 선정
        top_n = config.get_recommendation_config()['top_n']
        top_stocks = features_df.sort_values('최종점수', ascending=False).head(top_n)
        
        # 추천 결과 생성
        recommendations = []
        for _, row in top_stocks.iterrows():
            recommendation = {
                '종목코드': row['stock_code'],
                '종목명': row['stock_name'],
                '최종점수': row['최종점수'],
                '예측수익률': row['예측수익률'],
                '주요팩터': {
                    '1개월수익률': row['1개월수익률'],
                    '변동성': row['변동성_1개월'],
                    '감성점수': row['sentiment_score'],
                    'PER': row['per'],
                    'PBR': row['pbr'],
                    'ROE': row['roe'],
                    'EV': row['ev'],
                    'BPS': row['bps'],
                    '매출액': row['sale_amt'],
                    '영업이익': row['bus_pro'],
                    '순이익': row['cup_nga'],
                    '자본금': row['cap'],
                    '순이익률': row['profit_margin'],
                    '자산회전율': row['asset_turnover'],
                    '재무레버리지': row['financial_leverage'],
                    '보유여부': row['보유여부']
                }
            }
            # 추천 이유 생성
            recommendation['추천이유'] = generate_explanation(row, investment_type)
            recommendations.append(recommendation)
        
        # 정렬 추가
        recommendations.sort(key=lambda x: x['최종점수'], reverse=True)

        # # 결과 시각화
        # evaluator = RecommendationEvaluator()
        # evaluator.plot_recommendation_distribution(recommendations, '예측수익률')
        
        # DB에 저장
        save_recommendations_to_db(recommendations, user_id, investment_type)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"예측 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    # 명령행 인자 처리
    import argparse
    parser = argparse.ArgumentParser(description='주식 추천 예측 실행')
    parser.add_argument('--user_id', type=str, required=True, default='JunOh', help='사용자 ID')
    parser.add_argument('--investment_type', type=str, help='투자 유형 (지정하지 않으면 MongoDB에서 조회)')
    
    args = parser.parse_args()
    
    try:
        # 투자 유형이 지정되지 않은 경우 MongoDB에서 조회
        investment_type = args.investment_type if args.investment_type else get_user_investment_type(args.user_id)
        
        recommendations = recommend_stocks(args.user_id, investment_type)
        
        # 결과 출력
        print("\n추천 종목:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['종목명']} ({rec['종목코드']})")
            print(f"   최종점수: {rec['최종점수']:.2f}")
            print(f"   예측수익률: {rec['예측수익률']:.2f}%")
            print("   주요 지표:")
            for key, value in rec['주요팩터'].items():
                print(f"   - {key}: {value:.2f}")
            print(f"   추천이유: {rec['추천이유']}")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1) 