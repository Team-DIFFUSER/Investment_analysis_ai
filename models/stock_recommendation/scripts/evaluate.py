import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.recommendation_data_loader import RecommendationDataLoader
from data_processing.recommendation_data_processor import RecommendationDataProcessor
from mlp_model.recommendation_mlp_model import RecommendationMLP, RecommendationModelTrainer, RecommendationModelEvaluator
from evaluation_utils.recommendation_config import RecommendationConfig
from evaluation_utils.recommendation_evaluation import RecommendationEvaluator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_latest_model(user_id: str) -> RecommendationMLP:
    """가장 최근 모델 로드"""
    try:
        # saved 폴더에서 해당 사용자의 가장 최근 모델 찾기
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved')
        model_files = [f for f in os.listdir(model_dir) if f.startswith(f'model_{user_id}_')]
        
        if not model_files:
            raise FileNotFoundError(f"사용자 {user_id}의 모델을 찾을 수 없습니다.")
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model)
        
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
        
        logger.info(f"모델 로드 완료: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {e}")
        raise

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
        # 투자 유형 검증
        valid_types = ['공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형']
        if investment_type not in valid_types:
            raise ValueError(f"잘못된 투자 유형입니다. 다음 중 하나를 선택하세요: {', '.join(valid_types)}")
        
        # 설정 로드
        config = RecommendationConfig()
        
        # 데이터 로드 및 전처리
        data_loader = RecommendationDataLoader()
        data = data_loader.load_all_data(user_id)
        
        processor = RecommendationDataProcessor()
        processed_data = processor.process(data)
        
        # 모델 로드
        model = load_latest_model(user_id)
        
        # 예측 실행
        model.eval()
        with torch.no_grad():
            feature_cols = [
                '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm', '예측수익률_norm',
                '보유평가손익률_norm', 'per_norm', 'pbr_norm', 'roe_norm', 'ev_norm', 'bps_norm',
                'sale_amt_norm', 'bus_pro_norm', 'cup_nga_norm', 'cap_norm',
                'profit_margin_norm', 'asset_turnover_norm', 'financial_leverage_norm'
            ]
            X = torch.tensor(processed_data[feature_cols].values, dtype=torch.float32)
            predictions = model(X).squeeze().numpy()
        
        # 예측 결과에 종목 정보 추가
        processed_data['예측수익률'] = predictions
        
        # 투자 유형별 가중치 적용
        weights = config.get_investment_weights(investment_type)
        
        # 최종 점수 계산
        processed_data['최종점수'] = (
            weights['수익률'] * processed_data['1개월수익률_norm'] +
            weights['변동성'] * (1 - processed_data['변동성_1개월_norm']) +
            weights['감성'] * processed_data['sentiment_score_norm'] +
            weights['재무'] * (
                processed_data['per_norm'] +
                processed_data['pbr_norm'] +
                processed_data['roe_norm'] +
                processed_data['ev_norm'] +
                processed_data['bps_norm'] +
                processed_data['profit_margin_norm'] +
                processed_data['asset_turnover_norm'] +
                (1 - processed_data['financial_leverage_norm'])
            ) / 8
        ) * 100
        
        # 상위 종목 선정
        top_n = config.get_recommendation_config()['top_n']
        top_stocks = processed_data.sort_values('최종점수', ascending=False).head(top_n)
        
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
            recommendations.append(recommendation)
        
        # 결과 시각화
        evaluator = RecommendationEvaluator()
        evaluator.plot_recommendation_distribution(recommendations, '예측수익률')
        
        return recommendations
        
    except Exception as e:
        logger.error(f"예측 실행 중 오류 발생: {e}")
        raise

def evaluate_model(user_id: str, investment_type: str = '위험중립형'):
    """모델 평가 실행"""
    try:
        # 투자 유형 검증
        valid_types = ['공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형']
        if investment_type not in valid_types:
            raise ValueError(f"잘못된 투자 유형입니다. 다음 중 하나를 선택하세요: {', '.join(valid_types)}")
        
        # 설정 로드
        config = RecommendationConfig()
        
        # 데이터 로드 및 전처리
        data_loader = RecommendationDataLoader()
        data = data_loader.load_all_data(user_id)
        
        processor = RecommendationDataProcessor()
        processed_data = processor.process(data)
        
        # 모델 로드
        model = load_latest_model(user_id)
        
        # 모델 평가
        evaluator = RecommendationModelEvaluator(model)
        evaluation_metrics = evaluator.evaluate(processed_data)
        
        # 결과 출력
        print("\n모델 평가 지표:")
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 예측 결과 시각화
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(processed_data['1개월수익률'], processed_data['예측수익률'])
        plt.plot([-100, 100], [-100, 100], 'r--')
        plt.xlabel('실제 수익률')
        plt.ylabel('예측 수익률')
        plt.title('실제 vs 예측 수익률')
        
        plt.subplot(1, 2, 2)
        plt.hist(processed_data['예측수익률'], bins=50)
        plt.xlabel('예측 수익률')
        plt.ylabel('빈도')
        plt.title('예측 수익률 분포')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"모델 평가 중 오류 발생: {e}")
        raise

def evaluate_historical_performance(user_id: str, days: int = 30) -> Dict[str, Any]:
    """
    과거 예측 성능 평가
    
    Args:
        user_id (str): 사용자 ID
        days (int): 평가할 기간 (일)
    
    Returns:
        Dict[str, Any]: 과거 성능 평가 결과
    """
    try:
        # 설정 로드
        config = RecommendationConfig()
        
        # 데이터 로드
        data_loader = RecommendationDataLoader()
        data = data_loader.load_all_data(user_id)
        
        # 과거 데이터 필터링
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock_prices = data['stock_prices']
        stock_prices = stock_prices[
            (stock_prices['time'] >= start_date) & 
            (stock_prices['time'] <= end_date)
        ]
        
        # 일별 성능 평가
        daily_performance = []
        for date in pd.date_range(start_date, end_date):
            # 해당 날짜의 예측 결과
            predictions = recommend_stocks(user_id)
            
            # 실제 수익률 계산
            actual_returns = []
            for pred in predictions:
                stock_code = pred['종목코드']
                stock_data = stock_prices[stock_prices['stock_code'] == stock_code]
                if not stock_data.empty:
                    actual_return = (
                        stock_data['close_price'].iloc[-1] / 
                        stock_data['close_price'].iloc[0] - 1
                    ) * 100
                    actual_returns.append(actual_return)
            
            if actual_returns:
                daily_performance.append({
                    'date': date,
                    'predicted_return': np.mean([p['예측수익률'] for p in predictions]),
                    'actual_return': np.mean(actual_returns)
                })
        
        # 성능 지표 계산
        performance_df = pd.DataFrame(daily_performance)
        performance_df['error'] = performance_df['actual_return'] - performance_df['predicted_return']
        
        historical_metrics = {
            'mean_error': performance_df['error'].mean(),
            'std_error': performance_df['error'].std(),
            'direction_accuracy': np.mean(
                (performance_df['actual_return'] * performance_df['predicted_return']) > 0
            ),
            'correlation': performance_df['actual_return'].corr(performance_df['predicted_return'])
        }
        
        # 결과 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(performance_df['date'], performance_df['predicted_return'], label='예측수익률')
        plt.plot(performance_df['date'], performance_df['actual_return'], label='실제수익률')
        plt.title('과거 예측 성능')
        plt.xlabel('날짜')
        plt.ylabel('수익률 (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return historical_metrics
        
    except Exception as e:
        logger.error(f"과거 성능 평가 중 오류 발생: {e}")
        raise

def evaluate_recommendation_quality(recommendations: List[Dict[str, Any]], 
                                 actual_returns: pd.DataFrame,
                                 investment_type: str) -> Dict[str, float]:
    """
    추천 종목의 품질 평가
    
    Args:
        recommendations: 추천 종목 목록
        actual_returns: 실제 수익률 데이터
        investment_type: 투자 유형 ('공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형')
    
    Returns:
        Dict[str, float]: 평가 지표
    """
    try:
        # 투자 유형 검증
        valid_types = ['공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형']
        if investment_type not in valid_types:
            raise ValueError(f"잘못된 투자 유형입니다. 다음 중 하나를 선택하세요: {', '.join(valid_types)}")
        
        # 추천 종목의 실제 수익률 계산
        recommended_returns = []
        for rec in recommendations:
            stock_code = rec['종목코드']
            if stock_code in actual_returns.index:
                recommended_returns.append(actual_returns.loc[stock_code])
        
        if not recommended_returns:
            raise ValueError("추천 종목의 실제 수익률 데이터가 없습니다.")
        
        # 포트폴리오 수익률 계산
        portfolio_return = np.mean(recommended_returns)
        portfolio_volatility = np.std(recommended_returns)
        
        # 리스크 조정 수익률 (Sharpe Ratio)
        risk_free_rate = 0.02  # 연간 무위험 수익률
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # 투자 유형별 적합성 평가
        if investment_type == '공격투자형':
            type_score = 1.0 if portfolio_return > 0.15 and portfolio_volatility > 0.2 else 0.5
        elif investment_type == '적극투자형':
            type_score = 1.0 if portfolio_return > 0.12 and portfolio_volatility > 0.15 else 0.5
        elif investment_type == '위험중립형':
            type_score = 1.0 if 0.08 <= portfolio_return <= 0.12 and 0.1 <= portfolio_volatility <= 0.15 else 0.5
        elif investment_type == '안정추구형':
            type_score = 1.0 if portfolio_return > 0.05 and portfolio_volatility < 0.1 else 0.5
        else:  # 안정형
            type_score = 1.0 if portfolio_return > 0.03 and portfolio_volatility < 0.08 else 0.5
        
        # 분산도 평가
        sector_diversity = len(set(rec['주요팩터'].get('섹터', '') for rec in recommendations)) / len(recommendations)
        
        return {
            '포트폴리오_수익률': portfolio_return,
            '포트폴리오_변동성': portfolio_volatility,
            '샤프_비율': sharpe_ratio,
            '투자유형_적합성': type_score,
            '섹터_분산도': sector_diversity
        }
        
    except Exception as e:
        logger.error(f"추천 품질 평가 중 오류 발생: {e}")
        raise

def evaluate_recommendation_model(user_id: str, investment_type: str = '위험중립형'):
    """추천 모델 평가 실행"""
    try:
        # 투자 유형 검증
        valid_types = ['공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형']
        if investment_type not in valid_types:
            raise ValueError(f"잘못된 투자 유형입니다. 다음 중 하나를 선택하세요: {', '.join(valid_types)}")
        
        # 설정 로드
        config = RecommendationConfig()
        
        # 데이터 로드 및 전처리
        data_loader = RecommendationDataLoader()
        data = data_loader.load_all_data(user_id)
        
        processor = RecommendationDataProcessor()
        processed_data = processor.process(data)
        
        # 추천 종목 생성
        recommendations = recommend_stocks(user_id, investment_type)
        
        # 실제 수익률 데이터 준비
        actual_returns = processed_data.set_index('stock_code')['1개월수익률']
        
        # 추천 품질 평가
        quality_metrics = evaluate_recommendation_quality(
            recommendations, actual_returns, investment_type
        )
        
        # 결과 출력
        print("\n추천 모델 평가 결과:")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 시각화
        plt.figure(figsize=(12, 6))
        
        # 수익률 분포
        plt.subplot(1, 2, 1)
        returns = [rec['주요팩터']['1개월수익률'] for rec in recommendations]
        plt.hist(returns, bins=10)
        plt.xlabel('수익률 (%)')
        plt.ylabel('빈도')
        plt.title('추천 종목 수익률 분포')
        
        # 섹터 분포
        plt.subplot(1, 2, 2)
        sectors = [rec['주요팩터'].get('섹터', '기타') for rec in recommendations]
        sector_counts = pd.Series(sectors).value_counts()
        plt.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%')
        plt.title('섹터 분포')
        
        plt.tight_layout()
        plt.show()
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"추천 모델 평가 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    # 명령행 인자 처리
    import argparse
    parser = argparse.ArgumentParser(description='주식 추천 모델 평가')
    parser.add_argument('--user_id', type=str, required=True, help='사용자 ID')
    parser.add_argument('--investment_type', type=str, default='위험중립형', 
                       choices=['공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형'], 
                       help='투자 유형')
    
    args = parser.parse_args()
    
    try:
        # 추천 모델 평가
        quality_metrics = evaluate_recommendation_model(args.user_id, args.investment_type)
        
        # 추천 종목 출력
        recommendations = recommend_stocks(args.user_id, args.investment_type)
        print("\n추천 종목:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['종목명']} ({rec['종목코드']})")
            print(f"   최종점수: {rec['최종점수']:.2f}")
            print("   주요 지표:")
            for key, value in rec['주요팩터'].items():
                print(f"   - {key}: {value:.2f}")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1) 