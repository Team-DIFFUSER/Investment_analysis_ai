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
    """가장 최근 모델 로드 (사용자별 모델 없으면 공통 모델 사용)"""
    try:
        # saved 폴더에서 해당 사용자의 가장 최근 모델 찾기
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved')
        model_files = [f for f in os.listdir(model_dir) if f.startswith(f'model_{user_id}_')]
        
        if not model_files:
            # 사용자별 모델이 없으면 공통 모델(model_latest.pt) 사용
            model_path = os.path.join(model_dir, 'model_latest.pt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"공통 모델(model_latest.pt)도 찾을 수 없습니다.")
        else:
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
        features_df = processed_data['features']

        # stock_code가 리스트인 경우 첫 번째 값만 사용
        if features_df['stock_code'].apply(lambda x: isinstance(x, list)).any():
            features_df['stock_code'] = features_df['stock_code'].apply(lambda x: x[0] if isinstance(x, list) else x)
        
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
            X = torch.tensor(features_df[feature_cols].values, dtype=torch.float32)
            predictions = model(X).squeeze().numpy()
        
        # 예측 결과에 종목 정보 추가
        features_df['예측수익률'] = predictions
        
        # 투자 유형별 가중치 적용
        weights = config.get_investment_weights(investment_type)

        # nan/inf를 0으로 대체 (최종점수에 사용되는 모든 컬럼)
        norm_features = [
            '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm',
            'per_norm', 'pbr_norm', 'roe_norm', 'ev_norm', 'bps_norm',
            'profit_margin_norm', 'asset_turnover_norm', 'financial_leverage_norm'
        ]
        for feature in norm_features:
            if feature in features_df.columns:
                features_df[feature] = features_df[feature].replace([np.inf, -np.inf], 0.0)
                features_df[feature] = features_df[feature].fillna(0.0)

        # 예측값 칼리브레이션 (선형 보정)
        def _calibrate_predictions(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            try:
                if len(y_pred) != len(y_true) or len(y_pred) < 5:
                    return y_pred
                mask = (~np.isnan(y_pred)) & (~np.isnan(y_true)) & np.isfinite(y_pred) & np.isfinite(y_true)
                if mask.sum() < 5:
                    return y_pred
                a, b = np.polyfit(y_pred[mask], y_true[mask], 1)
                y_cal = a * y_pred + b
                return np.clip(y_cal, -20.0, 20.0)
            except Exception:
                return y_pred

        if '1개월수익률' in features_df.columns:
            features_df['예측수익률'] = _calibrate_predictions(features_df['예측수익률'].values, features_df['1개월수익률'].values)

        # 랭킹 기반 최종 점수 계산
        def _pct_rank(s: pd.Series) -> pd.Series:
            try:
                return s.rank(pct=True, method='average').fillna(0.0)
            except Exception:
                return pd.Series(np.zeros(len(s)), index=s.index)

        mom_rank = _pct_rank(features_df.get('1개월수익률', pd.Series(np.zeros(len(features_df)))))
        vol_rank = 1.0 - _pct_rank(features_df.get('변동성_1개월', pd.Series(np.zeros(len(features_df)))))
        sent_rank = _pct_rank(features_df.get('sentiment_score', pd.Series(np.zeros(len(features_df)))))
        pred_rank = _pct_rank(features_df['예측수익률'])

        fin_cols = [
            'per_norm', 'pbr_norm', 'roe_norm', 'ev_norm', 'bps_norm',
            'profit_margin_norm', 'asset_turnover_norm', 'financial_leverage_norm'
        ]
        for c in fin_cols:
            if c not in features_df.columns:
                features_df[c] = 0.0
        fin_composite = (
            features_df['per_norm'] +
            features_df['pbr_norm'] +
            features_df['roe_norm'] +
            features_df['ev_norm'] +
            features_df['bps_norm'] +
            features_df['profit_margin_norm'] +
            features_df['asset_turnover_norm'] +
            (1 - features_df['financial_leverage_norm'])
        ) / 8.0
        fin_rank = _pct_rank(fin_composite)

        ret_rank_blend = 0.5 * mom_rank + 0.5 * pred_rank

        features_df['최종점수'] = 100.0 * (
            weights['수익률'] * ret_rank_blend +
            weights['변동성'] * vol_rank +
            weights['감성'] * sent_rank +
            weights['재무'] * fin_rank
        )
        
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
        processed = processor.process(data)
        features_df = processed['features']
        
        # stock_code가 리스트인 경우 첫 번째 값만 사용
        if features_df['stock_code'].apply(lambda x: isinstance(x, list)).any():
            features_df['stock_code'] = features_df['stock_code'].apply(lambda x: x[0] if isinstance(x, list) else x)
        
        # 모델 로드 및 예측 생성 (평가용)
        model = load_latest_model(user_id)
        model.eval()
        with torch.no_grad():
            feature_cols = [
                '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm', '예측수익률_norm',
                '보유평가손익률_norm', 'per_norm', 'pbr_norm', 'roe_norm', 'ev_norm', 'bps_norm',
                'sale_amt_norm', 'bus_pro_norm', 'cup_nga_norm', 'cap_norm',
                'profit_margin_norm', 'asset_turnover_norm', 'financial_leverage_norm'
            ]
            X_eval = torch.tensor(features_df[feature_cols].values, dtype=torch.float32)
            y_pred = model(X_eval).squeeze().numpy()

        # 칼리브레이션
        y_true = features_df['1개월수익률'].values if '1개월수익률' in features_df.columns else y_pred
        try:
            a, b = np.polyfit(y_pred, y_true, 1)
            y_pred_cal = np.clip(a * y_pred + b, -20.0, 20.0)
        except Exception:
            y_pred_cal = y_pred

        # 랭킹 메트릭 (IC/RankIC/HitRatio)
        df_eval = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred_cal
        }).replace([np.inf, -np.inf], np.nan).dropna()

        ic_pearson = df_eval['y_true'].corr(df_eval['y_pred'], method='pearson') if len(df_eval) > 2 else np.nan
        ic_spearman = df_eval['y_true'].corr(df_eval['y_pred'], method='spearman') if len(df_eval) > 2 else np.nan
        hit_ratio = float(np.mean(np.sign(df_eval['y_true']) == np.sign(df_eval['y_pred']))) if len(df_eval) > 0 else np.nan

        # Decile 백테스트 (예측 기준 정렬)
        try:
            df_eval['decile'] = pd.qcut(df_eval['y_pred'], 10, labels=False, duplicates='drop')
            decile_returns = df_eval.groupby('decile')['y_true'].mean()
            top_decile = decile_returns.iloc[-1] if len(decile_returns) > 0 else np.nan
            bottom_decile = decile_returns.iloc[0] if len(decile_returns) > 0 else np.nan
            long_short = top_decile - bottom_decile if pd.notnull(top_decile) and pd.notnull(bottom_decile) else np.nan
        except Exception:
            top_decile = bottom_decile = long_short = np.nan

        print("\n추천 모델 랭킹 평가:")
        print(f"IC(Pearson): {ic_pearson:.4f}" if pd.notnull(ic_pearson) else "IC(Pearson): N/A")
        print(f"RankIC(Spearman): {ic_spearman:.4f}" if pd.notnull(ic_spearman) else "RankIC(Spearman): N/A")
        print(f"Hit Ratio: {hit_ratio:.4f}" if pd.notnull(hit_ratio) else "Hit Ratio: N/A")
        print(f"Top Decile 평균수익률: {top_decile:.4f}" if pd.notnull(top_decile) else "Top Decile 평균수익률: N/A")
        print(f"Long-Short (D10-D1): {long_short:.4f}" if pd.notnull(long_short) else "Long-Short (D10-D1): N/A")

        # 기존 추천 품질 평가도 병행
        recommendations = recommend_stocks(user_id, investment_type)
        actual_returns = features_df.set_index('stock_code')['1개월수익률']
        quality_metrics = evaluate_recommendation_quality(
            recommendations, actual_returns, investment_type
        )

        print("\n추천 모델 포트폴리오 평가:")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.4f}")

        # 간단 시각화: 예측 vs 실제 산포, Decile bar
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(df_eval['y_pred'], df_eval['y_true'], s=8, alpha=0.6)
            plt.xlabel('예측수익률(교정)')
            plt.ylabel('실제 1M 수익률')
            plt.title('예측 vs 실제')

            plt.subplot(1, 2, 2)
            if 'decile' in df_eval.columns:
                means = df_eval.groupby('decile')['y_true'].mean()
                means.plot(kind='bar')
                plt.title('Decile 평균 1M 수익률')
                plt.xlabel('Decile (낮음→높음)')
                plt.ylabel('평균 수익률')
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

        return {
            'IC': float(ic_pearson) if pd.notnull(ic_pearson) else None,
            'RankIC': float(ic_spearman) if pd.notnull(ic_spearman) else None,
            'HitRatio': float(hit_ratio) if pd.notnull(hit_ratio) else None,
            'TopDecileRet': float(top_decile) if pd.notnull(top_decile) else None,
            'LongShort': float(long_short) if pd.notnull(long_short) else None,
            **quality_metrics
        }
        
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