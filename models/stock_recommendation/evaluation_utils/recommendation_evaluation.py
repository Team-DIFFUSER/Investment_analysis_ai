import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import torch

logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """평가 지표 계산"""
        try:
            # 기본 지표
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # 방향 정확도
            direction_accuracy = np.mean((y_pred * y_true) > 0)
            
            # 상관계수
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            # 신뢰도 점수
            confidence_scores = 1 / (1 + np.abs(y_pred - y_true))
            
            self.metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                '방향정확도': direction_accuracy,
                '상관계수': correlation,
                '평균신뢰도': np.mean(confidence_scores)
            }
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"평가 지표 계산 중 오류 발생: {e}")
            raise
    
    def plot_predictions(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        title: str = "예측 vs 실제"):
        """예측 결과 시각화"""
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2)
            plt.xlabel('실제값')
            plt.ylabel('예측값')
            plt.title(title)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"예측 결과 시각화 중 오류 발생: {e}")
            raise
    
    def plot_metrics(self):
        """평가 지표 시각화"""
        try:
            if not self.metrics:
                raise ValueError("평가 지표가 계산되지 않았습니다.")
            
            plt.figure(figsize=(12, 6))
            metrics_df = pd.DataFrame({
                '지표': list(self.metrics.keys()),
                '값': list(self.metrics.values())
            })
            
            sns.barplot(x='지표', y='값', data=metrics_df)
            plt.title('모델 평가 지표')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"평가 지표 시각화 중 오류 발생: {e}")
            raise
    
    def analyze_recommendations(self, 
                              recommendations: List[Dict],
                              top_n: int = 10) -> pd.DataFrame:
        """추천 결과 분석"""
        try:
            # 추천 결과를 DataFrame으로 변환
            rec_df = pd.DataFrame(recommendations)
            
            # 상위 N개 추천 종목 분석
            top_recs = rec_df.head(top_n)
            
            # 주요 지표 통계
            stats = {
                '평균수익률': top_recs['주요팩터'].apply(lambda x: x['1개월수익률']).mean(),
                '평균변동성': top_recs['주요팩터'].apply(lambda x: x['변동성']).mean(),
                '평균PER': top_recs['주요팩터'].apply(lambda x: x['PER']).mean(),
                '평균PBR': top_recs['주요팩터'].apply(lambda x: x['PBR']).mean(),
                '평균ROE': top_recs['주요팩터'].apply(lambda x: x['ROE']).mean(),
                '보유종목비율': top_recs['주요팩터'].apply(lambda x: x['보유여부']).mean() * 100
            }
            
            return pd.DataFrame([stats])
            
        except Exception as e:
            logger.error(f"추천 결과 분석 중 오류 발생: {e}")
            raise
    
    def plot_recommendation_distribution(self, 
                                       recommendations: List[Dict],
                                       metric: str = '1개월수익률'):
        """추천 종목 분포 시각화"""
        try:
            # metric이 주요팩터에 없으면, recommendations의 최상위 키에서 값을 추출
            if len(recommendations) > 0 and metric in recommendations[0].get('주요팩터', {}):
                values = [rec['주요팩터'][metric] for rec in recommendations]
            else:
                values = [rec.get(metric, None) for rec in recommendations if rec.get(metric, None) is not None]

            plt.figure(figsize=(10, 6))
            sns.histplot(values, bins=20)
            plt.title(f'{metric} 분포')
            plt.xlabel(metric)
            plt.ylabel('빈도')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"추천 종목 분포 시각화 중 오류 발생: {e}")
            raise

    def evaluate_feature_importance(self, model, feature_cols, X_test):
        """특징 중요도 평가"""
        try:
            # 특징별 중요도 계산
            importance_scores = {}
            for i, feature in enumerate(feature_cols):
                # 원본 예측
                original_pred = model(torch.tensor(X_test, dtype=torch.float32))
                
                # 특징 제거 후 예측
                X_modified = X_test.copy()
                X_modified[:, i] = 0
                modified_pred = model(torch.tensor(X_modified, dtype=torch.float32))
                
                # 중요도 점수 계산 (예측 변화량)
                importance = torch.abs(original_pred - modified_pred).mean().item()
                importance_scores[feature] = importance
            
            # 중요도 점수 정규화
            total_importance = sum(importance_scores.values())
            normalized_scores = {k: v/total_importance for k, v in importance_scores.items()}
            
            # 특징 그룹별 평균 중요도 계산
            group_importance = {
                '기술적지표': np.mean([normalized_scores[f] for f in feature_cols if f in ['RSI', 'MA5', 'MA20', 'VOLUME_MA5', 'VOLUME_RATIO']]),
                '수익률지표': np.mean([normalized_scores[f] for f in feature_cols if f in ['return_1d', 'return_5d', 'return_20d']]),
                '변동성지표': np.mean([normalized_scores[f] for f in feature_cols if f in ['volatility_5d', 'volatility_20d']]),
                '감성분석지표': np.mean([normalized_scores[f] for f in feature_cols if f in ['sentiment_score', 'sentiment_volume']]),
                '가격예측지표': np.mean([normalized_scores[f] for f in feature_cols if f in ['price_prediction_1d', 'price_prediction_5d', 'price_prediction_20d']]),
                '재무제표지표': np.mean([normalized_scores[f] for f in feature_cols if f in ['per', 'roe', 'pbr', 'ev', 'bps', 'sale_amt', 'bus_pro', 'cup_nga', 'cap', 'profit_margin', 'asset_turnover', 'financial_leverage']])
            }
            
            return {
                'feature_importance': normalized_scores,
                'group_importance': group_importance
            }
            
        except Exception as e:
            logger.error(f"특징 중요도 평가 중 오류 발생: {e}")
            raise 