import os
import logging
from datetime import datetime
from typing import Dict, Any
import sys
import argparse
import numpy as np

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.recommendation_data_loader import RecommendationDataLoader
from data_processing.recommendation_data_processor import RecommendationDataProcessor
from mlp_model.recommendation_mlp_model import RecommendationMLP, RecommendationModelTrainer, RecommendationModelEvaluator
from evaluation_utils.recommendation_config import RecommendationConfig
from evaluation_utils.recommendation_evaluation import RecommendationEvaluator
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_latest_model(trainer):
    """공통 모델을 model_latest.pt로 저장"""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model_latest.pt')
    trainer.save_model(model_path)
    logger.info(f"공통 모델 저장 완료: {model_path}")

def train_model(user_id: str, investment_type: str = '위험중립형') -> Dict[str, Any]:
    """
    주식 추천 모델 학습
    
    Args:
        user_id (str): 사용자 ID
        investment_type (str): 투자 유형 ('공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형')
    
    Returns:
        Dict[str, Any]: 학습 결과 및 평가 지표
    """
    try:
        # 투자 유형 검증
        valid_types = ['공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형']
        if investment_type not in valid_types:
            raise ValueError(f"잘못된 투자 유형입니다. 다음 중 하나를 선택하세요: {', '.join(valid_types)}")
        
        # 설정 로드
        config = RecommendationConfig()
        
        # 데이터 로더 초기화
        data_loader = RecommendationDataLoader()
        
        # 데이터 로드
        logger.info("데이터 로드 중...")
        data = data_loader.load_all_data(user_id)
        
        # 데이터 전처리
        logger.info("데이터 전처리 중...")
        processor = RecommendationDataProcessor()
        processed_data = processor.process(data)
        
        # 모델 초기화
        logger.info("모델 초기화 중...")
        model = RecommendationMLP(
            input_dim=config.get_model_config()['input_dim'],  # 17개 특징
            hidden_dims=config.get_model_config()['hidden_dims'],
            dropout_rate=config.get_model_config()['dropout_rate']
        )
        
        # 모델 학습
        logger.info("모델 학습 중...")
        trainer = RecommendationModelTrainer(model)
        
        # 특징 컬럼 확인
        feature_cols = [
            '1개월수익률_norm', '변동성_1개월_norm', 'sentiment_score_norm', '예측수익률_norm',
            '보유평가손익률_norm', 'per_norm', 'pbr_norm', 'roe_norm', 'ev_norm', 'bps_norm',
            'sale_amt_norm', 'bus_pro_norm', 'cup_nga_norm', 'cap_norm',
            'profit_margin_norm', 'asset_turnover_norm', 'financial_leverage_norm'
        ]
        
        # features DataFrame에서 feature_cols로 선택 후 넘파이로 변환
        features = processed_data['features']
        X = features[feature_cols].values
        y = features['1개월수익률'].values
        
        # 데이터 분할 (train/val/test = 7:2:1)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
        
        # nan/inf를 0으로 대체 (train/val/test 전체)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
        print('X_train contains nan:', np.isnan(X_train).any())
        print('y_train contains nan:', np.isnan(y_train).any())
        print('X_val contains nan:', np.isnan(X_val).any())
        print('y_val contains nan:', np.isnan(y_val).any())
        print('X_test contains nan:', np.isnan(X_test).any())
        print('y_test contains nan:', np.isnan(y_test).any())
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 모델 학습
        trainer.train(train_loader, val_loader, **config.get_model_config())
        
        # 공통 모델로 저장
        save_latest_model(trainer)
        
        # 모델 평가
        logger.info("모델 평가 중...")
        evaluator = RecommendationModelEvaluator(model)
        eval_result = evaluator.evaluate(
            X_test,
            y_test
        )
        
        # 결과 저장
        result = {
            'train_result': trainer.train_result,
            'eval_result': eval_result,
            'investment_type': investment_type,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"모델 학습 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description='주식 추천 모델 학습')
    parser.add_argument('--user_id', type=str, required=True, help='사용자 ID')
    parser.add_argument('--investment_type', type=str, default='위험중립형', 
                       choices=['공격투자형', '적극투자형', '위험중립형', '안정추구형', '안정형'], 
                       help='투자 유형')
    
    args = parser.parse_args()
    
    try:
        result = train_model(args.user_id, args.investment_type)
        logger.info("학습 결과:")
        logger.info(f"평가 지표: {result['eval_result']}")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1) 