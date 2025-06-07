import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import List, Dict, Optional
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

class StockMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
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

def get_feature_columns():
    """모델에 사용할 특징 컬럼 목록 반환"""
    return [
        # 기술적 지표
        'RSI', 'MA5', 'MA20', 'VOLUME_MA5', 'VOLUME_RATIO',
        # 수익률 지표
        'return_1d', 'return_5d', 'return_20d',
        # 변동성 지표
        'volatility_5d', 'volatility_20d',
        # 감성분석 지표
        'sentiment_score', 'sentiment_volume',
        # 가격 예측 지표
        'price_prediction_1d', 'price_prediction_5d', 'price_prediction_20d',
        # 재무제표 지표
        'per', 'roe', 'pbr', 'ev', 'bps',
        'sale_amt', 'bus_pro', 'cup_nga', 'cap',
        'profit_margin', 'asset_turnover', 'financial_leverage'
    ]

class RecommendationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16], dropout_rate: float = 0.3):
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
    def __init__(self, patience: int = 10, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class RecommendationModelTrainer:
    def __init__(self, model: RecommendationMLP, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.loss_fn = nn.MSELoss()
        self.early_stopping = EarlyStopping(patience=10)
        
    def train(self, train_loader, val_loader, **kwargs):
        """모델 학습"""
        try:
            # 옵티마이저 설정
            optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get('learning_rate', 0.001))
            
            # 학습률 스케줄러 설정
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=kwargs.get('patience', 10),
                min_lr=1e-6
            )
            
            # 조기 종료 설정
            early_stopping = EarlyStopping(
                patience=kwargs.get('patience', 10),
                min_delta=kwargs.get('min_delta', 0.001)
            )
            
            # 학습 루프
            epochs = kwargs.get('epochs', 100)
            self.train_result = {
                'train_loss': [],
                'val_loss': [],
                'best_val_loss': float('inf')
            }
            
            for epoch in range(epochs):
                # 학습
                self.model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = F.mse_loss(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 검증
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        val_loss += F.mse_loss(outputs, batch_y).item()
                
                # 평균 손실 계산
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                # 학습률 조정
                scheduler.step(val_loss)
                
                # 결과 저장
                self.train_result['train_loss'].append(train_loss)
                self.train_result['val_loss'].append(val_loss)
                
                # 조기 종료 체크
                if early_stopping(val_loss):
                    logger.info(f"조기 종료: {epoch + 1} 에포크")
                    break
                
                # 로깅
                if (epoch + 1) % 10 == 0:
                    logger.info(f"에포크 {epoch + 1}/{epochs} - "
                              f"학습 손실: {train_loss:.4f}, "
                              f"검증 손실: {val_loss:.4f}")
            
            return self.train_result
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {e}")
            raise
    
    def save_model(self, path: str):
        """모델 저장"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }, path)
            logger.info(f"모델 저장 완료: {path}")
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {e}")
            raise
    
    def load_model(self, path: str):
        """모델 로드"""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"모델 로드 완료: {path}")
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            raise

class RecommendationModelEvaluator:
    def __init__(self, model: RecommendationMLP):
        self.model = model
    
    def evaluate(self, 
                X: np.ndarray, 
                y: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        try:
            self.model.eval()
            with torch.no_grad():
                pred = self.model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
            
            # 평가 지표 계산
            mae = np.mean(np.abs(pred - y))
            rmse = np.sqrt(np.mean((pred - y) ** 2))
            mape = np.mean(np.abs((y - pred) / y)) * 100
            
            # 방향 정확도 계산
            direction_accuracy = np.mean((pred * y) > 0)
            
            # 신뢰도 점수 계산
            confidence_scores = 1 / (1 + np.abs(pred - y))
            
            evaluation_metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                '방향정확도': direction_accuracy,
                'confidence_scores': confidence_scores,
                'predictions': pred
            }
            
            logger.info("모델 평가 완료")
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"모델 평가 중 오류 발생: {e}")
            raise 