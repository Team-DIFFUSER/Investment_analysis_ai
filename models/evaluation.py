import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        
    def evaluate_predictions(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """모델 예측 평가"""
        try:
            # 예측 수행
            predictions = self.model.predict(X_test)
            
            # 예측값을 실제 가격으로 변환
            last_prices = X_test[:, -1, 0]  # 마지막 가격
            actual_prices = []
            predicted_prices = []
            
            for i in range(len(predictions)):
                # 실제 가격 계산
                actual_relative = y_test[i]
                actual_price = last_prices[i] * (1 + actual_relative)
                actual_prices.append(actual_price)
                
                # 예측 가격 계산
                pred_relative = predictions[i]
                pred_price = last_prices[i] * (1 + pred_relative)
                predicted_prices.append(pred_price)
            
            actual_prices = np.array(actual_prices)
            predicted_prices = np.array(predicted_prices)
            
            # 역스케일링
            actual_prices = self.scaler.inverse_transform_price(actual_prices)
            predicted_prices = self.scaler.inverse_transform_price(predicted_prices)
            
            # 평가 지표 계산
            mse = mean_squared_error(actual_prices, predicted_prices)
            mae = mean_absolute_error(actual_prices, predicted_prices)
            rmse = np.sqrt(mse)
            
            # 방향 정확도 계산
            actual_direction = np.diff(actual_prices)
            pred_direction = np.diff(predicted_prices)
            direction_accuracy = np.mean((actual_direction > 0) == (pred_direction > 0))
            
            # 결과 저장
            results = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'direction_accuracy': direction_accuracy,
                'actual_prices': actual_prices,
                'predicted_prices': predicted_prices
            }
            
            return results
            
        except Exception as e:
            logger.error(f"예측 평가 중 오류 발생: {str(e)}")
            raise
    
    def plot_predictions(self, results: Dict, save_path: str = None):
        """예측 결과 시각화"""
        try:
            actual_prices = results['actual_prices']
            predicted_prices = results['predicted_prices']
            
            plt.figure(figsize=(12, 6))
            plt.plot(actual_prices, label='Actual', color='blue')
            plt.plot(predicted_prices, label='Predicted', color='red', linestyle='--')
            plt.title('Stock Price Prediction Results')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"예측 결과 시각화 중 오류 발생: {str(e)}")
            raise
    
    def plot_error_distribution(self, results: Dict, save_path: str = None):
        """예측 오차 분포 시각화"""
        try:
            actual_prices = results['actual_prices']
            predicted_prices = results['predicted_prices']
            errors = predicted_prices - actual_prices
            
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, kde=True)
            plt.title('Prediction Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"오차 분포 시각화 중 오류 발생: {str(e)}")
            raise
    
    def plot_direction_accuracy(self, results: Dict, save_path: str = None):
        """방향 정확도 시각화"""
        try:
            actual_prices = results['actual_prices']
            predicted_prices = results['predicted_prices']
            
            actual_direction = np.diff(actual_prices)
            pred_direction = np.diff(predicted_prices)
            
            correct_direction = (actual_direction > 0) == (pred_direction > 0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(correct_direction, label='Correct Direction', color='green')
            plt.title('Direction Prediction Accuracy')
            plt.xlabel('Time')
            plt.ylabel('Correct (1) / Incorrect (0)')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"방향 정확도 시각화 중 오류 발생: {str(e)}")
            raise
    
    def generate_evaluation_report(self, results: Dict, save_path: str = None) -> str:
        """평가 보고서 생성"""
        try:
            report = []
            report.append("Model Evaluation Report")
            report.append("=" * 50)
            report.append(f"Mean Squared Error (MSE): {results['mse']:.2f}")
            report.append(f"Root Mean Squared Error (RMSE): {results['rmse']:.2f}")
            report.append(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
            report.append(f"Direction Accuracy: {results['direction_accuracy']:.2%}")
            report.append("=" * 50)
            
            report_text = "\n".join(report)
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
            
            return report_text
            
        except Exception as e:
            logger.error(f"평가 보고서 생성 중 오류 발생: {str(e)}")
            raise 