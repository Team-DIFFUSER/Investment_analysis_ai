import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """예측 성능 지표 계산"""
        try:
            metrics = {}
            
            # MAE (Mean Absolute Error)
            metrics['MAE'] = mean_absolute_error(y_true, y_pred)
            
            # RMSE (Root Mean Square Error)
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # MAPE (Mean Absolute Percentage Error)
            metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Direction Accuracy
            true_direction = np.diff(y_true, axis=1)
            pred_direction = np.diff(y_pred, axis=1)
            direction_accuracy = np.mean(np.sign(true_direction) == np.sign(pred_direction)) * 100
            metrics['Direction_Accuracy'] = direction_accuracy
            
            # Trend Accuracy
            true_trend = np.mean(true_direction, axis=1)
            pred_trend = np.mean(pred_direction, axis=1)
            trend_accuracy = np.mean(np.sign(true_trend) == np.sign(pred_trend)) * 100
            metrics['Trend_Accuracy'] = trend_accuracy
            
            self.metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"성능 지표 계산 중 오류 발생: {str(e)}")
            raise
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        dates: List[str], stock_name: str) -> None:
        """예측 결과 시각화"""
        try:
            plt.figure(figsize=(15, 8))
            
            # 실제값과 예측값 플롯
            for i in range(y_true.shape[1]):
                plt.plot(dates, y_true[:, i], label=f'Actual Day {i+1}', marker='o')
                plt.plot(dates, y_pred[:, i], label=f'Predicted Day {i+1}', marker='x', linestyle='--')
            
            plt.title(f'Stock Price Predictions for {stock_name}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 그래프 저장
            plt.savefig(f'results/{stock_name}_predictions.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"예측 결과 시각화 중 오류 발생: {str(e)}")
            raise
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              stock_name: str) -> None:
        """오차 분포 시각화"""
        try:
            errors = y_pred - y_true
            
            plt.figure(figsize=(12, 6))
            sns.histplot(errors.flatten(), kde=True)
            plt.title(f'Error Distribution for {stock_name}')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # 그래프 저장
            plt.savefig(f'results/{stock_name}_error_distribution.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"오차 분포 시각화 중 오류 발생: {str(e)}")
            raise
    
    def generate_report(self, stock_code: str) -> str:
        """평가 보고서 생성"""
        try:
            report = f"""
            Stock Price Prediction Evaluation Report for {stock_code}
            =============================================
            
            Performance Metrics:
            -------------------
            Mean Absolute Error (MAE): {self.metrics['MAE']:.2f}
            Root Mean Square Error (RMSE): {self.metrics['RMSE']:.2f}
            Mean Absolute Percentage Error (MAPE): {self.metrics['MAPE']:.2f}%
            Direction Accuracy: {self.metrics['Direction_Accuracy']:.2f}%
            Trend Accuracy: {self.metrics['Trend_Accuracy']:.2f}%
            
            Interpretation:
            --------------
            - MAE: 평균적으로 예측값이 실제값과 {self.metrics['MAE']:.2f}원 차이가 납니다.
            - RMSE: 예측 오차의 표준편차는 {self.metrics['RMSE']:.2f}원입니다.
            - MAPE: 평균적으로 예측값이 실제값과 {self.metrics['MAPE']:.2f}% 차이가 납니다.
            - Direction Accuracy: 주가 방향 예측의 정확도는 {self.metrics['Direction_Accuracy']:.2f}%입니다.
            - Trend Accuracy: 주가 추세 예측의 정확도는 {self.metrics['Trend_Accuracy']:.2f}%입니다.
            """
            
            # 보고서 저장
            with open(f'results/{stock_code}_evaluation_report.txt', 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            logger.error(f"평가 보고서 생성 중 오류 발생: {str(e)}")
            raise

def evaluate_predictions(predicted_prices, target_dates, target_prices):
    """
    예측 결과를 평가하고 시각화하는 함수
    
    Args:
        predicted_prices (list): 예측된 가격 리스트
        target_dates (list): 목표 날짜 리스트
        target_prices (list): 실제 가격 리스트
    """
    try:
        # 날짜 문자열을 datetime 객체로 변환
        dates = [datetime.strptime(date, '%Y-%m-%d') if isinstance(date, str) else date 
                for date in target_dates]
        
        # 예측 결과와 실제 가격 비교
        results = pd.DataFrame({
            'Date': dates,
            'Actual': target_prices,
            'Predicted': predicted_prices
        })
        
        # 오차율 계산
        results['Error'] = results['Predicted'] - results['Actual']
        results['Error_Rate'] = (results['Error'] / results['Actual']) * 100
        
        # 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(results['Date'], results['Actual'], 'b-', label='Actual Price', linewidth=2)
        plt.plot(results['Date'], results['Predicted'], 'r--', label='Predicted Price', linewidth=2)
        
        # 오차율 표시
        for i, row in results.iterrows():
            plt.annotate(f'{row["Error_Rate"]:.1f}%', 
                        (row['Date'], row['Predicted']),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
        
        plt.title('Stock Price Prediction vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # 상세 분석 출력
        print("\n[예측 결과 상세 분석]")
        print("-" * 50)
        print(f"{'날짜':<12} {'실제가격':>10} {'예측가격':>10} {'오차':>10} {'오차율':>8}")
        print("-" * 50)
        
        for _, row in results.iterrows():
            print(f"{row['Date'].strftime('%Y-%m-%d'):<12} "
                  f"{row['Actual']:>10,.0f} "
                  f"{row['Predicted']:>10,.0f} "
                  f"{row['Error']:>10,.0f} "
                  f"{row['Error_Rate']:>8.1f}%")
        
        # 전체 성능 지표 계산
        mae = mean_absolute_error(results['Actual'], results['Predicted'])
        rmse = np.sqrt(mean_squared_error(results['Actual'], results['Predicted']))
        mape = np.mean(np.abs(results['Error_Rate']))
        
        print("\n[전체 성능 지표]")
        print(f"MAE (Mean Absolute Error): {mae:,.0f}")
        print(f"RMSE (Root Mean Squared Error): {rmse:,.0f}")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.1f}%")
        
        # 방향성 정확도 계산
        actual_direction = np.diff(results['Actual'])
        pred_direction = np.diff(results['Predicted'])
        direction_accuracy = np.mean(np.sign(actual_direction) == np.sign(pred_direction)) * 100
        
        print(f"방향성 정확도: {direction_accuracy:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        return False 