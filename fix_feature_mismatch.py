#!/usr/bin/env python3
"""
특성 불일치 수정 스크립트
prepare_data와 predict_next_five_days의 특성 목록을 일치시키고 모델 로딩 문제 수정
"""

import os
import re

def fix_feature_mismatch(file_path):
    """특성 불일치 수정"""
    print(f"특성 불일치 수정 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # predict_next_five_days의 특성 목록을 prepare_data와 일치하도록 수정
    old_features = """features = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'MA5', 'MA20', 'MA60', 'MA120',
                'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
                'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
                'Stoch_K', 'Stoch_D', 'TR', 'ATR',
                'Volume_MA5', 'Volume_MA20', 'Volume_Ratio',
                'Price_Change', 'Price_Change_MA5', 'Price_Change_MA20',
                'Volatility', 'Volatility_MA5', 'ROC', 'Momentum', 'ADX']"""
    
    new_features = """features = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'MA5', 'MA20', 'MA60', 'MA120',
                'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
                'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
                'Stoch_K', 'Stoch_D', 'ATR',
                'Volume_MA5', 'Volume_MA20', 'Volume_Ratio',
                'Price_Change', 'Price_Change_MA5', 'Price_Change_MA20',
                'Volatility', 'Volatility_MA5',
                'ROC', 'Momentum', 'ADX']"""
    
    content = content.replace(old_features, new_features)
    
    # 모델 로딩 후 검증 추가
    model_loading_code = """            # 예측
            predictions = []
            current_sequence = X.copy()
            
            # 모델이 제대로 로드되었는지 확인
            if self.model is None:
                self.logger.error("모델이 로드되지 않았습니다.")
                return []
            
            # 예측 날짜 계산 (마지막 데이터 다음날부터 5거래일)"""
    
    old_prediction_code = """            # 예측
            predictions = []
            current_sequence = X.copy()
            
            # 예측 날짜 계산 (마지막 데이터 다음날부터 5거래일)"""
    
    content = content.replace(old_prediction_code, model_loading_code)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"완료: {file_path}")

def main():
    """메인 함수"""
    models_dir = "models/stocks"
    
    # 모든 모델 파일 찾기
    model_files = [
        "lg_electronics.py",
        "sk_hynix.py", 
        "samsung_biologics.py",
        "lg_chemical.py",
        "hanwha.py",
        "hyundai_motor.py",
        "kia.py",
        "hd_hyundai.py",
        "samsung_electronics.py"
    ]
    
    for model_file in model_files:
        file_path = os.path.join(models_dir, model_file)
        if os.path.exists(file_path):
            fix_feature_mismatch(file_path)
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print("모든 모델 파일 특성 불일치 수정 완료!")

if __name__ == "__main__":
    main() 