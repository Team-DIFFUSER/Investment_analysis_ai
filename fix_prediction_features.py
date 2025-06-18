#!/usr/bin/env python3
"""
예측 특성 목록 수정 스크립트
predict_next_five_days 메서드의 특성 목록을 prepare_data와 일치하도록 수정
"""

import os
import re

def fix_prediction_features(file_path):
    """예측 메서드의 특성 목록 수정"""
    print(f"예측 특성 수정 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 잘못된 특성 목록을 올바른 특성 목록으로 교체
    old_features = """features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 
                       'MA5', 'MA20', 'MA60', 'Volatility',
                       'Volume_MA5', 'Volume_MA20', 'Price_Change',
                       'Price_Change_MA5', 'RSI', 'MACD']"""
    
    new_features = """features = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'MA5', 'MA20', 'MA60', 'MA120',
                'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
                'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
                'Stoch_K', 'Stoch_D', 'TR', 'ATR',
                'Volume_MA5', 'Volume_MA20', 'Volume_Ratio',
                'Price_Change', 'Price_Change_MA5', 'Price_Change_MA20',
                'Volatility', 'Volatility_MA5', 'ROC', 'Momentum', 'ADX']"""
    
    content = content.replace(old_features, new_features)
    
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
            fix_prediction_features(file_path)
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print("모든 모델 파일 예측 특성 수정 완료!")

if __name__ == "__main__":
    main() 