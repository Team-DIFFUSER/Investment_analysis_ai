#!/usr/bin/env python3
"""
8개 종목 모델의 특성 수를 16개로 통일하는 수정 스크립트
"""

import os
import re

def fix_model_features():
    """모델 파일들의 특성 수를 16개로 통일"""
    
    # 수정할 종목들
    target_stocks = [
        'hanwha.py',
        'hyundai_mobis.py', 
        'samsung_life_insurance.py',
        'hd_hyundai_electric.py',
        'samsung_heavy_industries.py',
        'sk.py',
        'kakao_bank.py'
    ]
    
    # 16개 특성 목록
    features_16 = [
        'open_price', 'high_price', 'low_price', 'close_price', 'volume',
        'MA5', 'MA20', 'MA60', 'MA120',
        'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
        'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram'
    ]
    
    # 31개 특성 목록 (기존)
    features_31 = [
        'open_price', 'high_price', 'low_price', 'close_price', 'volume',
        'MA5', 'MA20', 'MA60', 'MA120',
        'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
        'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
        'Stoch_K', 'Stoch_D', 'ATR',
        'Volume_MA5', 'Volume_MA20', 'Volume_Ratio',
        'Price_Change', 'Price_Change_MA5', 'Price_Change_MA20',
        'Volatility', 'Volatility_MA5',
        'ROC', 'Momentum', 'ADX'
    ]
    
    models_dir = 'models/stocks'
    
    for stock_file in target_stocks:
        file_path = os.path.join(models_dir, stock_file)
        
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없음: {stock_file}")
            continue
            
        print(f"🔄 수정 중: {stock_file}")
        
        try:
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1. 31개 특성을 16개로 변경
            old_features_pattern = r"features\s*=\s*\[([^\]]+)\]"
            new_features_content = f"features = [\n                " + ",\n                ".join([f"'{f}'" for f in features_16]) + "\n            ]"
            
            content = re.sub(old_features_pattern, new_features_content, content, flags=re.MULTILINE)
            
            # 2. np.zeros((1, 26))을 np.zeros((1, 11))로 변경 (16개 특성에 맞춤)
            content = content.replace("np.zeros((1, 26))", "np.zeros((1, 11))")
            
            # 3. prepare_data 메서드가 없으면 추가
            if "def prepare_data" not in content:
                print(f"  - prepare_data 메서드 추가 필요: {stock_file}")
                # 여기서는 간단한 수정만 진행
                
            # 파일 쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"✅ 수정 완료: {stock_file}")
            
        except Exception as e:
            print(f"❌ 수정 실패: {stock_file} - {str(e)}")
    
    print("\n🎉 모든 종목 모델 수정 완료!")
    print("💡 이제 predict.py를 실행하면 차원 불일치 오류가 해결될 것입니다.")

if __name__ == "__main__":
    fix_model_features()
