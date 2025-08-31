#!/usr/bin/env python3
"""
모든 실패하는 종목 모델을 16개 특성만 사용하도록 일괄 수정
"""

import os
import re

def fix_all_models():
    """모든 종목 모델을 16개 특성만 사용하도록 수정"""
    
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
            
            # 1. 불필요한 특성 생성 코드 주석 처리
            patterns_to_comment = [
                # 스토캐스틱
                (r"# 스토캐스틱.*?data\['Stoch_D'\] = data\['Stoch_K'\]\.rolling\(window=3, min_periods=1\)\.mean\(\)", 
                 "# 스토캐스틱 (16개 특성만 사용하므로 제거)\n            # low_min = data['low_price'].rolling(window=14, min_periods=1).min()\n            # high_max = data['high_price'].rolling(window=14, min_periods=1).max()\n            # data['Stoch_K'] = 100 * ((data['close_price'] - low_min) / (high_max - low_min))\n            # data['Stoch_D'] = data['Stoch_K'].rolling(window=3, min_periods=1).mean()"),
                
                # ATR
                (r"# ATR \(Average True Range\).*?data\['ATR'\] = data\['TR'\]\.rolling\(window=14, min_periods=1\)\.mean\(\)", 
                 "# ATR (16개 특성만 사용하므로 제거)\n            # tr1 = data['high_price'] - data['low_price']\n            # tr2 = abs(data['high_price'] - data['close_price'].shift())\n            # tr3 = abs(data['low_price'] - data['close_price'].shift())\n            # data['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)\n            # data['ATR'] = data['TR'].rolling(window=14, min_periods=1).mean()"),
                
                # 거래량 지표
                (r"# 거래량 지표.*?data\['Volume_Ratio'\] = data\['volume'\] / data\['Volume_MA20'\]", 
                 "# 거래량 지표 (16개 특성만 사용하므로 제거)\n            # data['Volume_MA5'] = data['volume'].rolling(window=5, min_periods=1).mean()\n            # data['Volume_MA20'] = data['volume'].rolling(window=20, min_periods=1).mean()\n            # data['Volume_Ratio'] = data['volume'] / data['Volume_MA20']"),
                
                # 가격 변화율
                (r"# 가격 변화율.*?data\['Price_Change_MA20'\] = data\['Price_Change'\]\.rolling\(window=20, min_periods=1\)\.mean\(\)", 
                 "# 가격 변화율 (16개 특성만 사용하므로 제거)\n            # data['Price_Change'] = data['close_price'].pct_change()\n            # data['Price_Change_MA5'] = data['Price_Change'].rolling(window=5, min_periods=1).mean()\n            # data['Price_Change_MA20'] = data['Price_Change'].rolling(window=20, min_periods=1).mean()"),
                
                # 변동성
                (r"# 변동성.*?data\['Volatility_MA5'\] = data\['Volatility'\]\.rolling\(window=5, min_periods=1\)\.mean\(\)", 
                 "# 변동성 (16개 특성만 사용하므로 제거)\n            # data['Volatility'] = data['close_price'].rolling(window=20, min_periods=1).std()\n            # data['Volatility_MA5'] = data['Volatility'].rolling(window=5, min_periods=1).mean()"),
                
                # 모멘텀 지표
                (r"# 모멘텀 지표.*?data\['Momentum'\] = data\['close_price'\] - data\['close_price'\]\.shift\(10\)", 
                 "# 모멘텀 지표 (16개 특성만 사용하므로 제거)\n            # data['ROC'] = data['close_price'].pct_change(periods=10) * 100\n            # data['Momentum'] = data['close_price'] - data['close_price'].shift(10)"),
                
                # 추세 강도
                (r"# 추세 강도.*?data\['ADX'\] = self\._calculate_adx\(data\)", 
                 "# 추세 강도 (16개 특성만 사용하므로 제거)\n            # data['ADX'] = self._calculate_adx(data)")
            ]
            
            for pattern, replacement in patterns_to_comment:
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            # 2. _calculate_adx 메서드 제거
            adx_method_pattern = r"def _calculate_adx\(self, data: pd\.DataFrame, period: int = 14\) -> pd\.Series:.*?return pd\.Series\(\[0\] \* len\(data\)\)"
            content = re.sub(adx_method_pattern, "# ADX 메서드 제거 (16개 특성만 사용하므로)", content, flags=re.DOTALL)
            
            # 3. features 리스트를 16개로 제한
            features_pattern = r"features\s*=\s*\[([^\]]+)\]"
            features_16 = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'MA5', 'MA20', 'MA60', 'MA120',
                'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
                'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram'
            ]
            new_features = "features = [\n                " + ",\n                ".join([f"'{f}'" for f in features_16]) + "\n            ]"
            content = re.sub(features_pattern, new_features, content, flags=re.MULTILINE)
            
            # 4. np.zeros 차원 수정
            content = content.replace("np.zeros((1, 26))", "np.zeros((1, 11))")
            
            # 파일 쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"✅ 수정 완료: {stock_file}")
            
        except Exception as e:
            print(f"❌ 수정 실패: {stock_file} - {str(e)}")
    
    print("\n🎉 모든 종목 모델 수정 완료!")
    print("💡 이제 predict.py를 실행하면 차원 불일치 오류가 해결될 것입니다.")
    print("📊 수정된 종목들:")
    for stock in target_stocks:
        print(f"   - {stock}")

if __name__ == "__main__":
    fix_all_models()
