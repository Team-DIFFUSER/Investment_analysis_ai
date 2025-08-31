#!/usr/bin/env python3
"""
모든 종목 모델에서 BB_middle 중복 특성을 제거하는 스크립트
"""

import os
import re

def fix_bb_middle():
    """BB_middle 중복 특성 제거"""
    
    # 수정할 종목들
    target_stocks = [
        'hanwha.py',
        'hyundai_mobis.py', 
        'samsung_life_insurance.py',
        'hd_hyundai_electric.py',
        'samsung_heavy_industries.py',
        'sk.py',
        'kakao_bank.py',
        'sk_telecom.py'
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
            
            # 1. BB_middle 생성 코드 제거
            content = re.sub(
                r"data\['BB_middle'\] = data\['close_price'\]\.rolling\(window=20, min_periods=1\)\.mean\(\)",
                "# BB_middle은 MA20과 동일하므로 제거",
                content
            )
            
            # 2. BB_upper, BB_lower에서 BB_middle을 MA20으로 변경
            content = content.replace("data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)", "data['BB_upper'] = data['MA20'] + (data['BB_std'] * 2)")
            content = content.replace("data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)", "data['BB_lower'] = data['MA20'] - (data['BB_std'] * 2)")
            
            # 3. features 리스트에서 BB_middle 제거
            content = re.sub(
                r"'BB_middle',\s*",
                "",
                content
            )
            
            # 파일 쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"✅ 수정 완료: {stock_file}")
            
        except Exception as e:
            print(f"❌ 수정 실패: {stock_file} - {str(e)}")
    
    print("\n🎉 BB_middle 중복 특성 제거 완료!")
    print("💡 이제 모든 종목이 정확히 16개 특성을 사용할 것입니다.")

if __name__ == "__main__":
    fix_bb_middle()
