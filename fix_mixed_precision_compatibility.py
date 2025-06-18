#!/usr/bin/env python3
"""
Mixed Precision 호환성 문제 해결 스크립트
모든 모델 파일에서 recurrent_dropout 제거 및 출력 레이어 dtype 수정
"""

import os
import re

def fix_model_file(file_path):
    """모델 파일의 Mixed Precision 호환성 문제 수정"""
    print(f"수정 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # recurrent_dropout 제거
    content = re.sub(
        r'recurrent_dropout=0\.1\)',
        ')',
        content
    )
    
    # 출력 레이어 dtype 수정
    content = re.sub(
        r'outputs = layers\.Dense\(1\)',
        'outputs = layers.Dense(1, dtype=\'float32\')',
        content
    )
    
    # build_model 메서드 주석 수정
    content = re.sub(
        r'"""삼성전자 전용 모델 구축"""',
        '"""삼성전자 전용 모델 구축 (Mixed Precision 호환)"""',
        content
    )
    
    # LSTM 레이어 주석 수정
    content = re.sub(
        r'# 첫 번째 LSTM 레이어',
        '# 첫 번째 LSTM 레이어 (recurrent_dropout 제거)',
        content
    )
    content = re.sub(
        r'# 두 번째 LSTM 레이어',
        '# 두 번째 LSTM 레이어 (recurrent_dropout 제거)',
        content
    )
    content = re.sub(
        r'# 세 번째 LSTM 레이어',
        '# 세 번째 LSTM 레이어 (recurrent_dropout 제거)',
        content
    )
    
    # 옵티마이저 주석 수정
    content = re.sub(
        r'# 옵티마이저 설정',
        '# 옵티마이저 설정 (Mixed Precision 호환)',
        content
    )
    
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
        "hd_hyundai.py"
    ]
    
    for model_file in model_files:
        file_path = os.path.join(models_dir, model_file)
        if os.path.exists(file_path):
            fix_model_file(file_path)
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print("모든 모델 파일 수정 완료!")

if __name__ == "__main__":
    main() 