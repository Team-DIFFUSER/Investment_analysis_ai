#!/usr/bin/env python3
"""
배치 크기 증가 스크립트
Mixed Precision 비활성화에 대한 성능 보완
"""

import os
import re

def fix_batch_size(file_path):
    """모델 파일의 배치 크기 증가"""
    print(f"배치 크기 수정 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 배치 크기를 64에서 128로 변경
    content = re.sub(
        r'batch_size=64',
        'batch_size=128',
        content
    )
    
    # 배치 크기 주석 수정
    content = re.sub(
        r'# 배치 사이즈 증가 \(속도 향상\)',
        '# 배치 사이즈 대폭 증가 (속도 향상)',
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
            fix_batch_size(file_path)
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print("모든 모델 파일 배치 크기 수정 완료!")

if __name__ == "__main__":
    main() 