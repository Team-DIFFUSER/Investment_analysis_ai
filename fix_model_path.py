#!/usr/bin/env python3
"""
모델 로딩 경로 수정 스크립트
모든 모델 파일의 load_model 메서드에서 경로 계산 수정
"""

import os
import re

def fix_model_path(file_path):
    """모델 파일의 로딩 경로 수정"""
    print(f"모델 경로 수정 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 잘못된 경로 계산 수정
    old_pattern = r'project_root = os\.path\.abspath\(os\.path\.join\(current_dir, \'\.\.\', \'\.\.\', \'\.\.\'\)\)'
    new_pattern = 'project_root = os.path.abspath(os.path.join(current_dir, \'..\', \'..\'))'
    
    content = re.sub(old_pattern, new_pattern, content)
    
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
            fix_model_path(file_path)
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print("모든 모델 파일 경로 수정 완료!")

if __name__ == "__main__":
    main() 