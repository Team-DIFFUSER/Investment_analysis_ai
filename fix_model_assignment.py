#!/usr/bin/env python3
"""
모델 할당 수정 스크립트
load_model 메서드에서 모델을 self.model에 할당하도록 수정
"""

import os
import re

def fix_model_assignment(file_path):
    """모델 할당 수정"""
    print(f"모델 할당 수정 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # load_model 메서드에서 self.model에 할당하도록 수정
    old_load_model = '''    def load_model(self):
        """모델 로드"""
        try:
            # 프로젝트 루트 디렉토리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            
            # 모델 파일 경로
            model_path = os.path.join(project_root, 'models', 'checkpoints', f'{self.stock_name}_model.h5')
            self.logger.info(f"모델 파일 검색: {model_path}")
            
            if os.path.exists(model_path):
                self.logger.info(f"모델 파일 발견: {model_path}")
                model = tf.keras.models.load_model(model_path)
                self.logger.info("모델 로드 성공")
                return model
            else:
                self.logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return None'''
    
    new_load_model = '''    def load_model(self):
        """모델 로드"""
        try:
            # 프로젝트 루트 디렉토리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            
            # 모델 파일 경로
            model_path = os.path.join(project_root, 'models', 'checkpoints', f'{self.stock_name}_model.h5')
            self.logger.info(f"모델 파일 검색: {model_path}")
            
            if os.path.exists(model_path):
                self.logger.info(f"모델 파일 발견: {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info("모델 로드 성공")
                return self.model
            else:
                self.logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
                self.model = None
                return None
                
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            self.model = None
            return None'''
    
    content = content.replace(old_load_model, new_load_model)
    
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
            fix_model_assignment(file_path)
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print("모든 모델 파일 모델 할당 수정 완료!")

if __name__ == "__main__":
    main() 