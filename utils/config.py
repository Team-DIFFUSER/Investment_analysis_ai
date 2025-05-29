import os
from pathlib import Path
import logging

class Config:
    def __init__(self):
        # 프로젝트 루트 디렉토리 설정
        self.project_root = Path(__file__).parent.parent
        
        # 기본 디렉토리 설정
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.project_root / 'models'
        self.results_dir = self.project_root / 'results'
        self.logs_dir = self.project_root / 'logs'
        
        # 디렉토리 생성
        self._create_directories()
        
        # 로깅 설정
        self._setup_logging()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """로깅 설정"""
        log_file = self.logs_dir / 'app.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_path(self, path_type: str) -> Path:
        """경로 반환"""
        paths = {
            'data': self.data_dir,
            'models': self.models_dir,
            'results': self.results_dir,
            'logs': self.logs_dir
        }
        return paths.get(path_type, self.project_root)
    
    def get_model_path(self, model_name: str) -> Path:
        """모델 경로 반환"""
        return self.models_dir / f'{model_name}_model.h5'
    
    def get_prediction_path(self, model_name: str) -> Path:
        """예측 결과 경로 반환"""
        return self.results_dir / f'{model_name}_predictions.csv'
    
    def get_evaluation_path(self, model_name: str) -> Path:
        """평가 결과 경로 반환"""
        return self.results_dir / f'{model_name}_evaluation.json' 