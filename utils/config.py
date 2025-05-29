import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any

class Config:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        self.config = self._load_config()
        
        # 기본 설정
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.project_root / 'models'
        self.results_dir = self.project_root / 'results'
        
        # 디렉토리 생성
        self._create_directories()
        
        # 로깅 설정
        self._setup_logging()
        
    def _load_config(self):
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"설정 파일 로드 중 오류 발생: {str(e)}")
            raise
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.data_dir,
            self.models_dir,
            self.results_dir,
            self.data_dir / 'raw',
            self.data_dir / 'processed',
            self.models_dir / 'stocks',
            self.results_dir / 'predictions',
            self.results_dir / 'evaluations'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'app.log'),
                logging.StreamHandler()
            ]
        )
    
    @property
    def database(self):
        """데이터베이스 설정"""
        return self.config.get('database', {})
    
    @property
    def model(self):
        """모델 설정"""
        return self.config.get('model', {})
    
    @property
    def training(self):
        """학습 설정"""
        return self.config.get('training', {})
    
    @property
    def prediction(self):
        """예측 설정"""
        return self.config.get('prediction', {})
    
    @property
    def evaluation(self):
        """평가 설정"""
        return self.config.get('evaluation', {})
    
    def get_stock_config(self, stock_code):
        """특정 종목의 설정"""
        stocks = self.config.get('stocks', {})
        return stocks.get(stock_code, {})
    
    def get_feature_config(self):
        """특성 설정"""
        return self.config.get('features', {})
    
    def get_technical_indicators(self):
        """기술적 지표 설정"""
        return self.config.get('technical_indicators', {})
    
    def get_sentiment_config(self):
        """감성 분석 설정"""
        return self.config.get('sentiment', {})
    
    def get_economic_config(self):
        """경제 지표 설정"""
        return self.config.get('economic', {})
    
    def get_path(self, path_type):
        """경로 설정"""
        paths = {
            'data': self.data_dir,
            'models': self.models_dir,
            'results': self.results_dir,
            'raw_data': self.data_dir / 'raw',
            'processed_data': self.data_dir / 'processed',
            'stock_models': self.models_dir / 'stocks',
            'predictions': self.results_dir / 'predictions',
            'evaluations': self.results_dir / 'evaluations',
            'logs': self.project_root / 'logs'
        }
        return paths.get(path_type)
    
    def get_model_path(self, stock_code):
        """모델 저장 경로"""
        return self.models_dir / 'stocks' / f'{stock_code}_model'
    
    def get_prediction_path(self, stock_code, date):
        """예측 결과 저장 경로"""
        return self.results_dir / 'predictions' / f'{stock_code}_{date}.csv'
    
    def get_evaluation_path(self, stock_code):
        """평가 결과 저장 경로"""
        return self.results_dir / 'evaluations' / f'{stock_code}_evaluation'

    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        return self.config.get(key, default)
    
    def get_database_config(self) -> Dict[str, str]:
        """데이터베이스 설정 조회"""
        return self.config['database']
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 조회"""
        return self.config['model']
    
    def get_paths(self) -> Dict[str, str]:
        """경로 설정 조회"""
        return self.config['paths']
    
    def get_logging_config(self) -> Dict[str, str]:
        """로깅 설정 조회"""
        return self.config['logging'] 