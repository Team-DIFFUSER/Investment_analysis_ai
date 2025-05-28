import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any

# 환경 변수 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """환경 변수에서 설정 로드"""
        try:
            # 데이터베이스 설정
            self.config['database'] = {
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT'),
                'name': os.getenv('DB_NAME')
            }
            
            # 모델 설정
            self.config['model'] = {
                'sequence_length': int(os.getenv('SEQUENCE_LENGTH', '60')),
                'prediction_days': int(os.getenv('PREDICTION_DAYS', '5')),
                'batch_size': int(os.getenv('BATCH_SIZE', '32')),
                'epochs': int(os.getenv('EPOCHS', '100')),
                'learning_rate': float(os.getenv('LEARNING_RATE', '0.001')),
                'validation_split': float(os.getenv('VALIDATION_SPLIT', '0.2'))
            }
            
            # 경로 설정
            self.config['paths'] = {
                'model_dir': os.getenv('MODEL_DIR', 'models/saved'),
                'data_dir': os.getenv('DATA_DIR', 'data'),
                'log_dir': os.getenv('LOG_DIR', 'logs'),
                'result_dir': os.getenv('RESULT_DIR', 'results')
            }
            
            # 로깅 설정
            self.config['logging'] = {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            }
            
            # 디렉토리 생성
            self._create_directories()
            
            logger.info("설정 로드 완료")
            
        except Exception as e:
            logger.error(f"설정 로드 중 오류 발생: {str(e)}")
            raise
    
    def _create_directories(self) -> None:
        """필요한 디렉토리 생성"""
        try:
            for path in self.config['paths'].values():
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            logger.error(f"디렉토리 생성 중 오류 발생: {str(e)}")
            raise
    
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