import logging
import sys
from pathlib import Path
from datetime import datetime
import os
from typing import Optional
from .config import Config

class Logger:
    def __init__(self, name: str, config: Optional[Config] = None):
        self.name = name
        self.config = config or Config()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(self.name)
        
        # 로그 레벨 설정
        log_level = getattr(logging, self.config.get_logging_config()['level'])
        logger.setLevel(log_level)
        
        # 이미 핸들러가 있다면 제거
        if logger.handlers:
            logger.handlers.clear()
        
        # 로그 디렉토리 생성
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # 파일 핸들러 설정
        log_file = log_dir / f'{self.name}_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def debug(self, message: str) -> None:
        """디버그 로그"""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """정보 로그"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """경고 로그"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """에러 로그"""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """치명적 에러 로그"""
        self.logger.critical(message)
    
    def exception(self, message: str) -> None:
        """예외 로그"""
        self.logger.exception(message)

def setup_logger(name, log_level=logging.INFO):
    """로거 설정"""
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 이미 핸들러가 있다면 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 로그 디렉토리 생성
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 파일 핸들러 설정
    log_file = log_dir / f'{name}_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name):
    """기존 로거 가져오기"""
    return logging.getLogger(name)

def set_log_level(logger, level):
    """로깅 레벨 설정"""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

def add_file_handler(logger, file_path, level=logging.INFO):
    """파일 핸들러 추가"""
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)

def remove_file_handler(logger, file_path):
    """파일 핸들러 제거"""
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == str(Path(file_path).absolute()):
                logger.removeHandler(handler)
                handler.close()

def clear_handlers(logger):
    """모든 핸들러 제거"""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

def setup_training_logger(name, log_dir='logs/training'):
    """학습용 로거 설정"""
    # 로그 디렉토리 생성
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 생성
    logger = setup_logger(name)
    
    # 학습 로그 파일 핸들러 추가
    training_log_file = log_dir / f'{name}_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    add_file_handler(logger, training_log_file)
    
    return logger

def setup_evaluation_logger(name, log_dir='logs/evaluation'):
    """평가용 로거 설정"""
    # 로그 디렉토리 생성
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 생성
    logger = setup_logger(name)
    
    # 평가 로그 파일 핸들러 추가
    evaluation_log_file = log_dir / f'{name}_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    add_file_handler(logger, evaluation_log_file)
    
    return logger

def setup_prediction_logger(name, log_dir='logs/prediction'):
    """예측용 로거 설정"""
    # 로그 디렉토리 생성
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 생성
    logger = setup_logger(name)
    
    # 예측 로그 파일 핸들러 추가
    prediction_log_file = log_dir / f'{name}_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    add_file_handler(logger, prediction_log_file)
    
    return logger 