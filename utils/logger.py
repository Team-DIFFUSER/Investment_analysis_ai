import logging
import os
from datetime import datetime
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
        
        # 포맷터 설정
        formatter = logging.Formatter(
            self.config.get_logging_config()['format']
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        log_dir = self.config.get_paths()['log_dir']
        log_file = os.path.join(
            log_dir,
            f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
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