import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
from sqlalchemy import create_engine

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

class RecommendationDataLoader:
    def __init__(self):
        self.mongo_uri = os.environ["MONGO_URI"]
        self.mongo_db_name = os.environ["MONGO_DB_NAME"]
        self.mongo_user_accounts = os.environ["MONGO_USER_ACCOUNTS"]
        self.mongo_user_holdings = os.environ["MONGO_USER_HOLDINGS"]
        
        # TimescaleDB 연결 정보
        self.ts_config = {
            "host": os.environ["DB_HOST"],
            "port": os.environ["DB_PORT"],
            "dbname": os.environ["DB_NAME"],
            "user": os.environ["DB_USER"],
            "password": os.environ["DB_PASSWORD"],
            "sslmode": os.environ.get("DB_SSL_MODE", "require")
        }
        
        # SQLAlchemy 엔진 생성
        self.engine = create_engine(
            f"postgresql://{self.ts_config['user']}:{self.ts_config['password']}@{self.ts_config['host']}:{self.ts_config['port']}/{self.ts_config['dbname']}?sslmode={self.ts_config['sslmode']}"
        )

    def get_mongo_connection(self):
        """MongoDB 연결"""
        try:
            client = MongoClient(self.mongo_uri)
            return client[self.mongo_db_name]
        except Exception as e:
            logger.error(f"MongoDB 연결 실패: {e}")
            raise

    def get_user_investment_type(self, username):
        """사용자 투자성향 정보 가져오기"""
        try:
            db = self.get_mongo_connection()
            user = db[self.mongo_user_accounts].find_one({'username': username})
            if user and 'investmentType' in user:
                return user['investmentType']
            return None
        except Exception as e:
            logger.error(f"투자성향 정보 조회 실패: {e}")
            raise

    def get_user_holdings(self, username):
        """사용자 보유자산 정보 가져오기"""
        try:
            db = self.get_mongo_connection()
            user = db[self.mongo_user_holdings].find_one({'username': username})
            if user and 'evltData' in user:
                return pd.DataFrame(user['evltData'])
            return pd.DataFrame(columns=['stockCode', 'name', 'quantity', 'avgPrice', 
                                       'currentPrice', 'evalAmount', 'plAmount', 'plRate'])
        except Exception as e:
            logger.error(f"보유자산 정보 조회 실패: {e}")
            raise

    def load_stock_meta(self):
        """종목 메타데이터 로드"""
        try:
            df = pd.read_sql("SELECT * FROM stock_items", self.engine)
            return df
        except Exception as e:
            logger.error(f"종목 메타데이터 로드 실패: {e}")
            raise

    def load_stock_prices(self):
        """종목 가격 데이터 로드"""
        try:
            df = pd.read_sql("SELECT * FROM stock_prices", self.engine)
            df['time'] = pd.to_datetime(df['time'])
            return df
        except Exception as e:
            logger.error(f"종목 가격 데이터 로드 실패: {e}")
            raise

    def load_news_sentiment(self):
        """뉴스 감성분석 데이터 로드"""
        try:
            df = pd.read_sql("SELECT * FROM news_sentiment", self.engine)
            return df
        except Exception as e:
            logger.error(f"뉴스 감성분석 데이터 로드 실패: {e}")
            raise

    def load_price_predictions(self):
        """가격 예측 데이터 로드 (22개 종목만)"""
        try:
            # 가격 예측이 가능한 22개 종목 리스트
            available_stocks = [
                'SK하이닉스', '한화', 'LG전자', '삼성전자', 'LG화학', 'NAVER', '기아', 
                '삼성바이오로직스', '현대모비스', 'HD현대', '삼성생명', '삼성화재', 
                '현대차', 'HD현대일렉트릭', '삼성중공업', 'SK이노베이션', '삼성SDI', 
                'SK텔레콤', 'SK', '카카오', '현대로템', '카카오뱅크'
            ]
            
            # 가격 예측 데이터 로드
            df = pd.read_sql("SELECT * FROM predicted_stock_prices", self.engine)
            
            # 22개 종목만 필터링
            if 'stock_name' in df.columns:
                df = df[df['stock_name'].isin(available_stocks)]
            elif 'stock_code' in df.columns:
                # stock_code로 필터링하는 경우를 위한 매핑 (필요시 수정)
                logger.info("stock_code 컬럼으로 필터링합니다. 필요시 stock_name 매핑을 추가하세요.")
                df = df[df['stock_code'].isin(available_stocks)]
            
            logger.info(f"가격 예측 가능한 종목 {len(df)}개 로드 완료")
            return df
            
        except Exception as e:
            logger.error(f"가격 예측 데이터 로드 실패: {e}")
            raise

    def get_available_stocks(self):
        """가격 예측이 가능한 종목 리스트 반환"""
        return [
            'SK하이닉스', '한화', 'LG전자', '삼성전자', 'LG화학', 'NAVER', '기아', 
            '삼성바이오로직스', '현대모비스', 'HD현대', '삼성생명', '삼성화재', 
            '현대차', 'HD현대일렉트릭', '삼성중공업', 'SK이노베이션', '삼성SDI', 
            'SK텔레콤', 'SK', '카카오', '현대로템', '카카오뱅크'
        ]
    
    def get_available_stock_codes(self):
        """가격 예측이 가능한 종목의 stock_code 리스트 반환 (A 접두사 포함)"""
        return [
            'A000270', 'A000660', 'A000810', 'A000880', 'A005930', 'A006400', 
            'A010140', 'A032830', 'A207940', 'A017670', 'A034730', 'A096770', 
            'A051910', 'A066570', 'A005380', 'A012330', 'A064350', 'A267250', 
            'A267260', 'A035720', 'A323410', 'A035420'
        ]
    
    def get_available_stock_codes_no_prefix(self):
        """가격 예측이 가능한 종목의 stock_code 리스트 반환 (A 접두사 제거)"""
        return [
            '000270', '000660', '000810', '000880', '005930', '006400', 
            '010140', '032830', '207940', '017670', '034730', '096770', 
            '051910', '066570', '005380', '012330', '064350', '267250', 
            '267260', '035720', '323410', '035420'
        ]

    def load_stock_meta_filtered(self):
        """가격 예측 가능한 종목의 메타데이터만 로드"""
        try:
            available_stocks = self.get_available_stocks()
            df = pd.read_sql("SELECT * FROM stock_items", self.engine)
            
            # 22개 종목만 필터링
            df = df[df['stock_name'].isin(available_stocks)]
            
            logger.info(f"필터링된 종목 메타데이터 {len(df)}개 로드 완료")
            return df
            
        except Exception as e:
            logger.error(f"필터링된 종목 메타데이터 로드 실패: {e}")
            raise

    def load_stock_prices_filtered(self):
        """가격 예측 가능한 종목의 가격 데이터만 로드"""
        try:
            available_stocks = self.get_available_stocks()
            df = pd.read_sql("SELECT * FROM stock_prices", self.engine)
            df['time'] = pd.to_datetime(df['time'])
            
            # 22개 종목만 필터링
            df = df[df['stock_name'].isin(available_stocks)]
            
            logger.info(f"필터링된 종목 가격 데이터 {len(df)}개 로드 완료")
            return df
            
        except Exception as e:
            logger.error(f"필터링된 종목 가격 데이터 로드 실패: {e}")
            raise

    def load_news_sentiment_filtered(self):
        """가격 예측 가능한 종목의 뉴스 감성분석 데이터만 로드"""
        try:
            stock_codes = self.get_available_stock_codes()
            df = pd.read_sql("SELECT * FROM news_sentiment", self.engine)
            
            # 22개 종목만 필터링
            df = df[df['stock_code'].isin(stock_codes)]
            
            logger.info(f"필터링된 뉴스 감성분석 데이터 {len(df)}개 로드 완료")
            return df
            
        except Exception as e:
            logger.error(f"필터링된 뉴스 감성분석 데이터 로드 실패: {e}")
            raise

    def load_financial_data_filtered(self):
        """가격 예측 가능한 종목의 재무제표 데이터만 로드"""
        try:
            stock_codes = self.get_available_stock_codes_no_prefix()  # A 접두사 제거된 코드 사용
            df = pd.read_sql("SELECT * FROM financial_statements", self.engine)
            
            # 22개 종목만 필터링
            df = df[df['stock_code'].isin(stock_codes)]
            
            logger.info(f"필터링된 재무제표 데이터 {len(df)}개 로드 완료")
            return df
            
        except Exception as e:
            logger.error(f"필터링된 재무제표 데이터 로드 실패: {e}")
            raise

    def load_all_data_filtered(self, username = 'JunOh'):
        """가격 예측 가능한 22개 종목의 데이터만 로드"""
        try:
            return {
                'investment_type': self.get_user_investment_type(username),
                'user_holdings': self.get_user_holdings(username),
                'stock_meta': self.load_stock_meta_filtered(),
                'stock_prices': self.load_stock_prices_filtered(),
                'news_sentiment': self.load_news_sentiment_filtered(),
                'price_predictions': self.load_price_predictions(),
                'financial_data': self.load_financial_data_filtered()
            }
        except Exception as e:
            logger.error(f"필터링된 데이터 로드 실패: {e}")
            raise

    def load_financial_data(self):
        """재무제표 데이터 로드"""
        try:
            df = pd.read_sql("SELECT * FROM financial_statements", self.engine)
            return df
        except Exception as e:
            logger.error(f"재무제표 데이터 로드 실패: {e}")
            raise

    def load_all_data(self, username = 'JunOh'):
        """모든 데이터 로드"""
        try:
            return {
                'investment_type': self.get_user_investment_type(username),
                'user_holdings': self.get_user_holdings(username),
                'stock_meta': self.load_stock_meta(),
                'stock_prices': self.load_stock_prices(),
                'news_sentiment': self.load_news_sentiment(),
                'price_predictions': self.load_price_predictions(),
                'financial_data': self.load_financial_data()
            }
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            raise 