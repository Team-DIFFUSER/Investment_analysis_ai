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
        """가격 예측 데이터 로드"""
        try:
            df = pd.read_sql("SELECT * FROM predicted_stock_prices", self.engine)
            return df
        except Exception as e:
            logger.error(f"가격 예측 데이터 로드 실패: {e}")
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