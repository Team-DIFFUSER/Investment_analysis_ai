import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager
import pandas as pd
from datetime import datetime
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# Timescale Cloud 연결 정보
DB_PARAMS = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'sslmode': os.getenv('DB_SSL_MODE', 'require')
}

# 환경 변수 검증
required_env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.Session = None
        self._connect()
    
    def _connect(self) -> None:
        """데이터베이스 연결"""
        try:
            # SQLAlchemy 엔진 생성
            db_url = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"
            self.engine = create_engine(db_url, pool_size=5, max_overflow=10)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("데이터베이스 연결 성공")
        except Exception as e:
            logger.error(f"데이터베이스 연결 중 오류 발생: {str(e)}")
            raise

    @contextmanager
    def get_db_connection(self):
        """PostgreSQL 연결 컨텍스트 매니저"""
        conn = None
        try:
            conn = psycopg2.connect(**DB_PARAMS)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_db_cursor(self, commit=True):
        """PostgreSQL 커서 컨텍스트 매니저"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()

    def get_session(self):
        """SQLAlchemy 세션 반환"""
        if not self.Session:
            self._connect()
        return self.Session()

    def close(self) -> None:
        """데이터베이스 연결 종료"""
        if self.engine:
            self.engine.dispose()
            logger.info("데이터베이스 연결 종료")

    def execute_query(self, query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        """SQL 쿼리 실행"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute(query, params or {})
                if cursor.description:
                    return cursor.fetchall()
                return []
        except Exception as e:
            logger.error(f"쿼리 실행 중 오류 발생: {str(e)}")
            raise

    def execute_values_query(self, query: str, data: List[tuple]) -> None:
        """여러 행의 데이터를 한 번에 삽입"""
        try:
            with self.get_db_cursor() as cursor:
                execute_values(cursor, query, data)
        except Exception as e:
            logger.error(f"데이터 삽입 중 오류 발생: {str(e)}")
            raise

    def execute_transaction(self, queries: List[tuple]) -> None:
        """트랜잭션 실행"""
        try:
            with self.get_db_cursor() as cursor:
                for query, params in queries:
                    if params is None:
                        cursor.execute(query)
                    else:
                        cursor.execute(query, params)
        except Exception as e:
            logger.error(f"트랜잭션 실행 중 오류 발생: {str(e)}")
            raise

    def create_tables(self) -> None:
        """필요한 테이블 생성"""
        queries = [
            ("""
            CREATE EXTENSION IF NOT EXISTS timescaledb;
            """, None),
            ("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                time TIMESTAMPTZ NOT NULL,
                stock_code VARCHAR(10) NOT NULL,
                stock_name VARCHAR(50) NOT NULL,
                open_price DECIMAL(10,2),
                high_price DECIMAL(10,2),
                low_price DECIMAL(10,2),
                close_price DECIMAL(10,2),
                volume BIGINT,
                market_cap DECIMAL(20,2),
                foreign_holding BIGINT,
                foreign_holding_ratio DECIMAL(5,2)
            );
            """, None),
            ("SELECT create_hypertable('stock_prices', 'time');", None),
            ("CREATE INDEX IF NOT EXISTS idx_stock_prices_code ON stock_prices (stock_code, time DESC);", None),
            ("""
            CREATE TABLE IF NOT EXISTS predicted_stock_prices (
                id SERIAL PRIMARY KEY,
                stock_code VARCHAR(10) NOT NULL,
                stock_name VARCHAR(50) NOT NULL,
                prediction_date TIMESTAMPTZ NOT NULL,
                target_date TIMESTAMPTZ NOT NULL,
                predicted_price DECIMAL(10,2) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
            """, None),
            ("CREATE INDEX IF NOT EXISTS idx_predicted_prices_date ON predicted_stock_prices (prediction_date, target_date);", None),
            ("CREATE INDEX IF NOT EXISTS idx_predicted_prices_stock ON predicted_stock_prices (stock_code);", None)
        ]
        self.execute_transaction(queries)
        logger.info("테이블 생성 완료")

    def save_prediction(self, stock_code: str, stock_name: str, prediction_date: datetime, 
                       target_date: datetime, predicted_price: float) -> None:
        """예측 결과 저장"""
        query = """
        INSERT INTO predicted_stock_prices (
            stock_code, stock_name, prediction_date, target_date, predicted_price
        ) VALUES (%s, %s, %s, %s, %s)
        """
        params = (stock_code, stock_name, prediction_date, target_date, predicted_price)
        self.execute_query(query, params)
        logger.info(f"예측 결과 저장 완료: {stock_code} - {target_date}")

    def get_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """주가 데이터 조회"""
        try:
            query = """
            SELECT 
                time as date,
                stock_code,
                stock_name,
                open_price as open,
                high_price as high,
                low_price as low,
                close_price as close,
                volume,
                market_cap,
                foreign_holding,
                foreign_holding_ratio as foreign_ratio
            FROM stock_prices
            WHERE stock_code = %s
            AND time BETWEEN %s AND %s
            ORDER BY time;
            """
            results = self.execute_query(query, (stock_code, start_date, end_date))
            
            if not results:
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            
            # 숫자형 컬럼 변환
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap', 'foreign_holding', 'foreign_ratio']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"주가 데이터 조회 중 오류 발생: {str(e)}")
            raise
    
    def clean_stock_code(self, stock_code):
        """종목코드에서 'A' 접두사 제거"""
        return stock_code.replace('A', '')

    def get_sentiment_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """감성 데이터 조회"""
        try:
            query = """
            SELECT 
                pub_date as date,
                title,
                description,
                stock_code
            FROM holding_articles
            WHERE stock_code = %s
            AND pub_date BETWEEN %s AND %s
            ORDER BY pub_date;
            """
            clean_code = self.clean_stock_code(stock_code)
            results = self.execute_query(query, (clean_code, start_date, end_date))
            
            if not results:
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            return df
            
        except Exception as e:
            logger.error(f"감성 데이터 조회 중 오류 발생: {str(e)}")
            raise
    
    def get_economic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """경제지표 데이터 조회"""
        try:
            query = """
            SELECT 
                time as date,
                treasury_10y,
                dollar_index,
                usd_krw,
                korean_bond_10y
            FROM economic_indicators
            WHERE time BETWEEN %s AND %s
            ORDER BY time;
            """
            results = self.execute_query(query, (start_date, end_date))
            
            if not results:
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            
            # 경제지표 변환
            economic_columns = ['treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y']
            for col in economic_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"경제지표 데이터 조회 중 오류 발생: {str(e)}")
            raise

def test_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        print("Connection successful!")
        conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")

def create_tables():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # TimescaleDB 확장 활성화
    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    
    # 주가 데이터 테이블 생성
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        time TIMESTAMPTZ NOT NULL,
        stock_code VARCHAR(10) NOT NULL,
        stock_name VARCHAR(50) NOT NULL,
        open_price DECIMAL(10,2),
        high_price DECIMAL(10,2),
        low_price DECIMAL(10,2),
        close_price DECIMAL(10,2),
        volume BIGINT
    );
    """)
    
    # TimescaleDB 하이퍼테이블로 변환
    cur.execute("SELECT create_hypertable('stock_prices', 'time');")
    
    # 인덱스 생성
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_prices_code ON stock_prices (stock_code, time DESC);")
    
    conn.commit()
    cur.close()
    conn.close()
    print("Tables created successfully!")

def insert_test_data():
    # 테스트용 데이터 생성
    test_data = pd.DataFrame({
        'time': pd.date_range(start='2024-03-24', periods=5, freq='D'),
        'stock_code': ['A066570'] * 5,  # LG전자
        'stock_name': ['LG전자'] * 5,
        'open_price': [69700, 67500, 67200, 66800, 65700],
        'high_price': [70000, 68000, 67500, 67000, 66000],
        'low_price': [69500, 67000, 67000, 66500, 65500],
        'close_price': [69700, 67500, 67200, 66800, 65700],
        'volume': [1000000] * 5
    })

    # 데이터 삽입
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    query = """
    INSERT INTO stock_prices (
        time, stock_code, stock_name, open_price, high_price,
        low_price, close_price, volume
    ) VALUES %s
    """
    
    data = [tuple(x) for x in test_data.values]
    execute_values(cur, query, data)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Test data inserted successfully!")

def query_test_data():
    # 데이터 조회 테스트
    conn = psycopg2.connect(**DB_PARAMS)
    
    query = """
    SELECT * FROM stock_prices
    WHERE stock_code = 'A066570'
    ORDER BY time DESC
    LIMIT 5
    """
    
    df = pd.read_sql_query(query, conn)
    print("\nRetrieved data:")
    print(df)
    
    conn.close()

# Standalone database functions
def execute_query(query: str, params: Optional[dict] = None, fetch: bool = True) -> List[Dict[str, Any]]:
    """Execute a SQL query and optionally fetch results"""
    db = DatabaseManager()
    try:
        with db.get_db_cursor() as cursor:
            cursor.execute(query, params or {})
            if fetch and cursor.description:
                return cursor.fetchall()
            return []
    finally:
        db.close()

def execute_values_query(query: str, data: List[tuple]) -> None:
    """Insert multiple rows of data at once"""
    db = DatabaseManager()
    try:
        with db.get_db_cursor() as cursor:
            execute_values(cursor, query, data)
    finally:
        db.close()

def execute_transaction(queries: List[tuple]) -> None:
    """Execute a transaction with multiple queries"""
    db = DatabaseManager()
    try:
        with db.get_db_cursor() as cursor:
            for query, params in queries:
                if params is None:
                    cursor.execute(query)
                else:
                    cursor.execute(query, params)
    finally:
        db.close()

if __name__ == "__main__":
    test_connection()
    create_tables()
    insert_test_data()
    query_test_data()