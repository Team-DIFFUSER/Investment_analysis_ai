import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Timescale Cloud 연결 정보
DB_PARAMS = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'sslmode': os.getenv('DB_SSL_MODE', 'require')
}

# 환경 변수가 제대로 설정되었는지 확인
required_env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

@contextmanager
def get_db_connection():
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
def get_db_cursor(commit=True):
    with get_db_connection() as conn:
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

def execute_query(query, params=None, fetch=True):
    with get_db_cursor() as cursor:
        cursor.execute(query, params)
        if fetch and cursor.description:
            return cursor.fetchall()
        return None

def execute_values_query(query, data):
    with get_db_cursor() as cursor:
        execute_values(cursor, query, data)

def execute_many_query(query, data):
    with get_db_cursor() as cursor:
        cursor.executemany(query, data)

def execute_transaction(queries):
    with get_db_cursor() as cursor:
        for query, params in queries:
            if params is None:
                cursor.execute(query)
            else:
                execute_values(cursor, query, params) 