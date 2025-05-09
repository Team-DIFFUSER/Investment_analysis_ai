import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from datetime import datetime

# Timescale Cloud 연결 정보
DB_PARAMS = {
    'host': 'g51c8i9urb.qkzqkclz66.tsdb.cloud.timescale.com',
    'port': '31216',
    'database': 'tsdb',
    'user': 'tsdbadmin',
    'password': 'ciyoiu2l3l4ybde0',
    'sslmode': 'require'
}

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

if __name__ == "__main__":
    test_connection()
    create_tables()
    insert_test_data()
    query_test_data()