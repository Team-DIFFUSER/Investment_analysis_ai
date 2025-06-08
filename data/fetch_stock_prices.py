import os
import sys
from pathlib import Path
import time
from requests.exceptions import RequestException
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

# 현재 파일의 절대 경로
current_file = Path(__file__).resolve()
# 프로젝트 루트 디렉토리의 절대 경로
project_root = current_file.parent.parent

# 프로젝트 루트 디렉토리를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database.database import execute_query, execute_values_query, execute_transaction
from pykrx import stock

def create_stock_prices_table():
    """주가 데이터 테이블 생성"""
    queries = [
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
            market_cap BIGINT,
            foreign_holding BIGINT,
            foreign_holding_ratio DECIMAL(5,2)
        );
        """, None),
        ("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'stock_prices') THEN PERFORM create_hypertable('stock_prices', 'time'); END IF; END $$;", None),
        ("CREATE INDEX IF NOT EXISTS idx_stock_prices_code ON stock_prices (stock_code, time DESC);", None),
        # Add unique constraint if it doesn't exist
        ("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'stock_prices_time_stock_code_key'
            ) THEN
                ALTER TABLE stock_prices ADD CONSTRAINT stock_prices_time_stock_code_key 
                UNIQUE (time, stock_code);
            END IF;
        END $$;
        """, None),
        # Add missing columns if they don't exist
        ("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stock_prices' AND column_name = 'market_cap') THEN
                ALTER TABLE stock_prices ADD COLUMN market_cap BIGINT;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stock_prices' AND column_name = 'foreign_holding') THEN
                ALTER TABLE stock_prices ADD COLUMN foreign_holding BIGINT;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stock_prices' AND column_name = 'foreign_holding_ratio') THEN
                ALTER TABLE stock_prices ADD COLUMN foreign_holding_ratio DECIMAL(5,2);
            END IF;
        END $$;
        """, None)
    ]
    execute_transaction(queries)
    print("Stock prices table created/updated successfully!")

def get_date_range():
    """2025년 3월 21일까지의 데이터를 가져오도록 설정"""
    end_date = datetime(2025, 3, 26)
    start_date = end_date - timedelta(days=500)
    # YYYYMMDD 형식으로 변환
    return start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')

def clean_stock_code(stock_code):
    """종목코드 정리
    한국거래소 API 형식 -> OpenAPI 형식
    예: 'A000120' -> '000120'
    """
    # 'A' 접두사 제거
    code = stock_code.replace('A', '')
    # 6자리로 맞추기
    return code.zfill(6)

def retry_on_error(func, max_retries=3, delay=1):
    """API 호출 실패 시 재시도하는 데코레이터"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (RequestException, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:  # 마지막 시도였다면
                    raise
                print(f"  - 시도 {attempt + 1}/{max_retries} 실패, {delay}초 후 재시도...")
                time.sleep(delay)
        return None
    return wrapper

@retry_on_error
def get_stock_data_with_retry(start_date, end_date, code):
    """재시도 로직이 포함된 주가 데이터 조회"""
    return stock.get_market_ohlcv_by_date(start_date, end_date, code)

@retry_on_error
def get_market_cap_with_retry(start_date, end_date, code):
    """재시도 로직이 포함된 시가총액 데이터 조회"""
    return stock.get_market_cap_by_date(start_date, end_date, code)

@retry_on_error
def get_foreign_holding_with_retry(code, start_date, end_date):
    """재시도 로직이 포함된 외국인 보유량 데이터 조회"""
    return stock.get_exhaustion_rates_of_foreign_investment_by_ticker(code, start_date, end_date)

def fetch_stock_data(stock_code, start_date, end_date):
    """PyKrx를 사용하여 주식 데이터 가져오기"""
    try:
        # 종목코드 정리
        clean_code = clean_stock_code(stock_code)
        print(f"  - 정리된 종목코드: {clean_code}")
        
        # 주가 데이터 가져오기
        try:
            print(f"  - 주가 데이터 가져오기 시도 중...")
            print(f"  - 시작일: {start_date}, 종료일: {end_date}")
            
            # 종목코드가 유효한지 먼저 확인
            try:
                stock_name = stock.get_market_ticker_name(clean_code)
                if not stock_name:
                    print(f"  - 유효하지 않은 종목코드")
                    return None, 0
                print(f"  - 종목명: {stock_name}")
            except Exception as e:
                print(f"  - 종목코드 확인 실패: {str(e)}")
                return None, 0
                
            # 주가 데이터 가져오기 (adjusted=False로 설정)
            df = stock.get_market_ohlcv_by_date(start_date, end_date, clean_code, adjusted=False)
            if df.empty:
                print(f"  - 데이터가 비어있음")
                return None, 0
                
            # 컬럼 확인
            required_columns = ['시가', '고가', '저가', '종가', '거래량', '거래대금']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"  - 누락된 컬럼: {missing_columns}")
                return None, 0
                
            print(f"  - 주가 데이터 가져오기 성공!")
            print(f"  - 데이터 샘플:\n{df.head()}")
        except Exception as e:
            print(f"  - 주가 데이터 가져오기 실패: {str(e)}")
            print(f"  - 상세 정보: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"  - API 응답: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
            return None, 0
            
        # 컬럼명 변경
        column_names = {
            '시가': 'open_price',
            '고가': 'high_price',
            '저가': 'low_price',
            '종가': 'close_price',
            '거래량': 'volume',
            '거래대금': 'trading_value'
        }
        df = df.rename(columns=column_names)
        
        # 종목코드와 종목명 추가
        df['stock_code'] = stock_code
        df['stock_name'] = stock_name
        
        # 날짜를 인덱스에서 컬럼으로 변경
        df = df.reset_index()
        df = df.rename(columns={'날짜': 'time'})
        
        return df, len(df)
        
    except Exception as e:
        print(f"데이터 수집 중 오류 발생: {e}")
        print(f"상세 정보: {type(e).__name__}")
        return None, 0

def modify_table_structure():
    """테이블 구조 수정"""
    queries = [
        ("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stock_prices' AND column_name = 'market_cap') THEN
                ALTER TABLE stock_prices ADD COLUMN market_cap BIGINT;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stock_prices' AND column_name = 'foreign_holding') THEN
                ALTER TABLE stock_prices ADD COLUMN foreign_holding BIGINT;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stock_prices' AND column_name = 'foreign_holding_ratio') THEN
                ALTER TABLE stock_prices ADD COLUMN foreign_holding_ratio DECIMAL(5,2);
            END IF;
        END $$;
        """, None)
    ]
    execute_transaction(queries)
    print("Table structure updated successfully!")

def fetch_stock_prices():
    """KOSPI200 주가 데이터 수집"""
    print("📢 KOSPI200 주가 데이터를 가져오는 중...")
    
    # 테이블 구조 수정
    modify_table_structure()
    
    # 시작일과 종료일 설정
    start_date, end_date = get_date_range()
    print(f"📆 조회 기간: {start_date} ~ {end_date}")
    
    try:
        # 데이터베이스에서 종목 리스트 가져오기
        query = "SELECT stock_code, stock_name FROM stock_items WHERE is_kospi200 = TRUE OR is_related = TRUE;"
        results = execute_query(query)
        stock_list = pd.DataFrame(results, columns=['stock_code', 'stock_name'])
    except Exception as e:
        print(f"종목 리스트를 가져올 수 없습니다: {e}")
        return None
    
    # 종목별 데이터를 담을 리스트
    all_stock_data = []
    success_count = 0
    fail_count = 0
    
    for idx, row in stock_list.iterrows():
        stock_code = row['stock_code']
        stock_name = row['stock_name']
        
        print(f"🔄 ({idx+1}/{len(stock_list)}) {stock_name}({stock_code}) 데이터 수집 중...")
        
        df, count = fetch_stock_data(stock_code, start_date, end_date)
        
        if df is not None and not df.empty:
            all_stock_data.append(df)
            success_count += 1
            print(f"✅ {stock_name}({stock_code}) - {count}개 데이터 수집 완료")
        else:
            fail_count += 1
            print(f"❌ {stock_name}({stock_code}) - 데이터 없음")
    
    if all_stock_data:
        # 모든 주가 데이터 합치기
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        # 트랜잭션으로 데이터 업데이트
        queries = [
            ("""
            INSERT INTO stock_prices (
                time, stock_code, stock_name, open_price, high_price,
                low_price, close_price, volume, market_cap,
                foreign_holding, foreign_holding_ratio
            ) VALUES %s
            ON CONFLICT (time, stock_code) DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume,
                market_cap = EXCLUDED.market_cap,
                foreign_holding = EXCLUDED.foreign_holding,
                foreign_holding_ratio = EXCLUDED.foreign_holding_ratio
            """, [(
                row['time'], row['stock_code'], row['stock_name'],
                row['open_price'], row['high_price'], row['low_price'],
                row['close_price'], row['volume'], row.get('market_cap'),
                row.get('foreign_holding'), row.get('foreign_holding_ratio')
            ) for _, row in combined_df.iterrows()])
        ]
        execute_values_query(queries[0][0], queries[0][1])
        
        print(f"\n💾 모든 데이터가 데이터베이스에 저장되었습니다.")
        print(f"📊 통계:")
        print(f"   - 성공한 종목 수: {success_count}/{len(stock_list)}")
        print(f"   - 실패한 종목 수: {fail_count}")
        print(f"   - 수집된 데이터 레코드 수: {len(combined_df)}개")
        
        return combined_df
    else:
        print("❌ 저장할 데이터가 없습니다.")
        return None

if __name__ == "__main__":
    print("📢 KOSPI200 주가 데이터를 가져오는 중...")
    create_stock_prices_table()
    stock_data = fetch_stock_prices()
    if stock_data is not None:
        print(f"✅ 모든 데이터 수집 및 저장 완료!")
    else:
        print("❌ 데이터 수집 실패!") 