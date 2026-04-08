import os
import sys
from pathlib import Path
import time
import random
from requests.exceptions import RequestException
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import numpy as np
import pytz
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logs_dir = Path(__file__).parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 콘솔 출력
        logging.FileHandler(logs_dir / 'stock_prices.log', encoding='utf-8')  # 파일 출력
    ]
)
logger = logging.getLogger(__name__)

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

# pykrx import
try:
    from pykrx import stock
except ImportError:
    logger.error("pykrx 라이브러리가 설치되지 않았습니다. 'pip install pykrx' 명령으로 설치해주세요.")
    sys.exit(1)

class DatabaseManager:
    def __init__(self):
        """데이터베이스 연결 초기화"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'stock_db'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'postgres')
            )
            self.cur = self.conn.cursor()
            self.create_tables()
            logger.info("데이터베이스 연결 성공")
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {str(e)}")
            raise

    def create_tables(self):
        """필요한 테이블 생성"""
        try:
            # stock_prices 테이블 생성
            self.cur.execute("""
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
                )
            """)
            
            self.conn.commit()
            logger.info("테이블 생성 완료")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"테이블 생성 중 오류 발생: {str(e)}")
            raise

    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> List[tuple]:
        """쿼리 실행"""
        try:
            if params is None:
                self.cur.execute(query)
            else:
                self.cur.execute(query, params)
            
            if fetch:
                # SELECT 쿼리인 경우에만 fetchall() 호출
                if query.strip().upper().startswith('SELECT'):
                    result = self.cur.fetchall()
                    if not result:
                        logger.warning("쿼리 결과가 없습니다.")
                        return []
                    return result
                else:
                    # INSERT, UPDATE, DELETE 등의 경우 빈 리스트 반환
                    return []
            
            self.conn.commit()
            return []
            
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"데이터베이스 오류 발생: {str(e)}")
            raise
        except Exception as e:
            self.conn.rollback()
            logger.error(f"쿼리 실행 중 오류 발생: {str(e)}")
            raise

    def execute_values_query(self, query: str, data: List[tuple]) -> None:
        """여러 행의 데이터를 한 번에 삽입"""
        try:
            execute_values(self.cur, query, data)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"데이터 삽입 중 오류 발생: {str(e)}")
            raise

    def close(self) -> None:
        """데이터베이스 연결 종료"""
        try:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()
            logger.info("데이터베이스 연결 종료")
        except Exception as e:
            logger.error(f"데이터베이스 연결 종료 중 오류 발생: {str(e)}")
            raise

# 전역 데이터베이스 매니저
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """데이터베이스 매니저 싱글톤 인스턴스 반환"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

# 편의 함수들
def execute_query(query: str, params: Optional[tuple] = None, fetch: bool = True) -> List[tuple]:
    """Execute a SQL query and optionally fetch results"""
    db = get_database_manager()
    return db.execute_query(query, params, fetch)

def execute_values_query(query: str, data: List[tuple]) -> None:
    """Insert multiple rows of data at once"""
    db = get_database_manager()
    db.execute_values_query(query, data)

def execute_transaction(queries: List[tuple]) -> None:
    """Execute a transaction with multiple queries"""
    db = get_database_manager()
    try:
        for query, params in queries:
            if params is None:
                db.cur.execute(query)
            else:
                db.cur.execute(query, params)
        db.conn.commit()
    except Exception as e:
        db.conn.rollback()
        logger.error(f"트랜잭션 실행 중 오류 발생: {str(e)}")
        raise

def create_stock_prices_table():
    """주가 데이터를 저장할 테이블을 생성합니다."""
    try:
        db = get_database_manager()
        db.create_tables()
        logger.info("Stock prices table created successfully!")
    except Exception as e:
        logger.error(f"Failed to create stock prices table: {str(e)}")
        raise

def _is_trading_day(date_str: str) -> bool:
    """지정한 일자에 일일 OHLCV가 존재하는지 삼성전자(005930)로 빠르게 확인"""
    try:
        df_probe = stock.get_market_ohlcv_by_date(date_str, date_str, '005930')
        return not df_probe.empty
    except Exception as e:
        logger.warning(f"거래일 확인 중 오류 발생: {e}")
        return False

def get_date_range():
    """한국 시간 기준으로 오늘이 거래일이면 오늘, 아니면 가장 최근 거래일 반환"""
    korea_tz = pytz.timezone('Asia/Seoul')
    today_korea = datetime.now(korea_tz)

    # 오늘이 주말인 경우 최근 평일 반환
    if today_korea.weekday() >= 5:
        logger.warning(f"⚠️ 오늘은 주말입니다 ({today_korea.strftime('%Y-%m-%d %A')})")
        for i in range(1, 8):
            d = (today_korea - timedelta(days=i)).strftime('%Y%m%d')
            if _is_trading_day(d):
                logger.info(f"📅 최근 거래일: {datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d %A')}")
                return d, d

    # 평일이면 오늘 데이터 존재 여부를 프로빙 후 결정
    today_str = today_korea.strftime('%Y%m%d')
    if _is_trading_day(today_str):
        logger.info(f"📅 오늘 날짜 사용 (한국 시간): {today_korea.strftime('%Y-%m-%d %A')}")
        return today_str, today_str
    else:
        logger.info("📅 오늘 일자 데이터가 없어 최근 거래일로 대체합니다")
        for i in range(1, 8):
            d = (today_korea - timedelta(days=i)).strftime('%Y%m%d')
            if _is_trading_day(d):
                logger.info(f"📅 최근 거래일: {datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d %A')}")
                return d, d

    # 최악의 경우 오늘 반환
    return today_str, today_str

def fetch_stock_data(stock_code, start_date, end_date):
    """단일 종목의 주가 데이터를 가져옵니다."""
    try:
        # 종목코드에서 'A' 접두사 제거
        clean_code = stock_code.replace('A', '')
        
        logger.info(f"  - 주가 데이터 가져오기 시도 중...")
        logger.info(f"  - 조회일: {start_date}")
        
        # 종목명 먼저 확인
        stock_name = stock.get_market_ticker_name(clean_code)
        if not stock_name:
            logger.warning(f"  - 유효하지 않은 종목코드")
            return None, 0
        logger.info(f"  - 종목명: {stock_name}")
        
        # 주가 데이터 가져오기
        df = stock.get_market_ohlcv_by_date(start_date, end_date, clean_code)
        
        # 데이터가 없는 경우
        if df.empty:
            logger.warning(f"  - {start_date} 데이터가 없음 (거래일이 아님)")
            return None, 0
        
        # 컬럼명 정리 (pykrx가 상황에 따라 '거래대금' 등 추가 컬럼을 포함할 수 있음)
        df = df.reset_index()
        # 필요한 첫 6개 컬럼만 사용: [date, open, high, low, close, volume]
        if len(df.columns) < 6:
            logger.warning(f"  - 예상보다 적은 컬럼 수: {len(df.columns)}")
            return None, 0
        df = df.iloc[:, :6]
        df.columns = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        
        # 종목코드와 종목명 추가
        df['stock_code'] = stock_code
        df['stock_name'] = stock_name
        
        # 날짜 형식 변환
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # 필요한 컬럼만 선택
        df = df[['date', 'stock_code', 'stock_name', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        
        logger.info(f"  - {start_date} 주가 데이터 가져오기 성공!")
        logger.debug(f"  - 데이터 샘플:\n{df.head()}")
        return df, len(df)
        
    except json.JSONDecodeError as e:
        logger.error(f"  - API 응답 파싱 실패: {str(e)}")
        logger.error(f"  - API 응답이 유효한 JSON 형식이 아닙니다.")
        # JSON 파싱 실패 시 재시도 필요
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"  - API 요청 실패: {str(e)}")
        logger.error(f"  - 네트워크 오류 또는 API 서버 문제")
        raise
    except Exception as e:
        logger.error(f"  - 데이터 가져오기 실패: {str(e)}")
        logger.error(f"  - 상세 정보: {type(e).__name__}")
        return None, 0

def fetch_stock_data_with_retry(stock_code, start_date, end_date, max_retries=2, delay=1):
    """재시도 로직이 포함된 주가 데이터 수집"""
    for attempt in range(max_retries):
        try:
            # API 호출 간격 조절 (rate limiting 방지)
            if attempt > 0:
                sleep_time = delay + random.uniform(0.1, 0.5)
                time.sleep(sleep_time)
            
            return fetch_stock_data(stock_code, start_date, end_date)
            
        except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
            if attempt == max_retries - 1:
                logger.error(f"종목 {stock_code} 최대 재시도 횟수 초과: {e}")
                return None, 0
            else:
                logger.warning(f"종목 {stock_code} {attempt+1}번째 시도 실패, 재시도 중...: {e}")
                continue
        except Exception as e:
            logger.error(f"종목 {stock_code} 예상치 못한 오류: {e}")
            return None, 0
    
    return None, 0

def delete_existing_data(today_str):
    """오늘 날짜의 기존 데이터를 삭제합니다."""
    query = "DELETE FROM stock_prices WHERE time = %s;"
    execute_query(query, (today_str,))
    logger.info(f"🗑️ {today_str} 기존 데이터 삭제 완료")

def get_stock_list():
    """데이터베이스에서 종목 리스트를 가져옵니다."""
    try:
        # 데이터베이스에서 종목 리스트 가져오기
        query = "SELECT stock_code, stock_name FROM stock_items WHERE is_kospi200 = TRUE OR is_related = TRUE;"
        results = execute_query(query)
        stock_list = pd.DataFrame(results, columns=['stock_code', 'stock_name'])
        logger.info(f"📋 종목 리스트 로드 완료: {len(stock_list)}개 종목")
        return stock_list
    except Exception as e:
        logger.error(f"종목 리스트를 가져올 수 없습니다: {e}")
        return None

def process_single_stock(args):
    """단일 종목 처리 함수 (병렬 처리용)"""
    idx, row, start_date, end_date = args
    stock_code = row['stock_code']
    stock_name = row['stock_name']
    
    logger.info(f"🔄 ({idx+1}) {stock_name}({stock_code}) {start_date} 데이터 수집 중...")
    
    # API 호출 간격 조절 (rate limiting 방지)
    if idx > 0:
        time.sleep(random.uniform(0.05, 0.2))
    
    df, count = fetch_stock_data_with_retry(stock_code, start_date, end_date)
    
    if df is not None and not df.empty:
        logger.info(f"✅ {stock_name}({stock_code}) - {start_date} 데이터 {count}개 수집 완료")
        return df, True
    else:
        logger.warning(f"❌ {stock_name}({stock_code}) - {start_date} 데이터 없음")
        return None, False

def fetch_stock_prices():
    """오늘 주가 데이터를 가져와 데이터베이스에 저장"""
    logger.info("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
    
    # 시작일과 종료일 설정 (한국 시간 기준)
    start_date, end_date = get_date_range()
    logger.info(f"📆 조회일: {start_date}")
    
    # 기존 데이터 삭제
    today_str = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
    delete_existing_data(today_str)
    
    # 종목 리스트 가져오기
    stock_list = get_stock_list()
    if stock_list is None or stock_list.empty:
        logger.error("❌ 종목 리스트를 가져올 수 없습니다.")
        return None
    
    # 병렬 처리를 위한 함수
    def process_stock_batch(stock_batch):
        """종목 배치 처리"""
        results = []
        for args in stock_batch:
            result = process_single_stock(args)
            results.append(result)
        return results
    
    # 병렬 처리로 데이터 수집
    all_stock_data = []
    success_count = 0
    fail_count = 0
    
    # 스레드 풀 크기 설정 (API 제한 고려)
    max_workers = 10
    
    # 종목 리스트를 배치로 나누기
    batch_size = max_workers
    stock_batches = []
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list.iloc[i:i+batch_size]
        batch_args = [(i+j, row, start_date, end_date) for j, (_, row) in enumerate(batch.iterrows())]
        stock_batches.append(batch_args)
    
    # 병렬 처리 실행
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 배치를 병렬로 처리
        future_to_batch = {executor.submit(process_stock_batch, batch): batch for batch in stock_batches}
        
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                for df, success in batch_results:
                    if success and df is not None:
                        all_stock_data.append(df)
                        success_count += 1
                    else:
                        fail_count += 1
            except Exception as e:
                logger.error(f"배치 처리 중 오류 발생: {e}")
                fail_count += len(future_to_batch[future])
    
    # 데이터 저장
    if all_stock_data:
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        # 데이터베이스에 저장할 데이터 준비
        query = """
            INSERT INTO stock_prices (
                time, stock_code, stock_name, open_price, high_price, 
                low_price, close_price, volume
            ) VALUES %s
        """
        
        data = [(
            row['date'], row['stock_code'], row['stock_name'], 
            row['open_price'], row['high_price'], row['low_price'], 
            row['close_price'], row['volume']
        ) for _, row in combined_df.iterrows()]
        
        execute_values_query(query, data)
        
        logger.info(f"\n💾 {start_date} 데이터가 데이터베이스에 저장되었습니다.")
        logger.info(f"📊 통계:")
        logger.info(f"   - 성공한 종목 수: {success_count}/{len(stock_list)}")
        logger.info(f"   - 실패한 종목 수: {fail_count}")
        logger.info(f"   - 수집된 데이터 레코드 수: {len(combined_df)}개")
        
        return combined_df
    else:
        logger.warning(f"❌ 저장할 {start_date} 데이터가 없습니다.")
        return None

def cleanup():
    """리소스 정리"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None

def main():
    """메인 실행 함수"""
    try:
        logger.info("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
        
        # 테이블 생성
        create_stock_prices_table()
        
        # 데이터 수집 및 저장
        stock_data = fetch_stock_prices()
        
        if stock_data is not None:
            logger.info("✅ 데이터 수집 및 저장 완료!")
        else:
            logger.error("❌ 데이터 수집 실패!")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    main()
