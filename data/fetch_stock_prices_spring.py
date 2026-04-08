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
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logs_dir = Path(__file__).parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / 'stock_prices_spring.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 현재 파일의 절대 경로
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# 프로젝트 루트 디렉토리를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pykrx import stock

class HikariConfig:
    """HikariCP 설정 클래스 (Spring 스타일)"""
    
    def __init__(self):
        self.jdbc_url = None
        self.username = None
        self.password = None
        self.driver_class_name = None
        self.maximum_pool_size = 10
        self.minimum_idle = 5
        self.connection_timeout = 30000
        self.idle_timeout = 600000
        self.max_lifetime = 1800000
    
    def set_jdbc_url(self, url: str):
        self.jdbc_url = url
    
    def set_username(self, username: str):
        self.username = username
    
    def set_password(self, password: str):
        self.password = password
    
    def set_driver_class_name(self, driver: str):
        self.driver_class_name = driver
    
    def set_maximum_pool_size(self, size: int):
        self.maximum_pool_size = size
    
    def set_minimum_idle(self, idle: int):
        self.minimum_idle = idle
    
    def set_connection_timeout(self, timeout: int):
        self.connection_timeout = timeout
    
    def set_idle_timeout(self, timeout: int):
        self.idle_timeout = timeout
    
    def set_max_lifetime(self, lifetime: int):
        self.max_lifetime = lifetime

class HikariDataSource:
    """HikariCP 스타일의 연결 풀"""
    
    def __init__(self, config: HikariConfig):
        self.config = config
        self.connection_pool = []
        self.active_connections = set()
        self.connection_lock = threading.Lock()
        self.is_closed = False
        
        # 연결 풀 초기화
        self._initialize_pool()
        logger.info(f"HikariDataSource initialized with max pool size: {config.maximum_pool_size}")
    
    def _initialize_pool(self):
        """연결 풀 초기화"""
        for _ in range(self.config.minimum_idle):
            try:
                conn = self._create_connection()
                self.connection_pool.append(conn)
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
    
    def _create_connection(self):
        """새로운 데이터베이스 연결 생성"""
        # JDBC URL을 PostgreSQL 연결 문자열로 변환
        if self.config.jdbc_url and self.config.jdbc_url.startswith("jdbc:postgresql://"):
            url_part = self.config.jdbc_url.replace("jdbc:postgresql://", "")
            if "/" in url_part:
                host_port, database = url_part.split("/", 1)
                if ":" in host_port:
                    host, port = host_port.split(":")
                    port = int(port)
                else:
                    host = host_port
                    port = 5432
            else:
                host = url_part
                port = 5432
                database = os.getenv('DB_NAME')
        else:
            host = os.getenv('DB_HOST')
            port = int(os.getenv('DB_PORT', 5432))
            database = os.getenv('DB_NAME')
        
        return psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=self.config.username,
            password=self.config.password
        )
    
    def get_connection(self):
        """연결 풀에서 연결 가져오기"""
        if self.is_closed:
            raise Exception("DataSource is closed")
        
        with self.connection_lock:
            if self.connection_pool:
                conn = self.connection_pool.pop()
                self.active_connections.add(conn)
                return conn
            else:
                if len(self.active_connections) < self.config.maximum_pool_size:
                    conn = self._create_connection()
                    self.active_connections.add(conn)
                    return conn
                else:
                    raise Exception("Connection pool exhausted")
    
    def return_connection(self, conn):
        """연결을 풀에 반환"""
        if self.is_closed:
            try:
                conn.close()
            except:
                pass
            return
        
        with self.connection_lock:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                try:
                    # 연결이 유효한지 확인
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    
                    # 풀에 반환
                    if len(self.connection_pool) < self.config.maximum_pool_size:
                        self.connection_pool.append(conn)
                    else:
                        conn.close()
                except:
                    try:
                        conn.close()
                    except:
                        pass
    
    def close(self):
        """데이터소스 종료"""
        self.is_closed = True
        
        with self.connection_lock:
            for conn in list(self.active_connections):
                try:
                    conn.close()
                except:
                    pass
            self.active_connections.clear()
            
            for conn in self.connection_pool:
                try:
                    conn.close()
                except:
                    pass
            self.connection_pool.clear()
        
        logger.info("HikariDataSource closed")

class JdbcTemplate:
    """Spring JdbcTemplate 스타일의 쿼리 실행 클래스"""
    
    def __init__(self, data_source: HikariDataSource):
        self.data_source = data_source
        logger.info("JdbcTemplate initialized")
    
    def execute(self, sql: str, *args):
        """INSERT, UPDATE, DELETE 쿼리 실행"""
        conn = None
        try:
            conn = self.data_source.get_connection()
            cursor = conn.cursor()
            
            if args:
                cursor.execute(sql, args)
            else:
                cursor.execute(sql)
            
            conn.commit()
            logger.debug(f"Query executed successfully: {sql[:50]}...")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Query execution failed: {str(e)}")
            raise
        finally:
            if conn:
                cursor.close()
                self.data_source.return_connection(conn)
    
    def query_for_list(self, sql: str, *args) -> List[Tuple]:
        """SELECT 쿼리 실행하여 리스트 반환"""
        conn = None
        try:
            conn = self.data_source.get_connection()
            cursor = conn.cursor()
            
            if args:
                cursor.execute(sql, args)
            else:
                cursor.execute(sql)
            
            results = cursor.fetchall()
            logger.debug(f"Query executed successfully, returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
        finally:
            if conn:
                cursor.close()
                self.data_source.return_connection(conn)
    
    def batch_update(self, sql: str, batch_args: List[Tuple]) -> List[int]:
        """배치 업데이트 실행"""
        conn = None
        try:
            conn = self.data_source.get_connection()
            cursor = conn.cursor()
            
            execute_values(cursor, sql, batch_args)
            conn.commit()
            logger.info(f"Batch update executed successfully: {len(batch_args)} rows")
            return [1] * len(batch_args)
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Batch update failed: {str(e)}")
            raise
        finally:
            if conn:
                cursor.close()
                self.data_source.return_connection(conn)

class AppConfig:
    """Spring Configuration 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("AppConfig initialized")
    
    def data_source(self) -> HikariDataSource:
        """DataSource Bean 생성"""
        config = HikariConfig()
        
        # 환경 변수에서 설정값 가져오기
        url = os.getenv('DB_URL', f"jdbc:postgresql://{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
        username = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        driver = "org.postgresql.Driver"
        
        config.set_jdbc_url(url)
        config.set_username(username)
        config.set_password(password)
        config.set_driver_class_name(driver)
        config.set_maximum_pool_size(10)
        
        return HikariDataSource(config)
    
    def jdbc_template(self, data_source: HikariDataSource) -> JdbcTemplate:
        """JdbcTemplate Bean 생성"""
        return JdbcTemplate(data_source)

class StockPriceRepository:
    """주가 데이터 Repository 클래스"""
    
    def __init__(self, jdbc_template: JdbcTemplate):
        self.jdbc_template = jdbc_template
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("StockPriceRepository initialized")
    
    def create_tables(self):
        """필요한 테이블 생성"""
        create_table_query = """
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
        """
        self.jdbc_template.execute(create_table_query)
        self.logger.info("Tables created successfully")
    
    def delete_existing_data(self, date_str: str):
        """기존 데이터 삭제"""
        query = "DELETE FROM stock_prices WHERE time = %s;"
        self.jdbc_template.execute(query, date_str)
        self.logger.info(f"🗑️ {date_str} 기존 데이터 삭제 완료")
    
    def get_stock_list(self) -> pd.DataFrame:
        """종목 리스트 조회"""
        query = "SELECT stock_code, stock_name FROM stock_items WHERE is_kospi200 = TRUE OR is_related = TRUE;"
        results = self.jdbc_template.query_for_list(query)
        stock_list = pd.DataFrame(results, columns=['stock_code', 'stock_name'])
        self.logger.info(f"📋 종목 리스트 로드 완료: {len(stock_list)}개 종목")
        return stock_list
    
    def save_stock_data(self, data: List[Tuple]):
        """주가 데이터 저장"""
        query = """
            INSERT INTO stock_prices (
                time, stock_code, stock_name, open_price, high_price, 
                low_price, close_price, volume
            ) VALUES %s
        """
        self.jdbc_template.batch_update(query, data)
        self.logger.info(f"💾 {len(data)}개 데이터 저장 완료")

class StockDataService:
    """주가 데이터 수집 서비스"""
    
    def __init__(self, repository: StockPriceRepository):
        self.repository = repository
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("StockDataService initialized")
    
    def get_date_range(self):
        """한국 시간 기준으로 오늘 날짜만 가져오기 위한 날짜 범위 계산"""
        korea_tz = pytz.timezone('Asia/Seoul')
        today_korea = datetime.now(korea_tz)
        
        if today_korea.weekday() >= 5:  # 주말
            self.logger.warning(f"⚠️ 오늘은 주말입니다 ({today_korea.strftime('%Y-%m-%d %A')})")
            self.logger.info("📅 가장 최근 거래일을 확인합니다...")
            
            for i in range(1, 8):
                check_date = today_korea - timedelta(days=i)
                if check_date.weekday() < 5:
                    self.logger.info(f"📅 최근 거래일: {check_date.strftime('%Y-%m-%d %A')}")
                    return check_date.strftime('%Y%m%d'), check_date.strftime('%Y%m%d')
        
        self.logger.info(f"📅 오늘 날짜 (한국 시간): {today_korea.strftime('%Y-%m-%d %A')}")
        return today_korea.strftime('%Y%m%d'), today_korea.strftime('%Y%m%d')
    
    def fetch_stock_data(self, stock_code: str, start_date: str, end_date: str):
        """단일 종목의 주가 데이터를 가져옵니다."""
        try:
            clean_code = stock_code.replace('A', '')
            
            self.logger.info(f"  - 주가 데이터 가져오기 시도 중...")
            self.logger.info(f"  - 조회일: {start_date}")
            
            stock_name = stock.get_market_ticker_name(clean_code)
            if not stock_name:
                self.logger.warning(f"  - 유효하지 않은 종목코드")
                return None, 0
            self.logger.info(f"  - 종목명: {stock_name}")
            
            df = stock.get_market_ohlcv_by_date(start_date, end_date, clean_code)
            
            if df.empty:
                self.logger.warning(f"  - {start_date} 데이터가 없음 (거래일이 아님)")
                return None, 0
            
            df = df.reset_index()
            df.columns = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
            df['stock_code'] = stock_code
            df['stock_name'] = stock_name
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[['date', 'stock_code', 'stock_name', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
            
            self.logger.info(f"  - {start_date} 주가 데이터 가져오기 성공!")
            return df, len(df)
            
        except Exception as e:
            self.logger.error(f"  - 데이터 가져오기 실패: {str(e)}")
            return None, 0
    
    def fetch_stock_data_with_retry(self, stock_code: str, start_date: str, end_date: str, max_retries=2, delay=1):
        """재시도 로직이 포함된 주가 데이터 수집"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    sleep_time = delay + random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)
                
                return self.fetch_stock_data(stock_code, start_date, end_date)
                
            except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"종목 {stock_code} 최대 재시도 횟수 초과: {e}")
                    return None, 0
                else:
                    self.logger.warning(f"종목 {stock_code} {attempt+1}번째 시도 실패, 재시도 중...: {e}")
                    continue
            except Exception as e:
                self.logger.error(f"종목 {stock_code} 예상치 못한 오류: {e}")
                return None, 0
        
        return None, 0
    
    def process_single_stock(self, args):
        """단일 종목 처리 함수 (병렬 처리용)"""
        idx, row, start_date, end_date = args
        stock_code = row['stock_code']
        stock_name = row['stock_name']
        
        self.logger.info(f"🔄 ({idx+1}) {stock_name}({stock_code}) {start_date} 데이터 수집 중...")
        
        if idx > 0:
            time.sleep(random.uniform(0.05, 0.2))
        
        df, count = self.fetch_stock_data_with_retry(stock_code, start_date, end_date)
        
        if df is not None and not df.empty:
            self.logger.info(f"✅ {stock_name}({stock_code}) - {start_date} 데이터 {count}개 수집 완료")
            return df, True
        else:
            self.logger.warning(f"❌ {stock_name}({stock_code}) - {start_date} 데이터 없음")
            return None, False
    
    def fetch_all_stock_prices(self):
        """모든 주가 데이터를 가져와 데이터베이스에 저장"""
        self.logger.info("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
        
        start_date, end_date = self.get_date_range()
        self.logger.info(f"📆 조회일: {start_date}")
        
        today_str = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
        self.repository.delete_existing_data(today_str)
        
        stock_list = self.repository.get_stock_list()
        if stock_list is None or stock_list.empty:
            self.logger.error("❌ 종목 리스트를 가져올 수 없습니다.")
            return None
        
        # 병렬 처리로 데이터 수집
        all_stock_data = []
        success_count = 0
        fail_count = 0
        max_workers = 10
        
        batch_size = max_workers
        stock_batches = []
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list.iloc[i:i+batch_size]
            batch_args = [(i+j, row, start_date, end_date) for j, (_, row) in enumerate(batch.iterrows())]
            stock_batches.append(batch_args)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(self._process_batch, batch): batch for batch in stock_batches}
            
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
                    self.logger.error(f"배치 처리 중 오류 발생: {e}")
                    fail_count += len(future_to_batch[future])
        
        # 데이터 저장
        if all_stock_data:
            combined_df = pd.concat(all_stock_data, ignore_index=True)
            
            data = [(
                row['date'], row['stock_code'], row['stock_name'], 
                row['open_price'], row['high_price'], row['low_price'], 
                row['close_price'], row['volume']
            ) for _, row in combined_df.iterrows()]
            
            self.repository.save_stock_data(data)
            
            self.logger.info(f"\n💾 {start_date} 데이터가 데이터베이스에 저장되었습니다.")
            self.logger.info(f"📊 통계:")
            self.logger.info(f"   - 성공한 종목 수: {success_count}/{len(stock_list)}")
            self.logger.info(f"   - 실패한 종목 수: {fail_count}")
            self.logger.info(f"   - 수집된 데이터 레코드 수: {len(combined_df)}개")
            
            return combined_df
        else:
            self.logger.warning(f"❌ 저장할 {start_date} 데이터가 없습니다.")
            return None
    
    def _process_batch(self, batch_args):
        """종목 배치 처리"""
        results = []
        for args in batch_args:
            result = self.process_single_stock(args)
            results.append(result)
        return results

class StockPriceApplication:
    """Spring 스타일의 메인 애플리케이션 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("StockPriceApplication initialized")
        
        # Spring 스타일 Bean 생성
        self.app_config = AppConfig()
        self.data_source = self.app_config.data_source()
        self.jdbc_template = self.app_config.jdbc_template(self.data_source)
        self.repository = StockPriceRepository(self.jdbc_template)
        self.stock_service = StockDataService(self.repository)
    
    def run(self):
        """애플리케이션 실행"""
        try:
            self.logger.info("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
            
            # 테이블 생성
            self.repository.create_tables()
            
            # 데이터 수집 및 저장
            result = self.stock_service.fetch_all_stock_prices()
            
            if result is not None:
                self.logger.info("✅ 데이터 수집 및 저장 완료!")
                return True
            else:
                self.logger.error("❌ 데이터 수집 실패!")
                return False
                
        except Exception as e:
            self.logger.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")
            return False
        finally:
            self.close()
    
    def close(self):
        """리소스 정리"""
        try:
            self.data_source.close()
            self.logger.info("애플리케이션 리소스 정리 완료")
        except Exception as e:
            self.logger.error(f"리소스 정리 중 오류 발생: {str(e)}")

def main():
    """메인 실행 함수"""
    app = StockPriceApplication()
    success = app.run()
    
    if success:
        logger.info("✅ 애플리케이션 실행 완료!")
    else:
        logger.error("❌ 애플리케이션 실행 실패!")

if __name__ == "__main__":
    main()
