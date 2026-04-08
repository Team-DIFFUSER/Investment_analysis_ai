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

# 로깅 설정
# logs 디렉토리 생성
logs_dir = Path(__file__).parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 콘솔 출력
        logging.FileHandler(logs_dir / 'stock_prices_spring.log', encoding='utf-8')  # 파일 출력
    ]
)
logger = logging.getLogger(__name__)

# 현재 파일의 절대 경로
current_file = Path(__file__).resolve()
# 프로젝트 루트 디렉토리의 절대 경로
project_root = current_file.parent.parent

# 프로젝트 루트 디렉토리를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database.spring_database import execute_query, execute_values_query, execute_transaction, create_tables
from pykrx import stock

def create_stock_prices_table():
    """주가 데이터를 저장할 테이블을 생성합니다."""
    try:
        create_tables()
        logger.info("Stock prices table created successfully!")
    except Exception as e:
        logger.error(f"Failed to create stock prices table: {str(e)}")
        raise

def get_date_range():
    """한국 시간 기준으로 오늘 날짜만 가져오기 위한 날짜 범위 계산"""
    # 한국 시간대 설정
    korea_tz = pytz.timezone('Asia/Seoul')
    today_korea = datetime.now(korea_tz)
    
    # 주말인지 확인
    if today_korea.weekday() >= 5:  # 5=토요일, 6=일요일
        logger.warning(f"⚠️ 오늘은 주말입니다 ({today_korea.strftime('%Y-%m-%d %A')})")
        logger.info("📅 가장 최근 거래일을 확인합니다...")
        
        # 가장 최근 거래일 찾기 (최대 7일 전까지)
        for i in range(1, 8):
            check_date = today_korea - timedelta(days=i)
            if check_date.weekday() < 5:  # 평일인 경우
                logger.info(f"📅 최근 거래일: {check_date.strftime('%Y-%m-%d %A')}")
                return check_date.strftime('%Y%m%d'), check_date.strftime('%Y%m%d')
    
    logger.info(f"📅 오늘 날짜 (한국 시간): {today_korea.strftime('%Y-%m-%d %A')}")
    return today_korea.strftime('%Y%m%d'), today_korea.strftime('%Y%m%d')

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
        
        # 컬럼명 정리
        df = df.reset_index()
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

class StockPriceService:
    """스프링 스타일의 주가 데이터 서비스 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("StockPriceService initialized")
    
    def fetch_and_save_stock_prices(self):
        """주가 데이터를 가져와서 저장하는 메인 메서드"""
        try:
            self.logger.info("Starting stock price data collection...")
            
            # 테이블 생성
            create_stock_prices_table()
            
            # 데이터 수집 및 저장
            result = fetch_stock_prices()
            
            if result is not None:
                self.logger.info("✅ Stock price data collection completed successfully!")
                return True
            else:
                self.logger.error("❌ Stock price data collection failed!")
                return False
                
        except Exception as e:
            self.logger.error(f"Stock price data collection failed: {str(e)}")
            return False

def main():
    """메인 실행 함수"""
    try:
        logger.info("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
        
        # 스프링 스타일 서비스 사용
        stock_service = StockPriceService()
        success = stock_service.fetch_and_save_stock_prices()
        
        if success:
            logger.info("✅ 데이터 수집 및 저장 완료!")
        else:
            logger.error("❌ 데이터 수집 실패!")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
