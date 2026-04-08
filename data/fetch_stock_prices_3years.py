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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 현재 파일의 절대 경로
current_file = Path(__file__).resolve()
# 프로젝트 루트 디렉토리의 절대 경로
project_root = current_file.parent.parent

# 프로젝트 루트 디렉토리를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database.database import execute_query, execute_values_query, execute_transaction
from pykrx import stock

def drop_stock_prices_table():
    """기존 주가 테이블을 삭제합니다."""
    query = "DROP TABLE IF EXISTS stock_prices CASCADE;"
    execute_query(query)
    print("🗑️ 기존 stock_prices 테이블 삭제 완료")

def create_stock_prices_table():
    """주가 데이터를 저장할 테이블을 생성합니다."""
    query = """
    CREATE TABLE IF NOT EXISTS stock_prices (
        time DATE,
        stock_code VARCHAR(20),
        stock_name VARCHAR(100),
        open_price DECIMAL(20,2),
        high_price DECIMAL(20,2),
        low_price DECIMAL(20,2),
        close_price DECIMAL(20,2),
        volume DECIMAL(20,2),
        market_cap DECIMAL(20,2),
        foreign_holding DECIMAL(20,2),
        foreign_holding_ratio DECIMAL(20,2),
        PRIMARY KEY (time, stock_code)
    );
    """
    execute_query(query)
    print("✅ Stock prices table created successfully!")

def get_date_range_3years():
    """3년치 데이터를 위한 날짜 범위 계산 (8월 28일까지)"""
    korea_tz = pytz.timezone('Asia/Seoul')
    end_date = datetime(2025, 9, 23)  # 8월 28일까지
    start_date = end_date - timedelta(days=4*365)  # 3년 전
    
    return start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')

def clean_stock_code(stock_code):
    """종목코드에서 'A' 접두사 제거"""
    return stock_code.replace('A', '')

def fetch_stock_data_with_retry(stock_code, start_date, end_date, max_retries=3, delay=2):
    """재시도 로직이 포함된 주가 데이터 수집"""
    for attempt in range(max_retries):
        try:
            # API 호출 간격 조절 (rate limiting 방지)
            if attempt > 0:
                sleep_time = delay + random.uniform(1.0, 3.0)
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

def fetch_stock_data(stock_code, start_date, end_date):
    """개별 종목의 주가 데이터를 가져옵니다."""
    try:
        # 종목코드 정리 (A 제거)
        clean_code = stock_code.replace('A', '')
        
        print(f"  - 주가 데이터 가져오기 시도 중...")
        print(f"  - 조회기간: {start_date} ~ {end_date}")
        
        try:
            # 종목명 먼저 확인
            stock_name = stock.get_market_ticker_name(clean_code)
            if not stock_name:
                print(f"  - 유효하지 않은 종목코드")
                return None, 0
            print(f"  - 종목명: {stock_name}")
            
            # 주가 데이터 가져오기 (3년치)
            df = stock.get_market_ohlcv_by_date(start_date, end_date, clean_code)
            
            # 데이터가 없는 경우
            if df.empty:
                print(f"  - {start_date}~{end_date} 기간에 데이터가 없음")
                return None, 0
                
            # 컬럼명 변경
            column_names = {
                '시가': 'open_price',
                '고가': 'high_price',
                '저가': 'low_price',
                '종가': 'close_price',
                '거래량': 'volume'
            }
            df = df.rename(columns=column_names)
            
            # 종목코드와 종목명 추가
            df['stock_code'] = stock_code
            df['stock_name'] = stock_name
            
            # 날짜를 인덱스에서 컬럼으로 변경
            df = df.reset_index()
            df = df.rename(columns={'날짜': 'time'})
            
            # 시가총액 데이터 가져오기 (선택적)
            try:
                market_cap = stock.get_market_cap_by_date(start_date, end_date, clean_code)
                if not market_cap.empty and '시가총액' in market_cap.columns:
                    # 시가총액 데이터를 주가 데이터와 병합
                    market_cap = market_cap.reset_index()
                    market_cap = market_cap.rename(columns={'날짜': 'time', '시가총액': 'market_cap'})
                    df = df.merge(market_cap[['time', 'market_cap']], on='time', how='left')
                else:
                    df['market_cap'] = None
            except Exception as e:
                print(f"  - 시가총액 데이터 가져오기 실패: {str(e)}")
                df['market_cap'] = None
            
            # 외국인/기관 보유량 데이터 가져오기 (선택적)
            try:
                foreign_holding = stock.get_exhaustion_rates_of_foreign_investment_by_ticker(clean_code, start_date, end_date)
                if not foreign_holding.empty and '외국인보유량' in foreign_holding.columns:
                    # 외국인 보유량 데이터를 주가 데이터와 병합
                    foreign_holding = foreign_holding.reset_index()
                    foreign_holding = foreign_holding.rename(columns={
                        '날짜': 'time', 
                        '외국인보유량': 'foreign_holding',
                        '외국인보유비율': 'foreign_holding_ratio'
                    })
                    df = df.merge(foreign_holding[['time', 'foreign_holding', 'foreign_holding_ratio']], on='time', how='left')
                else:
                    df['foreign_holding'] = None
                    df['foreign_holding_ratio'] = None
            except Exception as e:
                print(f"  - 외국인 보유량 데이터 가져오기 실패: {str(e)}")
                df['foreign_holding'] = None
                df['foreign_holding_ratio'] = None
            
            print(f"  - {start_date}~{end_date} 주가 데이터 가져오기 성공!")
            print(f"  - 수집된 데이터 수: {len(df)}개")
            return df, len(df)
            
        except json.JSONDecodeError as e:
            print(f"  - API 응답 파싱 실패: {str(e)}")
            print(f"  - API 응답이 유효한 JSON 형식이 아닙니다.")
            # JSON 파싱 실패 시 재시도 필요
            raise
        except requests.exceptions.RequestException as e:
            print(f"  - API 요청 실패: {str(e)}")
            print(f"  - 네트워크 오류 또는 API 서버 문제")
            raise
        except Exception as e:
            print(f"  - 데이터 가져오기 실패: {str(e)}")
            print(f"  - 상세 정보: {type(e).__name__}")
            return None, 0
        
    except Exception as e:
        print(f"데이터 수집 중 오류 발생: {e}")
        print(f"상세 정보: {type(e).__name__}")
        return None, 0

def fetch_stock_prices_3years():
    """3년치 주가 데이터를 가져와 데이터베이스에 저장"""
    print("📢 KOSPI200 3년치 주가 데이터를 가져오는 중...")
    
    # 시작일과 종료일 설정 (3년치, 8월 28일까지)
    start_date, end_date = get_date_range_3years()
    print(f"📆 조회기간: {start_date} ~ {end_date}")
    
    # 기존 테이블 삭제 후 새로 생성
    drop_stock_prices_table()
    create_stock_prices_table()
    
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
        
        print(f"🔄 ({idx+1}/{len(stock_list)}) {stock_name}({stock_code}) 3년치 데이터 수집 중...")
        
        # API 호출 간격 조절 (rate limiting 방지)
        if idx > 0:
            time.sleep(random.uniform(1.0, 2.0))
        
        df, count = fetch_stock_data_with_retry(stock_code, start_date, end_date)
        
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
        
        # BIGINT 범위 초과값 처리
        combined_df['volume'] = combined_df['volume'].fillna(0)
        
        # NaN 값을 None으로 변경 (PostgreSQL 호환성)
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.where(pd.notnull(combined_df), None)
        
        # 데이터베이스에 저장
        insert_query = """
        INSERT INTO stock_prices (
            time, stock_code, stock_name, open_price, high_price, low_price,
            close_price, volume, market_cap, foreign_holding, foreign_holding_ratio
        ) VALUES %s
        """
        
        data = [(
            row['time'], row['stock_code'], row['stock_name'],
            row['open_price'], row['high_price'], row['low_price'],
            row['close_price'], row['volume'], row['market_cap'],
            row['foreign_holding'], row['foreign_holding_ratio']
        ) for _, row in combined_df.iterrows()]
        
        execute_values_query(insert_query, data)
        
        print(f"\n💾 {start_date}~{end_date} 3년치 데이터가 데이터베이스에 저장되었습니다.")
        print(f"📊 통계:")
        print(f"   - 성공한 종목 수: {success_count}/{len(stock_list)}")
        print(f"   - 실패한 종목 수: {fail_count}")
        print(f"   - 수집된 데이터 레코드 수: {len(combined_df)}개")
        print("✅ 3년치 데이터 수집 및 저장 완료!")
        
        return combined_df
    else:
        print("❌ 수집된 데이터가 없습니다.")
        return None

if __name__ == "__main__":
    stock_data = fetch_stock_prices_3years()
