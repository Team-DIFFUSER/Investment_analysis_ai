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

# =================================================================================
# DEBUG: pykrx의 API 응답을 확인하기 위한 임시 코드 (v4 - Meta Debug)
# =================================================================================
print("!!! DEBUG-V4: PATCH CODE BLOCK IS RUNNING !!!") # Check if this block runs
import sys
# pykrx.stock.api를 임포트하면 내부적으로 pykrx.helper도 로드됩니다.
import pykrx.stock.api

print("!!! DEBUG-V4: pykrx modules in sys.modules: !!!")
pykrx_modules = [name for name in sys.modules if 'pykrx' in name]
print(pykrx_modules)


# 로드된 pykrx.helper 모듈을 sys.modules에서 직접 찾습니다.
if 'pykrx.helper' in sys.modules:
    print("!!! DEBUG-V4: Found 'pykrx.helper' in sys.modules. Applying patch. !!!")
    helper_module = sys.modules['pykrx.helper']

    # 원래 요청 함수를 직접 구현하여 중간에 응답을 확인합니다.
    def _debug_request_post(url, data, headers):
        response = requests.post(url, data=data, headers=headers)
        try:
            # JSON 파싱 시도
            return response.json()
        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시, 실제 응답 내용을 출력
            print("="*50)
            print("DEBUG: API가 JSON이 아닌 다른 응답을 반환했습니다.")
            print(f"URL: {url}")
            print(f"Request Data: {data}")
            print(f"Status Code: {response.status_code}")
            print("--- Response Text (first 500 chars) ---")
            print(response.text[:500])
            print("="*50)
            # 원래 에러를 다시 발생시켜 프로그램 흐름을 유지
            raise e

    # pykrx.helper 모듈 자체의 request_post 함수를 디버깅 함수로 교체합니다.
    helper_module.request_post = _debug_request_post
    print("!!! DEBUG-V4: Patch applied successfully. !!!")
else:
    print("!!! DEBUG-V4: COULD NOT FIND 'pykrx.helper' in sys.modules. Patch failed. !!!")
# =================================================================================


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
    print("Stock prices table created successfully!")

def get_date_range():
    """한국 시간 기준으로 오늘 날짜만 가져오기 위한 날짜 범위 계산"""
    # 한국 시간대 설정
    korea_tz = pytz.timezone('Asia/Seoul')
    today_korea = datetime.now(korea_tz)
    
    # 주말인지 확인
    if today_korea.weekday() >= 5:  # 5=토요일, 6=일요일
        print(f"⚠️ 오늘은 주말입니다 ({today_korea.strftime('%Y-%m-%d %A')})")
        print("📅 가장 최근 거래일을 확인합니다...")
        
        # 가장 최근 거래일 찾기 (최대 7일 전까지)
        for i in range(1, 8):
            check_date = today_korea - timedelta(days=i)
            if check_date.weekday() < 5:  # 평일인 경우
                print(f"📅 최근 거래일: {check_date.strftime('%Y-%m-%d %A')}")
                return check_date.strftime('%Y%m%d'), check_date.strftime('%Y%m%d')
    
    print(f"📅 오늘 날짜 (한국 시간): {today_korea.strftime('%Y-%m-%d %A')}")
    return today_korea.strftime('%Y%m%d'), today_korea.strftime('%Y%m%d')

def clean_stock_code(stock_code):
    """종목코드에서 'A' 접두사 제거"""
    return stock_code.replace('A', '')

def fetch_stock_data_with_retry(stock_code, start_date, end_date, max_retries=3, delay=2):
    """재시도 로직이 포함된 주가 데이터 수집"""
    for attempt in range(max_retries):
        try:
            # API 호출 간격 조절 (rate limiting 방지)
            if attempt > 0:
                sleep_time = delay + random.uniform(0.5, 2.0)
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
        print(f"  - 조회일: {start_date}")
        
        try:
            # 종목명 먼저 확인
            stock_name = stock.get_market_ticker_name(clean_code)
            if not stock_name:
                print(f"  - 유효하지 않은 종목코드")
                return None, 0
            print(f"  - 종목명: {stock_name}")
            
            # 주가 데이터 가져오기 (오늘 날짜만)
            df = stock.get_market_ohlcv_by_date(start_date, end_date, clean_code)
            
            # 데이터가 없는 경우
            if df.empty:
                print(f"  - {start_date} 데이터가 없음 (거래일이 아님)")
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
            
            # 시가총액 데이터 가져오기 (오늘 날짜만) - 에러 처리 강화
            try:
                market_cap = stock.get_market_cap_by_date(start_date, end_date, clean_code)
                if not market_cap.empty and '시가총액' in market_cap.columns:
                    df['market_cap'] = market_cap['시가총액']
                else:
                    df['market_cap'] = None
            except Exception as e:
                print(f"  - 시가총액 데이터 가져오기 실패: {str(e)}")
                df['market_cap'] = None
            
            # 외국인/기관 보유량 데이터 가져오기 (오늘 날짜만) - 에러 처리 강화
            try:
                foreign_holding = stock.get_exhaustion_rates_of_foreign_investment_by_ticker(clean_code, start_date, end_date)
                if not foreign_holding.empty and '외국인보유량' in foreign_holding.columns:
                    df['foreign_holding'] = foreign_holding['외국인보유량']
                    df['foreign_holding_ratio'] = foreign_holding['외국인보유비율']
                else:
                    df['foreign_holding'] = None
                    df['foreign_holding_ratio'] = None
            except Exception as e:
                print(f"  - 외국인 보유량 데이터 가져오기 실패: {str(e)}")
                df['foreign_holding'] = None
                df['foreign_holding_ratio'] = None
            
            print(f"  - {start_date} 주가 데이터 가져오기 성공!")
            print(f"  - 데이터 샘플:\n{df.head()}")
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

def delete_today_data():
    """오늘 날짜의 기존 데이터를 삭제합니다."""
    # 한국 시간 기준으로 오늘 날짜 계산
    korea_tz = pytz.timezone('Asia/Seoul')
    today_korea = datetime.now(korea_tz)
    today_str = today_korea.strftime('%Y-%m-%d')
    
    query = "DELETE FROM stock_prices WHERE time = %s;"
    execute_query(query, (today_str,))
    print(f"🗑️ {today_str} 기존 데이터 삭제 완료")

def fetch_stock_prices():
    """오늘 주가 데이터를 가져와 데이터베이스에 저장"""
    print("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
    
    # 시작일과 종료일 설정 (한국 시간 기준)
    start_date, end_date = get_date_range()
    print(f"📆 조회일: {start_date}")
    
    # 오늘 기존 데이터 삭제
    delete_today_data()
    
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
        
        print(f"🔄 ({idx+1}/{len(stock_list)}) {stock_name}({stock_code}) {start_date} 데이터 수집 중...")
        
        # API 호출 간격 조절 (rate limiting 방지)
        if idx > 0:
            time.sleep(random.uniform(0.5, 1.5))
        
        df, count = fetch_stock_data_with_retry(stock_code, start_date, end_date)
        
        if df is not None and not df.empty:
            all_stock_data.append(df)
            success_count += 1
            print(f"✅ {stock_name}({stock_code}) - {start_date} 데이터 {count}개 수집 완료")
        else:
            fail_count += 1
            print(f"❌ {stock_name}({stock_code}) - {start_date} 데이터 없음")
    
    if all_stock_data:
        # 모든 주가 데이터 합치기
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        # BIGINT 범위 초과값 처리
        for col in ['volume', 'foreign_holding']:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                combined_df[col] = combined_df[col].apply(lambda x: x if pd.notnull(x) and abs(x) <= 9223372036854775807 else None)
        
        # market_cap은 DECIMAL로 처리
        if 'market_cap' in combined_df.columns:
            combined_df['market_cap'] = pd.to_numeric(combined_df['market_cap'], errors='coerce')
        
        # NaN을 None으로 변환
        combined_df = combined_df.where(pd.notnull(combined_df), None)
        
        # 트랜잭션으로 데이터 업데이트
        query = """
        INSERT INTO stock_prices (
            time, stock_code, stock_name, open_price, high_price,
            low_price, close_price, volume, market_cap,
            foreign_holding, foreign_holding_ratio
        ) VALUES %s;
        """
        
        data = [(
            row['time'], row['stock_code'], row['stock_name'],
            row['open_price'], row['high_price'], row['low_price'],
            row['close_price'], row['volume'], row.get('market_cap'),
            row.get('foreign_holding'), row.get('foreign_holding_ratio')
        ) for _, row in combined_df.iterrows()]
        
        execute_values_query(query, data)
        
        print(f"\n💾 {start_date} 데이터가 데이터베이스에 저장되었습니다.")
        print(f"📊 통계:")
        print(f"   - 성공한 종목 수: {success_count}/{len(stock_list)}")
        print(f"   - 실패한 종목 수: {fail_count}")
        print(f"   - 수집된 데이터 레코드 수: {len(combined_df)}개")
        
        return combined_df
    else:
        print(f"❌ 저장할 {start_date} 데이터가 없습니다.")
        return None

if __name__ == "__main__":
    print("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
    create_stock_prices_table()
    stock_data = fetch_stock_prices()
    if stock_data is not None:
        print(f"✅ 데이터 수집 및 저장 완료!")
    else:
        print("❌ 데이터 수집 실패!") 