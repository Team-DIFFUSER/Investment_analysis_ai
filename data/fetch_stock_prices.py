import os
import sys
from pathlib import Path
import time
from requests.exceptions import RequestException
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import numpy as np

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
    """오늘 날짜만 가져오기 위한 날짜 범위 계산"""
    today = datetime.now()
    return today.strftime('%Y%m%d'), today.strftime('%Y%m%d')

def clean_stock_code(stock_code):
    """종목코드에서 'A' 접두사 제거"""
    return stock_code.replace('A', '')

def fetch_stock_data(stock_code, start_date, end_date):
    """PyKrx를 사용하여 주식 데이터 가져오기 (오늘 날짜만)"""
    try:
        # 종목코드 정리
        clean_code = clean_stock_code(stock_code)
        print(f"  - 정리된 종목코드: {clean_code}")
        
        # 일별 OHLCV 데이터 가져오기
        print(f"  - 주가 데이터 가져오기 시도 중...")
        print(f"  - 조회일: {start_date}")  # start_date와 end_date가 같으므로 하나만 표시
        
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
                print(f"  - 오늘 데이터가 없음 (거래일이 아님)")
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
            
            # 시가총액 데이터 가져오기 (오늘 날짜만)
            try:
                market_cap = stock.get_market_cap_by_date(start_date, end_date, clean_code)
                if not market_cap.empty:
                    df['market_cap'] = market_cap['시가총액']
            except Exception as e:
                print(f"  - 시가총액 데이터 가져오기 실패: {str(e)}")
            
            # 외국인/기관 보유량 데이터 가져오기 (오늘 날짜만)
            try:
                foreign_holding = stock.get_exhaustion_rates_of_foreign_investment_by_ticker(clean_code, start_date, end_date)
                if not foreign_holding.empty:
                    df['foreign_holding'] = foreign_holding['외국인보유량']
                    df['foreign_holding_ratio'] = foreign_holding['외국인보유비율']
            except Exception as e:
                print(f"  - 외국인 보유량 데이터 가져오기 실패: {str(e)}")
            
            print(f"  - 오늘 주가 데이터 가져오기 성공!")
            print(f"  - 데이터 샘플:\n{df.head()}")
            return df, len(df)
            
        except json.JSONDecodeError as e:
            print(f"  - API 응답 파싱 실패: {str(e)}")
            print(f"  - API 응답이 유효한 JSON 형식이 아닙니다.")
            return None, 0
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
    today = datetime.now().strftime('%Y-%m-%d')
    query = "DELETE FROM stock_prices WHERE time = %s;"
    execute_query(query, (today,))
    print(f"🗑️ {today} 기존 데이터 삭제 완료")

def fetch_stock_prices():
    """오늘 주가 데이터를 가져와 데이터베이스에 저장"""
    print("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
    
    # 시작일과 종료일 설정 (오늘 날짜만)
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
        
        print(f"🔄 ({idx+1}/{len(stock_list)}) {stock_name}({stock_code}) 오늘 데이터 수집 중...")
        
        df, count = fetch_stock_data(stock_code, start_date, end_date)
        
        if df is not None and not df.empty:
            all_stock_data.append(df)
            success_count += 1
            print(f"✅ {stock_name}({stock_code}) - 오늘 데이터 {count}개 수집 완료")
        else:
            fail_count += 1
            print(f"❌ {stock_name}({stock_code}) - 오늘 데이터 없음")
    
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
        
        print(f"\n💾 오늘 데이터가 데이터베이스에 저장되었습니다.")
        print(f"📊 통계:")
        print(f"   - 성공한 종목 수: {success_count}/{len(stock_list)}")
        print(f"   - 실패한 종목 수: {fail_count}")
        print(f"   - 수집된 데이터 레코드 수: {len(combined_df)}개")
        
        return combined_df
    else:
        print("❌ 저장할 오늘 데이터가 없습니다.")
        return None

if __name__ == "__main__":
    print("📢 KOSPI200 오늘 주가 데이터를 가져오는 중...")
    create_stock_prices_table()
    stock_data = fetch_stock_prices()
    if stock_data is not None:
        print(f"✅ 오늘 데이터 수집 및 저장 완료!")
    else:
        print("❌ 오늘 데이터 수집 실패!") 