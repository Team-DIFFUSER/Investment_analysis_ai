import os
import pandas as pd
import requests
import psycopg2
from dotenv import load_dotenv
import logging
from datetime import datetime
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# TimescaleDB 연결 설정
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'sslmode': os.getenv('DB_SSL_MODE', 'require')
}

# 백엔드 API 설정
BASE_URL = os.getenv('API_BASE_URL', 'http://52.79.34.229')
LOGIN_URL = f"{BASE_URL}/api/users/login"
API_BASE_URL = f"{BASE_URL}/api/assets"

def get_jwt_token(username, password):
    """JWT 토큰 발급받기"""
    try:
        login_data = {
            "username": username,
            "password": password
        }
        resp = requests.post(LOGIN_URL, json=login_data)
        resp.raise_for_status()
        jwt_token = resp.json().get("jwt")
        if not jwt_token:
            raise ValueError("JWT 토큰 발급 실패: 응답에 jwt가 없습니다.")
        return jwt_token
    except Exception as e:
        logger.error(f"JWT 토큰 발급 실패: {str(e)}")
        raise

def create_financial_statements_table():
    """TimescaleDB에 재무제표 테이블 생성"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # 테이블 생성 쿼리
        create_table_query = """
        CREATE TABLE IF NOT EXISTS financial_statements (
            id SERIAL PRIMARY KEY,
            stock_code VARCHAR(20) NOT NULL,
            per NUMERIC,
            roe NUMERIC,
            pbr NUMERIC,
            ev NUMERIC,
            bps NUMERIC,
            sale_amt NUMERIC,
            bus_pro NUMERIC,
            cup_nga NUMERIC,
            cap NUMERIC,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cur.execute(create_table_query)
        conn.commit()
        logger.info("재무제표 테이블 생성 완료")
        
    except Exception as e:
        logger.error(f"테이블 생성 중 오류 발생: {str(e)}")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def fetch_financial_statement(stock_code, headers):
    """백엔드 API에서 특정 종목의 재무제표 데이터 가져오기"""
    tried = False
    data = None
    
    # 1) A 없는 코드 시도
    if stock_code.startswith('A'):
        code_no_a = stock_code[1:]
        url = f"{API_BASE_URL}/{code_no_a}/financials"
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if 'stk_cd' not in data:
                    data['stk_cd'] = stock_code
                tried = True
            else:
                logger.warning(f"Try1: Error for {stock_code}({code_no_a}): {resp.status_code}")
        except Exception as e:
            logger.error(f"Try1: Error for {stock_code}({code_no_a}): {str(e)}")
        time.sleep(0.2)
    
    # 2) 실패시 A 붙은 코드도 시도
    if not tried:
        url = f"{API_BASE_URL}/{stock_code}/financials"
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if 'stk_cd' not in data:
                    data['stk_cd'] = stock_code
            else:
                logger.warning(f"Try2: Error for {stock_code}: {resp.status_code}")
        except Exception as e:
            logger.error(f"Try2: Error for {stock_code}: {str(e)}")
        time.sleep(0.2)
    
    return data

def save_financial_statement(data):
    """재무제표 데이터를 TimescaleDB에 저장"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO financial_statements (
            stock_code, per, roe, pbr, ev, bps, 
            sale_amt, bus_pro, cup_nga, cap
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        
        # 문자열 데이터를 숫자로 변환
        values = (
            data['stk_cd'],
            float(data['per']) if data.get('per') else None,
            float(data['roe']) if data.get('roe') else None,
            float(data['pbr']) if data.get('pbr') else None,
            float(data['ev']) if data.get('ev') else None,
            float(data['bps']) if data.get('bps') else None,
            float(data['sale_amt']) if data.get('sale_amt') else None,
            float(data['bus_pro']) if data.get('bus_pro') else None,
            float(data['cup_nga']) if data.get('cup_nga') else None,
            float(data['cap']) if data.get('cap') else None
        )
        
        cur.execute(insert_query, values)
        conn.commit()
        logger.info(f"종목 {data['stk_cd']}의 재무제표 데이터 저장 완료")
        
    except Exception as e:
        logger.error(f"데이터 저장 중 오류 발생: {str(e)}")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def main():
    try:
        # 테이블 생성
        create_financial_statements_table()
        
        # 사용자 정보
        username = "JunOh"
        password = "testPassword!"
        
        # JWT 토큰 발급
        jwt_token = get_jwt_token(username, password)
        headers = {
            "Authorization": f"Bearer {jwt_token}"
        }
        
        # 코스피 200 종목 목록 읽기
        df = pd.read_csv('data/kospi200_and_related.csv', dtype={'종목코드': str})
        
        # 각 종목에 대해 재무제표 데이터 가져오기
        for _, row in df.iterrows():
            stock_code = row['종목코드']
            logger.info(f"종목 {stock_code} 처리 중...")
            
            # 재무제표 데이터 가져오기
            financial_data = fetch_financial_statement(stock_code, headers)
            if financial_data:
                save_financial_statement(financial_data)
            
        logger.info("모든 종목의 재무제표 데이터 처리 완료")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 