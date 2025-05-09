import requests
import pandas as pd
import urllib3
from datetime import datetime
import os
from dotenv import load_dotenv
from ai_models.database.db_config import execute_query, execute_values_query, execute_transaction

# Load environment variables
load_dotenv()

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_stock_items_table():
    """주식 종목 테이블 생성"""
    queries = [
        ("""
        CREATE TABLE IF NOT EXISTS stock_items (
            stock_code VARCHAR(10) PRIMARY KEY,
            stock_name VARCHAR(50) NOT NULL,
            isin_code VARCHAR(12),
            company_name VARCHAR(100),
            is_kospi200 BOOLEAN DEFAULT FALSE,
            is_related BOOLEAN DEFAULT FALSE,
            last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """, None)
    ]
    execute_transaction(queries)
    print("Stock items table created successfully!")

def fetch_stock_items():
    """KOSPI 200 종목 및 관련 기업 정보를 가져와 데이터베이스에 저장"""
    print("KOSPI 200 종목 및 관련 기업 정보 가져오는 중...")
    
    # API 엔드포인트
    url = "http://apis.data.go.kr/1160100/service/GetKrxListedInfoService/getItemInfo"
    
    # API 키 설정
    service_key = os.getenv('STOCK_API_KEY')
    if not service_key:
        raise ValueError("STOCK_API_KEY environment variable is not set")
    
    # 파라미터 설정
    params = {
        "serviceKey": service_key,
        "resultType": "json",
        "mrktCtg": "KOSPI",
        "numOfRows": "3000"
    }

    try:
        # API 호출
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        if 'response' in data and 'body' in data['response']:
            items = data['response']['body']['items']['item']
            
            # 단일 아이템인 경우 리스트로 변환
            if isinstance(items, dict):
                items = [items]
            
            # 데이터프레임 생성
            df = pd.DataFrame(items)
            
            # 필요한 컬럼만 선택
            columns = ['srtnCd', 'itmsNm', 'isinCd', 'corpNm']
            df = df[columns].copy()  # 명시적 복사
            
            # 컬럼명 변경
            column_names = {
                'srtnCd': 'stock_code',
                'itmsNm': 'stock_name',
                'isinCd': 'isin_code',
                'corpNm': 'company_name'
            }
            df = df.rename(columns=column_names)
            
            # KOSPI 200 종목 선택 (처음 200개)
            kospi200 = df.head(200).copy()  # 명시적 복사
            kospi200.loc[:, 'is_kospi200'] = True
            kospi200.loc[:, 'is_related'] = False
            
            # 관련 기업 필터링
            related_companies = []
            major_groups = ['삼성', 'SK', 'LG', '현대', 'NAVER', '카카오', '네이버']
            
            print("\n[관련 기업 검색]")
            for group in major_groups:
                group_companies = df[df['company_name'].str.contains(group, na=False)].copy()  # 명시적 복사
                if not group_companies.empty:
                    print(f"- {group} 그룹: {len(group_companies)}개 기업 발견")
                    related_companies.extend(group_companies.to_dict('records'))
            
            # 중복 제거
            related_companies = pd.DataFrame(related_companies)
            related_companies = related_companies.drop_duplicates(subset=['stock_code'])
            
            # KOSPI 200에 포함된 관련 기업 제외
            related_companies = related_companies[~related_companies['stock_code'].isin(kospi200['stock_code'])].copy()  # 명시적 복사
            related_companies.loc[:, 'is_kospi200'] = False
            related_companies.loc[:, 'is_related'] = True
            
            # KOSPI 200과 관련 기업 합치기
            final_df = pd.concat([kospi200, related_companies])
            
            # 트랜잭션으로 데이터 업데이트
            queries = [
                ("DELETE FROM stock_items;", None),
                ("""
                INSERT INTO stock_items (
                    stock_code, stock_name, isin_code, company_name,
                    is_kospi200, is_related, last_updated
                ) VALUES %s
                """, [(
                    row['stock_code'], row['stock_name'], row['isin_code'],
                    row['company_name'], row['is_kospi200'], row['is_related'],
                    datetime.now()
                ) for _, row in final_df.iterrows()])
            ]
            execute_transaction(queries)
            
            print(f"\n✅ KOSPI 200 및 관련 기업 정보가 데이터베이스에 저장되었습니다.")
            print(f"- KOSPI 200 종목 수: {len(kospi200)}개")
            print(f"- 관련 기업 수: {len(related_companies)}개")
            print(f"- 총 {len(final_df)}개의 종목 정보 저장됨")
            
            return final_df
        else:
            print("❌ API 응답에 필요한 데이터가 없습니다.")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 중 오류 발생: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    create_stock_items_table()
    fetch_stock_items()
