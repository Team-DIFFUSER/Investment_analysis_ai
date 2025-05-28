import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from database.database import execute_query, execute_values_query, execute_transaction

def create_economic_indicators_table():
    """경제지표 데이터 테이블 생성"""
    try:
        # 기존 테이블 삭제
        drop_query = "DROP TABLE IF EXISTS economic_indicators CASCADE;"
        execute_query(drop_query, fetch=False)
        
        # 테이블 생성
        create_table_query = """
        CREATE TABLE economic_indicators (
            time TIMESTAMPTZ NOT NULL,
            treasury_10y DECIMAL(10,4),
            dollar_index DECIMAL(10,4),
            usd_krw DECIMAL(10,4),
            korean_bond_10y DECIMAL(10,4),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT economic_indicators_time_key UNIQUE (time)
        );
        """
        execute_query(create_table_query, fetch=False)
        
        # hypertable로 변환
        try:
            execute_query("SELECT create_hypertable('economic_indicators', 'time');", fetch=False)
        except Exception as e:
            print(f"Warning: Could not convert to hypertable: {e}")
        
        # 인덱스 생성
        execute_query("CREATE INDEX IF NOT EXISTS idx_economic_indicators_time ON economic_indicators (time DESC);", fetch=False)
        print("Economic indicators table created successfully!")
            
    except Exception as e:
        print(f"Error creating table: {e}")
        raise

def get_date_range():
    """고정된 종료일(20250321)과 그로부터 500일 전의 시작일을 계산하여 반환"""
    end_date = datetime.strptime('20250321', '%Y%m%d')
    start_date = end_date - timedelta(days=500)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def fetch_economic_indicators():
    """경제지표 데이터를 가져와 데이터베이스에 저장"""
    print("📢 경제지표 데이터를 가져오는 중...")
    
    # 시작일과 종료일 설정
    start_date, end_date = get_date_range()
    print(f"📆 조회 기간: {start_date} ~ {end_date}")
    
    try:
        # 미국 10년물 국채 금리
        treasury = yf.download('^TNX', start=start_date, end=end_date)
        treasury = treasury[['Close']].rename(columns={'Close': 'treasury_10y'})
        
        # 달러 인덱스
        dollar_index = yf.download('DX-Y.NYB', start=start_date, end=end_date)
        dollar_index = dollar_index[['Close']].rename(columns={'Close': 'dollar_index'})
        
        # 원달러 환율
        usdkrw = yf.download('USDKRW=X', start=start_date, end=end_date)
        usdkrw = usdkrw[['Close']].rename(columns={'Close': 'usd_krw'})
        
        # 한국 10년물 국채 금리 (대체 심볼 사용)
        korean_bond = yf.download('114260.KS', start=start_date, end=end_date)  # 한국 10년물 국채 선물
        korean_bond = korean_bond[['Close']].rename(columns={'Close': 'korean_bond_10y'})
        
        # 모든 경제지표 병합
        economic_data = pd.concat([treasury, dollar_index, usdkrw, korean_bond], axis=1)
        
        # MultiIndex 문제 해결
        if isinstance(economic_data.columns, pd.MultiIndex):
            economic_data.columns = economic_data.columns.get_level_values(0)
        
        # 결측치 처리
        economic_data = economic_data.ffill().bfill()
        
        # 데이터베이스에 저장할 데이터 준비
        data = [(
            index.to_pydatetime(),
            row['treasury_10y'],
            row['dollar_index'],
            row['usd_krw'],
            row['korean_bond_10y']
        ) for index, row in economic_data.iterrows()]
        
        # 트랜잭션으로 데이터 업데이트
        query = """
        INSERT INTO economic_indicators (
            time, treasury_10y, dollar_index, usd_krw, korean_bond_10y
        ) VALUES %s
        ON CONFLICT (time) DO UPDATE SET
            treasury_10y = EXCLUDED.treasury_10y,
            dollar_index = EXCLUDED.dollar_index,
            usd_krw = EXCLUDED.usd_krw,
            korean_bond_10y = EXCLUDED.korean_bond_10y
        """
        execute_values_query(query, data)
        
        print(f"✅ {len(data)}개의 경제지표 데이터가 데이터베이스에 저장되었습니다.")
        return economic_data
        
    except Exception as e:
        print(f"❌ 데이터 수집 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    create_economic_indicators_table()
    economic_data = fetch_economic_indicators()
    if economic_data is not None:
        print(f"✅ 모든 데이터 수집 및 저장 완료!")
    else:
        print("❌ 데이터 수집 실패!") 