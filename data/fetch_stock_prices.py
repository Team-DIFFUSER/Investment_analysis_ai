from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta
import os
from ai_models.database.db_config import execute_query, execute_values_query, execute_transaction

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
    """현재 날짜를 기준으로 과거 500일의 데이터를 가져오도록 설정"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)
    return start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')

def clean_stock_code(stock_code):
    """종목코드에서 'A' 접두사 제거"""
    return stock_code.replace('A', '')

def fetch_stock_data(stock_code, start_date, end_date):
    """PyKrx를 사용하여 주식 데이터 가져오기"""
    try:
        # 종목코드 정리
        clean_code = clean_stock_code(stock_code)
        
        # 주가 데이터 가져오기
        try:
            print(f"  - 주가 데이터 가져오기 시도 중...")
            df = stock.get_market_ohlcv_by_date(start_date, end_date, clean_code)
            if df.empty:
                print(f"  - 데이터가 비어있음")
                return None, 0
            print(f"  - 주가 데이터 가져오기 성공!")
        except Exception as e:
            print(f"  - 주가 데이터 가져오기 실패: {str(e)}")
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
        try:
            df['stock_name'] = stock.get_market_ticker_name(clean_code)
        except Exception as e:
            print(f"  - 종목명 가져오기 실패: {str(e)}")
            df['stock_name'] = stock_code
        
        # 날짜를 인덱스에서 컬럼으로 변경
        df = df.reset_index()
        df = df.rename(columns={'날짜': 'time'})
        
        # 시가총액 데이터 가져오기
        try:
            market_cap = stock.get_market_cap_by_date(start_date, end_date, clean_code)
            if not market_cap.empty:
                df = df.merge(market_cap[['시가총액']], left_on='time', right_index=True, how='left')
                df = df.rename(columns={'시가총액': 'market_cap'})
            else:
                df['market_cap'] = None
        except Exception as e:
            print(f"  - 시가총액 데이터 가져오기 실패: {str(e)}")
            df['market_cap'] = None
        
        # 외국인/기관 보유량 데이터 가져오기
        try:
            foreign_holding = stock.get_exhaustion_rates_of_foreign_investment_by_ticker(clean_code, start_date, end_date)
            if not foreign_holding.empty:
                df = df.merge(foreign_holding[['외국인보유량', '외국인보유비율']], left_on='time', right_index=True, how='left')
                df = df.rename(columns={
                    '외국인보유량': 'foreign_holding',
                    '외국인보유비율': 'foreign_holding_ratio'
                })
            else:
                df['foreign_holding'] = None
                df['foreign_holding_ratio'] = None
        except Exception as e:
            print(f"  - 외국인 보유량 데이터 가져오기 실패: {str(e)}")
            df['foreign_holding'] = None
            df['foreign_holding_ratio'] = None
        
        return df, len(df)
        
    except Exception as e:
        print(f"데이터 수집 중 오류 발생: {e}")
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
        execute_transaction(queries)
        
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