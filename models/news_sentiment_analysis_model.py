import pandas as pd
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from database.database import execute_query, execute_values_query, execute_transaction
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# MongoDB 연결
MONGO_URI = os.getenv('MONGO_URI')
print(f"🔌 MongoDB URI: {MONGO_URI}")
mongo_client = MongoClient(MONGO_URI)

# 사용 가능한 데이터베이스와 컬렉션 확인
print("\n📚 사용 가능한 데이터베이스:")
for db_name in mongo_client.list_database_names():
    print(f"- {db_name}")
    db = mongo_client[db_name]
    collections = db.list_collection_names()
    if collections:
        print(f"  컬렉션:")
        for coll in collections:
            count = db[coll].count_documents({})
            print(f"  - {coll} (문서 수: {count})")

# 데이터베이스 연결
db = mongo_client['user_accounts']  # stock_news에서 user_accounts로 변경
holding_articles = db['holding_articles']

def create_news_sentiment_table():
    """뉴스 감성 분석 결과 테이블 생성"""
    queries = [
        ("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id SERIAL PRIMARY KEY,
            stock_code VARCHAR(10) NOT NULL,
            title TEXT NOT NULL,
            pub_date TIMESTAMPTZ NOT NULL,
            finbert_positive DECIMAL(5,4),
            finbert_negative DECIMAL(5,4),
            finbert_neutral DECIMAL(5,4),
            finbert_sentiment VARCHAR(10),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """, None),
        ("CREATE INDEX IF NOT EXISTS idx_news_sentiment_date ON news_sentiment (pub_date DESC);", None),
        ("CREATE INDEX IF NOT EXISTS idx_news_sentiment_code ON news_sentiment (stock_code);", None)
    ]
    execute_transaction(queries)
    print("News sentiment table created successfully!")

# FinBERT 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# MongoDB에서 뉴스 데이터 로드
def load_news_from_mongo():
    """MongoDB에서 뉴스 데이터 로드"""
    try:
        # MongoDB 연결 상태 확인
        print(f"📊 MongoDB 서버 정보: {mongo_client.server_info()}")
        print(f"📚 전체 문서 수: {holding_articles.count_documents({})}")
        
        # MongoDB 쿼리 - 모든 데이터 조회
        cursor = holding_articles.find().sort('pubDate', -1)
        
        # DataFrame으로 변환
        news_list = []
        for doc in cursor:
            try:
                news_list.append({
                    '_id': doc['_id'],
                    'stock_code': doc['stockCode'],
                    'title': doc['title'],
                    'description': doc.get('description', ''),
                    'pub_date': doc['pubDate']
                })
            except KeyError as e:
                print(f"⚠️ 문서 처리 중 오류 발생: {str(e)}")
                print(f"문제가 된 문서: {doc}")
                continue
        
        if not news_list:
            print("⚠️ MongoDB에서 뉴스 데이터를 찾을 수 없습니다.")
            print("🔍 MongoDB 컬렉션 구조 확인:")
            sample_doc = holding_articles.find_one()
            if sample_doc:
                print("샘플 문서 구조:")
                for key, value in sample_doc.items():
                    print(f"- {key}: {type(value)}")
            return pd.DataFrame()
            
        print(f"✅ {len(news_list)}개의 뉴스 데이터를 성공적으로 로드했습니다.")
        return pd.DataFrame(news_list)
    except Exception as e:
        print(f"❌ MongoDB 데이터 로드 실패: {str(e)}")
        print(f"🔍 MongoDB 연결 정보: {MONGO_URI}")
        return pd.DataFrame()

# 감성 분석 함수
def get_finbert_sentiment(text):
    if pd.isna(text) or text == '':
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'sentiment': 'neutral'}
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 긍정(0), 부정(1), 중립(2) 클래스 확률
    positive = predictions[0][0].item()
    negative = predictions[0][1].item()
    neutral = predictions[0][2].item()
    
    # 가장 높은 확률의 감성 반환
    sentiment_labels = ['positive', 'negative', 'neutral']
    sentiment = sentiment_labels[np.argmax([positive, negative, neutral])]
    
    return {
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'sentiment': sentiment
    }

def process_news_sentiment():
    """뉴스 데이터를 처리하고 데이터베이스에 저장"""
    # MongoDB에서 뉴스 데이터 로드
    news_df = load_news_from_mongo()
    print(f"📊 {len(news_df)}개의 뉴스 데이터를 로드했습니다.")
    
    if news_df.empty:
        print("⚠️ 처리할 뉴스 데이터가 없습니다.")
        return
    
    # 뉴스 제목과 설명에 대한 감성 분석
    news_df['title_sentiment'] = news_df['title'].apply(get_finbert_sentiment)
    news_df['description_sentiment'] = news_df['description'].apply(get_finbert_sentiment)
    
    # 제목과 설명의 감성 점수 평균 계산
    news_df['finbert_positive'] = (news_df['title_sentiment'].apply(lambda x: x['positive']) + 
                                 news_df['description_sentiment'].apply(lambda x: x['positive'])) / 2
    news_df['finbert_negative'] = (news_df['title_sentiment'].apply(lambda x: x['negative']) + 
                                 news_df['description_sentiment'].apply(lambda x: x['negative'])) / 2
    news_df['finbert_neutral'] = (news_df['title_sentiment'].apply(lambda x: x['neutral']) + 
                                news_df['description_sentiment'].apply(lambda x: x['neutral'])) / 2
    
    # 최종 감성 결정
    news_df['finbert_sentiment'] = news_df.apply(
        lambda row: 'positive' if row['finbert_positive'] > max(row['finbert_negative'], row['finbert_neutral'])
        else 'negative' if row['finbert_negative'] > max(row['finbert_positive'], row['finbert_neutral'])
        else 'neutral', axis=1
    )
    
    # 데이터베이스에 저장할 데이터 준비
    data = [(
        row['stock_code'],
        row['title'],
        row['pub_date'],
        row['finbert_positive'],
        row['finbert_negative'],
        row['finbert_neutral'],
        row['finbert_sentiment']
    ) for _, row in news_df.iterrows()]
    
    # 트랜잭션으로 데이터 업데이트
    queries = [
        ("""
        INSERT INTO news_sentiment (
            stock_code, title, pub_date,
            finbert_positive, finbert_negative, finbert_neutral, finbert_sentiment
        ) VALUES %s
        ON CONFLICT (stock_code, pub_date) DO UPDATE SET
            finbert_positive = EXCLUDED.finbert_positive,
            finbert_negative = EXCLUDED.finbert_negative,
            finbert_neutral = EXCLUDED.finbert_neutral,
            finbert_sentiment = EXCLUDED.finbert_sentiment
        """, data)
    ]
    execute_transaction(queries)
    
    print(f"✅ {len(data)}개의 뉴스 감성 분석 결과가 데이터베이스에 저장되었습니다.")

if __name__ == "__main__":
    print("📢 뉴스 감성 분석을 시작합니다...")
    create_news_sentiment_table()
    process_news_sentiment()
    print("✅ 감성 분석 완료!")
