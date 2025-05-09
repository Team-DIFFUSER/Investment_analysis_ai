import requests
import datetime
import pandas as pd
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import time
from ai_models.database.db_config import execute_query, execute_values_query, execute_transaction

def create_news_table():
    """뉴스 데이터 테이블 생성"""
    queries = [
        ("DROP TABLE IF EXISTS financial_news;", None),
        ("""
        CREATE TABLE IF NOT EXISTS financial_news (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            link TEXT NOT NULL,
            pub_date TIMESTAMPTZ NOT NULL,
            source VARCHAR(20) NOT NULL,
            company VARCHAR(50) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title, link)
        );
        """, None),
        ("CREATE INDEX IF NOT EXISTS idx_financial_news_date ON financial_news (pub_date DESC);", None),
        ("CREATE INDEX IF NOT EXISTS idx_financial_news_company ON financial_news (company);", None)
    ]
    execute_transaction(queries)
    print("Financial news table created successfully!")

def parse_relative_date(relative_date):
    """상대적 날짜 문자열을 실제 날짜로 변환"""
    today = datetime.datetime.now()
    
    if '시간 전' in relative_date:
        hours = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(hours=hours)
    elif '일 전' in relative_date:
        days = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(days=days)
    elif '주 전' in relative_date:
        weeks = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(weeks=weeks)
    elif '개월 전' in relative_date:
        months = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(days=months*30)
    elif '년 전' in relative_date:
        years = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(days=years*365)
    else:
        return today

def fetch_news_google(query, start_date, end_date):
    """Google News에서 뉴스 가져오기"""
    base_url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "ko",
        "gl": "KR",
        "ceid": "KR:ko"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print("Failed to fetch news from Google")
        return []
    
    root = ET.fromstring(response.content)
    news_items = []
    
    for item in root.findall(".//item"):
        try:
            title = item.find("title").text
            link = item.find("link").text
            pub_date = item.find("pubDate").text
            
            if 'Z' in pub_date:
                pub_date = datetime.datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            else:
                pub_date = parse_relative_date(pub_date)
            
            news_items.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "source": "Google News"
            })
        except Exception as e:
            print(f"Error processing Google news item: {e}")
            continue
    
    return news_items

def fetch_news_naver(query, start_date, end_date):
    """네이버 뉴스에서 뉴스 가져오기"""
    base_url = "https://search.naver.com/search.naver"
    params = {
        "where": "news",
        "query": query,
        "sort": "1",  # 최신순
        "ds": start_date.strftime('%Y.%m.%d'),
        "de": end_date.strftime('%Y.%m.%d')
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch news from Naver")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = []
    
    for item in soup.select('.news_wrap'):
        try:
            title = item.select_one('.news_tit').text
            link = item.select_one('.news_tit')['href']
            date_text = item.select_one('.info').text.split(' ')[0]
            
            # 날짜 파싱 개선
            try:
                if '전' in date_text:
                    date = parse_relative_date(date_text)
                else:
                    # 다양한 날짜 형식 처리
                    date_formats = [
                        '%Y.%m.%d',
                        '%Y-%m-%d',
                        '%Y/%m/%d',
                        '%Y년 %m월 %d일',
                        '%m/%d/%Y',
                        '%d/%m/%Y'
                    ]
                    
                    date = None
                    for fmt in date_formats:
                        try:
                            date = datetime.datetime.strptime(date_text, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if date is None:
                        date = datetime.datetime.now()
                
                news_items.append({
                    "title": title,
                    "link": link,
                    "pub_date": date,
                    "source": "Naver News"
                })
            except Exception as e:
                print(f"Error parsing date '{date_text}': {e}")
                continue
                
        except Exception as e:
            print(f"Error processing Naver news item: {e}")
            continue
    
    return news_items

def save_news_to_db(news_items, company):
    """뉴스 데이터를 데이터베이스에 저장"""
    if not news_items:
        print(f"No news items to save for {company}")
        return
    
    try:
        # 데이터베이스에 저장할 데이터 준비
        data = [(
            item['title'],
            item['link'],
            item['pub_date'],
            item['source'],
            company
        ) for item in news_items]
        
        # execute_values를 사용하여 데이터 저장
        query = """
        INSERT INTO financial_news (
            title, link, pub_date, source, company
        ) VALUES %s
        ON CONFLICT (title, link) DO UPDATE SET
            pub_date = EXCLUDED.pub_date,
            source = EXCLUDED.source,
            company = EXCLUDED.company
        """
        execute_values_query(query, data)
        
        print(f"✅ {len(data)}개의 {company} 관련 뉴스가 데이터베이스에 저장되었습니다.")
    except Exception as e:
        print(f"❌ {company} 뉴스 저장 중 오류 발생: {str(e)}")
        return

def fetch_news_lg():
    """LG 및 계열사 관련 뉴스 수집"""
    # LG 계열사 목록
    affiliates = [
        "LG", "LG전자", "LG화학", "LG이노텍", "LG디스플레이",
        "LG유플러스", "LG하우시스", "LG생활건강", "LG에너지솔루션"
    ]
    
    # 확장된 검색 쿼리
    base_query = """
    {company}+(주가|실적|투자|분기|반기|연간|매출|이익|수익|성장|전망|전략|혁신|R&D|연구|개발|투자|M&A|인수|합병|협력|제휴|계약|수주|공급|출시|신제품|기술|특허)
    -스포츠 -야구 -축구 -농구 -배구 -골프 -테니스
    """
    
    end_date = datetime.datetime(2025, 3, 21)
    start_date = end_date - datetime.timedelta(days=180)
    
    print(f"Fetching news from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # 모든 계열사의 뉴스 수집
    total_news_count = 0
    for company in affiliates:
        print(f"\nFetching news for {company}...")
        query = base_query.format(company=company)
        
        # Google News
        print("Fetching from Google News...")
        google_news = fetch_news_google(query, start_date, end_date)
        save_news_to_db(google_news, company)
        total_news_count += len(google_news)
        
        # Naver News
        print("Fetching from Naver News...")
        naver_news = fetch_news_naver(query, start_date, end_date)
        save_news_to_db(naver_news, company)
        total_news_count += len(naver_news)
        
        # API 호출 간격 조절
        time.sleep(1)
    
    print(f"\nTotal {total_news_count}개의 뉴스가 수집되었습니다.")
    return total_news_count

if __name__ == "__main__":
    print("📢 LG 계열사 뉴스 수집을 시작합니다...")
    create_news_table()
    total_news = fetch_news_lg()
    if total_news > 0:
        print("✅ 뉴스 수집 및 저장 완료!")
    else:
        print("❌ 뉴스 수집 실패!")