Investment Analysis AI

프로젝트 개요
AI 기반 주식 투자 분석 및 종목 추천 시스템

#주요 기능#

1. 종목 추천 시스템
- 추천 대상: 가격 예측이 가능한 22개 주요 종목
- 추천 개수: 상위 3개 종목 추천
- 추천 기준: 
  - 예측수익률 (다른 팀원의 주가 예측 모델 기반)
  - 기술적 지표 (RSI, 이동평균, 거래량 등)
  - 뉴스 감성분석 (FinBERT 기반)
  - 재무제표 데이터 (PER, PBR, ROE 등)
  - 사용자 투자 성향별 맞춤 가중치

2. 지원 종목 (22개)
- SK하이닉스, 한화, LG전자, 삼성전자, LG화학
- NAVER, 기아, 삼성바이오로직스, 현대모비스, HD현대
- 삼성생명, 삼성화재, 현대차, HD현대일렉트릭, 삼성중공업
- SK이노베이션, 삼성SDI, SK텔레콤, SK, 카카오
- 현대로템, 카카오뱅크

3. 투자 성향별 맞춤 추천
- 공격투자형: 수익률 중심 (40%)
- 적극투자형: 수익률 + 감성 중심 (35% + 30%)
- 위험중립형: 균형잡힌 가중치 (30% + 30%)
- 안정추구형: 변동성 중심 (35%)
- 안정형: 변동성 최우선 (40%)

기술 스택
- Backend: Python, TensorFlow, PyTorch
- Database: TimescaleDB, MongoDB
- ML/AI: FinBERT, LSTM, MLP
- Data Processing: Pandas, NumPy, Scikit-learn

## 설치 및 실행
```bash
pip install -r requirements.txt
python models/stock_recommendation/scripts/train.py --user_id [사용자ID]
python models/stock_recommendation/scripts/predict.py --user_id [사용자ID]
```

## 최근 업데이트
- **2025년08월**: 22개 주요 종목으로 추천 대상 제한
- **2025년08월월**: 추천 개수를 3개로 최적화

