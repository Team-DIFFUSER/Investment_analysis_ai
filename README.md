# Investment Analysis AI

Team-DIFFUSER 서비스의 AI 전용 저장소입니다.  
이 리포지토리는 **주가 예측**, **개인화 종목 추천**, **투자 분석용 데이터 수집/가공**을 담당합니다.

중요한 점은, 이 저장소가 서비스 전체를 모두 담고 있지는 않다는 것입니다.

- 이 리포지토리: AI 모델 학습, 예측, 추천 로직, 데이터 적재
- 프론트엔드: 별도 저장소에서 추천 결과/예측 결과 UI 제공
- 백엔드: 별도 저장소에서 인증, 자산/재무 API, 사용자/보유 종목 관리 담당

즉, 현재 저장소는 **전체 서비스 중 AI 모듈**에 해당합니다.

---

## 1. 프로젝트가 하는 일

이 프로젝트는 크게 두 개의 AI 파이프라인으로 구성되어 있습니다.

### 1) 개별 종목 주가 예측

- 22개 주요 종목에 대해 시계열 기반 주가 예측 모델을 학습합니다.
- 각 종목별 TensorFlow 모델을 따로 관리합니다.
- 기술적 지표를 포함한 시계열 feature를 사용해 향후 5거래일 가격 흐름을 예측합니다.
- 예측 결과는 추천 모델의 입력 데이터로도 활용됩니다.

### 2) 사용자 맞춤 종목 추천

- PyTorch 기반 MLP 모델로 추천 점수를 계산합니다.
- 사용자 투자 성향, 보유 종목, 최근 수익률, 변동성, 뉴스 감성, 재무제표, 가격 예측값을 함께 반영합니다.
- 최종적으로 상위 3개 종목을 추천합니다.
- 추천 사유는 OpenAI API를 이용해 자연어로 생성합니다.

---

## 2. 전체 서비스 안에서의 역할

이 저장소는 단독 실행형 앱이라기보다, **프론트/백엔드와 연결되는 AI 서브시스템**입니다.

### 입력

- TimescaleDB의 시계열 주가 데이터
- TimescaleDB의 뉴스 감성분석 데이터
- TimescaleDB의 재무제표 데이터
- MongoDB의 사용자 계정/투자성향/보유 종목 정보
- 백엔드 API에서 내려주는 재무 데이터

### 출력

- 종목별 예측 모델 체크포인트
- 추천 모델 체크포인트
- 종목별 예측 결과
- 사용자별 추천 결과
- 추천 사유 텍스트

---

## 3. 핵심 기능

### 종목 추천 시스템

- 추천 대상: 가격 예측이 가능한 22개 주요 종목
- 추천 개수: 상위 3개
- 추천 기준:
  - 최근 수익률
  - 변동성
  - 뉴스 감성 점수
  - 개별 종목 가격 예측 수익률
  - 재무제표 지표
  - 사용자 보유 종목/평가손익률
  - 사용자 투자 성향별 가중치

### 투자 성향별 가중치

| 투자 성향 | 수익률 | 변동성 | 감성 | 재무 |
| --- | ---: | ---: | ---: | ---: |
| 공격투자형 | 0.40 | 0.10 | 0.30 | 0.20 |
| 적극투자형 | 0.35 | 0.15 | 0.30 | 0.20 |
| 위험중립형 | 0.30 | 0.30 | 0.20 | 0.20 |
| 안정추구형 | 0.25 | 0.35 | 0.20 | 0.20 |
| 안정형 | 0.20 | 0.40 | 0.20 | 0.20 |

### 지원 종목 (22개)

SK하이닉스, 한화, LG전자, 삼성전자, LG화학, NAVER, 기아, 삼성바이오로직스, 현대모비스, HD현대, 삼성생명, 삼성화재, 현대차, HD현대일렉트릭, 삼성중공업, SK이노베이션, 삼성SDI, SK텔레콤, SK, 카카오, 현대로템, 카카오뱅크

---

## 4. 디렉토리 구조

```text
Investment_analysis_ai/
├── data/                               # 데이터 수집/적재 스크립트 및 CSV/XLSX 샘플
├── database/                           # TimescaleDB 연결 및 스키마 관련 코드
├── models/
│   ├── stocks/                         # 22개 종목별 개별 예측 모델
│   ├── stock_recommendation/           # 사용자 맞춤 추천 모델
│   │   ├── data_processing/
│   │   ├── evaluation_utils/
│   │   ├── mlp_model/
│   │   ├── saved/
│   │   └── scripts/
│   └── base/                           # 공통 가격 예측 모델 로직
├── scripts/                            # 개별 종목 예측용 train / predict / evaluate
├── utils/                              # 설정, 날짜 처리, 로거
├── logs/                               # 로그 파일
└── README.md
```

---

## 5. 기술 스택

- Language: Python
- Price Prediction: TensorFlow, Keras, LSTM
- Recommendation: PyTorch, MLP
- NLP: FinBERT, OpenAI API
- Data Processing: pandas, numpy, scikit-learn, ta
- Data Source: pykrx, FinanceDataReader, yfinance, 외부 백엔드 API
- Database: TimescaleDB(PostgreSQL), MongoDB

---

## 6. 실행 전 준비

### 권장 환경

- Python 3.12 권장
- GPU가 있으면 TensorFlow 학습 속도 개선 가능

### 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. 환경 변수

`.env` 파일이 필요합니다. 최소한 아래 값들은 맞춰두는 것이 좋습니다.

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# TimescaleDB / PostgreSQL
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=your-db-name
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_SSL_MODE=require
TIMESCALE_URI=postgresql://user:password@host:5432/dbname

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=your_mongo_db
MONGO_USER_ACCOUNTS=user_accounts
MONGO_USER_HOLDINGS=user_holdings

# Backend API
API_BASE_URL=http://your-backend-host
```

### 참고

- `database/database.py`는 TimescaleDB 연결 정보가 없으면 바로 예외를 발생시킵니다.
- 추천 파이프라인은 MongoDB 컬렉션에서 사용자 투자 성향과 보유 종목을 읽습니다.
- 추천 사유 생성은 `OPENAI_API_KEY`가 없으면 동작하지 않습니다.
- 재무제표 수집 스크립트는 백엔드 API와 인증 흐름에 의존합니다.

---

## 8. 데이터 의존성

추천 파이프라인이 정상 동작하려면 아래 데이터가 준비돼 있어야 합니다.

### TimescaleDB 주요 테이블

- `stock_items`
- `stock_prices`
- `news_sentiment`
- `predicted_stock_prices`
- `financial_statements`
- `stock_recommendations`

### MongoDB 주요 컬렉션

- 사용자 계정 컬렉션
- 사용자 보유 종목 컬렉션

코드상에서 이 데이터들은 다음 정보에 사용됩니다.

- 사용자 투자 성향 조회
- 사용자 보유 여부 / 보유 평가손익률 반영
- 최근 1개월 수익률 / 변동성 계산
- 뉴스 감성 점수 집계
- 재무제표 feature 생성
- 개별 종목 예측값을 추천 feature로 반영

---

## 9. 빠른 실행 순서

처음 세팅할 때는 보통 아래 순서로 진행합니다.

1. 데이터베이스/백엔드/MongoDB 연결 정보 설정
2. 원천 데이터 수집 및 적재
3. 개별 종목 가격 예측 모델 학습
4. 추천 모델 학습
5. 사용자별 추천 실행

---

## 10. 데이터 수집 스크립트

### 주가 데이터 수집

```bash
python data/fetch_stock_prices.py
```

- KOSPI200 및 관련 종목의 가격 데이터를 수집합니다.
- 수집 결과는 `stock_prices` 테이블에 저장됩니다.

### 재무제표 데이터 수집

```bash
python data/fetch_financial_statements.py
```

- 백엔드 API를 호출해 종목별 재무 데이터를 가져옵니다.
- 수집 결과는 `financial_statements` 테이블에 저장됩니다.

### 기타

`data/` 폴더에는 뉴스, 경제지표, 종목 메타데이터 적재용 스크립트가 함께 들어 있습니다.

---

## 11. 개별 종목 가격 예측 파이프라인

### GPU 확인

```bash
python scripts/check_gpu.py
```

### 모델 학습

```bash
python scripts/train.py
```

- `models.ALL_STOCK_MODELS`에 등록된 22개 종목 모델을 병렬로 학습합니다.
- 학습 결과는 `models/checkpoints/` 또는 `models/backup/`에 저장됩니다.
- 학습 리포트는 `data/reports/` 아래에 생성됩니다.

### 예측 실행

```bash
python scripts/predict.py
```

- 각 종목 모델을 불러와 향후 5거래일 기준 가격 흐름 예측을 수행합니다.
- 종목별 모델이 없으면 내부적으로 학습을 먼저 시도합니다.

### 평가

```bash
python scripts/evaluate.py
```

- 전체 종목 모델의 MSE, MAE, R2 등을 계산합니다.
- 결과 리포트는 `results/` 아래에 저장됩니다.

---

## 12. 사용자 맞춤 추천 파이프라인

추천 모델은 `models/stock_recommendation/` 아래에서 관리합니다.

### 학습

```bash
python models/stock_recommendation/scripts/train.py \
  --user_id JunOh \
  --investment_type 위험중립형
```

- 입력 feature 수: 17개
- 모델: MLP
- 학습 완료 후 공통 추천 모델이 아래 경로에 저장됩니다.

```text
models/stock_recommendation/saved/model_latest.pt
```

### 추천 실행

```bash
python models/stock_recommendation/scripts/predict.py --user_id JunOh
```

필요하면 투자 성향을 직접 넘길 수도 있습니다.

```bash
python models/stock_recommendation/scripts/predict.py \
  --user_id JunOh \
  --investment_type 위험중립형
```

### 추천에 반영되는 대표 feature

- 최근 1개월 수익률
- 최근 1개월 변동성
- 뉴스 감성 점수
- 개별 종목 예측 수익률
- 보유 평가손익률
- PER / PBR / ROE / EV / BPS
- 매출액 / 영업이익 / 순이익 / 자본금
- 순이익률 / 자산회전율 / 재무레버리지

### 추천 결과

- 상위 3개 종목 선정
- 투자 성향별 가중치 적용
- 추천 이유 자연어 생성
- 추천 결과를 DB에 저장

---

## 13. 모델 산출물

### 개별 종목 예측 모델

- 저장 경로: `models/checkpoints/*.h5`
- 백업 경로: `models/backup/*.h5`

### 추천 모델

- 저장 경로: `models/stock_recommendation/saved/model_latest.pt`

### 로그 / 리포트

- 로그: `logs/`
- 학습 리포트: `data/reports/`
- 평가 리포트: `results/`

---

## 14. 현재 구조의 특징과 주의사항

이 저장소는 연구/개발과 서비스 연동 코드가 함께 들어 있는 형태입니다. 그래서 아래 특성이 있습니다.

- 완전히 독립적인 단일 앱이 아니라, 외부 DB와 백엔드 API에 의존합니다.
- 추천 모델은 사용자 데이터가 있는 MongoDB가 없으면 실제 운영 형태로 실행하기 어렵습니다.
- 일부 데이터 적재 스크립트는 운영/테스트 환경을 가정한 값에 의존하므로, 배포 전 환경 분리가 필요합니다.
- 프론트엔드와 백엔드는 이 저장소에 포함되어 있지 않습니다.

---

## 15. 권장 README 이해 포인트

이 저장소를 처음 보는 경우에는 아래 순서로 이해하면 빠릅니다.

1. `scripts/`  
   종목별 가격 예측 파이프라인

2. `models/stock_recommendation/`  
   사용자 맞춤 추천 파이프라인

3. `data/`  
   추천/예측에 필요한 원천 데이터 수집

4. `database/`  
   TimescaleDB 연동 및 테이블 생성

---

## 16. 한 줄 요약

이 저장소는 Team-DIFFUSER 서비스에서 **주가 예측 + 사용자 맞춤 종목 추천을 담당하는 AI 백엔드 레이어**입니다.
