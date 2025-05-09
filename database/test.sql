-- TimescaleDB 확장 활성화
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 주가 데이터 테이블
CREATE TABLE stock_prices (
    time TIMESTAMPTZ NOT NULL,
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(50) NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT
);

-- TimescaleDB 하이퍼테이블로 변환
SELECT create_hypertable('stock_prices', 'time');

-- 인덱스 생성
CREATE INDEX idx_stock_prices_code ON stock_prices (stock_code, time DESC);
