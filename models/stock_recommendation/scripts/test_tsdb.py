import os
import sys
import logging
import psycopg2
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TimescaleDBTester:
    def __init__(self):
        # 환경 변수 확인
        logger.info("환경 변수 확인:")
        logger.info(f"DB_HOST: {os.getenv('DB_HOST')}")
        logger.info(f"DB_PORT: {os.getenv('DB_PORT')}")
        logger.info(f"DB_NAME: {os.getenv('DB_NAME')}")
        logger.info(f"DB_USER: {os.getenv('DB_USER')}")
        logger.info(f"DB_SSL_MODE: {os.getenv('DB_SSL_MODE', 'require')}")
        
        # TimescaleDB 연결 정보
        self.ts_config = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'sslmode': os.getenv('DB_SSL_MODE', 'require')
        }
        logger.info(f"TimescaleDB 설정: {self.ts_config}")

    def get_timescale_connection(self):
        try:
            logger.info("TimescaleDB 연결 시도...")
            conn = psycopg2.connect(**self.ts_config)
            logger.info("TimescaleDB 연결 성공!")
            return conn
        except Exception as e:
            logger.error(f"TimescaleDB 연결 실패: {e}")
            raise

    def test_connection(self):
        """TimescaleDB 연결 테스트"""
        try:
            # 연결 시도
            conn = self.get_timescale_connection()
            cur = conn.cursor()
            
            # 테이블 목록 조회
            logger.info("테이블 목록 조회 중...")
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            tables = cur.fetchall()
            logger.info("\n사용 가능한 테이블 목록:")
            for table in tables:
                logger.info(f"- {table[0]}")
                
                # 각 테이블의 컬럼 정보 조회
                cur.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table[0]}'
                """)
                columns = cur.fetchall()
                logger.info("  컬럼 정보:")
                for col in columns:
                    logger.info(f"    - {col[0]}: {col[1]}")
            
            # 연결 종료
            cur.close()
            conn.close()
            logger.info("TimescaleDB 연결 종료")
            
            return True
            
        except Exception as e:
            logger.error(f"TimescaleDB 연결 실패: {e}")
            return False

if __name__ == "__main__":
    try:
        logger.info("TimescaleDB 테스트 시작")
        tester = TimescaleDBTester()
        success = tester.test_connection()
        if not success:
            logger.error("""
            TimescaleDB 연결에 실패했습니다. 다음 사항을 확인해주세요:
            1. PostgreSQL이 설치되어 있는지
            2. TimescaleDB 확장이 설치되어 있는지
            3. 환경 변수가 올바르게 설정되어 있는지
            4. 데이터베이스 서버가 실행 중인지
            5. 방화벽 설정이 올바른지
            """)
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}") 