import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models import ALL_STOCK_MODELS

from database.database import DatabaseManager

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_test_data(db_manager, start_date, end_date):
    """테스트 데이터 로드"""
    try:
        # 주가 데이터 로드
        stock_data = db_manager.get_stock_data(
            stock_code='066570',
            start_date=start_date,
            end_date=end_date
        )
        
        # 감성 데이터 로드
        sentiment_data = db_manager.get_sentiment_data(
            stock_code='066570',
            start_date=start_date,
            end_date=end_date
        )
        
        # 경제 데이터 로드
        economic_data = db_manager.get_economic_data(
            start_date=start_date,
            end_date=end_date
        )
        
        return stock_data, sentiment_data, economic_data
        
    except Exception as e:
        logger.error(f"테스트 데이터 로드 중 오류 발생: {str(e)}")
        raise

def main():
    logger = setup_logging()
    logger.info("모델 평가를 시작합니다.")
    
    try:
        # 결과 디렉토리 생성
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # 데이터베이스 연결 (여기서는 사용하지 않지만, 필요시 주석 해제)
        # db_manager = DatabaseManager()
        
        # 모든 모델 인스턴스 생성
        models = {name: model_class() for name, model_class in ALL_STOCK_MODELS.items()}

        all_results = {}
        for stock_name, model_instance in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"{stock_name} 모델 평가 시작")
            logger.info(f"{'='*50}")

            try:
                # 테스트 기간 설정 (각 모델의 load_data에서 처리되므로 여기서는 생략)
                # end_date = datetime.now()
                # start_date = end_date - timedelta(days=365)

                # 데이터 로드 및 전처리
                logger.info(f"{stock_name} 데이터 로드 및 전처리 중...")
                data = model_instance.load_data()
                if data.empty:
                    logger.error(f"{stock_name} 데이터 로드 실패. 다음 모델로 넘어갑니다.")
                    all_results[stock_name] = {'status': 'error', 'message': '데이터 로드 실패'}
                    continue
                
                processed_data = model_instance.enhanced_preprocessing(data)
                if processed_data.empty:
                    logger.error(f"{stock_name} 데이터 전처리 실패. 다음 모델로 넘어갑니다.")
                    all_results[stock_name] = {'status': 'error', 'message': '데이터 전처리 실패'}
                    continue

                X, y = model_instance.prepare_data(processed_data)
                if len(X) == 0 or len(y) == 0:
                    logger.error(f"{stock_name} 학습 데이터 준비 실패. 다음 모델로 넘어갑니다.")
                    all_results[stock_name] = {'status': 'error', 'message': '학습 데이터 준비 실패'}
                    continue

                # 모델 로드 (학습된 모델이 없으면 학습)
                logger.info(f"{stock_name} 모델 로드 중...")
                if model_instance.model is None:
                    logger.warning(f"{stock_name} 학습된 모델이 없습니다. 모델을 학습합니다.")
                    model_instance.train(X, y) # train 메서드 호출
                    model_instance.load_model() # 학습 후 모델 다시 로드

                if model_instance.model is None:
                    logger.error(f"{stock_name} 모델 로드 실패. 다음 모델로 넘어갑니다.")
                    all_results[stock_name] = {'status': 'error', 'message': '모델 로드 실패'}
                    continue

                # 평가 데이터 준비 (여기서는 X, y를 그대로 사용)
                X_test = X
                y_test = y

                # 모델 평가
                logger.info(f"{stock_name} 모델 평가 중...")
                metrics = model_instance.evaluate(X_test, y_test)
                all_results[stock_name] = {'status': 'success', 'metrics': metrics}

                # 결과 출력
                logger.info(f"\n=== {stock_name} 평가 결과 ===")
                logger.info(f"MSE: {metrics.get('mse', 'N/A'):.4f}")
                logger.info(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
                logger.info(f"R2: {metrics.get('r2', 'N/A'):.4f}")

            except Exception as e:
                logger.error(f"{stock_name} 모델 평가 중 오류 발생: {str(e)}")
                all_results[stock_name] = {'status': 'error', 'message': str(e)}

        # 최종 결과 리포트 생성
        success_count = sum(1 for r in all_results.values() if r['status'] == 'success')
        error_count = len(all_results) - success_count

        report_summary = f"\n{'='*50}\n전체 모델 평가 완료: {success_count}/{len(all_results)} 성공\n{'='*50}\n\n"
        report_details = []

        for stock_name, result in all_results.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                report_details.append(f"- {stock_name} (성공): MSE={metrics.get('mse', 'N/A'):.4f}, MAE={metrics.get('mae', 'N/A'):.4f}, R2={metrics.get('r2', 'N/A'):.4f}")
            else:
                report_details.append(f"- {stock_name} (실패): {result['message']}")

        final_report = report_summary + "\n".join(report_details)
        logger.info(final_report)

        # 리포트를 파일로 저장
        report_path = results_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        logger.info(f"전체 평가 리포트가 {report_path}에 저장되었습니다.")

        if error_count > 0:
            sys.exit(1)

    except Exception as e:
        logger.error(f"평가 실행 중 치명적인 오류 발생: {str(e)}")
        sys.exit(1)
    finally:
        # 데이터베이스 연결 종료 (필요시 주석 해제)
        # if 'db_manager' in locals():
        #     db_manager.close()
        logger.info("평가 프로세스 종료.")

if __name__ == "__main__":
    main()
 