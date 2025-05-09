# run_all_tests.py
import argparse
from stock_recommendation_model import StockRecommendationModel
from test_recommendation_model import test_data_processing, test_basic_recommendation, test_investment_types, test_backend_api

def run_tests(test_backend=False):
    # 데이터 로드 및 전처리 검증
    print("=== 데이터 처리 테스트 ===")
    test_data_processing()
    print("\n")
    
    # 기본 추천 테스트
    print("=== 기본 추천 테스트 ===")
    test_basic_recommendation()
    print("\n")
    
    # 투자성향별 테스트
    print("=== 투자성향별 테스트 ===")
    test_investment_types()
    print("\n")
    
    # 백엔드 API 연동 테스트 (선택적)
    if test_backend:
        print("=== 백엔드 API 연동 테스트 ===")
        test_backend_api()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='종목 추천 모델 테스트')
    parser.add_argument('--backend', action='store_true', help='백엔드 API 연동 테스트 포함')
    args = parser.parse_args()
    
    run_tests(test_backend=args.backend)
