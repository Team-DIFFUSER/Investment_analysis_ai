from stock_recommendation_model import StockRecommendationModel

def test_basic_recommendation():
    """기본 추천 기능 테스트"""
    model = StockRecommendationModel()
    recommendations = model.get_recommendations_for_user(top_n=5)
    
    print("==== 기본 추천 종목 목록 ====")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec['종목명']} ({rec['종목코드']}) - 점수: {rec['종합점수']:.4f}")
        print(f"   추천이유: {rec['추천이유']}")
        print()

def test_investment_types():
    """투자성향별 추천 테스트"""
    model = StockRecommendationModel()
    
    investment_types = ["안정형", "안정추구형", "위험중립형", "적극투자형", "공격투자형"]
    
    for inv_type in investment_types:
        print(f"\n==== {inv_type} 투자자 추천 종목 ====")
        model.calculate_scores(investment_type=inv_type)
        top_stocks = model.features.head(3)  # 각 유형별 상위 3개만
        
        for _, row in top_stocks.iterrows():
            print(f"{row['종목명']} ({row['종목코드']}) - 점수: {row['종합점수']:.4f}")
            print(f"추천이유: {model.generate_explanation(row, inv_type)}")
            print()

def test_data_processing():
    """데이터 로드 및 전처리 테스트"""
    model = StockRecommendationModel()
    
    print(f"종목 기본정보: {model.stock_meta.shape[0]}개 종목")
    print(f"주가 데이터: {model.stock_prices['종목코드'].nunique()}개 종목, {model.stock_prices['기준일자'].nunique()}일")
    print(f"감성 분석 결과: {model.news_sentiment.shape[0]}개 데이터")
    
    print(f"수익률 계산 결과: {model.returns_df.shape[0]}개 종목")
    print(f"변동성 계산 결과: {model.volatility_df.shape[0]}개 종목")
    print(f"감성점수 집계 결과: {model.sentiment_df.shape[0]}개 종목")
    
    print(f"최종 특성 데이터셋: {model.features.shape[0]}개 종목, {model.features.shape[1]}개 특성")
    print("특성 목록:", model.features.columns.tolist())

# def test_backend_api():
#     """백엔드 API 연동 테스트"""
#     model = StockRecommendationModel(
#         user_id="test_user",
#         backend_url="http://localhost:8000",
#         jwt_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."  # 실제 토큰으로 변경
#     )
    
#     profile = model.get_user_investment_profile()
#     print("사용자 투자성향 정보:", profile)
    
#     recommendations = model.get_recommendations_for_user(top_n=3)
#     for rec in recommendations:
#         print(f"{rec['종목명']} - {rec['추천이유']}")

if __name__ == "__main__":
    # 개별 테스트 함수 실행
    test_basic_recommendation()
    # test_investment_types()
    # test_data_processing()
    # test_backend_api()
