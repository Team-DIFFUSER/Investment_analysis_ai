from stock_recommendation_model import StockRecommendationModelV2

def test_basic_recommendation_v2():
    """MLP 기반 추천 기능 테스트"""
    model = StockRecommendationModelV2()
    model.train_mlp(
        target_col='1개월수익률',
        feature_cols=['1개월수익률_norm', '변동성_norm', '감성점수']
    )
    recommendations = model.get_recommendations_mlp(top_n=7)
    
    print("==== MLP 기반 추천 종목 목록 ====")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec['종목명']} ({rec['종목코드']}) - MLP_점수: {rec['MLP_점수_100']:.4f}")
        print(f"   추천이유: {rec['추천이유']}")
        print()

def test_data_processing_v2():
    """데이터 로드 및 전처리 테스트 (버전2)"""
    model = StockRecommendationModelV2()
    print(f"종목 기본정보: {model.stock_meta.shape[0]}개 종목")
    print(f"주가 데이터: {model.stock_prices['종목코드'].nunique()}개 종목, {model.stock_prices['기준일자'].nunique()}일")
    print(f"감성 분석 결과: {model.news_sentiment.shape[0]}개 데이터")
    print(f"수익률 계산 결과: {model.returns_df.shape[0]}개 종목")
    print(f"변동성 계산 결과: {model.volatility_df.shape[0]}개 종목")
    print(f"감성점수 집계 결과: {model.sentiment_df.shape[0]}개 종목")
    print(f"최종 특성 데이터셋: {model.features.shape[0]}개 종목, {model.features.shape[1]}개 특성")
    print("특성 목록:", model.features.columns.tolist())

if __name__ == "__main__":
    test_basic_recommendation_v2()
    # test_data_processing_v2()
