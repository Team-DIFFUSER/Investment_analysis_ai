# test_api_server.py
from flask import Flask, request, jsonify
from stock_recommendation_model import StockRecommendationModel

app = Flask(__name__)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    investment_type = data.get('investment_type')
    top_n = data.get('top_n', 5)
    
    model = StockRecommendationModel()
    
    if investment_type:
        # 직접 투자성향을 지정한 경우
        model.calculate_scores(investment_type)
        top_stocks = model.features.head(top_n)
        recommendations = []
        for _, row in top_stocks.iterrows():
            recommendations.append({
                '종목코드': row['종목코드'],
                '종목명': row['종목명'],
                '종합점수': float(row['종합점수']),
                '추천이유': model.generate_explanation(row, investment_type)
            })
    else:
        # 사용자 ID를 기반으로 투자성향 가져오기
        recommendations = model.get_recommendations_for_user(top_n=top_n)
    
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
