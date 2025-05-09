import requests

# API 엔드포인트 및 파라미터 설정
url = "https://apis.data.go.kr/1160100/service/GetCorpFinancialInfoService/getFinancialInfo"
params = {
    "serviceKey": "YOUR_API_KEY",
    "corpCode": "CORP_CODE",  # 기업 고유번호
    "bizYear": "2023",  # 사업연도
    "resultType": "json"
}



# API 호출
response = requests.get(url, params=params)

# 응답 데이터 출력
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")