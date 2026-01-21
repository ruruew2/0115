import requests

# 업비트 API에서 비트코인 가격 가져오기
url = "https://api.upbit.com/v1/ticker?markets=KRW-BTC"
data = requests.get(url).json()

price = data[0]['trade_price']
print(f"💰 [현재 비트코인 시세] {price:,} 원")
print("🚀 축하합니다! 이제 진짜 파이썬 마법사가 되셨습니다.")python -m pip install langchain langchain-openai