import yfinance as yf  # 맨 위에 이거 있어야 합니다! (pip install yfinance)
from langchain_core.tools import tool

@tool
def get_stock_price(symbol: str) -> str:
    """특정 종목의 주식 시세를 가져옵니다."""
    print(f'{symbol} 주식 시세 가져올 예정...')
    try:
        stock = yf.Ticker(symbol)
        # period는 "1d"(숫자1+d) 입니다.
        data = stock.history(period="1d")
        
        if data.empty:
            return f'{symbol}에 대한 정보를 찾을 수 없습니다.'
            
        # 데이터프레임에서 값 추출 (마지막 행의 값을 가져옴)
        # iloc[-1]을 써야 '값'만 깔끔하게 나옵니다.
        open_price = data['Open'].iloc[-1]
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close_price = data['Close'].iloc[-1]
        info = stock.info
        # print('info: ', info)

        name = info.get('longName','정보 없음')
        sector = info.get('sector',"정보 없음")
        industry = info.get('industry',"정보 없음")
        website = info.get('website',"정보 없음")
        market_cap = info.get('marketCap',"정보 없음")

# return과 문자열은 붙어있어야 하고, 변수는 { } 중괄호로 감쌉니다.
        return f"""
        [{symbol}] 시세 정보 (기업명: {name})
        - 현재가(종가): {close_price:.2f}
        - 시가: {open_price:.2f}
        - 고가: {high:.2f}
        - 저가: {low:.2f}
        - 산업(industry) : {industry}, 섹터 : {sector}
        - 시가 총액 : {market_cap}
        - 웹사이트 : {website}
        """

    except Exception as ex:
        return f'주식 정보 조회 중 오류 발생: {ex}'

# ==티커 (심볼)========================================
# SK하이닉스   000660   KOSPI   000660.KS
# 삼성전자       005930   KOSPI   005930.KS
# 카카오       035720   KOSPI   035720.KS
# 애플          AAPL    나스닥(.O)
# 구글          GOOG    뉴욕증권거래소(.N)
# 카카오게임즈와 같은 코스닥 종목을 조회한다면 293490.KQ 를 사용 (코스닥은 KQ)
# =====================================================

