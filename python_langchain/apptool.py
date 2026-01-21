import os
import pytz  # 시간대 계산용
from datetime import datetime
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# .env 로드
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    print(f"API Key loaded successfully: {api_key[:5]}...")
else:
    print('ERROR: API Key is None. Please check your .env file.')

# 1. 도구 정의 
@tool
def get_current_time(location: str, timezone: str) -> str:
    """
    특정 지역의 현재 시간을 알려주는 도구입니다.
    
    Args:
        location (str): 도시 이름 (예: 서울, 뉴욕)
        timezone (str): 타임존 문자열 (예: Asia/Seoul, America/New_York)
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        # 아래 줄 끝에 닫는 따옴표(")가 반드시 있어야 합니다.
        return f"{timezone} ({location})의 현재 시각: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"시간을 가져오는 중 오류 발생: {e}"

# --------------------- main 실행부 ----------------------------

def run_tool():
    # LLM 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # LLM에 도구 연결
    llm_with_tools = llm.bind_tools([get_current_time])

    # 테스트 질문
    query = "서울 지금 몇 시야?"
    print(f"\n질문: {query}")

    # 모델 호출
    messages = [HumanMessage(content=query)]
    ai_msg = llm_with_tools.invoke(messages)

    # AI가 도구를 사용하기로 했는지 확인 및 실행
    if ai_msg.tool_calls:
        print("\n[AI가 도구 호출을 결정함]")
        for tool_call in ai_msg.tool_calls:
            # 실제 함수 실행
            result = get_current_time.invoke(tool_call['args'])
            print(f"결과: {result}")
    else:
        print(f"\nAI 응답: {ai_msg.content}")

if __name__ == '__main__':
    run_tool()