"""
LangGraph 없이 직접 구현하는 반복 추론 Agent - TAVILY 검색
Agent의 핵심인 반복 추론(Reasoning Loop)을 직접 구현하여 동작 원리 이해
－－－－－－－－－－－－－
목표(goal)를 달성하기 위해 스스로 판단하고, 행동을 선택하고, 외부 도구를 사용할 수 있는 "자율적 실행자".
   자가 판단 → 행동 선택 → 환경 반영 → 결과 분석
이 사이클을 ａｉ스스로 수행한다는 점이 핵심
Agent는 단순히 “답을 생성”하는 것이 아니라 목표(goal)를 달성하기 위해 단계별로 작업한다

Agent가 일할 때 핵심 기술이 바로 반복 추론이다.
이걸 다른 말로는 chain-of-thought, multi-step reasoning, self-reflection 등으로 부르기도 한다.
반복추론이란¿
큰 문제를 한 번에 해결하지 않고, 여러 단계로 쪼개서 순차적으로 해결하는 방식.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# 환경 변수 로드
load_dotenv()

def main():
    """
    반복추론이 가능한 agent 구현 및 실행
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')

    if not openai_api_key or not tavily_api_key:
        print('# api key none. plz check your api key')
        return

    # 모델 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Tavily 검색 도구 생성 (오타 수정: TavilySearchResults)
    search_tool = TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_answers=True,
        include_raw_content=False,
        include_images=False,
        api_key=tavily_api_key
    )

    # 도구 리스트 및 딕셔너리 생성 (오타 수정: () -> {})
    tools = [search_tool]
    tool_dict = {search_tool.name: search_tool}

    # llm 모델에 도구 바인딩
    llm_with_tools = llm.bind_tools(tools)

    # 질문 리스트
    questions = [
        "2025년 롤드컵 우승자 정보",
        "비트코인 가격 10만 달러 넘었는지 확인, 넘었다면 시기 알려줘"
    ]

    # 반복추론 실행
    for q in questions:
        print('-'*50)
        print(f"Q : {q}")
        print('-'*50)
        run_agent(q, llm, llm_with_tools, tool_dict, max_iteration=5)
        print('-'*50)

def run_agent(question, llm, llm_with_tools, tool_dict, max_iteration=5):
    """
    Agent 반복 추론 루프 (실제 동작 로직)
    """
    # 1. 사용자님이 작성하신 프롬프트가 여기에 들어갑니다.
    messages = [
        SystemMessage(content="""
당신은 최신 정보를 검색하여 정확하게 답변하는 AI Agent입니다.

주요 역할:
- 질문에 답하기 위해 필요한 정보를 검색합니다.
- 검색 결과가 불충분하면 추가 검색을 수행합니다.
- 충분한 정보를 얻었다면 명확하고 자세한 최종 답변을 제공합니다.
- 한국어로 답변합니다.

도구 사용 전략:
- 최신 정보가 필요하면 검색 도구를 사용하세요.
- 검색 결과가 불완전하면 다른 키워드로 다시 검색하세요.
- 여러 정보가 필요하면 순차적으로 검색하세요.
- 충분한 정보가 모였다면 도구 호출 없이 최종 답변을 작성하세요.
        """),
        HumanMessage(content=question)
    ]

    iteration_count = 0

    for i in range(max_iteration):
        iteration_count += 1  # 루프 시작하자마자 1부터 카운트!
        response = llm_with_tools.invoke(messages)

        
    # 2. (생략되었던 부분) 루프 구현
    for i in range(max_iteration):
        # LLM에게 현재 상황 판단 요청
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 도구를 안 써도 된다면(최종 답변 완료) 종료
        if not response.tool_calls:
            print(f"A : {response.content}")
            break

        # 도구를 써야 한다면(추가 정보 필요) 실행
        for tool_call in response.tool_calls:
            print(f"[시스템] 도구 호출 중: {tool_call['name']}...")
            
            # 도구 이름에 맞는 함수 찾아 실행
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            actual_tool = tool_dict[tool_name]
            
            # 검색 결과 얻기
            observation = actual_tool.invoke(tool_args)
            
            # 검색 결과를 메시지 기록에 추가 (그래야 AI가 읽고 다음 판단을 함)
            messages.append(ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            ))

            print(f"💡 이 답변을 위해 총 {iteration_count}번의 추론(루프)을 거쳤습니다.")

if __name__ == '__main__':
    main()