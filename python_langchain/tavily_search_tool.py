import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

load_dotenv()

# 1. 모델 및 도구 설정 (상세 옵션 추가)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearchResults(
    max_results=3, 
    search_depth="advanced", 
    include_answer=True
)
llm_with_tools = llm.bind_tools([search_tool])

# 2. 시스템 템플릿 추가
template = "당신은 사용자 질문에 답변하는 assistant입니다. 최신 정보를 검색할 때는 타빌리 검색 도구를 사용하세요."

questions = [
    "2025 롤드컵 우승자 이름",
    "2026년 1월 현재 비트코인 가격", 
    "최신 발표된 휴머노이드 로봇에 대해 알려줘"
]

print("🚀 최종 버전 질문 답변 시작합니다...\n")

for q in questions:
    print(f"❓ 질문: {q}")
    
    # 시스템 메시지 + 질문으로 시작
    messages = [SystemMessage(content=template), HumanMessage(content=q)]
    
    res = llm_with_tools.invoke(messages)
    messages.append(res) # AI의 첫 번째 응답 저장
    
    if res.tool_calls:
        for tool_call in res.tool_calls:
            print(f"🔍 '{tool_call['name']}' 검색 중...")
            try:
                out = search_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(
                    tool_call_id=tool_call["id"], 
                    content=str(out)
                ))
            except Exception as e:
                messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(e)))
        
        # 전체 기록을 바탕으로 최종 답변 생성
        final_res = llm.invoke(messages)
        print(f"💡 답변: {final_res.content}\n")
    else:
        print(f"💡 답변: {res.content}\n")