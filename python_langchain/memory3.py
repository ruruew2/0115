import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. 모델 설정
llm = ChatOpenAI(model="gpt-4o", streaming=True)

# 2. 프롬프트 설정 (기억 보관함 포함)
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 독서 습관을 도와주는 친절한 독서 코치야. 이전 대화 내용을 바탕으로 구체적인 계획을 세워줘."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# 3. 체인 생성
chain = prompt | llm

# 4. 시뮬레이션할 질문 리스트
questions = [
    "내가 아까 추천해달라고 한 책들은 무엇이었지?",
    "서울의 인구는 2025년 기준 몇 명?"
]

# 5. 대화 기록 저장소
history = []

print("=== 📖 독서 코치 연속 대화 시뮬레이션 시작 ===\n")

for q in questions:
    print(f"나: {q}")
    
    # AI 답변 생성 (지금까지의 기록 history를 같이 보냄)
    response = chain.invoke({
        "input": q,
        "chat_history": history
    })
    
    print(f"AI: {response.content}")
    print("-" * 30)
    
    # 대화 기록 업데이트 (이게 있어야 다음 질문에서 기억을 함!)
    history.append(HumanMessage(content=q))
    history.append(AIMessage(content=response.content))
 
print("\n=== 시뮬레이션 종료 ===")

# k 값 = 최근 몇 턴의 대화를 유지할지 결정
# k=3: 최근 3턴의 대화만 유지 (사용자 질문 + AI 응답을 1턴으로 계산)
# 오래된 대화는 자동으로 삭제됨

# 권장 k 값:
# k=2~3: 간단한 FAQ, 단순 질의응답
# k=5 ⭐️: 가장 일반적, 시작 기본값으로 추천
# k=7~10: 상담, 교육, 코딩 도우미
# k=10~15: 복잡한 문제 해결, 장편 작업
# k=20+: 매우 특수한 경우 (비용 부담 큼)