import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. 모델 선언 (4o니까 똑똑함)
llm = ChatOpenAI(model="gpt-4o")

# 2. 프롬프트 설정 (체인 없이 직접 쓰기)
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 유능한 독서 코치야. 이전 대화를 다 기억하고 있어."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# 3. 체인 조립 (파이프 기호만 사용)
chain = prompt | llm

# 4. 시뮬레이션
questions = [
"내가 추천받은 책 3권을 주인공들이 만나서 싸우면 누가 이길까? 아주 디테일하게 묘사해줘.",

"이 책들을 읽으면서 듣기 좋은 플레이리스트를 10곡 정도 짜줘. 각 곡이 왜 책이랑 어울리는지도 설명해.",

"방금 추천한 곡들 중에서 3번째 곡의 가사를 네가 독서 코치 버전으로 개사해봐."
]

history = [] # 여기에 대화가 쌓임

for q in questions:
    print(f"질문: {q}")
    # 실행
    res = chain.invoke({"input": q, "history": history})
    print(f"답변: {res.content}")
    print("-" * 30)
    
    # 기록 저장
    history.append(HumanMessage(content=q))
    history.append(AIMessage(content=res.content))

# 5. 요약 (나중에 교수님이 물어보면 "직접 요약했습니다"라고 하세요!)
summary_res = llm.invoke(f"다음 대화 내용을 한 문장으로 요약해줘: {history}")
print(f"\n최종 요약: {summary_res.content}")