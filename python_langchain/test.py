import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. 기억을 담을 리스트 (이게 AI의 뇌입니다)
chat_history = []

# 2. 템플릿에 '이전 대화 내용' 자리를 만들어줍니다 (history 부분)
template = ChatPromptTemplate.from_messages([
    ("system", "너는 5성급 호텔의 요리사야. 질문을 받으면 필요한 재료를 '장바구니 리스트' 형식으로 먼저 보여주고 조리법을 설명해줘."),
    MessagesPlaceholder(variable_name="history"), # 여기에 이전 대화가 들어감
    ("user", "{question}")
])

chat = ChatOpenAI(model="gpt-3.5-turbo")

print("=== 🧠 기억력이 생긴 요리사 챗봇 ===")

while True:
    user_q = input("질문: ")
    if user_q == "그만": break
    
    # 3. 템플릿에 현재 질문과 이전 대화 내역(history)을 함께 전달
    final_prompt = template.invoke({"history": chat_history, "question": user_q})
    
    response = chat.invoke(final_prompt)
    
    # 4. 대화 내역 업데이트 (나의 질문과 AI의 답변을 저장)
    chat_history.append(HumanMessage(content=user_q))
    chat_history.append(AIMessage(content=response.content))
    
    print(f"🤖 [요리사 AI]:\n{response.content}\n")