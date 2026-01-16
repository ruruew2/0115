import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. 모델 설정
llm = ChatOpenAI(model="gpt-4.0", temperature=0.8)

# 2. 멀티턴용 프롬프트 설정 (MessagesPlaceholder가 핵심!)
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 열정적인 미슐랭 셰프야. 이전 대화 내용을 기억해서 손님과 티키타카 대화를 나눠줘."),
    MessagesPlaceholder(variable_name="chat_history"), # 이 자리에 대화 기록이 들어감
    ("user", "{input}")
])

# 3. 체인 생성
chain = prompt | llm   

# 4. 대화 기록을 담을 바구니 (메모리)
history = []

print("=== 👨‍🍳 셰프와의 1:1 대화 (종료하려면 'exit' 입력) ===")

while True:
    user_input = input("나: ")
    if user_input.lower() == 'exit':
        break

    # AI의 답변 생성 (기존 대화 기록인 history를 함께 전달)
    response = chain.invoke({
        "input": user_input,
        "chat_history": history
    })

    print(f"셰프: {response.content}")

    # 5. 대화 기록 업데이트 (나의 질문과 AI의 답변을 저장)
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))

    # (선택) 기록이 너무 길어지면 앞부분을 자르기도 하지만, 일단은 다 저장합니다!