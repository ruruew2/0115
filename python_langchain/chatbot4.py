import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. api key 로드
load_dotenv()

# 2. AI모델 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 3. 스트림릿 기본 설정
st.set_page_config(page_title="AI챗봇1-Basic", layout="centered")
st.header("😻기본 챗봇 (langchain+streamlit)")
st.caption("ChatPromptTemplate + 대화 기록 연동")

# 4. 세션 상태 초기화 (메시지 저장 공간)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 5. PromptTemplate으로 템플릿 생성
# 시스템에게 역할을 부여하고, 이전 대화 기록(history)을 포함하도록 구성합니다.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "너는 친절하고 유머러스한 AI 조수야. 사용자의 질문에 재치 있게 대답해줘."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 6. 세션 상태에 저장된 기존 메시지가 있으면 출력
for message in st.session_state["chat_history"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# 7. 사용자 입력 => 세션 상태에 내 입력 메시지 저장
if user_input := st.chat_input("메시지를 입력하세요..."):
    # 화면에 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(user_input)
    
    # 8. 모델 호출해서 응답 받기
    with st.chat_message("assistant"):
        with st.spinner("AI가 생각 중..."):
            # 템플릿에 현재 입력과 전체 대화 기록을 주입
            chain = prompt_template | llm
            response = chain.invoke({
                "input": user_input,
                "history": st.session_state["chat_history"]
            })
            
            ai_answer = response.content
            st.write(ai_answer)

    # 9. 세션 상태에 내 질문과 응답 내용을 객체로 저장
    st.session_state["chat_history"].append(HumanMessage(content=user_input))
    st.session_state["chat_history"].append(AIMessage(content=ai_answer))

    