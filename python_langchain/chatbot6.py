import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. 환경 변수 로드
load_dotenv()

st.header("세션별 관리 챗봇")
st.subheader("RunnableWithMessageHistory 방식")
st.caption("사용자 세션 ID에 따라 대화를 기억합니다.")

# 2. 세션 상태에 대화 기록 저장소(store) 초기화
if "store" not in st.session_state:
    st.session_state.store = {}

# 3. 세션 ID에 맞는 히스토리를 가져오는 함수
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# 현재 접속자의 세션 ID 설정 (나중에 로그인 정보로 대체 가능)
user_id = st.sidebar.text_input("사용자 세션 id 입력", value="tbdl")
SESSION_ID = "K"

# 4. 모델 및 프롬프트 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 여행 도우미야. 사용자의 질문에 구체적이고 현실적으로 답변해"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 5. 체인 생성 및 히스토리 관리자 연결
chain = chat_prompt | llm

# ★ 변수명을 하나로 통일했습니다: chain_with_history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 6. 이전 대화 기록 화면에 출력
history = get_session_history(SESSION_ID)
for msg in history.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# 7. 사용자 입력 및 AI 응답 처리
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(user_input)

    # 설정값 구성 (중괄호 사용 주의!)
    config = {"configurable": {"session_id": SESSION_ID}}
    
    # AI 응답 생성 및 표시
    with st.chat_message("assistant"):
        with st.spinner("생각 중...."):
            response = chain_with_history.invoke(
                {"input": user_input},
                config=config
            )
            st.write(response.content)
