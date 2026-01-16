# chatbot5.py
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence, RunnableLambda

load_dotenv()
# 1. 스트림릿 설정
st.header("🗨️ LangChain 멀티턴 Memory 챗봇")
st.subheader("PromptTemplate +BufferMemory")
st.caption("이전 대화를 기억하는 멀티턴 챗봇")

# 2. session_state에 메모리 저장
if "memory" not in st.session_state:
    st.session_state["memory"]=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
memory = st.session_state['memory']

# 3. PromptTemplate 정의
template="""
너는 사용자 질문에 친절하게 답변하는 AI야.
이전 대화 내용은 다음과 같아:

{chat_history}
---
사용자 질문 : {user_input}
AI 답변 :
"""
prompt = PromptTemplate(input_variables=["chat_history","user_input"],template=template)

# 4. 모델 생성
llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 5. run_chain()함수 구현
def run_chain(inputs):
    """LLM호출->메모리 저장->결과 반환"""
    # [1] 메모리에서 기존 대화 가져오기
    history = memory.load_memory_variables({})["chat_history"]

    # [2] 프롬프트 구성
    prompt_text = prompt.format(chat_history=history, user_input=inputs["user_input"])

    # [3] 모델 호출
    result  = llm.invoke(prompt_text)
    answer = result.content # 모델 답변

    # [4] 메모리에 저장
    memory.save_context({"input":inputs["user_input"]},{"output": answer})
    return {"text":answer}

# 6. 함수 호출
chain = RunnableLambda(run_chain)
# 객체 체인으로 연결: prompt |llm|str_parser
# 사용자 정의 함수를 체인의 일부로 넣고 싶다면=>RunnableLambda를 사용한다
# 변수=RunnableLambda(함수)
# 변수 | 파서

# 7. UI출력 (기존 대화 기록 출력)
st.subheader("대화 내용")
# messages = memory['chat_history'].messages # ==>에러
###################################
messages = memory.chat_memory.messages
print(memory)
###################################
# ConversationBufferMemory 속성
# memory_key="chat_history",  => 프롬프트에 전달될 변수 이름
# chat_memory =>(ChatMessageHistory타입. 이 안에 messages속성이 있다)=> 실제 대화를 저장하는 내부 저장소

for msg in messages:
    role = "user" if isinstance(msg,HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# 8. 사용자 입력
user_input = st.chat_input("메시지 입력하세요")

if user_input:
    # 사용자 메시지 출력
    st.chat_message("user").write(user_input)

    # llm호출
    response = chain.invoke({"user_input": user_input})

    # ai메시지 출력
    answer = response['text']
    with st.chat_message("assistant"):
        st.write(answer)
    st.rerun()

# streamlit run chatbot5.py