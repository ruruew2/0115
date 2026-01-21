import streamlit as st
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. API 키 설정 (보안상 직접 입력하거나 환경변수 사용) ---
os.environ["OPENAI_API_KEY"] = " "

st.set_page_config(page_title="City Plan RAG", page_icon="🏙️")
st.title("🏙️ 서울 & 뉴욕 도시계획 Q&A")

# --- 2. RAG 시스템 초기화 (피클 없이 직접 로드) ---
@st.cache_resource
def init_rag():
    # 실제 파일 경로 (이 부분이 정확해야 합니다!)
    files = [
        r'C:\Users\user\Desktop\0115\제외파일\g\data.pdf',
        r'C:\Users\user\Desktop\0115\제외파일\g\/OneNYC_2050_Strategic_Plan.pdf'
    ]
    
    docs = []
    for f in files:
        if os.path.exists(f):
            # PyMuPDFLoader는 한글 인식률이 매우 높습니다
            loader = PyMuPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("⚠️ 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요!")
        return None

    # 텍스트 나누기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 벡터 저장소 만들기
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db_new" # 아예 새로운 DB 폴더 사용
    )
    return vector_store.as_retriever(k=3)

# 리트리버 로드
retriever = init_rag()
llm = ChatOpenAI(model="gpt-4o-mini")

# --- 3. 체인 설정 ---
# 질문 보정용
q_augment_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 질문 보정 전문 AI야. 대화 맥락을 보고 사용자의 짧은 질문을 검색 가능한 명확한 질문으로 바꿔줘."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])
q_augment_chain = q_augment_prompt | llm | StrOutputParser()

# 답변용
qna_prompt = ChatPromptTemplate.from_messages([
    ("system", "아래의 컨텍스트를 참고해서 답변해:\n\n{context}"),
    MessagesPlaceholder(variable_name="messages"),
])
document_chain = create_stuff_documents_chain(llm, qna_prompt)

# --- 4. 채팅 UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

if prompt := st.chat_input("궁금한 점을 물어보세요!"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. 질문 보정
        aug_query = q_augment_chain.invoke({
            "chat_history": st.session_state.messages[:-1],
            "query": prompt
        })
        # 2. 검색 및 답변
        docs = retriever.invoke(aug_query)
        res = document_chain.invoke({"messages": st.session_state.messages, "context": docs})
        
        st.markdown(res)
        st.session_state.messages.append(AIMessage(content=res))



# import streamlit as st
# import os
# import pickle
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain.chains.combine_documents import create_stuff_documents_chain


# OPENAI_API_KEY = "여기에_API_키를_입력하세요" # 보안을 위해 실제 환경에선 st.secrets 사용 권장
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# st.set_page_config(page_title="City Plan RAG Chatbot", page_icon="🏙️")
# st.title("🏙️ 서울 & 뉴욕 도시계획 Q&A")

# # --- 2. 리소스 로드 함수 (캐싱 처리) ---
# @st.cache_resource
# def init_rag_system():
#     # PDF 로드 및 피클 저장 로직 (경로는 환경에 맞게 수정 필요)
#     pdf_path = 'C:/Users/user/Desktop/0115/g/data.pdf' # 예시 경로
#     pickle_path = 'data.pkl'
    
#     if os.path.exists(pickle_path):
#         with open(pickle_path, 'rb') as f:
#             data_seoul = pickle.load(f)
#     else:
#         loader = PyPDFLoader(pdf_path)
#         data_seoul = loader.load()
#         with open(pickle_path, 'wb') as f:
#             pickle.dump(data_seoul, f)

#     # 텍스트 분할
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     all_splits = text_splitter.split_documents(data_seoul)

#     # 임베딩 및 벡터스토어
#     embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
#     vector_store = Chroma.from_documents(
#         documents=all_splits, 
#         embedding=embeddings,
#         persist_directory="./chroma_db_streamlit"
#     )
#     return vector_store.as_retriever(k=3)

# retriever = init_rag_system()

# # LLM 설정
# llm = ChatOpenAI(model="gpt-4o-mini")

# # --- 3. 체인 생성 ---
# # 질문 보정(Augmentation) 체인
# q_augment_prompt = ChatPromptTemplate.from_messages([
#     ("system", "너는 질문 보정 전문 AI야. 이전 대화를 참고해서 사용자의 모호한 질문을 명확한 질문으로 교정해줘."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{query}")
# ])
# q_augment_chain = q_augment_prompt | llm | StrOutputParser()

# # 답변 생성 체인
# qna_prompt = ChatPromptTemplate.from_messages([
#     ("system", "아래 제공된 컨텍스트를 사용해서 질문에 답하세요.\n\n{context}"),
#     MessagesPlaceholder(variable_name="messages"),
# ])
# document_chain = create_stuff_documents_chain(llm, qna_prompt)

# # --- 4. 세션 상태 초기화 (채팅 기록 저장) ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # --- 5. 채팅 UI 구성 ---
# # 기존 메시지 출력
# for message in st.session_state.messages:
#     with st.chat_message(message.type):
#         st.markdown(message.content)

# # 사용자 입력 처리
# if prompt := st.chat_input("질문을 입력하세요..."):
#     # 1. 사용자 메시지 추가 및 화면 표시
#     st.session_state.messages.append(HumanMessage(content=prompt))
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("생각 중..."):
#             # 2. 질문 증강 (Query Augmentation)
#             augmented_query = q_augment_chain.invoke({
#                 "chat_history": st.session_state.messages[:-1],
#                 "query": prompt
#             })
            
#             # 3. 문서 검색 (Retriever)
#             docs = retriever.invoke(augmented_query)
            
#             # 4. 최종 답변 생성
#             response = document_chain.invoke({
#                 "messages": st.session_state.messages,
#                 "context": docs
#             })
            
#             st.markdown(response)
#             # 참고 문헌 표시 (선택 사항)
#             with st.expander("참고 문서 확인"):
#                 for i, doc in enumerate(docs):
#                     st.write(f"**Source {i+1}:** {doc.page_content[:200]}...")

#     # 5. AI 답변 저장
#     st.session_state.messages.append(AIMessage(content=response))