import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 변수 로드
load_dotenv()

# 2. 페이지 기본 설정
st.set_page_config(page_title="Game Dev Glossary", layout="centered")

# 3. 세련된 레드 포인트 디자인 (커서 애니메이션 추가)
st.markdown("""
    <style>
    /* 전체 배경 */
    [data-testid="stAppViewContainer"] { background-color: #ffffff; }
    
    /* 사이드바 배경 및 우측 경계선 포인트 */
    [data-testid="stSidebar"] {
        background-color: #fff9f9;
        border-right: 2px solid #ff4b4b;
    }
    
    /* 커서 깜빡임 애니메이션 추가 */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    .blinking-cursor {
        color: #ff4b4b; /* 레드 포인트 색상 */
        font-weight: bold;
        animation: blink 0.8s step-end infinite;
        margin-left: 4px;
    }

    /* 사이드바 내 구분선 */
    hr {
        border: 0;
        height: 1px;
        background: #ff4b4b;
        margin: 1.5rem 0;
        opacity: 0.5;
    }

    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        border: 1px solid #e0e0e0;
        background-color: white;
        color: #555;
        border-radius: 4px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        background-color: #fffafa;
    }

    /* 입력창 */
    .stTextInput>div>div>input:focus {
        border-color: #ff4b4b !important;
        box-shadow: none !important;
    }

    /* 타이틀 및 폰트 설정 */
    h1 {
        color: #111;
        font-weight: 800;
        display: inline-block; /* 커서와 나란히 배치 */
    }
    
    h3::before {
        content: "■ ";
        color: #ff4b4b;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. 랭체인 로직 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 게임 용어를 분석하고 설명하는 15년 차 시니어 게임 개발자입니다.
    가벼운 농담이나 풍선 효과 같은 군더더기는 빼고, 아래 구조에 맞춰 전문적이고 명확하게 답변하세요:
    
    1. 핵심 요약: 해당 용어의 기술적/운영적 정의
    2. 메커니즘 분석: 실제 게임 로직이나 시스템에서 어떻게 작동하는지
    3. 현업 사례: 실제 유명 게임에서의 구체적인 적용 예시"""),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

# 5. 사이드바 구성
with st.sidebar:
    st.title("📂 Reference")
    st.caption("자주 찾는 게임 개발 키워드")
    st.markdown("---")
    
    st.subheader("⚖️ System & Balance")
    if st.button("Meta-gaming (메타)"): st.session_state.q = "게임에서 메타라는 용어의 정확한 정의가 뭐야?"
    if st.button("Balance Patch (너프/버프)"): st.session_state.q = "밸런스 패치에서 너프와 버프가 결정되는 기준이 뭐야?"
    if st.button("RNG (확률 시스템)"): st.session_state.q = "게임 설계에서 RNG가 사용자 경험에 미치는 영향은?"
    
    st.markdown("---")
    
    st.subheader("🌐 Network & Tech")
    if st.button("Network Latency (핑)"): st.session_state.q = "네트워크 핑(Ping)과 응답 속도의 기술적 관계는?"
    if st.button("Tick Rate (틱레이트)"): st.session_state.q = "FPS 게임 서버 성능에서 틱레이트가 중요한 이유가 뭐야?"
    if st.button("Optimization (최적화)"): st.session_state.q = "리소스 최적화와 드로우콜의 관계는?"
    
    st.markdown("---")
    
    if st.button("🔄 Clear Search"):
        st.session_state.q = ""
        st.rerun()

# 6. 메인 UI 구성 (타이틀 옆에 커서 클래스 적용)
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Game Dictionary<span class="blinking-cursor">_</span></h1>
        <p class="hero-subtitle">Senior Developer's Insights & Analysis</p>
        <p style="color: #888; font-size: 0.9rem;">시니어 개발자의 관점에서 분석한 전문 게임 용어 사전</p>
    </div>
    """, unsafe_allow_html=True)

# 입력창
user_input = st.text_input(
    "질문을 입력하세요.", 
    key="q", 
    label_visibility="collapsed", 
    placeholder="분석할 게임 용어를 입력하세요 (예: 딜찍누, 히트박스, 가챠)..."
)

# 7. 결과 출력 영역
if user_input:
    with st.status("용어 분석 중...", expanded=True) as status:
        st.markdown(f"### 🔍 '{user_input}' 분석 결과")
        response = chain.invoke({"question": user_input})
        st.markdown(response)
        status.update(label="분석 완료", state="complete")