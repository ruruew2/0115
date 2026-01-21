import os
import pytz
import yfinance as yf
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# 환경 변수 로드
load_dotenv()

# --- 스타일 설정 ---
# --- 스타일 설정 (보라 & 핑크 에디션) ---
def apply_custom_style():
    st.markdown("""
    <style>
    /* 전체 배경을 살짝 어둡게 하거나 깔끔하게 유지 */
    .stApp {
        background-color: #ffffff;
    }
    
    .chat-container { display: flex; flex-direction: column; gap: 10px; }
    
    .chat-bubble { 
        padding: 12px 16px; 
        border-radius: 20px; 
        margin: 5px; 
        max-width: 75%; 
        line-height: 1.5; 
        word-wrap: break-word; 
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* 사용자: 핫핑크 느낌 */
    .user { 
        align-self: flex-end; 
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #fff !important; 
        border-bottom-right-radius: 2px;
    }

    /* AI: 연보라 느낌 */
    .ai { 
        align-self: flex-start; 
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        color: #444 !important; 
        border-bottom-left-radius: 2px;
    }

    /* 도구 결과: 조금 더 진한 보라색 테두리 */
    .tool { 
        align-self: flex-start; 
        background-color: #f3e5f5; 
        border-left: 5px solid #9c27b0; 
        font-size: 0.9em; 
        color: #4a148c !important;
    }

    .label { 
        font-weight: bold; 
        margin-bottom: 4px; 
        display: block; 
        color: #7b1fa2; 
    }
    
    /* 입력창 테두리 색상도 살짝 핑크로 */
    .stChatInputContainer {
        border-color: #fecfef !important;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# 1. 툴 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    '''현재 시간을 알려주는 tool입니다'''
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        return f'{timezone} ({location}) 현재 시각: {now}'
    except Exception as e: return f'오류: {e}'

@tool
def calculator(expression: str) -> str:
    '''간단한 산수 계산 도구'''
    return str(eval(expression))

@tool
def get_stock_price(symbol: str) -> str:
    '''주식 시세를 조회하는 도구'''
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if data.empty: return f'{symbol} 정보 없음'
        last = round(data['Close'].iloc[-1])
        info = stock.info
        name = info.get('longName', '정보 없음')
        return f"[{symbol}] {name} 현재가: {last:,}원"
    except Exception as ex: return f'오류: {ex}'

# 2. LLM 및 도구 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_current_time, calculator, get_stock_price]
tool_dict = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# 3. Streamlit UI
st.title("👻 AI _ Chat bot ")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "ai"
    st.markdown(f'<div class="chat-container"><div class="chat-bubble {role}">{message.content}</div></div>', unsafe_allow_html=True)

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.markdown(f'<div class="chat-container"><div class="chat-bubble user">{prompt}</div></div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        ai_msg = llm_with_tools.invoke(st.session_state.messages)
        
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                selected_tool = tool_dict[tool_call["name"]]
                tool_output = selected_tool.invoke(tool_call["args"])
                st.markdown(f"""
                    <div class="chat-bubble tool">
                        <span class="label">🛠️ {tool_call['name']} 결과</span>
                        {tool_output}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble ai">{ai_msg.content}</div>', unsafe_allow_html=True)