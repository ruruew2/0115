import os
import pytz
import yfinance as yf
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# 환경 변수 로드
load_dotenv()

# --- 스타일 설정 ---
def apply_custom_style():
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .chat-container { display: flex; flex-direction: column; gap: 10px; }
    .chat-bubble { 
        padding: 12px 16px; border-radius: 20px; margin: 5px; 
        max-width: 75%; line-height: 1.5; word-wrap: break-word; font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .user { 
        align-self: flex-end; 
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #fff !important; border-bottom-right-radius: 2px;
    }
    .ai { 
        align-self: flex-start; 
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        color: #444 !important; border-bottom-left-radius: 2px;
    }
    .tool { 
        align-self: flex-start; background-color: #f3e5f5; 
        border-left: 5px solid #9c27b0; font-size: 0.9em; color: #4a148c !important;
    }
    .label { font-weight: bold; margin-bottom: 4px; display: block; color: #7b1fa2; }
    .stChatInputContainer { border-color: #fecfef !important; }
    </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# 1. 도구 정의
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
        return f"[{symbol}] 현재가: {last:,}원"
    except Exception as ex: return f'오류: {ex}'

# --- Tavily 툴 생성 ---
tavily_tool = TavilySearchResults(max_results=2)

# 2. LLM 및 도구 바인딩 (한 번만 정의!)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_current_time, calculator, get_stock_price, tavily_tool]
tool_dict = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# 3. Streamlit UI
st.title("👻 ChatbotAILangChain ")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for message in st.session_state.messages:
    if isinstance(message, ToolMessage): continue # 툴 메시지는 화면에 따로 표시하므로 패스
    role = "user" if isinstance(message, HumanMessage) else "ai"
    st.markdown(f'<div class="chat-container"><div class="chat-bubble {role}">{message.content}</div></div>', unsafe_allow_html=True)

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.markdown(f'<div class="chat-container"><div class="chat-bubble user">{prompt}</div></div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        ai_msg = llm_with_tools.invoke(st.session_state.messages)
        st.session_state.messages.append(ai_msg)

        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                selected_tool = tool_dict[tool_call["name"]]
                tool_output = selected_tool.invoke(tool_call["args"])
                
                # 툴 실행 결과 출력
                st.markdown(f"""
                    <div class="chat-bubble tool">
                        <span class="label">🛠️ {tool_call['name']} 결과</span>
                        {tool_output}
                    </div>
                """, unsafe_allow_html=True)
                
                # 기록에 추가
                st.session_state.messages.append(ToolMessage(
                    tool_call_id=tool_call["id"], 
                    content=str(tool_output)
                ))
            
            # 최종 답변 생성
            final_ai_msg = llm.invoke(st.session_state.messages)
            st.session_state.messages.append(final_ai_msg)
            st.markdown(f'<div class="chat-bubble ai">{final_ai_msg.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble ai">{ai_msg.content}</div>', unsafe_allow_html=True)