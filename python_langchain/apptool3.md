👻 apptool3_md.md
📝 2026-01-19: LangChain & Streamlit 챗봇 실습

<--------------------------------------------------------------------->



1. 기술 스택


Framework: LangChain, Streamlit

Model: gpt-4o-mini

Tools: Tavily Search, yfinance, pytz



<--------------------------------------------------------------------->



2. 프로젝트 요약

주제: LangChain Tool Calling 기능을 활용한 지능형 챗봇 (Streamlit 기반)


주요 기능::

-Tavily Search: 최신 정보 검색 가능

-yfinance: 실시간 주식 시세 조회

-pytz: 전 세계 주요 도시 시간 조회

-Custom UI: 연핑크 & 연보라 그라데이션 테마 적용



<--------------------------------------------------------------------->



3. 핵심 메커니즘 (Tool Calling Flow)

-HumanMessage: 사용자의 질문 입력

-AIMessage (Tool Call): 모델이 질문을 분석하고 필요한 도구 호출 결정

-ToolMessage: 호출된 도구의 실제 실행 결과를 메시지 리스트에 추가

-Final AIMessage: 질문과 도구 결과를 종합하여 사용자에게 최종 답변 제공



<--------------------------------------------------------------------->