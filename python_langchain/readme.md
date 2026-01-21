# llm , langchain 학습 예제



오늘 하루 종일 패키지 에러와 오타 지옥을 뚫고 완성한 프로젝트네요! 고생하신 만큼 README.md도 멋지게 작성해서 박제해둡시다. 특히 langchain.memory 대신 최신 방식을 썼다는 점을 강조하면 훨씬 전문적으로 보입니다.

아래 내용을 그대로 복사해서 README.md 파일에 붙여넣으세요.

🤖 Multi-Session Travel Chatbot (LangChain & Streamlit)
이 프로젝트는 LangChain의 최신 RunnableWithMessageHistory 방식을 사용하여, 사용자별/세션별로 대화 흐름을 기억하는 멀티턴(Multi-turn) 여행 도우미 챗봇입니다.

🌟 주요 기능
세션별 대화 관리: 사이드바를 통해 Session ID를 변경하여 독립적인 대화 보관함을 운영할 수 있습니다.

실시간 대화 기억: ChatMessageHistory를 통해 이전 대화 내용을 AI가 기억하고 문맥에 맞는 답변을 제공합니다.

모던 UI: Streamlit의 chat_input, chat_message, spinner를 사용하여 깔끔한 채팅 인터페이스를 구현했습니다.

최신 LangChain 구조: langchain.memory(Legacy) 대신 최신 langchain-core 및 community 패키지 구조를 채택했습니다.

🛠 기술 스택
Language: Python 3.x

Framework: Streamlit

LLM: OpenAI GPT-4o-mini

Orchestration: LangChain (v0.3+)

Environment: python-dotenv


🚀 시작하기
1. 필수 패키지 설치
이 프로젝트는 패키지 분리 문제를 해결하기 위해 아래 패키지들을 사용합니다.

Bash
pip install streamlit langchain langchain-openai langchain-community python-dotenv
2. 환경 변수 설정
프로젝트 루트 폴더에 .env 파일을 생성하고 OpenAI API 키를 입력합니다.

코드 스니펫
OPENAI_API_KEY=your_api_key_here
3. 앱 실행
Bash
streamlit run chatbot5.py
📝 배운 점 (Troubleshooting)
langchain.memory 모듈이 최신 버전에서 분리된 문제를 해결하기 위해 langchain-community를 활용한 세션 관리 방식을 익혔습니다.

Streamlit의 session_state를 활용하여 새로고침 시에도 데이터가 휘발되지 않는 store 구조를 설계했습니다.

MessagesPlaceholder를 사용하여 프롬프트 내에 대화 기록이 주입되는 위치를 동적으로 제어하는 법을 배웠습니다.