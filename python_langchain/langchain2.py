import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 환경 변수 로드 (.env 파일의 내용을 읽어옴)
load_dotenv()

# 2. 모델 선언 (GPT 모델 설정)
# ChatOpenAI는 내부적으로 'OPENAI_API_KEY'라는 이름을 자동으로 찾습니다.
llm = ChatOpenAI(
    model="gpt-4o",          # 사용할 모델 이름 (gpt-3.5-turbo 등)
    temperature=0.7,         # 창의성 조절 (0에 가까울수록 엄격, 1에 가까울수록 창의적)
    # api_key=os.getenv("OPENAI_API_KEY") # 수동으로 넣고 싶을 땐 이렇게도 가능!
)

# 3. 프롬프트 템플릿 선언
prompt = ChatPromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해줘.")

# 4. 체인 생성 (모델과 프롬프트를 연결)
chain = prompt | llm

# 5. 실행 및 결과 출력
response = chain.invoke({"topic": "가상환경(venv)"})
print(response.content)