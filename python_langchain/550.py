import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 

load_dotenv()

# 1. 모델 선언 (요구사항: gpt-4o-mini, temp 0.7)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. 프롬프트 설계
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 게임 용어를 쉽게 설명하는 게임 개발자입니다.
초보자도 이해할 수 있도록 친근한 말투로 다음 형식을 지켜 답변하세요:
1. 간단한 정의 (한 줄 요약할 것)
2. 실생활 비유 (일상적인 상황에 빗대어 설명할 것)
3. 게임 예시 (유명 게임에서의 활용 사례 접목)"""),
    ("user", "{question}") # {input} 대신 {question}으로 변수명 맞춤
])

# 3. 출력 파서 선언
parser = StrOutputParser()

# 4. LCEL 문법으로 체인 완성 (요구사항: prompt | llm | parser)
chain = prompt | llm | parser

# 5. 테스트 실행
questions = ["메타란?", "핑이란?", "너프가 뭐야?"]

for q in questions:
    print(f"질문: {q}")
    response = chain.invoke({"question": q}) 
    print(f"답변:\n{response}")
    print("-" * 30)