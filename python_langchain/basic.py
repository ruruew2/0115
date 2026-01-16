import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. .env 파일에 저장된 보안 키를 불러옵니다
load_dotenv()
my_key = os.getenv("OPENAI_API_KEY")

# 2. OpenAI 클라이언트를 초기화합니다
client = OpenAI(api_key=my_key)

# 3. 인공지능에게 질문을 던집니다
response = client.chat.completions.create(
    model="gpt-3.5-turbo", # 또는 "gpt-4"
    messages=[
        {"role": "system", "content": "너는 아주 친절하고 유머러스한 파이썬 선생님이야."},
        {"role": "user", "content": "파이썬 가상환경 설정하느라 너무 힘들었어. 위로 한마디랑 앞으로의 응원 부탁해!"}
    ]
)

# 4. 인공지능의 답변을 출력합니다
print("🤖 AI의 답변:")
print(response.choices[0].message.content)