from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

# 전체를 다 기억하는 메모리
memory = ConversationBufferMemory()

# ConversationChain은 메모리가 내장된 편리한 체인입니다.
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True # AI가 기억을 어떻게 꺼내 쓰는지 로그를 보여줍니다.
)

print(conversation.predict(input="안녕, 내 이름은 셰프야!"))
print(conversation.predict(input="오늘 추천 메뉴가 뭐야?"))
print(conversation.predict(input="내 이름이 뭐라고 했지?"))






from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

# k=2 : 최근 2개의 대화(나-AI 1세트)만 기억함
memory = ConversationBufferWindowMemory(k=1) 

conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)

print(conversation.predict(input="내 이름은 홍길동이야."))
print(conversation.predict(input="나는 서울에 살아."))
# k=1이면 '홍길동'은 이미 잊어버렸을 확률이 높습니다!
print(conversation.predict(input="내 이름이 뭐야?"))






from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

# 대화를 요약할 때도 LLM을 써야 하므로 llm을 인자로 넣어줍니다.
memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)

print(conversation.predict(input="어제는 두바이 쿠키를 먹었고, 오늘은 마라탕을 먹었어."))
print(conversation.predict(input="내일은 초밥을 먹을 예정이야."))
# 메모리 안을 들여다보면 대화가 요약되어 저장된 걸 볼 수 있습니다.
print(memory.load_memory_variables({}))