import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_naver_data(query):
    url = f"https://search.naver.com/search.naver?query={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # 뉴스 및 블로그 제목 추출
    titles = soup.select(".news_tit, .api_txt_lines.total_tit")
    return "\n".join([f"- {t.get_text()}" for t in titles[:8]])

# 1. 모델 설정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8) # 창의력을 살짝 높임!

# 2. 페르소나 설정 (여기가 핵심!)
template = ChatPromptTemplate.from_messages([
    ("system", """너는 '맛에 미친 디저트 광인'이야. 말투는 '신사답지만 광기가 느껴지는' 스타일로 해줘.
    1. 제공된 데이터를 바탕으로 이야기하되, 너의 주관적인 감탄을 듬뿍 섞어줘.
    2. '두바이 쫀득 쿠키' 같은 단어가 나오면 환호성을 질러.
    3. 마지막엔 항상 '당장 먹으러 가야겠습니다...!'라고 마쳐줘."""),
    ("user", "오늘의 수집 데이터:\n{news_list}")
])

print("=== 🍪 광기의 디저트 특파원 ===")
keyword = input("어떤 맛집/뉴스를 털어볼까요?: ")

# 데이터 긁어오기
raw_data = get_naver_data(keyword)

if raw_data:
    chain = template | llm
    response = chain.invoke({"news_list": raw_data})
    
    print("\n" + "✨" * 20)
    print(response.content)
    print("✨" * 20)
else:
    print("아무것도 찾지 못했습니다... (눈물)")