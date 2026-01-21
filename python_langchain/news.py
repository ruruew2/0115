import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_naver_news(query):
    # 'where=news'를 빼고 일반 통합 검색으로 들어갑니다.
    url = f"https://search.naver.com/search.naver?query={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # 통합검색 결과(뉴스, 블로그, 웹사이트 등)에서 제목을 가져오는 태그들
    # .news_tit(뉴스), .api_txt_lines.total_tit(블로그/카페)
    titles = soup.select(".news_tit, .api_txt_lines.total_tit, .lnk_tit")
    
    results = []
    for i, title in enumerate(titles[:10]): # 정보량을 늘리기 위해 10개까지!
        results.append(f"{i+1}. {title.get_text().strip()}")
    
    return "\n".join(results)

# 1. 모델 설정
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 2. 템플릿 설정
template = ChatPromptTemplate.from_messages([
    ("system", "너는 실시간 뉴스 요약 전문가야. 제공된 최신 뉴스 목록을 보고 오늘의 핵심 상황을 브리핑해줘."),
    ("user", "키워드: {keyword}\n\n최신 뉴스 리스트:\n{news_list}")
])

print("=== 🗞️ 네이버 실시간 뉴스 요약기 ===")
keyword = input("궁금한 뉴스 키워드: ")

# 3. 크롤링 실행
print(f"\n🔍 네이버에서 '{keyword}'(으)로 검색 중...")
news_data = get_naver_news(keyword)

if news_data:
    # 4. GPT에게 요약 시키기
    chain = template | llm
    response = chain.invoke({"keyword": keyword, "news_list": news_data})
    
    print("\n" + "="*50)
    print(response.content)
    print("\n🔗 참고한 뉴스 출처:")
    print(news_data)
    print("="*50)
else:
    print("❌ 뉴스를 가져오지 못했습니다.")