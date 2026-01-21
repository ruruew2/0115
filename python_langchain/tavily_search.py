from langchain_community.tools.tavily_search import TavilySearchResults

OPENAI_API_KEY="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"
TAVILY_API_KEY="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"


print("API 키 로드 완료 \n")

# 1. 변수명 오타 수정 (tavily)
search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=TAVILY_API_KEY  # <--- "tavily_api_key"라고 정확히 써줘야 함!
)

question = "2025년 롤드컵 우승자를 알려줘"

print(f"질문: {question}\n")
print("Tavily로 검색 중 입니다. . . . \n")

# 2. 딕셔너리 문법 수정 (콜론(:)과 따옴표 위치)
search_result = search_tool.invoke({"query": question})

print("검색 완료. 결과를 출력합니다. \n")

# 3. 변수명 공백 제거 및 언더바(_) 확인
for i, result in enumerate(search_result, 1):
    # 슬라이싱 [:200] 위치 수정
    print(f"\n{i}. {result.get('content', '')[:200]}...")
    print(f"        출처 :  {result.get('url', '')}")