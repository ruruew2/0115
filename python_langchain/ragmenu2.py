"""
Pinecone을 사용한 간단한 RAG 예제
data 폴더의 텍스트 파일들을 임베딩하여 Pinecone에 저장하고 검색
"""
# pip install langchain-core langchain-openai langchain-pinecone langchain-community pinecone python-dotenv openai pillow
# pip install pinecone langchain pinecone
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec

# 환경 변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME=os.getenv('PINECONE_INDEX_NAME')
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')


def load_documents(path):
    document = [] # 변수명 리스트
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                # 도큐먼트에 메타정보 추가 (파일명)
                for doc in docs:
                    # [수정] == 가 아니라 = 로 할당해야 하며, 딕셔너리 접근은 [] 입니다.
                    doc.metadata['source'] = filename 

                document.extend(docs)
                print(f'load complete. {filename}=============')
            except Exception as ex:
                print(f"문서 로드 실패: {file_path} - {ex}")

    # [수정] 함수 이름이 아니라 결과 리스트를 반환해야 함
    return document





def split_document(documents, chunk_size=500, overlap_size=50):
    # [수정] 인자 이름을 documents로 맞춰줌
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap_size # [수정] 인자 이름은 chunk_overlap 입니다.
    )
    splits = text_splitter.split_documents(documents)
    # [수정] f-string 앞에 f가 빠지지 않았는지 확인
    print(f"문서 분할 완료, {len(splits)}개 청크 생성 완료")
    return splits



#pinecone db reset

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 기존 인덱스 확인
    existing_indexes=[index.name for index in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        #인덱스가 없으면 생성
        print(f'new index gen, {PINECONE_INDEX_NAME}')
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    else:
        # 기존 인덱스 사용
        print(f'기존 인덱스 사용: {PINECONE_INDEX_NAME}')
    return pc


def create_or_load_vectorstore(document=None,force_recreate=False):
    """
    Pinecone 벡터스토어 생성 또는 로드
    
    Args:
        documents: 임베딩할 문서 (새로 생성 시 필요)
        force_recreate: True면 기존 데이터 삭제하고 새로 생성
    """
    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # 인덱스 상태 확인 (벡터 수 카운트가 유일한 표준 방법)
    stats = index.describe_index_stats()
    vector_count = stats.get('total_vector_count', 0)
    
    # 기존 데이터가 있고 재생성 플래그가 없으면 로드
    if vector_count > 0 and not force_recreate:
        print(f"✓ 기존 벡터스토어 로드 (벡터 수: {vector_count})")
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
    else:
        # 데이터가 없거나 재생성 요청 시
        if force_recreate and vector_count > 0:
            print(f"기존 데이터 삭제 중... (벡터 수: {vector_count})")
            index.delete(delete_all=True)
        
        if document is None:
            raise ValueError("새로 생성하려면 documents 파라미터가 필요합니다")
        
        # Pinecone 벡터스토어 생성
        print(f"임베딩 생성 및 Pinecone 업로드 중... ({len(document)}개 문서)")
        vectorstore = PineconeVectorStore.from_documents(
            documents=document,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        print(f"✓ {len(document)}개 문서 임베딩 완료 및 Pinecone에 저장")
    
    return vectorstore

def create_rag_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 오타 수정: template
    template = '''
    당신은 레스토랑 정보를 제공하는 도우미입니다.
    다음 컨텍스트를 참고하여 질문에 답하세요.
    컨텍스트 정보가 없다면 '해당 정보 추적 불가'라고 답변하세요.

    컨텍스트 : {context}
    질문 : {question}
    답변: '''
    
    prompt = ChatPromptTemplate.from_template(template)

    # [수정] 검색된 문서들을 하나의 문자열로 합쳐주는 함수
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG 체인 구성 (LCEL 문법)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

    #참조 문서 메타정보 얻기
    source_docs= retriever.invoke(query)    

    print("="*50)
    print('참조 문서: ')
    for i, doc in enumerate(source_docs, start=1):
        print(f'{i} 출처: {doc.metadata.get('source','알 수 없음')}')
        print(f'내용 : {doc.page_content[:100]} ...')
    print("="*50)
    return response

def search_answer(query, rag_chain):
    # AI에게 질문 던지고 답변 받기
    answer = rag_chain.invoke(query)
    print(f"\n[AI 답변]: {answer}")
    print("-" * 50)

def main():
    # 1. 데이터 로드 및 분할
    document = load_documents('data')
    splits = split_document(document)

    # 2. Pinecone 초기화 및 벡터스토어 생성
    init_pinecone()
    # 처음 실행할 때는 force_recreate=True로 해서 데이터를 넣으세요!
    vector_store = create_or_load_vectorstore(document=splits, force_recreate=False)
    print("✓ 모든 준비가 완료되었습니다!")

    # 3. RAG 체인 생성
    rag_chain = create_rag_chain(vector_store)

    # 4. 테스트 질문 리스트
    example_queries = [
        "스테이크 메뉴의 가격과 특징을 알려주세요.",
        "해산물 요리는 어떤 것이 있나요?",
        "30만원 이상의 와인을 추천해주세요.",
        "디저트 메뉴가 있나요?",
    ]

    # print("\n" + "="*50)
    # print("레스토랑 AI 도우미 실행 결과")
    # print("="*50)

    # for query in example_queries:
    #     print(f"\n질문: {query}")
    #     # search_answer 대신 직접 invoke 호출
    #     answer = rag_chain.invoke(query)
    #     print(f"답변: {answer}")
    #     print("-" * 30)

    while True:
        user_query=input('질문 입력[종료시: exit]:').strip()
        if user_query.lower() in ['exit']:
            print('시스템을 종료합니다')
            exit(0)
        if not user_query:
            continue
        search_answer(user_query, rag_chain)


if __name__ == '__main__':
    main()

