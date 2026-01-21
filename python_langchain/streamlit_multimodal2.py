"""
멀티모달 RAG 예제
텍스트, 이미지, PDF 등 다양한 형식의 데이터를 처리

주요 기능:
1. 텍스트 파일 (.txt) 로드 및 처리
2. 이미지 파일 (.jpg, .png 등) GPT-4 Vision API로 분석
3. PDF 파일 (.pdf) 텍스트 추출
4. 모든 문서를 벡터 임베딩하여 Pinecone에 저장
5. 통합 검색 및 질의응답
"""

# 필수 패키지 설치
# pip install Pillow  # 이미지 처리 라이브러리 (열기/크기조절/포맷변환/base64인코딩)
# pip install langchain langchain-openai langchain-pinecone langchain-community
# pip install pinecone-client python-dotenv

import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain 핵심 컴포넌트
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 문서 분할
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI 임베딩 및 LLM
from langchain_pinecone import PineconeVectorStore  # Pinecone 벡터스토어

# 문서 로더
from langchain_community.document_loaders import (
    TextLoader,  # 텍스트 파일 로더
    UnstructuredImageLoader,  # 이미지 로더 (미사용)
    PyPDFLoader,  # PDF 로더
    DirectoryLoader  # 디렉토리 전체 로드
)

# LangChain 체인 구성 요소
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 템플릿
from langchain_core.runnables import RunnablePassthrough  # 체인 연결
from langchain_core.output_parsers import StrOutputParser  # 출력 파싱
from langchain_core.documents import Document  # 문서 객체
from langchain_core.messages import HumanMessage  # Vision API용 메시지

# Pinecone 및 이미지 처리
from pinecone import Pinecone, ServerlessSpec  # Pinecone 클라이언트
from PIL import Image  # 이미지 처리 (리사이징, 포맷 변환)
import base64  # 이미지 base64 인코딩
from io import BytesIO  # 메모리 버퍼

# =================================================================
# 환경 변수 로드 (.env 파일에서 API 키 읽기)
# =================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API 키
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Pinecone API 키
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # Pinecone 리전 (예: us-east-1)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "restaurant-multimodal")  # 인덱스 이름


# =================================================================
# 멀티모달 문서 로더 클래스
# =================================================================
class MultiModalDocumentLoader:
    """
    다양한 형식의 문서를 로드하는 클래스
    
    지원 형식:
    - 텍스트 파일 (.txt): DirectoryLoader로 일괄 로드
    - 이미지 파일 (.jpg, .png, .gif, .bmp): GPT-4 Vision API로 분석
    - PDF 파일 (.pdf): PyPDFLoader로 텍스트 추출
    
    Attributes:
        data_directory (str): 문서가 저장된 디렉토리 경로
        documents (list): 로드된 Document 객체 리스트
    """
    
    def __init__(self, data_directory):
        """초기화 메서드"""
        self.data_directory = data_directory  # 데이터 디렉토리 경로
        self.documents = []  # 로드된 문서 저장 리스트
    
    def load_text_files(self):
        """
        텍스트 파일 로드
        
        - DirectoryLoader로 data 폴더 내 모든 .txt 파일 검색
        - UTF-8 인코딩으로 텍스트 읽기
        - 메타데이터에 타입('text') 및 파일명 추가
        
        Returns:
            list: 로드된 Document 객체 리스트
        """
        print("📄 텍스트 파일 로드 중...")
        
        # DirectoryLoader: 디렉토리 내 특정 패턴 파일 일괄 로드
        loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.txt",  # 모든 하위 폴더의 .txt 파일
            loader_cls=TextLoader,  # 텍스트 로더 사용
            loader_kwargs={"encoding": "utf-8"}  # UTF-8 인코딩
        )
        docs = loader.load()
        
        # 각 문서에 메타데이터 추가
        for doc in docs:
            doc.metadata['type'] = 'text'  # 문서 타입 지정
            doc.metadata['source'] = os.path.basename(doc.metadata['source'])  # 파일명만 추출
        
        self.documents.extend(docs)  # 전체 문서 리스트에 추가
        print(f"✓ 텍스트 파일 {len(docs)}개 로드 완료")
        return docs
    
    def load_pdf_files(self):
        """
        PDF 파일 로드
        
        - PyPDFLoader로 PDF 파일의 텍스트 추출
        - 각 페이지가 별도 Document로 생성됨
        - 메타데이터에 타입('pdf') 및 파일명 추가
        
        Returns:
            list: 로드된 Document 객체 리스트 (페이지 단위)
        """
        print("📑 PDF 파일 로드 중...")
        
        # data 폴더 내 모든 .pdf 파일 검색
        pdf_files = list(Path(self.data_directory).glob("**/*.pdf"))
        docs = []
        
        # 각 PDF 파일 처리
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))  # PDF 로더 생성
                pdf_docs = loader.load()  # PDF 페이지별 텍스트 추출
                
                # 각 페이지에 메타데이터 추가
                for doc in pdf_docs:
                    doc.metadata['type'] = 'pdf'  # 문서 타입 지정
                    doc.metadata['source'] = pdf_file.name  # 파일명
                
                docs.extend(pdf_docs)  # 전체 문서 리스트에 추가
                print(f"✓ PDF 로드: {pdf_file.name}")
            except Exception as e:
                print(f"✗ PDF 로드 실패: {pdf_file.name} - {e}")
        
        self.documents.extend(docs)
        print(f"✓ PDF 파일 {len(docs)}페이지 로드 완료")
        return docs
    
    def load_image_files(self):
        """
        이미지 파일 로드 및 Vision API로 분석
        
        - 지원 형식: .jpg, .jpeg, .png, .gif, .bmp
        - GPT-4 Vision API로 이미지 내용 분석 (메뉴, 가격, 재료 등 추출)
        - 분석 결과를 텍스트로 변환하여 Document 생성
        
        Returns:
            list: Vision API 분석 결과가 담긴 Document 객체 리스트
        """
        print("🖼️  이미지 파일 로드 중...")
        
        # 지원하는 이미지 확장자 목록
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
        image_files = []
        
        # 각 확장자별로 파일 검색
        for ext in image_extensions:
            image_files.extend(Path(self.data_directory).glob(f"**/{ext}"))
        
        docs = []
        # 각 이미지 파일 처리
        for img_file in image_files:
            try:
                # GPT-4 Vision API로 이미지 분석 → 텍스트 설명 생성
                doc = self._process_image_with_vision(img_file)
                if doc:
                    docs.append(doc)
                    print(f"✓ 이미지 처리: {img_file.name}")
            except Exception as e:
                print(f"✗ 이미지 처리 실패: {img_file.name} - {e}")
        
        self.documents.extend(docs)  # 전체 문서 리스트에 추가
        print(f"✓ 이미지 파일 {len(docs)}개 처리 완료")
        return docs
    
    def _process_image_with_vision(self, image_path):
        """
        GPT-4 Vision API를 사용하여 이미지 내용 분석
        
        프로세스:
        1. 이미지를 PIL로 열기
        2. 크기 조절 (800x800 최대, 비용 절감 목적)
        3. PNG 포맷으로 변환
        4. base64 인코딩 (Vision API 전송용)
        5. GPT-4 Vision API 호출 (이미지 + 텍스트 프롬프트)
        6. 분석 결과를 Document 객체로 반환
        
        Args:
            image_path (Path): 이미지 파일 경로
            
        Returns:
            Document: 이미지 분석 결과가 담긴 Document 객체
            None: 처리 실패 시
        """
        try:
            # ========== Step 1: 이미지를 base64로 인코딩 ==========
            with Image.open(image_path) as img:
                # 이미지 크기 조절 (비용 절감 + 속도 향상)
                # 최대 800x800 픽셀, 비율 유지
                img.thumbnail((800, 800))
                
                # 메모리 버퍼에 PNG 포맷으로 저장
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                
                # base64 인코딩 (Vision API 전송 형식)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # ========== Step 2: GPT-4 Vision API 설정 ==========
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Vision 지원 모델
                # max_tokens=500  # 최대 응답 길이
            )
            
            # ========== Step 3: Vision API 호출 메시지 구성 ==========
            # HumanMessage: 텍스트 + 이미지 동시 전송 가능
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": """이 이미지를 자세히 분석해주세요.
만약 레스토랑 메뉴나 음식 사진이라면:
- 메뉴 이름
- 가격 (있다면)
- 재료나 설명
- 기타 특징

만약 와인이나 음료라면:
- 제품명
- 종류/품종
- 가격 (있다면)
- 특징

일반 이미지라면 주요 내용을 설명해주세요."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"  # base64 인코딩된 이미지
                        }
                    }
                ]
            )
            
            # ========== Step 4: Vision API 실행 ==========
            response = llm.invoke([message])  # GPT-4 Vision 호출
            description = response.content  # 이미지 분석 결과 텍스트
            
            # ========== Step 5: Document 객체 생성 ==========
            doc = Document(
                page_content=f"[이미지: {image_path.name}]\n{description}",  # 분석 결과 저장
                metadata={
                    'type': 'image',  # 문서 타입
                    'source': image_path.name,  # 파일명
                    'image_path': str(image_path)  # 원본 이미지 경로 (UI 표시용)
                }
            )
            
            return doc
            
        except Exception as e:
            print(f"이미지 처리 오류: {e}")
            return None
    
    def load_all(self):
        """모든 형식의 파일 로드"""
        print("\n" + "="*60)
        print("멀티모달 문서 로드 시작")
        print("="*60 + "\n")
        
        self.load_text_files()
        # self.load_pdf_files() #레스토랑에는 pdf파일 없음
        self.load_image_files()
        
        print(f"\n✓ 총 {len(self.documents)}개 문서 로드 완료")
        print(f"   - 텍스트: {sum(1 for d in self.documents if d.metadata.get('type') == 'text')}개")
        print(f"   - PDF: {sum(1 for d in self.documents if d.metadata.get('type') == 'pdf')}개")
        print(f"   - 이미지: {sum(1 for d in self.documents if d.metadata.get('type') == 'image')}개\n")
        
        return self.documents


# =================================================================
# 유틸리티 함수들
# =================================================================

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    문서를 작은 청크로 분할
    
    - RecursiveCharacterTextSplitter: 문단, 문장, 단어 단위로 재귀적 분할
    - chunk_size: 각 청크의 최대 길이 (문자 수)
    - chunk_overlap: 인접 청크 간 겹치는 부분 (문맥 유지)
    
    Args:
        documents (list): 분할할 Document 객체 리스트
        chunk_size (int): 청크 최대 크기 (default: 500)
        chunk_overlap (int): 청크 간 겹침 (default: 50)
        
    Returns:
        list: 분할된 Document 청크 리스트
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 청크 최대 길이
        chunk_overlap=chunk_overlap,  # 겹침는 문자 수
        length_function=len,  # 길이 계산 함수
    )
    
    splits = text_splitter.split_documents(documents)  # 분할 실행
    print(f"✓ 문서 분할 완료: {len(splits)}개 청크 생성")
    
    return splits


def initialize_pinecone():
    """
    Pinecone 초기화 및 인덱스 생성
    
    - Pinecone 클라이언트 초기화
    - 인덱스 존재 여부 확인
    - 없으면 새로 생성 (1536차원, cosine 유사도, AWS serverless)
    
    Returns:
        Pinecone: Pinecone 클라이언트 객체
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)  # Pinecone 클라이언트 초기화
    existing_indexes = [index.name for index in pc.list_indexes()]  # 기존 인덱스 목록
    
    # 인덱스가 없으면 새로 생성
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"새로운 인덱스 생성: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,  # 인덱스 이름
            dimension=1536,  # OpenAI text-embedding-3-small 차원
            metric='cosine',  # 유사도 계산 방식 (cosine similarity)
            spec=ServerlessSpec(
                cloud='aws',  # 클라우드 프로바이더
                region='us-east-1'  # 리전 (us-east-1 등)
            )
        )
        print(f"✓ 인덱스 생성 완료")
    else:
        print(f"✓ 기존 인덱스 사용: {PINECONE_INDEX_NAME}")
    
    return pc


def create_or_load_vectorstore(documents=None, force_recreate=False):
    """
    Pinecone 벡터스토어 생성 또는 로드
    
    작동 방식:
    1. force_recreate=False & 벡터 존재 → 기존 벡터 로드 (비용 절감)
    2. force_recreate=True → 기존 벡터 삭제 후 새로 생성
    3. 벡터 없음 → 새로 생성
    
    프로세스:
    - 문서 텍스트 → OpenAI Embeddings (1536차원 벡터)
    - 벡터 → Pinecone 저장
    - 메타데이터(타입, 소스, 경로) 포함
    
    Args:
        documents (list): 임베딩할 문서 (새로 생성 시 필요)
        force_recreate (bool): True면 기존 데이터 삭제하고 새로 생성
        
    Returns:
        PineconeVectorStore: Pinecone 벡터스토어 객체
    """
    # OpenAI Embeddings 모델 설정
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",  # 임베딩 모델
        dimensions=1536  # 벡터 차원
    )
    
    # Pinecone 인덱스 연결
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # ========== 인덱스 상태 확인 ==========
    stats = index.describe_index_stats()  # 인덱스 통계
    vector_count = stats.get('total_vector_count', 0)  # 저장된 벡터 개수
    
    # ========== 기존 데이터가 있고 재생성 플래그가 없으면 로드 ==========
    if vector_count > 0 and not force_recreate:
        print(f"✓ 기존 벡터스토어 로드 (벡터 수: {vector_count})")
        vectorstore = PineconeVectorStore(
            index=index,  # Pinecone 인덱스
            embedding=embeddings,  # 임베딩 모델
            text_key="text"  # 텍스트 필드 키
        )
    else:
        # ========== 데이터가 없거나 재생성 요청 시 ==========
        if force_recreate and vector_count > 0:
            print(f"기존 데이터 삭제 중... (벡터 수: {vector_count})")
            index.delete(delete_all=True)  # 모든 벡터 삭제
        
        if documents is None:
            raise ValueError("새로 생성하려면 documents 파라미터가 필요합니다")
        
        # 임베딩 생성 및 Pinecone 업로드
        print(f"임베딩 생성 및 Pinecone 업로드 중... ({len(documents)}개 문서)")
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,  # 문서 리스트
            embedding=embeddings,  # 임베딩 모델
            index_name=PINECONE_INDEX_NAME  # 인덱스 이름
        )
        print(f"✓ {len(documents)}개 문서 임베딩 완료 및 Pinecone에 저장")
    
    return vectorstore


def create_multimodal_rag_chain(vectorstore):
    """멀티모달 RAG 체인 생성"""
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini",
        temperature=0
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # 더 많은 문서 검색
    )
    
    template = """당신은 레스토랑 정보를 제공하는 도우미입니다.
다양한 형식의 문서(텍스트, PDF, 이미지)에서 정보를 가져왔습니다.
각 문서의 출처 타입을 고려하여 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        formatted = []
        for doc in docs:
            doc_type = doc.metadata.get('type', 'unknown')
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"[{doc_type.upper()} - {source}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✓ 멀티모달 RAG 체인 생성 완료")
    
    return rag_chain, retriever


def search_and_answer(rag_chain, retriever, query):
    """질문에 대한 답변 생성 (문서 타입 표시)"""
    print(f"\n{'='*60}")
    print(f"질문: {query}")
    print(f"{'='*60}")
    
    answer = rag_chain.invoke(query)
    print(f"\n답변:\n{answer}")
    
    source_docs = retriever.invoke(query)
    print(f"\n{'='*60}")
    print("참조 문서 (멀티모달):")
    for i, doc in enumerate(source_docs, 1):
        doc_type = doc.metadata.get('type', 'unknown')
        source = doc.metadata.get('source', 'Unknown')
        icon = {'text': '📄', 'pdf': '📑', 'image': '🖼️'}.get(doc_type, '📎')
        
        print(f"\n[{i}] {icon} {doc_type.upper()} - {source}")
        print(f"    내용: {doc.page_content[:150]}...")
    
    print(f"{'='*60}\n")
    
    return answer


def analyze_document_types(documents):
    """문서 타입별 통계"""
    print("\n" + "="*60)
    print("문서 타입 분석")
    print("="*60 + "\n")
    
    types = {}
    for doc in documents:
        doc_type = doc.metadata.get('type', 'unknown')
        types[doc_type] = types.get(doc_type, 0) + 1
    
    for doc_type, count in types.items():
        icon = {'text': '📄', 'pdf': '📑', 'image': '🖼️'}.get(doc_type, '📎')
        print(f"{icon} {doc_type.upper()}: {count}개")
    
    print()


# =================================================================
# 메인 실행 함수
# =================================================================

def main():
    """
    메인 실행 함수
    
    실행 단계:
    1. 멀티모달 문서 로드 (텍스트, 이미지, PDF)
    2. 문서 분할 (chunk 단위)
    3. Pinecone 초기화
    4. 벡터스토어 생성/로드
    5. RAG 체인 생성
    6. 예제 질문 실행
    7. 대화형 모드 진입
    """
    print("\n" + "="*60)
    print("멀티모달 RAG 예제")
    print("="*60 + "\n")
    
    # ========== 1단계: 멀티모달 문서 로드 ==========
    print("1단계: 멀티모달 문서 로드")
    data_directory = os.path.join(os.path.dirname(__file__), 'data')  # data 폴더 경로
    
    loader = MultiModalDocumentLoader(data_directory)  # 로더 생성
    documents = loader.load_all()  # 모든 형식 파일 로드
    
    # 문서가 없으면 종료
    if not documents:
        print("❌ 로드된 문서가 없습니다. data 폴더를 확인하세요.")
        return
    
    # 문서 타입별 통계 표시
    analyze_document_types(documents)
    
    # ========== 2단계: 문서 분할 ==========
    print("\n2단계: 문서 분할")
    splits = split_documents(documents)  # 문서를 청크로 분할
    
    # ========== 3단계: Pinecone 초기화 ==========
    print("\n3단계: Pinecone 초기화")
    initialize_pinecone()  # Pinecone 인덱스 생성 또는 확인
    
    # ========== 4단계: 벡터스토어 생성/로드 ==========
    print("\n4단계: 벡터스토어 로드/생성")
    # 주의: force_recreate=True로 설정하면 기존 데이터 삭제 후 재생성 
    # (Vision API 비용 발생)
    # 이후 실행 시에는 force_recreate=False로 변경 권장
    vectorstore = create_or_load_vectorstore(documents=splits, force_recreate=False)
    
    # ========== 5단계: 멀티모달 RAG 체인 생성 ==========
    print("\n5단계: 멀티모달 RAG 체인 생성")
    rag_chain, retriever = create_multimodal_rag_chain(vectorstore)
    
    # ========== 6단계: 질문 예제 ==========
    print("\n6단계: 멀티모달 검색 예제")
    
    example_queries = [
        "메뉴에 대한 모든 정보를 알려주세요",
        "이미지에서 찾을 수 있는 정보가 있나요?",
        "PDF 문서에는 어떤 내용이 있나요?",
    ]
    
    for query in example_queries:
        search_and_answer(rag_chain, retriever, query)
    
    # ========== 7단계: 대화형 모드 ==========
    print("\n대화형 모드 (종료하려면 'quit' 또는 'exit' 입력)")
    print("="*60 + "\n")
    
    while True:
        user_query = input("질문을 입력하세요: ").strip()
        
        # 종료 명령어 확인
        if user_query.lower() in ['quit', 'exit', '종료']:
            print("\n프로그램을 종료합니다.")
            break
        
        # 빈 입력 무시
        if not user_query:
            continue
        
        # 질문 처리
        search_and_answer(rag_chain, retriever, user_query)


if __name__ == "__main__":
    main()