import os
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from Embeddings import BaseEmbedding, EmbeddingModelFactory
from Retrievers import BaseRetriever, RetrieverFactory


class VectorStoreDao:
    """
    혐오표현 분류를 위한 벡터스토어 DAO 클래스
    """
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        embedding_model: Optional[BaseEmbedding] = None,
        collection_name: str = "hate_speech_collection"
    ):
        """
        VectorStoreDao 초기화
        
        Args:
            persist_directory: 벡터스토어 저장 경로
            embedding_model: 임베딩 모델 인스턴스
            collection_name: 컬렉션 이름
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model or EmbeddingModelFactory.create_embedding_model("openai")
        self.vector_store = None
        self.retriever = None
        self.documents = []  # 앙상블 리트리버용 문서 저장
        
        # 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents: Optional[List[Document]] = None) -> None:
        """
        벡터스토어 생성 또는 기존 로드
        
        Args:
            documents: 초기 문서 리스트 (새로 생성 시)
        """
        if documents:
            # 새로운 벡터스토어 생성
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            self.documents = documents.copy()
            print(f"새로운 벡터스토어 생성 완료: {len(documents)}개 문서")
        else:
            # 기존 벡터스토어 로드
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model.embedding_model,
                    collection_name=self.collection_name
                )
                print("기존 벡터스토어 로드 완료")
            except Exception as e:
                print(f"기존 벡터스토어 로드 실패: {e}")
                print("새로운 빈 벡터스토어 생성")
                self.vector_store = Chroma(
                    embedding_function=self.embedding_model.embedding_model,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        벡터스토어에 문서 추가
        
        Args:
            documents: 추가할 문서 리스트
        """
        if not self.vector_store:
            self.create_vector_store(documents)
            return
        
        self.vector_store.add_documents(documents)
        self.documents.extend(documents)
        print(f"{len(documents)}개 문서 추가 완료")
    
    def retrieve_documents(
        self, 
        query: str, 
        k: int = 5,
        retriever_type: str = "basic",
        **retriever_kwargs
    ) -> List[Document]:
        """
        문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            retriever_type: 리트리버 타입 ('basic', 'mmr', 'ensemble', 'self_query')
            **retriever_kwargs: 리트리버별 추가 파라미터
        
        Returns:
            검색된 문서 리스트
        """
        if not self.vector_store:
            raise ValueError("벡터스토어가 초기화되지 않았습니다. create_vector_store()를 먼저 호출하세요.")
        
        # 리트리버 설정
        self.set_retriever(retriever_type, k, **retriever_kwargs)
        
        # 검색 실행
        return self.retriever.retrieve(query)
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = 5,
        retriever_type: str = "basic",
        **retriever_kwargs
    ) -> List[tuple]:
        """
        점수와 함께 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            retriever_type: 리트리버 타입
            **retriever_kwargs: 리트리버별 추가 파라미터
        
        Returns:
            (문서, 점수) 튜플 리스트
        """
        if not self.vector_store:
            raise ValueError("벡터스토어가 초기화되지 않았습니다.")
        
        self.set_retriever(retriever_type, k, **retriever_kwargs)
        return self.retriever.retrieve_with_scores(query)
    
    def search_by_metadata(
        self, 
        query: str, 
        metadata_filter: Dict[str, Any],
        k: int = 5
    ) -> List[Document]:
        """
        메타데이터 필터를 사용한 검색
        
        Args:
            query: 검색 쿼리
            metadata_filter: 메타데이터 필터 (예: {"성별": 1, "욕설": 0})
            k: 반환할 문서 수
        
        Returns:
            필터링된 검색 결과
        """
        if not self.vector_store:
            raise ValueError("벡터스토어가 초기화되지 않았습니다.")
        
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter
        )
    
    def get_documents_by_category(self, category: str, value: int = 1, k: int = 10) -> List[Document]:
        """
        특정 카테고리의 문서 조회
        
        Args:
            category: 카테고리명 ('성별', '정체성', '연령', '기타', '욕설', '혐오없음')
            value: 카테고리 값 (0 또는 1)
            k: 반환할 문서 수
        
        Returns:
            해당 카테고리의 문서 리스트
        """
        if not self.vector_store:
            raise ValueError("벡터스토어가 초기화되지 않았습니다.")
        
        # 빈 쿼리로 검색하되 메타데이터 필터 적용
        return self.vector_store.similarity_search(
            query="",
            k=k,
            filter={category: value}
        )
    
    def set_retriever(
        self, 
        retriever_type: str, 
        k: int = 5, 
        **kwargs
    ) -> None:
        """
        리트리버 설정
        
        Args:
            retriever_type: 리트리버 타입
            k: 검색할 문서 수
            **kwargs: 리트리버별 추가 파라미터
        """
        if retriever_type == "ensemble" and "documents" not in kwargs:
            kwargs["documents"] = self.documents
        
        self.retriever = RetrieverFactory.create_retriever(
            retriever_type=retriever_type,
            vector_store=self.vector_store,
            k=k,
            **kwargs
        )
        print(f"리트리버 설정 완료: {retriever_type}")
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        문서 삭제
        
        Args:
            document_ids: 삭제할 문서 ID 리스트
        """
        if not self.vector_store:
            raise ValueError("벡터스토어가 초기화되지 않았습니다.")
        
        self.vector_store.delete(ids=document_ids)
        print(f"{len(document_ids)}개 문서 삭제 완료")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 조회
        
        Returns:
            컬렉션 정보 딕셔너리
        """
        if not self.vector_store:
            return {"status": "벡터스토어가 초기화되지 않음"}
        
        try:
            collection = self.vector_store._collection
            return {
                "collection_name": collection.name,
                "document_count": collection.count(),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": f"정보 조회 실패: {e}"}
    
    def persist(self) -> None:
        """
        벡터스토어 영구 저장
        """
        if self.vector_store:
            self.vector_store.persist()
            print("벡터스토어 저장 완료")
    
    def clear_collection(self) -> None:
        """
        컬렉션 초기화 (모든 문서 삭제)
        """
        if not self.vector_store:
            raise ValueError("벡터스토어가 초기화되지 않았습니다.")
        
        try:
            self.vector_store.delete_collection()
            self.documents.clear()
            print("컬렉션 초기화 완료")
        except Exception as e:
            print(f"컬렉션 초기화 실패: {e}")


# 사용 예시
if __name__ == "__main__":
    # DAO 인스턴스 생성
    dao = VectorStoreDao(
        persist_directory="./hate_speech_db",
        embedding_model=EmbeddingModelFactory.create_embedding_model("openai")
    )
    
    # 예시 문서 생성
    # sample_documents = [
    #     Document(
    #         page_content="일안하는 시간은 쉬고싶어서 그런게 아닐까",
    #         metadata={
    #             "성별": 0, "정체성": 0, "연령": 0, 
    #             "기타": 0, "욕설": 0, "혐오없음": 1
    #         }
    #     )
    # ]
    
    # # 벡터스토어 생성
    # dao.create_vector_store(sample_documents)
    dao.create_vector_store()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    
    # 검색 테스트
    results = dao.retrieve_documents("일하기 싫어", k=3, retriever_type="basic")
    print(f"검색 결과: {len(results)}개")
    print(results[0].metadata)
    
    results = dao.retrieve_documents("일하기 싫어", k=3, retriever_type="mmr")
    print(f"검색 결과: {len(results)}개")
    print(results[0].metadata)
    
    results = dao.retrieve_documents("일하기 싫어", k=3, retriever_type="self_query", llm=llm)
    print(f"검색 결과: {len(results)}개")
    print(results[0].metadata)
    
    # 컬렉션 정보 확인
    info = dao.get_collection_info()
    print(f"컬렉션 정보: {info}")