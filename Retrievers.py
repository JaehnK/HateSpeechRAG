from abc import ABC, abstractmethod
from typing import List, Any
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores import VectorStore


class BaseRetriever(ABC):
    """
    리트리버를 위한 추상 베이스 클래스
    """
    
    def __init__(self, vector_store: VectorStore, k: int = 5):
        self.vector_store = vector_store
        self.k = k
        self._retriever = None
        self._type = None
    
    @abstractmethod
    def _build_retriever(self) -> Any:
        """
        실제 리트리버를 구축하는 메서드
        """
        pass
    
    @property
    def retriever(self):
        """
        지연 로딩으로 리트리버 반환
        """
        if self._retriever is None:
            self._retriever = self._build_retriever()
        return self._retriever
    
    def retrieve(self, query: str) -> List[Document]:
        """
        문서 검색
        """
        # return self.retriever.get_relevant_documents(query)
        return self.retriever.invoke(query)
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        점수와 함께 문서 검색 (지원하는 경우)
        """
        if hasattr(self.retriever, 'similarity_search_with_score'):
            print(f"{self._type} Retriever: Retreive with scores")
            return self.retriever.similarity_search_with_score(query, k=self.k)
        else:
            # 점수가 없는 경우 기본 검색 결과 반환
            print(f"{self._type} Retriever: Invoke without Score")
            docs = self.retrieve(query)
            return [(doc, 0.0) for doc in docs]


class VectorStoreRetriever(BaseRetriever):
    """
    기본 벡터스토어 리트리버
    """
    
    def __init__(self, vector_store: VectorStore, k: int = 5, search_type: str = "similarity"):
        super().__init__(vector_store, k)
        self.search_type = search_type
        self._type = "VectorStore"
    
    def _build_retriever(self) -> Any:
        """
        기본 벡터스토어 리트리버 구축
        """
        return self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k}
        )
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        벡터스토어에서 직접 점수와 함께 검색
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=self.k)
            print(f"{self._type} Retriever: Retrieved with scores")
            return results
        except Exception as e:
            print(f"{self._type} Retriever: Error getting scores - {e}")
            docs = self.retrieve(query)
            return [(doc, 0.0) for doc in docs]

class MMRRetriever(BaseRetriever):
    """
    MMR (Maximal Marginal Relevance) 리트리버
    """
    
    def __init__(self, vector_store: VectorStore, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.7):
        super().__init__(vector_store, k)
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self._type = "mmr"
    
    def _build_retriever(self) -> Any:
        """
        MMR 리트리버 구축
        """
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.k,
                "fetch_k": self.fetch_k,
                "lambda_mult": self.lambda_mult
            }
        )

class EnsembleRetrieverWrapper(BaseRetriever):
    """
    앙상블 리트리버 (BM25 + Dense Vector)
    """
    
    def __init__(self, vector_store: VectorStore, documents: List[Document], k: int = 5, weights: List[float] = None):
        super().__init__(vector_store, k)
        self.documents = documents
        self.weights = weights or [0.7, 0.3]  # Dense 70%, BM25 30%
        self._type = "ensemble"
    
    def _build_retriever(self) -> Any:
        """
        앙상블 리트리버 구축
        """
        # Dense Vector 리트리버
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
        
        # BM25 리트리버
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.k
        
        # 앙상블 리트리버
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=self.weights
        )
        
        return ensemble_retriever

class SelfQueryRetrieverWrapper(BaseRetriever):
    """
    Self Query 리트리버 (메타데이터 기반 필터링)
    """
    
    def __init__(self, vector_store: VectorStore, llm: Any, k: int = 5):
        super().__init__(vector_store, k)
        self.llm = llm
        self._type = "self_query"
    
    def _build_retriever(self) -> Any:
        """
        Self Query 리트리버 구축
        """
        # 간소화된 혐오표현 메타데이터 필드 정의
        metadata_field_info = [
            AttributeInfo(name="성별", description="성별 관련 혐오 표현 (여성, 남성, 성소수자)", type="integer"),
            AttributeInfo(name="정체성", description="정체성 관련 혐오 표현 (인종, 국적, 지역, 종교)", type="integer"),
            AttributeInfo(name="연령", description="연령 관련 혐오 표현", type="integer"),
            AttributeInfo(name="기타", description="기타 혐오 표현 (기타, 개인지칭)", type="integer"),
            AttributeInfo(name="욕설", description="욕설 표현", type="integer"),
            AttributeInfo(name="혐오없음", description="혐오 표현이 없는 깨끗한 텍스트", type="integer"),
        ]
        
        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vector_store,
            document_contents="혐오 표현 분류를 위한 텍스트 데이터",
            metadata_field_info=metadata_field_info,
            verbose=True
        )


class RetrieverFactory:
    """
    리트리버 팩토리 클래스
    """
    
    @staticmethod
    def create_retriever(
        retriever_type: str,
        vector_store: VectorStore,
        k: int = 5,
        **kwargs
    ) -> BaseRetriever:
        """
        리트리버 생성 팩토리 메서드
        
        Args:
            retriever_type: 'basic', 'mmr', 'ensemble', 'self_query'
            vector_store: 벡터스토어 인스턴스
            k: 검색할 문서 수
            **kwargs: 각 리트리버별 추가 파라미터
        
        Returns:
            BaseRetriever: 리트리버 인스턴스
        """
        retriever_type = retriever_type.lower()
        
        if retriever_type == "basic":
            search_type = kwargs.get("search_type", "similarity")
            return VectorStoreRetriever(vector_store, k, search_type)
        
        elif retriever_type == "mmr":
            fetch_k = kwargs.get("fetch_k", 20)
            lambda_mult = kwargs.get("lambda_mult", 0.7)
            return MMRRetriever(vector_store, k, fetch_k, lambda_mult)
        
        elif retriever_type == "ensemble":
            documents = kwargs.get("documents")
            if not documents:
                raise ValueError("앙상블 리트리버를 위해서는 documents 파라미터가 필요합니다.")
            weights = kwargs.get("weights", [0.7, 0.3])
            return EnsembleRetrieverWrapper(vector_store, documents, k, weights)
        
        elif retriever_type == "self_query":
            llm = kwargs.get("llm")
            if not llm:
                raise ValueError("Self Query 리트리버를 위해서는 llm 파라미터가 필요합니다.")
            return SelfQueryRetrieverWrapper(vector_store, llm, k)
        
        else:
            raise ValueError(f"지원하지 않는 리트리버 타입입니다: {retriever_type}. "
                            f"'basic', 'mmr', 'ensemble', 'self_query' 중 하나를 사용하세요.")


if __name__ == "__main__":
    # 예시 사용법
    
    # 기본 리트리버
    basic_retriever = RetrieverFactory.create_retriever("basic", vector_store, k=5)
    
    # MMR 리트리버
    mmr_retriever = RetrieverFactory.create_retriever(
        "mmr", vector_store, k=5, fetch_k=20, lambda_mult=0.7
    )
    
    # 앙상블 리트리버
    ensemble_retriever = RetrieverFactory.create_retriever(
        "ensemble", vector_store, k=5, documents=documents, weights=[0.7, 0.3]
    )
    
    # Self Query 리트리버
    self_query_retriever = RetrieverFactory.create_retriever(
        "self_query", vector_store, k=5, llm=llm
    )
    
    # 검색 실행
    results = basic_retriever.retrieve("혐오 표현을 찾아주세요")
    print(f"검색 결과: {len(results)}개 문서")