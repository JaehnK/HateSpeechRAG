import os
from abc import ABC, abstractmethod
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings


class BaseEmbedding(ABC):
    
    def __init__(self, model_name: Optional[str] = None):
        load_dotenv()        
        self.model_name = model_name
        self._embedding_model = None
    
    def _initialize_model(self) -> None:
        """
        모델 초기화용 메서드
        """
        pass
    
    @property
    def embedding_model(self):
        """
        지연 로딩으로 임베딩 모델 반환
        """
        if self._embedding_model is None:
            self._initialize_model()
        return self._embedding_model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서들을 임베딩으로 변환
        """
        return self.embedding_model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        쿼리를 임베딩으로 변환
        """
        return self.embedding_model.embed_query(text)


class OpenAIEmbeddingModel(BaseEmbedding):
    """
    OpenAI 임베딩 모델 구현체
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or "text-embedding-ada-002"
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 api_key 파라미터를 제공하세요.")
    
    def _initialize_model(self) -> None:
        """
        OpenAI 임베딩 모델 초기화
        """
        self._embedding_model = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.api_key
        )

class UpstageEmbeddingModel(BaseEmbedding):
    """
    Upstage 임베딩 모델 구현=
    """
    
    def __init__(self, model_name: str = "solar-embedding-1-large", api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.model_name = model_name or "solar-embedding-1-large"
        
        if not self.api_key:
            raise ValueError("Upstage API 키가 필요합니다. 환경변수 UPSTAGE_API_KEY를 설정하거나 api_key 파라미터를 제공하세요.")
    
    def _initialize_model(self) -> None:
        """
        Upstage 임베딩 모델 초기화
        """
        self._embedding_model = UpstageEmbeddings(
            model=self.model_name,
            upstage_api_key=self.api_key
        )


class EmbeddingModelFactory:
    """
    임베딩 모델 팩토리 클래스
    """
    
    @staticmethod
    def create_embedding_model(provider: str, model_name: Optional[str] = None, api_key: Optional[str] = None) -> BaseEmbedding:
        """
        임베딩 모델 생성 팩토리 메서드
        
        Args:
            provider: 'openai' 또는 'upstage'
            model_name: 사용할 모델명 (선택사항)
            api_key: API 키 (선택사항, 환경변수 우선)
        
        Returns:
            BaseEmbeddingModel: 임베딩 모델 인스턴스
        """
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAIEmbeddingModel(model_name=model_name, api_key=api_key)
        elif provider == "upstage":
            return UpstageEmbeddingModel(model_name=model_name, api_key=api_key)
        else:
            raise ValueError(f"지원하지 않는 프로바이더입니다: {provider}. 'openai' 또는 'upstage'를 사용하세요.")


if __name__ == "__main__":
    # OpenAI 임베딩 모델 사용
    openai_embedding = EmbeddingModelFactory.create_embedding_model("openai")
    
    # Upstage 임베딩 모델 사용
    upstage_embedding = EmbeddingModelFactory.create_embedding_model("upstage")
    
    # 직접 클래스 인스턴스화도 가능
    # openai_embedding = OpenAIEmbeddingModel()
    # upstage_embedding = UpstageEmbeddingModel()
    
    # 임베딩 생성 예시
    texts = ["안녕하세요", "호이호이!"]
    text = "안녕하세용"
    
    try:
        # embeddings = openai_embedding.embed_documents(texts)
        embeddings = upstage_embedding.embed_documents(texts)
        embeddings = upstage_embedding.embed_query(text)
        print(f"임베딩 차원: {len(embeddings)}")
        print("임베딩 생성 성공!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
    

