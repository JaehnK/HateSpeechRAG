import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMProvider(Enum):
    """
    지원하는 LLM 제공자
    """
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class BaseLLMService(ABC):
    """LLM 서비스 추상 Base 클래스"""
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.llm = None
        self.kwargs = kwargs
    
    @abstractmethod
    def _initialize_llm(self) -> Any:
        """LLM 초기화"""
        pass
    
    @property
    def model(self):
        """지연 로딩으로 LLM 반환"""
        if self.llm is None:
            self.llm = self._initialize_llm()
        return self.llm


class OpenAILLMService(BaseLLMService):
    """OpenAI LLM 서비스"""
    
    def __init__(self, model_name: str = "gpt-5", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다.")
        
        self._llm = self._initialize_llm()
    
    def _initialize_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            # temperature=0,
            **self.kwargs
        )
        
class AnthropicLLMService(BaseLLMService):
    """Anthropic Claude LLM 서비스"""
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API 키가 필요합니다.")
    
    def _initialize_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            model=self.model_name,
            anthropic_api_key=self.api_key,
            temperature=0.0,
            **self.kwargs
        )
        
class GoogleLLMService(BaseLLMService):
    """Google Gemini LLM 서비스"""
    
    def __init__(self, model_name: str = "gemini-pro", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API 키가 필요합니다.")
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.0,
            **self.kwargs
        )


class LLMServiceFactory:
    """LLM 서비스 팩토리"""
    
    @staticmethod
    def create_llm_service(
        provider: Union[str, LLMProvider], 
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLMService:
        """LLM 서비스 생성"""
        load_dotenv()
        
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        if provider == LLMProvider.OPENAI:
            return OpenAILLMService(model_name or "gpt-5", **kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicLLMService(model_name, **kwargs)
        elif provider == LLMProvider.GOOGLE:
            return GoogleLLMService(model_name, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {provider}")
