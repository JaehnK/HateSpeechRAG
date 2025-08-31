import json
import asyncio
import time
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

from langfuse import observe, get_client

from openai import RateLimitError as OpenAIRateLimitError, BadRequestError as OpenAIBadRequestError
from anthropic import RateLimitError as AnthropicRateLimitError, BadRequestError as AnthropicBadRequestError


RETRY_DELAY = 30

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
        print(f"OpenAI LLM 서비스가 '{self._llm.model_name}' 모델로 초기화되었습니다.")
    
    def _initialize_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            # temperature=0,
            **self.kwargs
        )
    
    @observe(name="openai_llm_invoke", as_type="generation")    
    def invoke(self, messages:str, max_retries:int=3, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                result = self._llm.invoke(messages, **kwargs)
                print(result)
                return result
            
            except OpenAIRateLimitError as e:
                if attempt < max_retries:
                    wait_time = RETRY_DELAY * (2 ** attempt)  # 지수 백오프
                    print(f"Rate limit 에러 발생. {wait_time}초 후 재시도... (시도 {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    print("Rate limit 에러: 최대 재시도 횟수를 초과했습니다.")
                    raise
            
            except OpenAIBadRequestError as e:
                print(f"잘못된 요청 에러: {str(e)}\nInput: {messages}")
                raise
            
            except Exception as e:
                raise
        
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
    
    @observe(name="openai_llm_invoke", as_type="generation")    
    def invoke(self, messages: str, max_retries: int = 3, **kwargs):
        """Anthropic LLM 호출 (재시도 및 오류 처리 포함)"""
        for attempt in range(max_retries + 1):
            try:
                return self.llm.invoke(messages, **kwargs)
            except AnthropicRateLimitError as e:
                if attempt < max_retries:
                    wait_time = RETRY_DELAY * (2 ** attempt)  # 지수 백오프
                    print(f"Rate limit 에러 발생. {wait_time}초 후 재시도... (시도 {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    print("Rate limit 에러: 최대 재시도 횟수를 초과했습니다.")
                    raise
            except AnthropicBadRequestError as e:
                print(f"잘못된 요청 에러: {str(e)}\nInput: {messages}")
                raise
            except Exception as e:
                raise
        
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
        
        print(f"LLM Service Provider: {provider}")
        
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        if provider == LLMProvider.OPENAI:
            return OpenAILLMService(model_name or "gpt-5", **kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicLLMService(model_name or "claude-sonnet-4-20250514", **kwargs)  #claude-sonnet-4-20250514, claude-3-haiku-20240307"
        elif provider == LLMProvider.GOOGLE:
            return GoogleLLMService(model_name, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {provider}")
