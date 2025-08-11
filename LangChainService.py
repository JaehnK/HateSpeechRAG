from typing import List, Dict, Any, Tuple, Optional
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableParallel
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from langchain_teddynote import logging
from pydantic import BaseModel, Field
from operator import itemgetter
import json
import time

from VectorStoreDao import VectorStoreDao
from Embeddings import EmbeddingModelFactory
from LLMServices import LLMServiceFactory

from pprint import pprint
from dotenv import load_dotenv

class VectorStoreRetriever(Runnable):
    """VectorStoreDao를 Runnable로 래핑"""
    
    def __init__(self, dao: 'VectorStoreDao'):
        self.dao = dao
    
    def invoke(self, input_query: str, config=None) -> List[Tuple[Document, float]]:
        """쿼리를 받아 유사 문서들을 반환"""
        return self.dao.retrieve_with_scores(input_query)
    
    def batch(self, inputs: List[str], config=None) -> List[List[Tuple[Document, float]]]:
        """배치 처리"""
        # DAO에 캐시된 리트리버가 있으므로, 배치 호출도 효율적으로 처리될 수 있습니다.
        # LangChain의 batch 기능을 최대한 활용하기 위해 DAO에도 batch 인터페이스를 추가하는 것을 고려할 수 있습니다.
        return [self.invoke(query, config) for query in inputs]

class SimilarCasesFormatter(Runnable):
    """검색된 유사 사례들을 프롬프트용으로 포맷팅"""
    
    def invoke(self, search_results: List[Tuple[Document, float]], config=None) -> str:
        """검색 결과를 포맷된 문자열로 변환"""
        if not search_results:
            return "관련 유사 사례가 없습니다."
        
        formatted_cases = []
        for i, (doc, score) in enumerate(search_results, 1):
            # 메타데이터에서 혐오표현 카테고리 추출
            categories = self._extract_categories(doc.metadata)
            
            case_text = f"""
사례 {i}:
- 텍스트: {doc.page_content[:100]}...
- 분류: {categories}
- 유사도: {score:.3f}
"""
            formatted_cases.append(case_text)
        
        return "\n".join(formatted_cases)
    
    def _extract_categories(self, metadata: Dict[str, Any]) -> List[str]:
        """메타데이터에서 활성화된 카테고리 추출"""
        category_map = {
            '성별': metadata.get('성별', 0),
            '연령': metadata.get('연령', 0),
            '정체성': metadata.get('정체성', 0),
            '욕설': metadata.get('욕설', 0),
            '기타': metadata.get('기타', 0),
            '혐오없음': metadata.get('혐오없음', 0)
        }
        
        active_categories = [cat for cat, value in category_map.items() if value == 1]
        return active_categories if active_categories else ['미분류']

# Pydantic 모델 정의
class HateSpeechClassification(BaseModel):
    """혐오표현 분류 결과"""
    
    input_text: str = Field(description="분석 대상 텍스트")
    
    is_hate_speech: bool = Field(description="혐오표현 여부")
    
    categories: List[str] = Field(
        description="해당하는 혐오표현 카테고리들 (성별, 연령, 정체성, 욕설, 기타, 혐오없음)",
        default_factory=list
    )
    
    evidence_strength: float = Field(
        description="증거 강도 (검색 결과 기반, 0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    reasoning: str = Field(description="분류 근거 및 설명")
    
    similar_cases_used: List[str] = Field(
        description="참조한 유사 사례들 (번호)",
        default_factory=list
    )
    
    target_group: Optional[str] = Field(
        description="혐오 대상 집단 (해당하는 경우)",
        default=None
    )
    
    hate_type: Optional[str] = Field(
        description="혐오 유형 (모욕/비하/멸시/위협/선동 등)",
        default=None
    )

class HateSpeechRAGChain:
    """혐오표현 분류를 위한 RAG 체인"""
    
    def __init__(self, dao: 'VectorStoreDao', llm=None):
        self.dao = dao
        self.llm = llm
        
        # Runnable 컴포넌트들 초기화
        self.retriever = VectorStoreRetriever(dao)  # 파라미터 제거
        self.formatter = SimilarCasesFormatter()
        
        # Pydantic Parser 초기화
        self.output_parser = PydanticOutputParser(pydantic_object=HateSpeechClassification)
        
        # 프롬프트 템플릿 (JSON 출력 요청)
        self.prompt_template = ChatPromptTemplate.from_template("""
당신은 혐오표현 분류 전문가입니다.

=== 혐오표현 정의 ===
성별, 장애, 종교, 나이, 출신지역, 인종, 성적지향 등 특정 속성을 이유로  어떤 개인·집단에게 모욕, 비하, 멸시, 위협 또는 차별·폭력의 선전과 선동을 함으로써 차별을 정당화·조장·강화하는 효과를 갖는 표현

=== 분류 카테고리 (복수 선택 가능) ===
- 성별: 성별 기반 차별 및 비하 (예: "김치녀", "맘충")
- 연령: 연령대 기반 차별 및 비하 (예: "꼰대", "애새끼")
- 정체성: 출신지역, 인종, 성적지향, 종교 등 (예: "짱깨", "일베충")
- 욕설: 직접적 욕설이 포함된 혐오표현 (예: "시발", "새끼")
- 기타: 위 카테고리에 해당하지 않는 기타 혐오표현
- 혐오없음: 혐오표현이 아닌 경우

⚠️ 중요: 하나의 텍스트가 여러 카테고리에 동시에 해당할 수 있습니다.
예시: "김치녀들 시발 다 똑같아" → ["성별", "욕설"]

=== 참고할 유사 사례들 ===
{similar_cases}

=== 분석 대상 텍스트 ===
"{input_text}"

=== 분석 지침 ===
1. 검색된 유사 사례들의 분류와 일관성을 평가하세요
2. 증거 강도는 유사 사례들의 일관성과 유사도를 종합하여 계산하세요
3. 혐오표현 정의에 엄격히 따라 판단하세요
4. 해당하는 모든 카테고리를 포함하세요 (단일 또는 복수)
5. 복합 혐오표현의 경우 각각의 혐오 요소를 모두 식별하세요

=== 복합 혐오표현 예시 ===
- "김치녀들 시발 다 똑같아" → 성별 비하 + 욕설 → ["성별", "욕설"]
- "꼰대 일베충들" → 연령 비하 + 정체성 혐오 → ["연령", "정체성"]  
- "그 똥양인 새끼들 좆같네" → 정체성 혐오 + 욕설 → ["정체성", "욕설"]

{format_instructions}
""")
        
        # RAG 체인 구성
        self.rag_chain = self._build_chain()
    
    def _build_chain(self):
        """RAG 체인 구성"""
        return (
            RunnableParallel(
                input_text=RunnablePassthrough(),
                similar_cases=RunnablePassthrough() | self.retriever | self.formatter,
                format_instructions=lambda _: self.output_parser.get_format_instructions()
            )
            | self.prompt_template
        )
    
    def get_prompt(self, text: str) -> str:
        """텍스트에 대한 완성된 프롬프트 생성"""
        prompt_value = self.rag_chain.invoke(text)
        return prompt_value.to_string()
    
    def classify(self, text: str) -> HateSpeechClassification:
        """혐오표현 분류 수행 (구조화된 결과 반환)"""
        if self.llm is None:
            raise ValueError("LLM이 설정되지 않았습니다. 분류를 위해서는 LLM이 필요합니다.")

        print("\n--- Detailed Profiling ---")

        # 1. (Isolate) Query Embedding 시간 측정
        start_time = time.time()
        # DAO를 통해 임베딩 모델에 접근
        _ = self.dao.embedding_model.embedding_model.embed_query(text)
        embedding_time = time.time() - start_time
        print(f"1. (Isolated) Query Embedding: {embedding_time:.4f} 초")

        # 2. Retrieval 시간 측정 (내부적으로 임베딩 포함)
        start_time = time.time()
        retrieved_docs = self.retriever.invoke(text)
        retrieval_time = time.time() - start_time
        print(f"2. Full Retrieval (Embedding + Search): {retrieval_time:.4f} 초")

        # 3. Formatting 시간 측정
        start_time = time.time()
        formatted_cases = self.formatter.invoke(retrieved_docs)
        formatting_time = time.time() - start_time
        print(f"3. Formatting: {formatting_time:.4f} 초")

        # 4. Prompt 생성 시간 측정
        start_time = time.time()
        prompt_input = {
            "input_text": text,
            "similar_cases": formatted_cases,
            "format_instructions": self.output_parser.get_format_instructions()
        }
        prompt = self.prompt_template.invoke(prompt_input)
        prompt_time = time.time() - start_time
        print(f"4. Prompt Generation: {prompt_time:.4f} 초")

        # 5. LLM 호출 시간 측정
        start_time = time.time()
        llm_output = self.llm.invoke(prompt)
        llm_time = time.time() - start_time
        print(f"5. LLM Call: {llm_time:.4f} 초")

        # 6. Output Parsing 시간 측정
        start_time = time.time()
        parsed_output = self.output_parser.invoke(llm_output)
        parsing_time = time.time() - start_time
        print(f"6. Output Parsing: {parsing_time:.4f} 초")

        # 참고: 총 시간은 임베딩이 두 번 측정되므로 실제보다 약간 부풀려져 표시됩니다.
        total_time = embedding_time + retrieval_time + formatting_time + prompt_time + llm_time + parsing_time
        print(f"Total Execution Time (approx.): {total_time:.4f} 초")
        print("-------------------------")

        return parsed_output
    
    def classify_batch(self, texts: List[str]) -> List[HateSpeechClassification]:
        """배치 분류"""
        if self.llm is None:
            raise ValueError("LLM이 설정되지 않았습니다.")
        
        full_chain = self.rag_chain | self.llm | self.output_parser
        return full_chain.batch(texts)

# 테스트 및 사용 예시
def test_rag_pipeline_with_json(dao: 'VectorStoreDao', test_queries: List[str] = None) -> None:
    """JSON 출력을 포함한 RAG 파이프라인 테스트"""
    
    if test_queries is None:
        test_queries = [
            "여성을 비하하는 말",
            "욕설이 포함된 문장", 
            "깨끗한 일반 문장",
            "나이 관련 차별 표현"
        ]
    
    print("=== JSON 출력 RAG 파이프라인 테스트 ===")
    
    # RAG 체인 초기화
    rag_chain = HateSpeechRAGChain(dao)
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"쿼리: '{query}'")
        print(f"{'='*50}")
        
        try:
            # Pydantic 포맷 지침 확인
            format_instructions = rag_chain.output_parser.get_format_instructions()
            print(f"JSON 출력 형식:")
            print(format_instructions[:200] + "...")
            
            # 완성된 프롬프트 생성 (JSON 형식 지침 포함)
            full_prompt = rag_chain.get_prompt(query)
            print(f"\n프롬프트 끝부분 (JSON 형식 지침):")
            print("-" * 30)
            print(full_prompt[-300:])  # 마지막 300자만 출력
            print("-" * 30)
            
        except Exception as e:
            print(f"오류 발생: {e}")

def test_with_mock_llm(dao: 'VectorStoreDao') -> None:
    """Mock LLM으로 JSON 출력 테스트"""
    
    from langchain.schema.runnable import Runnable
    from langchain.schema import BaseMessage
    
    class MockLLM(Runnable):
        def invoke(self, input_data, config=None, **kwargs):
            # 입력이 메시지나 프롬프트인 경우 처리
            if hasattr(input_data, 'to_string'):
                prompt_text = input_data.to_string()
            elif isinstance(input_data, str):
                prompt_text = input_data
            else:
                prompt_text = str(input_data)
            
            # 프롬프트에서 실제 분석 대상 텍스트 추출
            import re
            # "분석 대상 텍스트 ===" 다음에 오는 텍스트를 찾음
            text_match = re.search(r'=== 분석 대상 텍스트 ===\s*"([^"]+)"', prompt_text)
            if text_match:
                actual_input = text_match.group(1)
            else:
                actual_input = "알 수 없는 텍스트"
            
            # 추출된 실제 입력에 따라 응답 생성 (복합 분류 포함)
            if "여성" in actual_input and "시발" in actual_input:
                return f"""{{"input_text": "{actual_input}", "is_hate_speech": true, "categories": ["성별", "욕설"], "evidence_strength": 0.95, "reasoning": "여성에 대한 비하 표현과 직접적 욕설이 동시에 포함된 복합 혐오표현", "similar_cases_used": ["사례1", "사례2", "사례3"], "target_group": "여성", "hate_type": "비하+욕설"}}"""
            elif "여성" in actual_input:
                return f"""{{"input_text": "{actual_input}", "is_hate_speech": true, "categories": ["성별"], "evidence_strength": 0.85, "reasoning": "검색된 유사 사례들이 모두 성별 기반 혐오표현으로 분류되어 있으며, 여성에 대한 비하적 표현임이 명확함", "similar_cases_used": ["사례1", "사례2", "사례3"], "target_group": "여성", "hate_type": "비하"}}"""
            elif "욕설" in actual_input:
                return f"""{{"input_text": "{actual_input}", "is_hate_speech": true, "categories": ["욕설"], "evidence_strength": 0.75, "reasoning": "직접적인 욕설이 포함된 표현으로 혐오표현에 해당함", "similar_cases_used": ["사례1", "사례3"], "target_group": null, "hate_type": "욕설"}}"""
            elif "꼰대" in actual_input and "일베" in actual_input:
                return f"""{{"input_text": "{actual_input}", "is_hate_speech": true, "categories": ["연령", "정체성"], "evidence_strength": 0.88, "reasoning": "연령 기반 비하('꼰대')와 특정 집단에 대한 정체성 혐오가 복합적으로 나타남", "similar_cases_used": ["사례1", "사례2"], "target_group": "중년층, 특정 온라인 집단", "hate_type": "비하+정체성혐오"}}"""
            elif "깨끗한" in actual_input or "일반" in actual_input:
                return f"""{{"input_text": "{actual_input}", "is_hate_speech": false, "categories": ["혐오없음"], "evidence_strength": 0.9, "reasoning": "검색된 사례 중 혐오없음으로 분류된 사례가 있고, 특정 집단에 대한 차별적 표현이 없음", "similar_cases_used": ["사례1"], "target_group": null, "hate_type": null}}"""
            elif "나이" in actual_input or "차별" in actual_input:
                return f"""{{"input_text": "{actual_input}", "is_hate_speech": true, "categories": ["연령"], "evidence_strength": 0.7, "reasoning": "연령 기반 차별 표현으로 판단됨", "similar_cases_used": ["사례1", "사례2"], "target_group": "특정 연령대", "hate_type": "차별"}}"""
            else:
                return f"""{{"input_text": "{actual_input}", "is_hate_speech": false, "categories": ["혐오없음"], "evidence_strength": 0.7, "reasoning": "특정 집단에 대한 차별적 표현이 발견되지 않음", "similar_cases_used": [], "target_group": null, "hate_type": null}}"""
        
        def batch(self, inputs, config=None, **kwargs):
            return [self.invoke(input_item, config, **kwargs) for input_item in inputs]
    
    print("=== Mock LLM JSON 출력 테스트 ===")
    
    mock_llm = MockLLM()
    rag_chain = HateSpeechRAGChain(dao, llm=mock_llm)
    
    test_cases = [
        "여성을 비하하는 말",
        "욕설이 포함된 문장", 
        "깨끗한 일반 문장",
        "김치녀들 시발 다 똑같아",  # 복합: 성별 + 욕설
        "꼰대 일베충들"  # 복합: 연령 + 정체성
    ]
    
    for test_text in test_cases:
        print(f"\n{'='*40}")
        print(f"테스트: '{test_text}'")
        print(f"{'='*40}")
        
        try:
            result = rag_chain.classify(test_text)
            print(f"분류 결과 타입: {type(result)}")
            print(f"혐오표현 여부: {result.is_hate_speech}")
            print(f"카테고리: {result.categories}")
            print(f"증거 강도: {result.evidence_strength}")
            print(f"근거: {result.reasoning}")
            print(f"대상 집단: {result.target_group}")
            print(f"혐오 유형: {result.hate_type}")
            
            print(f"\nJSON 출력:")
            print(result.model_dump_json(indent=2))
            
        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()

# 실제 사용 예시
def example_usage_with_real_llm():
    """실제 LLM과 함께 사용하는 예시"""
    
    # from langchain.llms import OpenAI
    # llm = OpenAI(temperature=0)
    # 
    # rag_chain = HateSpeechRAGChain(dao, llm=llm)
    # 
    # # 단일 분류
    # result: HateSpeechClassification = rag_chain.classify("김치녀들은 다 똑같아")
    # print(f"혐오표현 여부: {result.is_hate_speech}")
    # print(f"카테고리: {result.categories}")
    # print(f"증거 강도: {result.evidence_strength}")
    # print(f"근거: {result.reasoning}")
    # 
    # # 배치 분류
    # texts = ["여성 비하", "일반 문장", "나이 차별"]
    # results: List[HateSpeechClassification] = rag_chain.classify_batch(texts)
    # 
    # for result in results:
    #     print(f"{result.input_text} → {result.categories}")
    
    pass

# 개별 컴포넌트 테스트
def test_individual_components(dao: 'VectorStoreDao') -> None:
    """개별 Runnable 컴포넌트들 테스트"""
    
    print("=== 개별 컴포넌트 테스트 ===")
    
    # 1. 리트리버 테스트
    retriever = VectorStoreRetriever(dao, k=2)
    query = "여성을 비하하는 말"
    
    print(f"\n1. 리트리버 테스트 - 쿼리: '{query}'")
    search_results = retriever.invoke(query)
    for i, (doc, score) in enumerate(search_results):
        print(f"  결과 {i+1}: {doc.page_content[:50]}... (점수: {score:.3f})")
    
    # 2. 포맷터 테스트
    formatter = SimilarCasesFormatter()
    print(f"\n2. 포맷터 테스트:")
    formatted = formatter.invoke(search_results)
    print(formatted)
    
    # 3. 체인 조합 테스트
    print(f"\n3. 체인 조합 테스트:")
    combined_chain = retriever | formatter
    result = combined_chain.invoke(query)
    print(result)

# 배치 처리 테스트
def test_batch_processing(dao: 'VectorStoreDao') -> None:
    """배치 처리 테스트"""
    
    print("=== 배치 처리 테스트 ===")
    
    queries = ["여성 비하", "나이 차별", "일반 문장"]
    rag_chain = HateSpeechRAGChain(dao)
    
    # 배치로 프롬프트 생성
    for query in queries:
        print(f"\n쿼리: {query}")
        prompt = rag_chain.get_prompt(query)
        print(f"프롬프트 길이: {len(prompt)} 문자")

if __name__ == "__main__":
    load_dotenv()
    logging.langsmith("HateSpeechTest")
    dao = VectorStoreDao(
        persist_directory="./hate_speech_vectorstore",
        embedding_model = EmbeddingModelFactory.create_embedding_model('upstage'),
        collection_name="hate_speech_collection"
    )
    dao.create_vector_store()
    
    # 리트리버를 명시적으로 초기화 (k=3, basic retriever)
    dao.initialize_retriever(retriever_type="basic", k=3)
    
    llm_openai = LLMServiceFactory.create_llm_service("openai")
    
    rag_chain = HateSpeechRAGChain(
        dao=dao,
        llm=llm_openai.model)
    
    result = rag_chain.classify("우욱")
    pprint(f"입력 텍스트: {result.input_text}")
    pprint(f"혐오 발언 여부: {result.is_hate_speech}")
    pprint(f"혐오 카테고리: {result.categories}")
    pprint(f"신뢰성: {result.evidence_strength}")
    pprint(f"추론 이유: {result.reasoning}")
    pprint(f"혐오 타입: {result.hate_type}")