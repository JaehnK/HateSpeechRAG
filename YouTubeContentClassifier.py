import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_teddynote import logging
from dotenv import load_dotenv

from YouTubeDao import YouTubeDBSetup
from VectorStoreDao import VectorStoreDao
from Embeddings import EmbeddingModelFactory
from LLMServices import LLMServiceFactory
from LangChainService import HateSpeechRAGChain

class YOuTubeContentClassifier:
    def __init__(self):
        load_dotenv()
        # logging.langsmith("HateSpeechTest")
        
        self.youtube_dao = YouTubeDBSetup()
        self.connection = self.youtube_dao.get_connection()
        self.cursor = self.connection.cursor()
        
        self.vectorstore_dao = VectorStoreDao(
            persist_directory="./hate_speech_vectorstore",
            embedding_model = EmbeddingModelFactory.create_embedding_model('upstage'),
            collection_name="hate_speech_collection"
        )
        self.vectorstore_dao.create_vector_store()
        self.vectorstore_dao.initialize_retriever(retriever_type="basic", k=5)
        
        self.llm = LLMServiceFactory.create_llm_service("openai")
        self.rag_chain = HateSpeechRAGChain(
            dao=self.vectorstore_dao,
            llm=self.llm.model
        )
        
        self._init_text_splitter()

    def _load_script(self, video_id: str, absolute_dir: str = None) -> str:
        """비디오 ID로 유튜브 스크립트를 조회"""
        try:
            self.cursor.execute("SELECT script FROM videos WHERE video_id = %s", (video_id,))

            script_result = self.cursor.fetchone()
            # script_result = script_result.fetchone()
            if script_result and script_result[0]:
                script_content = script_result[0]
                
                # absolute_dir이 제공된 경우 절대경로를 붙여서 반환
                if absolute_dir:
                    import os
                    # 절대경로 정규화
                    abs_path = os.path.abspath(absolute_dir)
                    return os.path.join(abs_path, script_content)
                else:
                    return script_content
            else:
                print(f"No script found for video_id: {video_id}")
                return ""
                
        except Exception as e:
            print(f"Error loading script for video_id {video_id}: {e}")
            return ""

    def _init_text_splitter(self):
        """한국어 문장 단위 텍스트 스플리터 초기화"""
        korean_separators = [
            "\n\n",
            "\n", 
            ".",
            "!",
            "?",
            ";",
            ",",
            " ",
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 문장 단위로 자르기 위해 작게 설정
            chunk_overlap=50,
            separators=korean_separators,
            length_function=len,
        )

    def _split_text_simple(self, text: str) -> List[str]:
        """가장 간단한 문장 분할"""
        if not text or not text.strip():
            return []
        
        # 줄바꿈 정리
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 문장부호로 분할
        sentences = re.split(r'[.!?]+', text)
        
        # 정리
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 1:  # 너무 짧은 것 제외
                result.append(sentence)
        
        return result


    def _split_script_to_sentences(self, script_content: str) -> list:
        """스크립트를 문장 단위로 분할"""
        if not script_content:
            return []
        
        chunks = self.text_splitter.split_text(script_content)
        # 빈 문자열이나 너무 짧은 문장 제거
        sentences = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 5]
        return sentences

    def classify_video_script(self, video_id: str, absolute_dir: str = None) -> dict:
        """비디오 스크립트를 문장 단위로 분류 (ThreadPoolExecutor 사용)"""
        print(f"Processing video_id: {video_id}")
        
        # 1. 스크립트 로드
        script_content = self._load_script(video_id, absolute_dir)
        if not script_content:
            return {
                'video_id': video_id,
                'total_sentences': 0,
                'classifications': [],
                'hate_speech_count': 0,
                'error': 'No script found'
            }
        
        # 2. 스크립트가 파일 경로인 경우 파일 읽기
        if absolute_dir or script_content.endswith('.txt'):
            try:
                with open(script_content, 'r', encoding='utf-8') as f:
                    script_text = f.read()
            except Exception as e:
                print(f"Error reading script file {script_content}: {e}")
                return {
                    'video_id': video_id,
                    'total_sentences': 0,
                    'classifications': [],
                    'hate_speech_count': 0,
                    'error': f'File read error: {e}'
                }
        else:
            script_text = script_content
        
        # 3. 문장 단위로 분할
        # sentences = self._split_script_to_sentences(script_text)
        sentences = self._split_text_simple(script_text)
        print(f"Split into {len(sentences)} sentences")
        
        # 4. ThreadPoolExecutor로 병렬 분류 처리
        classifications = self._classify_sentences_async(sentences)
        
        return {
            'video_id': video_id,
            'total_sentences': len(sentences),
            'classifications': classifications,
        }
    
    def _classify_sentences_async(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """ThreadPoolExecutor를 사용한 병렬 분류 처리"""
        if not sentences:
            return []
        
        classifications = [None] * len(sentences)  # 순서 보장을 위한 초기화
        
        def classify_single_sentence(sentence_data):
            """단일 문장 분류 함수"""
            index, sentence = sentence_data
            try:
                script_result = self.rag_chain.classify(sentence)
                print(f"Sentence {index+1}/{len(sentences)}: {type(script_result)} - {str(script_result)[:100]}...")
                
                return {
                    'sentence_index': index,
                    'sentence': sentence,
                    'classification_result': script_result
                }
            except Exception as e:
                print(f"Error classifying sentence {index}: {e}")
                return {
                    'sentence_index': index,
                    'sentence': sentence,
                    'classification_result': None,
                    'error': str(e)
                }
        
        # ThreadPoolExecutor로 병렬 처리
        with ThreadPoolExecutor(max_workers=5) as executor:  # 동시 요청 수 조절
            # 문장과 인덱스를 함께 전달
            sentence_data = [(i, sentence) for i, sentence in enumerate(sentences)]
            
            # 모든 작업 제출
            future_to_index = {
                executor.submit(classify_single_sentence, data): data[0] 
                for data in sentence_data
            }
            
            # 완료된 작업들 수집 (순서 보장)
            for future in as_completed(future_to_index):
                try:
                    script_result = future.result()
                    classifications[script_result['sentence_index']] = script_result
                except Exception as e:
                    index = future_to_index[future]
                    print(f"Exception in future for sentence {index}: {e}")
                    classifications[index] = {
                        'sentence_index': index,
                        'sentence': sentences[index],
                        'classification_result': None,
                        'error': str(e)
                    }
        
        # None이 남아있다면 에러로 처리 (혹시 모를 상황 대비)
        for i, classification in enumerate(classifications):
            if classification is None:
                classifications[i] = {
                    'sentence_index': i,
                    'sentence': sentences[i],
                    'classification_result': None,
                    'error': 'Unknown error occurred'
                }
        
        return classifications

if "__main__" == __name__:
    from pprint import pprint
    classifier = YOuTubeContentClassifier()
    
    # print(classifier._load_script("k_byR7RQ-PI", "/home/jaehun/lab/YouTubeHateSpeech"))
    results = classifier.classify_video_script("k_byR7RQ-PI", "/home/jaehun/lab/YouTubeHateSpeech")
    for _, result in enumerate( results['classifications']):
        print("\n")
        pprint(f"입력 텍스트: {result['classification_result'].input_text}")
        pprint(f"혐오 발언 여부: {result['classification_result'].is_hate_speech}")
        pprint(f"혐오 카테고리: {result['classification_result'].categories}")
        pprint(f"신뢰성: {result['classification_result'].evidence_strength}")
        pprint(f"추론 이유: {result['classification_result'].reasoning}")
        pprint(f"혐오 타입: {result['classification_result'].hate_type}")
        print("\n")
        print("===="*20)
        pprint(f"프롬프트: {result['classification_result'].prompt}")
        print("===="*20)
    # print(result[0]['classifications'].prompt)