from typing import List, Dict, Optional, Any, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import os
import json
import pandas as pd
from openai import OpenAI
import time # time 모듈 추가

from VectorStoreDao import VectorStoreDao
from Embeddings import EmbeddingModelFactory
from LLMServices import LLMServiceFactory
from LangChainService import HateSpeechRAGChain, HateSpeechClassification
from HateSpeechDao import HateSpeechDBSetup


def load_dataset(file_path: str) -> List[str]:
    """
    tsv 파일에서 텍스트 데이터를 로드합니다.
    Args:
        file_path: tsv 파일 경로
    Returns:
        텍스트 리스트
    """
    df = pd.read_csv(file_path, sep='\t')
    return df['문장'].tolist()

def generate_prompts_for_batch(texts: List[str]) -> Tuple[List[Dict], Dict[str, str]]:
    """
    주어진 텍스트 리스트에 대해 HateSpeechRAGChain을 사용하여 프롬프트를 생성하고,
    OpenAI Batch API 형식에 맞는 딕셔너리 리스트와 custom_id-프롬프트 매핑을 반환합니다.
    Args:
        texts: 분류할 텍스트 리스트
    Returns:
        (OpenAI Batch API 요청 딕셔너리 리스트, custom_id-프롬프트 매핑 딕셔너리)
    """
    # VectorStoreDao 초기화
    dao = VectorStoreDao(
        persist_directory="./hate_speech_vectorstore",
        embedding_model = EmbeddingModelFactory.create_embedding_model('upstage'), 
        collection_name="hate_speech_collection"
    )
    dao.create_vector_store() # force_recreate=True 제거 또는 False로 변경
    dao.initialize_retriever(retriever_type="basic", k=3) # 또는 "mmr", "self_query"

    # LLM 서비스 초기화 (HateSpeechRAGChain 내부에서 처리되므로 여기서는 제거)
    # llm_service = LLMServiceFactory.create_llm_service("openai", "gpt-4o-mini") # 모델명 지정

    # HateSpeechRAGChain 초기화 (LLM 제공자 이름과 모델 이름을 직접 전달)
    rag_chain = HateSpeechRAGChain(dao=dao, llm="openai", model_name="gpt-5-mini")

    batch_requests = []
    prompt_mapping = {}
    for i, text in enumerate(texts):
        custom_id = f"request-{i}"
        prompt_content = rag_chain.get_prompt(text)
        batch_requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5-mini", # 실제 Batch API에서 사용할 모델
                "messages": [{"role": "user", "content": prompt_content}]
            }
        })
        prompt_mapping[custom_id] = prompt_content
        print(f"{i} request finished")
    return batch_requests, prompt_mapping


def save_batch_requests_to_jsonl(batch_requests: List[Dict], output_file: str) -> None:
    """
    OpenAI Batch API 요청 리스트를 JSONL 파일로 저장합니다.
    Args:
        batch_requests: OpenAI Batch API 요청 딕셔너리 리스트
        output_file: JSONL 파일 경로
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    print(f"Batch requests saved to {output_file}")

def upload_file_and_create_batch(file_path: str, client: OpenAI) -> str:
    """
    JSONL 파일을 OpenAI에 업로드하고 배치 작업을 생성합니다.
    Args:
        file_path: 업로드할 JSONL 파일 경로
        client: OpenAI 클라이언트 인스턴스
    Returns:
        생성된 배치 작업 ID
    """
    print(f"Uploading {file_path} to OpenAI...")
    file_object = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )
    print(f"File uploaded. File ID: {file_object.id}")

    print("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=file_object.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Batch job created. Batch ID: {batch_job.id}")
    return batch_job.id

def check_batch_status(batch_id: str, client: OpenAI, interval: int = 1800) -> Dict[str, Any]:
    """
    OpenAI 배치 작업의 상태를 주기적으로 확인합니다.
    Args:
        batch_id: 확인할 배치 작업 ID
        client: OpenAI 클라이언트 인스턴스
        interval: 상태 확인 주기 (초)
    Returns:
        최종 배치 작업 정보 딕셔너리
    """
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status
        print(f"Batch job {batch_id} status: {status}")

        if status in ["completed", "failed", "cancelled", "expired"]:
            print(f"Batch job {batch_id} finished with status: {status}")
            return batch_status.to_dict()
        
        print(f"Waiting for {interval} seconds before next check...")
        time.sleep(interval)

# def download_and_process_results(file_id: str, client: OpenAI, prompt_mapping: Dict[str, str], output_dir: str = ".") -> List[Dict]:
#     """
#     OpenAI에서 결과 파일을 다운로드하고 처리합니다.
#     Args:
#         file_id: 다운로드할 파일 ID
#         client: OpenAI 클라이언트 인스턴스
#         output_dir: 결과 파일을 저장할 디렉토리
#     Returns:
#         처리된 결과 딕셔너리 리스트
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     file_info = client.files.retrieve(file_id)
#     file_content = client.files.content(file_id).text

#     output_path = os.path.join(output_dir, f"batch_results_{file_id}.jsonl")
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(file_content)
#     print(f"Result file downloaded to {output_path}")

#     results = []
#     db_setup = HateSpeechDBSetup() # HateSpeechDBSetup 인스턴스 생성
#     db_setup.create_tables()

#     for line in file_content.strip().split('\n'):
#         if line:
#             response_data = json.loads(line)
#             try:
#                 # OpenAI Batch API 응답에서 content 추출
#                 # response_data 구조: {"custom_id": ..., "response": {"body": {"choices": [{"message": {"content": "..."}}]}}}
#                 content_str = response_data['response']['body']['choices'][0]['message']['content']
#                 parsed_result = HateSpeechClassification.model_validate_json(content_str)
                
#                 # 원본 입력 텍스트를 찾아서 HateSpeechClassification 객체에 추가 (필요하다면)
#                 # custom_id를 사용하여 generate_prompts_for_batch에서 사용된 원본 텍스트를 매핑할 수 있음
#                 # 현재는 parsed_result.input_text가 LLM 응답에서 추출되므로 별도 매핑 불필요

#                 # 데이터베이스에 저장
#                 original_prompt = prompt_mapping.get(response_data['custom_id'])
#                 db_setup.save_script(parsed_result, response_data['custom_id'], original_prompt)
#                 results.append(parsed_result.model_dump())
#             except Exception as e:
#                 print(f"Error parsing or saving result for custom_id {response_data.get('custom_id', 'N/A')}: {e}")
#                 results.append(response_data) # 파싱 실패 시 원본 응답 저장
            
#     return results

def download_and_process_results(file_id: str, client: OpenAI, prompt_mapping: Dict[str, str], output_dir: str = ".") -> List[Dict]:
    """
    OpenAI에서 결과 파일을 다운로드하고 처리합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_info = client.files.retrieve(file_id)
    file_content = client.files.content(file_id).text

    output_path = os.path.join(output_dir, f"batch_results_{file_id}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(file_content)
    print(f"Result file downloaded to {output_path}")

    # DB 저장 부분 제거, 파일 저장만
    results = []
    # db_setup = HateSpeechDBSetup()  # 주석 처리
    # db_setup.create_tables()       # 주석 처리

    for line in file_content.strip().split('\n'):
        if line:
            response_data = json.loads(line)
            # 원본 응답 그대로 저장
            results.append(response_data)
            
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 주석 처리

    # dataset_path = "/home/jaehun/lab/hateSpeechRAG/korean_unsmile_dataset/unsmile_valid_v1.0.tsv"
    # texts_to_classify = load_dataset(dataset_path)

    # print(f"Loaded {len(texts_to_classify)} texts from dataset.")
    
    # batch_prompts, prompt_mapping = generate_prompts_for_batch(texts_to_classify)
    # print(f"Generated {len(batch_prompts)} batch prompts.")
    
    # # 생성된 프롬프트 중 첫 번째를 출력하여 구조 확인
    # if batch_prompts:
    #     print("\n--- First Batch Prompt ---")
    #     print(json.dumps(batch_prompts[0], indent=2, ensure_ascii=False))

    output_jsonl_path = "./batch_requests.jsonl"
    # save_batch_requests_to_jsonl(batch_prompts, output_jsonl_path)
    
    # print(f"✅ JSONL 파일 생성 완료: {output_jsonl_path}")
    # print(f"📊 총 {len(batch_prompts)}개 요청 준비됨")
    
    # 여기서부터 모두 주석 처리
    
    # 배치 작업 생성 및 업로드
    batch_id = upload_file_and_create_batch(output_jsonl_path, client)
    print(f"Started OpenAI Batch Job with ID: {batch_id}")

    # 배치 작업 완료 대기 및 상태 확인
    batch_result = check_batch_status(batch_id, client)
    print(f"Batch job final status: {batch_result['status']}")
    
    if batch_result['status'] == "completed":
        print(f"Result file ID: {batch_result['output_file_id']}")
        results = download_and_process_results(batch_result['output_file_id'], client, {})  # 빈 딕셔너리로 수정
        print(f"Downloaded and processed {len(results)} results.")
    else:
        print("Batch job did not complete successfully.")
    