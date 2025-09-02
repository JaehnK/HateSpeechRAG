import pandas as pd
import argparse
import shutil
from typing import List
from langchain.schema import Document

from VectorStoreDao import VectorStoreDao
from Embeddings import EmbeddingModelFactory


def concat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 11개 컬럼을 6개 간소화 컬럼으로 변환
    """
    result_df = df.copy()
    result_df['성별'] = df[['여성/가족', '남성', '성소수자']].max(axis=1)
    result_df['정체성'] = df[['인종/국적', '지역', '종교']].max(axis=1)
    result_df['연령'] = df['연령']
    result_df['기타'] = df[['기타 혐오', '개인지칭']].max(axis=1)
    result_df['욕설'] = df['악플/욕설']
    result_df['혐오없음'] = df['clean']
    print("컬럼 변환 완료.")
    return result_df

def create_documents_from_df(df: pd.DataFrame) -> List[Document]:
    """
    데이터프레임을 Document 객체 리스트로 변환
    """
    documents = []
    for idx, row in df.iterrows():
        if pd.isna(row['문장']) or row['문장'].strip() == '':
            continue
        doc = Document(
            page_content=str(row['문장']).strip(),
            metadata={
                "성별": int(row['성별']), "정체성": int(row['정체성']),
                "연령": int(row['연령']), "기타": int(row['기타']),
                "욕설": int(row['욕설']), "혐오없음": int(row['혐오없음']),
                "row_index": idx
            }
        )
        documents.append(doc)
    print(f"Document 변환 완료: {len(documents)}개")
    return documents

def main(args):
    """메인 실행 함수"""
    # 1. 데이터 로드
    print(f"데이터 로드 중... 경로: {args.file_path}")
    df = pd.read_csv(args.file_path, sep='\t')
    print(f"데이터 로드 완료: {len(df)}개 행")

    # 2. 문서 객체 생성
    processed_df = concat_columns(df)
    documents = create_documents_from_df(processed_df)
    if not documents:
        print("업로드할 문서가 없습니다.")
        return

    # 3. 기존 데이터베이스 디렉토리 삭제
    if args.force_recreate:
        print(f"기존 디렉토리 삭제 중... 경로: {args.persist_directory}")
        shutil.rmtree(args.persist_directory, ignore_errors=True)

    # 4. DAO 및 임베딩 모델 초기화
    print(f"임베딩 모델 초기화... ({args.embedding_provider}) ")
    embedding_model = EmbeddingModelFactory.create_embedding_model(args.embedding_provider)
    dao = VectorStoreDao(
        persist_directory=args.persist_directory,
        embedding_model=embedding_model,
        collection_name=args.collection_name
    )

    # 5. HNSW 인덱스 설정 및 배치 업로드
    print(f"벡터스토어 생성 시작... (거리 함수: {args.distance_function})")
    collection_metadata = {"hnsw:space": args.distance_function}
    batch_size = args.batch_size
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 문서)")
        
        if i == 0:
            # 첫 배치는 벡터스토어 생성
            from langchain_chroma import Chroma
            dao.vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model.embedding_model,
                persist_directory=args.persist_directory,
                collection_name=args.collection_name,
                collection_metadata=collection_metadata
            )
        else:
            # 이후 배치는 문서 추가
            dao.add_documents(batch)

    print("\n=== 데이터베이스 구축 완료 ===")
    info = dao.get_collection_info()
    for key, value in info.items():
        print(f"- {key}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="혐오표현 데이터셋으로 ChromaDB 벡터스토어를 구축합니다.")
    parser.add_argument("--file_path", type=str, default="korean_unsmile_dataset/unsmile_train_v1.0.tsv", help="데이터셋 파일 경로")
    parser.add_argument("--persist_directory", type=str, default="./hate_speech_vectorstore", help="벡터스토어를 저장할 디렉토리")
    parser.add_argument("--collection_name", type=str, default="hate_speech_collection", help="ChromaDB 컬렉션 이름")
    parser.add_argument("--embedding_provider", type=str, default="upstage", choices=["upstage", "openai"], help="사용할 임베딩 모델")
    parser.add_argument("--distance_function", type=str, default="cosine", choices=["cosine", "l2", "ip"], help="HNSW 인덱스의 거리 함수")
    parser.add_argument("--batch_size", type=int, default=500, help="한 번에 처리할 문서의 수")
    parser.add_argument("--force_recreate", action="store_true", help="이 플래그가 있으면 기존 디렉토리를 강제로 삭제하고 다시 생성합니다.")
    
    args = parser.parse_args()
    main(args)
