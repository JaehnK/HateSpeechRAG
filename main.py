from datetime import datetime
import numpy as np
import pandas as pd
import logging
from langchain_teddynote import logging as langsmith_logging
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

from pprint import pprint
from LangChainService import HateSpeechRAGChain
from YouTubeDao import *
from VectorStoreDao import VectorStoreDao
from Embeddings import EmbeddingModelFactory


class HateSpeechAnalyzer:
    """혐오표현 분석 및 DB 업데이트 담당 클래스"""
    
    def __init__(self, youtube_db: YouTubeDBSetup, llm:str='openai'):
        self.youtube_db = youtube_db
        self.setup_logging()
        
        dao = VectorStoreDao(
        persist_directory="./hate_speech_vectorstore",
        embedding_model = EmbeddingModelFactory.create_embedding_model('upstage'),
        collection_name="hate_speech_collection"
        )
        dao.create_vector_store()
        dao.initialize_retriever(retriever_type="basic", k=3)
        
        self.rag_model = HateSpeechRAGChain(dao = dao, llm = llm)
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hate_speech_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_comment(self, comment: str) -> dict:
        self.logger.info(f"분석문장: {comment}")
        return self.rag_model.classify(comment)
    
    def process_all_comments(self, batch_size:int = 1000):
        """전체 댓글을 스트리밍으로 처리"""
        
        self.logger.info("전체 댓글 스트리밍 분석 시작...")
        total_processed = 0
        batch_results = []
        
        idx = 1
        for comment in self.youtube_db.get_comments_generator(batch_size=batch_size/2):
            print(f"idx: {idx} comment: {comment['comment_text']}")
            self.rag_model.classify(comment)
            batch_results.append(comment)
            idx+=1

    def test_unsmile_dataset(self, test_df:pd.DataFrame):
        pred_vecs, actual_vecs = [], []

        for idx, txt in enumerate(test_df['문장']):
            result = self.analyze_comment(txt)
            
            predicted_vec = [0,0,0,0,0,0]
            if "성별" in result.categories:
                predicted_vec[0] = 1
            if "연령" in result.categories:
                predicted_vec[1] = 1
            if "정체성" in result.categories:
                predicted_vec[2] = 1
            if "욕설" in result.categories:
                predicted_vec[3] = 1
            if "기타" in result.categories:
                predicted_vec[4] = 1
            if "혐오없음" in result.categories:
                predicted_vec[5] = 1
            pred_vecs.append(predicted_vec)

            actual_vec = [
            test_df.loc[idx, ['여성/가족', '남성', '성소수자']].max(),  # 성별
            test_df.loc[idx, '연령'],  # 연령
            test_df.loc[idx, ['인종/국적', '지역', '종교']].max(),  # 정체성
            test_df.loc[idx, '악플/욕설'],  # 욕설
            test_df.loc[idx, ['기타 혐오', '개인지칭']].max(),  # 기타
            test_df.loc[idx, 'clean']  # 혐오없음
            ]
            print(actual_vec)
            actual_vecs.append(actual_vec)
        
        return pred_vecs, actual_vecs
    
    def compute_performance_matrix(self, test_df: pd.DataFrame):
        pred, actual = self.test_unsmile_dataset(test_df)
        pred_array = np.array(pred)
        actual_array = np.array(actual)
        
        # 이진분류: 혐오 여부 (혐오없음의 반대)
        binary_pred = 1 - pred_array[:, 5]  # 혐오없음이 0이면 혐오=1
        binary_actual = 1 - actual_array[:, 5]  # clean이 0이면 혐오=1
        
        # 또는 혐오 카테고리 중 하나라도 1이면 혐오로 판단
        # binary_pred = (pred_array[:, :5].sum(axis=1) > 0).astype(int)
        # binary_actual = (actual_array[:, :5].sum(axis=1) > 0).astype(int)
        
        # 멀티라벨 (0~4 인덱스)
        multilabel_pred = pred_array[:, :5]
        multilabel_actual = actual_array[:, :5]
        
        return {
            'binary_f1': f1_score(binary_actual, binary_pred, zero_division=0),
            'hamming_loss': hamming_loss(actual_array, pred_array),
            'exact_match': accuracy_score(actual_array.tolist(), pred_array.tolist()),
            'macro_f1': f1_score(multilabel_actual, multilabel_pred, average='macro', zero_division=0)
        }
                

def test_model(analyzer: HateSpeechAnalyzer):
    df = pd.read_csv("./korean_unsmile_dataset/unsmile_valid_v1.0.tsv", sep='\t')
    result = analyzer.compute_performance_matrix(df)

    # 현재 시간을 파일명에 포함
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. JSON 형태로 상세 결과 저장
    json_filename = f"hate_speech_evaluation_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    # 2. 성능 요약을 텍스트 파일로 저장
    summary_filename = f"performance_summary_{timestamp}.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("=== 혐오표현 분류 모델 성능 평가 결과 ===\n\n")
        f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== 성능 지표 ===\n")
        f.write(f"Binary F1-Score: {result['binary_f1']:.4f}\n")
        f.write(f"Exact Match: {result['exact_match']:.4f}\n")
        f.write(f"Hamming Loss: {result['hamming_loss']:.4f}\n")
        f.write(f"Macro F1-Score: {result['macro_f1']:.4f}\n")
    
    print(f"결과가 다음 파일들로 저장되었습니다:")
    print(f"  - 상세 결과: {json_filename}")
    print(f"  - 성능 요약: {summary_filename}")



if __name__ == "__main__":
    load_dotenv()
    langsmith_logging.langsmith("HateSpeechTest")
    analyzer = HateSpeechAnalyzer(YouTubeDBSetup(), llm='anthropic')
    # test_model(analyzer)
    # analyzer.analyze_comment("생물학 화학 의학 약학 공부하는 년들아 제발 한남 골라 죽이는 생화학 무기나 바이러스 같은 것 좀 만들어 줘라 이기야광광")


    # pprint(f"입력 텍스트: {result.input_text}")
    # pprint(f"혐오 발언 여부: {result.is_hate_speech}")
    # pprint(f"혐오 카테고리: {result.categories}")
    # pprint(f"신뢰성: {result.evidence_strength}")
    # pprint(f"추론 이유: {result.reasoning}")
    # pprint(f"혐오 타입: {result.hate_type}")
    
