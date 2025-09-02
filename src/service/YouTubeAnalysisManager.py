import os
from typing import List

from dotenv import load_dotenv

from src.service.YouTubeCommentClassifier import YouTubeCommentClassifier
from src.service.YouTubeScriptClassifier import YouTubeScriptClassifier
from src.dao import YouTubeDBSetup
from src.dao import VectorStoreDao
from src.embedding import EmbeddingModelFactory
from src.llm import HateSpeechRAGChain


class YouTubeAnalysisManager:
    def __init__(self, llm: str = 'openai', model_name: str = None):
        load_dotenv()
        self._init_resources(llm, model_name)

        self.comment_classifier = YouTubeCommentClassifier(
            youtube_dao=self.youtube_dao,
            rag_chain=self.rag_chain
        )
        
        self.script_classifier = YouTubeScriptClassifier(
            youtube_dao=self.youtube_dao,
            rag_chain=self.rag_chain
        )

    def _init_resources(self, llm: str = 'openai', model_name: str = None):
        self.youtube_dao = YouTubeDBSetup()

        self.vectorstore_dao = VectorStoreDao(
            persist_directory="../../data/vectorstores/hate_speech_vectorstore",
            embedding_model=EmbeddingModelFactory.create_embedding_model('upstage'),
            collection_name="hate_speech_collection"
        )
        self.vectorstore_dao.create_vector_store()
        self.vectorstore_dao.initialize_retriever(retriever_type="basic", k=5)

        self.rag_chain = HateSpeechRAGChain(
            dao=self.vectorstore_dao,
            llm=llm,
            model_name=model_name
        )

    def classify_all_comments(self, batch_size: int = 1000, limit: int = None, offset: int = 0):
        """모든 댓글을 분류하는 함수 (관리자 역할)"""
        print("🔄 댓글 분류 시작...")
        
        # CommentClassifier에 위임
        results = self.comment_classifier.classify_all_comments(
            batch_size=batch_size, 
            limit=limit,
            offset=offset
        )
        
        # 통계 출력
        print(f"\n🎉 전체 처리 완료!")
        print(f"📊 총 처리된 댓글: {results['total_processed']}개")
        print(f"🚨 혐오발언 댓글: {results['total_hate_speech']}개")
        print(f"📈 혐오발언 비율: {results['hate_speech_ratio']:.2f}%")
        
        if results['failed_count'] > 0:
            print(f"❌ 실패한 댓글: {results['failed_count']}개")
        
        return results

    def classify_video_script(self, video_id: str, absolute_dir: str = None):
        """비디오 스크립트를 분류하는 함수"""
        return self.script_classifier.classify_video_script(video_id, absolute_dir)

    def analyze_video_complete(self, video_id: str, absolute_dir: str = None):
        """비디오 스크립트와 댓글을 모두 분석하는 함수"""
        print(f"🔄 비디오 {video_id} 전체 분석 시작...")
        
        # 스크립트 분석
        script_results = self.classify_video_script(video_id, absolute_dir)
        
        # 댓글 분석 (해당 비디오의 댓글만)
        # TODO: 특정 비디오의 댓글만 분석하는 메서드 추가 필요
        
        return {
            'video_id': video_id,
            'script_results': script_results,
            'comment_results': None  # TODO: 구현 필요
        }