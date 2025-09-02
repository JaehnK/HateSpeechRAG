from src.service.YouTubeAnalysisManager import BaseYouTubeClassifier
from src.dao import YouTubeDBSetup
from src.llm import HateSpeechRAGChain
from typing import List
import pandas as pd
import os
from datetime import datetime

class YouTubeCommentClassifier(BaseYouTubeClassifier):
    def __init__(self, youtube_dao: YouTubeDBSetup, rag_chain: HateSpeechRAGChain):
        super().__init__(youtube_dao, rag_chain)
    
    def classify_single_comment(self, comment_text: str):
        """단일 댓글 분류"""
        return self.rag_chain.classify(comment_text)
    
    def _load_comments_to_dataframe(self, limit: int = None):
        """DB에서 댓글을 DataFrame으로 로드 (필요시 사용)"""
        
        query = """
        SELECT comment_id, comment_text, video_id, author
        FROM comments 
        WHERE is_hate_speech IS NULL
        ORDER BY published_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            temp_connection = self.youtube_dao.get_connection()
            df = pd.read_sql(query, temp_connection)
            temp_connection.close()
            
            print(f" {len(df)}개의 미분류 댓글을 로드했습니다.")
            return df
            
        except Exception as e:
            print(f"❌ 댓글 로드 실패: {e}")
            return pd.DataFrame()

    def classify_all_comments(self, batch_size: int = 1000, limit: int = None, offset: int = 0):
        """모든 댓글을 분류하는 함수"""
        total_processed = 0
        total_hate_speech = 0
        batch_count = 0
        failed_comments = []
        current_offset = offset  # 현재 offset 초기화
        
        # 배치 결과를 저장할 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("data", "batch_results", timestamp)
        os.makedirs(save_dir, exist_ok=True)
        print(f"📄 배치 결과는 '{save_dir}'에 저장됩니다.")
        
        print("🔄 댓글 분류 시작...")
        
        for batch_comments in self.youtube_dao.get_comments_generator(
            batch_size=batch_size, 
            include_analysis=True, # False -> True
            _offset=offset
        ):
            unanalyzed_comments = [
                comment for comment in batch_comments 
                if comment.get('is_hate_speech') is None
            ]
            
            if not unanalyzed_comments:
                continue
                
            batch_count += 1
            print(f"🔄 배치 {batch_count}: {len(unanalyzed_comments)}개 미분석 댓글 처리 중...")
            
            batch_results = []
            for comment in unanalyzed_comments:
                try:
                    classification_result = self.classify_single_comment(
                        comment_text=comment['comment_text']
                    )
                    
                    batch_results.append({
                        'comment_id': comment['comment_id'],
                        'is_hate_speech': classification_result.is_hate_speech,
                        'categories': classification_result.categories,
                        'similar_cases_used': classification_result.similar_cases_used,
                        'target_group': classification_result.target_group,
                        'hate_type': classification_result.hate_type,
                        'used_prompt': classification_result.prompt,
                        'classification_result': classification_result
                    })
                    
                except Exception as e:
                    # 에러 타입별 처리
                    error_info = self._handle_classification_error(e, comment, batch_count)
                    failed_comments.append(error_info)
                    
                    batch_results.append({
                        'comment_id': comment['comment_id'],
                        'is_hate_speech': False,
                        'categories': [],
                        'similar_cases_used': [],
                        'target_group': None,
                        'hate_type': None,
                        'used_prompt': None,
                        'classification_result': None,
                        'error': str(e)
                    })
            
            # 배치 결과를 데이터프레임으로 저장
            if batch_results:
                try:
                    # classification_result 객체를 문자열로 변환하여 저장 준비
                    results_for_df = []
                    for r in batch_results:
                        res_copy = r.copy()
                        if 'classification_result' in res_copy and res_copy['classification_result'] is not None:
                            res_copy['classification_result'] = str(res_copy['classification_result'])
                        results_for_df.append(res_copy)
                    
                    df = pd.DataFrame(results_for_df)
                    file_path = os.path.join(save_dir, f"batch_{batch_count}_results.csv")
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    print(f"   💾 배치 {batch_count}의 결과가 {file_path}에 저장되었습니다.")
                except Exception as e:
                    print(f"   ⚠️ 배치 {batch_count}의 결과를 파일로 저장하는 데 실패했습니다: {e}")

            # DB 업데이트
            update_result = self.youtube_dao.update_hate_speech_analysis_batch(batch_results)
        
            # DB 업데이트 실패한 댓글들도 수집
            if update_result['failed_count'] > 0:
                for failed_update in update_result['failed_comments']:
                    failed_comment = {
                        'comment_id': failed_update['comment_id'],
                        'error': failed_update['error'],
                        'error_type': 'DB_UPDATE_ERROR',
                        'batch_number': batch_count
                    }
                    failed_comments.append(failed_comment)
            
            # 통계 업데이트
            batch_hate_speech = sum(1 for result in batch_results if result['is_hate_speech'])
            total_processed += len(batch_results)
            total_hate_speech += batch_hate_speech
            current_offset += len(batch_results)  # offset 업데이트
            
            print(f"✅ 배치 {batch_count} 완료: {len(batch_results)}개 처리, {batch_hate_speech}개 혐오발언")
            print(f"   DB 업데이트: {update_result['success_count']}개 성공, {update_result['failed_count']}개 실패")
            print(f"   현재 offset: {current_offset}")
        
            # limit 체크 (한 번만)
            if limit and total_processed >= limit:
                break
            
        hate_speech_ratio = total_hate_speech / total_processed * 100 if total_processed > 0 else 0.0
        
        return {
            'total_processed': total_processed,
            'total_hate_speech': total_hate_speech,
            'hate_speech_ratio': hate_speech_ratio,
            'failed_comments': failed_comments,
            'failed_count': len(failed_comments),
            'current_offset': current_offset,
            'start_offset': offset
            }

            
    def _handle_classification_error(self, error: Exception, comment: dict, batch_count: int) -> dict:
        """분류 에러를 타입별로 처리"""
        from openai import RateLimitError as OpenAIRateLimitError, BadRequestError as OpenAIBadRequestError
        from anthropic import RateLimitError as AnthropicRateLimitError, BadRequestError as AnthropicBadRequestError
        
        error_type = type(error).__name__
        error_message = str(error)
        
        # 에러 타입별 처리
        if isinstance(error, (OpenAIRateLimitError, AnthropicRateLimitError)):
            print(f"⚠️ Rate Limit 에러 - 댓글 {comment['comment_id']}: {error_message}")
            
        elif isinstance(error, (OpenAIBadRequestError, AnthropicBadRequestError)):
            print(f"❌ Bad Request 에러 (400) - 댓글 {comment['comment_id']}: {error_message}")
            
        elif "timeout" in error_message.lower():
            print(f"⏰ Timeout 에러 - 댓글 {comment['comment_id']}: {error_message}")
            
        elif "connection" in error_message.lower():
            print(f"❌ Connection 에러 - 댓글 {comment['comment_id']}: {error_message}")
            
        else:
            print(f"❌ 기타 에러 - 댓글 {comment['comment_id']}: {error_message}")
        
        return {
            'comment_id': comment['comment_id'],
            'video_id': comment.get('video_id'),
            'author': comment.get('author'),
            'comment_text': comment['comment_text'][:100] + '...' if len(comment['comment_text']) > 100 else comment['comment_text'],
            'error': error_message,
            'error_type': error_type,
            'batch_number': batch_count,
            'is_retryable': self._is_retryable_error(error)
        }

    def _is_retryable_error(self, error: Exception) -> bool:
        """에러가 재시도 가능한지 판단"""
        from openai import RateLimitError as OpenAIRateLimitError, BadRequestError as OpenAIBadRequestError
        from anthropic import RateLimitError as AnthropicRateLimitError, BadRequestError as AnthropicBadRequestError
        
        # 400 Bad Request는 재시도하면 안 됨
        if isinstance(error, (OpenAIBadRequestError, AnthropicBadRequestError)):
            return False
        
        # Rate Limit은 LLM 서비스에서 재시도하므로 여기서는 재시도하지 않음
        if isinstance(error, (OpenAIRateLimitError, AnthropicRateLimitError)):
            return False
        
        # Timeout, Connection 에러는 재시도 가능
        error_message = str(error).lower()
        if any(keyword in error_message for keyword in ['timeout', 'connection', 'network']):
            return True
        
        # 기타 에러는 재시도하지 않음
        return False