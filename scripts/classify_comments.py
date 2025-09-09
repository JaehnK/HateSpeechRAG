import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.service import YouTubeAnalysisManager


if __name__ == "__main__":
    # 분석 매니저 초기화
    analysis = YouTubeAnalysisManager(llm='openai', model_name='gpt-5')
    
    print("댓글 분류 시작...")
    
    # 댓글 분류 실행
    result = analysis.classify_all_comments(
        batch_size=5,  # 배치 크기
        limit=None,       # 제한 없음 (전체 처리)
        offset=0,          # 시작점
        do_once=True
    )
    
    # 결과를 JSON 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'comment_rag_result_{timestamp}.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ 결과가 {filename}에 저장되었습니다.")
    print(f"처리 결과:")
    print(f"   - 총 처리된 댓글: {result['total_processed']}개")
    print(f"   - 혐오발언 댓글: {result['total_hate_speech']}개")
    print(f"   - 혐오발언 비율: {result['hate_speech_ratio']:.2f}%")
    print(f"   - 실패한 댓글: {result['failed_count']}개")
    
    if result['failed_count'] > 0:
        print(f"\n❌ 실패한 댓글들:")
        for failed in result['failed_comments'][:5]:  # 처음 5개만 출력
            print(f"   - {failed['comment_id']}: {failed['error_type']} - {failed['error']}")
        if len(result['failed_comments']) > 5:
            print(f"   ... 외 {len(result['failed_comments']) - 5}개")