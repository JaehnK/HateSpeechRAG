import sys
import os
from datetime import datetime
import asyncio
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.service import YouTubeAnalysisManager


def main():
    # Initialize analysis manager with gpt-5-mini
    analysis = YouTubeAnalysisManager(llm='openai', model_name='gpt-5-mini')

    print("테스트 배치(50) 분류 시작 (gpt-5-mini, DB 업로드 없음)...")

    # 설정값
    batch_size = 50
    offset = 0
    max_concurrency = 10
    timeout_per_task = 90
    ensure_model_name = 'gpt-5-mini'

    # 모델 확인 (가능한 경우에만 검증)
    try:
        underlying_llm = getattr(analysis.comment_classifier.rag_chain.llm, '_llm', None)
        current_model_name = getattr(underlying_llm, 'model_name', None)
        if ensure_model_name and current_model_name and current_model_name != ensure_model_name:
            raise ValueError(f"요청한 모델({ensure_model_name})과 현재 모델({current_model_name})이 다릅니다.")
    except Exception:
        # 확인 불가 시는 통과 (기본값이 gpt-5-mini이므로)
        pass

    # 결과 저장 디렉터리 준비
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("data", "batch_results", f"test_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 기본 결과 템플릿
    result = {
        'file_path': None,
        'total_processed': 0,
        'total_hate_speech': 0,
        'hate_speech_ratio': 0.0,
        'failed_count': 0
    }

    # 배치 로드 (첫 배치만)
    batch_comments = None
    for comments in analysis.youtube_dao.get_comments_generator(batch_size=batch_size, include_analysis=True, _offset=offset):
        batch_comments = comments
        break

    if not batch_comments:
        print("⚠️ 로드된 댓글 배치가 없습니다.")
    else:
        # is_hate_speech == False 인 댓글만 필터링
        target_comments = [c for c in batch_comments if c.get('is_hate_speech') is False]

        if not target_comments:
            print("⚠️ is_hate_speech=False 인 댓글이 없는 배치입니다.")
        else:
            print(f"🔄 테스트 배치 분류 시작 (is_hate_speech=False): {len(target_comments)}개 (동시성 {max_concurrency})")
            # 기존 비동기 분류 로직 재사용
            results = asyncio.run(
                analysis.comment_classifier._classify_batch_async(
                    target_comments,
                    max_concurrency=max_concurrency,
                    timeout_per_task=timeout_per_task,
                    batch_number=1
                )
            )

            total_processed = len(results)
            total_hate_speech = sum(1 for r in results if r.get('is_hate_speech'))
            failed_count = sum(1 for r in results if r.get('error'))
            hate_speech_ratio = (total_hate_speech / total_processed * 100) if total_processed > 0 else 0.0

            # XLSX 저장
            df = pd.DataFrame(results)
            file_path = os.path.join(save_dir, "test_batch_results.xlsx")
            try:
                df.to_excel(file_path, index=False)
                print(f"   💾 테스트 배치 결과가 {file_path}에 저장되었습니다.")
            except Exception as e:
                print(f"   ⚠️ XLSX 저장 실패: {e}. CSV로 대체 저장합니다.")
                file_path = os.path.join(save_dir, "test_batch_results.csv")
                df.to_csv(file_path, index=False, encoding='utf-8-sig')

            print(f"✅ 테스트 배치 완료: {total_processed}개 처리, {total_hate_speech}개 혐오발언, 실패 {failed_count}개")

            result = {
                'file_path': file_path,
                'total_processed': total_processed,
                'total_hate_speech': total_hate_speech,
                'hate_speech_ratio': hate_speech_ratio,
                'failed_count': failed_count
            }

    file_path = result.get('file_path')
    if file_path:
        print(f"✅ XLSX 저장 완료: {file_path}")
    else:
        print("⚠️ 저장된 파일이 없습니다 (배치 비어있음 또는 미분석 댓글 없음)")

    print("처리 결과:")
    print(f"   - 총 처리된 댓글: {result.get('total_processed', 0)}개")
    print(f"   - 혐오발언 댓글: {result.get('total_hate_speech', 0)}개")
    print(f"   - 혐오발언 비율: {result.get('hate_speech_ratio', 0.0):.2f}%")
    print(f"   - 실패한 댓글: {result.get('failed_count', 0)}개")


if __name__ == "__main__":
    main()


