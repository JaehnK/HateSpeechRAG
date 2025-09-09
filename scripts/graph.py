"""
사회연결망 그래프 사용 예시
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import YouTubeDBSetup, GraphFactory, GraphBuilder, GraphType
from src.graph import create_comment_reply_graph, build_graph_from_video, build_graph_from_videos


def example_basic_usage():
    """기본 사용법 예시"""
    print("=== 기본 사용법 예시 ===")
    
    # 1. 데이터베이스 연결
    db_setup = YouTubeDBSetup()
    
    # 2. 그래프 빌더 생성
    builder = GraphBuilder(db_setup)
    
    # 3. 비디오 ID 목록 조회
    video_ids = db_setup.get_unique_video_ids()
    if not video_ids:
        print("데이터베이스에 비디오가 없습니다.")
        return
    
    # 4. 첫 번째 비디오로 그래프 구축
    first_video_id = video_ids[0]
    print(f"비디오 {first_video_id}로 그래프 구축 중...")
    
    graph = builder.build_comment_reply_graph_by_video(
        video_id=first_video_id,
        graph_id=f"example_graph_{first_video_id}"
    )
    
    # 5. 그래프 기본 정보 출력
    stats = graph.get_statistics()
    print(f"그래프 통계: {stats}")
    
    # 6. 상위 대댓글 작성자들
    top_repliers = graph.get_top_repliers(limit=5)
    print(f"상위 대댓글 작성자: {top_repliers}")
    
    # 7. 혐오표현 네트워크 분석
    hate_stats = graph.get_hate_speech_network_stats()
    print(f"혐오표현 네트워크 통계: {hate_stats}")


def example_factory_usage():
    """팩토리 패턴 사용 예시"""
    print("\n=== 팩토리 패턴 사용 예시 ===")
    
    # 1. 사용 가능한 그래프 타입 확인
    available_types = GraphFactory.get_available_types()
    print(f"사용 가능한 그래프 타입: {available_types}")
    
    # 2. 팩토리로 그래프 생성
    graph1 = GraphFactory.create_graph(
        GraphType.COMMENT_REPLY,
        graph_id="factory_example_1",
        directed=True
    )
    
    # 3. 편의 함수로 그래프 생성
    graph2 = create_comment_reply_graph(
        graph_id="factory_example_2",
        directed=False  # 무방향 그래프
    )
    
    print(f"그래프 1: {graph1.graph_id}, 방향성: {graph1.directed}")
    print(f"그래프 2: {graph2.graph_id}, 방향성: {graph2.directed}")


def example_multiple_videos():
    """여러 비디오로 그래프 구축 예시"""
    print("\n=== 여러 비디오 그래프 구축 예시 ===")
    
    db_setup = YouTubeDBSetup()
    builder = GraphBuilder(db_setup)
    
    # 상위 3개 비디오로 통합 그래프 구축
    video_ids = db_setup.get_unique_video_ids()
    if len(video_ids) < 3:
        print("비디오가 3개 미만입니다.")
        return
    
    selected_videos = video_ids[:3]
    print(f"선택된 비디오들: {selected_videos}")
    
    # 통합 그래프 구축
    combined_graph = builder.build_comment_reply_graph_by_videos(
        video_ids=selected_videos,
        graph_id="multi_video_example"
    )
    
    stats = combined_graph.get_statistics()
    print(f"통합 그래프 통계: {stats}")
    
    # 가장 많이 대댓글을 받은 작성자들
    most_replied = combined_graph.get_most_replied_authors(limit=5)
    print(f"가장 많이 대댓글을 받은 작성자: {most_replied}")


def example_all_comments_graph():
    """DB의 모든 댓글로 그래프 구축 예시"""
    print("\n=== 모든 댓글 그래프 구축 예시 ===")
    
    db_setup = YouTubeDBSetup()
    
    # 전체 댓글로 그래프 구축
    all_comments_graph = create_comment_reply_graph(graph_id="all_comments_graph")
    success = all_comments_graph.build_from_all_comments(db_setup)
    
    if not success:
        print("모든 댓글로 그래프 구축 실패")
        return
    
    # 그래프 통계 출력
    stats = all_comments_graph.get_statistics()
    print(f"전체 댓글 그래프 통계: {stats}")
    
    # 상위 대댓글 작성자들
    top_repliers = all_comments_graph.get_top_repliers(limit=5)
    print(f"상위 대댓글 작성자: {top_repliers}")
    
    # 혐오표현 네트워크 분석
    hate_stats = all_comments_graph.get_hate_speech_network_stats()
    print(f"전체 댓글 혐오표현 네트워크 통계: {hate_stats}")
    
    # 가장 많이 대댓글을 받은 작성자들
    most_replied = all_comments_graph.get_most_replied_authors(limit=5)
    print(f"가장 많이 대댓글을 받은 작성자: {most_replied}")
    
    # 그래프 저장
    filepath = "./all_comments_graph.gexf"
    save_success = all_comments_graph.save_to_file(filepath, format='gexf')
    if save_success:
        print(f"전체 댓글 그래프가 {filepath}에 저장되었습니다.")
    else:
        print("그래프 저장 실패")


def example_hate_speech_analysis():
    """혐오표현 네트워크 분석 예시"""
    print("\n=== 혐오표현 네트워크 분석 예시 ===")
    
    db_setup = YouTubeDBSetup()
    
    # 샘플 그래프들 생성
    builder = GraphBuilder(db_setup)
    sample_graphs = builder.build_sample_graphs(sample_size=3)
    
    if not sample_graphs:
        print("샘플 그래프 생성 실패")
        return
    
    # 각 그래프별 혐오표현 분석
    for graph_name, graph in sample_graphs.items():
        print(f"\n--- {graph_name} 분석 ---")
        
        # 기본 통계
        stats = graph.get_statistics()
        print(f"노드 수: {stats['node_count']}, 엣지 수: {stats['edge_count']}")
        
        # 혐오표현 통계
        hate_stats = graph.get_hate_speech_network_stats()
        print(f"혐오표현 작성자: {hate_stats['hate_authors_count']}명")
        print(f"평균 혐오표현 비율: {hate_stats['avg_hate_ratio_per_author']:.2%}")
        
        # 혐오표현 클러스터 찾기
        clusters = graph.find_hate_speech_clusters(min_hate_ratio=0.2)
        if clusters:
            print(f"혐오표현 클러스터 {len(clusters)}개 발견:")
            for i, cluster in enumerate(clusters):
                print(f"  클러스터 {i+1}: {cluster}")
        else:
            print("혐오표현 클러스터 없음")


def example_graph_persistence():
    """그래프 저장/로드 예시"""
    print("\n=== 그래프 저장/로드 예시 ===")
    
    db_setup = YouTubeDBSetup()
    video_ids = db_setup.get_unique_video_ids()
    
    if not video_ids:
        print("사용할 수 있는 비디오가 없습니다.")
        return
    
    # 그래프 생성 및 구축
    graph = build_graph_from_video(
        db_setup=db_setup,
        video_id=video_ids[0],
        graph_id="persistence_example"
    )
    
    if graph.get_node_count() == 0:
        print("빈 그래프입니다.")
        return
    
    print(f"원본 그래프 - 노드: {graph.get_node_count()}, 엣지: {graph.get_edge_count()}")
    
    # 파일로 저장
    filepath = f"./{video_ids[0]}.gexf"
    success = graph.save_to_file(filepath, format='gexf')
    if success:
        print(f"그래프가 {filepath}에 저장되었습니다.")
        
        # 새 그래프 생성 후 로드
        new_graph = create_comment_reply_graph(graph_id="loaded_graph")
        load_success = new_graph.load_from_file(filepath, format='gexf')
        
        if load_success:
            print(f"로드된 그래프 - 노드: {new_graph.get_node_count()}, 엣지: {new_graph.get_edge_count()}")
        else:
            print("그래프 로드 실패")
    else:
        print("그래프 저장 실패")


def example_custom_analysis():
    """커스텀 분석 예시"""
    print("\n=== 커스텀 분석 예시 ===")
    
    db_setup = YouTubeDBSetup()
    builder = GraphBuilder(db_setup)
    
    # 샘플 그래프 하나 생성
    video_ids = db_setup.get_unique_video_ids()
    if not video_ids:
        return
    
    graph = builder.build_comment_reply_graph_by_video(video_ids[0])
    
    # NetworkX 그래프로 변환하여 고급 분석
    nx_graph = graph.to_networkx()
    
    print(f"NetworkX 그래프 정보:")
    print(f"  노드 수: {nx_graph.number_of_nodes()}")
    print(f"  엣지 수: {nx_graph.number_of_edges()}")
    
    if nx_graph.number_of_nodes() > 0:
        # 중심성 분석 예시 (networkx 사용)
        try:
            import networkx as nx
            
            # 차수 중심성 (Degree Centrality)
            degree_centrality = nx.degree_centrality(nx_graph.to_undirected())
            top_central_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"상위 중심성 노드: {top_central_nodes}")
            
            # 밀도 (Density)
            density = nx.density(nx_graph)
            print(f"그래프 밀도: {density:.4f}")
            
        except ImportError:
            print("NetworkX가 설치되지 않아 고급 분석을 수행할 수 없습니다.")


if __name__ == "__main__":
    # 모든 예시 실행
    try:
        # example_basic_usage()
        # example_factory_usage()
        # example_multiple_videos()
        example_all_comments_graph()
        # example_hate_speech_analysis()
        # example_graph_persistence()
        # example_custom_analysis()
        
        print("\n=== 모든 예시 완료 ===")
        
    except Exception as e:
        print(f"예시 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()