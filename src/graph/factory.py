from typing import Dict, Any, List, Optional, Type
from enum import Enum
from ..dao.YouTubeDao import YouTubeDBSetup
from .base import BaseGraph
from .comment_reply_graph import CommentReplyGraph


class GraphType(Enum):
    """지원하는 그래프 타입들"""
    COMMENT_REPLY = "comment_reply"
    # 향후 추가될 그래프 타입들
    # AUTHOR_SIMILARITY = "author_similarity"
    # VIDEO_INTERACTION = "video_interaction" 
    # HATE_SPEECH_PROPAGATION = "hate_propagation"


class GraphFactory:
    """그래프 객체 생성을 위한 팩토리 클래스"""
    
    _graph_classes: Dict[GraphType, Type[BaseGraph]] = {
        GraphType.COMMENT_REPLY: CommentReplyGraph,
    }
    
    @classmethod
    def create_graph(cls, graph_type: GraphType, graph_id: Optional[str] = None, **kwargs) -> BaseGraph:
        """
        그래프 타입에 따라 적절한 그래프 객체 생성
        
        Args:
            graph_type: 생성할 그래프 타입
            graph_id: 그래프 고유 식별자 (None시 기본값 사용)
            **kwargs: 그래프별 추가 파라미터
        
        Returns:
            생성된 그래프 객체
        """
        if graph_type not in cls._graph_classes:
            raise ValueError(f"지원하지 않는 그래프 타입: {graph_type}")
        
        graph_class = cls._graph_classes[graph_type]
        
        # graph_id가 없으면 기본값 사용
        if graph_id is None:
            if hasattr(graph_class, '__init__'):
                # 임시 객체를 생성해서 기본 graph_id 확인
                temp_obj = graph_class()
                graph_id = temp_obj.graph_id
                del temp_obj
        
        return graph_class(graph_id=graph_id, **kwargs)
    
    @classmethod
    def create_comment_reply_graph(cls, graph_id: Optional[str] = None, directed: bool = True) -> CommentReplyGraph:
        """댓글-대댓글 관계 그래프 생성"""
        return cls.create_graph(GraphType.COMMENT_REPLY, graph_id, directed=directed)
    
    @classmethod
    def register_graph_type(cls, graph_type: GraphType, graph_class: Type[BaseGraph]) -> None:
        """새로운 그래프 타입 등록 (확장성을 위한 메서드)"""
        if not issubclass(graph_class, BaseGraph):
            raise ValueError("그래프 클래스는 BaseGraph를 상속해야 합니다")
        cls._graph_classes[graph_type] = graph_class
    
    @classmethod
    def get_available_types(cls) -> List[GraphType]:
        """사용 가능한 그래프 타입 목록 반환"""
        return list(cls._graph_classes.keys())


class GraphBuilder:
    """데이터베이스와 연동하여 그래프를 구축하는 빌더 클래스"""
    
    def __init__(self, db_setup: YouTubeDBSetup):
        self.db_setup = db_setup
    
    def build_comment_reply_graph_by_video(self, video_id: str, graph_id: Optional[str] = None) -> CommentReplyGraph:
        """특정 비디오의 댓글로 댓글-대댓글 그래프 구축"""
        graph = GraphFactory.create_comment_reply_graph(graph_id=graph_id)
        
        success = graph.build_from_video_id(self.db_setup, video_id)
        if not success:
            raise RuntimeError(f"비디오 {video_id}로 그래프 구축 실패")
        
        return graph
    
    def build_comment_reply_graph_by_videos(self, video_ids: List[str], graph_id: Optional[str] = None) -> CommentReplyGraph:
        """여러 비디오의 댓글로 댓글-대댓글 그래프 구축"""
        graph = GraphFactory.create_comment_reply_graph(graph_id=graph_id)
        
        success = graph.build_from_multiple_videos(self.db_setup, video_ids)
        if not success:
            raise RuntimeError(f"비디오들 {video_ids}로 그래프 구축 실패")
        
        return graph
    
    def build_comment_reply_graph_from_data(self, comments_data: List[Dict[str, Any]], graph_id: Optional[str] = None) -> CommentReplyGraph:
        """직접 제공된 댓글 데이터로 댓글-대댓글 그래프 구축"""
        graph = GraphFactory.create_comment_reply_graph(graph_id=graph_id)
        
        success = graph.build_graph(comments_data)
        if not success:
            raise RuntimeError("제공된 댓글 데이터로 그래프 구축 실패")
        
        return graph
    
    def build_sample_graphs(self, sample_size: int = 5) -> Dict[str, BaseGraph]:
        """샘플 그래프들 생성 (테스트/데모용)"""
        try:
            # 비디오 ID 목록 조회
            video_ids = self.db_setup.get_unique_video_ids()
            if not video_ids:
                raise RuntimeError("데이터베이스에 비디오가 없습니다")
            
            # 샘플 비디오들 선택
            sample_video_ids = video_ids[:min(sample_size, len(video_ids))]
            
            graphs = {}
            
            # 각 비디오별로 개별 그래프 생성
            for i, video_id in enumerate(sample_video_ids):
                try:
                    graph = self.build_comment_reply_graph_by_video(
                        video_id, 
                        graph_id=f"sample_video_{i+1}_{video_id}"
                    )
                    if graph.get_node_count() > 0:  # 노드가 있는 경우만 추가
                        graphs[f"video_{video_id}"] = graph
                except Exception as e:
                    print(f"비디오 {video_id} 그래프 생성 실패: {e}")
            
            # 전체 샘플 통합 그래프 생성
            try:
                combined_graph = self.build_comment_reply_graph_by_videos(
                    sample_video_ids, 
                    graph_id=f"combined_sample_{len(sample_video_ids)}_videos"
                )
                if combined_graph.get_node_count() > 0:
                    graphs["combined_sample"] = combined_graph
            except Exception as e:
                print(f"통합 샘플 그래프 생성 실패: {e}")
            
            return graphs
            
        except Exception as e:
            print(f"샘플 그래프 생성 실패: {e}")
            return {}


# 편의 함수들
def create_comment_reply_graph(graph_id: str = None, directed: bool = True) -> CommentReplyGraph:
    """댓글-대댓글 그래프 생성 편의 함수"""
    return GraphFactory.create_comment_reply_graph(graph_id, directed)


def build_graph_from_video(db_setup: YouTubeDBSetup, video_id: str, graph_id: str = None) -> CommentReplyGraph:
    """비디오 ID로 그래프 구축 편의 함수"""
    builder = GraphBuilder(db_setup)
    return builder.build_comment_reply_graph_by_video(video_id, graph_id)


def build_graph_from_videos(db_setup: YouTubeDBSetup, video_ids: List[str], graph_id: str = None) -> CommentReplyGraph:
    """여러 비디오 ID들로 그래프 구축 편의 함수"""
    builder = GraphBuilder(db_setup)
    return builder.build_comment_reply_graph_by_videos(video_ids, graph_id)