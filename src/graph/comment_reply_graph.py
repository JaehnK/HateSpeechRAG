from typing import Dict, List, Any, Optional, Set
from ..dao.YouTubeDao import YouTubeDBSetup
from .base import BaseGraph, Node, Edge


class CommentAuthorNode(Node):
    """댓글 작성자 노드"""
    
    def __init__(self, author: str, author_channel_id: str = None, **kwargs):
        attributes = {
            'author_channel_id': author_channel_id,
            'comment_count': 0,
            'reply_count': 0,
            'total_likes': 0,
            'hate_speech_count': 0,
            'is_hate_author': False,
            **kwargs
        }
        super().__init__(id=author, node_type='author', attributes=attributes)
    
    def add_comment_stats(self, like_count: int = 0, is_reply: bool = False, is_hate_speech: bool = False):
        """댓글 통계 업데이트"""
        if is_reply:
            self.attributes['reply_count'] += 1
        else:
            self.attributes['comment_count'] += 1
        
        self.attributes['total_likes'] += like_count or 0
        
        if is_hate_speech:
            self.attributes['hate_speech_count'] += 1
            
        # 혐오표현 작성자 여부 판단 (혐오표현 댓글이 1개 이상)
        self.attributes['is_hate_author'] = self.attributes['hate_speech_count'] > 0
    
    def get_total_comments(self) -> int:
        """전체 댓글 수 (댓글 + 대댓글)"""
        return self.attributes['comment_count'] + self.attributes['reply_count']
    
    def get_hate_speech_ratio(self) -> float:
        """혐오표현 비율"""
        total = self.get_total_comments()
        return self.attributes['hate_speech_count'] / total if total > 0 else 0.0


class ReplyEdge(Edge):
    """대댓글 관계를 나타내는 엣지"""
    
    def __init__(self, replier: str, replied_to: str, **kwargs):
        attributes = {
            'reply_count': 1,
            'latest_reply_time': kwargs.get('published_at'),
            'avg_like_count': kwargs.get('like_count', 0),
            'hate_reply_count': 0,
            **{k: v for k, v in kwargs.items() if k not in ['published_at', 'like_count']}
        }
        super().__init__(
            source=replier, 
            target=replied_to, 
            edge_type='reply_to',
            weight=1.0,
            attributes=attributes
        )
    
    def add_reply(self, like_count: int = 0, published_at: str = None, is_hate_speech: bool = False):
        """추가 대댓글 정보 업데이트"""
        self.attributes['reply_count'] += 1
        
        # 평균 좋아요 수 업데이트
        current_avg = self.attributes['avg_like_count']
        current_count = self.attributes['reply_count']
        self.attributes['avg_like_count'] = ((current_avg * (current_count - 1)) + like_count) / current_count
        
        # 최신 대댓글 시간 업데이트
        if published_at and (not self.attributes['latest_reply_time'] or published_at > self.attributes['latest_reply_time']):
            self.attributes['latest_reply_time'] = published_at
        
        # 혐오표현 대댓글 카운트
        if is_hate_speech:
            self.attributes['hate_reply_count'] += 1
        
        # 가중치 업데이트 (대댓글 수에 비례)
        self.weight = float(self.attributes['reply_count'])
    
    def get_hate_reply_ratio(self) -> float:
        """혐오표현 대댓글 비율"""
        total = self.attributes['reply_count']
        return self.attributes['hate_reply_count'] / total if total > 0 else 0.0


class CommentReplyGraph(BaseGraph):
    """댓글 작성자간 대댓글 관계를 표현하는 그래프"""
    
    def __init__(self, graph_id: str = "comment_reply_network", directed: bool = True):
        super().__init__(graph_id, directed)
        self.metadata.update({
            'graph_type': 'comment_reply',
            'description': '댓글 작성자간 대댓글 관계 네트워크'
        })
    
    def build_graph(self, data: List[Dict[str, Any]]) -> bool:
        """댓글 데이터로부터 그래프 구축"""
        try:
            self.clear()
            
            # 1단계: 모든 작성자를 노드로 추가
            authors_data = {}  # author -> 댓글 정보 리스트
            parent_authors = {}  # parent_id -> author
            
            # 데이터 전처리
            for comment in data:
                author = comment.get('author')
                if not author:
                    continue
                
                comment_id = comment.get('comment_id')
                is_reply = comment.get('is_reply', False)
                parent_id = comment.get('parent_id')
                
                # 작성자 정보 수집
                if author not in authors_data:
                    authors_data[author] = []
                authors_data[author].append(comment)
                
                # 부모 댓글 작성자 매핑 (대댓글이 아닌 경우에만)
                if not is_reply and comment_id:
                    parent_authors[comment_id] = author
            
            # 2단계: 작성자 노드 생성
            for author, comments in authors_data.items():
                # 첫 번째 댓글에서 channel_id 가져오기
                author_channel_id = next((c.get('author_channel_id') for c in comments), None)
                
                node = CommentAuthorNode(author=author, author_channel_id=author_channel_id)
                
                # 댓글별 통계 추가
                for comment in comments:
                    is_reply = comment.get('is_reply', False)
                    like_count = comment.get('like_count', 0)
                    is_hate_speech = comment.get('is_hate_speech', False)
                    
                    node.add_comment_stats(
                        like_count=like_count,
                        is_reply=is_reply,
                        is_hate_speech=is_hate_speech
                    )
                
                self.add_node(node)
            
            # 3단계: 대댓글 관계 엣지 생성
            reply_relationships = {}  # (replier, replied_to) -> 댓글 리스트
            
            for comment in data:
                is_reply = comment.get('is_reply', False)
                if not is_reply:
                    continue
                
                replier = comment.get('author')
                parent_id = comment.get('parent_id')
                
                if not replier or not parent_id:
                    continue
                
                # 부모 댓글의 작성자 찾기
                replied_to = parent_authors.get(parent_id)
                if not replied_to or replier == replied_to:  # 자기 자신에게 대댓글은 제외
                    continue
                
                relationship_key = (replier, replied_to)
                if relationship_key not in reply_relationships:
                    reply_relationships[relationship_key] = []
                reply_relationships[relationship_key].append(comment)
            
            # 4단계: 엣지 추가
            for (replier, replied_to), replies in reply_relationships.items():
                # 첫 번째 대댓글로 엣지 생성
                first_reply = replies[0]
                edge = ReplyEdge(
                    replier=replier,
                    replied_to=replied_to,
                    like_count=first_reply.get('like_count', 0),
                    published_at=first_reply.get('published_at'),
                    is_hate_speech=first_reply.get('is_hate_speech', False)
                )
                
                # 나머지 대댓글들 추가
                for reply in replies[1:]:
                    edge.add_reply(
                        like_count=reply.get('like_count', 0),
                        published_at=reply.get('published_at'),
                        is_hate_speech=reply.get('is_hate_speech', False)
                    )
                
                self.add_edge(edge)
            
            # 메타데이터 업데이트
            self.metadata.update({
                'build_time': str(__import__('datetime').datetime.now()),
                'total_comments_processed': len(data),
                'unique_authors': len(authors_data),
                'reply_relationships': len(reply_relationships)
            })
            
            return True
            
        except Exception as e:
            print(f"그래프 구축 실패: {e}")
            return False
    
    def build_from_video_id(self, db_setup: YouTubeDBSetup, video_id: str) -> bool:
        """특정 비디오의 댓글로 그래프 구축"""
        try:
            comments = db_setup.get_comments_by_video_id(video_id, verbose=False)
            if not comments:
                print(f"비디오 {video_id}에 대한 댓글이 없습니다.")
                return False
            
            result = self.build_graph(comments)
            if result:
                self.metadata['source_video_id'] = video_id
            return result
            
        except Exception as e:
            print(f"비디오별 그래프 구축 실패: {e}")
            return False
    
    def build_from_multiple_videos(self, db_setup: YouTubeDBSetup, video_ids: List[str]) -> bool:
        """여러 비디오의 댓글로 그래프 구축"""
        try:
            all_comments = []
            processed_videos = []
            
            for video_id in video_ids:
                comments = db_setup.get_comments_by_video_id(video_id, verbose=False)
                if comments:
                    all_comments.extend(comments)
                    processed_videos.append(video_id)
            
            if not all_comments:
                print("처리할 댓글이 없습니다.")
                return False
            
            result = self.build_graph(all_comments)
            if result:
                self.metadata['source_video_ids'] = processed_videos
                self.metadata['total_videos'] = len(processed_videos)
            return result
            
        except Exception as e:
            print(f"다중 비디오 그래프 구축 실패: {e}")
            return False
    
    def build_from_all_comments(self, db_setup: YouTubeDBSetup) -> bool:
        """DB의 모든 댓글로 그래프 구축"""
        try:
            all_comments = db_setup.get_all_comments(verbose=False)
            
            if not all_comments:
                print("처리할 댓글이 없습니다.")
                return False
            
            result = self.build_graph(all_comments)
            if result:
                self.metadata['source_type'] = 'all_comments'
                self.metadata['total_comments'] = len(all_comments)
            return result
            
        except Exception as e:
            print(f"전체 댓글 그래프 구축 실패: {e}")
            return False
    
    def get_top_repliers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """가장 많이 대댓글을 작성한 사용자들"""
        replier_stats = {}
        
        for edge in self._edges:
            if edge.edge_type != 'reply_to':
                continue
            
            replier = edge.source
            reply_count = edge.attributes.get('reply_count', 1)
            
            if replier not in replier_stats:
                replier_stats[replier] = {
                    'author': replier,
                    'total_replies': 0,
                    'replied_to_count': 0,
                    'hate_replies': 0
                }
            
            replier_stats[replier]['total_replies'] += reply_count
            replier_stats[replier]['replied_to_count'] += 1
            replier_stats[replier]['hate_replies'] += edge.attributes.get('hate_reply_count', 0)
        
        # 정렬 후 상위 반환
        sorted_repliers = sorted(replier_stats.values(), key=lambda x: x['total_replies'], reverse=True)
        return sorted_repliers[:limit]
    
    def get_most_replied_authors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """가장 많이 대댓글을 받은 사용자들"""
        target_stats = {}
        
        for edge in self._edges:
            if edge.edge_type != 'reply_to':
                continue
            
            target = edge.target
            reply_count = edge.attributes.get('reply_count', 1)
            
            if target not in target_stats:
                target_stats[target] = {
                    'author': target,
                    'received_replies': 0,
                    'repliers_count': 0,
                    'received_hate_replies': 0
                }
            
            target_stats[target]['received_replies'] += reply_count
            target_stats[target]['repliers_count'] += 1
            target_stats[target]['received_hate_replies'] += edge.attributes.get('hate_reply_count', 0)
        
        # 정렬 후 상위 반환
        sorted_targets = sorted(target_stats.values(), key=lambda x: x['received_replies'], reverse=True)
        return sorted_targets[:limit]
    
    def get_hate_speech_network_stats(self) -> Dict[str, Any]:
        """혐오표현 관련 네트워크 통계"""
        stats = {
            'hate_authors_count': 0,
            'total_hate_comments': 0,
            'total_hate_replies': 0,
            'hate_reply_relationships': 0,
            'avg_hate_ratio_per_author': 0.0
        }
        
        hate_ratios = []
        
        # 노드 기반 통계
        for node in self._nodes.values():
            if node.node_type != 'author':
                continue
            
            if node.attributes.get('is_hate_author', False):
                stats['hate_authors_count'] += 1
            
            stats['total_hate_comments'] += node.attributes.get('hate_speech_count', 0)
            hate_ratios.append(node.get_hate_speech_ratio())
        
        # 엣지 기반 통계
        for edge in self._edges:
            if edge.edge_type != 'reply_to':
                continue
            
            hate_reply_count = edge.attributes.get('hate_reply_count', 0)
            stats['total_hate_replies'] += hate_reply_count
            
            if hate_reply_count > 0:
                stats['hate_reply_relationships'] += 1
        
        # 평균 혐오표현 비율
        if hate_ratios:
            stats['avg_hate_ratio_per_author'] = sum(hate_ratios) / len(hate_ratios)
        
        return stats
    
    def find_hate_speech_clusters(self, min_hate_ratio: float = 0.3) -> List[List[str]]:
        """혐오표현 비율이 높은 작성자들의 클러스터 찾기"""
        import networkx as nx
        
        # 혐오표현 비율이 높은 노드들만 필터링
        hate_authors = set()
        for node in self._nodes.values():
            if node.node_type == 'author' and node.get_hate_speech_ratio() >= min_hate_ratio:
                hate_authors.add(node.id)
        
        if len(hate_authors) < 2:
            return []
        
        # 혐오표현 작성자들 간의 서브그래프 생성
        subgraph = self._graph.subgraph(hate_authors)
        
        # 연결된 컴포넌트 찾기
        if self.directed:
            components = list(nx.weakly_connected_components(subgraph))
        else:
            components = list(nx.connected_components(subgraph))
        
        # 2명 이상의 클러스터만 반환
        return [list(component) for component in components if len(component) >= 2]
    
    def get_node_list(self) -> List[Dict[str, Any]]:
        """그래프의 모든 노드를 리스트로 반환"""
        node_list = []
        
        for node in self._nodes.values():
            if node.node_type != 'author':
                continue
                
            node_data = {
                'id': node.id,
                'author': node.id,
                'author_channel_id': node.attributes.get('author_channel_id'),
                'comment_count': node.attributes.get('comment_count', 0),
                'reply_count': node.attributes.get('reply_count', 0),
                'total_likes': node.attributes.get('total_likes', 0),
                'hate_speech_count': node.attributes.get('hate_speech_count', 0),
                'is_hate_author': node.attributes.get('is_hate_author', False),
                'total_comments': node.get_total_comments(),
                'hate_speech_ratio': node.get_hate_speech_ratio()
            }
            node_list.append(node_data)
        
        return node_list
    
    def get_edge_list(self) -> List[Dict[str, Any]]:
        """그래프의 모든 엣지를 리스트로 반환"""
        edge_list = []
        
        for edge in self._edges:
            if edge.edge_type != 'reply_to':
                continue
                
            edge_data = {
                'source': edge.source,
                'target': edge.target,
                'weight': edge.weight,
                'reply_count': edge.attributes.get('reply_count', 1),
                'latest_reply_time': edge.attributes.get('latest_reply_time'),
                'avg_like_count': edge.attributes.get('avg_like_count', 0),
                'hate_reply_count': edge.attributes.get('hate_reply_count', 0),
                'hate_reply_ratio': edge.get_hate_reply_ratio()
            }
            edge_list.append(edge_data)
        
        return edge_list