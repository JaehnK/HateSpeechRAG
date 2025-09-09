from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class Node:
    """그래프 노드를 표현하는 기본 클래스"""
    id: str
    node_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("Node ID는 필수입니다")
        if not self.node_type:
            raise ValueError("Node type은 필수입니다")
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)
    
    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'node_type': self.node_type,
            **self.attributes
        }


@dataclass
class Edge:
    """그래프 엣지를 표현하는 기본 클래스"""
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.source or not self.target:
            raise ValueError("Source와 Target은 필수입니다")
        if not self.edge_type:
            raise ValueError("Edge type은 필수입니다")
        if self.weight <= 0:
            raise ValueError("Weight는 양수여야 합니다")
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)
    
    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'weight': self.weight,
            **self.attributes
        }


class BaseGraph(ABC):
    """모든 그래프 클래스의 기본 추상 클래스"""
    
    def __init__(self, graph_id: str, directed: bool = True):
        self.graph_id = graph_id
        self.directed = directed
        self._graph = nx.DiGraph() if directed else nx.Graph()
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_node(self, node: Node) -> bool:
        """노드 추가"""
        try:
            if node.id in self._nodes:
                # 기존 노드 업데이트
                self._nodes[node.id] = node
                self._graph.add_node(node.id, **node.to_dict())
                return True
            
            self._nodes[node.id] = node
            self._graph.add_node(node.id, **node.to_dict())
            return True
        except Exception as e:
            print(f"노드 추가 실패 ({node.id}): {e}")
            return False
    
    def add_edge(self, edge: Edge) -> bool:
        """엣지 추가"""
        try:
            # 소스와 타겟 노드가 존재하는지 확인
            if edge.source not in self._nodes:
                raise ValueError(f"소스 노드 {edge.source}가 존재하지 않습니다")
            if edge.target not in self._nodes:
                raise ValueError(f"타겟 노드 {edge.target}가 존재하지 않습니다")
            
            self._edges.append(edge)
            self._graph.add_edge(edge.source, edge.target, **edge.to_dict())
            return True
        except Exception as e:
            print(f"엣지 추가 실패 ({edge.source} -> {edge.target}): {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """노드 조회"""
        return self._nodes.get(node_id)
    
    def get_edges_by_type(self, edge_type: str) -> List[Edge]:
        """특정 타입의 엣지들 조회"""
        return [edge for edge in self._edges if edge.edge_type == edge_type]
    
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """노드의 이웃 노드들 조회"""
        if node_id not in self._graph:
            return []
        
        neighbors = list(self._graph.neighbors(node_id))
        
        if edge_type:
            # 특정 엣지 타입으로 필터링
            filtered_neighbors = []
            for neighbor in neighbors:
                edge_data = self._graph.get_edge_data(node_id, neighbor)
                if edge_data and edge_data.get('edge_type') == edge_type:
                    filtered_neighbors.append(neighbor)
            return filtered_neighbors
        
        return neighbors
    
    def get_node_count(self) -> int:
        """노드 개수 반환"""
        return len(self._nodes)
    
    def get_edge_count(self) -> int:
        """엣지 개수 반환"""
        return len(self._edges)
    
    def get_statistics(self) -> Dict[str, Any]:
        """그래프 통계 정보"""
        stats = {
            'graph_id': self.graph_id,
            'directed': self.directed,
            'node_count': self.get_node_count(),
            'edge_count': self.get_edge_count(),
            'node_types': {},
            'edge_types': {}
        }
        
        # 노드 타입별 개수
        for node in self._nodes.values():
            node_type = node.node_type
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # 엣지 타입별 개수
        for edge in self._edges:
            edge_type = edge.edge_type
            stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
        return stats
    
    @abstractmethod
    def build_graph(self, data: List[Dict[str, Any]]) -> bool:
        """데이터로부터 그래프 구축 (하위 클래스에서 구현)"""
        pass
    
    def clear(self) -> None:
        """그래프 초기화"""
        self._graph.clear()
        self._nodes.clear()
        self._edges.clear()
        self.metadata.clear()
    
    def to_networkx(self) -> nx.Graph:
        """NetworkX 그래프 객체 반환"""
        return self._graph.copy()
    
    def save_to_file(self, filepath: str, format: str = 'gexf') -> bool:
        """그래프를 파일로 저장"""
        try:
            if format == 'gexf':
                nx.write_gexf(self._graph, filepath)
            elif format == 'gml':
                nx.write_gml(self._graph, filepath)
            elif format == 'graphml':
                nx.write_graphml(self._graph, filepath)
            else:
                raise ValueError(f"지원하지 않는 형식: {format}")
            return True
        except Exception as e:
            print(f"그래프 저장 실패: {e}")
            return False
    
    def load_from_file(self, filepath: str, format: str = 'gexf') -> bool:
        """파일에서 그래프 로드"""
        try:
            if format == 'gexf':
                loaded_graph = nx.read_gexf(filepath)
            elif format == 'gml':
                loaded_graph = nx.read_gml(filepath)
            elif format == 'graphml':
                loaded_graph = nx.read_graphml(filepath)
            else:
                raise ValueError(f"지원하지 않는 형식: {format}")
            
            # 기존 그래프 초기화 후 로드된 그래프 적용
            self.clear()
            self._graph = loaded_graph
            
            # 노드와 엣지 정보 재구축
            for node_id, node_data in self._graph.nodes(data=True):
                node = Node(
                    id=node_id,
                    node_type=node_data.get('node_type', 'unknown'),
                    attributes={k: v for k, v in node_data.items() if k not in ['id', 'node_type']}
                )
                self._nodes[node_id] = node
            
            for source, target, edge_data in self._graph.edges(data=True):
                edge = Edge(
                    source=source,
                    target=target,
                    edge_type=edge_data.get('edge_type', 'unknown'),
                    weight=edge_data.get('weight', 1.0),
                    attributes={k: v for k, v in edge_data.items() 
                              if k not in ['source', 'target', 'edge_type', 'weight']}
                )
                self._edges.append(edge)
            
            return True
        except Exception as e:
            print(f"그래프 로드 실패: {e}")
            return False