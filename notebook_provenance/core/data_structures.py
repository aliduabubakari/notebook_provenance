"""
Data Structures Module
======================

Core data structures for representing provenance information.

This module defines:
- DFGNode: Nodes in the data flow graph
- DFGEdge: Edges in the data flow graph
- DataFlowGraph: The complete data flow graph
- CellDependency: Dependencies between notebook cells
- DataArtifact: Important data objects
- Transformation: Transformations between artifacts
- PipelineStageNode: High-level pipeline stages
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Any, Optional
import networkx as nx

from notebook_provenance.core.enums import (
    NodeType,
    EdgeType,
    TaskType,
    PipelineStage,
    ArtifactType,
)


@dataclass
class DFGNode:
    """
    A node in the data flow graph.
    
    Attributes:
        id: Unique identifier for the node
        label: Human-readable label (e.g., variable name, function name)
        node_type: Type of the node (from NodeType enum)
        code_snippet: Code snippet this node represents
        line_number: Line number in the source code
        cell_id: ID of the cell containing this node
        embedding: Optional vector embedding for the node
        task_type: Optional task type classification
        metadata: Additional metadata
    """
    id: str
    label: str
    node_type: NodeType
    code_snippet: str = ""
    line_number: int = 0
    cell_id: str = ""
    embedding: Optional[List[float]] = None
    task_type: Optional[TaskType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize node data"""
        if isinstance(self.node_type, str):
            self.node_type = NodeType(self.node_type)
        if self.task_type is not None and isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = asdict(self)
        data['node_type'] = self.node_type.value
        if self.task_type:
            data['task_type'] = self.task_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DFGNode':
        """Create node from dictionary"""
        return cls(**data)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, DFGNode):
            return False
        return self.id == other.id


@dataclass
class DFGEdge:
    """
    An edge in the data flow graph.
    
    Attributes:
        from_node: ID of the source node
        to_node: ID of the target node
        edge_type: Type of the edge (from EdgeType enum)
        weight: Weight of the edge (default 1.0)
        operation: Description of the operation
        metadata: Additional metadata
    """
    from_node: str
    to_node: str
    edge_type: EdgeType
    weight: float = 1.0
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize edge data"""
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType(self.edge_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = asdict(self)
        data['edge_type'] = self.edge_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DFGEdge':
        """Create edge from dictionary"""
        return cls(**data)
    
    def __hash__(self):
        return hash((self.from_node, self.to_node, self.edge_type))
    
    def __eq__(self, other):
        if not isinstance(other, DFGEdge):
            return False
        return (self.from_node == other.from_node and 
                self.to_node == other.to_node and 
                self.edge_type == other.edge_type)


@dataclass
class DataFlowGraph:
    """
    Complete data flow graph representation.
    
    Attributes:
        nodes: Dictionary mapping node IDs to DFGNode objects
        edges: List of DFGEdge objects
        metadata: Additional graph-level metadata
    """
    nodes: Dict[str, DFGNode] = field(default_factory=dict)
    edges: List[DFGEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: DFGNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: DFGEdge) -> None:
        """Add an edge to the graph"""
        if edge.from_node not in self.nodes:
            raise ValueError(f"Source node {edge.from_node} not in graph")
        if edge.to_node not in self.nodes:
            raise ValueError(f"Target node {edge.to_node} not in graph")
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[DFGNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_edges_from(self, node_id: str) -> List[DFGEdge]:
        """Get all edges originating from a node"""
        return [e for e in self.edges if e.from_node == node_id]
    
    def get_edges_to(self, node_id: str) -> List[DFGEdge]:
        """Get all edges pointing to a node"""
        return [e for e in self.edges if e.to_node == node_id]
    
    def get_neighbors(self, node_id: str, direction: str = 'out') -> List[str]:
        """
        Get neighbor node IDs.
        
        Args:
            node_id: The node to get neighbors for
            direction: 'out' for outgoing, 'in' for incoming, 'both' for both
        """
        neighbors = []
        if direction in ['out', 'both']:
            neighbors.extend([e.to_node for e in self.get_edges_from(node_id)])
        if direction in ['in', 'both']:
            neighbors.extend([e.from_node for e in self.get_edges_to(node_id)])
        return list(set(neighbors))
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph"""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.to_dict())
        
        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(edge.from_node, edge.to_node, **edge.to_dict())
        
        return G
    
    @classmethod
    def from_networkx(cls, G: nx.DiGraph) -> 'DataFlowGraph':
        """Create DataFlowGraph from NetworkX graph"""
        dfg = cls()
        
        # Add nodes
        for node_id, data in G.nodes(data=True):
            node = DFGNode.from_dict(data)
            dfg.add_node(node)
        
        # Add edges
        for from_node, to_node, data in G.edges(data=True):
            edge = DFGEdge.from_dict(data)
            dfg.add_edge(edge)
        
        return dfg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataFlowGraph':
        """Create DataFlowGraph from dictionary"""
        dfg = cls()
        dfg.metadata = data.get('metadata', {})
        
        # Add nodes
        for node_id, node_data in data.get('nodes', {}).items():
            node = DFGNode.from_dict(node_data)
            dfg.nodes[node_id] = node
        
        # Add edges
        for edge_data in data.get('edges', []):
            edge = DFGEdge.from_dict(edge_data)
            dfg.edges.append(edge)
        
        return dfg
    
    def filter_by_node_type(self, node_types: Set[NodeType]) -> 'DataFlowGraph':
        """Create a filtered graph containing only specified node types"""
        filtered = DataFlowGraph()
        filtered.metadata = self.metadata.copy()
        
        # Add filtered nodes
        for node_id, node in self.nodes.items():
            if node.node_type in node_types:
                filtered.add_node(node)
        
        # Add edges between filtered nodes
        for edge in self.edges:
            if edge.from_node in filtered.nodes and edge.to_node in filtered.nodes:
                filtered.add_edge(edge)
        
        return filtered
    
    def __len__(self) -> int:
        """Return number of nodes"""
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if node exists in graph"""
        return node_id in self.nodes


@dataclass
class CellDependency:
    """
    Track dependencies between notebook cells.
    
    Attributes:
        cell_id: Unique identifier for the cell
        depends_on: Set of cell IDs this cell depends on
        produces: Set of variable names produced by this cell
        consumes: Set of variable names consumed by this cell
        imports: Set of modules imported by this cell
    """
    cell_id: str
    depends_on: Set[str] = field(default_factory=set)
    produces: Set[str] = field(default_factory=set)
    consumes: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'cell_id': self.cell_id,
            'depends_on': list(self.depends_on),
            'produces': list(self.produces),
            'consumes': list(self.consumes),
            'imports': list(self.imports),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CellDependency':
        """Create from dictionary"""
        return cls(
            cell_id=data['cell_id'],
            depends_on=set(data.get('depends_on', [])),
            produces=set(data.get('produces', [])),
            consumes=set(data.get('consumes', [])),
            imports=set(data.get('imports', [])),
        )


@dataclass
class DataArtifact:
    """
    Represents a key data object in the pipeline.
    
    Attributes:
        id: Unique identifier
        name: Variable name
        type: Type of artifact (from ArtifactType enum)
        created_in_cell: Cell ID where artifact was created
        transformations: List of transformation IDs applied to this artifact
        schema_info: Optional schema information
        importance_score: Computed importance score
        metadata: Additional metadata
    """
    id: str
    name: str
    type: str
    created_in_cell: str
    transformations: List[str] = field(default_factory=list)
    schema_info: Optional[Dict] = None
    importance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataArtifact':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class Transformation:
    """
    Represents a transformation between artifacts.
    
    Attributes:
        id: Unique identifier
        operation: Name of the transformation operation
        source_artifacts: List of source artifact IDs
        target_artifact: Target artifact ID
        function_calls: List of function calls involved
        cell_id: Cell where transformation occurs
        description: Human-readable description
        semantic_type: Semantic type of transformation
    """
    id: str
    operation: str
    source_artifacts: List[str]
    target_artifact: str
    function_calls: List[str]
    cell_id: str
    description: str = ""
    semantic_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transformation':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class PipelineStageNode:
    """
    High-level pipeline stage.
    
    Attributes:
        id: Unique identifier
        stage_type: Type of stage (from PipelineStage enum)
        cells: List of cell IDs in this stage
        operations: List of operations performed
        input_artifacts: List of input artifact IDs
        output_artifacts: List of output artifact IDs
        description: Human-readable description
        confidence: Confidence score for stage classification
    """
    id: str
    stage_type: PipelineStage
    cells: List[str]
    operations: List[str]
    input_artifacts: List[str]
    output_artifacts: List[str]
    description: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate and normalize stage data"""
        if isinstance(self.stage_type, str):
            self.stage_type = PipelineStage(self.stage_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = asdict(self)
        data['stage_type'] = self.stage_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineStageNode':
        """Create from dictionary"""
        return cls(**data)


# Export all data structures
__all__ = [
    "DFGNode",
    "DFGEdge",
    "DataFlowGraph",
    "CellDependency",
    "DataArtifact",
    "Transformation",
    "PipelineStageNode",
]