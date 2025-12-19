"""
Artifact Analyzer Module
========================

Analyze and track data artifacts with importance scoring.

This module provides the DataArtifactAnalyzer class which:
- Identifies key data artifacts in the graph
- Computes importance scores
- Builds artifact lineage graphs
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx

from notebook_provenance.core.data_structures import (
    DataFlowGraph,
    DFGNode,
    DataArtifact,
    CellDependency,
)
from notebook_provenance.core.enums import NodeType


class DataArtifactAnalyzer:
    """
    Analyze and track data artifacts with importance scoring.
    
    This class identifies important data objects (DataFrames, tables, models, etc.)
    and computes their importance based on usage patterns and connectivity.
    
    Example:
        >>> analyzer = DataArtifactAnalyzer()
        >>> artifacts = analyzer.identify_artifacts(dfg, cell_dependencies)
        >>> lineage_graph = analyzer.build_artifact_lineage(dfg, artifacts)
    """
    
    def __init__(self, llm_analyzer=None):
        """
        Initialize the artifact analyzer.
        
        Args:
            llm_analyzer: Optional LLM analyzer for enhanced descriptions
        """
        self.llm_analyzer = llm_analyzer
        self.artifacts = {}
        
        # Patterns for identifying artifacts
        self.artifact_patterns = {
            'dataframe': {
                'names': ['df', 'data', 'dataset', '_df', '_data'],
                'functions': ['read_csv', 'DataFrame', 'read_excel', 'read_parquet'],
                'base_importance': 10
            },
            'table': {
                'names': ['table', 'tbl', '_table'],
                'functions': ['add_table', 'get_table', 'create_table'],
                'base_importance': 10
            },
            'model': {
                'names': ['model', 'classifier', 'regressor', 'estimator'],
                'functions': ['fit', 'train', 'compile'],
                'base_importance': 9
            },
            'result': {
                'names': ['result', 'output', 'prediction', 'predictions', 'score'],
                'functions': ['predict', 'transform', 'score'],
                'base_importance': 7
            },
            'matrix': {
                'names': ['matrix', 'array', 'tensor', 'X', 'y'],
                'functions': ['array', 'tensor', 'matrix'],
                'base_importance': 8
            },
        }
    
    def identify_artifacts(self, dfg: DataFlowGraph, 
                          cell_dependencies: Dict[str, CellDependency]) -> List[DataArtifact]:
        """
        Identify key data artifacts in the graph.
        
        Args:
            dfg: Data flow graph
            cell_dependencies: Cell dependency information
            
        Returns:
            List of DataArtifact objects sorted by importance
        """
        artifacts = []
        
        for node_id, node in dfg.nodes.items():
            # Only consider variables and data artifacts
            if node.node_type not in [NodeType.VARIABLE, NodeType.DATA_ARTIFACT]:
                continue
            
            # Calculate importance score
            artifact_type, importance = self._calculate_importance(node, dfg)
            
            # Threshold for important artifacts
            if importance >= 7:
                # Find which cell created this
                created_in_cell = node.cell_id or "unknown"
                
                artifact = DataArtifact(
                    id=node_id,
                    name=node.label,
                    type=artifact_type,
                    created_in_cell=created_in_cell,
                    importance_score=importance,
                    metadata={
                        'node_type': node.node_type.value,
                        'code_snippet': node.code_snippet,
                        'line_number': node.line_number
                    }
                )
                
                artifacts.append(artifact)
                self.artifacts[node_id] = artifact
        
        # Sort by importance
        artifacts.sort(key=lambda x: x.importance_score, reverse=True)
        
        return artifacts
    
    def _calculate_importance(self, node: DFGNode, dfg: DataFlowGraph) -> Tuple[str, float]:
        """
        Calculate artifact importance score.
        
        Importance is based on:
        - Name patterns matching known artifact types
        - Connectivity (incoming/outgoing edges)
        - Involvement in important operations
        
        Args:
            node: DFGNode to score
            dfg: Data flow graph
            
        Returns:
            Tuple of (artifact_type, importance_score)
        """
        score = 0.0
        artifact_type = 'unknown'
        
        # Check name patterns
        for atype, config in self.artifact_patterns.items():
            for pattern in config['names']:
                if pattern.lower() in node.label.lower():
                    score += config['base_importance']
                    artifact_type = atype
                    break
            if artifact_type != 'unknown':
                break
        
        # Check if involved in important operations
        for edge in dfg.edges:
            if edge.from_node == node.id or edge.to_node == node.id:
                score += 0.5
                
                # Bonus for specific operation types
                if edge.operation:
                    important_ops = ['read', 'load', 'save', 'fit', 'transform']
                    if any(op in edge.operation.lower() for op in important_ops):
                        score += 1.0
        
        # Connectivity bonus
        incoming_edges = sum(1 for e in dfg.edges if e.to_node == node.id)
        outgoing_edges = sum(1 for e in dfg.edges if e.from_node == node.id)
        
        # Heavily used artifacts are important
        score += incoming_edges * 1.5  # Being produced by many operations
        score += outgoing_edges * 2.0  # Being consumed by many operations
        
        # Cap the score
        score = min(score, 100.0)
        
        return artifact_type, score
    
    def build_artifact_lineage(self, dfg: DataFlowGraph, 
                               artifacts: List[DataArtifact]) -> nx.DiGraph:
        """
        Build lineage graph showing artifact transformations.
        
        Args:
            dfg: Data flow graph
            artifacts: List of identified artifacts
            
        Returns:
            NetworkX DiGraph of artifact lineage
        """
        G = nx.DiGraph()
        
        # Add artifact nodes
        for artifact in artifacts:
            G.add_node(
                artifact.id,
                name=artifact.name,
                type=artifact.type,
                cell=artifact.created_in_cell,
                importance=artifact.importance_score
            )
        
        # Find transformations between artifacts
        artifact_ids = {a.id for a in artifacts}
        
        for edge in dfg.edges:
            if edge.from_node in artifact_ids and edge.to_node in artifact_ids:
                # Find the operation between them
                operation = self._find_operation_between(dfg, edge.from_node, edge.to_node)
                
                G.add_edge(
                    edge.from_node,
                    edge.to_node,
                    operation=operation,
                    edge_type=edge.edge_type.value
                )
        
        return G
    
    def _find_operation_between(self, dfg: DataFlowGraph, 
                               from_id: str, to_id: str) -> str:
        """
        Find operation that transforms one artifact to another.
        
        Args:
            dfg: Data flow graph
            from_id: Source artifact ID
            to_id: Target artifact ID
            
        Returns:
            Operation name
        """
        # Look for intermediate function calls
        for edge in dfg.edges:
            if edge.from_node == from_id:
                intermediate = dfg.nodes.get(edge.to_node)
                if intermediate and intermediate.node_type == NodeType.FUNCTION_CALL:
                    # Check if this intermediate leads to target
                    for edge2 in dfg.edges:
                        if edge2.from_node == intermediate.id and edge2.to_node == to_id:
                            return intermediate.label
        
        return "transform"
    
    def get_artifact_by_name(self, name: str) -> Optional[DataArtifact]:
        """
        Get artifact by name.
        
        Args:
            name: Artifact name
            
        Returns:
            DataArtifact or None
        """
        for artifact in self.artifacts.values():
            if artifact.name == name:
                return artifact
        return None
    
    def get_artifact_stats(self, artifacts: List[DataArtifact]) -> Dict:
        """
        Get statistics about identified artifacts.
        
        Args:
            artifacts: List of artifacts
            
        Returns:
            Dictionary of statistics
        """
        if not artifacts:
            return {
                'total': 0,
                'by_type': {},
                'avg_importance': 0.0,
            }
        
        by_type = defaultdict(int)
        for artifact in artifacts:
            by_type[artifact.type] += 1
        
        return {
            'total': len(artifacts),
            'by_type': dict(by_type),
            'avg_importance': sum(a.importance_score for a in artifacts) / len(artifacts),
            'max_importance': max(a.importance_score for a in artifacts),
            'min_importance': min(a.importance_score for a in artifacts),
        }


__all__ = [
    "DataArtifactAnalyzer",
]