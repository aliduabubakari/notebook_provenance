"""
Artifact Analyzer Module - Updated with Hybrid Classifier
==========================================================
"""

from typing import Dict, List, Optional
from collections import defaultdict
import networkx as nx

from notebook_provenance.core.data_structures import (
    DataFlowGraph,
    DFGNode,
    DataArtifact,
    CellDependency,
)
from notebook_provenance.core.enums import NodeType
from notebook_provenance.semantic.artifact_classifier import (
    HybridArtifactClassifier,
    ArtifactClassification,
)


class DataArtifactAnalyzer:
    """
    Analyze and track data artifacts using hybrid LLM + embedding classification.
    """
    
    def __init__(self, llm_analyzer=None, use_embeddings: bool = True):
        """
        Initialize analyzer with hybrid classifier.
        
        Args:
            llm_analyzer: LLM analyzer for semantic understanding
            use_embeddings: Whether to use embedding-based similarity
        """
        self.llm_analyzer = llm_analyzer
        self.hybrid_classifier = HybridArtifactClassifier(
            llm_analyzer=llm_analyzer,
            use_embeddings=use_embeddings
        )
        self.artifacts = {}
        self.classifications = {}
    
    def identify_artifacts(self, dfg: DataFlowGraph, 
                      cell_dependencies: Dict[str, CellDependency],
                      code_cells: Optional[Dict[str, str]] = None,
                      max_llm_calls: int = 15) -> List[DataArtifact]:
        """
        Identify artifacts with FIXED deduplication.
        """
        artifacts = []
        seen_names = set()  # Track by NAME, not node_id
        
        # Get code context
        code_context = code_cells or {}
        
        # Filter to relevant nodes
        relevant_nodes = [
            (node_id, node) for node_id, node in dfg.nodes.items()
            if node.node_type in [NodeType.VARIABLE, NodeType.DATA_ARTIFACT]
        ]
        
        print(f"  Classifying {len(relevant_nodes)} nodes...")
        
        # Group by name FIRST
        by_name = {}
        for node_id, node in relevant_nodes:
            name = node.label
            if name not in by_name:
                by_name[name] = []
            by_name[name].append((node_id, node))
        
        print(f"  Found {len(by_name)} unique variable names")
        
        # Classify each unique name only ONCE
        for var_name, node_list in by_name.items():
            # Use first occurrence (usually the definition)
            node_id, node = node_list[0]
            
            # Get context
            cell_code = code_context.get(node.cell_id, "")
            function_calls = self._get_related_functions(dfg, node_id)
            
            # Classify
            try:
                classification = self.hybrid_classifier.classify(
                    node=node,
                    code_context=cell_code,
                    function_calls=function_calls
                )
            except Exception as e:
                print(f"  ⚠ Classification failed for '{var_name}': {e}")
                classification = ArtifactClassification(
                    category='utility',
                    importance=3.0,
                    confidence=0.3,
                    reasoning='Classification failed',
                    source='fallback'
                )
            
            self.classifications[node_id] = classification
            
            # Filter: only keep core_data and important payloads
            if classification.category not in ['core_data', 'payload']:
                continue
            
            if classification.importance < 5:
                continue
            
            # Create artifact (ONCE per unique name)
            artifact = DataArtifact(
                id=node_id,
                name=var_name,
                type=classification.semantic_type or classification.category,
                created_in_cell=node.cell_id or "unknown",
                importance_score=classification.importance * classification.confidence * 5,
                metadata={
                    'category': classification.category,
                    'classification_source': classification.source,
                    'confidence': classification.confidence,
                    'reasoning': classification.reasoning,
                    'created_by': node.metadata.get('created_by'),  # ADD THIS
                    'occurrences': len(node_list),
                    'all_node_ids': [nid for nid, _ in node_list]
                }
            )
            
            artifacts.append(artifact)
            self.artifacts[node_id] = artifact
        
        print(f"  ✓ Classification complete: {len(artifacts)} unique artifacts")
        
        # Sort by importance
        artifacts.sort(key=lambda x: x.importance_score, reverse=True)
        
        return artifacts

    def _get_related_functions(self, dfg: DataFlowGraph, node_id: str) -> List[str]:
        """Get function calls related to a node."""
        functions = []
        
        # Check incoming edges (what created this)
        for edge in dfg.edges:
            if edge.to_node == node_id:
                source = dfg.get_node(edge.from_node)
                if source and source.node_type == NodeType.FUNCTION_CALL:
                    functions.append(source.label)
        
        # Check outgoing edges (what this is passed to)
        for edge in dfg.edges:
            if edge.from_node == node_id:
                target = dfg.get_node(edge.to_node)
                if target and target.node_type == NodeType.FUNCTION_CALL:
                    functions.append(target.label)
        
        return functions
    
    def get_classification_stats(self) -> Dict:
        """Get statistics about classifications."""
        return self.hybrid_classifier.get_statistics()


__all__ = [
    "DataArtifactAnalyzer",
]