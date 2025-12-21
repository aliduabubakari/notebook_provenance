"""
Semantic Deduplication Module
==============================

Uses embeddings to identify semantically equivalent variables.

This module provides:
- SemanticDeduplicator: Deduplicate variables using semantic similarity
- VariableCluster: Group semantically equivalent variables
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from notebook_provenance.core.data_structures import DFGNode
from notebook_provenance.core.enums import NodeType
from notebook_provenance.semantic.llm_analyzer import LLMSemanticAnalyzer


@dataclass
class VariableCluster:
    """
    A cluster of semantically similar variables.
    
    Attributes:
        canonical_name: The representative name for this cluster
        canonical_id: The representative node ID
        members: List of (name, node_id, similarity_score) tuples
    """
    canonical_name: str
    canonical_id: str
    members: List[Tuple[str, str, float]]  # (name, node_id, similarity)
    
    def add_member(self, name: str, node_id: str, similarity: float):
        """Add a member to this cluster."""
        self.members.append((name, node_id, similarity))
    
    @property
    def size(self) -> int:
        """Number of members in cluster."""
        return len(self.members)


class SemanticDeduplicator:
    """
    Deduplicate variables using semantic similarity.
    
    This handles cases where:
    - Same variable referenced multiple times (var_df_30, var_df_45)
    - Semantically equivalent names (customer_data, customers_df)
    - Versioned variables (df_v1, df_v2 → track as versions)
    
    Example:
        >>> deduplicator = SemanticDeduplicator(llm_analyzer)
        >>> canonical_map = deduplicator.deduplicate(dfg.nodes)
        >>> # canonical_map: {node_id -> canonical_node_id}
    """
    
    def __init__(self, llm_analyzer: Optional[LLMSemanticAnalyzer] = None,
                 similarity_threshold: float = 0.85,
                 use_semantic_merge: bool = True):
        """
        Initialize deduplicator.
        
        Args:
            llm_analyzer: LLM analyzer for embeddings
            similarity_threshold: Threshold for considering variables equivalent
            use_semantic_merge: Whether to use semantic similarity (requires LLM)
        """
        self.llm_analyzer = llm_analyzer
        self.similarity_threshold = similarity_threshold
        self.use_semantic_merge = use_semantic_merge
        self.embeddings = {}
        self.clusters = []
    
    def deduplicate(self, nodes: Dict[str, DFGNode], 
                   verbose: bool = False) -> Dict[str, str]:
        """
        Deduplicate nodes and return mapping to canonical IDs.
        
        Args:
            nodes: Dictionary of node_id -> DFGNode
            verbose: Whether to print deduplication info
            
        Returns:
            Dictionary mapping any node_id to its canonical node_id
        """
        if verbose:
            print(f"\n[Deduplication] Processing {len(nodes)} nodes...")
        
        # Filter to only variables and data artifacts
        var_nodes = {
            nid: node for nid, node in nodes.items()
            if node.node_type in [NodeType.VARIABLE, NodeType.DATA_ARTIFACT]
        }
        
        if verbose:
            print(f"  Filtered to {len(var_nodes)} variable/artifact nodes")
        
        # Group by exact name first
        by_name = {}
        for node_id, node in var_nodes.items():
            name = node.label
            if name not in by_name:
                by_name[name] = []
            by_name[name].append((node_id, node))
        
        if verbose:
            print(f"  Found {len(by_name)} unique variable names")
        
        # For each name group, pick canonical (earliest or most important)
        canonical_map = {}
        
        for name, node_list in by_name.items():
            # Sort by cell_id (earliest first) then by line number
            node_list.sort(key=lambda x: (x[1].cell_id, x[1].line_number))
            
            # First one is canonical
            canonical_id = node_list[0][0]
            
            for node_id, _ in node_list:
                canonical_map[node_id] = canonical_id
            
            if len(node_list) > 1 and verbose:
                print(f"    {name}: merged {len(node_list)} occurrences")
        
        # Now check for semantic similarity between different names
        if (self.use_semantic_merge and 
            self.llm_analyzer and 
            self.llm_analyzer.enabled):
            
            if verbose:
                print(f"\n  Checking semantic similarity across names...")
            
            canonical_map = self._semantic_merge(by_name, canonical_map, verbose)
        
        if verbose:
            unique_after = len(set(canonical_map.values()))
            print(f"\n[Deduplication] Reduced from {len(nodes)} to {unique_after} unique artifacts")
        
        return canonical_map
    
    def _semantic_merge(self, by_name: Dict, canonical_map: Dict,
                       verbose: bool = False) -> Dict:
        """
        Merge semantically similar variable names.
        
        Args:
            by_name: Dictionary of name -> [(node_id, node)]
            canonical_map: Current canonical mapping
            verbose: Whether to print merge info
            
        Returns:
            Updated canonical mapping
        """
        unique_names = list(by_name.keys())
        
        if len(unique_names) < 2:
            return canonical_map
        
        # Generate embeddings for each unique name
        embeddings = {}
        for name in unique_names:
            emb = self._get_embedding(name)
            if emb is not None:
                embeddings[name] = emb
        
        if len(embeddings) < 2:
            if verbose:
                print(f"    Warning: Could not generate embeddings")
            return canonical_map
        
        # Find similar pairs
        merged_count = 0
        merged = set()
        
        for i, name1 in enumerate(unique_names):
            if name1 in merged or name1 not in embeddings:
                continue
            
            for name2 in unique_names[i+1:]:
                if name2 in merged or name2 not in embeddings:
                    continue
                
                similarity = self._cosine_similarity(
                    embeddings[name1], 
                    embeddings[name2]
                )
                
                # Check if they're semantically equivalent
                if similarity >= self.similarity_threshold:
                    # Merge name2 into name1 (name1 is canonical)
                    canonical_id = by_name[name1][0][0]
                    
                    for node_id, _ in by_name[name2]:
                        canonical_map[node_id] = canonical_id
                    
                    merged.add(name2)
                    merged_count += 1
                    
                    if verbose:
                        print(f"    Merged '{name2}' → '{name1}' (similarity: {similarity:.2f})")
        
        if verbose and merged_count > 0:
            print(f"    Total semantic merges: {merged_count}")
        
        return canonical_map
    
    def _get_embedding(self, name: str) -> Optional[List[float]]:
        """
        Get embedding for a variable name.
        
        Args:
            name: Variable name
            
        Returns:
            Embedding vector or None if failed
        """
        if name in self.embeddings:
            return self.embeddings[name]
        
        try:
            response = self.llm_analyzer.client.embeddings.create(
                model="text-embedding-3-small",
                input=f"variable name: {name}"
            )
            embedding = response.data[0].embedding
            self.embeddings[name] = embedding
            return embedding
        except Exception as e:
            # Silently fail - will use exact name matching instead
            return None
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        if norm < 1e-10:
            return 0.0
        return float(np.dot(a_np, b_np) / norm)
    
    def get_statistics(self) -> Dict:
        """Get deduplication statistics."""
        return {
            'embeddings_generated': len(self.embeddings),
            'similarity_threshold': self.similarity_threshold,
            'use_semantic_merge': self.use_semantic_merge
        }


__all__ = [
    "SemanticDeduplicator",
    "VariableCluster",
]