"""
Provenance Comparator Module
=============================

Compare multiple notebook provenance results.

This module provides the ProvenanceComparator class which:
- Compares notebooks at various levels
- Generates comparison reports
- Identifies differences and similarities
- Supports batch comparison
"""

from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict


class ProvenanceComparator:
    """
    Compare multiple notebook provenance results.
    
    This class provides comprehensive comparison capabilities
    for analyzing differences between notebooks.
    
    Example:
        >>> comparator = ProvenanceComparator()
        >>> diff = comparator.compare_two(result1, result2)
        >>> report = comparator.compare_multiple([r1, r2, r3], names)
    """
    
    @staticmethod
    def summarize_result(result: Dict) -> Dict[str, Any]:
        """
        Summarize a provenance result for comparison.
        
        Args:
            result: Provenance analysis result
            
        Returns:
            Dictionary of summarized data
        """
        # Extract artifacts as set of (name, type) tuples
        artifacts = {
            (a.name, a.type) 
            for a in result.get('artifacts', [])
        }
        
        # Extract transformations as set of tuples
        transformations = {
            (tuple(sorted(t.source_artifacts)), t.target_artifact, t.operation)
            for t in result.get('transformations', [])
        }
        
        # Extract stage sequence
        stages_seq = [s.stage_type.value for s in result.get('stages', [])]
        
        # Extract columns created
        cols_created = set(result.get('column_lineage', {}).get('created', {}).keys())
        
        return {
            'artifacts': artifacts,
            'transformations': transformations,
            'stages_seq': stages_seq,
            'cols_created': cols_created,
            'stats': result.get('statistics', {})
        }
    
    @staticmethod
    def compare_two(base: Dict, other: Dict) -> Dict[str, Any]:
        """
        Compare two notebook results.
        
        Args:
            base: Base notebook result
            other: Other notebook result
            
        Returns:
            Dictionary of differences
        """
        A = ProvenanceComparator.summarize_result(base)
        B = ProvenanceComparator.summarize_result(other)
        
        # Artifact differences
        added_artifacts = B['artifacts'] - A['artifacts']
        removed_artifacts = A['artifacts'] - B['artifacts']
        
        # Transformation differences
        added_trans = B['transformations'] - A['transformations']
        removed_trans = A['transformations'] - B['transformations']
        
        # Column differences
        added_cols = B['cols_created'] - A['cols_created']
        removed_cols = A['cols_created'] - B['cols_created']
        
        # Stage sequence similarity (LCS-based)
        stage_similarity = ProvenanceComparator._compute_sequence_similarity(
            A['stages_seq'], B['stages_seq']
        )
        
        return {
            'added_artifacts': sorted(list(added_artifacts)),
            'removed_artifacts': sorted(list(removed_artifacts)),
            'added_transformations': len(added_trans),
            'removed_transformations': len(removed_trans),
            'added_columns_created': sorted(list(added_cols)),
            'removed_columns_created': sorted(list(removed_cols)),
            'stage_sequence_similarity': round(stage_similarity, 3),
            'stats_base': A['stats'],
            'stats_other': B['stats']
        }
    
    @staticmethod
    def compare_multiple(results: List[Dict], names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple notebook results.
        
        Args:
            results: List of provenance results
            names: List of notebook names
            
        Returns:
            Dictionary with comparison report
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 results to compare")
        
        report = {
            'baseline': names[0],
            'comparisons': {}
        }
        
        base = results[0]
        
        for name, res in zip(names[1:], results[1:]):
            report['comparisons'][name] = ProvenanceComparator.compare_two(base, res)
        
        # Add aggregate statistics
        report['aggregate'] = ProvenanceComparator._compute_aggregate_stats(results, names)
        
        return report
    
    @staticmethod
    def _compute_sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
        """
        Compute similarity between two sequences using LCS.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Similarity score (0-1)
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Longest Common Subsequence
        lcs_len = ProvenanceComparator._lcs_length(seq1, seq2)
        
        # Normalize by max length
        max_len = max(len(seq1), len(seq2))
        return lcs_len / max_len if max_len > 0 else 0.0
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def _compute_aggregate_stats(results: List[Dict], names: List[str]) -> Dict:
        """Compute aggregate statistics across all notebooks."""
        stats = {
            'total_notebooks': len(results),
            'notebook_names': names,
            'metrics': {}
        }
        
        # Aggregate metrics
        metrics = ['total_cells', 'artifacts', 'transformations', 'stages']
        
        for metric in metrics:
            values = []
            for result in results:
                if metric in ['artifacts', 'transformations', 'stages']:
                    values.append(len(result.get(metric, [])))
                else:
                    values.append(result.get('statistics', {}).get(metric, 0))
            
            stats['metrics'][metric] = {
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'avg': sum(values) / len(values) if values else 0,
                'values': values
            }
        
        # Find common stages
        all_stages = [
            set(s.stage_type.value for s in result.get('stages', []))
            for result in results
        ]
        if all_stages:
            common_stages = set.intersection(*all_stages)
            stats['common_stages'] = sorted(list(common_stages))
        else:
            stats['common_stages'] = []
        
        # Find common artifacts
        all_artifacts = [
            set((a.name, a.type) for a in result.get('artifacts', []))
            for result in results
        ]
        if all_artifacts:
            common_artifacts = set.intersection(*all_artifacts)
            stats['common_artifacts'] = sorted(list(common_artifacts))
        else:
            stats['common_artifacts'] = []
        
        return stats
    
    @staticmethod
    def find_similar_notebooks(results: List[Dict], names: List[str],
                              threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find pairs of similar notebooks.
        
        Args:
            results: List of provenance results
            names: List of notebook names
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (name1, name2, similarity) tuples
        """
        similar_pairs = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                similarity = ProvenanceComparator._compute_overall_similarity(
                    results[i], results[j]
                )
                
                if similarity >= threshold:
                    similar_pairs.append((names[i], names[j], similarity))
        
        # Sort by similarity descending
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    @staticmethod
    def _compute_overall_similarity(result1: Dict, result2: Dict) -> float:
        """
        Compute overall similarity between two results.
        
        Uses weighted combination of various similarity metrics.
        """
        A = ProvenanceComparator.summarize_result(result1)
        B = ProvenanceComparator.summarize_result(result2)
        
        # Artifact similarity (Jaccard)
        artifact_sim = ProvenanceComparator._jaccard_similarity(
            A['artifacts'], B['artifacts']
        )
        
        # Stage sequence similarity
        stage_sim = ProvenanceComparator._compute_sequence_similarity(
            A['stages_seq'], B['stages_seq']
        )
        
        # Column similarity
        col_sim = ProvenanceComparator._jaccard_similarity(
            A['cols_created'], B['cols_created']
        )
        
        # Weighted combination
        weights = {
            'artifacts': 0.4,
            'stages': 0.4,
            'columns': 0.2
        }
        
        overall_sim = (
            weights['artifacts'] * artifact_sim +
            weights['stages'] * stage_sim +
            weights['columns'] * col_sim
        )
        
        return overall_sim
    
    @staticmethod
    def _jaccard_similarity(set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


__all__ = [
    "ProvenanceComparator",
]