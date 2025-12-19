"""
Provenance Evaluator Module
============================

Multi-dimensional evaluation metrics for provenance extraction.

This module provides the ProvenanceEvaluator class which computes:
- Node classification metrics
- Artifact detection metrics
- Lineage accuracy metrics
- Stage sequence similarity
- Graph edit distance
- Composite provenance score
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np

from notebook_provenance.core.data_structures import (
    DataFlowGraph,
    DataArtifact,
    PipelineStageNode,
    Transformation,
)
from notebook_provenance.evaluation.ground_truth import GroundTruthAnnotation


class ProvenanceEvaluator:
    """
    Multi-dimensional evaluation for notebook provenance extraction.
    
    This class computes various metrics to evaluate the quality of
    provenance extraction against ground truth annotations.
    
    Example:
        >>> evaluator = ProvenanceEvaluator()
        >>> report = evaluator.evaluate(predicted_result, ground_truth)
        >>> print(report['composite_score'])
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize evaluator with optional custom weights.
        
        Args:
            weights: Optional dictionary of metric weights for composite score
        """
        self.weights = weights or {
            'node_classification_f1': 0.15,
            'artifact_detection_f1': 0.20,
            'lineage_edge_f1': 0.25,
            'stage_sequence_lcs': 0.15,
            'transformation_accuracy': 0.15,
            'column_lineage_f1': 0.10,
        }
    
    def evaluate(self, predicted: Dict, 
                ground_truth: GroundTruthAnnotation) -> Dict[str, Any]:
        """
        Comprehensive evaluation of predicted provenance.
        
        Args:
            predicted: Predicted provenance result
            ground_truth: Ground truth annotation
            
        Returns:
            Dictionary containing all metrics and composite score
        """
        report = {
            'notebook_id': ground_truth.notebook_id,
            'metrics': {},
        }
        
        # 1. Node/Cell Classification Metrics
        report['metrics']['node_classification'] = self.evaluate_node_classification(
            predicted, ground_truth
        )
        
        # 2. Artifact Detection Metrics
        report['metrics']['artifact_detection'] = self.evaluate_artifact_detection(
            predicted.get('artifacts', []),
            ground_truth.artifact_annotations
        )
        
        # 3. Lineage Edge Accuracy
        report['metrics']['lineage_accuracy'] = self.evaluate_lineage_accuracy(
            predicted.get('transformations', []),
            ground_truth.lineage_edges,
            ground_truth.artifact_annotations
        )
        
        # 4. Stage Sequence Similarity
        report['metrics']['stage_sequence'] = self.evaluate_stage_sequence(
            predicted.get('stages', []),
            ground_truth.stage_sequence
        )
        
        # 5. Transformation Classification
        report['metrics']['transformation_classification'] = self.evaluate_transformations(
            predicted.get('transformations', []),
            ground_truth.artifact_annotations
        )
        
        # 6. Column Lineage (if available)
        if ground_truth.column_operations:
            report['metrics']['column_lineage'] = self.evaluate_column_lineage(
                predicted.get('column_lineage', {}),
                ground_truth.column_operations
            )
        
        # Compute composite score
        report['composite_score'] = self._compute_composite_score(report['metrics'])
        
        return report
    
    def evaluate_node_classification(self, predicted: Dict,
                                    ground_truth: GroundTruthAnnotation) -> Dict:
        """
        Evaluate node/cell type classification.
        
        Args:
            predicted: Predicted result
            ground_truth: Ground truth annotation
            
        Returns:
            Dictionary of classification metrics
        """
        # Extract predicted cell task types
        pred_task_types = {}
        
        # Get task types from parsed cells or stages
        parsed_cells = predicted.get('parsed_cells', [])
        for cell in parsed_cells:
            if hasattr(cell, 'cell_id'):
                cell_id = cell.cell_id
            else:
                cell_id = cell.get('cell_id', '')
            
            # Infer task type from stages
            task_type = 'other'
            for stage in predicted.get('stages', []):
                if cell_id in stage.cells:
                    task_type = stage.stage_type.value
                    break
            
            pred_task_types[cell_id] = task_type
        
        # Ground truth task types
        gt_task_types = {
            cell.cell_id: cell.task_type 
            for cell in ground_truth.cell_annotations
        }
        
        # Compute metrics for common cells
        common_cells = set(pred_task_types.keys()) & set(gt_task_types.keys())
        
        if not common_cells:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'num_cells': 0
            }
        
        # Calculate per-class metrics
        pred_labels = [pred_task_types[c] for c in common_cells]
        gt_labels = [gt_task_types[c] for c in common_cells]
        
        accuracy = sum(p == g for p, g in zip(pred_labels, gt_labels)) / len(common_cells)
        
        # Compute macro F1
        all_labels = set(pred_labels) | set(gt_labels)
        f1_scores = []
        
        for label in all_labels:
            tp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == label and g == label)
            fp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == label and g != label)
            fn = sum(1 for p, g in zip(pred_labels, gt_labels) if p != label and g == label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'num_cells': len(common_cells),
            'per_class_f1': dict(zip(all_labels, f1_scores))
        }
    
    def evaluate_artifact_detection(self, predicted_artifacts: List[DataArtifact],
                                   gt_artifacts: List) -> Dict:
        """
        Evaluate artifact detection accuracy.
        
        Args:
            predicted_artifacts: List of predicted artifacts
            gt_artifacts: List of ground truth artifact annotations
            
        Returns:
            Dictionary of artifact detection metrics
        """
        # Extract artifact names and types
        pred_set = {(a.name, a.type) for a in predicted_artifacts}
        gt_set = {(a.name, a.artifact_type) for a in gt_artifacts}
        
        # Calculate precision, recall, F1
        true_positives = len(pred_set & gt_set)
        
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(gt_set) if gt_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate by type
        pred_by_type = defaultdict(set)
        for name, atype in pred_set:
            pred_by_type[atype].add(name)
        
        gt_by_type = defaultdict(set)
        for a in gt_artifacts:
            gt_by_type[a.artifact_type].add(a.name)
        
        per_type_f1 = {}
        for atype in set(pred_by_type.keys()) | set(gt_by_type.keys()):
            p_set = pred_by_type.get(atype, set())
            g_set = gt_by_type.get(atype, set())
            
            tp = len(p_set & g_set)
            p = tp / len(p_set) if p_set else 0
            r = tp / len(g_set) if g_set else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            per_type_f1[atype] = f
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'predicted_count': len(pred_set),
            'ground_truth_count': len(gt_set),
            'per_type_f1': per_type_f1
        }
    
    def evaluate_lineage_accuracy(self, predicted_transformations: List[Transformation],
                                 gt_lineage_edges: List[Tuple],
                                 gt_artifacts: List) -> Dict:
        """
        Evaluate artifact lineage accuracy.
        
        Args:
            predicted_transformations: List of predicted transformations
            gt_lineage_edges: List of ground truth (source, target) edges
            gt_artifacts: Ground truth artifact annotations
            
        Returns:
            Dictionary of lineage accuracy metrics
        """
        # Build artifact name mapping
        gt_artifact_names = {a.name for a in gt_artifacts}
        
        # Extract predicted edges (using artifact names)
        pred_edges = set()
        for trans in predicted_transformations:
            # We need to map artifact IDs to names
            # For now, assume IDs contain the name or extract from label
            target = trans.target_artifact
            for source in trans.source_artifacts:
                pred_edges.add((source, target))
        
        # Convert to comparable format (using names)
        gt_edges = set(tuple(e) for e in gt_lineage_edges)
        
        # Calculate metrics
        true_positives = len(pred_edges & gt_edges)
        
        precision = true_positives / len(pred_edges) if pred_edges else 0.0
        recall = true_positives / len(gt_edges) if gt_edges else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'predicted_edges': len(pred_edges),
            'ground_truth_edges': len(gt_edges),
            'missing_edges': list(gt_edges - pred_edges)[:10],  # Sample of missing
            'extra_edges': list(pred_edges - gt_edges)[:10]  # Sample of extra
        }
    
    def evaluate_stage_sequence(self, predicted_stages: List[PipelineStageNode],
                               gt_stage_sequence: List[str]) -> Dict:
        """
        Evaluate pipeline stage detection using sequence metrics.
        
        Args:
            predicted_stages: List of predicted stages
            gt_stage_sequence: Ground truth stage sequence
            
        Returns:
            Dictionary of stage sequence metrics
        """
        pred_sequence = [s.stage_type.value for s in predicted_stages]
        
        # Longest Common Subsequence ratio
        lcs_len = self._lcs_length(pred_sequence, gt_stage_sequence)
        max_len = max(len(pred_sequence), len(gt_stage_sequence))
        lcs_ratio = lcs_len / max_len if max_len > 0 else 1.0
        
        # Stage-level precision/recall
        pred_set = set(pred_sequence)
        gt_set = set(gt_stage_sequence)
        
        precision = len(pred_set & gt_set) / len(pred_set) if pred_set else 0.0
        recall = len(pred_set & gt_set) / len(gt_set) if gt_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Kendall's Tau for ordering (simplified)
        ordering_score = self._compute_ordering_score(pred_sequence, gt_stage_sequence)
        
        # Exact match
        exact_match = 1.0 if pred_sequence == gt_stage_sequence else 0.0
        
        return {
            'lcs_ratio': lcs_ratio,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ordering_score': ordering_score,
            'exact_match': exact_match,
            'predicted_sequence': pred_sequence,
            'ground_truth_sequence': gt_stage_sequence
        }
    
    def evaluate_transformations(self, predicted_transformations: List[Transformation],
                                gt_artifacts: List) -> Dict:
        """
        Evaluate transformation type classification.
        
        Args:
            predicted_transformations: List of predicted transformations
            gt_artifacts: Ground truth artifact annotations
            
        Returns:
            Dictionary of transformation metrics
        """
        # Build ground truth transformation types from artifacts
        gt_trans_types = {
            a.name: a.transformation_type 
            for a in gt_artifacts 
            if a.transformation_type
        }
        
        # Match predicted transformations
        correct = 0
        total = 0
        
        for trans in predicted_transformations:
            if trans.target_artifact in gt_trans_types:
                total += 1
                gt_type = gt_trans_types[trans.target_artifact]
                
                # Flexible matching (semantic type contains or equals)
                if (trans.semantic_type == gt_type or 
                    gt_type in trans.semantic_type or
                    trans.semantic_type in gt_type):
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_column_lineage(self, predicted_column_lineage: Dict,
                               gt_column_operations: Dict) -> Dict:
        """
        Evaluate column-level lineage accuracy.
        
        Args:
            predicted_column_lineage: Predicted column lineage
            gt_column_operations: Ground truth column operations
            
        Returns:
            Dictionary of column lineage metrics
        """
        pred_created = set(predicted_column_lineage.get('created', {}).keys())
        gt_created = set(gt_column_operations.get('created', {}).keys())
        
        pred_dropped = set(predicted_column_lineage.get('dropped', {}).keys())
        gt_dropped = set(gt_column_operations.get('dropped', {}).keys())
        
        # Calculate F1 for created columns
        created_tp = len(pred_created & gt_created)
        created_precision = created_tp / len(pred_created) if pred_created else 0.0
        created_recall = created_tp / len(gt_created) if gt_created else 0.0
        created_f1 = 2 * created_precision * created_recall / (created_precision + created_recall) if (created_precision + created_recall) > 0 else 0.0
        
        # Calculate F1 for dropped columns
        dropped_tp = len(pred_dropped & gt_dropped)
        dropped_precision = dropped_tp / len(pred_dropped) if pred_dropped else 0.0
        dropped_recall = dropped_tp / len(gt_dropped) if gt_dropped else 0.0
        dropped_f1 = 2 * dropped_precision * dropped_recall / (dropped_precision + dropped_recall) if (dropped_precision + dropped_recall) > 0 else 0.0
        
        # Combined F1
        combined_f1 = (created_f1 + dropped_f1) / 2
        
        return {
            'created_f1': created_f1,
            'dropped_f1': dropped_f1,
            'combined_f1': combined_f1,
            'created_precision': created_precision,
            'created_recall': created_recall,
            'dropped_precision': dropped_precision,
            'dropped_recall': dropped_recall
        }
    
    def compute_graph_edit_distance(self, pred_dfg: DataFlowGraph,
                                   gt_dfg: DataFlowGraph,
                                   timeout: int = 30) -> float:
        """
        Compute normalized Graph Edit Distance.
        
        Note: This is computationally expensive for large graphs.
        
        Args:
            pred_dfg: Predicted data flow graph
            gt_dfg: Ground truth data flow graph
            timeout: Timeout in seconds
            
        Returns:
            Normalized GED (0 = perfect match, 1 = completely different)
        """
        try:
            import networkx as nx
            
            pred_G = pred_dfg.to_networkx()
            gt_G = gt_dfg.to_networkx()
            
            # Node match function
            def node_match(n1, n2):
                return n1.get('node_type') == n2.get('node_type')
            
            ged = nx.graph_edit_distance(
                pred_G, gt_G,
                node_match=node_match,
                timeout=timeout
            )
            
            if ged is None:
                return 1.0  # Timeout, assume very different
            
            # Normalize
            max_edits = (len(pred_G.nodes) + len(gt_G.nodes) + 
                        len(pred_G.edges) + len(gt_G.edges))
            
            return ged / max_edits if max_edits > 0 else 0.0
            
        except Exception as e:
            print(f"Warning: GED computation failed: {e}")
            return 1.0
    
    def _compute_composite_score(self, metrics: Dict) -> float:
        """
        Compute single composite score from all metrics.
        
        Args:
            metrics: Dictionary of all metrics
            
        Returns:
            Composite score (0-1)
        """
        score = 0.0
        total_weight = 0.0
        
        # Map metric names to their values
        metric_values = {
            'node_classification_f1': metrics.get('node_classification', {}).get('macro_f1', 0),
            'artifact_detection_f1': metrics.get('artifact_detection', {}).get('f1', 0),
            'lineage_edge_f1': metrics.get('lineage_accuracy', {}).get('f1', 0),
            'stage_sequence_lcs': metrics.get('stage_sequence', {}).get('lcs_ratio', 0),
            'transformation_accuracy': metrics.get('transformation_classification', {}).get('accuracy', 0),
            'column_lineage_f1': metrics.get('column_lineage', {}).get('combined_f1', 0),
        }
        
        for metric_name, weight in self.weights.items():
            if metric_name in metric_values:
                score += weight * metric_values[metric_name]
                total_weight += weight
        
        # Normalize by actual weight used
        return score / total_weight if total_weight > 0 else 0.0
    
    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _compute_ordering_score(self, pred_seq: List, gt_seq: List) -> float:
        """Compute ordering consistency score."""
        if not pred_seq or not gt_seq:
            return 0.0
        
        # Find common elements
        common = set(pred_seq) & set(gt_seq)
        if len(common) < 2:
            return 1.0 if pred_seq == gt_seq else 0.0
        
        # Check pairwise ordering
        correct_pairs = 0
        total_pairs = 0
        
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                a, b = common_list[i], common_list[j]
                
                # Get positions in both sequences
                try:
                    pred_pos_a = pred_seq.index(a)
                    pred_pos_b = pred_seq.index(b)
                    gt_pos_a = gt_seq.index(a)
                    gt_pos_b = gt_seq.index(b)
                    
                    total_pairs += 1
                    
                    # Check if relative order is same
                    if (pred_pos_a < pred_pos_b) == (gt_pos_a < gt_pos_b):
                        correct_pairs += 1
                except ValueError:
                    continue
        
        return correct_pairs / total_pairs if total_pairs > 0 else 0.0


__all__ = [
    "ProvenanceEvaluator",
]