"""
Evaluation Benchmark Module
============================

Benchmark framework for systematic evaluation.

This module provides the EvaluationBenchmark class which:
- Runs evaluation across multiple notebooks
- Aggregates results by complexity and domain
- Generates comprehensive reports
"""

from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
import time
import numpy as np

from notebook_provenance.evaluation.metrics import ProvenanceEvaluator
from notebook_provenance.evaluation.ground_truth import (
    GroundTruthManager,
    GroundTruthAnnotation,
)


class EvaluationBenchmark:
    """
    Benchmark framework for systematic evaluation.
    
    This class provides a standardized way to evaluate the provenance
    system across a dataset of annotated notebooks.
    
    Example:
        >>> benchmark = EvaluationBenchmark("annotations/")
        >>> results = benchmark.run_benchmark(analysis_function)
        >>> benchmark.save_results(results, "benchmark_results.json")
    """
    
    def __init__(self, annotations_dir: str, 
                 evaluator: Optional[ProvenanceEvaluator] = None):
        """
        Initialize benchmark.
        
        Args:
            annotations_dir: Directory containing ground truth annotations
            evaluator: Optional custom evaluator
        """
        self.gt_manager = GroundTruthManager(annotations_dir)
        self.evaluator = evaluator or ProvenanceEvaluator()
        self.results = []
    
    def run_benchmark(self, analysis_function: Callable,
                     notebook_dir: Optional[str] = None,
                     subset: Optional[List[str]] = None,
                     verbose: bool = True) -> Dict[str, Any]:
        """
        Run benchmark across all annotated notebooks.
        
        Args:
            analysis_function: Function that takes (code_cells, cell_ids) and returns result
            notebook_dir: Directory containing notebooks (if different from annotations)
            subset: Optional list of notebook IDs to evaluate
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing benchmark results
        """
        annotations = self.gt_manager.load_all_annotations()
        
        if subset:
            annotations = {k: v for k, v in annotations.items() if k in subset}
        
        if verbose:
            print(f"Running benchmark on {len(annotations)} notebooks...")
        
        results = []
        errors = []
        
        for notebook_id, annotation in annotations.items():
            if verbose:
                print(f"  Evaluating {notebook_id}...", end=" ")
            
            try:
                start_time = time.time()
                
                # Load notebook
                notebook_path = annotation.notebook_path
                if notebook_dir:
                    notebook_path = str(Path(notebook_dir) / Path(notebook_path).name)
                
                # Run analysis
                from notebook_provenance.parsing.notebook_loader import NotebookLoader
                code_cells, cell_ids = NotebookLoader.load_notebook(notebook_path)
                
                predicted = analysis_function(code_cells, cell_ids)
                
                # Evaluate
                eval_result = self.evaluator.evaluate(predicted, annotation)
                eval_result['runtime_seconds'] = time.time() - start_time
                eval_result['complexity'] = annotation.complexity_level
                eval_result['domain'] = annotation.domain
                
                results.append(eval_result)
                
                if verbose:
                    print(f"✓ (score: {eval_result['composite_score']:.3f})")
                
            except Exception as e:
                if verbose:
                    print(f"✗ ({e})")
                errors.append({
                    'notebook_id': notebook_id,
                    'error': str(e)
                })
        
        self.results = results
        
        # Aggregate results
        return self._aggregate_results(results, errors)
    
    def _aggregate_results(self, results: List[Dict], 
                          errors: List[Dict]) -> Dict[str, Any]:
        """Aggregate individual results into summary."""
        if not results:
            return {
                'summary': {'total': 0, 'successful': 0, 'failed': len(errors)},
                'errors': errors
            }
        
        # Overall statistics
        composite_scores = [r['composite_score'] for r in results]
        
        summary = {
            'total_notebooks': len(results) + len(errors),
            'successful': len(results),
            'failed': len(errors),
            'composite_score': {
                'mean': np.mean(composite_scores),
                'std': np.std(composite_scores),
                'min': np.min(composite_scores),
                'max': np.max(composite_scores),
                'median': np.median(composite_scores)
            }
        }
        
        # By complexity
        by_complexity = {}
        for complexity in ['simple', 'medium', 'complex']:
            subset = [r for r in results if r.get('complexity') == complexity]
            if subset:
                scores = [r['composite_score'] for r in subset]
                by_complexity[complexity] = {
                    'count': len(subset),
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
        
        # By domain
        by_domain = {}
        domains = set(r.get('domain', 'unknown') for r in results)
        for domain in domains:
            subset = [r for r in results if r.get('domain') == domain]
            if subset:
                scores = [r['composite_score'] for r in subset]
                by_domain[domain] = {
                    'count': len(subset),
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
        
        # Per-metric statistics
        metric_stats = {}
        metric_keys = [
            ('node_classification', 'macro_f1'),
            ('artifact_detection', 'f1'),
            ('lineage_accuracy', 'f1'),
            ('stage_sequence', 'lcs_ratio'),
            ('transformation_classification', 'accuracy'),
        ]
        
        for metric_name, value_key in metric_keys:
            values = []
            for r in results:
                if metric_name in r.get('metrics', {}):
                    val = r['metrics'][metric_name].get(value_key)
                    if val is not None:
                        values.append(val)
            
            if values:
                metric_stats[f"{metric_name}_{value_key}"] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return {
            'summary': summary,
            'by_complexity': by_complexity,
            'by_domain': by_domain,
            'metric_statistics': metric_stats,
            'detailed_results': results,
            'errors': errors
        }
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save benchmark results to JSON.
        
        Args:
            results: Benchmark results
            output_path: Output file path
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results = convert_numpy(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Benchmark results saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """Print formatted summary of results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        summary = results.get('summary', {})
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total notebooks: {summary.get('total_notebooks', 0)}")
        print(f"  • Successful: {summary.get('successful', 0)}")
        print(f"  • Failed: {summary.get('failed', 0)}")
        
        if 'composite_score' in summary:
            cs = summary['composite_score']
            print(f"\n📈 Composite Score:")
            print(f"  • Mean: {cs.get('mean', 0):.4f}")
            print(f"  • Std:  {cs.get('std', 0):.4f}")
            print(f"  • Min:  {cs.get('min', 0):.4f}")
            print(f"  • Max:  {cs.get('max', 0):.4f}")
        
        if results.get('by_complexity'):
            print(f"\n🎯 By Complexity:")
            for complexity, stats in results['by_complexity'].items():
                print(f"  • {complexity.capitalize()}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        if results.get('by_domain'):
            print(f"\n🏷️  By Domain:")
            for domain, stats in results['by_domain'].items():
                print(f"  • {domain}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        if results.get('metric_statistics'):
            print(f"\n📉 Per-Metric Statistics:")
            for metric, stats in results['metric_statistics'].items():
                print(f"  • {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print("\n" + "=" * 60)


__all__ = [
    "EvaluationBenchmark",
]