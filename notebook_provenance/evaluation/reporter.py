"""
Metrics Reporter Module
========================

Generate evaluation reports in various formats.

This module provides the MetricsReporter class which:
- Generates Markdown reports
- Creates LaTeX tables for papers
- Produces visualizations of metrics
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class MetricsReporter:
    """
    Generate evaluation reports in various formats.
    
    This class produces reports suitable for documentation
    and research papers.
    
    Example:
        >>> reporter = MetricsReporter()
        >>> reporter.generate_markdown_report(results, "report.md")
        >>> reporter.generate_latex_table(results, "table.tex")
    """
    
    def generate_markdown_report(self, benchmark_results: Dict, 
                                output_path: str) -> str:
        """
        Generate comprehensive Markdown report.
        
        Args Benchmark results dictionary
            output_path: Output file path
            
        Returns:
            Markdown content
        """
        lines = []
        
        # Header
        lines.append("# Provenance Extraction Evaluation Report\n")
        lines.append(f"*Generated automatically*\n")
        lines.append("---\n")
        
        # Summary
        summary = benchmark_results.get('summary', {})
        lines.append("## Summary\n")
        lines.append(f"- **Total notebooks evaluated:** {summary.get('total_notebooks', 0)}")
        lines.append(f"- **Successful:** {summary.get('successful', 0)}")
        lines.append(f"- **Failed:** {summary.get('failed', 0)}\n")
        
        # Composite Score
        if 'composite_score' in summary:
            cs = summary['composite_score']
            lines.append("## Composite Score\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean | {cs.get('mean', 0):.4f} |")
            lines.append(f"| Std | {cs.get('std', 0):.4f} |")
            lines.append(f"| Min | {cs.get('min', 0):.4f} |")
            lines.append(f"| Max | {cs.get('max', 0):.4f} |")
            lines.append(f"| Median | {cs.get('median', 0):.4f} |\n")
        
        # By Complexity
        if benchmark_results.get('by_complexity'):
            lines.append("## Results by Complexity\n")
            lines.append("| Complexity | Count | Mean Score | Std |")
            lines.append("|------------|-------|------------|-----|")
            for complexity, stats in benchmark_results['by_complexity'].items():
                lines.append(f"| {complexity.capitalize()} | {stats['count']} | {stats['mean']:.4f} | {stats['std']:.4f} |")
            lines.append("")
        
        # By Domain
        if benchmark_results.get('by_domain'):
            lines.append("## Results by Domain\n")
            lines.append("| Domain | Count | Mean Score | Std |")
            lines.append("|--------|-------|------------|-----|")
            for domain, stats in benchmark_results['by_domain'].items():
                lines.append(f"| {domain} | {stats['count']} | {stats['mean']:.4f} | {stats['std']:.4f} |")
            lines.append("")
        
        # Per-Metric Statistics
        if benchmark_results.get('metric_statistics'):
            lines.append("## Per-Metric Results\n")
            lines.append("| Metric | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")
            for metric, stats in benchmark_results['metric_statistics'].items():
                lines.append(f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
            lines.append("")
        
        # Detailed Results
        if benchmark_results.get('detailed_results'):
            lines.append("## Detailed Results\n")
            lines.append("| Notebook | Composite Score | Complexity | Domain |")
            lines.append("|----------|-----------------|------------|--------|")
            for r in benchmark_results['detailed_results'][:20]:  # Limit to 20
                lines.append(f"| {r.get('notebook_id', 'N/A')} | {r.get('composite_score', 0):.4f} | {r.get('complexity', 'N/A')} | {r.get('domain', 'N/A')} |")
            
            if len(benchmark_results['detailed_results']) > 20:
                lines.append(f"\n*... and {len(benchmark_results['detailed_results']) - 20} more notebooks*\n")
        
        # Errors
        if benchmark_results.get('errors'):
            lines.append("## Errors\n")
            for error in benchmark_results['errors']:
                lines.append(f"- **{error['notebook_id']}**: {error['error']}")
            lines.append("")
        
        content = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Markdown report saved to {output_path}")
        return content
    
    def generate_latex_table(self, benchmark_results: Dict,
                            output_path: str,
                            caption: str = "Provenance Extraction Results",
                            label: str = "tab:results") -> str:
        """
        Generate LaTeX table for research papers.
        
        Args:
            benchmark_results: Benchmark results dictionary
            output_path: Output file path
            caption: Table caption
            label: LaTeX label
            
        Returns:
            LaTeX content
        """
        lines = []
        
        # Main results table
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("Metric & Mean & Std & Range \\\\")
        lines.append("\\midrule")
        
        # Add metric rows
        if benchmark_results.get('metric_statistics'):
            for metric, stats in benchmark_results['metric_statistics'].items():
                metric_display = metric.replace('_', ' ').title()
                lines.append(f"{metric_display} & {stats['mean']:.3f} & {stats['std']:.3f} & [{stats['min']:.3f}, {stats['max']:.3f}] \\\\")
        
        # Add composite score
        summary = benchmark_results.get('summary', {})
        if 'composite_score' in summary:
            cs = summary['composite_score']
            lines.append("\\midrule")
            lines.append(f"\\textbf{{Composite Score}} & \\textbf{{{cs['mean']:.3f}}} & {cs['std']:.3f} & [{cs['min']:.3f}, {cs['max']:.3f}] \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        content = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ LaTeX table saved to {output_path}")
        return content
    
    def generate_latex_complexity_table(self, benchmark_results: Dict,
                                       output_path: str) -> str:
        """
        Generate LaTeX table showing results by complexity.
        
        Args:
            benchmark_results: Benchmark results dictionary
            output_path: Output file path
            
        Returns:
            LaTeX content
        """
        lines = []
        
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append("\\caption{Results by Notebook Complexity}")
        lines.append("\\label{tab:complexity}")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("Complexity & Count & Mean Score & Std \\\\")
        lines.append("\\midrule")
        
        if benchmark_results.get('by_complexity'):
            for complexity in ['simple', 'medium', 'complex']:
                if complexity in benchmark_results['by_complexity']:
                    stats = benchmark_results['by_complexity'][complexity]
                    lines.append(f"{complexity.capitalize()} & {stats['count']} & {stats['mean']:.3f} & {stats['std']:.3f} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        content = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ LaTeX complexity table saved to {output_path}")
        return content
    
    def generate_visualization(self, benchmark_results: Dict,
                              output_path: str):
        """
        Generate visualization plots of evaluation results.
        
        Args:
            benchmark_results: Benchmark results dictionary
            output_path: Output file path (without extension)
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Warning: matplotlib not available for visualization")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Score Distribution
        ax = axes[0, 0]
        if benchmark_results.get('detailed_results'):
            scores = [r['composite_score'] for r in benchmark_results['detailed_results']]
            ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
            ax.set_xlabel('Composite Score')
            ax.set_ylabel('Count')
            ax.set_title('Score Distribution')
            ax.legend()
        
        # 2. By Complexity
        ax = axes[0, 1]
        if benchmark_results.get('by_complexity'):
            complexities = list(benchmark_results['by_complexity'].keys())
            means = [benchmark_results['by_complexity'][c]['mean'] for c in complexities]
            stds = [benchmark_results['by_complexity'][c]['std'] for c in complexities]
            
            x = range(len(complexities))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([c.capitalize() for c in complexities])
            ax.set_ylabel('Composite Score')
            ax.set_title('Results by Complexity')
        
        # 3. Per-Metric Scores
        ax = axes[1, 0]
        if benchmark_results.get('metric_statistics'):
            metrics = list(benchmark_results['metric_statistics'].keys())
            means = [benchmark_results['metric_statistics'][m]['mean'] for m in metrics]
            
            # Shorten metric names
            short_names = [m.replace('_f1', '').replace('_', '\n')[:15] for m in metrics]
            
            x = range(len(metrics))
            ax.barh(x, means, alpha=0.7)
            ax.set_yticks(x)
            ax.set_yticklabels(short_names, fontsize=8)
            ax.set_xlabel('Score')
            ax.set_title('Per-Metric Results')
            ax.set_xlim(0, 1)
        
        # 4. By Domain
        ax = axes[1, 1]
        if benchmark_results.get('by_domain'):
            domains = list(benchmark_results['by_domain'].keys())
            means = [benchmark_results['by_domain'][d]['mean'] for d in domains]
            counts = [benchmark_results['by_domain'][d]['count'] for d in domains]
            
            # Create pie chart with domain sizes
            ax.pie(counts, labels=domains, autopct='%1.1f%%', startangle=90)
            ax.set_title('Dataset Distribution by Domain')
        
        plt.tight_layout()
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {output_path}.png")
    
    def generate_json_summary(self, benchmark_results: Dict,
                             output_path: str) -> Dict:
        """
        Generate JSON summary for programmatic access.
        
        Args:
            benchmark_results: Benchmark results dictionary
            output_path: Output file path
            
        Returns:
            Summary dictionary
        """
        summary = {
            'overview': {
                'total_notebooks': benchmark_results.get('summary', {}).get('total_notebooks', 0),
                'successful': benchmark_results.get('summary', {}).get('successful', 0),
                'failed': benchmark_results.get('summary', {}).get('failed', 0),
            },
            'composite_score': benchmark_results.get('summary', {}).get('composite_score', {}),
            'by_complexity': benchmark_results.get('by_complexity', {}),
            'by_domain': benchmark_results.get('by_domain', {}),
            'metrics': benchmark_results.get('metric_statistics', {})
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ JSON summary saved to {output_path}")
        return summary


__all__ = [
    "MetricsReporter",
]