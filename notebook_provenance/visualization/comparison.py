"""
Comparison Visualizer Module
=============================

Notebook comparison visualizations.

This module provides the ComparisonVisualizer class which:
- Compares multiple notebook analyses
- Generates side-by-side visualizations
- Creates diff reports
- Highlights differences
"""

from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import json

from notebook_provenance.core.data_structures import (
    DataArtifact,
    PipelineStageNode,
    Transformation,
)
from notebook_provenance.core.enums import PipelineStage


class ComparisonVisualizer:
    """
    Visualize and compare multiple notebook analyses.
    
    This class creates visualizations that highlight differences
    between multiple notebooks or versions of the same notebook.
    
    Example:
        >>> visualizer = ComparisonVisualizer()
        >>> fig = visualizer.visualize_stage_comparison([result1, result2], names)
        >>> report = visualizer.generate_diff_report(result1, result2)
    """
    
    def __init__(self):
        """Initialize comparison visualizer."""
        # Color schemes for differences
        self.diff_colors = {
            'added': '#e6ffe6',
            'removed': '#ffe6e6',
            'changed': '#fff3e6',
            'unchanged': '#f5f5f5'
        }
        
        self.stage_colors = {
            PipelineStage.SETUP: '#E8E8E8',
            PipelineStage.DATA_LOADING: '#AED6F1',
            PipelineStage.DATA_PREPARATION: '#A9DFBF',
            PipelineStage.RECONCILIATION: '#FAD7A0',
            PipelineStage.ENRICHMENT: '#F9E79F',
            PipelineStage.TRANSFORMATION: '#F5B7B1',
            PipelineStage.ANALYSIS: '#D7BDE2',
            PipelineStage.OUTPUT: '#D5DBDB'
        }
    
    def visualize_stage_comparison(self, results: List[Dict], 
                                   names: List[str],
                                   figsize: Tuple = (20, 6)) -> plt.Figure:
        """
        Create side-by-side comparison of pipeline stages.
        
        Args:
            results: List of analysis results
            names: List of notebook names
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_notebooks = len(results)
        fig, axes = plt.subplots(1, n_notebooks, figsize=figsize)
        
        # Handle single notebook case
        if n_notebooks == 1:
            axes = [axes]
        
        for idx, (result, name) in enumerate(zip(results, names)):
            ax = axes[idx]
            stages = result.get('stages', [])
            
            if not stages:
                ax.text(0.5, 0.5, "No stages", ha='center', va='center')
                ax.set_title(name)
                ax.axis('off')
                continue
            
            # Draw stages vertically
            y_spacing = 1.5
            for i, stage in enumerate(stages):
                y = -i * y_spacing
                
                # Get color
                color = self.stage_colors.get(stage.stage_type, '#FFFFFF')
                
                # Draw stage box
                rect = FancyBboxPatch(
                    (-0.8, y - 0.5), 1.6, 1.0,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor='#333',
                    linewidth=2,
                    alpha=0.9
                )
                ax.add_patch(rect)
                
                # Stage name
                stage_name = stage.stage_type.value.replace('_', '\n')
                ax.text(0, y, stage_name,
                       ha='center', va='center',
                       fontsize=9, fontweight='bold')
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-len(stages) * y_spacing, 1)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle('Pipeline Stage Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_artifact_comparison(self, results: List[Dict],
                                     names: List[str],
                                     figsize: Tuple = (20, 8)) -> plt.Figure:
        """
        Create side-by-side comparison of artifacts.
        
        Args:
            results: List of analysis results
            names: List of notebook names
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_notebooks = len(results)
        fig, axes = plt.subplots(1, n_notebooks, figsize=figsize)
        
        if n_notebooks == 1:
            axes = [axes]
        
        for idx, (result, name) in enumerate(zip(results, names)):
            ax = axes[idx]
            artifacts = result.get('artifacts', [])
            
            if not artifacts:
                ax.text(0.5, 0.5, "No artifacts", ha='center', va='center')
                ax.set_title(name)
                ax.axis('off')
                continue
            
            # Sort by importance
            artifacts = sorted(artifacts, key=lambda a: a.importance_score, reverse=True)[:10]
            
            # Draw artifacts as horizontal bars
            y_pos = range(len(artifacts))
            scores = [a.importance_score for a in artifacts]
            labels = [a.name[:20] for a in artifacts]
            
            bars = ax.barh(y_pos, scores, color='#3498DB', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Importance Score', fontsize=10)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Invert y-axis to have highest at top
            ax.invert_yaxis()
        
        fig.suptitle('Data Artifact Comparison (Top 10 by Importance)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def generate_diff_report(self, base: Dict, other: Dict, 
                            base_name: str = "Base",
                            other_name: str = "Other") -> str:
        """
        Generate human-readable diff report.
        
        Args:
            base: Base analysis result
            other: Other analysis result to compare
            base_name: Name of base notebook
            other_name: Name of other notebook
            
        Returns:
            Markdown-formatted diff report
        """
        report = []
        
        report.append(f"# Notebook Comparison Report\n")
        report.append(f"**Base:** {base_name}")
        report.append(f"**Compared:** {other_name}\n")
        report.append("---\n")
        
        # Pipeline structure differences
        report.append("## Pipeline Structure Differences\n")
        
        base_stages = [s.stage_type.value for s in base.get('stages', [])]
        other_stages = [s.stage_type.value for s in other.get('stages', [])]
        
        if base_stages != other_stages:
            report.append(f"- **Base pipeline:** {' → '.join(base_stages)}")
            report.append(f"- **Compared pipeline:** {' → '.join(other_stages)}")
            
            added_stages = set(other_stages) - set(base_stages)
            removed_stages = set(base_stages) - set(other_stages)
            
            if added_stages:
                report.append(f"- **Stages added:** {', '.join(added_stages)}")
            if removed_stages:
                report.append(f"- **Stages removed:** {', '.join(removed_stages)}")
        else:
            report.append("✓ Pipeline structure is identical\n")
        
        report.append("")
        
        # Data artifact differences
        report.append("## Data Artifact Differences\n")
        
        base_artifacts = {(a.name, a.type) for a in base.get('artifacts', [])}
        other_artifacts = {(a.name, a.type) for a in other.get('artifacts', [])}
        
        new_artifacts = other_artifacts - base_artifacts
        removed_artifacts = base_artifacts - other_artifacts
        
        if new_artifacts:
            report.append(f"- **New artifacts ({len(new_artifacts)}):**")
            for name, atype in sorted(new_artifacts):
                report.append(f"  - {name} ({atype})")
        
        if removed_artifacts:
            report.append(f"- **Removed artifacts ({len(removed_artifacts)}):**")
            for name, atype in sorted(removed_artifacts):
                report.append(f"  - {name} ({atype})")
        
        if not new_artifacts and not removed_artifacts:
            report.append("✓ Artifacts are identical\n")
        
        report.append("")
        
        # Transformation differences
        report.append("## Transformation Differences\n")
        
        base_trans_count = len(base.get('transformations', []))
        other_trans_count = len(other.get('transformations', []))
        
        report.append(f"- **Base transformations:** {base_trans_count}")
        report.append(f"- **Compared transformations:** {other_trans_count}")
        report.append(f"- **Difference:** {other_trans_count - base_trans_count:+d}\n")
        
        # Column lineage differences
        if 'column_lineage' in base and 'column_lineage' in other:
            report.append("## Column Lineage Differences\n")
            
            base_cols = set(base['column_lineage'].get('created', {}).keys())
            other_cols = set(other['column_lineage'].get('created', {}).keys())
            
            added_cols = other_cols - base_cols
            removed_cols = base_cols - other_cols
            
            if added_cols:
                report.append(f"- **New columns ({len(added_cols)}):** {', '.join(sorted(added_cols))}")
            if removed_cols:
                report.append(f"- **Removed columns ({len(removed_cols)}):** {', '.join(sorted(removed_cols))}")
            
            if not added_cols and not removed_cols:
                report.append("✓ Column set is identical\n")
        
        report.append("")
        
        # Statistics comparison
        report.append("## Statistics Comparison\n")
        report.append("| Metric | Base | Compared | Difference |")
        report.append("|--------|------|----------|------------|")
        
        metrics = ['total_cells', 'artifacts', 'transformations', 'stages']
        for metric in metrics:
            base_val = base.get('statistics', {}).get(metric, 0)
            other_val = other.get('statistics', {}).get(metric, 0)
            diff = other_val - base_val
            report.append(f"| {metric} | {base_val} | {other_val} | {diff:+d} |")
        
        return '\n'.join(report)
    
    def create_comparison_html(self, results: List[Dict], names: List[str],
                              output_file: str = "comparison.html"):
        """
        Create interactive HTML comparison.
        
        Args:
            results: List of analysis results
            names: List of notebook names
            output_file: Output file path
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Notebook Comparison</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
            background: #f5f5f5; 
            padding: 20px;
        }}
        .header {{ 
            background: white; 
            padding: 20px; 
            margin-bottom: 20px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .comparison-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 20px; 
            margin-bottom: 20px;
        }}
        .notebook-panel {{ 
            background: white; 
            border-radius: 8px; 
            padding: 20px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .notebook-panel h2 {{ 
            color: #333; 
            margin-bottom: 15px; 
            padding-bottom: 10px; 
            border-bottom: 2px solid #3498DB;
        }}
        .metric {{ 
            display: flex; 
            justify-content: space-between; 
            padding: 8px 0; 
            border-bottom: 1px solid #eee;
        }}
        .metric-label {{ color: #666; }}
        .metric-value {{ font-weight: bold; color: #333; }}
        .stage-list {{ 
            list-style: none; 
            padding: 0;
        }}
        .stage-item {{ 
            background: #f8f9fa; 
            margin: 5px 0; 
            padding: 10px; 
            border-radius: 4px;
            border-left: 4px solid #3498DB;
        }}
        .artifact-list {{ 
            list-style: none; 
            padding: 0;
        }}
        .artifact-item {{ 
            padding: 8px 0; 
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }}
        .diff-section {{ 
            background: white; 
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .diff-added {{ 
            background: #e6ffe6; 
            padding: 5px 10px; 
            border-radius: 4px; 
            margin: 5px 0;
        }}
        .diff-removed {{ 
            background: #ffe6e6; 
            padding: 5px 10px; 
            border-radius: 4px; 
            margin: 5px 0;
        }}
        .diff-changed {{ 
            background: #fff3e6; 
            padding: 5px 10px; 
            border-radius: 4px; 
            margin: 5px 0;
        }}
        .badge {{ 
            display: inline-block; 
            padding: 2px 8px; 
            border-radius: 12px; 
            font-size: 12px; 
            font-weight: bold;
        }}
        .badge-added {{ background: #2ECC71; color: white; }}
        .badge-removed {{ background: #E74C3C; color: white; }}
        .badge-changed {{ background: #F39C12; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Notebook Comparison</h1>
        <p>Comparing {len(results)} notebooks: {', '.join(names)}</p>
    </div>
"""
        
        # Summary comparison
        html += self._build_summary_section(results, names)
        
        # Side-by-side comparison
        html += '<div class="comparison-grid">'
        for result, name in zip(results, names):
            html += self._build_notebook_panel(result, name)
        html += '</div>'
        
        # Differences section (if comparing 2)
        if len(results) == 2:
            html += self._build_diff_section(results[0], results[1], names[0], names[1])
        
        html += """
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Comparison HTML saved to {output_file}")
    
    def _build_summary_section(self, results: List[Dict], names: List[str]) -> str:
        """Build summary comparison section."""
        html = '<div class="diff-section">'
        html += '<h2>Summary Comparison</h2>'
        html += '<table style="width: 100%; border-collapse: collapse; margin-top: 15px;">'
        html += '<thead><tr style="background: #f8f9fa;">'
        html += '<th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Metric</th>'
        
        for name in names:
            html += f'<th style="padding: 10px; text-align: center; border: 1px solid #ddd;">{name}</th>'
        html += '</tr></thead><tbody>'
        
        metrics = [
            ('Pipeline Stages', 'stages'),
            ('Data Artifacts', 'artifacts'),
            ('Transformations', 'transformations'),
            ('Total Cells', 'total_cells'),
            ('Graph Nodes', 'clean_nodes')
        ]
        
        for label, key in metrics:
            html += f'<tr><td style="padding: 10px; border: 1px solid #ddd;">{label}</td>'
            
            values = []
            for result in results:
                if key in ['stages', 'artifacts', 'transformations']:
                    value = len(result.get(key, []))
                else:
                    value = result.get('statistics', {}).get(key, 0)
                values.append(value)
                html += f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd;">{value}</td>'
            
            html += '</tr>'
        
        html += '</tbody></table></div>'
        return html
    
    def _build_notebook_panel(self, result: Dict, name: str) -> str:
        """Build individual notebook panel."""
        html = f'<div class="notebook-panel">'
        html += f'<h2>{name}</h2>'
        
        # Statistics
        stats = result.get('statistics', {})
        html += '<div style="margin-bottom: 20px;">'
        html += f'<div class="metric"><span class="metric-label">Total Cells</span><span class="metric-value">{stats.get("total_cells", 0)}</span></div>'
        html += f'<div class="metric"><span class="metric-label">Pipeline Stages</span><span class="metric-value">{len(result.get("stages", []))}</span></div>'
        html += f'<div class="metric"><span class="metric-label">Data Artifacts</span><span class="metric-value">{len(result.get("artifacts", []))}</span></div>'
        html += f'<div class="metric"><span class="metric-label">Transformations</span><span class="metric-value">{len(result.get("transformations", []))}</span></div>'
        html += '</div>'
        
        # Pipeline stages
        stages = result.get('stages', [])
        if stages:
            html += '<h3 style="margin-top: 20px; margin-bottom: 10px;">Pipeline Flow</h3>'
            html += '<ul class="stage-list">'
            for stage in stages:
                html += f'<li class="stage-item">{stage.stage_type.value.replace("_", " ").title()}</li>'
            html += '</ul>'
        
        # Top artifacts
        artifacts = result.get('artifacts', [])
        if artifacts:
            top_artifacts = sorted(artifacts, key=lambda a: a.importance_score, reverse=True)[:5]
            html += '<h3 style="margin-top: 20px; margin-bottom: 10px;">Top Artifacts</h3>'
            html += '<ul class="artifact-list">'
            for artifact in top_artifacts:
                html += f'<li class="artifact-item">'
                html += f'<span>{artifact.name}</span>'
                html += f'<span style="color: #666; font-size: 12px;">{artifact.type}</span>'
                html += '</li>'
            html += '</ul>'
        
        html += '</div>'
        return html
    
    def _build_diff_section(self, base: Dict, other: Dict, 
                           base_name: str, other_name: str) -> str:
        """Build differences section."""
        html = '<div class="diff-section">'
        html += f'<h2>Differences: {base_name} vs {other_name}</h2>'
        
        # Stage differences
        base_stages = {s.stage_type.value for s in base.get('stages', [])}
        other_stages = {s.stage_type.value for s in other.get('stages', [])}
        
        added_stages = other_stages - base_stages
        removed_stages = base_stages - other_stages
        
        if added_stages or removed_stages:
            html += '<h3 style="margin-top: 20px;">Pipeline Stages</h3>'
            if added_stages:
                html += '<div class="diff-added">'
                html += '<span class="badge badge-added">ADDED</span> '
                html += ', '.join(s.replace('_', ' ').title() for s in added_stages)
                html += '</div>'
            if removed_stages:
                html += '<div class="diff-removed">'
                html += '<span class="badge badge-removed">REMOVED</span> '
                html += ', '.join(s.replace('_', ' ').title() for s in removed_stages)
                html += '</div>'
        
        # Artifact differences
        base_artifacts = {(a.name, a.type) for a in base.get('artifacts', [])}
        other_artifacts = {(a.name, a.type) for a in other.get('artifacts', [])}
        
        new_artifacts = other_artifacts - base_artifacts
        removed_artifacts = base_artifacts - other_artifacts
        
        if new_artifacts or removed_artifacts:
            html += '<h3 style="margin-top: 20px;">Data Artifacts</h3>'
            if new_artifacts:
                html += '<div class="diff-added">'
                html += '<span class="badge badge-added">ADDED</span> '
                html += ', '.join(f'{name} ({atype})' for name, atype in new_artifacts)
                html += '</div>'
            if removed_artifacts:
                html += '<div class="diff-removed">'
                html += '<span class="badge badge-removed">REMOVED</span> '
                html += ', '.join(f'{name} ({atype})' for name, atype in removed_artifacts)
                html += '</div>'
        
        # Column differences
        if 'column_lineage' in base and 'column_lineage' in other:
            base_cols = set(base['column_lineage'].get('created', {}).keys())
            other_cols = set(other['column_lineage'].get('created', {}).keys())
            
            added_cols = other_cols - base_cols
            removed_cols = base_cols - other_cols
            
            if added_cols or removed_cols:
                html += '<h3 style="margin-top: 20px;">Columns</h3>'
                if added_cols:
                    html += '<div class="diff-added">'
                    html += '<span class="badge badge-added">ADDED</span> '
                    html += ', '.join(sorted(added_cols))
                    html += '</div>'
                if removed_cols:
                    html += '<div class="diff-removed">'
                    html += '<span class="badge badge-removed">REMOVED</span> '
                    html += ', '.join(sorted(removed_cols))
                    html += '</div>'
        
        html += '</div>'
        return html
    
    def compare_multiple_summary(self, results: List[Dict], 
                                names: List[str]) -> Dict[str, Any]:
        """
        Generate summary comparison data.
        
        Args:
            results: List of analysis results
            names: List of notebook names
            
        Returns:
            Dictionary with comparison summary
        """
        summary = {
            'notebooks': names,
            'count': len(results),
            'metrics': {},
            'commonalities': {},
            'differences': {}
        }
        
        # Collect metrics
        for metric in ['stages', 'artifacts', 'transformations']:
            summary['metrics'][metric] = [
                len(result.get(metric, [])) for result in results
            ]
        
        # Find common stages
        all_stages = [
            {s.stage_type.value for s in result.get('stages', [])}
            for result in results
        ]
        if all_stages:
            common_stages = set.intersection(*all_stages)
            summary['commonalities']['stages'] = list(common_stages)
        
        # Find common artifacts
        all_artifacts = [
            {(a.name, a.type) for a in result.get('artifacts', [])}
            for result in results
        ]
        if all_artifacts:
            common_artifacts = set.intersection(*all_artifacts)
            summary['commonalities']['artifacts'] = [
                {'name': name, 'type': atype} 
                for name, atype in common_artifacts
            ]
        
        return summary


__all__ = [
    "ComparisonVisualizer",
]