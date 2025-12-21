"""
Provenance Visualizer Module
=============================

Multi-level static provenance visualizations.

This module provides the ProvenanceVisualizer class which:
- Visualizes pipeline stages
- Visualizes artifact lineage
- Visualizes data flow graphs
- Creates simplified views
"""

from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

from notebook_provenance.core.data_structures import (
    DataFlowGraph,
    DataArtifact,
    Transformation,
    PipelineStageNode,
)
from notebook_provenance.core.enums import PipelineStage, NodeType
from notebook_provenance.core.config import VisualizationConfig


class ProvenanceVisualizer:
    """
    Multi-level provenance visualization.
    
    This class creates various static visualizations of the notebook
    provenance, including pipeline stages, artifact lineage, and data flow.
    
    Example:
        >>> visualizer = ProvenanceVisualizer()
        >>> fig = visualizer.visualize_pipeline_stages(stages)
        >>> fig.savefig('stages.png', dpi=300, bbox_inches='tight')
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Optional visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Color schemes
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
        
        self.artifact_colors = {
            'dataframe': '#3498DB',
            'table': '#E74C3C',
            'model': '#2ECC71',
            'result': '#F39C12',
            'matrix': '#9B59B6',
            'unknown': '#95A5A6'
        }
        
        self.node_type_colors = {
            NodeType.DATA_ARTIFACT: '#2ECC71',
            NodeType.FUNCTION_CALL: '#E74C3C',
            NodeType.VARIABLE: '#3498DB',
            NodeType.INTERMEDIATE: '#F39C12',
            NodeType.ATTRIBUTE_ACCESS: '#9B59B6',
        }
    
    @staticmethod
    def _wrap_text(text: str, max_chars: int = 35, max_lines: int = 2) -> str:
        """
        Wrap text for display.
        
        Args:
            text: Text to wrap
            max_chars: Maximum characters per line
            max_lines: Maximum number of lines
            
        Returns:
            Wrapped text
        """
        words = text.split()
        lines = []
        current = []
        length = 0
        
        for w in words:
            word_len = len(w)
            if length + word_len + len(current) > max_chars:
                if current:
                    lines.append(' '.join(current))
                    current = [w]
                    length = word_len
                else:
                    # Single word longer than max_chars
                    lines.append(w[:max_chars])
                    length = 0
            else:
                current.append(w)
                length += word_len
        
        if current:
            lines.append(' '.join(current))
        
        # Limit to max_lines
        lines = lines[:max_lines]
        
        # Add ellipsis if truncated
        if len(lines) == max_lines and len(' '.join(lines)) < len(text):
            last = lines[-1]
            if not last.endswith('...'):
                lines[-1] = last[:max(0, len(last)-3)] + '...'
        
        return '\n'.join(lines)
    
    def visualize_pipeline_stages(self, stages: List[PipelineStageNode],
                                  figsize: Optional[Tuple] = None) -> plt.Figure:
        """
        Visualize high-level pipeline stages with improved layout.
        
        Args:
            stages: List of pipeline stages
            figsize: Optional figure size
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.config.figsize_stages
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if not stages:
            ax.text(0.5, 0.5, "No pipeline stages identified",
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        # Calculate positions for horizontal flow
        n_stages = len(stages)
        spacing = 5
        
        for i, stage in enumerate(stages):
            x = i * spacing
            y = 0
            
            # Draw stage box
            color = self.stage_colors.get(stage.stage_type, '#FFFFFF')
            
            rect = FancyBboxPatch(
                (x - 1.5, y - 0.8), 3, 1.6,
                boxstyle="round,pad=0.15,rounding_size=0.05",
                facecolor=color,
                edgecolor='#333333',
                linewidth=2.5,
                alpha=0.95,
                zorder=10
            )
            ax.add_patch(rect)
            
            # Add stage name
            stage_name = stage.stage_type.value.replace('_', '\n').title()
            ax.text(x, y, stage_name,
                   ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   zorder=11)
            
            # Add wrapped description below
            description = self._wrap_text(stage.description or "", max_chars=40, max_lines=2)
            ax.text(x, y - 1.55, description,
                   ha='center', va='top',
                   fontsize=9, style='italic',
                   color='#555',
                   bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', 
                             edgecolor='none',
                             alpha=0.7),
                   zorder=9)
            
            # Add cell count above
            cell_text = f"{len(stage.cells)} cell{'s' if len(stage.cells) > 1 else ''}"
            ax.text(x, y + 1.1, cell_text,
                   ha='center', va='bottom',
                   fontsize=9, color='#666',
                   zorder=11)
            
            # Draw arrow to next stage
            if i < n_stages - 1:
                arrow = FancyArrowPatch(
                    (x + 1.5, y), (x + spacing - 1.5, y),
                    arrowstyle='-|>',
                    mutation_scale=30,
                    linewidth=3,
                    color='#777',
                    alpha=0.7,
                    zorder=5
                )
                ax.add_patch(arrow)
        
        ax.set_xlim(-2, n_stages * spacing)
        ax.set_ylim(-3.2, 2.2)
        ax.axis('off')
        ax.set_title("Pipeline Stages (High-Level View)", 
                    fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def visualize_artifact_lineage(self, artifacts: List[DataArtifact],
                               transformations: List[Transformation],
                               figsize: Optional[Tuple] = None) -> plt.Figure:
        """
        Visualize artifact lineage with FIXED hierarchical layout.
        """
        if figsize is None:
            figsize = self.config.figsize_lineage
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if not artifacts:
            ax.text(0.5, 0.5, "No data artifacts identified",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        # Build transformation graph
        G = nx.DiGraph()
        
        for artifact in artifacts:
            G.add_node(artifact.id, artifact=artifact)
        
        # Add edges from transformations
        for trans in transformations:
            for source_id in trans.source_artifacts:
                if source_id in G.nodes() and trans.target_artifact in G.nodes():
                    G.add_edge(source_id, trans.target_artifact, 
                            transformation=trans, label=trans.description[:20])
        
        # Try different layouts for better visualization
        if G.number_of_nodes() > 0:
            try:
                # Try graphviz DOT layout (best for DAGs)
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR -Gnodesep=2 -Granksep=3')
            except:
                try:
                    # Try Kamada-Kawai for good node separation
                    pos = nx.kamada_kawai_layout(G, scale=5)
                except:
                    # Fall back to spring layout with larger k
                    pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
        else:
            pos = {}
        
        # Scale positions for better visibility
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            
            x_range = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 1
            y_range = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 1
            
            # Normalize and scale
            for node_id in pos:
                x, y = pos[node_id]
                pos[node_id] = (
                    (x - min(x_coords)) / max(x_range, 0.1) * 20,
                    (y - min(y_coords)) / max(y_range, 0.1) * 10
                )
        
        # Get importance range for sizing
        if artifacts:
            min_imp = min(a.importance_score for a in artifacts)
            max_imp = max(a.importance_score for a in artifacts)
            imp_range = max(1, max_imp - min_imp)
        else:
            min_imp, imp_range = 0, 1
        
        # Draw artifacts
        for artifact in artifacts:
            if artifact.id not in pos:
                continue
            
            x, y = pos[artifact.id]
            color = self.artifact_colors.get(artifact.type, '#95A5A6')
            
            # Size based on importance
            norm_imp = (artifact.importance_score - min_imp) / imp_range
            width = 2.5 + norm_imp * 1.5
            height = 1.2
            
            # Draw box
            rect = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.15",
                facecolor=color,
                edgecolor='#333',
                linewidth=2,
                alpha=0.85,
                zorder=10
            )
            ax.add_patch(rect)
            
            # Name (truncate if needed)
            name = artifact.name if len(artifact.name) <= 18 else artifact.name[:15] + '...'
            ax.text(x, y + 0.15, name,
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white', zorder=11)
            
            # Type
            ax.text(x, y - 0.25, artifact.type,
                    ha='center', va='center',
                    fontsize=8, style='italic',
                    color='white', zorder=11)
        
        # Draw transformation edges
        for trans in transformations:
            for source_id in trans.source_artifacts:
                if source_id not in pos or trans.target_artifact not in pos:
                    continue
                
                x1, y1 = pos[source_id]
                x2, y2 = pos[trans.target_artifact]
                
                # Calculate edge endpoints (outside boxes)
                dx = x2 - x1
                dy = y2 - y1
                dist = (dx**2 + dy**2)**0.5
                
                if dist > 0:
                    # Offset from center of boxes
                    offset = 1.3
                    x1_adj = x1 + (dx/dist) * offset
                    y1_adj = y1 + (dy/dist) * 0.6
                    x2_adj = x2 - (dx/dist) * offset
                    y2_adj = y2 - (dy/dist) * 0.6
                    
                    arrow = FancyArrowPatch(
                        (x1_adj, y1_adj), (x2_adj, y2_adj),
                        arrowstyle='-|>',
                        mutation_scale=20,
                        linewidth=2,
                        color='#555',
                        alpha=0.7,
                        connectionstyle="arc3,rad=0.1",
                        zorder=5
                    )
                    ax.add_patch(arrow)
                    
                    # Add label at midpoint
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2 + 0.5
                    
                    desc = trans.description if len(trans.description) <= 25 else trans.description[:22] + '...'
                    ax.text(mid_x, mid_y, desc,
                            ha='center', va='bottom',
                            fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='white',
                                    edgecolor='#ccc',
                                    alpha=0.9),
                            zorder=6)
        
        # Set limits with padding
        if pos:
            all_x = [p[0] for p in pos.values()]
            all_y = [p[1] for p in pos.values()]
            ax.set_xlim(min(all_x) - 3, max(all_x) + 3)
            ax.set_ylim(min(all_y) - 2, max(all_y) + 2)
        
        ax.axis('off')
        ax.set_title("Data Artifact Lineage", fontsize=16, fontweight='bold', pad=20)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=color, label=atype.title(), edgecolor='#333')
            for atype, color in self.artifact_colors.items()
            if atype != 'unknown'
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        return fig

    def visualize_simplified_lineage(self, artifacts: List[DataArtifact],
                                 transformations: List[Transformation],
                                 figsize: Optional[Tuple] = None) -> plt.Figure:
        """
        Create simplified linear view showing the MAIN data flow path.
        
        FIXED: 
        - Only show core data artifacts
        - Order by actual data flow, not just cell order
        - Start from source (df) and follow transformations
        """
        if figsize is None:
            figsize = (18, 5)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if not artifacts:
            ax.text(0.5, 0.5, "No data artifacts identified",
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        # Filter to CORE data artifacts only
        # Exclude: payloads, html_table, metadata IDs
        exclude_patterns = ['payload', 'html', 'id', 'name', 'sent']
        
        core_artifacts = [
            a for a in artifacts
            if not any(p in a.name.lower() for p in exclude_patterns)
            and a.type in ['dataframe', 'table', 'result']
        ]
        
        if not core_artifacts:
            # Fallback: just use top artifacts by importance
            core_artifacts = sorted(artifacts, key=lambda a: a.importance_score, reverse=True)[:5]
        
        # Build transformation graph to determine order
        flow_order = self._determine_data_flow_order(core_artifacts, transformations)
        
        if not flow_order:
            # Fallback to cell order
            flow_order = sorted(core_artifacts, key=lambda a: a.created_in_cell)
        
        # Limit to reasonable number
        main_artifacts = flow_order[:6]
        
        # Draw
        spacing = 4.5
        for i, artifact in enumerate(main_artifacts):
            x = i * spacing
            y = 0
            color = self.artifact_colors.get(artifact.type, '#95A5A6')
            
            # Box
            rect = FancyBboxPatch(
                (x - 1.4, y - 0.65), 2.8, 1.3,
                boxstyle="round,pad=0.12",
                facecolor=color,
                edgecolor='#333',
                linewidth=2,
                alpha=0.9,
                zorder=10
            )
            ax.add_patch(rect)
            
            # Name
            name = artifact.name if len(artifact.name) <= 14 else artifact.name[:11] + '...'
            ax.text(x, y + 0.12, name,
                    ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white', zorder=11)
            
            # Type
            ax.text(x, y - 0.28, artifact.type,
                    ha='center', va='center',
                    fontsize=8, color='white', zorder=11)
            
            # Arrow to next
            if i < len(main_artifacts) - 1:
                next_artifact = main_artifacts[i + 1]
                
                # Find transformation label
                trans_label = self._find_transformation_label(
                    artifact, next_artifact, transformations
                )
                
                arrow = FancyArrowPatch(
                    (x + 1.5, y), (x + spacing - 1.5, y),
                    arrowstyle='-|>',
                    mutation_scale=20,
                    linewidth=2.5,
                    color='#666',
                    alpha=0.75,
                    zorder=5
                )
                ax.add_patch(arrow)
                
                # Label
                if trans_label:
                    ax.text(x + spacing/2, y + 0.7, trans_label,
                            ha='center', va='bottom',
                            fontsize=8, style='italic',
                            bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='white',
                                    edgecolor='#ddd',
                                    alpha=0.9),
                            zorder=6)
        
        ax.set_xlim(-2, len(main_artifacts) * spacing)
        ax.set_ylim(-1.5, 1.8)
        ax.axis('off')
        ax.set_title("Main Data Flow (Simplified)", fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        return fig

    def _determine_data_flow_order(self, artifacts: List[DataArtifact],
                                transformations: List[Transformation]) -> List[DataArtifact]:
        """
        Determine the order of artifacts based on actual data flow.
        
        Uses topological sort on the transformation graph.
        """
        import networkx as nx
        
        G = nx.DiGraph()
        artifact_by_id = {a.id: a for a in artifacts}
        artifact_by_name = {a.name: a for a in artifacts}
        
        for a in artifacts:
            G.add_node(a.id)
        
        for trans in transformations:
            target = trans.target_artifact
            for source in trans.source_artifacts:
                if source in artifact_by_id and target in artifact_by_id:
                    G.add_edge(source, target)
                # Also check by name (in case IDs don't match)
                elif source in artifact_by_name and target in artifact_by_name:
                    G.add_edge(artifact_by_name[source].id, artifact_by_name[target].id)
        
        # Topological sort
        try:
            sorted_ids = list(nx.topological_sort(G))
            return [artifact_by_id[aid] for aid in sorted_ids if aid in artifact_by_id]
        except nx.NetworkXUnfeasible:
            # Cycle detected, fall back to importance order
            return sorted(artifacts, key=lambda a: a.importance_score, reverse=True)

    def _find_transformation_label(self, source: DataArtifact, 
                                target: DataArtifact,
                                transformations: List[Transformation]) -> str:
        """Find the transformation label between two artifacts."""
        for trans in transformations:
            if (source.id in trans.source_artifacts and trans.target_artifact == target.id):
                if trans.description:
                    return trans.description[:25] + ('...' if len(trans.description) > 25 else '')
                return trans.operation[:20] if trans.operation else '→'
            # Also check by name
            if (source.name in trans.source_artifacts and trans.target_artifact == target.name):
                if trans.description:
                    return trans.description[:25] + ('...' if len(trans.description) > 25 else '')
                return trans.operation[:20] if trans.operation else '→'
        
        return '→'
       
    def visualize_clean_dfg(self, clean_dfg: DataFlowGraph,
                           figsize: Optional[Tuple] = None) -> plt.Figure:
        """
        Visualize cleaned data flow graph.
        
        Args:
            clean_dfg: Cleaned data flow graph
            figsize: Optional figure size
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.config.figsize_dfg
        
        fig, ax = plt.subplots(figsize=figsize)
        
        G = clean_dfg.to_networkx()
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No nodes in cleaned graph",
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        # Layout
        try:
            pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G)
        
        # Prepare node colors and sizes
        node_colors = []
        node_sizes = []
        
        for node_id in G.nodes():
            node = clean_dfg.nodes[node_id]
            
            # Get color
            color = self.node_type_colors.get(node.node_type, '#95A5A6')
            node_colors.append(color)
            
            # Get size
            if node.node_type == NodeType.DATA_ARTIFACT:
                node_sizes.append(3000)
            elif node.node_type == NodeType.FUNCTION_CALL:
                node_sizes.append(2500)
            elif node.node_type == NodeType.VARIABLE:
                node_sizes.append(2000)
            else:
                node_sizes.append(1500)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='#333',
            linewidths=2,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#666',
            arrows=True,
            arrowsize=20,
            width=1.5,
            alpha=0.6,
            ax=ax,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels
        labels = {}
        for node_id in G.nodes():
            node = clean_dfg.nodes[node_id]
            label = node.label
            if len(label) > 15:
                label = label[:12] + '...'
            labels[node_id] = label
        
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title("Clean Data Flow Graph", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#2ECC71', label='Data Artifact', edgecolor='#333'),
            mpatches.Patch(facecolor='#E74C3C', label='Function Call', edgecolor='#333'),
            mpatches.Patch(facecolor='#3498DB', label='Variable', edgecolor='#333'),
            mpatches.Patch(facecolor='#F39C12', label='Intermediate', edgecolor='#333'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_all_visualizations(self, result: Dict) -> Dict[str, plt.Figure]:
        """
        Create all standard visualizations.
        
        Args:
            result: Complete analysis result
            
        Returns:
            Dictionary mapping visualization names to figures
        """
        visualizations = {}
        
        # Pipeline stages
        if 'stages' in result and result['stages']:
            try:
                visualizations['stages'] = self.visualize_pipeline_stages(result['stages'])
            except Exception as e:
                print(f"Warning: Failed to create stages visualization: {e}")
        
        # Artifact lineage
        if 'artifacts' in result and 'transformations' in result:
            try:
                visualizations['lineage'] = self.visualize_artifact_lineage(
                    result['artifacts'],
                    result['transformations']
                )
            except Exception as e:
                print(f"Warning: Failed to create lineage visualization: {e}")
        
        # Simplified lineage
        if 'artifacts' in result and 'transformations' in result:
            try:
                visualizations['simplified_lineage'] = self.visualize_simplified_lineage(
                    result['artifacts'],
                    result['transformations']
                )
            except Exception as e:
                print(f"Warning: Failed to create simplified lineage: {e}")
        
        # Clean DFG
        if 'clean_dfg' in result:
            try:
                visualizations['clean_dfg'] = self.visualize_clean_dfg(result['clean_dfg'])
            except Exception as e:
                print(f"Warning: Failed to create DFG visualization: {e}")
        
        return visualizations


__all__ = [
    "ProvenanceVisualizer",
]