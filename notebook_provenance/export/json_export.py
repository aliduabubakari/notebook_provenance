"""
JSON Exporter Module
====================

Export provenance data to JSON format.

This module provides the JSONExporter class which:
- Exports complete provenance to JSON
- Supports various serialization options
- Handles data structure conversion
- Provides filtering options
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from notebook_provenance.core.data_structures import (
    DataFlowGraph,
    DFGNode,
    DFGEdge,
    DataArtifact,
    Transformation,
    PipelineStageNode,
    CellDependency,
)


class JSONExporter:
    """
    Export provenance data to JSON format.
    
    This class converts provenance results to JSON format with
    various options for filtering and formatting.
    
    Example:
        >>> exporter = JSONExporter()
        >>> exporter.export_complete(result, "output.json")
        >>> exporter.export_summary(result, "summary.json")
    """
    
    def __init__(self, indent: int = 2, sort_keys: bool = True):
        """
        Initialize JSON exporter.
        
        Args:
            indent: Indentation level for JSON
            sort_keys: Whether to sort dictionary keys
        """
        self.indent = indent
        self.sort_keys = sort_keys
    
    def export_complete(self, result: Dict, output_file: str) -> Dict:
        """
        Export complete provenance to JSON.
        
        Args:
            result: Complete analysis result
            output_file: Output file path
            
        Returns:
            Dictionary of exported data
        """
        export_data = {
            'metadata': self._build_metadata(result),
            'pipeline_stages': self._export_stages(result.get('stages', [])),
            'data_artifacts': self._export_artifacts(result.get('artifacts', [])),
            'transformations': self._export_transformations(result.get('transformations', [])),
            'column_lineage': result.get('column_lineage', {}),
            'cell_dependencies': self._export_cell_dependencies(result.get('cell_dependencies', {})),
            'graph_statistics': self._export_statistics(result),
            'clean_dfg': self._export_dfg(result.get('clean_dfg')),
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=self.indent, sort_keys=self.sort_keys)
        
        print(f"✓ Complete provenance exported to {output_file}")
        return export_data
    
    def export_summary(self, result: Dict, output_file: str) -> Dict:
        """
        Export summary provenance (without full graph).
        
        Args:
            result: Complete analysis result
            output_file: Output file path
            
        Returns:
            Dictionary of exported summary data
        """
        export_data = {
            'metadata': self._build_metadata(result),
            'pipeline_stages': self._export_stages(result.get('stages', [])),
            'data_artifacts': self._export_artifacts(result.get('artifacts', [])),
            'transformations': self._export_transformations(result.get('transformations', [])),
            'column_lineage': result.get('column_lineage', {}),
            'statistics': self._export_statistics(result),
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=self.indent, sort_keys=self.sort_keys)
        
        print(f"✓ Summary provenance exported to {output_file}")
        return export_data
    
    def export_stages_only(self, stages: List[PipelineStageNode], 
                          output_file: str) -> Dict:
        """
        Export only pipeline stages.
        
        Args:
            stages: List of pipeline stages
            output_file: Output file path
            
        Returns:
            Dictionary of exported stages
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'stages': self._export_stages(stages)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=self.indent, sort_keys=self.sort_keys)
        
        print(f"✓ Pipeline stages exported to {output_file}")
        return export_data
    
    def export_artifacts_only(self, artifacts: List[DataArtifact],
                             transformations: List[Transformation],
                             output_file: str) -> Dict:
        """
        Export only artifacts and transformations.
        
        Args:
            artifacts: List of data artifacts
            transformations: List of transformations
            output_file: Output file path
            
        Returns:
            Dictionary of exported artifacts
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'artifacts': self._export_artifacts(artifacts),
            'transformations': self._export_transformations(transformations)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=self.indent, sort_keys=self.sort_keys)
        
        print(f"✓ Artifacts and transformations exported to {output_file}")
        return export_data
    
    def _build_metadata(self, result: Dict) -> Dict:
        """Build metadata section."""
        return {
            'timestamp': datetime.now().isoformat(),
            'version': '0.2.0',
            'notebook_cells': result.get('statistics', {}).get('total_cells', 0),
            'pipeline_stages': len(result.get('stages', [])),
            'data_artifacts': len(result.get('artifacts', [])),
            'transformations': len(result.get('transformations', [])),
        }
    
    def _export_stages(self, stages: List[PipelineStageNode]) -> List[Dict]:
        """Export pipeline stages."""
        return [
            {
                'id': stage.id,
                'type': stage.stage_type.value,
                'cells': stage.cells,
                'description': stage.description,
                'operations': stage.operations[:10],  # Limit operations
                'input_artifacts': stage.input_artifacts,
                'output_artifacts': stage.output_artifacts,
                'confidence': stage.confidence
            }
            for stage in stages
        ]
    
    def _export_artifacts(self, artifacts: List[DataArtifact]) -> List[Dict]:
        """Export data artifacts."""
        return [
            {
                'id': artifact.id,
                'name': artifact.name,
                'type': artifact.type,
                'created_in_cell': artifact.created_in_cell,
                'importance_score': artifact.importance_score,
                'transformations': artifact.transformations,
                'schema_info': artifact.schema_info,
                'metadata': artifact.metadata
            }
            for artifact in artifacts
        ]
    
    def _export_transformations(self, transformations: List[Transformation]) -> List[Dict]:
        """Export transformations."""
        return [
            {
                'id': trans.id,
                'operation': trans.operation,
                'source_artifacts': trans.source_artifacts,
                'target_artifact': trans.target_artifact,
                'description': trans.description,
                'function_calls': trans.function_calls[:5],  # Limit function calls
                'cell_id': trans.cell_id,
                'semantic_type': trans.semantic_type
            }
            for trans in transformations
        ]
    
    def _export_cell_dependencies(self, cell_dependencies: Dict[str, CellDependency]) -> Dict:
        """Export cell dependencies."""
        return {
            cell_id: dep.to_dict()
            for cell_id, dep in cell_dependencies.items()
        }
    
    def _export_statistics(self, result: Dict) -> Dict:
        """Export statistics."""
        stats = result.get('statistics', {})
        
        return {
            'raw_nodes': stats.get('raw_nodes', 0),
            'raw_edges': len(result.get('raw_dfg', {}).get('edges', [])) if 'raw_dfg' in result else 0,
            'clean_nodes': stats.get('clean_nodes', 0),
            'clean_edges': len(result.get('clean_dfg', {}).get('edges', [])) if 'clean_dfg' in result else 0,
            'noise_reduction': stats.get('noise_reduction', 0),
            'artifacts': stats.get('artifacts', 0),
            'transformations': stats.get('transformations', 0),
            'stages': stats.get('stages', 0),
            'total_cells': stats.get('total_cells', 0),
            'valid_cells': stats.get('valid_cells', 0)
        }
    
    def _export_dfg(self, dfg: Optional[DataFlowGraph]) -> Optional[Dict]:
        """Export data flow graph."""
        if not dfg:
            return None
        
        return {
            'nodes': {
                node_id: {
                    'id': node.id,
                    'label': node.label,
                    'type': node.node_type.value,
                    'cell_id': node.cell_id,
                    'line_number': node.line_number,
                    'code_snippet': node.code_snippet[:100] if node.code_snippet else ""
                }
                for node_id, node in dfg.nodes.items()
            },
            'edges': [
                {
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'type': edge.edge_type.value,
                    'operation': edge.operation
                }
                for edge in dfg.edges
            ],
            'metadata': dfg.metadata
        }
    
    def export_for_paper(self, result: Dict, output_file: str) -> Dict:
        """
        Export data in format suitable for research paper.
        
        This includes aggregated statistics and key findings,
        without verbose details.
        
        Args:
            result: Complete analysis result
            output_file: Output file path
            
        Returns:
            Dictionary of paper-ready data
        """
        export_data = {
            'notebook_id': output_file.split('/')[-1].replace('.json', ''),
            'summary': {
                'total_cells': result.get('statistics', {}).get('total_cells', 0),
                'pipeline_stages': len(result.get('stages', [])),
                'data_artifacts': len(result.get('artifacts', [])),
                'transformations': len(result.get('transformations', [])),
                'noise_reduction_pct': result.get('statistics', {}).get('noise_reduction', 0),
            },
            'pipeline_flow': [
                stage.stage_type.value for stage in result.get('stages', [])
            ],
            'artifact_types': self._aggregate_artifact_types(result.get('artifacts', [])),
            'transformation_types': self._aggregate_transformation_types(result.get('transformations', [])),
            'complexity_metrics': {
                'graph_nodes': result.get('statistics', {}).get('clean_nodes', 0),
                'graph_edges': len(result.get('clean_dfg', DataFlowGraph()).edges),
                'avg_stage_size': self._avg_stage_size(result.get('stages', [])),
            },
            'column_operations': {
                'created': len(result.get('column_lineage', {}).get('created', {})),
                'dropped': len(result.get('column_lineage', {}).get('dropped', {})),
                'renamed': len(result.get('column_lineage', {}).get('renamed', {})),
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=self.indent, sort_keys=self.sort_keys)
        
        print(f"✓ Paper-ready data exported to {output_file}")
        return export_data
    
    def _aggregate_artifact_types(self, artifacts: List[DataArtifact]) -> Dict:
        """Aggregate artifact types."""
        from collections import Counter
        types = Counter(a.type for a in artifacts)
        return dict(types)
    
    def _aggregate_transformation_types(self, transformations: List[Transformation]) -> Dict:
        """Aggregate transformation types."""
        from collections import Counter
        types = Counter(t.semantic_type for t in transformations if t.semantic_type)
        return dict(types)
    
    def _avg_stage_size(self, stages: List[PipelineStageNode]) -> float:
        """Calculate average stage size (cells per stage)."""
        if not stages:
            return 0.0
        return sum(len(s.cells) for s in stages) / len(stages)


__all__ = [
    "JSONExporter",
]