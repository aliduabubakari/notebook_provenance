"""
Transformation Extractor Module
================================

Extract and describe transformations between artifacts.

This module provides the TransformationExtractor class which:
- Identifies transformations between data artifacts
- Extracts operation details
- Generates human-readable descriptions
- Optionally uses LLM for enhanced descriptions
"""

from typing import Dict, List, Optional, Set
import networkx as nx

from notebook_provenance import NodeType
from notebook_provenance.core.data_structures import (
    DataFlowGraph,
    DataArtifact,
    Transformation,
)
from notebook_provenance.parsing.ast_parser import ParsedCell


class TransformationExtractor:
    """
    Extract and describe transformations between artifacts.
    
    This class identifies how data artifacts are transformed from one to another,
    extracting the operations and generating descriptions.
    
    Example:
        >>> extractor = TransformationExtractor(llm_analyzer)
        >>> transformations = extractor.extract_transformations(
        ...     dfg, artifacts, parsed_cells
        ... )
    """
    
    def __init__(self, llm_analyzer=None):
        """
        Initialize the transformation extractor.
        
        Args:
            llm_analyzer: Optional LLM analyzer for enhanced descriptions
        """
        self.llm_analyzer = llm_analyzer
        self.transformations = []
    
    def extract_transformations(self, dfg: DataFlowGraph, 
                           artifacts: List[DataArtifact],
                           parsed_cells: List['ParsedCell']) -> List[Transformation]:
        """
        Extract ONLY direct transformations between artifacts.
        
        FIXED: Don't create transformations between all pairs.
        Only when there's a clear function creating one artifact from another.
        """
        transformations = []
        artifact_set = {a.id for a in artifacts}
        artifact_by_name = {a.name: a for a in artifacts}
        
        # Map of cell_id to parsed cell
        cell_map = {cell.cell_id: cell for cell in parsed_cells if not cell.error}
        
        for artifact in artifacts:
            # Skip if no creation info
            created_by = artifact.metadata.get('created_by')
            if not created_by:
                continue
            
            # Find what artifacts were inputs to the creating function
            source_artifacts = self._find_source_artifacts(
                dfg, artifact, artifacts, created_by
            )
            
            if not source_artifacts:
                # Check if this is from file load
                if any(f in created_by.lower() for f in ['read_', 'load', 'fetch']):
                    # This is a source artifact, no transformation needed
                    continue
            
            # Create transformation
            trans = Transformation(
                id=f"trans_{len(transformations)}",
                operation=created_by,
                source_artifacts=[s.id for s in source_artifacts],
                target_artifact=artifact.id,
                function_calls=[created_by],
                cell_id=artifact.created_in_cell,
                description=self._generate_description(
                    source_artifacts, artifact, created_by
                ),
                semantic_type=self._classify_transformation([created_by])
            )
            
            transformations.append(trans)
        
        self.transformations = transformations
        return transformations

    def _find_source_artifacts(self, dfg: DataFlowGraph, 
                            target_artifact: DataArtifact,
                            all_artifacts: List[DataArtifact],
                            creating_function: str) -> List[DataArtifact]:
        """
        Find artifacts that were inputs to the function that created target.
        """
        sources = []
        artifact_names = {a.name for a in all_artifacts}
        artifact_by_name = {a.name: a for a in all_artifacts}
        
        # Find the function call node
        for node_id, node in dfg.nodes.items():
            if (node.node_type == NodeType.FUNCTION_CALL and 
                node.label == creating_function and
                node.cell_id == target_artifact.created_in_cell):
                
                # Find inputs to this function
                for edge in dfg.edges:
                    if edge.to_node == node_id:
                        source_node = dfg.get_node(edge.from_node)
                        if source_node and source_node.label in artifact_names:
                            source_artifact = artifact_by_name.get(source_node.label)
                            if source_artifact and source_artifact.id != target_artifact.id:
                                sources.append(source_artifact)
                break
        
        return sources

    def _generate_description(self, sources: List[DataArtifact],
                            target: DataArtifact,
                            function: str) -> str:
        """Generate human-readable transformation description."""
        func_lower = function.lower()
        
        # Function-based descriptions
        if 'read_csv' in func_lower:
            return f"Load {target.name} from CSV file"
        if 'read_' in func_lower:
            return f"Load {target.name} from file"
        if 'add_table' in func_lower:
            if sources:
                return f"Store {sources[0].name} in table manager"
            return f"Create table {target.name}"
        if 'get_table' in func_lower:
            return f"Retrieve table data"
        if 'reconcile' in func_lower:
            if sources:
                return f"Reconcile {sources[0].name}"
            return f"Reconcile table data"
        if 'extend' in func_lower:
            if sources:
                return f"Extend {sources[0].name} with additional data"
            return f"Extend table with additional data"
        if 'merge' in func_lower or 'join' in func_lower:
            if len(sources) >= 2:
                return f"Merge {sources[0].name} and {sources[1].name}"
            return f"Merge data"
        
        # Generic
        if sources:
            return f"Transform {sources[0].name} → {target.name}"
        return f"Create {target.name}"
    
    def _find_path(self, dfg: DataFlowGraph, source_id: str, target_id: str) -> Optional[List[str]]:
        """
        Find path between two nodes using BFS.
        
        Args:
            dfg: Data flow graph
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            List of node IDs in path, or None if no path exists
        """
        G = dfg.to_networkx()
        
        try:
            return nx.shortest_path(G, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _extract_operations_from_path(self, dfg: DataFlowGraph, path: List[str]) -> List[str]:
        """
        Extract operations from path.
        
        Args:
            dfg: Data flow graph
            path: List of node IDs in path
            
        Returns:
            List of operation names
        """
        operations = []
        
        for node_id in path:
            node = dfg.nodes.get(node_id)
            if node and node.node_type.value == 'function_call':
                operations.append(node.label)
        
        return operations
    
    def _heuristic_describe_transformation(self, source_name: str, 
                                          target_name: str,
                                          function_calls: List[str]) -> str:
        """
        Generate heuristic-based description.
        
        Args:
            source_name: Source artifact name
            target_name: Target artifact name
            function_calls: List of function calls
            
        Returns:
            Description string
        """
        if not function_calls:
            return f"Transform {source_name} to {target_name}"
        
        func = function_calls[0].lower()
        
        transformation_map = {
            'reconcile': f'Reconcile entities in {source_name}',
            'extend': f'Extend {source_name} with external data',
            'extend_column': f'Extend {source_name} with new column',
            'read_csv': f'Load data from CSV',
            'add_table': f'Store {source_name} in table',
            'merge': f'Merge {source_name} with other data',
            'join': f'Join {source_name} with other data',
            'filter': f'Filter rows from {source_name}',
            'aggregate': f'Aggregate values in {source_name}',
            'groupby': f'Group {source_name} by key',
            'pivot': f'Pivot {source_name}',
            'melt': f'Melt {source_name}',
            'drop': f'Drop columns from {source_name}',
            'rename': f'Rename columns in {source_name}',
            'sort': f'Sort {source_name}',
            'osm': f'Enrich {source_name} with OSM data',
            'geocode': f'Geocode locations in {source_name}',
            'llm': f'Classify {source_name} with LLM',
            'fit': f'Train model on {source_name}',
            'predict': f'Generate predictions from {source_name}',
            'transform': f'Transform {source_name}',
        }
        
        for key, desc in transformation_map.items():
            if key in func:
                return desc
        
        return f"Apply {function_calls[0]} to {source_name}"
    
    def _classify_transformation(self, function_calls: List[str]) -> str:
        """
        Classify transformation type.
        
        Args:
            function_calls: List of function calls
            
        Returns:
            Semantic type string
        """
        if not function_calls:
            return "generic"
        
        func = function_calls[0].lower()
        
        # Classification map
        if any(kw in func for kw in ['read', 'load', 'fetch']):
            return "data_loading"
        elif any(kw in func for kw in ['merge', 'join', 'concat']):
            return "data_combination"
        elif any(kw in func for kw in ['filter', 'select', 'drop', 'where']):
            return "data_selection"
        elif any(kw in func for kw in ['groupby', 'aggregate', 'agg', 'sum', 'mean']):
            return "aggregation"
        elif any(kw in func for kw in ['pivot', 'melt', 'reshape']):
            return "reshaping"
        elif any(kw in func for kw in ['reconcile', 'match', 'dedupe']):
            return "reconciliation"
        elif any(kw in func for kw in ['extend', 'enrich', 'augment']):
            return "enrichment"
        elif any(kw in func for kw in ['fit', 'train']):
            return "model_training"
        elif any(kw in func for kw in ['predict', 'score']):
            return "prediction"
        elif any(kw in func for kw in ['save', 'write', 'export', 'to_']):
            return "data_export"
        else:
            return "transformation"
    
    def _extract_multi_source_transformations(self, dfg: DataFlowGraph,
                                              artifacts: List[DataArtifact],
                                              cell_code_map: Dict) -> List[Transformation]:
        """
        Extract transformations that combine multiple source artifacts.
        
        This handles operations like joins, merges, concatenations that
        take multiple inputs.
        
        Args:
            dfg: Data flow graph
            artifacts: List of artifacts
            cell_code_map: Mapping of cell IDs to parsed cells
            
        Returns:
            List of multi-source Transformation objects
        """
        transformations = []
        artifact_ids = {a.id for a in artifacts}
        artifact_map = {a.id: a for a in artifacts}
        
        # For each artifact, check if it has multiple artifact predecessors
        for artifact in artifacts:
            # Get all incoming edges
            incoming = dfg.get_edges_to(artifact.id)
            
            # Find source artifacts
            source_artifact_ids = []
            operations = []
            
            for edge in incoming:
                # Check if source is an artifact
                if edge.from_node in artifact_ids:
                    source_artifact_ids.append(edge.from_node)
                else:
                    # Check if it's a function that takes artifacts
                    source_node = dfg.get_node(edge.from_node)
                    if source_node and source_node.node_type.value == 'function_call':
                        operations.append(source_node.label)
                        
                        # Find artifacts feeding into this function
                        func_incoming = dfg.get_edges_to(source_node.id)
                        for func_edge in func_incoming:
                            if func_edge.from_node in artifact_ids:
                                source_artifact_ids.append(func_edge.from_node)
            
            # If multiple sources found, create transformation
            if len(source_artifact_ids) >= 2:
                source_artifact_ids = list(set(source_artifact_ids))
                
                # Get code snippet
                cell_info = cell_code_map.get(artifact.created_in_cell)
                code_snippet = cell_info.code[:500] if cell_info else ''
                
                # Generate description
                source_names = [artifact_map[sid].name for sid in source_artifact_ids]
                description = self._generate_multi_source_description(
                    source_names, artifact.name, operations, code_snippet
                )
                
                transformation = Transformation(
                    id=f"trans_{len(self.transformations) + len(transformations)}",
                    operation=operations[0] if operations else "combine",
                    source_artifacts=source_artifact_ids,
                    target_artifact=artifact.id,
                    function_calls=operations,
                    cell_id=artifact.created_in_cell,
                    description=description,
                    semantic_type="data_combination"
                )
                
                transformations.append(transformation)
        
        return transformations
    
    def _generate_multi_source_description(self, source_names: List[str],
                                           target_name: str,
                                           operations: List[str],
                                           code_snippet: str) -> str:
        """
        Generate description for multi-source transformation.
        
        Args:
            source_names: List of source artifact names
            target_name: Target artifact name
            operations: List of operations
            code_snippet: Code snippet
            
        Returns:
            Description string
        """
        # Use LLM if available
        if self.llm_analyzer and self.llm_analyzer.enabled:
            try:
                prompt_sources = ", ".join(source_names)
                return self.llm_analyzer.describe_transformation(
                    prompt_sources,
                    target_name,
                    operations,
                    code_snippet
                )
            except:
                pass
        
        # Heuristic description
        sources_str = " and ".join(source_names)
        
        if operations:
            op = operations[0].lower()
            if 'merge' in op or 'join' in op:
                return f"Join {sources_str}"
            elif 'concat' in op:
                return f"Concatenate {sources_str}"
            elif 'union' in op:
                return f"Union {sources_str}"
        
        return f"Combine {sources_str} into {target_name}"
    
    def get_transformation_by_target(self, target_artifact_id: str) -> Optional[Transformation]:
        """
        Get transformation that produces a specific artifact.
        
        Args:
            target_artifact_id: Target artifact ID
            
        Returns:
            Transformation or None
        """
        for trans in self.transformations:
            if trans.target_artifact == target_artifact_id:
                return trans
        return None
    
    def get_transformations_by_source(self, source_artifact_id: str) -> List[Transformation]:
        """
        Get all transformations that use a specific artifact as source.
        
        Args:
            source_artifact_id: Source artifact ID
            
        Returns:
            List of Transformation objects
        """
        return [
            trans for trans in self.transformations
            if source_artifact_id in trans.source_artifacts
        ]
    
    def get_transformation_stats(self) -> Dict:
        """
        Get statistics about transformations.
        
        Returns:
            Dictionary of statistics
        """
        if not self.transformations:
            return {
                'total': 0,
                'by_type': {},
                'avg_sources': 0.0,
            }
        
        from collections import Counter
        
        type_counts = Counter(t.semantic_type for t in self.transformations)
        avg_sources = sum(len(t.source_artifacts) for t in self.transformations) / len(self.transformations)
        
        return {
            'total': len(self.transformations),
            'by_type': dict(type_counts),
            'avg_sources': avg_sources,
            'multi_source_count': sum(1 for t in self.transformations if len(t.source_artifacts) > 1),
        }


__all__ = [
    "TransformationExtractor",
]