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
                               parsed_cells: List[ParsedCell]) -> List[Transformation]:
        """
        Extract transformations between artifacts.
        
        Args:
            dfg: Data flow graph
            artifacts: List of data artifacts
            parsed_cells: List of parsed cells
            
        Returns:
            List of Transformation objects
        """
        transformations = []
        artifact_map = {a.id: a for a in artifacts}
        
        # Build a mapping of cells to their code
        cell_code_map = {cell.cell_id: cell for cell in parsed_cells if not cell.error}
        
        # Find paths between artifacts
        for i, source_artifact in enumerate(artifacts):
            for target_artifact in artifacts[i+1:]:
                # Check if there's a path
                path = self._find_path(dfg, source_artifact.id, target_artifact.id)
                
                if path:
                    # Extract operations along the path
                    operations = self._extract_operations_from_path(dfg, path)
                    
                    # Skip if no meaningful operations
                    if not operations:
                        continue
                    
                    # Find the cell where transformation happens
                    transform_cell = target_artifact.created_in_cell
                    
                    # Get code snippet
                    cell_info = cell_code_map.get(transform_cell)
                    code_snippet = cell_info.code[:500] if cell_info else ''
                    
                    # Get function calls
                    function_calls = [op for op in operations if op]
                    
                    # Generate description
                    description = self._generate_description(
                        source_artifact.name,
                        target_artifact.name,
                        function_calls,
                        code_snippet
                    )
                    
                    # Determine semantic type
                    semantic_type = self._classify_transformation(function_calls)
                    
                    transformation = Transformation(
                        id=f"trans_{len(transformations)}",
                        operation=function_calls[0] if function_calls else "transform",
                        source_artifacts=[source_artifact.id],
                        target_artifact=target_artifact.id,
                        function_calls=function_calls,
                        cell_id=transform_cell,
                        description=description,
                        semantic_type=semantic_type
                    )
                    
                    transformations.append(transformation)
        
        # Also check for multi-source transformations (joins, merges, etc.)
        multi_source_transforms = self._extract_multi_source_transformations(
            dfg, artifacts, cell_code_map
        )
        transformations.extend(multi_source_transforms)
        
        self.transformations = transformations
        return transformations
    
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
    
    def _generate_description(self, source_name: str, target_name: str,
                             function_calls: List[str], code_snippet: str) -> str:
        """
        Generate human-readable description of transformation.
        
        Args:
            source_name: Source artifact name
            target_name: Target artifact name
            function_calls: List of function calls
            code_snippet: Code snippet
            
        Returns:
            Description string
        """
        # Use LLM if available
        if self.llm_analyzer and self.llm_analyzer.enabled:
            try:
                return self.llm_analyzer.describe_transformation(
                    source_name,
                    target_name,
                    function_calls,
                    code_snippet
                )
            except Exception as e:
                print(f"Warning: LLM description failed: {e}")
                # Fall through to heuristic
        
        # Heuristic description
        return self._heuristic_describe_transformation(
            source_name, target_name, function_calls
        )
    
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