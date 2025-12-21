"""
Orchestrator Module - UPDATED WITH HYBRID CLASSIFICATION
=========================================================

Main orchestrator with integrated hybrid artifact classification.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import time

import matplotlib.pyplot as plt

from notebook_provenance.core.config import Config, get_default_config
from notebook_provenance.core.data_structures import (
    DataFlowGraph,
    DataArtifact,
    Transformation,
    PipelineStageNode,
    CellDependency,
)

from notebook_provenance.parsing.ast_parser import EnhancedCodeCellParser, ParsedCell
from notebook_provenance.parsing.notebook_loader import NotebookLoader
from notebook_provenance.parsing.renderer import NotebookRenderer

from notebook_provenance.graph.dfg_builder import SmartDFGBuilder
from notebook_provenance.graph.artifact_analyzer import DataArtifactAnalyzer
from notebook_provenance.graph.transformation import TransformationExtractor
from notebook_provenance.graph.column_lineage import ColumnLineageTracker

from notebook_provenance.semantic.llm_analyzer import LLMSemanticAnalyzer
from notebook_provenance.semantic.stage_builder import PipelineStageBuilder
from notebook_provenance.semantic.artifact_classifier import HybridArtifactClassifier
from notebook_provenance.semantic.deduplicator import SemanticDeduplicator

from notebook_provenance.visualization.provenance_viz import ProvenanceVisualizer
from notebook_provenance.visualization.interactive import InteractiveVisualizer

from notebook_provenance.export.json_export import JSONExporter


class NotebookProvenanceSystem:
    """
    Main orchestrator for notebook provenance extraction and analysis.
    
    UPDATED: Now includes hybrid artifact classification and semantic deduplication.
    """
    
    def __init__(self, config: Optional[Config] = None,
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.deepinfra.com/v1/openai",
                 use_llm: bool = True):
        """
        Initialize the provenance system.
        
        Args:
            config: Optional configuration object
            api_key: API key for LLM (used if config not provided)
            base_url: Base URL for LLM API
            use_llm: Whether to use LLM features
        """
        # Build config if not provided
        if config is None:
            from notebook_provenance.core.config import (
                LLMConfig, 
                VisualizationConfig,
                ClassificationConfig
            )
            config = Config(
                llm=LLMConfig(
                    enabled=use_llm and api_key is not None,
                    api_key=api_key,
                    base_url=base_url,
                ),
                classification=ClassificationConfig(
                    use_llm=use_llm and api_key is not None,
                    use_embeddings=use_llm and api_key is not None,
                    use_semantic_deduplication=use_llm and api_key is not None,
                ),
                visualization=VisualizationConfig(),
            )
        
        self.config = config
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all system components."""
        # Parsing
        self.parser = EnhancedCodeCellParser(config=self.config.parsing)
        
        # Graph building
        self.dfg_builder = SmartDFGBuilder(config=self.config.graph)
        
        # Semantic analysis (LLM)
        self.llm_analyzer = None
        if self.config.llm.enabled:
            self.llm_analyzer = LLMSemanticAnalyzer(config=self.config.llm)
            
            if self.config.verbose:
                print(f"✓ LLM enabled: {self.config.llm.model}")
        
        # NEW: Hybrid artifact classifier
        self.artifact_classifier = HybridArtifactClassifier(
            llm_analyzer=self.llm_analyzer,
            use_embeddings=self.config.classification.use_embeddings
        )
        
        # NEW: Semantic deduplicator
        self.deduplicator = SemanticDeduplicator(
            llm_analyzer=self.llm_analyzer,
            similarity_threshold=self.config.classification.similarity_threshold,
            use_semantic_merge=self.config.classification.use_semantic_deduplication
        )
        
        # Artifact analysis (now uses hybrid classifier)
        self.artifact_analyzer = DataArtifactAnalyzer(
            llm_analyzer=self.llm_analyzer,
            use_embeddings=self.config.classification.use_embeddings
        )
        
        # Transformation extraction
        self.transformation_extractor = TransformationExtractor(self.llm_analyzer)
        
        # Stage building
        self.stage_builder = PipelineStageBuilder(self.llm_analyzer)
        
        # Column lineage
        self.column_tracker = ColumnLineageTracker()
        
        # Visualization
        self.visualizer = ProvenanceVisualizer(config=self.config.visualization)
        self.interactive_visualizer = InteractiveVisualizer()
        
        # Export
        self.json_exporter = JSONExporter()
    
    def analyze_notebook(self, notebook_cells: List[str], 
                        cell_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Complete notebook analysis pipeline with hybrid classification.
        
        Args:
            notebook_cells: List of code cell contents
            cell_ids: Optional list of cell identifiers
            
        Returns:
            Dictionary containing all analysis results
        """
        if self.config.verbose:
            print("=" * 80)
            print("NOTEBOOK PROVENANCE ANALYSIS")
            print("=" * 80)
        
        # Generate cell IDs if not provided
        if cell_ids is None:
            cell_ids = [f"cell_{i}" for i in range(len(notebook_cells))]
        
        # Create cell code mapping for context
        cell_code_map = {cid: code for cid, code in zip(cell_ids, notebook_cells)}
        
        # ========== LAYER 1: PARSE CELLS ==========
        if self.config.verbose:
            print("\n[Layer 1] Parsing notebook cells...")
        
        start_time = time.time()
        parsed_cells = self.parser.parse_cells(notebook_cells, cell_ids)
        
        valid_cells = [c for c in parsed_cells if c.error is None]
        parse_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"  ✓ Parsed {len(valid_cells)}/{len(parsed_cells)} cells successfully ({parse_time:.2f}s)")
        
        # ========== LAYER 2: BUILD GRAPHS ==========
        if self.config.verbose:
            print("\n[Layer 2] Building data flow graphs...")
        
        start_time = time.time()
        raw_dfg, cell_dependencies = self.dfg_builder.build_dfg_from_cells(parsed_cells)
        
        if self.config.verbose:
            print(f"  ✓ Raw DFG: {len(raw_dfg.nodes)} nodes, {len(raw_dfg.edges)} edges")
        
        # NEW: Semantic deduplication
        if self.config.classification.use_semantic_deduplication:
            if self.config.verbose:
                print(f"\n  [2.5] Semantic deduplication...")
            
            canonical_map = self.deduplicator.deduplicate(
                raw_dfg.nodes,
                verbose=self.config.verbose
            )
            
            # Apply deduplication to DFG
            raw_dfg = self._apply_deduplication(raw_dfg, canonical_map)
            
            if self.config.verbose:
                dedup_stats = self.deduplicator.get_statistics()
                print(f"  ✓ Deduplication: {dedup_stats}")
        
        # Build clean DFG
        if self.config.graph.build_clean_dfg:
            clean_dfg = self.dfg_builder.build_clean_dfg(raw_dfg)
            reduction = (1 - len(clean_dfg.nodes) / len(raw_dfg.nodes)) * 100 if raw_dfg.nodes else 0
            
            if self.config.verbose:
                print(f"  ✓ Clean DFG: {len(clean_dfg.nodes)} nodes, {len(clean_dfg.edges)} edges")
                print(f"  → Noise reduction: {reduction:.1f}%")
        else:
            clean_dfg = raw_dfg
            reduction = 0
        
        graph_time = time.time() - start_time
        
        # ========== LAYER 3: SEMANTIC ANALYSIS WITH HYBRID CLASSIFICATION ==========
        if self.config.verbose:
            print("\n[Layer 3] Semantic analysis with hybrid classification...")
        
        start_time = time.time()
        
        # Identify artifacts using hybrid classifier
        artifacts = self.artifact_analyzer.identify_artifacts(
            raw_dfg, 
            cell_dependencies,
            code_cells=cell_code_map
        )
        
        if self.config.verbose:
            print(f"  ✓ Identified {len(artifacts)} data artifacts")
            
            # Show classification statistics
            if hasattr(self.artifact_analyzer, 'hybrid_classifier'):
                class_stats = self.artifact_analyzer.hybrid_classifier.get_statistics()
                print(f"  Classification sources:")
                for source, count in class_stats.get('by_source', {}).items():
                    pct = class_stats['percentages'].get(source, 0)
                    print(f"    - {source}: {count} ({pct:.1f}%)")
            
            # Show top artifacts
            for artifact in artifacts[:5]:
                print(f"    - {artifact.name} ({artifact.type}, importance: {artifact.importance_score:.1f})")
                if self.config.verbose and 'reasoning' in artifact.metadata:
                    print(f"      Reason: {artifact.metadata['reasoning']}")
            if len(artifacts) > 5:
                print(f"    ... and {len(artifacts) - 5} more")
        
        # Extract transformations
        transformations = self.transformation_extractor.extract_transformations(
            raw_dfg, artifacts, parsed_cells
        )
        
        if self.config.verbose:
            print(f"  ✓ Extracted {len(transformations)} transformations")
        
        # Build pipeline stages
        stages = self.stage_builder.build_stages(parsed_cells, artifacts, cell_dependencies)
        
        if self.config.verbose:
            print(f"  ✓ Identified {len(stages)} pipeline stages:")
            for i, stage in enumerate(stages, 1):
                print(f"    {i}. {stage.stage_type.value}: {stage.description}")
        
        semantic_time = time.time() - start_time
        
        # ========== LAYER 3.5: COLUMN LINEAGE ==========
        if self.config.verbose:
            print("\n[Layer 3.5] Extracting column-level lineage...")
        
        parsed_cell_dicts = [
            {
                'cell_id': cell.cell_id,
                'code': cell.code,
                'error': cell.error,
            }
            for cell in parsed_cells
        ]
        
        column_lineage = self.column_tracker.extract_column_lineage(parsed_cell_dicts)
        
        if self.config.verbose:
            created_cols = len(column_lineage.get('created', {}))
            dropped_cols = len(column_lineage.get('dropped', {}))
            renamed_cols = len(column_lineage.get('renamed', {}))
            print(f"  ✓ Columns created: {created_cols}, dropped: {dropped_cols}, renamed: {renamed_cols}")
        
        # ========== LAYER 4: VISUALIZATION ==========
        visualizations = {}
        
        if self.config.visualization.enabled:
            if self.config.verbose:
                print("\n[Layer 4] Creating visualizations...")
            
            visualizations = self.visualizer.create_all_visualizations({
                'stages': stages,
                'artifacts': artifacts,
                'transformations': transformations,
                'clean_dfg': clean_dfg,
            })
            
            if self.config.verbose:
                for name in visualizations:
                    print(f"  ✓ {name} visualization")
        
        # ========== COMPILE RESULTS ==========
        result = {
            'parsed_cells': parsed_cells,
            'raw_dfg': raw_dfg,
            'clean_dfg': clean_dfg,
            'cell_dependencies': cell_dependencies,
            'artifacts': artifacts,
            'transformations': transformations,
            'stages': stages,
            'column_lineage': column_lineage,
            'visualizations': visualizations,
            'statistics': {
                'total_cells': len(notebook_cells),
                'valid_cells': len(valid_cells),
                'raw_nodes': len(raw_dfg.nodes),
                'raw_edges': len(raw_dfg.edges),
                'clean_nodes': len(clean_dfg.nodes),
                'clean_edges': len(clean_dfg.edges),
                'noise_reduction': reduction,
                'artifacts': len(artifacts),
                'transformations': len(transformations),
                'stages': len(stages),
            },
            'timing': {
                'parse_time': parse_time,
                'graph_time': graph_time,
                'semantic_time': semantic_time,
            },
            'classification_stats': self.artifact_analyzer.get_classification_stats() if hasattr(self.artifact_analyzer, 'get_classification_stats') else {}
        }
        
        if self.config.verbose:
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print("=" * 80)
        
        return result
    
    def _apply_deduplication(self, dfg: DataFlowGraph, 
                           canonical_map: Dict[str, str]) -> DataFlowGraph:
        """
        Apply deduplication mapping to DFG.
        
        Args:
            dfg: Original data flow graph
            canonical_map: Mapping from node_id to canonical node_id
            
        Returns:
            Deduplicated data flow graph
        """
        new_dfg = DataFlowGraph()
        new_dfg.metadata = dfg.metadata.copy()
        
        # Add canonical nodes only
        canonical_nodes = set(canonical_map.values())
        for node_id in canonical_nodes:
            if node_id in dfg.nodes:
                new_dfg.add_node(dfg.nodes[node_id])
        
        # Add edges, mapping to canonical IDs
        for edge in dfg.edges:
            from_canonical = canonical_map.get(edge.from_node, edge.from_node)
            to_canonical = canonical_map.get(edge.to_node, edge.to_node)
            
            # Only add if both nodes exist and it's not a self-loop
            if (from_canonical in new_dfg.nodes and 
                to_canonical in new_dfg.nodes and 
                from_canonical != to_canonical):
                
                # Create new edge with canonical IDs
                from notebook_provenance.core.data_structures import DFGEdge
                new_edge = DFGEdge(
                    from_node=from_canonical,
                    to_node=to_canonical,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    operation=edge.operation,
                    metadata=edge.metadata
                )
                new_dfg.add_edge(new_edge)
        
        return new_dfg
    
    def save_all(self, result: Dict, prefix: str = "provenance"):
        """
        Save all outputs to files.
        
        Args:
            result: Analysis result dictionary
            prefix: Output file prefix
        """
        if self.config.verbose:
            print("\n[Layer 5] Saving outputs...")
        
        # Save visualizations
        if self.config.visualization.enabled:
            for name, fig in result.get('visualizations', {}).items():
                filename = f"{prefix}_{name}.png"
                try:
                    fig.savefig(filename, dpi=self.config.visualization.dpi, bbox_inches='tight')
                    if self.config.verbose:
                        print(f"  ✓ Saved {filename}")
                    plt.close(fig)
                except Exception as e:
                    print(f"  ⚠ Failed to save {filename}: {e}")
        
        # Export JSON
        try:
            self.json_exporter.export_complete(result, f"{prefix}_complete.json")
        except Exception as e:
            print(f"  ⚠ Failed to export JSON: {e}")
        
        # Create interactive HTML
        if self.config.visualization.interactive_html:
            try:
                self.interactive_visualizer.create_interactive_html(
                    result, f"{prefix}_interactive.html"
                )
            except Exception as e:
                print(f"  ⚠ Failed to create interactive HTML: {e}")
        
        # Save classification cache if enabled
        if (self.config.classification.cache_classifications and 
            self.config.classification.cache_path and
            hasattr(self.artifact_analyzer, 'hybrid_classifier')):
            try:
                cache_path = str(self.config.classification.cache_path)
                self.artifact_analyzer.hybrid_classifier.save_cache(cache_path)
                if self.config.verbose:
                    print(f"  ✓ Saved classification cache to {cache_path}")
            except Exception as e:
                print(f"  ⚠ Failed to save classification cache: {e}")
        
        if self.config.verbose:
            print("\n✅ All outputs saved successfully!")
    
    def analyze_file(self, file_path: str, 
                    output_prefix: Optional[str] = None,
                    save_outputs: bool = True) -> Dict[str, Any]:
        """
        Convenience method to analyze a notebook file.
        
        Args:
            file_path: Path to notebook file
            output_prefix: Optional output prefix
            save_outputs: Whether to save outputs
            
        Returns:
            Analysis result dictionary
        """
        # Load notebook
        code_cells, cell_ids = NotebookLoader.load_notebook(file_path)
        
        # Analyze
        result = self.analyze_notebook(code_cells, cell_ids)
        
        # Save if requested
        if save_outputs:
            if output_prefix is None:
                output_prefix = Path(file_path).stem + "_provenance"
            self.save_all(result, prefix=output_prefix)
        
        return result


# Convenience function for simple usage
def analyze_notebook_file(notebook_path: str,
                         api_key: Optional[str] = None,
                         output_prefix: Optional[str] = None,
                         use_llm: bool = True,
                         use_embeddings: bool = True,
                         save_outputs: bool = True) -> Dict[str, Any]:
    """
    Convenience function to analyze a notebook file with hybrid classification.
    
    Args:
        notebook_path: Path to .ipynb or .py file
        api_key: API key for LLM (optional)
        output_prefix: Prefix for output files (optional)
        use_llm: Whether to use LLM (default: True if api_key provided)
        use_embeddings: Whether to use embedding-based similarity
        save_outputs: Whether to save outputs (default: True)
    
    Returns:
        Analysis result dictionary
    """
    from notebook_provenance.core.config import Config, LLMConfig, ClassificationConfig
    
    config = Config(
        llm=LLMConfig(
            enabled=use_llm and api_key is not None,
            api_key=api_key
        ),
        classification=ClassificationConfig(
            use_llm=use_llm and api_key is not None,
            use_embeddings=use_embeddings and api_key is not None,
            use_semantic_deduplication=use_embeddings and api_key is not None,
        )
    )
    
    system = NotebookProvenanceSystem(config=config)
    
    return system.analyze_file(
        notebook_path,
        output_prefix=output_prefix,
        save_outputs=save_outputs
    )


__all__ = [
    "NotebookProvenanceSystem",
    "analyze_notebook_file",
]