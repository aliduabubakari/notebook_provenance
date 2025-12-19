"""
Notebook Provenance Analysis System
====================================

A comprehensive system for extracting, analyzing, and visualizing data provenance
from computational notebooks and scripts.

Main Components:
- Core: Data structures, enums, and configuration
- Parsing: Code parsing and notebook loading
- Graph: Data flow graph construction and analysis
- Semantic: LLM-based semantic analysis
- Visualization: Multi-level provenance visualization
- Export: Export to various formats (JSON, Neo4j, etc.)
- Evaluation: Metrics and benchmarking framework

Usage:
    >>> from notebook_provenance import NotebookProvenanceSystem
    >>> system = NotebookProvenanceSystem(api_key="your_key")
    >>> result = system.analyze_notebook(code_cells, cell_ids)
    >>> system.save_all(result, prefix="output")
    
    # Or use convenience function
    >>> from notebook_provenance import analyze_notebook_file
    >>> result = analyze_notebook_file("notebook.ipynb", api_key="your_key")
"""

__version__ = "0.2.0"
__author__ = "Your Name"
__license__ = "MIT"

# Import main classes for convenience
from notebook_provenance.core.data_structures import (
    DFGNode,
    DFGEdge,
    DataFlowGraph,
    CellDependency,
    DataArtifact,
    Transformation,
    PipelineStageNode,
)

from notebook_provenance.core.enums import (
    NodeType,
    EdgeType,
    TaskType,
    PipelineStage,
)

from notebook_provenance.core.config import (
    Config,
    get_default_config,
)

# Main orchestrator
from notebook_provenance.orchestrator import (
    NotebookProvenanceSystem,
    analyze_notebook_file,
)

# Parsing utilities
from notebook_provenance.parsing.notebook_loader import NotebookLoader
from notebook_provenance.parsing.renderer import NotebookRenderer

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    
    # Main system
    "NotebookProvenanceSystem",
    "analyze_notebook_file",
    
    # Core data structures
    "DFGNode",
    "DFGEdge",
    "DataFlowGraph",
    "CellDependency",
    "DataArtifact",
    "Transformation",
    "PipelineStageNode",
    
    # Enums
    "NodeType",
    "EdgeType",
    "TaskType",
    "PipelineStage",
    
    # Configuration
    "Config",
    "get_default_config",
    
    # Utilities
    "NotebookLoader",
    "NotebookRenderer",
]