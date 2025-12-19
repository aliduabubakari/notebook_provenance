"""
Core Module
===========

Fundamental data structures, enums, and configuration for the provenance system.

This module provides:
- Data structures for representing data flow graphs, artifacts, and transformations
- Enumerations for node types, edge types, task types, and pipeline stages
- Configuration management for system-wide settings
"""

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
    LLMConfig,
    VisualizationConfig,
    EvaluationConfig,
    get_default_config,
)

__all__ = [
    # Data structures
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
    "LLMConfig",
    "VisualizationConfig",
    "EvaluationConfig",
    "get_default_config",
]