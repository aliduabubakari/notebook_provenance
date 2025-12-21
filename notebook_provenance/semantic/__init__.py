"""
Semantic Module
===============

LLM-based semantic analysis and reasoning.

This module provides:
- LLMSemanticAnalyzer: LLM integration for semantic understanding
- PipelineStageBuilder: Build high-level pipeline stages
- HybridOperationClassifier: Hybrid classification with reasoning
- DynamicTaxonomy: Expandable operation taxonomy
- HybridArtifactClassifier: LLM + embedding artifact classification (NEW)
- SemanticDeduplicator: Semantic variable deduplication (NEW)
"""

from notebook_provenance.semantic.llm_analyzer import LLMSemanticAnalyzer
from notebook_provenance.semantic.stage_builder import PipelineStageBuilder
from notebook_provenance.semantic.reasoning import (
    HybridOperationClassifier,
    DynamicTaxonomy,
)
from notebook_provenance.semantic.artifact_classifier import (
    HybridArtifactClassifier,
    ArtifactClassification,
)
from notebook_provenance.semantic.deduplicator import (
    SemanticDeduplicator,
    VariableCluster,
)

__all__ = [
    "LLMSemanticAnalyzer",
    "PipelineStageBuilder",
    "HybridOperationClassifier",
    "DynamicTaxonomy",
    "HybridArtifactClassifier",
    "ArtifactClassification",
    "SemanticDeduplicator",
    "VariableCluster",
]