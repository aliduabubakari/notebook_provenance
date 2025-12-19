"""
Semantic Module
===============

LLM-based semantic analysis and reasoning.

This module provides:
- LLMSemanticAnalyzer: LLM integration for semantic understanding
- PipelineStageBuilder: Build high-level pipeline stages
- Reasoning submodule: Hybrid classification and dynamic taxonomy
"""

from notebook_provenance.semantic.llm_analyzer import LLMSemanticAnalyzer
from notebook_provenance.semantic.stage_builder import PipelineStageBuilder

# Import reasoning submodule
from notebook_provenance.semantic import reasoning

__all__ = [
    "LLMSemanticAnalyzer",
    "PipelineStageBuilder",
    "reasoning",
]