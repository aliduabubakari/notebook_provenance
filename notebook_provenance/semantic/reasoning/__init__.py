"""
Reasoning Submodule
===================

Advanced reasoning capabilities for operation classification.

This submodule provides:
- HybridOperationClassifier: Combines fixed taxonomy with LLM reasoning
- DynamicTaxonomy: Expandable operation taxonomy
- Prompt templates for consistent LLM interactions
"""

from notebook_provenance.semantic.reasoning.classifier import HybridOperationClassifier
from notebook_provenance.semantic.reasoning.taxonomy import DynamicTaxonomy
from notebook_provenance.semantic.reasoning import prompts

__all__ = [
    "HybridOperationClassifier",
    "DynamicTaxonomy",
    "prompts",
]