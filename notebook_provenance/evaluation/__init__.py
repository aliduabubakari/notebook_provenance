"""
Evaluation Module
=================

Metrics and benchmarking framework for provenance extraction evaluation.

This module provides:
- ProvenanceEvaluator: Multi-dimensional evaluation metrics
- EvaluationBenchmark: Benchmark framework for dataset evaluation
- GroundTruthManager: Ground truth annotation management
- MetricsReporter: Generate evaluation reports
"""

from notebook_provenance.evaluation.metrics import ProvenanceEvaluator
from notebook_provenance.evaluation.benchmark import EvaluationBenchmark
from notebook_provenance.evaluation.ground_truth import (
    GroundTruthManager,
    GroundTruthAnnotation,
    CellAnnotation,
    ArtifactAnnotation,
)
from notebook_provenance.evaluation.reporter import MetricsReporter

__all__ = [
    "ProvenanceEvaluator",
    "EvaluationBenchmark",
    "GroundTruthManager",
    "GroundTruthAnnotation",
    "CellAnnotation",
    "ArtifactAnnotation",
    "MetricsReporter",
]