"""
Enumerations Module
===================

Defines all enumeration types used throughout the provenance system.

This module contains:
- NodeType: Types of nodes in the data flow graph
- EdgeType: Types of edges representing different relationships
- TaskType: Types of computational tasks in notebooks
- PipelineStage: High-level pipeline stages
"""

from enum import Enum, auto
from typing import Dict, List


class NodeType(Enum):
    """
    Types of nodes in the data flow graph.
    
    - VARIABLE: A named variable in the code
    - FUNCTION_CALL: A function invocation
    - LITERAL: A literal value (string, number, etc.)
    - INTERMEDIATE: An intermediate result in a computation
    - DATA_ARTIFACT: An important data object (DataFrame, table, model, etc.)
    - IMPORT: An imported module or function
    - CLASS_INSTANCE: An instance of a class
    - ATTRIBUTE_ACCESS: Access to an object's attribute
    """
    VARIABLE = "variable"
    FUNCTION_CALL = "function_call"
    LITERAL = "literal"
    INTERMEDIATE = "intermediate"
    DATA_ARTIFACT = "data_artifact"
    IMPORT = "import"
    CLASS_INSTANCE = "class_instance"
    ATTRIBUTE_ACCESS = "attribute_access"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_data_related(self) -> bool:
        """Check if this node type represents data"""
        return self in {NodeType.VARIABLE, NodeType.DATA_ARTIFACT, 
                       NodeType.INTERMEDIATE}


class EdgeType(Enum):
    """
    Types of edges representing relationships between nodes.
    
    - DATA_FLOW: Data flows from one node to another
    - CONTROL_FLOW: Control flow dependency
    - TRANSFORMATION: A transformation operation
    - CALLER: Relationship between caller and callee
    - INPUT: Input to a function
    - BRANCH: Conditional branch
    - LOOP: Loop dependency
    - FUNCTION_CALL: Function call edge (for recursive functions)
    - OMITTED: Omitted edge (for graph simplification)
    """
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    TRANSFORMATION = "transformation"
    CALLER = "caller"
    INPUT = "input"
    BRANCH = "branch"
    LOOP = "loop"
    FUNCTION_CALL = "function_call"
    OMITTED = "omitted"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_control_flow(self) -> bool:
        """Check if this edge represents control flow"""
        return self in {EdgeType.CONTROL_FLOW, EdgeType.BRANCH, 
                       EdgeType.LOOP, EdgeType.FUNCTION_CALL}


class TaskType(Enum):
    """
    Types of computational tasks found in notebooks.
    
    These represent the semantic purpose of code cells:
    - DATA_LOADING: Loading data from external sources
    - DATA_CLEANING: Data preprocessing and cleaning
    - RECONCILIATION: Entity matching, linking, deduplication
    - EXTENSION: Data enrichment from external sources
    - TRANSFORMATION: Data transformation and feature engineering
    - ANALYSIS: Data analysis and exploration
    - VISUALIZATION: Creating plots and visualizations
    - MODEL_TRAINING: Training machine learning models
    - EVALUATION: Evaluating model performance
    - SETUP: Environment setup, configuration, imports
    - API_INTEGRATION: Integrating with external APIs
    - SCHEMA_VALIDATION: Validating data schemas
    - DATA_PARTITIONING: Partitioning data
    - OTHER: Other operations
    """
    DATA_LOADING = "data_loading"
    DATA_CLEANING = "data_cleaning"
    RECONCILIATION = "reconciliation"
    EXTENSION = "extension"
    TRANSFORMATION = "transformation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    SETUP = "setup"
    API_INTEGRATION = "api_integration"
    SCHEMA_VALIDATION = "schema_validation"
    DATA_PARTITIONING = "data_partitioning"
    OTHER = "other"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def get_keywords(cls) -> Dict[str, List[str]]:
        """Get keyword patterns for heuristic classification"""
        return {
            cls.DATA_LOADING.value: [
                'read_csv', 'read_excel', 'read_json', 'read_parquet',
                'load', 'fetch', 'download', 'query', 'connect', 'from_csv'
            ],
            cls.DATA_CLEANING.value: [
                'dropna', 'fillna', 'drop_duplicates', 'clean', 'strip',
                'replace', 'remove', 'filter_invalid'
            ],
            cls.RECONCILIATION.value: [
                'reconcile', 'match', 'dedupe', 'link', 'merge_fuzzy',
                'geocode', 'geocoding'
            ],
            cls.EXTENSION.value: [
                'extend', 'enrich', 'augment', 'lookup', 'join_external',
                'osm', 'openmeteo', 'api'
            ],
            cls.TRANSFORMATION.value: [
                'transform', 'apply', 'map', 'groupby', 'aggregate',
                'pivot', 'melt', 'reshape'
            ],
            cls.ANALYSIS.value: [
                'describe', 'summary', 'statistics', 'correlation',
                'analyze', 'explore'
            ],
            cls.VISUALIZATION.value: [
                'plot', 'chart', 'visualize', 'plt.', 'sns.', 'px.',
                'figure', 'show'
            ],
            cls.MODEL_TRAINING.value: [
                'fit', 'train', 'compile', 'model.', 'estimator',
                'classifier', 'regressor'
            ],
            cls.EVALUATION.value: [
                'evaluate', 'predict', 'score', 'accuracy', 'metrics',
                'test', 'validate'
            ],
            cls.SETUP.value: [
                'import', 'authmanager', 'config', 'setup', 'initialize',
                'token', 'credentials'
            ],
            cls.API_INTEGRATION.value: [
                'requests', 'api', 'endpoint', 'rest', 'graphql',
                'client.'
            ],
            cls.SCHEMA_VALIDATION.value: [
                'validate', 'check_schema', 'assert', 'dtype', 'type_check'
            ],
            cls.DATA_PARTITIONING.value: [
                'partition', 'split', 'chunk', 'batch', 'shard'
            ],
        }


class PipelineStage(Enum):
    """
    High-level pipeline stages representing workflow phases.
    
    These represent logical groupings of operations:
    - SETUP: Environment configuration and initialization
    - DATA_LOADING: Loading datasets from sources
    - DATA_PREPARATION: Cleaning and preprocessing
    - RECONCILIATION: Entity resolution and matching
    - ENRICHMENT: Augmenting data with external information
    - TRANSFORMATION: Feature engineering and transformations
    - ANALYSIS: Statistical analysis and modeling
    - OUTPUT: Saving results and generating outputs
    """
    SETUP = "setup"
    DATA_LOADING = "data_loading"
    DATA_PREPARATION = "data_preparation"
    RECONCILIATION = "reconciliation"
    ENRICHMENT = "enrichment"
    TRANSFORMATION = "transformation"
    ANALYSIS = "analysis"
    OUTPUT = "output"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name(self) -> str:
        """Get human-readable display name"""
        return self.value.replace('_', ' ').title()
    
    @classmethod
    def from_task_type(cls, task_type: TaskType) -> 'PipelineStage':
        """Map TaskType to PipelineStage"""
        mapping = {
            TaskType.SETUP: cls.SETUP,
            TaskType.DATA_LOADING: cls.DATA_LOADING,
            TaskType.DATA_CLEANING: cls.DATA_PREPARATION,
            TaskType.RECONCILIATION: cls.RECONCILIATION,
            TaskType.EXTENSION: cls.ENRICHMENT,
            TaskType.API_INTEGRATION: cls.ENRICHMENT,
            TaskType.TRANSFORMATION: cls.TRANSFORMATION,
            TaskType.ANALYSIS: cls.ANALYSIS,
            TaskType.VISUALIZATION: cls.OUTPUT,
            TaskType.MODEL_TRAINING: cls.ANALYSIS,
            TaskType.EVALUATION: cls.ANALYSIS,
            TaskType.SCHEMA_VALIDATION: cls.DATA_PREPARATION,
            TaskType.DATA_PARTITIONING: cls.TRANSFORMATION,
            TaskType.OTHER: cls.TRANSFORMATION,
        }
        return mapping.get(task_type, cls.TRANSFORMATION)
    
    @classmethod
    def get_typical_order(cls) -> List['PipelineStage']:
        """Get typical order of pipeline stages"""
        return [
            cls.SETUP,
            cls.DATA_LOADING,
            cls.DATA_PREPARATION,
            cls.RECONCILIATION,
            cls.ENRICHMENT,
            cls.TRANSFORMATION,
            cls.ANALYSIS,
            cls.OUTPUT,
        ]


class ArtifactType(Enum):
    """
    Types of data artifacts identified in the pipeline.
    
    - DATAFRAME: Pandas/Polars/Spark DataFrame
    - TABLE: Database table or similar structure
    - MODEL: Machine learning model
    - RESULT: Computation result or prediction
    - DATASET: Generic dataset
    - MATRIX: Numerical matrix (numpy array, etc.)
    - GRAPH: Graph structure
    - UNKNOWN: Type could not be determined
    """
    DATAFRAME = "dataframe"
    TABLE = "table"
    MODEL = "model"
    RESULT = "result"
    DATASET = "dataset"
    MATRIX = "matrix"
    GRAPH = "graph"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value


# Export all enums
__all__ = [
    "NodeType",
    "EdgeType",
    "TaskType",
    "PipelineStage",
    "ArtifactType",
]