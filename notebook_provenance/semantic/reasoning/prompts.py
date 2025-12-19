"""
Prompt Templates Module
=======================

Centralized prompt templates for LLM reasoning tasks.

This module contains:
- Classification prompts
- ReAct-style reasoning prompts
- Few-shot examples
- Prompt builders
"""

from typing import Dict, List, Optional


class PromptTemplates:
    """
    Centralized prompt templates for LLM interactions.
    
    This class provides consistent, well-tested prompts for various
    reasoning tasks in the provenance system.
    """
    
    # Base system message for all tasks
    SYSTEM_MESSAGE = """You are an expert data engineer analyzing Python code from data pipeline notebooks.
Your task is to understand the semantic meaning and purpose of code operations.
Provide accurate, concise responses focused on data operations and transformations."""
    
    # Task classification prompt
    TASK_CLASSIFICATION = """Analyze this Python code cell and classify its primary task type.

CODE:
```python
{code}
```

METADATA:
- Variables defined: {variables_defined}
- Function calls: {function_calls}
- Imports: {imports}

TASK TYPES:
{task_type_descriptions}

Respond ONLY with valid JSON:
{{"task_type": "task_name", "confidence": 0.95, "reasoning": "brief explanation"}}"""

    # ReAct-style reasoning prompt
    REACT_CLASSIFICATION = """You are analyzing a code snippet from a data pipeline notebook.

CODE:
```python
{code}
```

CONTEXT:
- Variables defined: {variables_defined}
- Function calls: {function_calls}
- Previous operations: {previous_ops}

KNOWN OPERATION TYPES: {known_types}

Use the following reasoning process:

THOUGHT: What is this code doing at a high level?
OBSERVATION: What specific functions/patterns do I see?
REASONING: How does this fit (or not fit) into known categories?
CONCLUSION: Either classify into a known type OR propose a new descriptive type.

If proposing a new type, it should be:
- Descriptive and generalizable
- Following snake_case naming
- Examples: "api_integration", "schema_validation", "data_partitioning"

Respond in JSON:
{{
    "thought": "...",
    "observation": "...", 
    "reasoning": "...",
    "operation_type": "existing_or_new_type",
    "confidence": 0.0-1.0,
    "is_new_type": true/false
}}"""

    # Transformation description prompt
    TRANSFORMATION_DESCRIPTION = """Describe the data transformation in 5-10 words.

SOURCE: {source_artifact}
TARGET: {target_artifact}
OPERATIONS: {operations}

CODE SNIPPET:
```python
{code_snippet}
```

Provide a concise, human-readable description of what this transformation does.
Examples: "Geocode locations", "Enrich with weather data", "Load CSV into table"

Respond with ONLY the description, no JSON, no explanation."""

    # Multi-source transformation description
    MULTI_SOURCE_TRANSFORMATION = """Describe how multiple data sources are combined.

SOURCES: {sources}
TARGET: {target}
OPERATIONS: {operations}

CODE SNIPPET:
```python
{code_snippet}
```

Provide a concise description (5-10 words) of how these sources are combined.
Examples: "Join customer and order data", "Merge weather with location data"

Respond with ONLY the description."""

    # Stage description prompt
    STAGE_DESCRIPTION = """Describe this pipeline stage in one sentence (max 15 words).

STAGE TYPE: {stage_type}
OPERATIONS: {operations}

CODE:
```python
{code}
```

Provide a clear, concise description of what this stage does.
Example: "Load air quality data from CSV and store in table"

Respond with ONLY the description."""

    # Pipeline stage detection prompt
    STAGE_DETECTION = """Analyze this Jupyter notebook and identify logical pipeline stages.

CELLS:
{cell_summaries}

Group related cells into pipeline stages. Typical stages:
- setup: Imports, configuration, authentication
- data_loading: Loading datasets
- data_preparation: Cleaning, preprocessing
- reconciliation: Entity matching, geocoding
- enrichment: Data augmentation from external sources
- transformation: Feature engineering, aggregations
- analysis: Statistical analysis, modeling
- output: Saving results, visualization

Respond with JSON array grouping cell indices by stage:
[
  {{"stage": "setup", "cells": [0], "confidence": 0.95}},
  {{"stage": "data_loading", "cells": [1], "confidence": 0.9}},
  {{"stage": "enrichment", "cells": [3, 4, 5], "confidence": 0.85}},
  ...
]

Respond ONLY with the JSON array."""

    # Few-shot examples for classification
    FEW_SHOT_EXAMPLES = {
        "data_loading": [
            {
                "code": "df = pd.read_csv('data.csv')",
                "type": "data_loading",
                "reasoning": "Loading data from CSV file into DataFrame"
            },
            {
                "code": "data = requests.get(api_url).json()",
                "type": "data_loading",
                "reasoning": "Fetching data from external API"
            }
        ],
        "reconciliation": [
            {
                "code": "reconciled = reconcile_client.reconcile(df, 'name', 'company')",
                "type": "reconciliation",
                "reasoning": "Entity matching using reconciliation service"
            }
        ],
        "enrichment": [
            {
                "code": "df = extend_column(df, 'city', osm_service)",
                "type": "enrichment",
                "reasoning": "Enriching data with external OSM service"
            }
        ],
        "transformation": [
            {
                "code": "result = df.groupby('category').agg({'value': 'mean'})",
                "type": "transformation",
                "reasoning": "Aggregating data by category"
            }
        ]
    }
    
    @staticmethod
    def build_task_classification_prompt(code: str, metadata: Dict,
                                        task_type_descriptions: Dict[str, str]) -> str:
        """
        Build task classification prompt.
        
        Args:
            code: Code snippet
            metadata: Cell metadata
            task_type_descriptions: Dictionary of task type descriptions
            
        Returns:
            Formatted prompt string
        """
        desc_lines = [f"- {t}: {d}" for t, d in task_type_descriptions.items()]
        
        return PromptTemplates.TASK_CLASSIFICATION.format(
            code=code[:500],
            variables_defined=metadata.get('variables_defined', []),
            function_calls=metadata.get('function_calls', []),
            imports=metadata.get('imports', []),
            task_type_descriptions='\n'.join(desc_lines)
        )
    
    @staticmethod
    def build_react_prompt(code: str, context: Dict, known_types: List[str]) -> str:
        """
        Build ReAct-style reasoning prompt.
        
        Args:
            code: Code snippet
            context: Context dictionary
            known_types: List of known operation types
            
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.REACT_CLASSIFICATION.format(
            code=code[:800],
            variables_defined=context.get('variables_defined', []),
            function_calls=context.get('function_calls', []),
            previous_ops=context.get('previous_ops', []),
            known_types=', '.join(known_types)
        )
    
    @staticmethod
    def build_transformation_prompt(source: str, target: str,
                                   operations: List[str], code: str) -> str:
        """
        Build transformation description prompt.
        
        Args:
            source: Source artifact name
            target: Target artifact name
            operations: List of operations
            code: Code snippet
            
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.TRANSFORMATION_DESCRIPTION.format(
            source_artifact=source,
            target_artifact=target,
            operations=', '.join(operations[:5]),
            code_snippet=code[:300]
        )
    
    @staticmethod
    def build_multi_source_transformation_prompt(sources: List[str], target: str,
                                                 operations: List[str], code: str) -> str:
        """
        Build multi-source transformation prompt.
        
        Args:
            sources: List of source artifact names
            target: Target artifact name
            operations: List of operations
            code: Code snippet
            
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.MULTI_SOURCE_TRANSFORMATION.format(
            sources=', '.join(sources),
            target=target,
            operations=', '.join(operations[:5]),
            code_snippet=code[:300]
        )
    
    @staticmethod
    def build_stage_description_prompt(stage_type: str, operations: List[str],
                                      code: str) -> str:
        """
        Build stage description prompt.
        
        Args:
            stage_type: Type of pipeline stage
            operations: List of operations
            code: Combined code from stage
            
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.STAGE_DESCRIPTION.format(
            stage_type=stage_type,
            operations=', '.join(operations[:5]),
            code=code[:500]
        )
    
    @staticmethod
    def build_stage_detection_prompt(cell_summaries: str) -> str:
        """
        Build stage detection prompt.
        
        Args:
            cell_summaries: Formatted cell summaries
            
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.STAGE_DETECTION.format(
            cell_summaries=cell_summaries
        )
    
    @staticmethod
    def build_few_shot_prompt(base_prompt: str, operation_type: str,
                             num_examples: int = 2) -> str:
        """
        Build few-shot prompt with examples.
        
        Args:
            base_prompt: Base prompt template
            operation_type: Type of operation for examples
            num_examples: Number of examples to include
            
        Returns:
            Prompt with few-shot examples
        """
        examples = PromptTemplates.FEW_SHOT_EXAMPLES.get(operation_type, [])
        
        if not examples:
            return base_prompt
        
        example_text = "\n\nEXAMPLES:\n"
        for i, ex in enumerate(examples[:num_examples], 1):
            example_text += f"\nExample {i}:\n"
            example_text += f"Code: {ex['code']}\n"
            example_text += f"Type: {ex['type']}\n"
            example_text += f"Reasoning: {ex['reasoning']}\n"
        
        return base_prompt + example_text


class ChainOfThoughtPrompts:
    """
    Chain-of-thought prompts for complex reasoning tasks.
    
    These prompts encourage step-by-step reasoning for better
    accuracy on difficult classification tasks.
    """
    
    COT_CLASSIFICATION = """Analyze this code step by step to determine its operation type.

CODE:
```python
{code}
```

CONTEXT:
- Function calls: {function_calls}
- Variables: {variables}

Let's think through this step by step:

Step 1: What data structures are being used?
Step 2: What operations are being performed?
Step 3: What is the input and output?
Step 4: What is the high-level purpose?

Based on this analysis, classify the operation type.

Respond in JSON:
{{
    "step1_analysis": "...",
    "step2_analysis": "...",
    "step3_analysis": "...",
    "step4_analysis": "...",
    "operation_type": "type_name",
    "confidence": 0.0-1.0
}}"""
    
    @staticmethod
    def build_cot_prompt(code: str, context: Dict) -> str:
        """Build chain-of-thought prompt."""
        return ChainOfThoughtPrompts.COT_CLASSIFICATION.format(
            code=code[:600],
            function_calls=', '.join(context.get('function_calls', [])[:5]),
            variables=', '.join(context.get('variables_defined', [])[:5])
        )


__all__ = [
    "PromptTemplates",
    "ChainOfThoughtPrompts",
]