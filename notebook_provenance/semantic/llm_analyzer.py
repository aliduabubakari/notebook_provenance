"""
LLM Analyzer Module
===================

LLM integration for semantic analysis of code and data flow.

This module provides the LLMSemanticAnalyzer class which:
- Classifies cell task types using LLM
- Describes transformations
- Detects pipeline stages
- Provides reasoning capabilities
"""

import json
import time
from typing import Dict, List, Tuple, Optional, Any

from notebook_provenance.core.enums import TaskType, PipelineStage
from notebook_provenance.core.config import LLMConfig


class LLMSemanticAnalyzer:
    """
    Use LLM for semantic analysis of code and data flow.
    
    This class integrates with LLM APIs to provide intelligent
    semantic understanding of notebook code, including task classification,
    transformation descriptions, and pipeline stage detection.
    
    Example:
        >>> analyzer = LLMSemanticAnalyzer(api_key="your_key")
        >>> task_type, confidence = analyzer.classify_cell_task(code, metadata)
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = "https://api.deepinfra.com/v1/openai",
                 config: Optional[LLMConfig] = None):
        """
        Initialize LLM analyzer.
        
        Args:
            api_key: API key for LLM service
            base_url: Base URL for LLM API
            config: Optional LLMConfig object
        """
        if config:
            self.config = config
        else:
            self.config = LLMConfig(
                api_key=api_key,
                base_url=base_url
            )
        
        self.enabled = False
        self.client = None
        
        # Try to initialize OpenAI client
        if self.config.enabled and self.config.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
                self.enabled = True
            except ImportError:
                print("Warning: openai package not installed. LLM features disabled.")
                print("Install with: pip install openai")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
    
    def classify_cell_task(self, cell_code: str, 
                          cell_metadata: Dict) -> Tuple[TaskType, float]:
        """
        Use LLM to classify cell task type.
        
        Args:
            cell_code: Code content of the cell
            cell_metadata: Metadata including variables, functions, imports
            
        Returns:
            Tuple of (TaskType, confidence_score)
        """
        if not self.enabled:
            return self._heuristic_classify(cell_code, cell_metadata)
        
        prompt = self._build_classification_prompt(cell_code, cell_metadata)
        
        try:
            response = self._call_llm(
                prompt,
                temperature=0,
                max_tokens=200
            )
            
            result = json.loads(response.strip())
            task_type = TaskType(result['task_type'])
            confidence = float(result.get('confidence', 0.5))
            
            return task_type, confidence
            
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return self._heuristic_classify(cell_code, cell_metadata)
    
    def _build_classification_prompt(self, code: str, metadata: Dict) -> str:
        """Build prompt for task classification."""
        task_types = [t.value for t in TaskType]
        
        prompt = f"""Analyze this Python code cell and classify its primary task type.

CODE:
```python
{code[:500]}
```

METADATA:
- Variables defined: {metadata.get('variables_defined', [])}
- Function calls: {metadata.get('function_calls', [])}
- Imports: {metadata.get('imports', [])}

TASK TYPES:
{chr(10).join(f"- {t}: {self._get_task_description(t)}" for t in task_types)}

Respond ONLY with valid JSON:
{{"task_type": "task_name", "confidence": 0.95, "reasoning": "brief explanation"}}"""
        
        return prompt
    
    def _get_task_description(self, task_type: str) -> str:
        """Get human-readable description of task type."""
        descriptions = {
            'data_loading': 'Loading data from files, databases, or APIs',
            'data_cleaning': 'Data preprocessing, cleaning, deduplication',
            'reconciliation': 'Entity matching, geocoding, linking',
            'extension': 'Data enrichment from external sources',
            'transformation': 'Data transformation, feature engineering',
            'analysis': 'Data analysis, statistics, exploration',
            'visualization': 'Plotting, charting, visual representation',
            'model_training': 'Machine learning model training',
            'evaluation': 'Model evaluation, metrics computation',
            'setup': 'Configuration, imports, initialization',
            'api_integration': 'Integrating with external APIs',
            'schema_validation': 'Validating data schemas',
            'data_partitioning': 'Partitioning or sharding data',
            'other': 'Other operations',
        }
        return descriptions.get(task_type, 'Unknown task')
    
    def describe_transformation(self, source_artifact: str, target_artifact: str,
                               function_calls: List[str], code_snippet: str) -> str:
        """
        Use LLM to describe transformation between artifacts.
        
        Args:
            source_artifact: Source artifact name
            target_artifact: Target artifact name
            function_calls: List of function calls involved
            code_snippet: Code snippet showing transformation
            
        Returns:
            Human-readable description string
        """
        if not self.enabled:
            return self._heuristic_describe_transformation(function_calls)
        
        prompt = f"""Describe the data transformation in 5-10 words.

SOURCE: {source_artifact}
TARGET: {target_artifact}
OPERATIONS: {', '.join(function_calls[:5])}

CODE SNIPPET:
```python
{code_snippet[:300]}
```

Provide a concise, human-readable description of what this transformation does.
Examples: "Geocode locations", "Enrich with weather data", "Load CSV into table"

Respond with ONLY the description, no JSON, no explanation."""

        try:
            response = self._call_llm(
                prompt,
                temperature=0.3,
                max_tokens=50
            )
            
            description = response.strip().strip('"').strip("'")
            return description[:100]
            
        except Exception as e:
            print(f"LLM transformation description failed: {e}")
            return self._heuristic_describe_transformation(function_calls)
    
    def detect_pipeline_stages(self, parsed_cells: List[Dict]) -> List[Dict]:
        """
        Use LLM to detect pipeline stage boundaries.
        
        Args:
            parsed_cells: List of parsed cell dictionaries
            
        Returns:
            List of dictionaries with 'stage' and 'cells' keys
        """
        if not self.enabled:
            return self._heuristic_stage_detection(parsed_cells)
        
        # Create cell summaries
        cell_summaries = []
        for i, cell in enumerate(parsed_cells):
            if 'error' in cell:
                continue
            
            summary = {
                'index': i,
                'cell_id': cell['cell_id'],
                'variables': cell.get('variables_defined', [])[:3],
                'functions': cell.get('function_calls', [])[:3],
                'code_preview': cell['code'][:150]
            }
            cell_summaries.append(summary)
        
        prompt = self._build_stage_detection_prompt(cell_summaries)
        
        try:
            response = self._call_llm(
                prompt,
                temperature=0,
                max_tokens=1000
            )
            
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            print(f"LLM stage detection failed: {e}")
            return self._heuristic_stage_detection(parsed_cells)
    
    def _build_stage_detection_prompt(self, cell_summaries: List[Dict]) -> str:
        """Build prompt for stage detection."""
        cells_text = self._format_cell_summaries(cell_summaries)
        
        prompt = f"""Analyze this Jupyter notebook and identify logical pipeline stages.

CELLS:
{cells_text}

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
        
        return prompt
    
    def _format_cell_summaries(self, summaries: List[Dict]) -> str:
        """Format cell summaries for LLM."""
        lines = []
        for s in summaries:
            lines.append(f"Cell {s['index']} ({s['cell_id']}):")
            if s['variables']:
                lines.append(f"  Vars: {', '.join(s['variables'])}")
            if s['functions']:
                lines.append(f"  Funcs: {', '.join(s['functions'])}")
            lines.append(f"  Code: {s['code_preview']}...")
            lines.append("")
        return '\n'.join(lines)
    
    def reason_and_classify(self, code: str, context: Dict) -> Tuple[str, float, str]:
        """
        Use ReAct-style reasoning to classify operation.
        
        Args:
            code: Code snippet
            context: Context dictionary
            
        Returns:
            Tuple of (operation_type, confidence, reasoning)
        """
        if not self.enabled:
            return "other", 0.5, "LLM not available"
        
        from notebook_provenance.core.enums import TaskType
        known_types = [t.value for t in TaskType]
        
        prompt = f"""You are analyzing a code snippet from a data pipeline notebook.

CODE:
```python
{code[:800]}
```

CONTEXT:
- Variables defined: {context.get('variables_defined', [])}
- Function calls: {context.get('function_calls', [])}
- Previous operations: {context.get('previous_ops', [])}

KNOWN OPERATION TYPES: {', '.join(known_types)}

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

        try:
            response = self._call_llm(prompt, temperature=0.3, max_tokens=400)
            result = json.loads(response.strip())
            
            return (
                result['operation_type'],
                result['confidence'],
                result['reasoning']
            )
        except Exception as e:
            print(f"LLM reasoning failed: {e}")
            return "other", 0.5, f"Error: {e}"
    
    def _call_llm(self, prompt: str, temperature: float = 0.0, 
                  max_tokens: int = 500) -> str:
        """
        Call LLM API with retry logic.
        
        Args:
            prompt: Prompt text
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response text
        """
        if not self.enabled or not self.client:
            raise RuntimeError("LLM client not initialized")
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.timeout
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise e
    
    def _heuristic_classify(self, code: str, metadata: Dict) -> Tuple[TaskType, float]:
        """
        Fallback heuristic classification.
        
        Args:
            code: Code content
            metadata: Cell metadata
            
        Returns:
            Tuple of (TaskType, confidence)
        """
        code_lower = code.lower()
        
        patterns = [
            (['import', 'authmanager', 'token', 'config'], TaskType.SETUP, 0.9),
            (['read_csv', 'load', 'pd.read', 'from_csv'], TaskType.DATA_LOADING, 0.9),
            (['reconcile', 'geocoding', 'match'], TaskType.RECONCILIATION, 0.85),
            (['extend', 'enrich', 'osm', 'openmeteo', 'api'], TaskType.EXTENSION, 0.85),
            (['drop', 'fillna', 'dropna', 'clean'], TaskType.DATA_CLEANING, 0.8),
            (['plt.', 'plot', 'chart', 'visualize'], TaskType.VISUALIZATION, 0.8),
            (['fit', 'train', 'model.'], TaskType.MODEL_TRAINING, 0.8),
            (['merge', 'join', 'concat'], TaskType.TRANSFORMATION, 0.75),
        ]
        
        for keywords, task_type, confidence in patterns:
            if any(kw in code_lower for kw in keywords):
                return task_type, confidence
        
        return TaskType.OTHER, 0.5
    
    def _heuristic_describe_transformation(self, function_calls: List[str]) -> str:
        """Fallback transformation description."""
        if not function_calls:
            return "Transform data"
        
        func = function_calls[0].lower()
        
        transformation_map = {
            'reconcile': 'Reconcile entities',
            'extend_column': 'Extend with external data',
            'extend': 'Enrich data',
            'read_csv': 'Load from CSV',
            'add_table': 'Store in table',
            'merge': 'Merge datasets',
            'join': 'Join tables',
            'filter': 'Filter data',
            'aggregate': 'Aggregate values',
            'osm': 'Enrich with OSM data',
            'llm': 'Classify with LLM',
            'geocode': 'Geocode locations',
        }
        
        for key, desc in transformation_map.items():
            if key in func:
                return desc
        
        return f"Apply {function_calls[0]}"
    
    def _heuristic_stage_detection(self, parsed_cells: List[Dict]) -> List[Dict]:
        """Fallback heuristic stage detection."""
        stages = []
        current_stage = None
        current_cells = []
        
        for i, cell in enumerate(parsed_cells):
            if 'error' in cell:
                continue
            
            # Classify cell
            stage_type = self._classify_cell_simple(cell)
            
            if current_stage is None or current_stage == stage_type:
                current_stage = stage_type
                current_cells.append(i)
            else:
                # Save previous stage
                if current_cells:
                    stages.append({
                        'stage': current_stage,
                        'cells': current_cells,
                        'confidence': 0.7
                    })
                
                # Start new stage
                current_stage = stage_type
                current_cells = [i]
        
        # Don't forget last stage
        if current_cells:
            stages.append({
                'stage': current_stage,
                'cells': current_cells,
                'confidence': 0.7
            })
        
        return stages
    
    def _classify_cell_simple(self, cell: Dict) -> str:
        """Simple cell classification."""
        code = cell['code'].lower()
        
        if any(kw in code for kw in ['import', 'authmanager']):
            return 'setup'
        elif any(kw in code for kw in ['read_csv', 'load']):
            return 'data_loading'
        elif 'reconcile' in code:
            return 'reconciliation'
        elif any(kw in code for kw in ['extend', 'enrich', 'osm', 'llm']):
            return 'enrichment'
        else:
            return 'transformation'


__all__ = [
    "LLMSemanticAnalyzer",
]