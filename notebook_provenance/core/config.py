"""
Configuration Module
====================

Configuration management for the provenance system.

This module provides:
- Config: Main configuration class
- LLMConfig: LLM-specific configuration
- VisualizationConfig: Visualization settings
- EvaluationConfig: Evaluation framework settings
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import os


@dataclass
class LLMConfig:
    """
    Configuration for LLM integration.
    
    Attributes:
        enabled: Whether to use LLM features
        api_key: API key for LLM service
        base_url: Base URL for LLM API
        model: Model identifier
        temperature: Temperature for generation
        max_tokens: Maximum tokens per request
        timeout: Request timeout in seconds
        retry_attempts: Number of retry attempts
    """
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: str = "https://api.deepinfra.com/v1/openai"
    model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 30
    retry_attempts: int = 3
    
    def __post_init__(self):
        """Load API key from environment if not provided"""
        if self.api_key is None:
            self.api_key = os.getenv('PROVENANCE_API_KEY')
        
        # Disable if no API key available
        if not self.api_key:
            self.enabled = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)"""
        data = asdict(self)
        data['api_key'] = '***' if self.api_key else None
        return data


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization generation.
    
    Attributes:
        enabled: Whether to generate visualizations
        dpi: Resolution for output images
        figsize_stages: Figure size for pipeline stages
        figsize_lineage: Figure size for artifact lineage
        figsize_dfg: Figure size for DFG visualization
        color_scheme: Color scheme name
        interactive_html: Whether to generate interactive HTML
    """
    enabled: bool = True
    dpi: int = 300
    figsize_stages: tuple = (24, 8)
    figsize_lineage: tuple = (24, 12)
    figsize_dfg: tuple = (20, 14)
    color_scheme: str = "default"
    interactive_html: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation framework.
    
    Attributes:
        enabled: Whether evaluation is enabled
        metrics: List of metrics to compute
        ground_truth_path: Path to ground truth annotations
        output_path: Path to save evaluation results
        compute_ged: Whether to compute graph edit distance (expensive)
    """
    enabled: bool = False
    metrics: List[str] = field(default_factory=lambda: [
        'node_classification_f1',
        'artifact_detection_f1',
        'lineage_edge_f1',
        'stage_sequence_lcs'
    ])
    ground_truth_path: Optional[Path] = None
    output_path: Path = Path("evaluation_results")
    compute_ged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.ground_truth_path:
            data['ground_truth_path'] = str(self.ground_truth_path)
        data['output_path'] = str(self.output_path)
        return data


@dataclass
class ParsingConfig:
    """
    Configuration for code parsing.
    
    Attributes:
        min_cell_lines: Minimum lines to consider a cell valid
        max_complexity: Maximum complexity threshold
        filter_noise: Whether to filter noise patterns
        noise_patterns: Patterns to filter as noise
    """
    min_cell_lines: int = 1
    max_complexity: int = 1000
    filter_noise: bool = True
    noise_patterns: List[str] = field(default_factory=lambda: [
        'manager', 'client', 'config', 'token', 'api_key',
        'username', 'password', 'url', 'connection', 'session'
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class GraphConfig:
    """
    Configuration for graph construction.
    
    Attributes:
        build_clean_dfg: Whether to build a cleaned DFG
        artifact_importance_threshold: Threshold for artifact importance
        max_nodes: Maximum nodes in graph (for performance)
        include_literals: Whether to include literal nodes
        include_imports: Whether to include import nodes
    """
    build_clean_dfg: bool = True
    artifact_importance_threshold: float = 7.0
    max_nodes: int = 10000
    include_literals: bool = False
    include_imports: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Config:
    """
    Main configuration for the provenance system.
    
    Attributes:
        llm: LLM configuration
        visualization: Visualization configuration
        evaluation: Evaluation configuration
        parsing: Parsing configuration
        graph: Graph construction configuration
        output_dir: Output directory for results
        verbose: Verbosity level
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    output_dir: Path = Path("provenance_output")
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize configuration"""
        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure evaluation output directory exists
        if self.evaluation.enabled and self.evaluation.output_path:
            self.evaluation.output_path = Path(self.evaluation.output_path)
            self.evaluation.output_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'llm': self.llm.to_dict(),
            'visualization': self.visualization.to_dict(),
            'evaluation': self.evaluation.to_dict(),
            'parsing': self.parsing.to_dict(),
            'graph': self.graph.to_dict(),
            'output_dir': str(self.output_dir),
            'verbose': self.verbose,
        }
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load configuration from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            llm=LLMConfig(**data.get('llm', {})),
            visualization=VisualizationConfig(**data.get('visualization', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            parsing=ParsingConfig(**data.get('parsing', {})),
            graph=GraphConfig(**data.get('graph', {})),
            output_dir=Path(data.get('output_dir', 'provenance_output')),
            verbose=data.get('verbose', True),
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        return cls(
            llm=LLMConfig(**data.get('llm', {})),
            visualization=VisualizationConfig(**data.get('visualization', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            parsing=ParsingConfig(**data.get('parsing', {})),
            graph=GraphConfig(**data.get('graph', {})),
            output_dir=Path(data.get('output_dir', 'provenance_output')),
            verbose=data.get('verbose', True),
        )


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def get_config_from_env() -> Config:
    """
    Create configuration from environment variables.
    
    Environment variables:
        PROVENANCE_API_KEY: API key for LLM
        PROVENANCE_BASE_URL: Base URL for LLM API
        PROVENANCE_OUTPUT_DIR: Output directory
        PROVENANCE_VERBOSE: Verbosity (true/false)
        PROVENANCE_DPI: DPI for visualizations
    """
    config = Config()
    
    # LLM configuration from environment
    if os.getenv('PROVENANCE_API_KEY'):
        config.llm.api_key = os.getenv('PROVENANCE_API_KEY')
        config.llm.enabled = True
    
    if os.getenv('PROVENANCE_BASE_URL'):
        config.llm.base_url = os.getenv('PROVENANCE_BASE_URL')
    
    # Output configuration
    if os.getenv('PROVENANCE_OUTPUT_DIR'):
        config.output_dir = Path(os.getenv('PROVENANCE_OUTPUT_DIR'))
    
    # Verbosity
    if os.getenv('PROVENANCE_VERBOSE'):
        config.verbose = os.getenv('PROVENANCE_VERBOSE').lower() == 'true'
    
    # Visualization
    if os.getenv('PROVENANCE_DPI'):
        config.visualization.dpi = int(os.getenv('PROVENANCE_DPI'))
    
    return config


# Export configuration classes
__all__ = [
    "Config",
    "LLMConfig",
    "VisualizationConfig",
    "EvaluationConfig",
    "ParsingConfig",
    "GraphConfig",
    "get_default_config",
    "get_config_from_env",
]