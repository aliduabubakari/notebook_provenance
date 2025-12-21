"""
Test with pattern-only classification (no LLM)
"""

from notebook_provenance import NotebookProvenanceSystem
from notebook_provenance.core.config import Config, ClassificationConfig, LLMConfig

# Config with LLM disabled for classification
config = Config(
    llm=LLMConfig(
        enabled=False,  # Disable LLM entirely
    ),
    classification=ClassificationConfig(
        use_llm=False,
        use_embeddings=False,
        use_semantic_deduplication=False,
    ),
    verbose=True
)

system = NotebookProvenanceSystem(config=config)

# Analyze
result = system.analyze_file(
    "test_notebooks/base_notebook_file_2025-09-05_13-27.ipynb",
    output_prefix="test_pattern_only",
    save_outputs=True
)

print("\n✅ Analysis complete!")
print(f"Found {len(result['artifacts'])} artifacts")
print(f"Found {len(result['transformations'])} transformations")