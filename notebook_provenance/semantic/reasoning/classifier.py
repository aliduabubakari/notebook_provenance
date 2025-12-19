"""
Hybrid Operation Classifier Module
===================================

Combines fixed taxonomy with LLM reasoning for operation classification.

This module provides the HybridOperationClassifier which:
- Uses fixed taxonomy for known patterns
- Falls back to LLM reasoning for novel operations
- Learns and expands taxonomy over time
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from notebook_provenance.core.enums import TaskType
from notebook_provenance.semantic.llm_analyzer import LLMSemanticAnalyzer
from notebook_provenance.semantic.reasoning.taxonomy import DynamicTaxonomy
from notebook_provenance.semantic.reasoning.prompts import PromptTemplates


class HybridOperationClassifier:
    """
    Combines fixed taxonomy with LLM reasoning for operation classification.
    
    Uses ReAct-style reasoning when operation doesn't fit predefined classes.
    This allows the system to handle novel operations while maintaining
    consistency for known patterns.
    
    Example:
        >>> classifier = HybridOperationClassifier(llm_analyzer)
        >>> op_type, confidence, reasoning = classifier.classify(
        ...     code_snippet, function_calls, context
        ... )
    """
    
    # Fixed taxonomy of known operations
    FIXED_TAXONOMY = {
        "data_loading": [
            "read_csv", "read_excel", "read_parquet", "read_json",
            "load", "fetch", "download", "query", "from_csv"
        ],
        "data_cleaning": [
            "dropna", "fillna", "drop_duplicates", "clean",
            "strip", "replace", "remove", "filter_invalid"
        ],
        "transformation": [
            "merge", "join", "groupby", "pivot", "melt",
            "apply", "transform", "map", "reshape"
        ],
        "enrichment": [
            "extend", "enrich", "geocode", "lookup",
            "augment", "join_external"
        ],
        "reconciliation": [
            "match", "dedupe", "reconcile", "link",
            "resolve", "merge_fuzzy"
        ],
        "aggregation": [
            "sum", "mean", "count", "aggregate", "agg",
            "median", "std", "var"
        ],
        "validation": [
            "validate", "check", "assert", "verify",
            "test", "check_schema"
        ],
        "output": [
            "to_csv", "to_parquet", "to_json", "save",
            "write", "export", "upload"
        ],
        "setup": [
            "import", "config", "initialize", "setup",
            "auth", "connect"
        ],
        "api_integration": [
            "requests", "api", "endpoint", "rest",
            "graphql", "client"
        ],
    }
    
    def __init__(self, llm_analyzer: Optional[LLMSemanticAnalyzer] = None,
                 confidence_threshold: float = 0.8,
                 use_few_shot: bool = False):
        """
        Initialize hybrid classifier.
        
        Args:
            llm_analyzer: Optional LLM analyzer for reasoning
            confidence_threshold: Threshold for using LLM reasoning
            use_few_shot: Whether to use few-shot learning
        """
        self.llm_analyzer = llm_analyzer
        self.confidence_threshold = confidence_threshold
        self.use_few_shot = use_few_shot
        self.dynamic_taxonomy = DynamicTaxonomy()
        self.prompt_templates = PromptTemplates()
        
        # Statistics
        self.classification_stats = {
            'total': 0,
            'fixed_taxonomy': 0,
            'dynamic_taxonomy': 0,
            'llm_reasoning': 0,
            'fallback': 0
        }
    
    def classify(self, code_snippet: str, function_calls: List[str], 
                 context: Dict) -> Tuple[str, float, str]:
        """
        Classify operation using hybrid approach.
        
        Args:
            code_snippet: Code to classify
            function_calls: List of function calls in the code
            context: Additional context (variables, previous ops, etc.)
            
        Returns:
            Tuple of (operation_type, confidence, reasoning)
        """
        self.classification_stats['total'] += 1
        
        # Step 1: Try fixed taxonomy matching
        matched_type, confidence = self._match_fixed_taxonomy(function_calls)
        
        if confidence >= self.confidence_threshold:
            self.classification_stats['fixed_taxonomy'] += 1
            return matched_type, confidence, "Matched via fixed taxonomy"
        
        # Step 2: Try dynamic taxonomy (learned types)
        dynamic_type, dynamic_conf = self.dynamic_taxonomy.match(function_calls)
        
        if dynamic_conf >= self.confidence_threshold:
            self.classification_stats['dynamic_taxonomy'] += 1
            return dynamic_type, dynamic_conf, "Matched via dynamic taxonomy"
        
        # Step 3: Use LLM reasoning for uncertain cases
        if self.llm_analyzer and self.llm_analyzer.enabled:
            try:
                result = self._llm_reason_and_classify(code_snippet, context)
                self.classification_stats['llm_reasoning'] += 1
                return result
            except Exception as e:
                print(f"Warning: LLM reasoning failed: {e}")
        
        # Fallback
        self.classification_stats['fallback'] += 1
        fallback_type = matched_type if matched_type != "other" else "transformation"
        return fallback_type, confidence, "Heuristic match (fallback)"
    
    def _match_fixed_taxonomy(self, function_calls: List[str]) -> Tuple[str, float]:
        """
        Match against fixed taxonomy.
        
        Args:
            function_calls: List of function calls
            
        Returns:
            Tuple of (operation_type, confidence)
        """
        if not function_calls:
            return "other", 0.3
        
        # Score each category
        scores = defaultdict(float)
        
        for func in function_calls:
            func_lower = func.lower()
            
            for category, keywords in self.FIXED_TAXONOMY.items():
                for keyword in keywords:
                    if keyword in func_lower:
                        # Exact match gets higher score
                        if keyword == func_lower or f".{keyword}" in func_lower:
                            scores[category] += 1.0
                        else:
                            scores[category] += 0.5
        
        if not scores:
            return "other", 0.3
        
        # Get best match
        best_category = max(scores, key=scores.get)
        max_score = scores[best_category]
        
        # Normalize confidence based on score and number of function calls
        total_calls = len(function_calls)
        confidence = min(0.95, 0.5 + (max_score / total_calls) * 0.45)
        
        return best_category, confidence
    
    def _llm_reason_and_classify(self, code: str, context: Dict) -> Tuple[str, float, str]:
        """
        Use LLM with ReAct-style reasoning for classification.
        
        Args:
            code: Code snippet
            context: Context dictionary
            
        Returns:
            Tuple of (operation_type, confidence, reasoning)
        """
        # Build prompt
        known_types = list(self.FIXED_TAXONOMY.keys()) + list(self.dynamic_taxonomy.discovered_types.keys())
        
        if self.use_few_shot:
            # Get relevant few-shot examples
            base_prompt = self.prompt_templates.build_react_prompt(code, context, known_types)
            # Add few-shot examples based on function calls
            prompt = base_prompt  # Can enhance with few-shot later
        else:
            prompt = self.prompt_templates.build_react_prompt(code, context, known_types)
        
        # Call LLM
        response = self.llm_analyzer._call_llm(prompt, temperature=0.3, max_tokens=400)
        
        # Parse response
        import json
        result = json.loads(response.strip())
        
        operation_type = result['operation_type']
        confidence = result['confidence']
        reasoning = result['reasoning']
        
        # If it's a new type, add to dynamic taxonomy
        if result.get('is_new_type', False) and operation_type not in self.FIXED_TAXONOMY:
            self.dynamic_taxonomy.add_type(
                operation_type,
                examples=context.get('function_calls', []),
                confidence=confidence
            )
        
        return operation_type, confidence, reasoning
    
    def _is_known_type(self, operation_type: str) -> bool:
        """Check if operation type is in fixed taxonomy."""
        return operation_type in self.FIXED_TAXONOMY
    
    def get_statistics(self) -> Dict:
        """
        Get classification statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.classification_stats.copy()
        
        if stats['total'] > 0:
            stats['fixed_taxonomy_pct'] = stats['fixed_taxonomy'] / stats['total'] * 100
            stats['dynamic_taxonomy_pct'] = stats['dynamic_taxonomy'] / stats['total'] * 100
            stats['llm_reasoning_pct'] = stats['llm_reasoning'] / stats['total'] * 100
            stats['fallback_pct'] = stats['fallback'] / stats['total'] * 100
        
        # Add dynamic taxonomy stats
        stats['dynamic_taxonomy_info'] = self.dynamic_taxonomy.get_statistics()
        
        return stats
    
    def reset_statistics(self):
        """Reset classification statistics."""
        self.classification_stats = {
            'total': 0,
            'fixed_taxonomy': 0,
            'dynamic_taxonomy': 0,
            'llm_reasoning': 0,
            'fallback': 0
        }


__all__ = [
    "HybridOperationClassifier",
]