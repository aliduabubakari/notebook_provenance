"""
Reasoning Module
================

Hybrid operation classification with ReAct-style reasoning.

This module provides:
- HybridOperationClassifier: Combines fixed taxonomy with LLM reasoning
- DynamicTaxonomy: Expandable operation taxonomy
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime

from notebook_provenance.core.enums import TaskType
from notebook_provenance.semantic.llm_analyzer import LLMSemanticAnalyzer


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
        "data_loading": ["read_csv", "load", "fetch", "download", "query", "read_excel", "read_parquet"],
        "data_cleaning": ["dropna", "fillna", "drop_duplicates", "clean", "strip", "replace"],
        "transformation": ["merge", "join", "groupby", "pivot", "melt", "apply", "transform"],
        "enrichment": ["extend", "enrich", "geocode", "lookup", "augment"],
        "reconciliation": ["match", "dedupe", "reconcile", "link", "resolve"],
        "aggregation": ["sum", "mean", "count", "aggregate", "agg"],
        "validation": ["validate", "check", "assert", "verify", "test"],
        "output": ["to_csv", "save", "write", "export", "upload", "to_parquet"],
        "setup": ["import", "config", "initialize", "setup", "auth"],
    }
    
    def __init__(self, llm_analyzer: Optional[LLMSemanticAnalyzer] = None,
                 confidence_threshold: float = 0.8):
        """
        Initialize hybrid classifier.
        
        Args:
            llm_analyzer: Optional LLM analyzer for reasoning
            confidence_threshold: Threshold for using LLM reasoning
        """
        self.llm_analyzer = llm_analyzer
        self.confidence_threshold = confidence_threshold
        self.dynamic_taxonomy = DynamicTaxonomy()
    
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
        # Step 1: Try fixed taxonomy matching
        matched_type, confidence = self._match_fixed_taxonomy(function_calls)
        
        if confidence >= self.confidence_threshold:
            return matched_type, confidence, "Matched via fixed taxonomy"
        
        # Step 2: Try dynamic taxonomy (learned types)
        dynamic_type, dynamic_conf = self.dynamic_taxonomy.match(function_calls)
        
        if dynamic_conf >= self.confidence_threshold:
            return dynamic_type, dynamic_conf, "Matched via dynamic taxonomy"
        
        # Step 3: Use LLM reasoning for uncertain cases
        if self.llm_analyzer and self.llm_analyzer.enabled:
            return self._llm_reason_and_classify(code_snippet, context)
        
        # Fallback
        return matched_type if matched_type != "other" else "transformation", confidence, "Heuristic match"
    
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
                        scores[category] += 1.0
        
        if not scores:
            return "other", 0.3
        
        # Get best match
        best_category = max(scores, key=scores.get)
        max_score = scores[best_category]
        
        # Normalize confidence
        confidence = min(0.9, 0.5 + (max_score / len(function_calls)) * 0.4)
        
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
        try:
            operation_type, confidence, reasoning = self.llm_analyzer.reason_and_classify(
                code, context
            )
            
            # If it's a new type, add to dynamic taxonomy
            if not self._is_known_type(operation_type):
                self.dynamic_taxonomy.add_type(
                    operation_type,
                    examples=context.get('function_calls', []),
                    confidence=confidence
                )
            
            return operation_type, confidence, reasoning
            
        except Exception as e:
            return "other", 0.4, f"LLM reasoning failed: {e}"
    
    def _is_known_type(self, operation_type: str) -> bool:
        """Check if operation type is in fixed taxonomy."""
        return operation_type in self.FIXED_TAXONOMY


class DynamicTaxonomy:
    """
    Maintains and expands the operation taxonomy based on LLM discoveries.
    
    This class learns new operation types over time and can promote
    frequently-seen types to the fixed taxonomy.
    
    Example:
        >>> taxonomy = DynamicTaxonomy()
        >>> taxonomy.add_type("api_integration", ["requests.get", "api.call"])
        >>> matched_type, confidence = taxonomy.match(["requests.post"])
    """
    
    def __init__(self):
        """Initialize dynamic taxonomy."""
        self.discovered_types = {}  # type -> metadata
        self.type_examples = defaultdict(list)  # type -> list of examples
        self.type_patterns = defaultdict(set)  # type -> set of patterns
    
    def add_type(self, type_name: str, examples: List[str], confidence: float):
        """
        Add a new discovered type.
        
        Args:
            type_name: Name of the operation type
            examples: Example function calls
            confidence: Confidence in this type
        """
        if type_name not in self.discovered_types:
            self.discovered_types[type_name] = {
                'confidence': confidence,
                'example_count': len(examples),
                'first_seen': datetime.now(),
                'last_seen': datetime.now()
            }
        else:
            # Update existing type
            self.discovered_types[type_name]['example_count'] += len(examples)
            self.discovered_types[type_name]['last_seen'] = datetime.now()
            # Update confidence (running average)
            old_conf = self.discovered_types[type_name]['confidence']
            self.discovered_types[type_name]['confidence'] = (old_conf + confidence) / 2
        
        # Add examples
        self.type_examples[type_name].extend(examples)
        
        # Extract patterns (simple word extraction)
        for example in examples:
            words = example.lower().split('.')
            for word in words:
                if len(word) > 2:  # Skip very short words
                    self.type_patterns[type_name].add(word)
    
    def match(self, function_calls: List[str]) -> Tuple[str, float]:
        """
        Match function calls against dynamic taxonomy.
        
        Args:
            function_calls: List of function calls
            
        Returns:
            Tuple of (matched_type, confidence)
        """
        if not function_calls or not self.discovered_types:
            return "other", 0.0
        
        # Score each type
        scores = defaultdict(float)
        
        for func in function_calls:
            func_lower = func.lower()
            words = func_lower.split('.')
            
            for type_name, patterns in self.type_patterns.items():
                matches = sum(1 for word in words if word in patterns)
                if matches > 0:
                    scores[type_name] += matches / len(patterns)
        
        if not scores:
            return "other", 0.0
        
        # Get best match
        best_type = max(scores, key=scores.get)
        
        # Confidence based on score and type metadata
        base_confidence = scores[best_type]
        type_confidence = self.discovered_types[best_type]['confidence']
        example_bonus = min(0.2, self.discovered_types[best_type]['example_count'] / 50)
        
        confidence = min(0.9, base_confidence * 0.5 + type_confidence * 0.3 + example_bonus)
        
        return best_type, confidence
    
    def promote_to_fixed(self, type_name: str, threshold: int = 10) -> Optional[Dict]:
        """
        Promote frequently-seen types to fixed taxonomy.
        
        Args:
            type_name: Type to promote
            threshold: Minimum examples required
            
        Returns:
            Dictionary with type info if promoted, None otherwise
        """
        if type_name not in self.discovered_types:
            return None
        
        if len(self.type_examples[type_name]) >= threshold:
            # Extract common patterns
            patterns = self._extract_common_patterns(type_name)
            
            return {
                'type': type_name,
                'patterns': patterns,
                'examples': len(self.type_examples[type_name]),
                'confidence': self.discovered_types[type_name]['confidence']
            }
        
        return None
    
    def _extract_common_patterns(self, type_name: str) -> List[str]:
        """
        Extract common patterns for a type.
        
        Args:
            type_name: Type name
            
        Returns:
            List of common patterns
        """
        examples = self.type_examples[type_name]
        
        # Count word frequencies
        word_counts = defaultdict(int)
        for example in examples:
            words = example.lower().split('.')
            for word in words:
                if len(word) > 2:
                    word_counts[word] += 1
        
        # Get top patterns (appearing in at least 30% of examples)
        threshold = len(examples) * 0.3
        common_patterns = [
            word for word, count in word_counts.items()
            if count >= threshold
        ]
        
        return sorted(common_patterns, key=lambda w: word_counts[w], reverse=True)[:10]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the dynamic taxonomy.
        
        Returns:
            Dictionary of statistics
        """
        total_types = len(self.discovered_types)
        total_examples = sum(len(examples) for examples in self.type_examples.values())
        
        promotable = sum(
            1 for type_name in self.discovered_types
            if len(self.type_examples[type_name]) >= 10
        )
        
        return {
            'total_types': total_types,
            'total_examples': total_examples,
            'avg_examples_per_type': total_examples / total_types if total_types > 0 else 0,
            'promotable_types': promotable,
            'types': list(self.discovered_types.keys())
        }
    
    def save(self, filepath: str):
        """Save dynamic taxonomy to file."""
        import json
        
        data = {
            'types': self.discovered_types,
            'examples': {k: list(v) for k, v in self.type_examples.items()},
            'patterns': {k: list(v) for k, v in self.type_patterns.items()}
        }
        
        # Convert datetime objects to strings
        for type_name, metadata in data['types'].items():
            metadata['first_seen'] = metadata['first_seen'].isoformat()
            metadata['last_seen'] = metadata['last_seen'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load dynamic taxonomy from file."""
        import json
        from datetime import datetime
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.discovered_types = data['types']
        self.type_examples = defaultdict(list, data['examples'])
        self.type_patterns = defaultdict(set, {
            k: set(v) for k, v in data['patterns'].items()
        })
        
        # Convert strings back to datetime
        for type_name, metadata in self.discovered_types.items():
            metadata['first_seen'] = datetime.fromisoformat(metadata['first_seen'])
            metadata['last_seen'] = datetime.fromisoformat(metadata['last_seen'])


__all__ = [
    "HybridOperationClassifier",
    "DynamicTaxonomy",
]