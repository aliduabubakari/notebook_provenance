"""
Dynamic Taxonomy Module
=======================

Maintains and expands the operation taxonomy based on LLM discoveries.

This module provides the DynamicTaxonomy class which:
- Learns new operation types over time
- Extracts patterns from examples
- Promotes frequently-seen types
- Persists learned knowledge
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime
import json


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
    
    def add_type(self, type_name: str, examples: List[str], 
                 confidence: float, metadata: Optional[Dict] = None):
        """
        Add a new discovered type.
        
        Args:
            type_name: Name of the operation type
            examples: Example function calls
            confidence: Confidence in this type
            metadata: Optional additional metadata
        """
        if type_name not in self.discovered_types:
            self.discovered_types[type_name] = {
                'confidence': confidence,
                'example_count': len(examples),
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'metadata': metadata or {}
            }
        else:
            # Update existing type
            self.discovered_types[type_name]['example_count'] += len(examples)
            self.discovered_types[type_name]['last_seen'] = datetime.now()
            
            # Update confidence (running average)
            old_conf = self.discovered_types[type_name]['confidence']
            old_count = self.discovered_types[type_name]['example_count'] - len(examples)
            new_count = self.discovered_types[type_name]['example_count']
            
            # Weighted average
            weighted_conf = (old_conf * old_count + confidence * len(examples)) / new_count
            self.discovered_types[type_name]['confidence'] = weighted_conf
        
        # Add examples
        self.type_examples[type_name].extend(examples)
        
        # Extract patterns (simple word extraction)
        for example in examples:
            words = example.lower().replace('.', ' ').replace('_', ' ').split()
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
            func_lower = func.lower().replace('.', ' ').replace('_', ' ')
            words = func_lower.split()
            
            for type_name, patterns in self.type_patterns.items():
                matches = sum(1 for word in words if word in patterns)
                if matches > 0:
                    # Score based on proportion of patterns matched
                    scores[type_name] += matches / len(patterns)
        
        if not scores:
            return "other", 0.0
        
        # Get best match
        best_type = max(scores, key=scores.get)
        
        # Confidence based on score and type metadata
        base_confidence = min(1.0, scores[best_type])
        type_confidence = self.discovered_types[best_type]['confidence']
        
        # Bonus for mature types (more examples)
        example_count = self.discovered_types[best_type]['example_count']
        maturity_bonus = min(0.2, example_count / 50)
        
        # Combined confidence
        confidence = min(0.95, 
                        base_confidence * 0.4 + 
                        type_confidence * 0.4 + 
                        maturity_bonus)
        
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
                'confidence': self.discovered_types[type_name]['confidence'],
                'first_seen': self.discovered_types[type_name]['first_seen'].isoformat(),
                'example_list': self.type_examples[type_name][:20]  # Sample examples
            }
        
        return None
    
    def _extract_common_patterns(self, type_name: str, min_frequency: float = 0.3) -> List[str]:
        """
        Extract common patterns for a type.
        
        Args:
            type_name: Type name
            min_frequency: Minimum frequency threshold (0-1)
            
        Returns:
            List of common patterns
        """
        examples = self.type_examples[type_name]
        
        # Count word frequencies
        word_counts = defaultdict(int)
        for example in examples:
            words = example.lower().replace('.', ' ').replace('_', ' ').split()
            for word in words:
                if len(word) > 2:
                    word_counts[word] += 1
        
        # Get top patterns (appearing in at least min_frequency of examples)
        threshold = len(examples) * min_frequency
        common_patterns = [
            word for word, count in word_counts.items()
            if count >= threshold
        ]
        
        # Sort by frequency
        common_patterns.sort(key=lambda w: word_counts[w], reverse=True)
        
        return common_patterns[:15]  # Return top 15
    
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
        
        # Most common types
        top_types = sorted(
            self.discovered_types.items(),
            key=lambda x: x[1]['example_count'],
            reverse=True
        )[:5]
        
        return {
            'total_types': total_types,
            'total_examples': total_examples,
            'avg_examples_per_type': total_examples / total_types if total_types > 0 else 0,
            'promotable_types': promotable,
            'type_names': list(self.discovered_types.keys()),
            'top_types': [(name, meta['example_count']) for name, meta in top_types]
        }
    
    def save(self, filepath: str):
        """
        Save dynamic taxonomy to file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'version': '1.0',
            'saved_at': datetime.now().isoformat(),
            'types': {},
            'examples': {k: list(v) for k, v in self.type_examples.items()},
            'patterns': {k: list(v) for k, v in self.type_patterns.items()}
        }
        
        # Convert datetime objects to strings
        for type_name, metadata in self.discovered_types.items():
            data['types'][type_name] = {
                'confidence': metadata['confidence'],
                'example_count': metadata['example_count'],
                'first_seen': metadata['first_seen'].isoformat(),
                'last_seen': metadata['last_seen'].isoformat(),
                'metadata': metadata.get('metadata', {})
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load dynamic taxonomy from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load types
        self.discovered_types = {}
        for type_name, metadata in data['types'].items():
            self.discovered_types[type_name] = {
                'confidence': metadata['confidence'],
                'example_count': metadata['example_count'],
                'first_seen': datetime.fromisoformat(metadata['first_seen']),
                'last_seen': datetime.fromisoformat(metadata['last_seen']),
                'metadata': metadata.get('metadata', {})
            }
        
        # Load examples and patterns
        self.type_examples = defaultdict(list, data['examples'])
        self.type_patterns = defaultdict(set, {
            k: set(v) for k, v in data['patterns'].items()
        })
    
    def merge_with(self, other: 'DynamicTaxonomy'):
        """
        Merge with another dynamic taxonomy.
        
        Args:
            other: Another DynamicTaxonomy instance
        """
        for type_name, metadata in other.discovered_types.items():
            examples = other.type_examples[type_name]
            self.add_type(
                type_name,
                examples,
                metadata['confidence'],
                metadata.get('metadata')
            )


__all__ = [
    "DynamicTaxonomy",
]