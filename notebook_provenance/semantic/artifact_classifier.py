"""
Hybrid Artifact Classifier
===========================

Uses LLM + Embeddings + Heuristics for generalizable artifact classification.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import numpy as np

from notebook_provenance.core.data_structures import DFGNode
from notebook_provenance.core.enums import NodeType
from notebook_provenance.semantic.llm_analyzer import LLMSemanticAnalyzer


@dataclass
class ArtifactClassification:
    """Result of artifact classification."""
    category: str  # core_data, metadata, payload, config, display, utility
    importance: float  # 1-10
    confidence: float  # 0-1
    reasoning: str
    source: str  # 'pattern', 'embedding', 'llm'
    semantic_type: Optional[str] = None  # dataframe, table, model, etc.


@dataclass
class EmbeddingCache:
    """Cache for variable embeddings and classifications."""
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    classifications: Dict[str, ArtifactClassification] = field(default_factory=dict)
    
    def add(self, key: str, embedding: List[float], classification: ArtifactClassification):
        self.embeddings[key] = embedding
        self.classifications[key] = classification
    
    def get_similar(self, embedding: List[float], threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """Find most similar cached item."""
        if not self.embeddings:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for key, cached_emb in self.embeddings.items():
            similarity = self._cosine_similarity(embedding, cached_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = key
        
        if best_similarity >= threshold:
            return best_match, best_similarity
        return None
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-10))


class HybridArtifactClassifier:
    """
    Hybrid classifier using patterns, embeddings, and LLM reasoning.
    
    This provides generalizable artifact classification that:
    1. Uses fast patterns for obvious cases
    2. Uses embedding similarity for learned classifications
    3. Falls back to LLM for truly ambiguous cases
    4. Learns from classifications to improve over time
    """
    
    # High-confidence patterns (only very obvious ones)
    HIGH_CONFIDENCE_PATTERNS = {
        'core_data': {
            'exact': ['df', 'dataframe', 'dataset'],
            'suffix': ['_df', '_data', '_table', '_result'],
            'prefix': ['raw_', 'clean_', 'final_', 'merged_', 'joined_'],
        },
        'config': {
            'exact': ['url', 'uri', 'token', 'password', 'username', 'api_key'],
            'suffix': ['_url', '_uri', '_token', '_key', '_password', '_config'],
            'contains': ['manager', 'client', 'handler', 'service', 'auth'],
        },
        'display': {
            'exact': ['html', 'plot', 'figure', 'chart'],
            'suffix': ['_html', '_plot', '_fig', '_chart'],
            'prefix': ['html_', 'plot_', 'fig_'],
        },
    }
    
    # Category descriptions for LLM
    CATEGORY_DESCRIPTIONS = {
        'core_data': 'Main data artifacts: DataFrames, tables, datasets, transformation results',
        'metadata': 'Identifiers, names, keys: table_id, column_name, dataset_id',
        'payload': 'API request/response data: payloads, responses, request bodies',
        'config': 'Configuration and setup: URLs, credentials, managers, clients',
        'display': 'Visualization objects: HTML tables, plots, charts, figures',
        'utility': 'Helper objects: counters, flags, temporary variables',
    }
    
    def __init__(self, llm_analyzer: Optional[LLMSemanticAnalyzer] = None,
                 use_embeddings: bool = True):
        """
        Initialize hybrid classifier.
        
        Args:
            llm_analyzer: LLM analyzer for semantic reasoning
            use_embeddings: Whether to use embedding-based similarity
        """
        self.llm_analyzer = llm_analyzer
        self.use_embeddings = use_embeddings
        self.embedding_cache = EmbeddingCache()
        
        # Statistics
        self.stats = defaultdict(int)
    
    def classify(self, node: DFGNode, code_context: str = "",
                 function_calls: List[str] = None) -> ArtifactClassification:
        """
        Classify an artifact using hybrid approach.
        
        Args:
            node: DFG node to classify
            code_context: Surrounding code context
            function_calls: Related function calls
            
        Returns:
            ArtifactClassification result
        """
        variable_name = node.label
        function_calls = function_calls or []
        
        # Layer 1: Fast pattern matching (high confidence only)
        pattern_result = self._pattern_match(variable_name)
        if pattern_result and pattern_result.confidence >= 0.9:
            self.stats['pattern_match'] += 1
            return pattern_result
        
        # Layer 2: Embedding similarity (if enabled and we have cache)
        if self.use_embeddings and self.embedding_cache.embeddings:
            embedding_result = self._embedding_match(variable_name, code_context)
            if embedding_result and embedding_result.confidence >= 0.85:
                self.stats['embedding_match'] += 1
                return embedding_result
        
        # Layer 3: LLM semantic reasoning
        if self.llm_analyzer and self.llm_analyzer.enabled:
            llm_result = self._llm_classify(
                variable_name, code_context, function_calls, node
            )
            if llm_result:
                self.stats['llm_classify'] += 1
                
                # Cache for future similarity matching
                if self.use_embeddings:
                    self._cache_classification(variable_name, code_context, llm_result)
                
                return llm_result
        
        # Layer 4: Fallback to pattern with lower confidence
        self.stats['fallback'] += 1
        return pattern_result or ArtifactClassification(
            category='utility',
            importance=3.0,
            confidence=0.3,
            reasoning='No clear classification found',
            source='fallback'
        )
    
    def _pattern_match(self, name: str) -> Optional[ArtifactClassification]:
        """
        Fast pattern matching for obvious cases.
        """
        name_lower = name.lower()
        
        for category, patterns in self.HIGH_CONFIDENCE_PATTERNS.items():
            # Exact match
            if name_lower in patterns.get('exact', []):
                return ArtifactClassification(
                    category=category,
                    importance=8.0 if category == 'core_data' else 3.0,
                    confidence=0.95,
                    reasoning=f"Exact match for {category}",
                    source='pattern'
                )
            
            # Suffix match
            for suffix in patterns.get('suffix', []):
                if name_lower.endswith(suffix):
                    return ArtifactClassification(
                        category=category,
                        importance=7.0 if category == 'core_data' else 3.0,
                        confidence=0.9,
                        reasoning=f"Suffix '{suffix}' matches {category}",
                        source='pattern'
                    )
            
            # Prefix match
            for prefix in patterns.get('prefix', []):
                if name_lower.startswith(prefix):
                    return ArtifactClassification(
                        category=category,
                        importance=7.0 if category == 'core_data' else 3.0,
                        confidence=0.9,
                        reasoning=f"Prefix '{prefix}' matches {category}",
                        source='pattern'
                    )
            
            # Contains match
            for substr in patterns.get('contains', []):
                if substr in name_lower:
                    return ArtifactClassification(
                        category=category,
                        importance=4.0 if category == 'core_data' else 2.0,
                        confidence=0.75,
                        reasoning=f"Contains '{substr}' suggesting {category}",
                        source='pattern'
                    )
        
        # Check for common data patterns (lower confidence)
        data_hints = ['data', 'table', 'result', 'output', 'df', 'frame']
        if any(hint in name_lower for hint in data_hints):
            return ArtifactClassification(
                category='core_data',
                importance=6.0,
                confidence=0.7,
                reasoning=f"Name suggests data artifact",
                source='pattern'
            )
        
        return None
    
    def _embedding_match(self, name: str, context: str) -> Optional[ArtifactClassification]:
        """
        Find similar cached classification using embeddings.
        """
        # Generate embedding for this variable
        embedding = self._generate_embedding(name, context)
        if not embedding:
            return None
        
        # Find similar in cache
        match = self.embedding_cache.get_similar(embedding, threshold=0.85)
        if match:
            matched_key, similarity = match
            cached_class = self.embedding_cache.classifications[matched_key]
            
            return ArtifactClassification(
                category=cached_class.category,
                importance=cached_class.importance,
                confidence=similarity * cached_class.confidence,
                reasoning=f"Similar to '{matched_key}' (similarity: {similarity:.2f})",
                source='embedding',
                semantic_type=cached_class.semantic_type
            )
        
        return None
    
    def _generate_embedding(self, name: str, context: str) -> Optional[List[float]]:
        """
        Generate embedding for variable name + context.
        """
        if not self.llm_analyzer or not self.llm_analyzer.enabled:
            return None
        
        # Combine name and context for embedding
        text = f"Variable: {name}\nContext: {context[:200]}"
        
        try:
            # Use OpenAI embeddings if available
            response = self.llm_analyzer.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            # Fallback: use simple hash-based pseudo-embedding
            return self._simple_embedding(name, context)
    
    def _simple_embedding(self, name: str, context: str) -> List[float]:
        """
        Simple fallback embedding using character n-grams.
        Not as good as real embeddings but works offline.
        """
        text = f"{name} {context[:100]}".lower()
        
        # Character trigram counts
        trigrams = defaultdict(int)
        for i in range(len(text) - 2):
            trigrams[text[i:i+3]] += 1
        
        # Convert to fixed-size vector (simple hash)
        embedding = [0.0] * 128
        for trigram, count in trigrams.items():
            idx = hash(trigram) % 128
            embedding[idx] += count
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding
    
    def _llm_classify(self, name: str, context: str, 
                  function_calls: List[str], node: DFGNode) -> Optional[ArtifactClassification]:
        prompt = self._build_classification_prompt(name, context, function_calls, node)
        
        try:
            # Add explicit timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM classification timeout")
            
            # Set 10 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            
            try:
                response = self.llm_analyzer._call_llm(
                    prompt,
                    temperature=0.1,
                    max_tokens=300
                )
                
                # Cancel alarm
                signal.alarm(0)
                
                # Parse JSON response
                result = json.loads(response.strip())
                
                return ArtifactClassification(
                    category=result.get('category', 'utility'),
                    importance=float(result.get('importance', 5)),
                    confidence=float(result.get('confidence', 0.7)),
                    reasoning=result.get('reasoning', ''),
                    source='llm',
                    semantic_type=result.get('semantic_type')
                )
            except TimeoutError:
                print(f"  ⚠ LLM timeout for variable '{name}', using fallback")
                signal.alarm(0)  # Cancel alarm
                return None
            
        except Exception as e:
            print(f"  ⚠ LLM classification failed for '{name}': {e}")
            return None
    
    def _build_classification_prompt(self, name: str, context: str,
                                    function_calls: List[str], node: DFGNode) -> str:
        """Build prompt for LLM classification."""
        
        categories_desc = "\n".join(
            f"- {cat}: {desc}" 
            for cat, desc in self.CATEGORY_DESCRIPTIONS.items()
        )
        
        return f"""Analyze this variable from a data pipeline notebook and classify it.

VARIABLE NAME: {name}

CODE CONTEXT:
```python
{context[:500]}
```

RELATED FUNCTION CALLS: {', '.join(function_calls[:5]) if function_calls else 'None'}

CELL ID: {node.cell_id}

CATEGORIES:
{categories_desc}

SEMANTIC TYPES (for core_data):
- dataframe: Pandas/Polars DataFrame
- table: Database table or structured table data
- model: ML model or trained estimator
- result: Transformation result, reconciliation output
- matrix: Numpy array or tensor

Classify this variable and assess its importance for data lineage tracking.

Respond ONLY with JSON:
{{
    "category": "one of: core_data, metadata, payload, config, display, utility",
    "semantic_type": "if core_data: dataframe/table/model/result/matrix, else null",
    "importance": 1-10 (10 = critical for lineage, 1 = not important),
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
    
    def _cache_classification(self, name: str, context: str, 
                              classification: ArtifactClassification):
        """Cache classification for future similarity matching."""
        key = f"{name}_{hash(context[:100]) % 10000}"
        embedding = self._generate_embedding(name, context)
        
        if embedding:
            self.embedding_cache.add(key, embedding, classification)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        total = sum(self.stats.values())
        
        return {
            'total_classifications': total,
            'by_source': dict(self.stats),
            'cache_size': len(self.embedding_cache.embeddings),
            'percentages': {
                k: (v / total * 100 if total > 0 else 0)
                for k, v in self.stats.items()
            }
        }
    
    def save_cache(self, filepath: str):
        """Save embedding cache to file."""
        data = {
            'embeddings': self.embedding_cache.embeddings,
            'classifications': {
                k: {
                    'category': v.category,
                    'importance': v.importance,
                    'confidence': v.confidence,
                    'reasoning': v.reasoning,
                    'source': v.source,
                    'semantic_type': v.semantic_type
                }
                for k, v in self.embedding_cache.classifications.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_cache(self, filepath: str):
        """Load embedding cache from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.embedding_cache.embeddings = data.get('embeddings', {})
        self.embedding_cache.classifications = {
            k: ArtifactClassification(**v)
            for k, v in data.get('classifications', {}).items()
        }


__all__ = [
    "HybridArtifactClassifier",
    "ArtifactClassification",
    "EmbeddingCache",
]