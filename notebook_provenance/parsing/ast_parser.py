"""
AST Parser Module
=================

Advanced AST-based code cell parsing with context awareness.

This module provides the EnhancedCodeCellParser class which:
- Parses Python code using AST
- Extracts variables, functions, imports, and control structures
- Computes complexity scores
- Handles various Python syntax constructs
"""

import ast
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field

from notebook_provenance.core.config import ParsingConfig


@dataclass
class ParsedCell:
    """
    Result of parsing a code cell.
    
    Attributes:
        cell_id: Unique identifier for the cell
        code: Original code content
        variables_defined: List of variables defined in the cell
        variables_used: List of variables used in the cell
        variables_modified: List of variables modified (augmented assignment)
        function_calls: List of function calls
        function_definitions: List of function definitions
        imports: List of imports
        class_definitions: List of class definitions
        decorators: List of decorators used
        control_structures: List of control structures (For, While, If)
        string_literals: List of string literals
        numeric_literals: List of numeric literals
        ast_tree: The AST tree
        complexity_score: Computed complexity score
        error: Optional error message if parsing failed
    """
    cell_id: str
    code: str
    variables_defined: List[str] = field(default_factory=list)
    variables_used: List[str] = field(default_factory=list)
    variables_modified: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    function_definitions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    class_definitions: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    control_structures: List[str] = field(default_factory=list)
    string_literals: List[str] = field(default_factory=list)
    numeric_literals: List[float] = field(default_factory=list)
    ast_tree: Optional[ast.AST] = None
    complexity_score: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding AST tree)"""
        data = {
            'cell_id': self.cell_id,
            'code': self.code,
            'variables_defined': self.variables_defined,
            'variables_used': self.variables_used,
            'variables_modified': self.variables_modified,
            'function_calls': self.function_calls,
            'function_definitions': self.function_definitions,
            'imports': self.imports,
            'class_definitions': self.class_definitions,
            'decorators': self.decorators,
            'control_structures': self.control_structures,
            'complexity_score': self.complexity_score,
        }
        if self.error:
            data['error'] = self.error
        return data


class EnhancedCodeCellParser:
    """
    Advanced AST-based cell parser with context awareness.
    
    This parser extracts comprehensive metadata from Python code cells:
    - Variable definitions and usages
    - Function calls and definitions
    - Imports and class definitions
    - Control flow structures
    - Complexity metrics
    
    Example:
        >>> parser = EnhancedCodeCellParser()
        >>> result = parser.parse_cell("x = df.read_csv('data.csv')", "cell_0")
        >>> print(result.variables_defined)
        ['x']
        >>> print(result.function_calls)
        ['df.read_csv']
    """
    
    def __init__(self, config: Optional[ParsingConfig] = None):
        """
        Initialize the parser.
        
        Args:
            config: Optional parsing configuration
        """
        self.config = config or ParsingConfig()
        self.builtin_functions = set(dir(__builtins__))
    
    def parse_cell(self, code: str, cell_id: Optional[str] = None) -> ParsedCell:
        """
        Parse a single code cell with enhanced metadata extraction.
        
        Args:
            code: Source code to parse
            cell_id: Optional cell identifier
            
        Returns:
            ParsedCell object containing all extracted information
        """
        # Generate cell ID if not provided
        if cell_id is None:
            cell_id = f"cell_{hash(code) % 10000}"
        
        # Try to parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ParsedCell(
                cell_id=cell_id,
                code=code,
                error=f"Syntax error: {e}"
            )
        except Exception as e:
            return ParsedCell(
                cell_id=cell_id,
                code=code,
                error=f"Parse error: {e}"
            )
        
        # Initialize parsed cell
        parsed = ParsedCell(
            cell_id=cell_id,
            code=code,
            ast_tree=tree
        )
        
        # Walk AST and extract information
        for node in ast.walk(tree):
            self._process_node(node, parsed)
        
        # Calculate complexity
        parsed.complexity_score = self._calculate_complexity(parsed)
        
        # Deduplicate lists
        parsed.variables_used = list(set(parsed.variables_used))
        parsed.function_calls = list(set(parsed.function_calls))
        parsed.imports = list(set(parsed.imports))
        
        return parsed
    
    def parse_cells(self, cells: List[str], 
                    cell_ids: Optional[List[str]] = None) -> List[ParsedCell]:
        """
        Parse multiple code cells.
        
        Args:
            cells: List of code strings
            cell_ids: Optional list of cell identifiers
            
        Returns:
            List of ParsedCell objects
        """
        if cell_ids is None:
            cell_ids = [f"cell_{i}" for i in range(len(cells))]
        
        return [self.parse_cell(code, cell_id) 
                for code, cell_id in zip(cells, cell_ids)]
    
    def _process_node(self, node: ast.AST, parsed: ParsedCell) -> None:
        """
        Process individual AST node and extract information.
        
        Args:
            node: AST node to process
            parsed: ParsedCell to populate
        """
        # Variable assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    parsed.variables_defined.append(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            parsed.variables_defined.append(elt.id)
        
        # Augmented assignments (+=, -=, etc.)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                parsed.variables_modified.append(node.target.id)
        
        # Annotated assignments (x: int = 5)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                parsed.variables_defined.append(node.target.id)
        
        # Function calls
        elif isinstance(node, ast.Call):
            func_name = self._extract_function_name(node)
            if func_name and func_name not in self.builtin_functions:
                parsed.function_calls.append(func_name)
        
        # Function definitions
        elif isinstance(node, ast.FunctionDef):
            parsed.function_definitions.append(node.name)
            # Extract decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    parsed.decorators.append(decorator.id)
                elif isinstance(decorator, ast.Call):
                    dec_name = self._extract_function_name(decorator)
                    if dec_name:
                        parsed.decorators.append(dec_name)
        
        # Async function definitions
        elif isinstance(node, ast.AsyncFunctionDef):
            parsed.function_definitions.append(node.name)
        
        # Class definitions
        elif isinstance(node, ast.ClassDef):
            parsed.class_definitions.append(node.name)
        
        # Variable usage (loading)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            parsed.variables_used.append(node.id)
        
        # Imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parsed.imports.append(alias.name)
        
        # From imports
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                if module:
                    parsed.imports.append(f"{module}.{alias.name}")
                else:
                    parsed.imports.append(alias.name)
        
        # Control structures
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            parsed.control_structures.append('For')
        elif isinstance(node, ast.While):
            parsed.control_structures.append('While')
        elif isinstance(node, ast.If):
            parsed.control_structures.append('If')
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            parsed.control_structures.append('With')
        elif isinstance(node, ast.Try):
            parsed.control_structures.append('Try')
        
        # String literals
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str) and len(value) > 3:
                # Truncate long strings
                parsed.string_literals.append(value[:100])
            elif isinstance(value, (int, float)):
                parsed.numeric_literals.append(value)
        
        # Legacy nodes (Python < 3.8)
        elif isinstance(node, ast.Str):
            if len(node.s) > 3:
                parsed.string_literals.append(node.s[:100])
        elif isinstance(node, ast.Num):
            parsed.numeric_literals.append(node.n)
    
    def _extract_function_name(self, call_node: ast.Call) -> Optional[str]:
        """
        Extract full function name from Call node.
        
        Examples:
            - func() -> "func"
            - obj.method() -> "obj.method"
            - module.submodule.func() -> "module.submodule.func"
        
        Args:
            call_node: AST Call node
            
        Returns:
            Full function name or None
        """
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        
        elif isinstance(call_node.func, ast.Attribute):
            parts = []
            node = call_node.func
            
            # Walk up the attribute chain
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            
            # Add the base name
            if isinstance(node, ast.Name):
                parts.append(node.id)
            
            # Reverse to get correct order
            return '.'.join(reversed(parts))
        
        return None
    
    def _calculate_complexity(self, parsed: ParsedCell) -> float:
        """
        Calculate cell complexity score.
        
        The complexity is computed based on:
        - Number of variables defined (weight: 1)
        - Number of function calls (weight: 2)
        - Number of function definitions (weight: 3)
        - Number of class definitions (weight: 5)
        - Number of control structures (weight: 2)
        
        Args:
            parsed: ParsedCell to calculate complexity for
            
        Returns:
            Complexity score
        """
        score = 0.0
        score += len(parsed.variables_defined) * 1
        score += len(parsed.function_calls) * 2
        score += len(parsed.function_definitions) * 3
        score += len(parsed.class_definitions) * 5
        score += len(parsed.control_structures) * 2
        return score
    
    def get_statistics(self, parsed_cells: List[ParsedCell]) -> Dict[str, Any]:
        """
        Get statistics for a collection of parsed cells.
        
        Args:
            parsed_cells: List of ParsedCell objects
            
        Returns:
            Dictionary of statistics
        """
        total_cells = len(parsed_cells)
        valid_cells = [c for c in parsed_cells if c.error is None]
        error_cells = [c for c in parsed_cells if c.error is not None]
        
        if not valid_cells:
            return {
                'total_cells': total_cells,
                'valid_cells': 0,
                'error_cells': len(error_cells),
                'errors': [c.error for c in error_cells]
            }
        
        return {
            'total_cells': total_cells,
            'valid_cells': len(valid_cells),
            'error_cells': len(error_cells),
            'total_variables_defined': sum(len(c.variables_defined) for c in valid_cells),
            'total_function_calls': sum(len(c.function_calls) for c in valid_cells),
            'total_imports': sum(len(c.imports) for c in valid_cells),
            'avg_complexity': sum(c.complexity_score for c in valid_cells) / len(valid_cells),
            'max_complexity': max(c.complexity_score for c in valid_cells),
            'min_complexity': min(c.complexity_score for c in valid_cells),
        }


__all__ = [
    "EnhancedCodeCellParser",
    "ParsedCell",
]