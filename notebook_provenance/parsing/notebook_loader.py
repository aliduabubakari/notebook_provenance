"""
Notebook Loader Module
======================

Load and parse notebooks from various formats.

This module provides:
- NotebookLoader: Load notebooks from .ipynb and .py files
- Automatic format detection
- Smart cell splitting for Python files
"""

import re
import ast
from pathlib import Path
from typing import List, Tuple, Optional
import sys

try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False
    print("Warning: nbformat not installed. .ipynb support disabled.")
    print("Install with: pip install nbformat")


class NotebookLoader:
    """
    Load and parse notebooks from various formats.
    
    Supports:
    - Jupyter notebooks (.ipynb)
    - Python files (.py) with various cell splitting strategies
    
    Example:
        >>> loader = NotebookLoader()
        >>> code_cells, cell_ids = loader.load_notebook("analysis.ipynb")
        >>> print(f"Loaded {len(code_cells)} cells")
    """
    
    @staticmethod
    def load_notebook(file_path: str) -> Tuple[List[str], List[str]]:
        """
        Auto-detect format and load notebook.
        
        Args:
            file_path: Path to notebook file
            
        Returns:
            Tuple of (code_cells, cell_ids)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.ipynb':
            return NotebookLoader.load_ipynb(file_path)
        elif suffix == '.py':
            return NotebookLoader.load_py(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .ipynb, .py"
            )
    
    @staticmethod
    def load_ipynb(file_path: str) -> Tuple[List[str], List[str]]:
        """
        Load Jupyter notebook (.ipynb) and extract code cells.
        
        Args:
            file_path: Path to .ipynb file
            
        Returns:
            Tuple of (code_cells, cell_ids)
            
        Raises:
            ImportError: If nbformat is not installed
            Exception: If loading fails
        """
        if not NBFORMAT_AVAILABLE:
            raise ImportError(
                "nbformat is required to load .ipynb files. "
                "Install with: pip install nbformat"
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            code_cells = []
            cell_ids = []
            
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    code_cells.append(cell.source)
                    # Use cell ID if available, otherwise generate
                    cell_id = cell.get('id', f"cell_{i}")
                    cell_ids.append(cell_id)
            
            print(f"✓ Loaded {len(code_cells)} code cells from {file_path}")
            return code_cells, cell_ids
            
        except Exception as e:
            raise Exception(f"Error loading .ipynb file: {e}")
    
    @staticmethod
    def load_py(file_path: str) -> Tuple[List[str], List[str]]:
        """
        Load Python file (.py) and split into cells.
        
        Supports multiple splitting strategies:
        1. Cell markers (# %%, #%%)
        2. Function/class definitions
        3. Blank lines
        
        Args:
            file_path: Path to .py file
            
        Returns:
            Tuple of (code_cells, cell_ids)
            
        Raises:
            Exception: If loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Strategy 1: Split by cell markers
            if '# %%' in content or '#%%' in content:
                return NotebookLoader._split_by_cell_markers(content)
            
            # Strategy 2: Split by function definitions
            elif 'def ' in content or 'class ' in content:
                return NotebookLoader._split_by_functions(content)
            
            # Strategy 3: Split by blank lines
            else:
                return NotebookLoader._split_by_blank_lines(content)
                
        except Exception as e:
            raise Exception(f"Error loading .py file: {e}")
    
    @staticmethod
    def _split_by_cell_markers(content: str) -> Tuple[List[str], List[str]]:
        """
        Split Python file by cell markers (VS Code/PyCharm style).
        
        Recognizes patterns like:
        - # %%
        - #%%
        - # %% [markdown]
        - # %% cell_name
        
        Args:
            content: File content
            
        Returns:
            Tuple of (code_cells, cell_ids)
        """
        # Pattern: # %%, #%%, # %% [markdown], etc.
        cell_pattern = re.compile(r'^#\s*%%(?:\s*\[.*?\])?\s*.*$', re.MULTILINE)
        
        # Find all cell boundaries
        matches = list(cell_pattern.finditer(content))
        
        if not matches:
            # No markers found, return as single cell
            return [content], ["cell_0"]
        
        code_cells = []
        cell_ids = []
        
        # Extract cells between markers
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            
            cell_content = content[start:end].strip()
            
            if cell_content:  # Only add non-empty cells
                code_cells.append(cell_content)
                
                # Try to extract cell name from marker
                marker_text = match.group(0)
                cell_name_match = re.search(r'%%\s+(.+)', marker_text)
                if cell_name_match:
                    cell_name = cell_name_match.group(1).strip()
                    # Sanitize cell name
                    cell_name = re.sub(r'[^\w\s-]', '', cell_name)[:20]
                    cell_ids.append(f"cell_{i}_{cell_name}")
                else:
                    cell_ids.append(f"cell_{i}")
        
        print(f"✓ Split into {len(code_cells)} cells using cell markers")
        return code_cells, cell_ids
    
    @staticmethod
    def _split_by_functions(content: str) -> Tuple[List[str], List[str]]:
        """
        Split Python file by function/class definitions.
        
        Args:
            content: File content
            
        Returns:
            Tuple of (code_cells, cell_ids)
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If parse fails, fall back to blank line splitting
            return NotebookLoader._split_by_blank_lines(content)
        
        lines = content.split('\n')
        code_cells = []
        cell_ids = []
        
        # Collect imports first
        imports = []
        other_statements = []
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                imports.extend(lines[start:end])
            else:
                other_statements.append(node)
        
        # Add imports as first cell
        if imports:
            code_cells.append('\n'.join(imports))
            cell_ids.append("cell_0_imports")
        
        # Add each function/class as separate cell
        for i, node in enumerate(other_statements):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            
            cell_content = '\n'.join(lines[start:end])
            
            if cell_content.strip():
                code_cells.append(cell_content)
                
                # Generate descriptive cell ID
                if isinstance(node, ast.FunctionDef):
                    cell_ids.append(f"cell_{len(cell_ids)}_func_{node.name}")
                elif isinstance(node, ast.ClassDef):
                    cell_ids.append(f"cell_{len(cell_ids)}_class_{node.name}")
                else:
                    cell_ids.append(f"cell_{len(cell_ids)}")
        
        print(f"✓ Split into {len(code_cells)} cells by functions/classes")
        return code_cells, cell_ids
    
    @staticmethod
    def _split_by_blank_lines(content: str, min_lines: int = 3) -> Tuple[List[str], List[str]]:
        """
        Split Python file by blank lines (simple heuristic).
        
        Args:
            content: File content
            min_lines: Minimum lines to form a cell
            
        Returns:
            Tuple of (code_cells, cell_ids)
        """
        lines = content.split('\n')
        code_cells = []
        cell_ids = []
        
        current_cell = []
        
        for line in lines:
            if line.strip():  # Non-empty line
                current_cell.append(line)
            else:  # Empty line
                # If we have accumulated enough lines, create a cell
                if len(current_cell) >= min_lines:
                    code_cells.append('\n'.join(current_cell))
                    cell_ids.append(f"cell_{len(cell_ids)}")
                    current_cell = []
                elif current_cell:
                    # Add empty line to current cell if we have content
                    current_cell.append(line)
        
        # Add remaining lines as last cell
        if current_cell:
            code_cells.append('\n'.join(current_cell))
            cell_ids.append(f"cell_{len(cell_ids)}")
        
        # If no cells created, add everything as one cell
        if not code_cells:
            code_cells = [content]
            cell_ids = ["cell_0"]
        
        print(f"✓ Split into {len(code_cells)} cells by blank lines")
        return code_cells, cell_ids
    
    @staticmethod
    def detect_format(file_path: str) -> str:
        """
        Detect notebook format from file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Format string: 'ipynb', 'py', or 'unknown'
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.ipynb':
            return 'ipynb'
        elif suffix == '.py':
            return 'py'
        else:
            return 'unknown'


__all__ = [
    "NotebookLoader",
]