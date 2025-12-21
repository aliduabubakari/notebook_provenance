"""
DFG Builder Module
==================

Build data flow graphs with intelligent filtering and artifact detection.

This module provides the SmartDFGBuilder class which:
- Constructs data flow graphs from parsed cells
- Tracks variable scoping and dependencies
- Identifies data artifacts
- Creates cleaned graphs with noise filtering
"""

import ast
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

from notebook_provenance.core.data_structures import (
    DataFlowGraph,
    DFGNode,
    DFGEdge,
    CellDependency,
)
from notebook_provenance.core.enums import NodeType, EdgeType
from notebook_provenance.core.config import GraphConfig
from notebook_provenance.parsing.ast_parser import ParsedCell


class SmartDFGBuilder:
    """
    Build data flow graphs with intelligent filtering and artifact detection.
    
    This class constructs a complete data flow graph from parsed cells,
    tracking variable definitions, usages, and dependencies between cells.
    
    Example:
        >>> builder = SmartDFGBuilder()
        >>> raw_dfg, dependencies = builder.build_dfg_from_cells(parsed_cells)
        >>> clean_dfg = builder.build_clean_dfg(raw_dfg)
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        """
        Initialize the DFG builder.
        
        Args:
            config: Optional graph configuration
        """
        self.config = config or GraphConfig()
        self.dfg = DataFlowGraph()
        self.variable_to_node = {}
        self.variable_to_cell = {}
        self.cell_dependencies = {}
        self.node_counter = 0
        self.global_scope = {}
        
        # Patterns for identifying important elements
        self.data_artifact_patterns = {
            'dataframe': ['df', 'data', 'dataset', '_data', '_df'],
            'table': ['table', 'tbl', '_table'],
            'model': ['model', 'classifier', 'regressor', 'estimator'],
            'result': ['result', 'output', 'prediction', 'predictions'],
            'matrix': ['matrix', 'array', 'tensor'],
        }
        
        # Patterns to filter as noise
        self.noise_patterns = [
            'manager', 'client', 'config', 'token', 'api_key',
            'username', 'password', 'url', 'connection', 'session',
            'auth', 'credentials', 'secret', 'key'
        ]
    
    def build_dfg_from_cells(self, parsed_cells: List[ParsedCell]) -> Tuple[DataFlowGraph, Dict[str, CellDependency]]:
        """
        Build complete DFG with dependencies from parsed cells.
        
        Args:
            parsed_cells: List of ParsedCell objects
            
        Returns:
            Tuple of (DataFlowGraph, cell_dependencies dict)
        """
        # Reset state
        self.dfg = DataFlowGraph()
        self.variable_to_node = {}
        self.variable_to_cell = {}
        self.cell_dependencies = {}
        self.node_counter = 0
        self.global_scope = {}
        
        # First pass: identify cell dependencies
        for cell in parsed_cells:
            if cell.error:
                continue
            
            cell_id = cell.cell_id
            self.cell_dependencies[cell_id] = CellDependency(cell_id=cell_id)
            
            # Track what this cell produces
            for var in cell.variables_defined:
                self.cell_dependencies[cell_id].produces.add(var)
                self.variable_to_cell[var] = cell_id
            
            # Track imports
            self.cell_dependencies[cell_id].imports.update(cell.imports)
        
        # Second pass: identify what each cell consumes
        for cell in parsed_cells:
            if cell.error:
                continue
            
            cell_id = cell.cell_id
            
            for var in cell.variables_used:
                self.cell_dependencies[cell_id].consumes.add(var)
                
                # Track dependencies on other cells
                if var in self.variable_to_cell:
                    producer_cell = self.variable_to_cell[var]
                    if producer_cell != cell_id:
                        self.cell_dependencies[cell_id].depends_on.add(producer_cell)
        
        # Third pass: build the graph
        for cell in parsed_cells:
            if cell.error:
                continue
            self._process_cell(cell)
        
        return self.dfg, self.cell_dependencies
    
    def _process_cell(self, cell: ParsedCell) -> None:
        """
        Process a cell and build graph nodes/edges.
        
        Args:
            cell: ParsedCell to process
        """
        cell_id = cell.cell_id
        cell_scope = {}
        
        # Process each statement in the AST
        if cell.ast_tree:
            for stmt in cell.ast_tree.body:
                if isinstance(stmt, ast.Assign):
                    self._process_assignment(stmt, cell_id, cell.code, cell_scope)
                elif isinstance(stmt, ast.AugAssign):
                    self._process_aug_assignment(stmt, cell_id, cell.code, cell_scope)
                elif isinstance(stmt, ast.AnnAssign):
                    self._process_ann_assignment(stmt, cell_id, cell.code, cell_scope)
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    self._process_standalone_call(stmt.value, cell_id, cell.code, cell_scope)
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    self._process_import(stmt, cell_id, cell_scope)
        
        # Update global scope
        self.global_scope.update(cell_scope)
    
    def _process_assignment(self, assign_node: ast.Assign, cell_id: str, 
                       code: str, cell_scope: Dict) -> None:
        """
        Process assignment WITH function tracking.
        """
        for target in assign_node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                
                # Process right-hand side and track if it's a function call
                value_node = self._process_expression(
                    assign_node.value, cell_id, code, cell_scope
                )
                
                # Check if value is a function call
                creating_function = None
                if isinstance(assign_node.value, ast.Call):
                    func_name = self._extract_function_name(assign_node.value)
                    creating_function = func_name
                
                # Create variable node WITH metadata
                var_node = self._create_variable_node(
                    var_name, cell_id, code, 
                    getattr(assign_node, 'lineno', 0)
                )
                
                # ADD: Store creating function in metadata
                if creating_function:
                    var_node.metadata['created_by'] = creating_function
                
                # Add edge from value to variable
                if value_node and value_node.id != var_node.id:
                    edge = DFGEdge(
                        from_node=value_node.id,
                        to_node=var_node.id,
                        edge_type=EdgeType.DATA_FLOW,
                        operation=creating_function or "assign",
                        metadata={'creating_function': creating_function} if creating_function else {}
                    )
                    self.dfg.add_edge(edge)
                
                # Update scopes
                cell_scope[var_name] = var_node
                self.variable_to_node[var_name] = var_node

    def _process_aug_assignment(self, aug_assign_node: ast.AugAssign, 
                               cell_id: str, code: str, cell_scope: Dict) -> None:
        """
        Process augmented assignment (+=, -=, etc.).
        
        Args:
            aug_assign_node: AST AugAssign node
            cell_id: Current cell ID
            code: Source code
            cell_scope: Current cell's scope
        """
        if isinstance(aug_assign_node.target, ast.Name):
            var_name = aug_assign_node.target.id
            
            # Get existing variable node or create new one
            if var_name in cell_scope:
                var_node = cell_scope[var_name]
            elif var_name in self.variable_to_node:
                var_node = self.variable_to_node[var_name]
            else:
                var_node = self._create_variable_node(
                    var_name, cell_id, code,
                    getattr(aug_assign_node, 'lineno', 0)
                )
            
            # Process value
            value_node = self._process_expression(
                aug_assign_node.value, cell_id, code, cell_scope
            )
            
            # Create intermediate node for the operation
            op_name = aug_assign_node.op.__class__.__name__
            intermediate = DFGNode(
                id=f"intermediate_{self.node_counter}",
                label=f"{var_name} {op_name}= ...",
                node_type=NodeType.INTERMEDIATE,
                code_snippet=f"{var_name} {op_name}= ...",
                line_number=getattr(aug_assign_node, 'lineno', 0),
                cell_id=cell_id
            )
            self.node_counter += 1
            self.dfg.add_node(intermediate)
            
            # Add edges: var -> intermediate, value -> intermediate
            if value_node:
                self.dfg.add_edge(DFGEdge(
                    from_node=var_node.id,
                    to_node=intermediate.id,
                    edge_type=EdgeType.DATA_FLOW
                ))
                self.dfg.add_edge(DFGEdge(
                    from_node=value_node.id,
                    to_node=intermediate.id,
                    edge_type=EdgeType.DATA_FLOW
                ))
            
            # Update variable to point to intermediate result
            self.dfg.add_edge(DFGEdge(
                from_node=intermediate.id,
                to_node=var_node.id,
                edge_type=EdgeType.DATA_FLOW
            ))
    
    def _process_ann_assignment(self, ann_assign_node: ast.AnnAssign,
                               cell_id: str, code: str, cell_scope: Dict) -> None:
        """
        Process annotated assignment (x: int = 5).
        
        Args:
            ann_assign_node: AST AnnAssign node
            cell_id: Current cell ID
            code: Source code
            cell_scope: Current cell's scope
        """
        if isinstance(ann_assign_node.target, ast.Name):
            var_name = ann_assign_node.target.id
            
            if ann_assign_node.value:
                # Process value
                value_node = self._process_expression(
                    ann_assign_node.value, cell_id, code, cell_scope
                )
                
                # Create variable node
                var_node = self._create_variable_node(
                    var_name, cell_id, code,
                    getattr(ann_assign_node, 'lineno', 0)
                )
                
                # Add edge
                if value_node:
                    self.dfg.add_edge(DFGEdge(
                        from_node=value_node.id,
                        to_node=var_node.id,
                        edge_type=EdgeType.DATA_FLOW
                    ))
                
                # Update scopes
                cell_scope[var_name] = var_node
                self.variable_to_node[var_name] = var_node
    
    def _process_tuple_assignment(self, target_tuple: ast.AST, value: ast.AST,
                                  cell_id: str, code: str, cell_scope: Dict) -> None:
        """
        Handle tuple unpacking (a, b = func()).
        
        Args:
            target_tuple: Tuple/List AST node
            value: Right-hand side AST node
            cell_id: Current cell ID
            code: Source code
            cell_scope: Current cell's scope
        """
        value_node = self._process_expression(value, cell_id, code, cell_scope)
        
        elements = target_tuple.elts if hasattr(target_tuple, 'elts') else []
        
        for i, target in enumerate(elements):
            if isinstance(target, ast.Name):
                var_name = target.id
                
                # Create intermediate node for tuple element
                intermediate = DFGNode(
                    id=f"intermediate_{self.node_counter}",
                    label=f"{var_name} (from tuple[{i}])",
                    node_type=NodeType.INTERMEDIATE,
                    code_snippet=f"{var_name} = ...[{i}]",
                    line_number=getattr(target, 'lineno', 0),
                    cell_id=cell_id
                )
                self.node_counter += 1
                self.dfg.add_node(intermediate)
                
                # Add edge from value to intermediate
                if value_node:
                    self.dfg.add_edge(DFGEdge(
                        from_node=value_node.id,
                        to_node=intermediate.id,
                        edge_type=EdgeType.DATA_FLOW,
                        operation=f"unpack[{i}]"
                    ))
                
                # Create variable node
                var_node = self._create_variable_node(
                    var_name, cell_id, code, getattr(target, 'lineno', 0)
                )
                
                # Add edge from intermediate to variable
                self.dfg.add_edge(DFGEdge(
                    from_node=intermediate.id,
                    to_node=var_node.id,
                    edge_type=EdgeType.DATA_FLOW
                ))
                
                # Update scopes
                cell_scope[var_name] = var_node
                self.variable_to_node[var_name] = var_node
    
    def _process_expression(self, expr_node: ast.AST, cell_id: str, 
                           code: str, cell_scope: Dict) -> Optional[DFGNode]:
        """
        Process an expression node and return the resulting node.
        
        Args:
            expr_node: AST expression node
            cell_id: Current cell ID
            code: Source code
            cell_scope: Current cell's scope
            
        Returns:
            DFGNode or None
        """
        if isinstance(expr_node, ast.Call):
            return self._process_function_call(expr_node, cell_id, code, cell_scope)
        
        elif isinstance(expr_node, ast.Name):
            var_name = expr_node.id
            # Look up in cell scope, then global scope
            return cell_scope.get(var_name, self.variable_to_node.get(var_name))
        
        elif isinstance(expr_node, ast.Attribute):
            return self._process_attribute(expr_node, cell_id, code, cell_scope)
        
        elif isinstance(expr_node, (ast.Constant, ast.Str, ast.Num)):
            if self.config.include_literals:
                value = getattr(expr_node, 'value', 
                              getattr(expr_node, 's', 
                              getattr(expr_node, 'n', 'unknown')))
                return self._create_literal_node(str(value), cell_id)
            return None
        
        elif isinstance(expr_node, (ast.List, ast.Tuple, ast.Set)):
            # Process collection
            return self._process_collection(expr_node, cell_id, code, cell_scope)
        
        elif isinstance(expr_node, ast.Dict):
            return self._process_dict(expr_node, cell_id, code, cell_scope)
        
        elif isinstance(expr_node, ast.BinOp):
            return self._process_binop(expr_node, cell_id, code, cell_scope)
        
        return None
    
    def _process_function_call(self, call_node: ast.Call, cell_id: str, 
                              code: str, cell_scope: Dict) -> DFGNode:
        """
        Process function call and create node with edges.
        
        Args:
            call_node: AST Call node
            cell_id: Current cell ID
            code: Source code
            cell_scope: Current cell's scope
            
        Returns:
            DFGNode for the function call
        """
        func_name = self._extract_function_name(call_node)
        
        # Get code snippet
        code_snippet = (ast.unparse(call_node)[:100] 
                       if hasattr(ast, 'unparse') else func_name)
        
        # Create function call node
        func_node = DFGNode(
            id=f"func_{self.node_counter}",
            label=func_name,
            node_type=NodeType.FUNCTION_CALL,
            code_snippet=code_snippet,
            line_number=getattr(call_node, 'lineno', 0),
            cell_id=cell_id
        )
        self.node_counter += 1
        self.dfg.add_node(func_node)
        
        # Process arguments
        for arg in call_node.args:
            arg_node = self._process_expression(arg, cell_id, code, cell_scope)
            if arg_node:
                self.dfg.add_edge(DFGEdge(
                    from_node=arg_node.id,
                    to_node=func_node.id,
                    edge_type=EdgeType.INPUT,
                    operation="argument"
                ))
        
        # Process keyword arguments
        for keyword in call_node.keywords:
            arg_node = self._process_expression(keyword.value, cell_id, code, cell_scope)
            if arg_node:
                self.dfg.add_edge(DFGEdge(
                    from_node=arg_node.id,
                    to_node=func_node.id,
                    edge_type=EdgeType.INPUT,
                    operation=f"kwarg_{keyword.arg or 'kwargs'}"
                ))
        
        return func_node
    
    def _process_attribute(self, attr_node: ast.Attribute, cell_id: str,
                          code: str, cell_scope: Dict) -> Optional[DFGNode]:
        """
        Process attribute access (obj.attr).
        
        Args:
            attr_node: AST Attribute node
            cell_id: Current cell ID
            code: Source code
            cell_scope: Current cell's scope
            
        Returns:
            DFGNode or None
        """
        # Process base object
        base_node = self._process_expression(attr_node.value, cell_id, code, cell_scope)
        
        attr_name = attr_node.attr
        full_name = f"{getattr(base_node, 'label', 'obj')}.{attr_name}"
        
        # Create attribute access node
        attr_access_node = DFGNode(
            id=f"attr_{self.node_counter}",
            label=full_name,
            node_type=NodeType.ATTRIBUTE_ACCESS,
            code_snippet=full_name,
            line_number=getattr(attr_node, 'lineno', 0),
            cell_id=cell_id
        )
        self.node_counter += 1
        self.dfg.add_node(attr_access_node)
        
        # Add edge from base to attribute
        if base_node:
            self.dfg.add_edge(DFGEdge(
                from_node=base_node.id,
                to_node=attr_access_node.id,
                edge_type=EdgeType.DATA_FLOW,
                operation="attribute_access"
            ))
        
        return attr_access_node
    
    def _process_collection(self, coll_node: ast.AST, cell_id: str,
                           code: str, cell_scope: Dict) -> Optional[DFGNode]:
        """Process list, tuple, or set literal."""
        if not self.config.include_literals:
            return None
        
        # Create collection node
        coll_type = type(coll_node).__name__
        coll_node_dfg = DFGNode(
            id=f"literal_{self.node_counter}",
            label=f"{coll_type}[...]",
            node_type=NodeType.LITERAL,
            code_snippet=f"{coll_type}[...]",
            line_number=getattr(coll_node, 'lineno', 0),
            cell_id=cell_id
        )
        self.node_counter += 1
        self.dfg.add_node(coll_node_dfg)
        
        return coll_node_dfg
    
    def _process_dict(self, dict_node: ast.Dict, cell_id: str,
                     code: str, cell_scope: Dict) -> Optional[DFGNode]:
        """Process dictionary literal."""
        if not self.config.include_literals:
            return None
        
        dict_node_dfg = DFGNode(
            id=f"literal_{self.node_counter}",
            label="dict{...}",
            node_type=NodeType.LITERAL,
            code_snippet="dict{...}",
            line_number=getattr(dict_node, 'lineno', 0),
            cell_id=cell_id
        )
        self.node_counter += 1
        self.dfg.add_node(dict_node_dfg)
        
        return dict_node_dfg
    
    def _process_binop(self, binop_node: ast.BinOp, cell_id: str,
                      code: str, cell_scope: Dict) -> Optional[DFGNode]:
        """Process binary operation."""
        # Create node for the operation
        op_name = type(binop_node.op).__name__
        op_node = DFGNode(
            id=f"intermediate_{self.node_counter}",
            label=f"BinOp({op_name})",
            node_type=NodeType.INTERMEDIATE,
            code_snippet=f"... {op_name} ...",
            line_number=getattr(binop_node, 'lineno', 0),
            cell_id=cell_id
        )
        self.node_counter += 1
        self.dfg.add_node(op_node)
        
        # Process operands
        left_node = self._process_expression(binop_node.left, cell_id, code, cell_scope)
        right_node = self._process_expression(binop_node.right, cell_id, code, cell_scope)
        
        if left_node:
            self.dfg.add_edge(DFGEdge(
                from_node=left_node.id,
                to_node=op_node.id,
                edge_type=EdgeType.DATA_FLOW
            ))
        if right_node:
            self.dfg.add_edge(DFGEdge(
                from_node=right_node.id,
                to_node=op_node.id,
                edge_type=EdgeType.DATA_FLOW
            ))
        
        return op_node
    
    def _process_import(self, import_node: ast.AST, cell_id: str, cell_scope: Dict) -> None:
        """
        Process import statement.
        
        Args:
            import_node: AST Import or ImportFrom node
            cell_id: Current cell ID
            cell_scope: Current cell's scope
        """
        if not self.config.include_imports:
            return
        
        if isinstance(import_node, ast.Import):
            for alias in import_node.names:
                module_name = alias.name
                as_name = alias.asname or alias.name
                
                import_node_dfg = DFGNode(
                    id=f"import_{self.node_counter}",
                    label=f"import {module_name}",
                    node_type=NodeType.IMPORT,
                    code_snippet=f"import {module_name}",
                    line_number=getattr(import_node, 'lineno', 0),
                    cell_id=cell_id
                )
                self.node_counter += 1
                self.dfg.add_node(import_node_dfg)
                
                cell_scope[as_name] = import_node_dfg
                self.variable_to_node[as_name] = import_node_dfg
        
        elif isinstance(import_node, ast.ImportFrom):
            module = import_node.module or ''
            for alias in import_node.names:
                name = alias.name
                as_name = alias.asname or name
                
                import_node_dfg = DFGNode(
                    id=f"import_{self.node_counter}",
                    label=f"from {module} import {name}",
                    node_type=NodeType.IMPORT,
                    code_snippet=f"from {module} import {name}",
                    line_number=getattr(import_node, 'lineno', 0),
                    cell_id=cell_id
                )
                self.node_counter += 1
                self.dfg.add_node(import_node_dfg)
                
                cell_scope[as_name] = import_node_dfg
                self.variable_to_node[as_name] = import_node_dfg
    
    def _process_standalone_call(self, call_node: ast.Call, cell_id: str, 
                                code: str, cell_scope: Dict) -> None:
        """Process standalone function call (not assigned to variable)."""
        self._process_function_call(call_node, cell_id, code, cell_scope)
    
    def _create_variable_node(self, var_name: str, cell_id: str, 
                             code: str, line_number: int) -> DFGNode:
        """
        Create a variable node.
        
        Args:
            var_name: Variable name
            cell_id: Current cell ID
            code: Source code
            line_number: Line number
            
        Returns:
            DFGNode for the variable
        """
        # Check if this is a data artifact
        is_artifact = any(
            pattern in var_name.lower()
            for patterns in self.data_artifact_patterns.values()
            for pattern in patterns
        )
        
        node_type = NodeType.DATA_ARTIFACT if is_artifact else NodeType.VARIABLE
        
        var_node = DFGNode(
            id=f"var_{var_name}_{self.node_counter}",
            label=var_name,
            node_type=node_type,
            code_snippet=var_name,
            line_number=line_number,
            cell_id=cell_id
        )
        self.node_counter += 1
        self.dfg.add_node(var_node)
        return var_node
    
    def _create_literal_node(self, value: str, cell_id: str) -> DFGNode:
        """Create a literal node."""
        literal_node = DFGNode(
            id=f"lit_{self.node_counter}",
            label=value[:50],
            node_type=NodeType.LITERAL,
            code_snippet=value,
            line_number=0,
            cell_id=cell_id
        )
        self.node_counter += 1
        self.dfg.add_node(literal_node)
        return literal_node
    
    def _extract_function_name(self, call_node: ast.Call) -> str:
        """Extract function name from Call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            parts = []
            node = call_node.func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return '.'.join(reversed(parts))
        return "unknown_function"
    
    def should_include_node(self, node: DFGNode) -> bool:
        """
        Decide if node should be in cleaned graph.
        
        Args:
            node: DFGNode to check
            
        Returns:
            True if node should be included
        """
        # Always exclude noise patterns
        if any(pattern in node.label.lower() for pattern in self.noise_patterns):
            return False
        
        # Always include data artifacts
        if node.node_type == NodeType.DATA_ARTIFACT:
            return True
        
        # Exclude most literals
        if node.node_type == NodeType.LITERAL:
            if not self.config.include_literals:
                return False
            # Include only paths, URLs, etc.
            if any(pattern in node.label.lower() 
                  for pattern in ['.csv', '.json', '.parquet', 'http://', 'https://', 'api']):
                return True
            return False
        
        # Exclude imports unless configured
        if node.node_type == NodeType.IMPORT:
            return self.config.include_imports
        
        # Include important function calls
        if node.node_type == NodeType.FUNCTION_CALL:
            important_functions = [
                'read', 'load', 'reconcile', 'extend', 'transform',
                'merge', 'join', 'filter', 'aggregate', 'add_table',
                'get_table', 'query', 'execute', 'fit', 'predict',
                'train', 'evaluate', 'save', 'to_csv', 'to_parquet'
            ]
            if any(func in node.label.lower() for func in important_functions):
                return True
            return False
        
        # Include variables that aren't configuration
        if node.node_type == NodeType.VARIABLE:
            return True
        
        # Include intermediates and attributes
        if node.node_type in [NodeType.INTERMEDIATE, NodeType.ATTRIBUTE_ACCESS]:
            return True
        
        return True
    
    def build_clean_dfg(self, raw_dfg: DataFlowGraph) -> DataFlowGraph:
        """
        Create cleaned DFG by filtering noise.
        
        Args:
            raw_dfg: Raw data flow graph
            
        Returns:
            Cleaned DataFlowGraph
        """
        clean_dfg = DataFlowGraph()
        clean_dfg.metadata = raw_dfg.metadata.copy()
        
        # Add filtered nodes
        for node_id, node in raw_dfg.nodes.items():
            if self.should_include_node(node):
                clean_dfg.add_node(node)
        
        # Add edges between included nodes
        for edge in raw_dfg.edges:
            if (edge.from_node in clean_dfg.nodes and 
                edge.to_node in clean_dfg.nodes):
                clean_dfg.add_edge(edge)
        
        return clean_dfg
    
    def _process_cell(self, cell: 'ParsedCell') -> None:
        """
        Process a cell and build graph nodes/edges.
        
        FIXED: Better tracking of function calls and their results.
        """
        cell_id = cell.cell_id
        cell_scope = {}
        
        # Track function call results for this cell
        self._current_cell_function_results = {}
        
        if cell.ast_tree:
            for stmt in cell.ast_tree.body:
                self._process_statement(stmt, cell_id, cell.code, cell_scope)
        
        # Update global scope
        self.global_scope.update(cell_scope)

    def _process_statement(self, stmt, cell_id: str, code: str, cell_scope: Dict) -> None:
        """Process a single statement."""
        import ast
        
        if isinstance(stmt, ast.Assign):
            self._process_assignment(stmt, cell_id, code, cell_scope)
        elif isinstance(stmt, ast.AugAssign):
            self._process_aug_assignment(stmt, cell_id, code, cell_scope)
        elif isinstance(stmt, ast.AnnAssign):
            self._process_ann_assignment(stmt, cell_id, code, cell_scope)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._process_standalone_call(stmt.value, cell_id, code, cell_scope)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            self._process_import(stmt, cell_id, cell_scope)
        elif isinstance(stmt, ast.Try):
            # Process try block contents
            for try_stmt in stmt.body:
                self._process_statement(try_stmt, cell_id, code, cell_scope)
        elif isinstance(stmt, (ast.For, ast.While)):
            # Process loop body
            for loop_stmt in stmt.body:
                self._process_statement(loop_stmt, cell_id, code, cell_scope)
        elif isinstance(stmt, ast.If):
            # Process if body
            for if_stmt in stmt.body:
                self._process_statement(if_stmt, cell_id, code, cell_scope)

    def should_include_node(self, node: 'DFGNode') -> bool:
        """
        Decide if node should be in cleaned graph.
        
        FIXED: More aggressive filtering of configuration noise.
        """
        label_lower = node.label.lower()
        
        # Expanded noise patterns - exclude configuration/setup
        noise_patterns = [
            # Authentication/credentials
            'manager', 'client', 'handler', 'service', 'provider',
            'auth', 'token', 'password', 'username', 'credential', 'api_key',
            # Configuration
            'config', 'setting', 'option', 'url', 'uri', 'endpoint', 'base_url', 'api_url',
            # User input
            'input', 'prompt', 'default', 'getpass',
            # Utilities
            'utility', 'util', 'helper',
            # Messages
            'message', 'success', 'error', 'warning',
        ]
        
        # Check noise patterns
        for pattern in noise_patterns:
            if pattern in label_lower:
                # Exception: keep if it's clearly data (table_data, etc.)
                if any(data_word in label_lower for data_word in ['data', 'table', 'df', 'result', 'payload']):
                    if 'table_data' in label_lower or 'reconciled' in label_lower or 'extended' in label_lower:
                        return True
                return False
        
        # Always include data artifacts
        if node.node_type == NodeType.DATA_ARTIFACT:
            return True
        
        # Include important function calls
        if node.node_type == NodeType.FUNCTION_CALL:
            important_functions = [
                'read_csv', 'read_excel', 'read_json', 'read_parquet',
                'add_table', 'get_table', 'create_table',
                'reconcile', 'extend', 'extend_column',
                'merge', 'join', 'filter', 'aggregate',
                'fit', 'predict', 'transform',
                'save', 'to_csv', 'to_parquet', 'push_to_backend',
                'display', 'show',
            ]
            if any(func in label_lower for func in important_functions):
                return True
            return False
        
        # Include variables that look like data
        if node.node_type == NodeType.VARIABLE:
            data_patterns = ['df', 'data', 'table', 'result', 'payload', 'reconciled', 'extended']
            if any(pattern in label_lower for pattern in data_patterns):
                return True
            # Exclude other variables
            return False
        
        # Include intermediates only if they're data-related
        if node.node_type == NodeType.INTERMEDIATE:
            data_patterns = ['table', 'data', 'result', 'payload']
            if any(pattern in label_lower for pattern in data_patterns):
                return True
            return False
        
        return False


__all__ = [
    "SmartDFGBuilder",
]