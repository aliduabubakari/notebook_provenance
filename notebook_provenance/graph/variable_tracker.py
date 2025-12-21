"""
Variable Tracker Module
========================

Track unique variables across cells and deduplicate artifacts.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from notebook_provenance.core.data_structures import DataArtifact, DFGNode
from notebook_provenance.core.enums import NodeType


@dataclass
class VariableVersion:
    """Track a single version of a variable."""
    node_id: str
    cell_id: str
    line_number: int
    created_by: Optional[str] = None  # Function that created this version
    is_reassignment: bool = False


@dataclass 
class TrackedVariable:
    """Track all versions of a variable across the notebook."""
    name: str
    versions: List[VariableVersion] = field(default_factory=list)
    artifact_type: Optional[str] = None
    
    @property
    def latest_version(self) -> Optional[VariableVersion]:
        return self.versions[-1] if self.versions else None
    
    @property
    def first_version(self) -> Optional[VariableVersion]:
        return self.versions[0] if self.versions else None
    
    @property
    def is_multi_version(self) -> bool:
        return len(self.versions) > 1


class VariableTracker:
    """
    Track and deduplicate variables across notebook cells.
    
    This class maintains a registry of all variables and their versions, proper deduplication of artifacts in the lineage graph.
    """
    
    def __init__(self):
        self.variables: Dict[str, TrackedVariable] = {}
        self.node_to_variable: Dict[str, str] = {}  # node_id -> variable_name
        
        # Categories for filtering
        self.core_data_patterns = [
            'df', 'data', 'table_data', 'dataset', 
            'result', 'reconciled', 'extended', 'merged', 'joined',
            'transformed', 'filtered', 'aggregated'
        ]
        
        self.metadata_patterns = [
            'id', 'name', 'key', 'index', 'column', 'columns',
            'table_id', 'dataset_id', 'column_name'
        ]
        
        self.payload_patterns = [
            'payload', 'request', 'response', 'body'
        ]
        
        self.display_patterns = [
            'html', 'display', 'plot', 'figure', 'chart', 'viz'
        ]
        
        self.config_patterns = [
            'url', 'uri', 'endpoint', 'token', 'password', 'username',
            'auth', 'manager', 'client', 'config', 'setting'
        ]
    
    def register_node(self, node: DFGNode, created_by: Optional[str] = None):
        """
        Register a node and track it as a variable version.
        
        Args:
            node: DFG node representing a variable
            created_by: Optional function name that created this variable
        """
        if node.node_type not in [NodeType.VARIABLE, NodeType.DATA_ARTIFACT]:
            return
        
        var_name = node.label
        
        # Create or get tracked variable
        if var_name not in self.variables:
            self.variables[var_name] = TrackedVariable(
                name=var_name,
                artifact_type=self._classify_variable(var_name)
            )
        
        # Check if this is a reassignment (same var name in same cell)
        tracked = self.variables[var_name]
        is_reassignment = any(
            v.cell_id == node.cell_id 
            for v in tracked.versions
        )
        
        # Add version
        version = VariableVersion(
            node_id=node.id,
            cell_id=node.cell_id,
            line_number=node.line_number,
            created_by=created_by,
            is_reassignment=is_reassignment
        )
        
        tracked.versions.append(version)
        self.node_to_variable[node.id] = var_name
    
    def _classify_variable(self, name: str) -> str:
        """Classify variable into category."""
        name_lower = name.lower()
        
        # Check config first (to exclude)
        for pattern in self.config_patterns:
            if pattern in name_lower:
                return 'config'
        
        # Check display (lower priority in lineage)
        for pattern in self.display_patterns:
            if pattern in name_lower:
                return 'display'
        
        # Check metadata
        for pattern in self.metadata_patterns:
            if pattern in name_lower:
                return 'metadata'
        
        # Check payload
        for pattern in self.payload_patterns:
            if pattern in name_lower:
                return 'payload'
        
        # Check core data
        for pattern in self.core_data_patterns:
            if pattern in name_lower or name_lower.startswith(pattern):
                return 'core_data'
        
        return 'unknown'
    
    def get_unique_artifacts(self, 
                            include_types: Optional[Set[str]] = None,
                            exclude_types: Optional[Set[str]] = None) -> List[TrackedVariable]:
        """
        Get unique artifacts (deduplicated by name).
        
        Args:
            include_types: Only include these artifact types
            exclude_types: Exclude these artifact types
            
        Returns:
            List of unique TrackedVariable objects
        """
        if include_types is None:
            include_types = {'core_data', 'payload', 'metadata'}
        
        if exclude_types is None:
            exclude_types = {'config', 'display'}
        
        unique = []
        for var in self.variables.values():
            if var.artifact_type in exclude_types:
                continue
            if include_types and var.artifact_type not in include_types:
                continue
            unique.append(var)
        
        return unique
    
    def get_core_data_artifacts(self) -> List[TrackedVariable]:
        """Get only core data artifacts for simplified lineage."""
        return self.get_unique_artifacts(include_types={'core_data'})
    
    def get_canonical_node_id(self, var_name: str) -> Optional[str]:
        """Get the canonical (first or most important) node ID for a variable."""
        if var_name not in self.variables:
            return None
        
        tracked = self.variables[var_name]
        
        # Prefer the version created by a data-creating function
        for version in tracked.versions:
            if version.created_by and any(
                func in version.created_by.lower()
                for func in ['read_', 'load', 'add_table', 'reconcile', 'extend', 'merge', 'join']
            ):
                return version.node_id
        
        # Otherwise return first version
        return tracked.first_version.node_id if tracked.first_version else None
    
    def build_deduplication_map(self) -> Dict[str, str]:
        """
        Build a map from all node IDs to canonical node IDs.
        
        Returns:
            Dict mapping any node_id to its canonical node_id
        """
        dedup_map = {}
        
        for var_name, tracked in self.variables.items():
            canonical = self.get_canonical_node_id(var_name)
            if canonical:
                for version in tracked.versions:
                    dedup_map[version.node_id] = canonical
        
        return dedup_map
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        type_counts = defaultdict(int)
        multi_version_count = 0
        
        for var in self.variables.values():
            type_counts[var.artifact_type] += 1
            if var.is_multi_version:
                multi_version_count += 1
        
        return {
            'total_unique_variables': len(self.variables),
            'total_node_registrations': len(self.node_to_variable),
            'multi_version_variables': multi_version_count,
            'by_type': dict(type_counts)
        }


__all__ = [
    "VariableTracker",
    "TrackedVariable",
    "VariableVersion",
]