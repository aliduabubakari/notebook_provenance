"""
Column Lineage Tracker Module
==============================

Track column-level lineage via regex heuristics.

This module provides the ColumnLineageTracker class which:
- Tracks column creation, modification, and deletion
- Identifies column renames
- Uses regex patterns to detect column operations
"""

import re
from typing import Dict, List, Set, Any
from collections import defaultdict

from notebook_provenance.parsing.ast_parser import ParsedCell


class ColumnLineageTracker:
    """
    Track column-level lineage via regex heuristics.
    
    This class uses pattern matching to identify column-level operations
    such as creation, deletion, and renaming in data manipulation code.
    
    Example:
        >>> tracker = ColumnLineageTracker()
        >>> lineage = tracker.extract_column_lineage(parsed_cells)
        >>> print(lineage['created'])
    """
    
    def __init__(self):
        """Initialize column lineage tracker with patterns."""
        # Patterns focus on common pandas and dict-style operations
        
        # df['col'] = ...
        self.create_pat = re.compile(
            r"([A-Za-z_]\w*)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*="
        )
        
        # df.drop(columns=['a','b']) or df.drop(labels=['a',...], axis=1)
        self.drop_pat = re.compile(
            r"\.drop\(\s*(?:columns|labels)\s*=\s*\[([^\]]+)\]"
        )
        
        # df.rename(columns={'old':'new', ...})
        self.rename_pat = re.compile(
            r"\.rename\(\s*columns\s*=\s*\{([^}]+)\}"
        )
        
        # table_data['columns']['col'] = ...
        self.create_tabledata_pat = re.compile(
            r"(table_data|.*_table)\s*\[\s*['\"]columns['\"]\s*\]\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"
        )
        
        # df.assign(new_col=...)
        self.assign_pat = re.compile(
            r"\.assign\(\s*([A-Za-z_]\w*)\s*="
        )
        
        # df['col'].apply(...) or df['col'].map(...)
        self.modify_pat = re.compile(
            r"([A-Za-z_]\w*)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*\.\s*(?:apply|map|transform)"
        )
    
    def extract_column_lineage(self, parsed_cells) -> Dict[str, Dict[str, Any]]:
        """
        Extract column-level lineage from parsed cells.
        
        FIXED: Now handles both ParsedCell objects and dictionaries.
        
        Args:
            parsed_cells: List of ParsedCell objects OR dictionaries
            
        Returns:
            Dictionary with keys:
                - 'created': {col_name: cell_id}
                - 'modified': {col_name: [cell_ids]}
                - 'dropped': {col_name: cell_id}
                - 'renamed': {old_name: new_name}
        """
        lineage = {
            'created': {},      # col -> first cell_id where created
            'modified': defaultdict(list),  # col -> list[cell_id] where modified
            'dropped': {},      # col -> cell_id where dropped
            'renamed': {},      # old_name -> new_name
        }
        
        for cell in parsed_cells:
            # Handle both ParsedCell objects and dictionaries
            if isinstance(cell, dict):
                if cell.get('error'):
                    continue
                code = cell.get('code', '')
                cell_id = cell.get('cell_id', '')
            else:
                # It's a ParsedCell object
                if cell.error:
                    continue
                code = cell.code
                cell_id = cell.cell_id
            
            # Track column creation: df['new_col'] = ...
            for match in self.create_pat.finditer(code):
                df_name = match.group(1)
                col_name = match.group(2)
                
                if col_name not in lineage['created']:
                    lineage['created'][col_name] = cell_id
                else:
                    # Column exists, this is a modification
                    lineage['modified'][col_name].append(cell_id)
            
            # Track table_data column creation
            for match in self.create_tabledata_pat.finditer(code):
                col_name = match.group(2)
                
                if col_name not in lineage['created']:
                    lineage['created'][col_name] = cell_id
                else:
                    lineage['modified'][col_name].append(cell_id)
            
            # Track df.assign(col=...)
            for match in self.assign_pat.finditer(code):
                col_name = match.group(1)
                
                if col_name not in lineage['created']:
                    lineage['created'][col_name] = cell_id
                else:
                    lineage['modified'][col_name].append(cell_id)
            
            # Track column modification: df['col'].apply(...)
            for match in self.modify_pat.finditer(code):
                col_name = match.group(2)
                lineage['modified'][col_name].append(cell_id)
            
            # Track column drops
            for match in self.drop_pat.finditer(code):
                cols = self._extract_list_of_strings(match.group(1))
                for col in cols:
                    lineage['dropped'][col] = cell_id
            
            # Track column renames
            for match in self.rename_pat.finditer(code):
                pairs = self._extract_rename_pairs(match.group(1))
                lineage['renamed'].update(pairs)
        
        # Convert defaultdict to regular dict for JSON serialization
        lineage['modified'] = dict(lineage['modified'])
        
        return lineage
    
    def _extract_list_of_strings(self, text: str) -> List[str]:
        """
        Extract list of quoted strings from text.
        
        Example: "'a', 'b', 'c'" -> ['a', 'b', 'c']
        
        Args:
            text: Text containing quoted strings
            
        Returns:
            List of extracted strings
        """
        return re.findall(r"['\"]([^'\"]+)['\"]", text)
    
    def _extract_rename_pairs(self, text: str) -> Dict[str, str]:
        """
        Extract rename pairs from rename dict.
        
        Example: "'old':'new', 'x': 'y'" -> {'old': 'new', 'x': 'y'}
        
        Args:
            text: Text containing rename pairs
            
        Returns:
            Dictionary of old -> new names
        """
        pairs = re.findall(r"['\"]([^'\"]+)['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
        return {old: new for old, new in pairs}
    
    def apply_renames(self, lineage: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Apply rename operations to normalize column names.
        
        This updates created and modified dictionaries to use final column names
        after all renames have been applied.
        
        Args:
            lineage: Lineage dictionary
            
        Returns:
            Updated lineage dictionary
        """
        renamed = lineage.get('renamed', {})
        
        if not renamed:
            return lineage
        
        # Build rename chain (handle multiple renames)
        final_names = {}
        for old, new in renamed.items():
            # Follow the chain
            current = old
            visited = {current}
            
            while current in renamed:
                current = renamed[current]
                if current in visited:
                    # Circular rename, break
                    break
                visited.add(current)
            
            final_names[old] = current
        
        # Apply renames to created
        new_created = {}
        for col, cell_id in lineage['created'].items():
            final_name = final_names.get(col, col)
            new_created[final_name] = cell_id
        lineage['created'] = new_created
        
        # Apply renames to modified
        new_modified = defaultdict(list)
        for col, cell_ids in lineage['modified'].items():
            final_name = final_names.get(col, col)
            new_modified[final_name].extend(cell_ids)
        lineage['modified'] = dict(new_modified)
        
        # Apply renames to dropped
        new_dropped = {}
        for col, cell_id in lineage['dropped'].items():
            final_name = final_names.get(col, col)
            new_dropped[final_name] = cell_id
        lineage['dropped'] = new_dropped
        
        return lineage
    
    def get_column_history(self, lineage: Dict[str, Dict[str, Any]], 
                          column_name: str) -> Dict[str, Any]:
        """
        Get complete history of a specific column.
        
        Args:
            lineage: Lineage dictionary
            column_name: Column name to track
            
        Returns:
            Dictionary with column history:
                - created_in: cell_id or None
                - modified_in: list of cell_ids
                - dropped_in: cell_id or None
                - renamed_from: original name or None
        """
        # Check if this column was renamed
        renamed_from = None
        for old, new in lineage.get('renamed', {}).items():
            if new == column_name:
                renamed_from = old
                break
        
        return {
            'column': column_name,
            'created_in': lineage['created'].get(column_name),
            'modified_in': lineage['modified'].get(column_name, []),
            'dropped_in': lineage['dropped'].get(column_name),
            'renamed_from': renamed_from,
        }
    
    def get_active_columns(self, lineage: Dict[str, Dict[str, Any]]) -> Set[str]:
        """
        Get set of columns that are currently active (created but not dropped).
        
        Args:
            lineage: Lineage dictionary
            
        Returns:
            Set of active column names
        """
        created = set(lineage['created'].keys())
        dropped = set(lineage['dropped'].keys())
        return created - dropped
    
    def get_lineage_stats(self, lineage: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about column lineage.
        
        Args:
            lineage: Lineage dictionary
            
        Returns:
            Dictionary of statistics
        """
        created_count = len(lineage['created'])
        dropped_count = len(lineage['dropped'])
        renamed_count = len(lineage['renamed'])
        modified_count = len(lineage['modified'])
        active_count = len(self.get_active_columns(lineage))
        
        # Most modified columns
        most_modified = []
        if lineage['modified']:
            sorted_modified = sorted(
                lineage['modified'].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            most_modified = [(col, len(cells)) for col, cells in sorted_modified[:5]]
        
        return {
            'total_created': created_count,
            'total_dropped': dropped_count,
            'total_renamed': renamed_count,
            'total_modified': modified_count,
            'active_columns': active_count,
            'most_modified': most_modified,
        }
    
    def visualize_column_flow(self, lineage: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a text-based visualization of column flow.
        
        Args:
            lineage: Lineage dictionary
            
        Returns:
            Text visualization
        """
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("COLUMN LINEAGE FLOW")
        lines.append("=" * 60)
        
        # Show active columns
        active = self.get_active_columns(lineage)
        lines.append(f"\n📊 Active Columns ({len(active)}):")
        for col in sorted(active):
            created_cell = lineage['created'].get(col, 'unknown')
            modified_cells = lineage['modified'].get(col, [])
            
            status = f"  • {col}"
            status += f" (created in {created_cell}"
            if modified_cells:
                status += f", modified {len(modified_cells)} times"
            status += ")"
            lines.append(status)
        
        # Show dropped columns
        if lineage['dropped']:
            lines.append(f"\n🗑️  Dropped Columns ({len(lineage['dropped'])}):")
            for col, cell_id in sorted(lineage['dropped'].items()):
                lines.append(f"  • {col} (dropped in {cell_id})")
        
        # Show renames
        if lineage['renamed']:
            lines.append(f"\n🔄 Renamed Columns ({len(lineage['renamed'])}):")
            for old, new in sorted(lineage['renamed'].items()):
                lines.append(f"  • {old} → {new}")
        
        lines.append("\n" + "=" * 60 + "\n")
        
        return '\n'.join(lines)


__all__ = [
    "ColumnLineageTracker",
]