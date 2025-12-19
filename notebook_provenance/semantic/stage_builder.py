"""
Stage Builder Module
====================

Build high-level pipeline stages from cells.

This module provides the PipelineStageBuilder class which:
- Groups cells into logical pipeline stages
- Uses LLM for intelligent stage detection
- Generates stage descriptions
- Identifies input/output artifacts for each stage
"""

from typing import Dict, List, Optional
from collections import defaultdict

from notebook_provenance.core.data_structures import (
    PipelineStageNode,
    DataArtifact,
    CellDependency,
)
from notebook_provenance.core.enums import PipelineStage, TaskType
from notebook_provenance.parsing.ast_parser import ParsedCell
from notebook_provenance.semantic.llm_analyzer import LLMSemanticAnalyzer


class PipelineStageBuilder:
    """
    Build high-level pipeline stages from cells.
    
    This class groups related cells into logical pipeline stages,
    using either LLM-based analysis or heuristic methods.
    
    Example:
        >>> builder = PipelineStageBuilder(llm_analyzer)
        >>> stages = builder.build_stages(parsed_cells, artifacts, dependencies)
    """
    
    def __init__(self, llm_analyzer: Optional[LLMSemanticAnalyzer] = None):
        """
        Initialize stage builder.
        
        Args:
            llm_analyzer: Optional LLM analyzer for intelligent stage detection
        """
        self.llm_analyzer = llm_analyzer
    
    def build_stages(self, parsed_cells: List[ParsedCell], 
                    artifacts: List[DataArtifact],
                    cell_dependencies: Dict[str, CellDependency]) -> List[PipelineStageNode]:
        """
        Build pipeline stages from cells.
        
        Args:
            parsed_cells: List of ParsedCell objects
            artifacts: List of identified data artifacts
            cell_dependencies: Cell dependency information
            
        Returns:
            List of PipelineStageNode objects
        """
        # Convert ParsedCell objects to dicts for LLM analyzer
        cell_dicts = [self._parsed_cell_to_dict(cell) for cell in parsed_cells]
        
        # Use LLM to detect stage boundaries
        if self.llm_analyzer and self.llm_analyzer.enabled:
            stage_groups = self.llm_analyzer.detect_pipeline_stages(cell_dicts)
        else:
            stage_groups = self._heuristic_stage_detection(parsed_cells)
        
        stages = []
        
        for group in stage_groups:
            stage_type_str = group.get('stage', 'other')
            cell_indices_or_ids = group.get('cells', [])
            confidence = group.get('confidence', 0.8)
            
            # Convert to cell IDs if indices
            cell_ids = []
            for item in cell_indices_or_ids:
                if isinstance(item, int):
                    if 0 <= item < len(parsed_cells):
                        cell_ids.append(parsed_cells[item].cell_id)
                else:
                    cell_ids.append(item)
            
            if not cell_ids:
                continue
            
            # Determine stage type
            try:
                stage_type = PipelineStage(stage_type_str)
            except ValueError:
                # Try to map from task type
                try:
                    task_type = TaskType(stage_type_str)
                    stage_type = PipelineStage.from_task_type(task_type)
                except ValueError:
                    stage_type = PipelineStage.TRANSFORMATION
            
            # Collect operations
            operations = []
            for cell_id in cell_ids:
                cell = self._get_cell_by_id(parsed_cells, cell_id)
                if cell:
                    operations.extend(cell.function_calls[:10])  # Limit operations
            
            # Find input/output artifacts
            input_artifacts, output_artifacts = self._find_stage_artifacts(
                cell_ids, artifacts, cell_dependencies
            )
            
            # Generate description
            description = self._generate_stage_description(
                stage_type, cell_ids, operations, parsed_cells
            )
            
            stage = PipelineStageNode(
                id=f"stage_{'_'.join(cell_ids[:3])}",  # Use first 3 cell IDs
                stage_type=stage_type,
                cells=cell_ids,
                operations=list(set(operations)),  # Deduplicate
                input_artifacts=input_artifacts,
                output_artifacts=output_artifacts,
                description=description,
                confidence=confidence
            )
            
            stages.append(stage)
        
        return stages
    
    def _parsed_cell_to_dict(self, cell: ParsedCell) -> Dict:
        """Convert ParsedCell to dictionary for LLM analyzer."""
        return {
            'cell_id': cell.cell_id,
            'code': cell.code,
            'variables_defined': cell.variables_defined,
            'function_calls': cell.function_calls,
            'imports': cell.imports,
            'error': cell.error,
        }
    
    def _get_cell_by_id(self, parsed_cells: List[ParsedCell], 
                       cell_id: str) -> Optional[ParsedCell]:
        """Get cell by ID."""
        for cell in parsed_cells:
            if cell.cell_id == cell_id:
                return cell
        return None
    
    def _find_stage_artifacts(self, cell_ids: List[str],
                             artifacts: List[DataArtifact],
                             cell_dependencies: Dict[str, CellDependency]) -> tuple:
        """
        Find input and output artifacts for a stage.
        
        Args:
            cell_ids: List of cell IDs in the stage
            artifacts: All data artifacts
            cell_dependencies: Cell dependency information
            
        Returns:
            Tuple of (input_artifact_ids, output_artifact_ids)
        """
        input_artifacts = []
        output_artifacts = []
        
        cell_id_set = set(cell_ids)
        
        for artifact in artifacts:
            # Output: artifact created in this stage
            if artifact.created_in_cell in cell_id_set:
                output_artifacts.append(artifact.id)
            else:
                # Input: artifact used by cells in this stage
                for cell_id in cell_ids:
                    deps = cell_dependencies.get(cell_id)
                    if deps and artifact.name in deps.consumes:
                        input_artifacts.append(artifact.id)
                        break
        
        return list(set(input_artifacts)), list(set(output_artifacts))
    
    def _generate_stage_description(self, stage_type: PipelineStage, 
                                   cell_ids: List[str], 
                                   operations: List[str],
                                   parsed_cells: List[ParsedCell]) -> str:
        """
        Generate human-readable stage description.
        
        Args:
            stage_type: Type of pipeline stage
            cell_ids: Cell IDs in the stage
            operations: Operations performed in the stage
            parsed_cells: All parsed cells
            
        Returns:
            Description string
        """
        # Try LLM description
        if self.llm_analyzer and self.llm_analyzer.enabled:
            code_snippets = []
            for cell_id in cell_ids[:3]:  # Limit to first 3 cells
                cell = self._get_cell_by_id(parsed_cells, cell_id)
                if cell:
                    code_snippets.append(cell.code[:200])
            
            combined_code = '\n\n'.join(code_snippets)
            
            prompt = f"""Describe this pipeline stage in one sentence (max 15 words).

STAGE TYPE: {stage_type.value}
OPERATIONS: {', '.join(operations[:5])}

CODE:
{combined_code[:500]}

Provide a clear, concise description of what this stage does.
Example: "Load air quality data from CSV and store in table"

Respond with ONLY the description."""

            try:
                response = self.llm_analyzer._call_llm(
                    prompt,
                    temperature=0.3,
                    max_tokens=50
                )
                
                description = response.strip().strip('"').strip("'")
                return description[:150]
            except:
                pass
        
        # Fallback to heuristic descriptions
        return self._heuristic_stage_description(stage_type, operations)
    
    def _heuristic_stage_description(self, stage_type: PipelineStage,
                                    operations: List[str]) -> str:
        """Generate heuristic-based stage description."""
        descriptions = {
            PipelineStage.SETUP: "Initialize environment and authenticate services",
            PipelineStage.DATA_LOADING: "Load data from external sources",
            PipelineStage.DATA_PREPARATION: "Clean and prepare data",
            PipelineStage.RECONCILIATION: "Reconcile and match entities",
            PipelineStage.ENRICHMENT: "Enrich data from external APIs",
            PipelineStage.TRANSFORMATION: "Transform and engineer features",
            PipelineStage.ANALYSIS: "Analyze and model data",
            PipelineStage.OUTPUT: "Save and visualize results",
        }
        
        base_desc = descriptions.get(stage_type, "Process data")
        
        # Add operation details if available
        if operations:
            key_ops = operations[:2]
            base_desc += f" ({', '.join(key_ops)})"
        
        return base_desc
    
    def _heuristic_stage_detection(self, parsed_cells: List[ParsedCell]) -> List[Dict]:
        """
        Fallback heuristic stage detection.
        
        Args:
            parsed_cells: List of ParsedCell objects
            
        Returns:
            List of stage group dictionaries
        """
        stages = []
        current_stage = None
        current_cells = []
        
        for cell in parsed_cells:
            if cell.error:
                continue
            
            # Classify cell
            stage_type = self._classify_cell_stage(cell)
            
            if current_stage is None or current_stage == stage_type:
                current_stage = stage_type
                current_cells.append(cell.cell_id)
            else:
                # Save previous stage
                if current_cells:
                    stages.append({
                        'stage': current_stage,
                        'cells': current_cells,
                        'confidence': 0.7
                    })
                
                # Start new stage
                current_stage = stage_type
                current_cells = [cell.cell_id]
        
        # Don't forget last stage
        if current_cells:
            stages.append({
                'stage': current_stage,
                'cells': current_cells,
                'confidence': 0.7
            })
        
        return stages
    
    def _classify_cell_stage(self, cell: ParsedCell) -> str:
        """Classify cell into stage type."""
        code = cell.code.lower()
        
        if any(kw in code for kw in ['import', 'authmanager', 'token']):
            return 'setup'
        elif any(kw in code for kw in ['read_csv', 'read_', 'load', 'fetch']):
            return 'data_loading'
        elif any(kw in code for kw in ['dropna', 'fillna', 'clean', 'drop_duplicates']):
            return 'data_preparation'
        elif 'reconcile' in code:
            return 'reconciliation'
        elif any(kw in code for kw in ['extend', 'enrich', 'osm', 'api', 'llm']):
            return 'enrichment'
        elif any(kw in code for kw in ['to_csv', 'save', 'write', 'plot']):
            return 'output'
        else:
            return 'transformation'
    
    def merge_adjacent_stages(self, stages: List[PipelineStageNode]) -> List[PipelineStageNode]:
        """
        Merge adjacent stages of the same type.
        
        Args:
            stages: List of pipeline stages
            
        Returns:
            List of merged stages
        """
        if not stages:
            return []
        
        merged = []
        current = stages[0]
        
        for next_stage in stages[1:]:
            if next_stage.stage_type == current.stage_type:
                # Merge into current
                current.cells.extend(next_stage.cells)
                current.operations.extend(next_stage.operations)
                current.input_artifacts = list(set(
                    current.input_artifacts + next_stage.input_artifacts
                ))
                current.output_artifacts = list(set(
                    current.output_artifacts + next_stage.output_artifacts
                ))
                # Average confidence
                current.confidence = (current.confidence + next_stage.confidence) / 2
            else:
                # Save current and start new
                merged.append(current)
                current = next_stage
        
        # Don't forget last stage
        merged.append(current)
        
        return merged


__all__ = [
    "PipelineStageBuilder",
]