"""
Ground Truth Manager Module
============================

Manage ground truth annotations for evaluation.

This module provides:
- Ground truth data structures
- Annotation loading/saving
- Annotation validation
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Any
from pathlib import Path


@dataclass
class CellAnnotation:
    """
    Ground truth annotation for a single cell.
    
    Attributes:
        cell_id: Cell identifier
        task_type: Expected task type
        stage: Expected pipeline stage
        produces_artifacts: List of artifact names produced
        consumes_artifacts: List of artifact names consumed
        is_important: Whether this cell is important for the pipeline
        notes: Optional annotator notes
    """
    cell_id: str
    task_type: str
    stage: str
    produces_artifacts: List[str] = field(default_factory=list)
    consumes_artifacts: List[str] = field(default_factory=list)
    is_important: bool = True
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CellAnnotation':
        return cls(**data)


@dataclass
class ArtifactAnnotation:
    """
    Ground truth annotation for a data artifact.
    
    Attributes:
        name: Artifact name
        artifact_type: Type of artifact (dataframe, table, model, etc.)
        created_in_cell: Cell where artifact is created
        source_artifacts: List of source artifact names (lineage)
        transformation_type: Type of transformation that created it
        importance: Importance level (1-10)
    """
    name: str
    artifact_type: str
    created_in_cell: str
    source_artifacts: List[str] = field(default_factory=list)
    transformation_type: str = ""
    importance: int = 5
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ArtifactAnnotation':
        return cls(**data)


@dataclass
class GroundTruthAnnotation:
    """
    Complete ground truth annotation for a notebook.
    
    Attributes:
        notebook_id: Unique identifier for the notebook
        notebook_path: Path to the notebook file
        cell_annotations: List of cell annotations
        artifact_annotations: List of artifact annotations
        stage_sequence: Expected sequence of pipeline stages
        lineage_edges: List of (source, target) lineage edges
        column_operations: Column lineage information
        complexity_level: Notebook complexity (simple, medium, complex)
        domain: Domain of the notebook (etl, ml, analytics, etc.)
        annotator: Who created the annotation
        annotation_date: When annotation was created
        notes: Additional notes
    """
    notebook_id: str
    notebook_path: str
    cell_annotations: List[CellAnnotation] = field(default_factory=list)
    artifact_annotations: List[ArtifactAnnotation] = field(default_factory=list)
    stage_sequence: List[str] = field(default_factory=list)
    lineage_edges: List[tuple] = field(default_factory=list)
    column_operations: Dict[str, Any] = field(default_factory=dict)
    complexity_level: str = "medium"
    domain: str = "general"
    annotator: str = ""
    annotation_date: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['cell_annotations'] = [c.to_dict() for c in self.cell_annotations]
        data['artifact_annotations'] = [a.to_dict() for a in self.artifact_annotations]
        data['lineage_edges'] = [list(e) for e in self.lineage_edges]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GroundTruthAnnotation':
        cell_annotations = [
            CellAnnotation.from_dict(c) for c in data.pop('cell_annotations', [])
        ]
        artifact_annotations = [
            ArtifactAnnotation.from_dict(a) for a in data.pop('artifact_annotations', [])
        ]
        lineage_edges = [tuple(e) for e in data.pop('lineage_edges', [])]
        
        return cls(
            cell_annotations=cell_annotations,
            artifact_annotations=artifact_annotations,
            lineage_edges=lineage_edges,
            **data
        )
    
    def validate(self) -> List[str]:
        """
        Validate the annotation for consistency.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check that all artifacts in lineage are defined
        defined_artifacts = {a.name for a in self.artifact_annotations}
        
        for source, target in self.lineage_edges:
            if source not in defined_artifacts:
                errors.append(f"Lineage source '{source}' not in artifact annotations")
            if target not in defined_artifacts:
                errors.append(f"Lineage target '{target}' not in artifact annotations")
        
        # Check that produced artifacts are defined
        for cell in self.cell_annotations:
            for artifact in cell.produces_artifacts:
                if artifact not in defined_artifacts:
                    errors.append(f"Cell {cell.cell_id} produces undefined artifact '{artifact}'")
        
        # Check stage sequence contains valid stages
        valid_stages = {
            'setup', 'data_loading', 'data_preparation', 'reconciliation',
            'enrichment', 'transformation', 'analysis', 'output'
        }
        for stage in self.stage_sequence:
            if stage not in valid_stages:
                errors.append(f"Invalid stage '{stage}' in stage_sequence")
        
        return errors


class GroundTruthManager:
    """
    Manage ground truth annotations for evaluation.
    
    This class handles loading, saving, and validating ground truth
    annotations for the evaluation benchmark.
    
    Example:
        >>> manager = GroundTruthManager("annotations/")
        >>> annotation = manager.load_annotation("notebook_001")
        >>> manager.save_annotation(annotation)
    """
    
    def __init__(self, annotations_dir: str):
        """
        Initialize ground truth manager.
        
        Args:
            annotations_dir: Directory containing annotation files
        """
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
    
    def load_annotation(self, notebook_id: str) -> Optional[GroundTruthAnnotation]:
        """
        Load annotation for a specific notebook.
        
        Args:
            notebook_id: Notebook identifier
            
        Returns:
            GroundTruthAnnotation or None if not found
        """
        if notebook_id in self._cache:
            return self._cache[notebook_id]
        
        annotation_path = self.annotations_dir / f"{notebook_id}.json"
        
        if not annotation_path.exists():
            return None
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotation = GroundTruthAnnotation.from_dict(data)
        self._cache[notebook_id] = annotation
        
        return annotation
    
    def save_annotation(self, annotation: GroundTruthAnnotation):
        """
        Save annotation to file.
        
        Args:
            annotation: GroundTruthAnnotation to save
        """
        annotation_path = self.annotations_dir / f"{annotation.notebook_id}.json"
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation.to_dict(), f, indent=2)
        
        self._cache[annotation.notebook_id] = annotation
        print(f"✓ Saved annotation for {annotation.notebook_id}")
    
    def load_all_annotations(self) -> Dict[str, GroundTruthAnnotation]:
        """
        Load all annotations in the directory.
        
        Returns:
            Dictionary mapping notebook_id to GroundTruthAnnotation
        """
        annotations = {}
        
        for path in self.annotations_dir.glob("*.json"):
            notebook_id = path.stem
            annotation = self.load_annotation(notebook_id)
            if annotation:
                annotations[notebook_id] = annotation
        
        return annotations
    
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate all annotations.
        
        Returns:
            Dictionary mapping notebook_id to list of errors
        """
        results = {}
        annotations = self.load_all_annotations()
        
        for notebook_id, annotation in annotations.items():
            errors = annotation.validate()
            if errors:
                results[notebook_id] = errors
        
        return results
    
    def create_template(self, notebook_id: str, notebook_path: str,
                       cell_ids: List[str]) -> GroundTruthAnnotation:
        """
        Create an empty annotation template.
        
        Args:
            notebook_id: Notebook identifier
            notebook_path: Path to notebook
            cell_ids: List of cell IDs in the notebook
            
        Returns:
            Template GroundTruthAnnotation
        """
        from datetime import datetime
        
        cell_annotations = [
            CellAnnotation(
                cell_id=cell_id,
                task_type="other",
                stage="transformation",
            )
            for cell_id in cell_ids
        ]
        
        annotation = GroundTruthAnnotation(
            notebook_id=notebook_id,
            notebook_path=notebook_path,
            cell_annotations=cell_annotations,
            annotation_date=datetime.now().isoformat(),
        )
        
        return annotation
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the annotation dataset.
        
        Returns:
            Dictionary of statistics
        """
        annotations = self.load_all_annotations()
        
        if not annotations:
            return {'total': 0}
        
        from collections import Counter
        
        complexity_counts = Counter(a.complexity_level for a in annotations.values())
        domain_counts = Counter(a.domain for a in annotations.values())
        
        total_cells = sum(len(a.cell_annotations) for a in annotations.values())
        total_artifacts = sum(len(a.artifact_annotations) for a in annotations.values())
        
        return {
            'total_notebooks': len(annotations),
            'total_cells': total_cells,
            'total_artifacts': total_artifacts,
            'by_complexity': dict(complexity_counts),
            'by_domain': dict(domain_counts),
            'avg_cells_per_notebook': total_cells / len(annotations),
            'avg_artifacts_per_notebook': total_artifacts / len(annotations),
        }


__all__ = [
    "CellAnnotation",
    "ArtifactAnnotation",
    "GroundTruthAnnotation",
    "GroundTruthManager",
]