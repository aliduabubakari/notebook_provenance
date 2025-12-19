"""
Neo4j Exporter Module
=====================

Export provenance to Neo4j graph database.

This module provides the Neo4jExporter class which:
- Exports artifacts as nodes
- Exports transformations as relationships
- Creates indexed properties
- Supports batch operations
"""

from typing import Dict, List, Optional, Any
import time

from notebook_provenance.core.data_structures import (
    DataArtifact,
    Transformation,
    PipelineStageNode,
)


class Neo4jExporter:
    """
    Export provenance to Neo4j graph database.
    
    This class exports provenance data to a Neo4j graph database,
    allowing for powerful graph queries and analysis.
    
    Example:
        >>> exporter = Neo4jExporter(uri, user, password)
        >>> exporter.export_to_neo4j(result)
        >>> exporter.close()
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j exporter.
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
        # Try to import neo4j
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print(f"✓ Connected to Neo4j at {uri}")
        except ImportError:
            print("⚠ neo4j package not installed. Run: pip install neo4j")
            raise
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            raise
    
    def export_to_neo4j(self, result: Dict, clear_existing: bool = True):
        """
        Export complete provenance to Neo4j.
        
        Args:
            result: Complete analysis result
            clear_existing: Whether to clear existing data
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        with self.driver.session() as session:
            # Clear existing data if requested
            if clear_existing:
                print("Clearing existing data...")
                session.run("MATCH (n) DETACH DELETE n")
            
            # Create constraints and indexes
            self._create_constraints(session)
            
            # Export artifacts
            artifacts = result.get('artifacts', [])
            print(f"Exporting {len(artifacts)} artifacts...")
            self._export_artifacts(session, artifacts)
            
            # Export transformations
            transformations = result.get('transformations', [])
            print(f"Exporting {len(transformations)} transformations...")
            self._export_transformations(session, transformations)
            
            # Export stages
            stages = result.get('stages', [])
            print(f"Exporting {len(stages)} pipeline stages...")
            self._export_stages(session, stages)
            
            # Create relationships between stages and artifacts
            self._link_stages_artifacts(session, stages)
        
        print(f"✓ Provenance exported to Neo4j at {self.uri}")
    
    def _create_constraints(self, session):
        """Create constraints and indexes."""
        # Create uniqueness constraints
        try:
            session.run("CREATE CONSTRAINT artifact_id IF NOT EXISTS FOR (a:Artifact) REQUIRE a.id IS UNIQUE")
            session.run("CREATE CONSTRAINT stage_id IF NOT EXISTS FOR (s:Stage) REQUIRE s.id IS UNIQUE")
        except Exception as e:
            print(f"Note: Could not create constraints: {e}")
        
        # Create indexes
        try:
            session.run("CREATE INDEX artifact_name IF NOT EXISTS FOR (a:Artifact) ON (a.name)")
            session.run("CREATE INDEX artifact_type IF NOT EXISTS FOR (a:Artifact) ON (a.type)")
            session.run("CREATE INDEX stage_type IF NOT EXISTS FOR (s:Stage) ON (s.stage_type)")
        except Exception as e:
            print(f"Note: Could not create indexes: {e}")
    
    def _export_artifacts(self, session, artifacts: List[DataArtifact]):
        """Export artifacts as nodes."""
        for artifact in artifacts:
            session.run("""
                CREATE (a:Artifact {
                    id: $id,
                    name: $name,
                    type: $type,
                    created_in_cell: $cell,
                    importance: $importance
                })
            """,
            id=artifact.id,
            name=artifact.name,
            type=artifact.type,
            cell=artifact.created_in_cell,
            importance=artifact.importance_score
            )
    
    def _export_transformations(self, session, transformations: List[Transformation]):
        """Export transformations as relationships."""
        for trans in transformations:
            for source_id in trans.source_artifacts:
                session.run("""
                    MATCH (a1:Artifact {id: $source})
                    MATCH (a2:Artifact {id: $target})
                    CREATE (a1)-[:TRANSFORMS_TO {
                        transformation_id: $trans_id,
                        operation: $operation,
                        description: $description,
                        semantic_type: $semantic_type
                    }]->(a2)
                """,
                source=source_id,
                target=trans.target_artifact,
                trans_id=trans.id,
                operation=trans.operation,
                description=trans.description,
                semantic_type=trans.semantic_type
                )
    
    def _export_stages(self, session, stages: List[PipelineStageNode]):
        """Export pipeline stages as nodes."""
        for stage in stages:
            session.run("""
                CREATE (s:Stage {
                    id: $id,
                    stage_type: $stage_type,
                    description: $description,
                    cell_count: $cell_count,
                    confidence: $confidence
                })
            """,
            id=stage.id,
            stage_type=stage.stage_type.value,
            description=stage.description,
            cell_count=len(stage.cells),
            confidence=stage.confidence
            )
    
    def _link_stages_artifacts(self, session, stages: List[PipelineStageNode]):
        """Create relationships between stages and artifacts."""
        for stage in stages:
            # Link input artifacts
            for artifact_id in stage.input_artifacts:
                session.run("""
                    MATCH (s:Stage {id: $stage_id})
                    MATCH (a:Artifact {id: $artifact_id})
                    CREATE (a)-[:INPUT_TO]->(s)
                """,
                stage_id=stage.id,
                artifact_id=artifact_id
                )
            
            # Link output artifacts
            for artifact_id in stage.output_artifacts:
                session.run("""
                    MATCH (s:Stage {id: $stage_id})
                    MATCH (a:Artifact {id: $artifact_id})
                    CREATE (s)-[:PRODUCES]->(a)
                """,
                stage_id=stage.id,
                artifact_id=artifact_id
                )
    
    def query_artifact_lineage(self, artifact_name: str) -> List[Dict]:
        """
        Query lineage of a specific artifact.
        
        Args:
            artifact_name: Name of the artifact
            
        Returns:
            List of lineage records
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (source:Artifact)-[:TRANSFORMS_TO*]->(target:Artifact {name: $name})
                RETURN source.name as source_name, 
                       target.name as target_name,
                       [rel in relationships(path) | rel.operation] as operations
                ORDER BY length(path)
            """, name=artifact_name)
            
            return [dict(record) for record in result]
    
    def query_stage_flow(self) -> List[Dict]:
        """
        Query the pipeline stage flow.
        
        Returns:
            List of stage records in order
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Stage)
                RETURN s.id as id, 
                       s.stage_type as stage_type, 
                       s.description as description,
                       s.cell_count as cell_count
                ORDER BY s.id
            """)
            
            return [dict(record) for record in result]
    
    def query_transformation_by_type(self, semantic_type: str) -> List[Dict]:
        """
        Query transformations by semantic type.
        
        Args:
            semantic_type: Type of transformation
            
        Returns:
            List of transformation records
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a1:Artifact)-[t:TRANSFORMS_TO]->(a2:Artifact)
                WHERE t.semantic_type = $type
                RETURN a1.name as source,
                       a2.name as target,
                       t.operation as operation,
                       t.description as description
            """, type=semantic_type)
            
            return [dict(record) for record in result]
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed")


__all__ = [
    "Neo4jExporter",
]