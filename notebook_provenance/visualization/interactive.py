"""
Interactive Visualizer Module
==============================

Create interactive HTML visualizations using vis.js.

This module provides the InteractiveVisualizer class which:
- Creates interactive graph visualizations
- Adds search and filter capabilities
- Generates responsive HTML with controls
"""

from typing import Dict, List, Any
import json

from notebook_provenance.core.data_structures import DataFlowGraph


class InteractiveVisualizer:
    """
    Create interactive HTML visualizations.
    
    This class generates interactive, web-based visualizations
    using vis.js for graph rendering with search and filter capabilities.
    
    Example:
        >>> visualizer = InteractiveVisualizer()
        >>> visualizer.create_interactive_html(result, "output.html")
    """
    
    def create_interactive_html(self, result: Dict, 
                               output_file: str = "provenance_interactive.html"):
        """
        Create enhanced interactive HTML visualization.
        
        Args:
            result: Complete analysis result
            output_file: Output file path
        """
        # Extract data
        clean_dfg = result.get('clean_dfg')
        stages = result.get('stages', [])
        artifacts = result.get('artifacts', [])
        transformations = result.get('transformations', [])
        statistics = result.get('statistics', {})
        
        # Build nodes and edges for vis.js
        nodes_data, edges_data = self._build_vis_data(clean_dfg)
        
        # Create HTML
        html_content = self._build_html_template(
            nodes_data, edges_data, stages, artifacts, 
            transformations, statistics
        )
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Enhanced interactive HTML saved to {output_file}")
    
    def _build_vis_data(self, dfg: DataFlowGraph) -> tuple:
        """
        Build nodes and edges data for vis.js.
        
        Args:
            dfg: Data flow graph
            
        Returns:
            Tuple of (nodes_data, edges_data)
        """
        if not dfg:
            return [], []
        
        nodes_data = []
        edges_data = []
        
        # Node colors by type
        node_colors = {
            'data_artifact': '#2ECC71',
            'function_call': '#E74C3C',
            'variable': '#3498DB',
            'intermediate': '#F39C12',
            'attribute_access': '#9B59B6',
        }
        
        # Node shapes by type
        node_shapes = {
            'data_artifact': 'box',
            'function_call': 'ellipse',
            'variable': 'ellipse',
            'intermediate': 'diamond',
            'attribute_access': 'dot',
        }
        
        # Add nodes
        for node_id, node in dfg.nodes.items():
            node_type = node.node_type.value
            
            nodes_data.append({
                'id': node_id,
                'label': node.label[:20],
                'title': self._build_node_tooltip(node),
                'color': node_colors.get(node_type, '#95A5A6'),
                'shape': node_shapes.get(node_type, 'ellipse'),
                'size': self._get_node_size(node),
                'group': node_type
            })
        
        # Add edges
        for edge in dfg.edges:
            edges_data.append({
                'from': edge.from_node,
                'to': edge.to_node,
                'arrows': 'to',
                'color': {'color': '#666', 'opacity': 0.6},
                'title': edge.operation or edge.edge_type.value
            })
        
        return nodes_data, edges_data
    
    def _build_node_tooltip(self, node) -> str:
        """Build HTML tooltip for node."""
        tooltip = f"<b>{node.node_type.value}</b><br>"
        tooltip += f"{node.label}<br><br>"
        if node.code_snippet:
            snippet = node.code_snippet[:100]
            tooltip += f"<code>{snippet}</code>"
        return tooltip
    
    def _get_node_size(self, node) -> int:
        """Get node size based on type."""
        sizes = {
            'data_artifact': 30,
            'function_call': 25,
            'variable': 20,
            'intermediate': 18,
            'attribute_access': 15,
        }
        return sizes.get(node.node_type.value, 20)
    
    def _build_html_template(self, nodes_data: List, edges_data: List,
                            stages: List, artifacts: List,
                            transformations: List, statistics: Dict) -> str:
        """Build complete HTML template."""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Notebook Provenance - Interactive</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
            background: #f5f5f5; 
            color: #333;
        }}
        .header {{ 
            background: white; 
            padding: 20px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .stats {{ 
            background: white; 
            padding: 15px 20px; 
            margin-top: 2px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
        }}
        .stat {{ display: flex; flex-direction: column; }}
        .stat-label {{ color: #666; font-size: 14px; margin-bottom: 5px; }}
        .stat-value {{ color: #333; font-size: 24px; font-weight: bold; }}
        .controls {{ 
            background: white; 
            padding: 15px 20px; 
            margin-top: 2px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            display: flex; 
            gap: 15px; 
            align-items: center; 
            flex-wrap: wrap; 
        }}
        .control-group {{ display: flex; align-items: center; gap: 8px; }}
        label {{ font-weight: 500; color: #666; }}
        select, input[type="text"], button {{ 
            padding: 8px 12px; 
            border: 1px solid #ddd; 
            border-radius: 4px; 
            font-size: 14px; 
        }}
        button {{ 
            background: #3498DB; 
            color: white; 
            border: none; 
            cursor: pointer; 
            font-weight: 500; 
            transition: background 0.2s;
        }}
        button:hover {{ background: #2980B9; }}
        #mynetwork {{ 
            width: 100%; 
            height: calc(100vh - 280px); 
            border: 1px solid #ddd; 
            background: white; 
            margin-top: 2px; 
        }}
        .legend {{ 
            background: white; 
            padding: 15px 20px; 
            margin-top: 2px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }}
        .legend-items {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ 
            width: 20px; 
            height: 20px; 
            border-radius: 4px; 
            border: 2px solid #333; 
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Notebook Data Provenance</h1>
        <p>Interactive visualization of data flow and transformations</p>
    </div>
    
    <div class="stats">
        <div class="stat">
            <span class="stat-label">Pipeline Stages</span>
            <span class="stat-value">{len(stages)}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Data Artifacts</span>
            <span class="stat-value">{len(artifacts)}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Transformations</span>
            <span class="stat-value">{len(transformations)}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Graph Nodes</span>
            <span class="stat-value">{len(nodes_data)}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Noise Reduction</span>
            <span class="stat-value">{statistics.get('noise_reduction', 0):.0f}%</span>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label for="filterType">Filter by Type:</label>
            <select id="filterType">
                <option value="all">All Nodes</option>
                <option value="data_artifact">Data Artifacts Only</option>
                <option value="function_call">Functions Only</option>
                <option value="variable">Variables Only</option>
            </select>
        </div>
        <div class="control-group">
            <label for="searchNode">Search:</label>
            <input type="text" id="searchNode" placeholder="Node name..." />
        </div>
        <button onclick="network.fit()">Reset View</button>
        <button onclick="exportGraph()">Export as PNG</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
    </div>
    
    <div id="mynetwork"></div>
    
    <div class="legend">
        <h3 style="margin-bottom: 10px;">Legend</h3>
        <div class="legend-items">
            <div class="legend-item">
                <div class="legend-color" style="background: #2ECC71;"></div>
                <span>Data Artifact</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #E74C3C;"></div>
                <span>Function Call</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #3498DB;"></div>
                <span>Variable</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #F39C12;"></div>
                <span>Intermediate</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9B59B6;"></div>
                <span>Attribute Access</span>
            </div>
        </div>
    </div>
    
    <script type="text/javascript">
        // Data
        const allNodes = {json.dumps(nodes_data)};
        const allEdges = {json.dumps(edges_data)};
        
        const nodes = new vis.DataSet(allNodes);
        const edges = new vis.DataSet(allEdges);
        
        const container = document.getElementById('mynetwork');
        const data = {{ nodes: nodes, edges: edges }};
        
        let physicsEnabled = true;
        
        const options = {{
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 200,
                    springConstant: 0.08
                }},
                maxVelocity: 50,
                stabilization: {{ iterations: 150 }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }},
            nodes: {{
                font: {{ size: 14, face: 'Arial' }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                width: 2,
                shadow: true,
                smooth: {{ type: 'continuous' }}
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        
        // Filter functionality
        document.getElementById('filterType').addEventListener('change', function(e) {{
            const filterValue = e.target.value;
            if (filterValue === 'all') {{
                nodes.clear();
                nodes.add(allNodes);
                edges.clear();
                edges.add(allEdges);
            }} else {{
                const filtered = allNodes.filter(n => n.group === filterValue);
                const ids = new Set(filtered.map(n => n.id));
                const filteredEdges = allEdges.filter(e => ids.has(e.from) && ids.has(e.to));
                nodes.clear();
                nodes.add(filtered);
                edges.clear();
                edges.add(filteredEdges);
            }}
            network.fit();
        }});
        
        // Search functionality
        document.getElementById('searchNode').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            if (!searchTerm) {{
                allNodes.forEach(n => nodes.update({{ id: n.id, color: n.color, borderWidth: 2 }}));
                return;
            }}
            allNodes.forEach(n => {{
                const match = n.label.toLowerCase().includes(searchTerm);
                nodes.update({{
                    id: n.id,
                    color: match ? '#FFD700' : n.color,
                    borderWidth: match ? 4 : 2
                }});
            }});
            const m = allNodes.find(n => n.label.toLowerCase().includes(searchTerm));
            if (m) network.focus(m.id, {{ scale: 1.5, animation: true }});
        }});
        
        // Toggle physics
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}
        
        // Export as PNG
        function exportGraph() {{
            const canvas = document.querySelector('#mynetwork canvas');
            const link = document.createElement('a');
            link.download = 'provenance_graph.png';
            link.href = canvas.toDataURL();
            link.click();
        }}
    </script>
</body>
</html>
"""
        return html


__all__ = [
    "InteractiveVisualizer",
]