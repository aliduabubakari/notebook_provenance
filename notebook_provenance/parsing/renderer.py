"""
Notebook Renderer Module
========================

Render notebook cells for visualization and export.

This module provides:
- NotebookRenderer: Render notebooks to Markdown and HTML
- Syntax highlighting support (with pygments)
- Summary statistics
"""

from pathlib import Path
from typing import List, Optional

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    print("Warning: pygments not installed. Syntax highlighting disabled.")
    print("Install with: pip install pygments")


class NotebookRenderer:
    """
    Render notebook cells for visualization and export.
    
    Supports rendering to:
    - Markdown (.md)
    - HTML (.html) with optional syntax highlighting
    
    Example:
        >>> renderer = NotebookRenderer()
        >>> renderer.render_to_html(code_cells, cell_ids, "output.html")
    """
    
    @staticmethod
    def render_to_markdown(code_cells: List[str], 
                          cell_ids: List[str],
                          output_file: str = "notebook_rendered.md") -> str:
        """
        Render notebook as Markdown.
        
        Args:
            code_cells: List of code cell contents
            cell_ids: List of cell identifiers
            output_file: Output file path
            
        Returns:
            Markdown content as string
        """
        lines = []
        
        # Header
        lines.append("# Notebook Code Cells\n")
        lines.append(f"*Total cells: {len(code_cells)}*\n")
        lines.append("---\n")
        
        # Render each cell
        for i, (code, cell_id) in enumerate(zip(code_cells, cell_ids), 1):
            lines.append(f"## Cell {i}: `{cell_id}`\n")
            lines.append("```python")
            lines.append(code)
            lines.append("```\n")
            lines.append("---\n")
        
        content = '\n'.join(lines)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Rendered notebook to {output_file}")
        return content
    
    @staticmethod
    def render_to_html(code_cells: List[str],
                      cell_ids: List[str],
                      output_file: str = "notebook_rendered.html",
                      title: str = "Notebook Rendering") -> str:
        """
        Render notebook as HTML.
        
        Args:
            code_cells: List of code cell contents
            cell_ids: List of cell identifiers
            output_file: Output file path
            title: Page title
            
        Returns:
            HTML content as string
        """
        html_parts = []
        
        # HTML header
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .stats {{
            color: #666;
            font-size: 14px;
        }}
        .cell {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .cell-header {{
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .cell-id {{
            font-family: 'Monaco', 'Menlo', monospace;
            background: #f0f0f0;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        .cell-number {{
            font-weight: bold;
            color: #3498db;
        }}
        .cell-code {{
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            background: #f8f8f8;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            line-height: 1.5;
        }}
        pre {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
""")
        
        # Add pygments CSS if available
        if PYGMENTS_AVAILABLE:
            formatter = HtmlFormatter(style='default')
            html_parts.append(f"<style>{formatter.get_style_defs('.highlight')}</style>")
        
        html_parts.append("""
    </style>
</head>
<body>
    <div class="header">
        <h1>""" + title + """</h1>
        <div class="stats">
            <p><strong>Total cells:</strong> """ + str(len(code_cells)) + """</p>
            <p><strong>Total lines:</strong> """ + str(sum(len(c.split('\n')) for c in code_cells)) + """</p>
        </div>
    </div>
""")
        
        # Render each cell
        for i, (code, cell_id) in enumerate(zip(code_cells, cell_ids), 1):
            html_parts.append('<div class="cell">')
            html_parts.append('<div class="cell-header">')
            html_parts.append(f'<span><span class="cell-number">Cell {i}:</span> <span class="cell-id">{cell_id}</span></span>')
            html_parts.append(f'<span style="color: #999; font-size: 12px;">{len(code.split(chr(10)))} lines</span>')
            html_parts.append('</div>')
            html_parts.append('<div class="cell-code">')
            
            # Syntax highlighting
            if PYGMENTS_AVAILABLE:
                try:
                    highlighted = highlight(code, PythonLexer(), formatter)
                    html_parts.append(highlighted)
                except:
                    # Fall back to plain text if highlighting fails
                    html_parts.append(f'<pre>{NotebookRenderer._escape_html(code)}</pre>')
            else:
                html_parts.append(f'<pre>{NotebookRenderer._escape_html(code)}</pre>')
            
            html_parts.append('</div>')
            html_parts.append('</div>')
        
        # HTML footer
        html_parts.append("""
</body>
</html>
""")
        
        html_content = '\n'.join(html_parts)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Rendered notebook to {output_file}")
        return html_content
    
    @staticmethod
    def render_summary(code_cells: List[str], 
                      cell_ids: List[str],
                      max_preview_cells: int = 5) -> str:
        """
        Print summary of notebook structure.
        
        Args:
            code_cells: List of code cell contents
            cell_ids: List of cell identifiers
            max_preview_cells: Maximum cells to preview
            
        Returns:
            Summary string
        """
        lines = []
        
        lines.append("\n" + "=" * 80)
        lines.append("NOTEBOOK STRUCTURE SUMMARY")
        lines.append("=" * 80)
        
        # Basic statistics
        total_lines = sum(len(cell.split('\n')) for cell in code_cells)
        avg_lines = total_lines / len(code_cells) if code_cells else 0
        
        lines.append(f"\n📊 Basic Statistics:")
        lines.append(f"  • Total cells: {len(code_cells)}")
        lines.append(f"  • Total lines: {total_lines}")
        lines.append(f"  • Avg lines per cell: {avg_lines:.1f}")
        
        # Cell preview
        lines.append(f"\n📝 Cell Preview:")
        preview_cells = min(max_preview_cells, len(code_cells))
        for i in range(preview_cells):
            code = code_cells[i]
            cell_id = cell_ids[i]
            cell_lines = code.split('\n')
            first_line = cell_lines[0][:60] + '...' if len(cell_lines[0]) > 60 else cell_lines[0]
            lines.append(f"  {i+1}. {cell_id}: {first_line} ({len(cell_lines)} lines)")
        
        if len(code_cells) > max_preview_cells:
            lines.append(f"  ... and {len(code_cells) - max_preview_cells} more cells")
        
        lines.append("\n" + "=" * 80 + "\n")
        
        summary = '\n'.join(lines)
        print(summary)
        return summary
    
    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))


__all__ = [
    "NotebookRenderer",
]