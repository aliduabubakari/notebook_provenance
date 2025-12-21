"""
CLI Module
==========

Command-line interface for the Notebook Provenance Analysis System.

This module provides:
- Main CLI entry point
- Argument parsing
- Command handlers for various operations
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from notebook_provenance import __version__
from notebook_provenance.core.config import ClassificationConfig


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all commands and options.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="notebook-provenance",
        description="Notebook Provenance Analysis System - Extract, analyze, and visualize data provenance from computational notebooks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a Jupyter notebook with LLM
  notebook-provenance analyze notebook.ipynb --api-key YOUR_KEY
  
  # Analyze a Python file without LLM
  notebook-provenance analyze script.py --no-llm
  
  # Custom output prefix
  notebook-provenance analyze notebook.ipynb --output my_analysis
  
  # Render notebook only (no analysis)
  notebook-provenance render notebook.ipynb --format html
  
  # Compare multiple notebooks
  notebook-provenance compare notebook1.ipynb notebook2.ipynb notebook3.ipynb
  
  # Export to Neo4j
  notebook-provenance analyze notebook.ipynb --neo4j-uri bolt://localhost:7687
        """
    )
    
    parser.add_argument(
        '--version', '-V',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ========== ANALYZE COMMAND ==========
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze a notebook and extract provenance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Add this argument to analyze_parser
    analyze_parser.add_argument(
        '--no-llm-classification',
        action='store_true',
        help='Disable LLM for artifact classification (use patterns only)'
    )
    
    analyze_parser.add_argument(
        'notebook',
        type=str,
        help='Path to notebook file (.ipynb or .py)'
    )
    
    # LLM configuration
    llm_group = analyze_parser.add_argument_group('LLM Configuration')
    llm_group.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for LLM service (or set PROVENANCE_API_KEY env var)'
    )
    llm_group.add_argument(
        '--base-url',
        type=str,
        default='https://api.deepinfra.com/v1/openai',
        help='Base URL for LLM API'
    )
    llm_group.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-Coder-480B-A35B-Instruct',
        help='LLM model to use'
    )
    llm_group.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM and use heuristics only'
    )
    
    # Output configuration
    output_group = analyze_parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file prefix (default: derived from notebook name)'
    )
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory (default: current directory)'
    )
    output_group.add_argument(
        '--format',
        choices=['all', 'json', 'html', 'png'],
        default='all',
        help='Output format (default: all)'
    )
    
    # Visualization options
    viz_group = analyze_parser.add_argument_group('Visualization Options')
    viz_group.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualization images'
    )
    viz_group.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip generating interactive HTML'
    )
    viz_group.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output images (default: 300)'
    )
    
    # Neo4j export
    neo4j_group = analyze_parser.add_argument_group('Neo4j Export')
    neo4j_group.add_argument(
        '--neo4j-uri',
        type=str,
        default=None,
        help='Neo4j database URI'
    )
    neo4j_group.add_argument(
        '--neo4j-user',
        type=str,
        default='neo4j',
        help='Neo4j username'
    )
    neo4j_group.add_argument(
        '--neo4j-password',
        type=str,
        default=None,
        help='Neo4j password'
    )
    
    # Verbosity
    analyze_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    analyze_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Increase output verbosity'
    )
    
    # ========== RENDER COMMAND ==========
    render_parser = subparsers.add_parser(
        'render',
        help='Render notebook to HTML or Markdown (no analysis)'
    )
    
    render_parser.add_argument(
        'notebook',
        type=str,
        help='Path to notebook file'
    )
    render_parser.add_argument(
        '--format',
        choices=['html', 'markdown', 'both'],
        default='html',
        help='Output format (default: html)'
    )
    render_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path'
    )
    
    # ========== COMPARE COMMAND ==========
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare multiple notebooks'
    )
    
    compare_parser.add_argument(
        'notebooks',
        nargs='+',
        help='Paths to notebook files to compare'
    )
    compare_parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for LLM service'
    )
    compare_parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM'
    )
    compare_parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory'
    )
    compare_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    # ========== INFO COMMAND ==========
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about a notebook'
    )
    
    info_parser.add_argument(
        'notebook',
        type=str,
        help='Path to notebook file'
    )
    
    return parser


def cmd_analyze(args: argparse.Namespace) -> int:
    """
    Handle the analyze command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    from notebook_provenance.orchestrator import NotebookProvenanceSystem
    from notebook_provenance.parsing.notebook_loader import NotebookLoader
    from notebook_provenance.parsing.renderer import NotebookRenderer
    
    # Print header
    if not args.quiet:
        print_header()
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        notebook_path = Path(args.notebook)
        output_prefix = notebook_path.stem + "_provenance"
    
    # Add output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / output_prefix)
    
    # Load notebook
    if not args.quiet:
        print(f"📂 Loading notebook: {args.notebook}")
    
    try:
        code_cells, cell_ids = NotebookLoader.load_notebook(args.notebook)
    except Exception as e:
        print(f"✗ Error loading notebook: {e}")
        return 1
    
    # Show summary
    if not args.quiet:
        NotebookRenderer.render_summary(code_cells, cell_ids)
    
    # Determine LLM usage
    use_llm = not args.no_llm and args.api_key is not None
    
    if not args.quiet:
        if use_llm:
            print(f"\n🤖 LLM Analysis: ENABLED (using {args.base_url})")
        else:
            print(f"\n🤖 LLM Analysis: DISABLED (using heuristics)")
            if not args.no_llm and args.api_key is None:
                print("   💡 Tip: Provide --api-key to enable LLM features")
    
    # Build configuration
    from notebook_provenance.core.config import Config, LLMConfig, VisualizationConfig
    
    config = Config(
        llm=LLMConfig(
            enabled=use_llm,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        ),
        classification=ClassificationConfig(
            use_llm=use_llm and not args.no_llm_classification,  # NEW
            use_embeddings=use_llm and not args.no_llm_classification,  # NEW
            use_semantic_deduplication=False,  # Disable for now
        ),
        visualization=VisualizationConfig(
            enabled=not args.no_visualizations,
            dpi=args.dpi,
            interactive_html=not args.no_interactive,
        ),
        output_dir=output_dir,
        verbose=not args.quiet,
    )
    
    # Initialize system and analyze
    try:
        system = NotebookProvenanceSystem(config=config)
        result = system.analyze_notebook(code_cells, cell_ids)
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Save outputs
    if not args.quiet:
        print(f"\n💾 Saving outputs to: {output_dir}")
    
    try:
        system.save_all(result, prefix=output_prefix)
    except Exception as e:
        print(f"✗ Error saving outputs: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Neo4j export if requested
    if args.neo4j_uri and args.neo4j_password:
        if not args.quiet:
            print(f"\n🔗 Exporting to Neo4j...")
        try:
            from notebook_provenance.export.neo4j_export import Neo4jExporter
            neo4j_exporter = Neo4jExporter(
                args.neo4j_uri,
                args.neo4j_user,
                args.neo4j_password
            )
            neo4j_exporter.export_to_neo4j(result)
            neo4j_exporter.close()
        except Exception as e:
            print(f"  ⚠ Neo4j export failed: {e}")
    
    # Print summary
    if not args.quiet:
        print_analysis_summary(result, output_prefix)
    
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """
    Handle the render command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    from notebook_provenance.parsing.notebook_loader import NotebookLoader
    from notebook_provenance.parsing.renderer import NotebookRenderer
    
    # Load notebook
    try:
        code_cells, cell_ids = NotebookLoader.load_notebook(args.notebook)
    except Exception as e:
        print(f"✗ Error loading notebook: {e}")
        return 1
    
    # Determine output file
    if args.output:
        output_base = args.output
    else:
        output_base = Path(args.notebook).stem + "_rendered"
    
    # Render
    try:
        if args.format in ['html', 'both']:
            output_file = f"{output_base}.html" if not args.output else args.output
            NotebookRenderer.render_to_html(code_cells, cell_ids, output_file)
        
        if args.format in ['markdown', 'both']:
            output_file = f"{output_base}.md" if not args.output else args.output
            NotebookRenderer.render_to_markdown(code_cells, cell_ids, output_file)
        
        print("✅ Rendering complete!")
        return 0
        
    except Exception as e:
        print(f"✗ Rendering failed: {e}")
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """
    Handle the compare command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    from notebook_provenance.orchestrator import NotebookProvenanceSystem
    from notebook_provenance.parsing.notebook_loader import NotebookLoader
    from notebook_provenance.export.comparison import ProvenanceComparator
    from notebook_provenance.visualization.comparison import ComparisonVisualizer
    from notebook_provenance.core.config import Config, LLMConfig
    import json
    
    notebooks = args.notebooks
    
    if len(notebooks) < 2:
        print("✗ Need at least 2 notebooks to compare")
        return 1
    
    if not args.quiet:
        print_header()
        print(f"📂 Comparing {len(notebooks)} notebooks...")
    
    # Build configuration
    use_llm = not args.no_llm and args.api_key is not None
    config = Config(
        llm=LLMConfig(
            enabled=use_llm,
            api_key=args.api_key,
        ),
        verbose=not args.quiet,
    )
    
    # Analyze each notebook
    results = []
    names = []
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    system = NotebookProvenanceSystem(config=config)
    
    for notebook_path in notebooks:
        if not args.quiet:
            print(f"\n→ Analyzing: {notebook_path}")
        
        try:
            code_cells, cell_ids = NotebookLoader.load_notebook(notebook_path)
            result = system.analyze_notebook(code_cells, cell_ids)
            results.append(result)
            names.append(Path(notebook_path).name)
            
            # Save individual results
            prefix = str(output_dir / (Path(notebook_path).stem + "_provenance"))
            system.save_all(result, prefix=prefix)
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if len(results) < 2:
        print("✗ Need at least 2 successful analyses to compare")
        return 1
    
    # Generate comparison
    if not args.quiet:
        print(f"\n📊 Generating comparison...")
    
    comparator = ProvenanceComparator()
    comparison_report = comparator.compare_multiple(results, names)
    
    # Save comparison report
    comparison_file = str(output_dir / "comparison_report.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=2, default=str)
    print(f"✓ Comparison report saved to {comparison_file}")
    
    # Generate comparison visualization
    visualizer = ComparisonVisualizer()
    visualizer.create_comparison_html(
        results, names,
        str(output_dir / "comparison.html")
    )
    
    # Print summary
    if not args.quiet:
        print_comparison_summary(comparison_report, names)
    
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """
    Handle the info command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    from notebook_provenance.parsing.notebook_loader import NotebookLoader
    from notebook_provenance.parsing.ast_parser import EnhancedCodeCellParser
    from notebook_provenance.parsing.renderer import NotebookRenderer
    
    # Load notebook
    try:
        code_cells, cell_ids = NotebookLoader.load_notebook(args.notebook)
    except Exception as e:
        print(f"✗ Error loading notebook: {e}")
        return 1
    
    # Parse cells
    parser = EnhancedCodeCellParser()
    parsed_cells = parser.parse_cells(code_cells, cell_ids)
    
    # Show summary
    NotebookRenderer.render_summary(code_cells, cell_ids)
    
    # Show parsing statistics
    stats = parser.get_statistics(parsed_cells)
    
    print("\n📊 Parsing Statistics:")
    print(f"  • Total variables defined: {stats.get('total_variables_defined', 0)}")
    print(f"  • Total function calls: {stats.get('total_function_calls', 0)}")
    print(f"  • Total imports: {stats.get('total_imports', 0)}")
    print(f"  • Average complexity: {stats.get('avg_complexity', 0):.1f}")
    print(f"  • Max complexity: {stats.get('max_complexity', 0):.1f}")
    
    return 0


def print_header():
    """Print CLI header."""
    print("\n" + "=" * 70)
    print(" " * 15 + "NOTEBOOK PROVENANCE ANALYSIS SYSTEM")
    print(" " * 20 + f"Version {__version__}")
    print("=" * 70 + "\n")


def print_analysis_summary(result: dict, output_prefix: str):
    """Print analysis summary."""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    stats = result.get('statistics', {})
    
    print(f"\n📊 Statistics:")
    print(f"  • Cells analyzed: {stats.get('total_cells', 0)}")
    print(f"  • Pipeline stages: {stats.get('stages', 0)}")
    print(f"  • Data artifacts: {stats.get('artifacts', 0)}")
    print(f"  • Transformations: {stats.get('transformations', 0)}")
    print(f"  • Graph nodes: {stats.get('clean_nodes', 0)}")
    print(f"  • Noise reduction: {stats.get('noise_reduction', 0):.1f}%")
    
    # Key artifacts
    artifacts = result.get('artifacts', [])
    if artifacts:
        print(f"\n🗂️ Key Data Artifacts:")
        for artifact in artifacts[:5]:
            print(f"  • {artifact.name} ({artifact.type}, importance: {artifact.importance_score:.1f})")
        if len(artifacts) > 5:
            print(f"  ... and {len(artifacts) - 5} more")
    
    # Pipeline stages
    stages = result.get('stages', [])
    if stages:
        print(f"\n🔄 Pipeline Stages:")
        for i, stage in enumerate(stages, 1):
            print(f"  {i}. {stage.stage_type.value.upper()}")
            print(f"     {stage.description}")
    
    # Column lineage
    col_lineage = result.get('column_lineage', {})
    if col_lineage:
        print(f"\n🧱 Column Lineage:")
        print(f"  • Created: {len(col_lineage.get('created', {}))}")
        print(f"  • Dropped: {len(col_lineage.get('dropped', {}))}")
        print(f"  • Renamed: {len(col_lineage.get('renamed', {}))}")
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print(f"\n📁 Output files:")
    print(f"  • {output_prefix}_stages.png")
    print(f"  • {output_prefix}_lineage.png")
    print(f"  • {output_prefix}_simplified_lineage.png")
    print(f"  • {output_prefix}_clean_dfg.png")
    print(f"  • {output_prefix}_complete.json")
    print(f"  • {output_prefix}_interactive.html")
    print()


def print_comparison_summary(report: dict, names: list):
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    baseline = report.get('baseline', names[0])
    print(f"\nBaseline: {baseline}")
    
    for name, comp in report.get('comparisons', {}).items():
        print(f"\n📊 Against: {name}")
        print(f"  • Added artifacts: {len(comp.get('added_artifacts', []))}")
        print(f"  • Removed artifacts: {len(comp.get('removed_artifacts', []))}")
        print(f"  • Added transformations: {comp.get('added_transformations', 0)}")
        print(f"  • Removed transformations: {comp.get('removed_transformations', 0)}")
        print(f"  • Stage sequence similarity: {comp.get('stage_sequence_similarity', 0)*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ Comparison complete!")
    print("=" * 70 + "\n")


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # If no command, print help
    if args.command is None:
        parser.print_help()
        return 0
    
    # Route to appropriate command handler
    try:
        if args.command == 'analyze':
            return cmd_analyze(args)
        elif args.command == 'render':
            return cmd_render(args)
        elif args.command == 'compare':
            return cmd_compare(args)
        elif args.command == 'info':
            return cmd_info(args)
        else:
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())