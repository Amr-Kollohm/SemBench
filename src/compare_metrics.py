"""
Compare metrics between two benchmark runs.

This script takes two metric JSON files and creates comparison graphs
for various metrics like execution time, tokens, cost, energy, etc.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# Define metrics to compare with their display names and units
METRIC_DEFINITIONS = {
    'execution_time': {'label': 'Execution Time', 'unit': 'seconds'},
    'token_usage': {'label': 'Token Usage', 'unit': 'tokens'},
    'money_cost': {'label': 'Cost', 'unit': 'dollars'},
    'energy_consumed': {'label': 'Energy Consumed', 'unit': 'kWh'},
    'carbon_produced': {'label': 'Carbon Produced', 'unit': 'kg CO2'},
    'water_consumed': {'label': 'Water Consumed', 'unit': 'liters'},
    'accuracy': {'label': 'Accuracy', 'unit': 'score'},
    'precision': {'label': 'Precision', 'unit': 'score'},
    'recall': {'label': 'Recall', 'unit': 'score'},
    'f1_score': {'label': 'F1 Score', 'unit': 'score'},
    'absolute_error': {'label': 'Absolute Error', 'unit': 'error'},
    'relative_error': {'label': 'Relative Error', 'unit': 'error'},
    'mean_absolute_percentage_error': {'label': 'MAPE', 'unit': '%'},
    'spearman_correlation': {'label': 'Spearman Correlation', 'unit': 'coefficient'},
    'kendall_tau': {'label': 'Kendall Tau', 'unit': 'coefficient'},
    'adjusted_rand_index': {'label': 'Adjusted Rand Index', 'unit': 'score'},
    'omega_index': {'label': 'Omega Index', 'unit': 'score'},
}

# Group metrics by category
METRIC_GROUPS = {
    'performance': {
        'title': 'Performance Metrics',
        'metrics': ['precision', 'recall', 'f1_score', 'accuracy', 'absolute_error', 'relative_error', 'mean_absolute_percentage_error'],
    },
    'cost': {
        'title': 'Cost & Resource Metrics',
        'metrics': ['execution_time', 'money_cost', 'token_usage'],
    },
    'environmental': {
        'title': 'Environmental Metrics',
        'metrics': ['water_consumed', 'energy_consumed', 'carbon_produced'],
    },
    'ranking': {
        'title': 'Ranking & Clustering Metrics',
        'metrics': ['spearman_correlation', 'kendall_tau', 'adjusted_rand_index', 'omega_index'],
    },
}


def load_metrics(file_path: str) -> Dict:
    """Load metrics from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_metric_values(metrics1: Dict, metrics2: Dict, metric_name: str) -> Tuple[List, List, List]:
    """
    Extract metric values from two metric dictionaries.
    
    Returns:
        Tuple of (query_ids, values1, values2)
    """
    query_ids = []
    values1 = []
    values2 = []
    
    # Get all query IDs from both files
    all_queries = set(metrics1.keys()) | set(metrics2.keys())
    
    # Handle special case for clustering metrics stored as accuracy with metric_type
    metric_type_map = {
        'adjusted_rand_index': 'adjusted-rand-index',
        'omega_index': 'omega-index'
    }
    
    for query_id in sorted(all_queries):
        # Only include if both have the metric
        if query_id in metrics1 and query_id in metrics2:
            val1 = None
            val2 = None
            
            # Check if this is a clustering metric stored with metric_type
            if metric_name in metric_type_map:
                expected_type = metric_type_map[metric_name]
                if (metrics1[query_id].get('metric_type') == expected_type and 
                    'accuracy' in metrics1[query_id]):
                    val1 = metrics1[query_id]['accuracy']
                if (metrics2[query_id].get('metric_type') == expected_type and 
                    'accuracy' in metrics2[query_id]):
                    val2 = metrics2[query_id]['accuracy']
            # Regular metric extraction
            elif metric_name in metrics1[query_id] and metric_name in metrics2[query_id]:
                val1 = metrics1[query_id][metric_name]
                val2 = metrics2[query_id][metric_name]
            
            # Skip None values
            if val1 is not None and val2 is not None:
                query_ids.append(query_id)
                values1.append(val1)
                values2.append(val2)
    
    return query_ids, values1, values2


def create_comparison_graph(
    query_ids: List[str],
    values1: List[float],
    values2: List[float],
    metric_name: str,
    label1: str,
    label2: str,
    output_path: Path
):
    """Create a comparison bar graph for a specific metric."""
    if not query_ids:
        print(f"Skipping {metric_name}: No common data found")
        return
    
    metric_def = METRIC_DEFINITIONS.get(metric_name, {'label': metric_name, 'unit': ''})
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Position of bars on x-axis
    x = np.arange(len(query_ids))
    width = 0.25  # Reduced from 0.35 to make bars smaller
    
    # Create bars
    bars1 = ax.bar(x - width/2, values1, width, label=label1, alpha=0.8)
    bars2 = ax.bar(x + width/2, values2, width, label=label2, alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Query', fontsize=12, fontweight='bold')
    ax.set_ylabel(f"{metric_def['label']} ({metric_def['unit']})", fontsize=12, fontweight='bold')
    ax.set_title(f"{metric_def['label']} Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(query_ids)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2e}' if height > 1000 or height < 0.01 else f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_grouped_comparison(
    metrics1: Dict,
    metrics2: Dict,
    group_name: str,
    group_config: Dict,
    label1: str,
    label2: str,
    output_path: Path
):
    """Create a grouped comparison graph with multiple metrics in subplots."""
    metrics_in_group = group_config['metrics']
    
    # Find which metrics actually have data
    available_metrics = []
    metric_data = {}
    
    for metric_name in metrics_in_group:
        query_ids, values1, values2 = extract_metric_values(metrics1, metrics2, metric_name)
        if query_ids:
            available_metrics.append(metric_name)
            metric_data[metric_name] = (query_ids, values1, values2)
    
    if not available_metrics:
        print(f"Skipping {group_name}: No common data found")
        return
    
    # Create subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric in its subplot
    for idx, metric_name in enumerate(available_metrics):
        ax = axes[idx]
        query_ids, values1, values2 = metric_data[metric_name]
        metric_def = METRIC_DEFINITIONS.get(metric_name, {'label': metric_name, 'unit': ''})
        
        # Position of bars on x-axis
        x = np.arange(len(query_ids))
        width = 0.25  # Smaller bar width
        
        # Create bars
        bars1 = ax.bar(x - width/2, values1, width, label=label1, alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x + width/2, values2, width, label=label2, alpha=0.8, color='#ff7f0e')
        
        # Add labels and title
        ax.set_xlabel('Query', fontsize=11, fontweight='bold')
        ax.set_ylabel(f"{metric_def['label']}", fontsize=11, fontweight='bold')
        ax.set_title(metric_def['label'], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(query_ids, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Use log scale for certain metrics if values vary widely
        if metric_name in ['execution_time', 'token_usage', 'money_cost', 'absolute_error', 'relative_error']:
            # Check if log scale is appropriate
            all_vals = values1 + values2
            if max(all_vals) / min([v for v in all_vals if v > 0], default=1) > 100:
                ax.set_yscale('log')
    
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=2, frameon=False, fontsize=11)
    
    # Add overall title
    fig.suptitle(group_config['title'], fontsize=14, fontweight='bold', y=0.02)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved grouped graph: {output_path}")


def create_summary_table(metrics1: Dict, metrics2: Dict, label1: str, label2: str, output_path: Path):
    """Create a summary comparison table."""
    with open(output_path, 'w') as f:
        f.write(f"# Metrics Comparison Summary\n\n")
        f.write(f"Comparing: {label1} vs {label2}\n\n")
        
        # Get all queries
        all_queries = sorted(set(metrics1.keys()) | set(metrics2.keys()))
        
        for query_id in all_queries:
            f.write(f"\n## {query_id}\n\n")
            f.write("| Metric | {} | {} | Difference |\n".format(label1, label2))
            f.write("|--------|----------|----------|------------|\n")
            
            # Get all metrics for this query
            all_metrics = set()
            if query_id in metrics1:
                all_metrics.update(metrics1[query_id].keys())
            if query_id in metrics2:
                all_metrics.update(metrics2[query_id].keys())
            
            for metric in sorted(all_metrics):
                if metric in ['query_id', 'status', 'available_models', 'concurrent_llm_worker', 'row_count', 'metric_type']:
                    continue
                    
                val1 = metrics1.get(query_id, {}).get(metric, 'N/A')
                val2 = metrics2.get(query_id, {}).get(metric, 'N/A')
                
                # Calculate difference if both are numeric
                diff = 'N/A'
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff_val = val2 - val1
                    pct_change = (diff_val / val1 * 100) if val1 != 0 else 0
                    diff = f"{diff_val:+.2f} ({pct_change:+.1f}%)"
                
                # Format values
                val1_str = f"{val1:.4f}" if isinstance(val1, float) else str(val1)
                val2_str = f"{val2:.4f}" if isinstance(val2, float) else str(val2)
                
                f.write(f"| {metric} | {val1_str} | {val2_str} | {diff} |\n")
    
    print(f"Saved summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics between two benchmark runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two palimpzest runs
  python src/compare_metrics.py \\
      files/movie/metrics/palimpzest_run1.json \\
      files/movie/metrics/palimpzest_run2.json \\
      --output comparison_graphs

  # Compare with custom labels
  python src/compare_metrics.py \\
      files/ecomm/metrics/palimpzest.json \\
      files/ecomm/metrics/lotus.json \\
      --label1 "Palimpzest" --label2 "LOTUS" \\
      --output ecomm_comparison
        """
    )
    
    parser.add_argument(
        'file1',
        help='Path to first metrics JSON file'
    )
    
    parser.add_argument(
        'file2',
        help='Path to second metrics JSON file'
    )
    
    parser.add_argument(
        '--output',
        default='comparison',
        help='Output directory name for graphs (default: comparison)'
    )
    
    parser.add_argument(
        '--label1',
        help='Label for first dataset (default: derived from filename)'
    )
    
    parser.add_argument(
        '--label2',
        help='Label for second dataset (default: derived from filename)'
    )
    
    args = parser.parse_args()
    
    # Load metrics
    print(f"Loading metrics from:\n  {args.file1}\n  {args.file2}")
    metrics1 = load_metrics(args.file1)
    metrics2 = load_metrics(args.file2)
    
    # Determine labels
    label1 = args.label1 or Path(args.file1).stem
    label2 = args.label2 or Path(args.file2).stem
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving graphs to: {output_dir}")
    
    # Create grouped comparison graphs
    print("\nGenerating grouped comparison graphs...")
    for group_name, group_config in METRIC_GROUPS.items():
        output_path = output_dir / f"{group_name}_metrics_comparison.png"
        create_grouped_comparison(
            metrics1, metrics2, group_name, group_config,
            label1, label2, output_path
        )
    
    # Also create individual graphs for reference (optional)
    print("\nGenerating individual comparison graphs...")
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)
    
    for metric_name in METRIC_DEFINITIONS.keys():
        query_ids, values1, values2 = extract_metric_values(metrics1, metrics2, metric_name)
        
        if query_ids:
            output_path = individual_dir / f"{metric_name}_comparison.png"
            create_comparison_graph(
                query_ids, values1, values2, metric_name,
                label1, label2, output_path
            )
    
    # Create summary table
    summary_path = output_dir / "comparison_summary.md"
    create_summary_table(metrics1, metrics2, label1, label2, summary_path)
    
    print(f"\n✓ Comparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
