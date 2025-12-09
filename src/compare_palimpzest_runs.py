"""
Compare two Palimpzest runs side-by-side.

Usage:
    python src/compare_palimpzest_runs.py <path_to_run1.json> <path_to_run2.json> [--output-dir <dir>]

Example:
    python src/compare_palimpzest_runs.py \
        files/movie/metrics/palimpzest_baseline.json \
        files/movie/metrics/palimpzest_optimized.json \
        --output-dir figures/movie/comparison
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class PalimpzestRunComparator:
    def __init__(self, run1_path: str, run2_path: str, output_dir: str = None):
        """
        Initialize comparator with two Palimpzest run JSON files.
        
        Args:
            run1_path: Path to first run's metrics JSON
            run2_path: Path to second run's metrics JSON
            output_dir: Directory to save comparison plots
        """
        self.run1_path = Path(run1_path)
        self.run2_path = Path(run2_path)
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("figures") / "palimpzest_comparison"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        with open(self.run1_path, 'r') as f:
            self.run1_data = json.load(f)
        with open(self.run2_path, 'r') as f:
            self.run2_data = json.load(f)
        
        # Extract run names from file names
        self.run1_name = self.run1_path.stem
        self.run2_name = self.run2_path.stem
        
        # Colors for the two runs
        self.colors = {
            self.run1_name: "#1f77b4",  # Blue
            self.run2_name: "#ff7f0e",  # Orange
        }
    
    def get_common_queries(self):
        """Get queries that exist in both runs."""
        queries1 = set(self.run1_data.keys())
        queries2 = set(self.run2_data.keys())
        common = queries1 & queries2
        return sorted(common, key=lambda x: int(x.replace('Q', '')))
    
    def plot_execution_time_comparison(self):
        """Compare execution time between two runs."""
        queries = self.get_common_queries()
        
        # Separate Q7 from other queries for better visualization
        queries_without_q7 = [q for q in queries if q != 'Q7']
        has_q7 = 'Q7' in queries
        
        # Create subplots: one for all queries except Q7, one for Q7 alone
        fig, axes = plt.subplots(2 if has_q7 else 1, 1, figsize=(12, 12 if has_q7 else 6))
        if not has_q7:
            axes = [axes]
        
        # First subplot: All queries except Q7
        ax = axes[0]
        x_pos = np.arange(len(queries_without_q7))
        width = 0.35
        
        # Gather data (excluding Q7)
        times1 = [self.run1_data[q].get('execution_time', 0) for q in queries_without_q7]
        times2 = [self.run2_data[q].get('execution_time', 0) for q in queries_without_q7]
        
        # Plot bars
        bars1 = ax.bar(x_pos - width/2, times1, width, 
                      label=self.run1_name, 
                      color=self.colors[self.run1_name],
                      alpha=0.8, edgecolor='black', linewidth=1.0)
        bars2 = ax.bar(x_pos + width/2, times2, width,
                      label=self.run2_name,
                      color=self.colors[self.run2_name],
                      alpha=0.8, edgecolor='black', linewidth=1.0)
        
        # Add value labels on bars
        for bars, times in [(bars1, times1), (bars2, times2)]:
            for bar, time in zip(bars, times):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                           f'{time:.1f}s',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        title = 'Execution Time Comparison (Queries 1-10, excluding Q7)' if has_q7 else 'Execution Time Comparison'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(queries_without_q7)
        ax.legend(loc='upper right', fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        
        # Second subplot: Q7 only (if exists)
        if has_q7:
            ax = axes[1]
            x_pos_q7 = np.array([0])
            width_q7 = 0.2  # Narrower bars for Q7
            
            time1_q7 = self.run1_data['Q7'].get('execution_time', 0)
            time2_q7 = self.run2_data['Q7'].get('execution_time', 0)
            
            bars1 = ax.bar(x_pos_q7 - width_q7/2, [time1_q7], width_q7,
                          label=self.run1_name,
                          color=self.colors[self.run1_name],
                          alpha=0.8, edgecolor='black', linewidth=1.0)
            bars2 = ax.bar(x_pos_q7 + width_q7/2, [time2_q7], width_q7,
                          label=self.run2_name,
                          color=self.colors[self.run2_name],
                          alpha=0.8, edgecolor='black', linewidth=1.0)
            
            # Add value labels
            for bar, time in [(bars1[0], time1_q7), (bars2[0], time2_q7)]:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                           f'{time:.1f}s',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
            ax.set_title('Execution Time Comparison (Q7 only - note different scale)', fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(x_pos_q7)
            ax.set_xticklabels(['Q7'])
            ax.set_xlim(-0.5, 0.5)  # Limit x-axis range
            ax.legend(loc='upper right', fontsize=10, frameon=True)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved execution_time_comparison.png")
    
    def plot_cost_comparison(self):
        """Compare cost (token usage and money) between two runs."""
        queries = self.get_common_queries()
        
        # Separate Q7 from other queries
        queries_without_q7 = [q for q in queries if q != 'Q7']
        has_q7 = 'Q7' in queries
        
        # Create 4 subplots if Q7 exists: tokens (no Q7), cost (no Q7), tokens (Q7), cost (Q7)
        fig, axes = plt.subplots(4 if has_q7 else 2, 1, figsize=(12, 20 if has_q7 else 10))
        
        x_pos = np.arange(len(queries_without_q7))
        width = 0.35
        
        # Token Usage (without Q7)
        ax1 = axes[0]
        tokens1 = [self.run1_data[q].get('token_usage', 0) for q in queries_without_q7]
        tokens2 = [self.run2_data[q].get('token_usage', 0) for q in queries_without_q7]
        
        bars1 = ax1.bar(x_pos - width/2, tokens1, width,
                       label=self.run1_name,
                       color=self.colors[self.run1_name],
                       alpha=0.8, edgecolor='black', linewidth=1.0)
        bars2 = ax1.bar(x_pos + width/2, tokens2, width,
                       label=self.run2_name,
                       color=self.colors[self.run2_name],
                       alpha=0.8, edgecolor='black', linewidth=1.0)
        
        # Add labels for token usage
        for bars, tokens in [(bars1, tokens1), (bars2, tokens2)]:
            for bar, token in zip(bars, tokens):
                height = bar.get_height()
                if height > 0:
                    label = f'{token/1000:.0f}K' if token >= 1000 else str(int(token))
                    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                            label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Query ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Token Usage', fontsize=12, fontweight='bold')
        title = 'Token Usage Comparison (Queries 1-10, excluding Q7)' if has_q7 else 'Token Usage Comparison'
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(queries_without_q7)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_axisbelow(True)
        ax1.spines[['top', 'right']].set_visible(False)
        
        # Money Cost (without Q7)
        ax2 = axes[1]
        costs1 = [self.run1_data[q].get('money_cost', 0) for q in queries_without_q7]
        costs2 = [self.run2_data[q].get('money_cost', 0) for q in queries_without_q7]
        
        bars1 = ax2.bar(x_pos - width/2, costs1, width,
                       label=self.run1_name,
                       color=self.colors[self.run1_name],
                       alpha=0.8, edgecolor='black', linewidth=1.0)
        bars2 = ax2.bar(x_pos + width/2, costs2, width,
                       label=self.run2_name,
                       color=self.colors[self.run2_name],
                       alpha=0.8, edgecolor='black', linewidth=1.0)
        
        # Add labels for money cost
        for bars, costs in [(bars1, costs1), (bars2, costs2)]:
            for bar, cost in zip(bars, costs):
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                            f'${cost:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Query ID', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Money Cost ($)', fontsize=12, fontweight='bold')
        title = 'Money Cost Comparison (Queries 1-10, excluding Q7)' if has_q7 else 'Money Cost Comparison'
        ax2.set_title(title, fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(queries_without_q7)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_axisbelow(True)
        ax2.spines[['top', 'right']].set_visible(False)
        # Q7 Token Usage (if exists)
        if has_q7:
            ax3 = axes[2]
            x_pos_q7 = np.array([0])
            width_q7 = 0.2  # Narrower bars for Q7
            
            tokens1_q7 = self.run1_data['Q7'].get('token_usage', 0)
            tokens2_q7 = self.run2_data['Q7'].get('token_usage', 0)
            
            bars1 = ax3.bar(x_pos_q7 - width_q7/2, [tokens1_q7], width_q7,
                           label=self.run1_name,
                           color=self.colors[self.run1_name],
                           alpha=0.8, edgecolor='black', linewidth=1.0)
            bars2 = ax3.bar(x_pos_q7 + width_q7/2, [tokens2_q7], width_q7,
                           label=self.run2_name,
                           color=self.colors[self.run2_name],
                           alpha=0.8, edgecolor='black', linewidth=1.0)
            
            for bar, token in [(bars1[0], tokens1_q7), (bars2[0], tokens2_q7)]:
                height = bar.get_height()
                if height > 0:
                    label = f'{token/1000000:.1f}M' if token >= 1000000 else f'{token/1000:.0f}K'
                    ax3.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                            label, ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax3.set_title('Token Usage Comparison (Q7 only - note different scale)', fontsize=13, fontweight='bold')
            ax3.set_xticks(x_pos_q7)
            ax3.set_xticklabels(['Q7'])
            ax3.set_xlim(-0.5, 0.5)  # Limit x-axis range
            ax3.legend(loc='upper right', fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax3.set_axisbelow(True)
            ax3.spines[['top', 'right']].set_visible(False)
            
            # Q7 Money Cost
            ax4 = axes[3]
            
            cost1_q7 = self.run1_data['Q7'].get('money_cost', 0)
            cost2_q7 = self.run2_data['Q7'].get('money_cost', 0)
            
            bars1 = ax4.bar(x_pos_q7 - width_q7/2, [cost1_q7], width_q7,
                           label=self.run1_name,
                           color=self.colors[self.run1_name],
                           alpha=0.8, edgecolor='black', linewidth=1.0)
            bars2 = ax4.bar(x_pos_q7 + width_q7/2, [cost2_q7], width_q7,
                           label=self.run2_name,
                           color=self.colors[self.run2_name],
                           alpha=0.8, edgecolor='black', linewidth=1.0)
            
            for bar, cost in [(bars1[0], cost1_q7), (bars2[0], cost2_q7)]:
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                            f'${cost:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax4.set_xlabel('Query ID', fontsize=12, fontweight='bold')
            ax4.set_title('Money Cost Comparison (Q7 only - note different scale)', fontsize=13, fontweight='bold')
            ax4.set_xticks(x_pos_q7)
            ax4.set_xticklabels(['Q7'])
            ax4.set_xlim(-0.5, 0.5)  # Limit x-axis range
            ax4.legend(loc='upper right', fontsize=10)
            ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax4.set_axisbelow(True)
            ax4.spines[['top', 'right']].set_visible(False)
            ax4.spines[['top', 'right']].set_visible(False)
        
        plt.suptitle('Cost Comparison', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(self.output_dir / 'cost_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved cost_comparison.png")
    
    def plot_quality_comparison(self):
        """Compare quality metrics between two runs."""
        queries = self.get_common_queries()
        
        # Separate queries by metric type
        retrieval_queries = []
        aggregation_queries = []
        ranking_queries = []
        
        for q in queries:
            q1_data = self.run1_data.get(q, {})
            q2_data = self.run2_data.get(q, {})
            
            if 'f1_score' in q1_data or 'f1_score' in q2_data:
                retrieval_queries.append(q)
            elif 'relative_error' in q1_data or 'relative_error' in q2_data:
                aggregation_queries.append(q)
            elif 'spearman_correlation' in q1_data or 'spearman_correlation' in q2_data:
                ranking_queries.append(q)
        
        # Determine number of subplots needed
        n_plots = sum([len(retrieval_queries) > 0, len(aggregation_queries) > 0, len(ranking_queries) > 0])
        
        if n_plots == 0:
            print("⚠️  No quality metrics found")
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        ax_idx = 0
        
        # Plot retrieval metrics (F1 score)
        if retrieval_queries:
            ax = axes[ax_idx]
            ax_idx += 1
            
            x_pos = np.arange(len(retrieval_queries))
            width = 0.35
            
            f1_scores1 = [self.run1_data[q].get('f1_score', 0) for q in retrieval_queries]
            f1_scores2 = [self.run2_data[q].get('f1_score', 0) for q in retrieval_queries]
            
            bars1 = ax.bar(x_pos - width/2, f1_scores1, width,
                          label=self.run1_name,
                          color=self.colors[self.run1_name],
                          alpha=0.8, edgecolor='black', linewidth=1.0)
            bars2 = ax.bar(x_pos + width/2, f1_scores2, width,
                          label=self.run2_name,
                          color=self.colors[self.run2_name],
                          alpha=0.8, edgecolor='black', linewidth=1.0)
            
            # Add labels
            for bars, scores in [(bars1, f1_scores1), (bars2, f1_scores2)]:
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
            ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
            ax.set_title('Retrieval Quality (F1 Score)', fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(retrieval_queries)
            ax.set_ylim(0, 1.15)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        
        # Plot aggregation metrics (Relative Error)
        if aggregation_queries:
            ax = axes[ax_idx]
            ax_idx += 1
            
            x_pos = np.arange(len(aggregation_queries))
            width = 0.35
            
            errors1 = [self.run1_data[q].get('relative_error', 0) for q in aggregation_queries]
            errors2 = [self.run2_data[q].get('relative_error', 0) for q in aggregation_queries]
            
            bars1 = ax.bar(x_pos - width/2, errors1, width,
                          label=self.run1_name,
                          color=self.colors[self.run1_name],
                          alpha=0.8, edgecolor='black', linewidth=1.0)
            bars2 = ax.bar(x_pos + width/2, errors2, width,
                          label=self.run2_name,
                          color=self.colors[self.run2_name],
                          alpha=0.8, edgecolor='black', linewidth=1.0)
            
            # Add labels
            for bars, errors in [(bars1, errors1), (bars2, errors2)]:
                for bar, error in zip(bars, errors):
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                               f'{error:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
            ax.set_ylabel('Relative Error', fontsize=12, fontweight='bold')
            ax.set_title('Aggregation Quality (Relative Error - Lower is Better)', fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(aggregation_queries)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        
        # Plot ranking metrics (Spearman Correlation)
        if ranking_queries:
            ax = axes[ax_idx]
            
            x_pos = np.arange(len(ranking_queries))
            width = 0.35
            
            corr1 = [self.run1_data[q].get('spearman_correlation', 0) for q in ranking_queries]
            corr2 = [self.run2_data[q].get('spearman_correlation', 0) for q in ranking_queries]
            
            bars1 = ax.bar(x_pos - width/2, corr1, width,
                          label=self.run1_name,
                          color=self.colors[self.run1_name],
                          alpha=0.8, edgecolor='black', linewidth=1.0)
            bars2 = ax.bar(x_pos + width/2, corr2, width,
                          label=self.run2_name,
                          color=self.colors[self.run2_name],
                          alpha=0.8, edgecolor='black', linewidth=1.0)
            
            # Add labels
            for bars, corrs in [(bars1, corr1), (bars2, corr2)]:
                for bar, corr in zip(bars, corrs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{corr:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
            ax.set_ylabel('Spearman Correlation', fontsize=12, fontweight='bold')
            ax.set_title('Ranking Quality (Spearman Correlation)', fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(ranking_queries)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        
        plt.suptitle('Quality Comparison', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(self.output_dir / 'quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved quality_comparison.png")
    
    def plot_summary_table(self):
        """Generate a summary comparison table."""
        queries = self.get_common_queries()
        
        # Calculate totals
        total_time1 = sum(self.run1_data[q].get('execution_time', 0) for q in queries)
        total_time2 = sum(self.run2_data[q].get('execution_time', 0) for q in queries)
        
        total_tokens1 = sum(self.run1_data[q].get('token_usage', 0) for q in queries)
        total_tokens2 = sum(self.run2_data[q].get('token_usage', 0) for q in queries)
        
        total_cost1 = sum(self.run1_data[q].get('money_cost', 0) for q in queries)
        total_cost2 = sum(self.run2_data[q].get('money_cost', 0) for q in queries)
        
        # Calculate average quality scores
        f1_queries = [q for q in queries if 'f1_score' in self.run1_data.get(q, {})]
        avg_f1_1 = np.mean([self.run1_data[q]['f1_score'] for q in f1_queries]) if f1_queries else 0
        avg_f1_2 = np.mean([self.run2_data[q]['f1_score'] for q in f1_queries]) if f1_queries else 0
        
        # Print summary
        summary = f"""
{'='*80}
                    PALIMPZEST RUN COMPARISON SUMMARY
{'='*80}

Run 1: {self.run1_name}
Run 2: {self.run2_name}

Queries Compared: {len(queries)}
Query IDs: {', '.join(queries)}

{'='*80}
EXECUTION TIME
{'='*80}
{self.run1_name:30} {total_time1:>15.2f} seconds
{self.run2_name:30} {total_time2:>15.2f} seconds
Difference:                     {total_time2 - total_time1:>15.2f} seconds ({((total_time2-total_time1)/total_time1*100):+.1f}%)

{'='*80}
TOKEN USAGE
{'='*80}
{self.run1_name:30} {total_tokens1:>15,.0f} tokens
{self.run2_name:30} {total_tokens2:>15,.0f} tokens
Difference:                     {total_tokens2 - total_tokens1:>15,.0f} tokens ({((total_tokens2-total_tokens1)/total_tokens1*100):+.1f}%)

{'='*80}
MONEY COST
{'='*80}
{self.run1_name:30} ${total_cost1:>14.4f}
{self.run2_name:30} ${total_cost2:>14.4f}
Difference:                     ${total_cost2 - total_cost1:>14.4f} ({((total_cost2-total_cost1)/total_cost1*100):+.1f}%)

{'='*80}
AVERAGE QUALITY (F1 Score - {len(f1_queries)} queries)
{'='*80}
{self.run1_name:30} {avg_f1_1:>15.4f}
{self.run2_name:30} {avg_f1_2:>15.4f}
Difference:                     {avg_f1_2 - avg_f1_1:>15.4f} ({((avg_f1_2-avg_f1_1)/avg_f1_1*100 if avg_f1_1 > 0 else 0):+.1f}%)

{'='*80}
"""
        
        print(summary)
        
        # Save to file
        with open(self.output_dir / 'comparison_summary.txt', 'w') as f:
            f.write(summary)
        print(f"✅ Saved comparison_summary.txt")
    
    def generate_all_comparisons(self):
        """Generate all comparison plots."""
        print(f"\n{'='*80}")
        print(f"Comparing Palimpzest Runs")
        print(f"{'='*80}")
        print(f"Run 1: {self.run1_path}")
        print(f"Run 2: {self.run2_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"{'='*80}\n")
        
        self.plot_execution_time_comparison()
        self.plot_cost_comparison()
        self.plot_quality_comparison()
        self.plot_summary_table()
        
        print(f"\n{'='*80}")
        print(f"✅ All comparison plots generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare two Palimpzest runs side-by-side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare baseline vs optimized
  python src/compare_palimpzest_runs.py \\
      files/movie/metrics/palimpzest_baseline.json \\
      files/movie/metrics/palimpzest_optimized.json

  # Compare with custom output directory
  python src/compare_palimpzest_runs.py \\
      files/movie/metrics/palimpzest_baseline.json \\
      files/movie/metrics/palimpzest_optimized.json \\
      --output-dir figures/movie/my_comparison
        """
    )
    
    parser.add_argument('run1', help='Path to first run metrics JSON file')
    parser.add_argument('run2', help='Path to second run metrics JSON file')
    parser.add_argument('--output-dir', '-o', help='Output directory for comparison plots')
    
    args = parser.parse_args()
    
    comparator = PalimpzestRunComparator(args.run1, args.run2, args.output_dir)
    comparator.generate_all_comparisons()


if __name__ == '__main__':
    main()
