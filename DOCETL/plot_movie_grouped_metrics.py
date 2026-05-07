"""Generate grouped plots for movie scenario metrics by system.

Creates time, money, and performance plots with queries on the x-axis and
system-colored bars for each query. Output is organized into subdirectories
under the selected output directory.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

IEEE_SINGLE_COL_WIDTH = 3.5
IEEE_DOUBLE_COL_WIDTH = 7.16

SYSTEM_COLORS = {
    "docetl": "#2ca02c",
    "lotus": "#ff7f0e",
    "palimpzest": "#9467bd",
}

TIME_GROUPS = {
    "q5_q6_q7": ["Q5", "Q6", "Q7"],
    "others": ["Q1", "Q2", "Q3", "Q4", "Q8", "Q9", "Q10"],
}

MONEY_GROUPS = {
    "q5_q6_q7": ["Q5", "Q6", "Q7"],
    "q10": ["Q10"],
    "others": ["Q1", "Q2", "Q3", "Q4", "Q8", "Q9"],
}

PERFORMANCE_GROUPS = {
    "f1_score": ["Q1", "Q2", "Q5", "Q6", "Q7"],
    "mean_absolute_percentage_error": ["Q3", "Q4", "Q8"],
    "spearman_correlation": ["Q9", "Q10"],
}

PERFORMANCE_LABELS = {
    "f1_score": "F1 Score",
    "mean_absolute_percentage_error": "MAPE (%)",
    "spearman_correlation": "Spearman Correlation",
}


def configure_ieee_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 0.9,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def load_metrics(input_dir: Path) -> Dict[str, Dict[str, dict]]:
    metrics = {}
    for path in sorted(input_dir.glob("*.json")):
        system_name = path.stem
        with path.open("r", encoding="utf-8") as handle:
            metrics[system_name] = json.load(handle)
    return metrics


def _numeric_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_group_average(
    metrics: Dict[str, dict],
    queries: List[str],
    metric_key: str,
) -> Optional[float]:
    values = []
    for query in queries:
        query_data = metrics.get(query)
        if not query_data:
            continue
        value = _numeric_value(query_data.get(metric_key))
        if value is None:
            continue
        values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def format_value(metric_key: str, value: float) -> str:
    if metric_key == "money_cost":
        return f"{value:.1f}"
    if metric_key == "execution_time":
        return f"{value:.3g}"
    if metric_key == "mean_absolute_percentage_error":
        return f"{value:.2f}"
    return f"{value:.3g}"


def plot_grouped_queries(
    output_path: Path,
    title: str,
    ylabel: str,
    systems: List[str],
    queries: List[str],
    values_by_system: Dict[str, Dict[str, Optional[float]]],
    figsize: tuple[float, float],
    metric_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(queries))
    group_width = 0.45 if (len(queries) == 1 and metric_key == "money_cost") else 0.7
    bar_width = group_width / max(len(systems), 1)
    offsets = (
        np.arange(len(systems)) - (len(systems) - 1) / 2
    ) * bar_width

    max_value = max(
        (
            v
            for values in values_by_system.values()
            for v in values.values()
            if v is not None
        ),
        default=0.0,
    )
    label_offset = max_value * 0.02 if max_value > 0 else 0.01

    for idx, system in enumerate(systems):
        system_values = [
            values_by_system.get(system, {}).get(query)
            for query in queries
        ]
        plotted_values = [
            value if value is not None else 0.0
            for value in system_values
        ]
        color = SYSTEM_COLORS.get(system, "#7f7f7f")

        bars = ax.bar(
            x + offsets[idx],
            plotted_values,
            width=bar_width,
            color=color,
            edgecolor="black",
            label=system,
        )

        label_rotation = (
            90
            if metric_key == "money_cost" and len(queries) > 1
            else 0
        )
        if metric_key != "money_cost":
            for bar, value in zip(bars, system_values):
                if value is None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + label_offset,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=label_rotation,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + label_offset,
                        format_value(metric_key, value),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=label_rotation,
                    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    if max_value > 0:
        ax.set_ylim(0, max_value * 1.15)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=max(len(systems), 1),
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grouped plots for movie scenario metrics."
    )
    parser.add_argument(
        "--input-dir",
        default="DOCETL",
        help="Directory containing system metrics JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots (default: <input-dir>/plots).",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Optional list of systems to include (e.g., docetl lotus palimpzest).",
    )
    parser.add_argument(
        "--ieee-column",
        choices=["single", "double"],
        default="double",
        help="IEEE column width to target for figures.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = (
        Path(args.output_dir) if args.output_dir else input_dir / "plots"
    )

    metrics_by_system = load_metrics(input_dir)
    systems = sorted(metrics_by_system.keys())
    if args.systems:
        systems = [s for s in systems if s in args.systems]

    if not systems:
        raise ValueError("No systems found to plot.")

    configure_ieee_style()

    if args.ieee_column == "single":
        fig_width = IEEE_SINGLE_COL_WIDTH
        fig_height = 2.4
    else:
        fig_width = IEEE_DOUBLE_COL_WIDTH
        fig_height = 3.0

    figsize = (fig_width, fig_height)

    # Time plots
    for group_name, queries in TIME_GROUPS.items():
        values_by_system = {
            system: {
                query: _numeric_value(
                    metrics_by_system[system].get(query, {}).get(
                        "execution_time"
                    )
                )
                for query in queries
            }
            for system in systems
        }
        plot_grouped_queries(
            output_dir / "time" / f"time_{group_name}.png",
            title=f"Movie scenario execution time ({', '.join(queries)})",
            ylabel="Execution Time (s)",
            systems=systems,
            queries=queries,
            values_by_system=values_by_system,
            figsize=figsize,
            metric_key="execution_time",
        )

    # Money plots
    for group_name, queries in MONEY_GROUPS.items():
        values_by_system = {
            system: {
                query: _numeric_value(
                    metrics_by_system[system].get(query, {}).get(
                        "money_cost"
                    )
                )
                for query in queries
            }
            for system in systems
        }
        plot_grouped_queries(
            output_dir / "money" / f"money_{group_name}.png",
            title=f"Movie scenario money cost ({', '.join(queries)})",
            ylabel="Money Cost (USD)",
            systems=systems,
            queries=queries,
            values_by_system=values_by_system,
            figsize=figsize,
            metric_key="money_cost",
        )

    # Performance plots
    for metric_key, queries in PERFORMANCE_GROUPS.items():
        values_by_system = {
            system: {
                query: _numeric_value(
                    metrics_by_system[system].get(query, {}).get(metric_key)
                )
                for query in queries
            }
            for system in systems
        }
        plot_grouped_queries(
            output_dir / "performance" / f"performance_{metric_key}.png",
            title=f"Movie scenario performance ({', '.join(queries)})",
            ylabel=PERFORMANCE_LABELS[metric_key],
            systems=systems,
            queries=queries,
            values_by_system=values_by_system,
            figsize=figsize,
            metric_key=metric_key,
        )


if __name__ == "__main__":
    main()
