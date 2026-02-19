"""
DocETL Query 6b: Find airlines with flights to Germany.

Semantic filter on table data (Airlines + Destinations columns).
Requires world knowledge to know which destinations are in Germany.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, FilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q6b using DocETL: Find airlines flying to Germany.

    Args:
        data_dir: Path to the data directory
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with airline names, cost)
    """
    # Load table data
    table_df = pd.read_csv(os.path.join(data_dir, "tampa_international_airport.csv"))

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q6b_input.csv")
    table_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Semantic filter: does this airline fly to Germany?
    filter_op = FilterOp(
        name="filter_germany",
        type="filter",
        prompt=(
            "Given destinations of an airline, the airline has flights to Germany.\n\n"
            "Airline: {{ input.Airlines }}\n"
            "Destinations: {{ input.Destinations }}"
        ),
        output={"schema": {"flies_to_germany": "bool"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="filter_airlines",
        input="airlines",
        operations=["filter_germany"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q6b_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q6b_germany_airlines",
        datasets={"airlines": dataset},
        operations=[filter_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results
    results_df = pd.read_json(output_file)
    return results_df[["Airlines"]], cost
