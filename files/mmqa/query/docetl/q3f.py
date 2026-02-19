"""
DocETL Query 3f: Find romantic comedy movies from Lizzy Caplan's filmography.

Semantic filter on text data to identify romantic comedy movies.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, FilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q3f using DocETL: Filter for romantic comedy movies.

    Args:
        data_dir: Path to the data directory
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with romantic comedy movie titles, cost)
    """
    # Load text data
    text_df = pd.read_csv(os.path.join(data_dir, "lizzy_caplan_text_data.csv"))

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q3f_input.csv")
    text_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Semantic filter: is this a romantic comedy?
    filter_op = FilterOp(
        name="filter_romcom",
        type="filter",
        prompt=(
            "Determine if a movie is a romantic comedy given their description.\n\n"
            "Movie title: {{ input.title }}\n"
            "Movie description: {{ input.text }}"
        ),
        output={"schema": {"is_romcom": "bool"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="filter_romcoms",
        input="movies",
        operations=["filter_romcom"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q3f_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q3f_romcom_movies",
        datasets={"movies": dataset},
        operations=[filter_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results
    results_df = pd.read_json(output_file)
    return results_df[["title"]], cost
