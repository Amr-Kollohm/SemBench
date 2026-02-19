"""
DocETL Query 1: Find the director of the movie where Ben Piazza plays Bob Whitewood.

Joins table data with text data, extracts director from movie description,
then filters by role.
"""

import os
import pandas as pd
from docetl.api import (
    Pipeline, Dataset, MapOp, CodeFilterOp, PipelineStep, PipelineOutput,
)


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q1 using DocETL: Extract director for Ben Piazza's Bob Whitewood role.

    Args:
        data_dir: Path to the data directory
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with director name, cost)
    """
    # Load and join table + text data (same as Palimpzest runner)
    table_df = pd.read_csv(os.path.join(data_dir, "ben_piazza.csv"))
    text_df = pd.read_csv(os.path.join(data_dir, "ben_piazza_text_data.csv"))
    joined_df = table_df.merge(text_df, left_on="Title", right_on="title", how="left")
    joined_df = joined_df.fillna("")

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q1_input.csv")
    joined_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Filter by role first (non-semantic, code filter)
    role_filter = CodeFilterOp(
        name="filter_by_role",
        type="code_filter",
        code="def transform(doc):\n    return doc.get('Role') == 'Bob Whitewood'",
    )

    # Semantic map: extract director from description
    extract_director = MapOp(
        name="extract_director",
        type="map",
        prompt=(
            "Extract the director name from the movie description.\n\n"
            "Movie description: {{ input.text }}"
        ),
        output={"schema": {"director": "str"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="find_director",
        input="movies",
        operations=["filter_by_role", "extract_director"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q1_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q1_director",
        datasets={"movies": dataset},
        operations=[role_filter, extract_director],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results
    results_df = pd.read_json(output_file)
    return results_df[["director"]], cost
