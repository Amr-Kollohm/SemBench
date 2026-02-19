"""
DocETL Query 5: Find the actor who played a role in all listed movies.

Uses ReduceOp to aggregate all movie descriptions and determine the common actor.
"""

import os
import pandas as pd
from docetl.api import (
    Pipeline, Dataset, ReduceOp, CodeMapOp, PipelineStep, PipelineOutput,
)


# Target movies for this query (same as Palimpzest runner)
TARGET_MOVIES = [
    "Love Is the Drug",
    "Crashing",
    "Cloverfield",
    "My Best Friend's Girl",
    "Hot Tub Time Machine",
    "The Last Rites of Ransom Pride",
    "Save the Date",
    "Bachelorette",
    "3, 2, 1... Frankie Go Boom",
    "Queens of Country",
    "Item 47",
    "The Night Before",
    "Now You See Me 2",
    "Allied",
    "Extinction",
    "Cobweb",
]


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q5 using DocETL: Find actor who played in all listed movies.

    Args:
        data_dir: Path to the data directory
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with actor name, cost)
    """
    # Load and filter text data to target movies
    text_df = pd.read_csv(
        os.path.join(data_dir, "lizzy_caplan_text_data.csv"),
        sep=",",
        quotechar='"',
    )
    text_df = text_df[text_df["title"].isin(TARGET_MOVIES)]

    # Add a constant group key for aggregation
    text_df["group_key"] = "all"

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q5_input.csv")
    text_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Reduce all movies into a single answer
    reduce_op = ReduceOp(
        name="find_common_actor",
        type="reduce",
        reduce_key=["group_key"],
        prompt=(
            "Who has played a role in all the movies listed below given their descriptions? "
            "Simply give the name of the actor.\n\n"
            "{% for doc in inputs %}"
            "Movie: {{ doc.title }}\n"
            "Description: {{ doc.text }}\n\n"
            "{% endfor %}"
        ),
        output={"schema": {"actor": "str"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="aggregate_movies",
        input="movies",
        operations=["find_common_actor"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q5_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q5_common_actor",
        datasets={"movies": dataset},
        operations=[reduce_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results
    results_df = pd.read_json(output_file)
    return results_df[["actor"]], cost
