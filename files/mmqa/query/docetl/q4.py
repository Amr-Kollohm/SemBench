"""
DocETL Query 4: Categorize movies by genre.

Semantic map to extract genres for each movie, then pandas post-processing
to pivot into genre → movies table.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput


# Target movies for this query (same as Palimpzest runner)
TARGET_MOVIES = [
    "Orange County",
    "Mean Girls",
    "Love Is the Drug",
    "Crashing",
    "Cloverfield",
    "My Best Friend's Girl",
    "Crossing Over",
    "Hot Tub Time Machine",
    "The Last Rites of Ransom Pride",
    "127 Hours",
    "High Road",
    "Save the Date",
    "Bachelorette",
    "3, 2, 1... Frankie Go Boom",
    "Queens of Country",
    "Item 47",
    "The Interview",
    "The Night Before",
    "Now You See Me 2",
    "Allied",
    "The Disaster Artist",
    "Extinction",
    "The People We Hate at the Wedding",
    "Cobweb",
]


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q4 using DocETL: Categorize movies by genre.

    Args:
        data_dir: Path to the data directory
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with genre and movies_in_genre, cost)
    """
    # Load and filter text data to target movies
    text_df = pd.read_csv(os.path.join(data_dir, "lizzy_caplan_text_data.csv"))
    text_df = text_df[text_df["title"].isin(TARGET_MOVIES)]

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q4_input.csv")
    text_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Semantic map: extract genres
    map_op = MapOp(
        name="extract_genres",
        type="map",
        prompt=(
            "What are the genres of this movie? Return the genres separated by commas.\n\n"
            "Movie description: {{ input.text }}"
        ),
        output={"schema": {"genres": "str"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="map_genres",
        input="movies",
        operations=["extract_genres"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q4_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q4_genres",
        datasets={"movies": dataset},
        operations=[map_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results and pivot into genre → movies table
    output_df = pd.read_json(output_file)

    expanded_data = []
    for _, row in output_df.iterrows():
        movie_title = row["title"]
        genres = []
        if isinstance(row["genres"], str):
            genres = [genre.lower().strip() for genre in row["genres"].split(",")]
        for genre in genres:
            expanded_data.append({"genre": genre, "title": movie_title})

    df_expanded = pd.DataFrame(expanded_data)
    genre_movies_table = (
        df_expanded.groupby("genre")["title"]
        .apply(lambda x: ", ".join(x))
        .reset_index()
    )
    genre_movies_table.rename(columns={"title": "movies_in_genre"}, inplace=True)

    return genre_movies_table, cost
