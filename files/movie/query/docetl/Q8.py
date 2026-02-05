"""
DocETL Query 8: Calculate the number of positive and negative reviews for movie "taken_3"

This query groups reviews by sentiment and counts each group.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, MapOp, CodeFilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q8 using DocETL: Count positive and negative reviews for taken_3.
    
    Args:
        data_dir: Path to the data directory containing Reviews.csv
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)
    
    Returns:
        Tuple of (DataFrame with sentiment counts, cost)
    """
    # Parameters are passed directly from the runner
    # No fallback needed since runner always provides these values
    
    # Define input dataset
    input_file = os.path.join(data_dir, 'Reviews.csv')
    
    dataset = Dataset(
        type="file",
        path=input_file,
        source="csv"
    )
    
    # Filter by movie ID
    movie_filter = CodeFilterOp(
        name="filter_by_movie",
        type="code_filter",
        code="def transform(doc):\n    return doc.get('id') == 'taken_3'"
    )
    
    # Add sentiment column
    add_sentiment = MapOp(
        name="add_sentiment",
        type="map",
        prompt="Return POSITIVE if the following review is positive, and NEGATIVE if the review is not positive. Only output POSITIVE or NEGATIVE with no additional commentary\n\nReview: {{ input.reviewText }}",
        output={
            "schema": {
                "sentiment": "str"
            }
        }
    )
    
    # Define pipeline step (no LLM-powered groupby, we'll use pandas for grouping/counting)
    step = PipelineStep(
        name="count_sentiments",
        input="reviews",
        operations=["filter_by_movie", "add_sentiment"]
    )
    
    # Define output
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    output_file = os.path.join(intermediate_dir, "q8_results.json")
    
    output = PipelineOutput(
        type="file",
        path=output_file,
        intermediate_dir=intermediate_dir
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        name="q8_sentiment_counts",
        datasets={"reviews": dataset},
        operations=[movie_filter, add_sentiment],
        steps=[step],
        output=output,
        default_model=model_name,
    )
    
    cost = pipeline.run(max_threads=max_threads)
    
    # Load results and group by sentiment using pandas
    results_df = pd.read_json(output_file)
    
    # Group by sentiment and count
    sentiment_counts = results_df.groupby('sentiment').size().reset_index(name='count_sentiment')
    
    return sentiment_counts[['sentiment', 'count_sentiment']], cost
