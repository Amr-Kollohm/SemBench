"""
DocETL Query 2: Find five positive reviews for movie "taken_3"

This query filters by movie ID and then finds positive reviews.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, FilterOp, CodeFilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q2 using DocETL: Find 5 positive reviews for taken_3.
    
    Args:
        data_dir: Path to the data directory containing Reviews.csv
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)
    
    Returns:
        Tuple of (DataFrame with 5 positive review IDs for taken_3, cost)
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
    
    # Filter by movie ID (similar to Palimpzest's lambda filter)
    movie_filter = CodeFilterOp(
        name="filter_by_movie",
        type="code_filter",
        code="def transform(doc):\n    return doc.get('id') == 'taken_3'"
    )
    
    # Filter for positive reviews
    positive_filter = FilterOp(
        name="filter_positive_reviews",
        type="filter",
        limit=5,
        prompt="Determine if the following movie review is clearly positive.\n\nReview: {{ input.reviewText }}",
        output={
            "schema": {
                "is_positive": "bool"
            }
        }
    )
    
    # Define pipeline step
    step = PipelineStep(
        name="find_positive_reviews",
        input="reviews",
        operations=["filter_by_movie", "filter_positive_reviews"]
    )
    
    # Define output
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    output_file = os.path.join(intermediate_dir, "q2_results.json")
    
    output = PipelineOutput(
        type="file",
        path=output_file,
        intermediate_dir=intermediate_dir
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        name="q2_positive_reviews_taken_3",
        datasets={"reviews": dataset},
        operations=[movie_filter, positive_filter],
        steps=[step],
        output=output,
        default_model=model_name,
    )
    
    cost = pipeline.run(max_threads=max_threads)
    
    # Load results
    results_df = pd.read_json(output_file)
    
    # Return just the reviewId
    return results_df[['reviewId']], cost
