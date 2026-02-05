"""
DocETL Query 1: Find clearly positive movie reviews

This query demonstrates a basic semantic filter operation using DocETL.
It filters movie reviews to find clearly positive ones and returns the top 5.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, FilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q1 using DocETL: Find 5 clearly positive movie reviews.
    
    Args:
        data_dir: Path to the data directory containing Reviews.csv
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)
    
    Returns:
        Tuple of (DataFrame with 5 clearly positive review IDs, cost)
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
    
    # Define the filter operation to identify positive reviews
    # Similar to Palimpzest's sem_filter approach
    # limit=5 will stop after finding 5 positive reviews (early stopping)
    filter_op = FilterOp(
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
        name="filter_positive",
        input="reviews",
        operations=["filter_positive_reviews"]
    )
    
    # Define output
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    output_file = os.path.join(intermediate_dir, "q1_results.json")
    
    output = PipelineOutput(
        type="file",
        path=output_file,
        intermediate_dir=intermediate_dir
    )
    
    # Create and run pipeline
    # Similar to Palimpzest: config from runner, not hardcoded
    pipeline = Pipeline(
        name="q1_positive_reviews",
        datasets={"reviews": dataset},
        operations=[filter_op],
        steps=[step],
        output=output,
        default_model=model_name,  # Use model from runner configuration
    )
    
    # Run the pipeline with max_threads (similar to Palimpzest's concurrent workers)
    cost = pipeline.run(max_threads=max_threads)
    
    # Load results - FilterOp with limit=5 already returned exactly 5 positive reviews
    results_df = pd.read_json(output_file)
    
    # Return just the reviewId as a DataFrame (matching Palimpzest's .project(["reviewId"]))
    return results_df[['reviewId']], cost