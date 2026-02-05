"""
DocETL Query 3: Count of positive reviews for movie "taken_3"

This query filters by movie ID, finds positive reviews, and counts them.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, FilterOp, CodeFilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q3 using DocETL: Count positive reviews for taken_3.
    
    Args:
        data_dir: Path to the data directory containing Reviews.csv
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)
    
    Returns:
        Tuple of (DataFrame with count of positive reviews, cost)
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
    
    # Filter for positive reviews
    positive_filter = FilterOp(
        name="filter_positive_reviews",
        type="filter",
        prompt="Determine if the following movie review is clearly positive.\n\nReview: {{ input.reviewText }}",
        output={
            "schema": {
                "is_positive": "bool"
            }
        }
    )
    
    # Define pipeline step (no LLM-powered count, we'll use pandas for counting)
    step = PipelineStep(
        name="count_positive_reviews",
        input="reviews",
        operations=["filter_by_movie", "filter_positive_reviews"]
    )
    
    # Define output
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    output_file = os.path.join(intermediate_dir, "q3_results.json")
    
    output = PipelineOutput(
        type="file",
        path=output_file,
        intermediate_dir=intermediate_dir
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        name="q3_count_positive_reviews",
        datasets={"reviews": dataset},
        operations=[movie_filter, positive_filter],
        steps=[step],
        output=output,
        default_model=model_name,
    )
    
    cost = pipeline.run(max_threads=max_threads)
    
    # Load results and count using pandas
    results_df = pd.read_json(output_file)
    
    # Count the positive reviews (simple row count)
    positive_review_cnt = len(results_df)
    
    # Return as DataFrame with single row
    final_df = pd.DataFrame([{'positive_review_cnt': positive_review_cnt}])
    
    return final_df, cost
