"""
DocETL Query 4: Positivity ratio (average of 0/1) for movie "taken_3"

This query calculates the average positivity score for reviews.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, MapOp, CodeFilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q4 using DocETL: Calculate positivity ratio for taken_3.
    
    Args:
        data_dir: Path to the data directory containing Reviews.csv
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)
    
    Returns:
        Tuple of (DataFrame with positivity ratio, cost)
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
    
    # Add positivity column (0 or 1)
    add_positivity = MapOp(
        name="add_positivity",
        type="map",
        prompt="Return 1 if the following review is positive, and 0 if the review is not positive. Only output a single numeric value (1 or 0) with no additional commentary\n\nReview: {{ input.reviewText }}",
        output={
            "schema": {
                "positivity": "int"
            }
        }
    )
    
    # Define pipeline step (no LLM-powered reduce, we'll use pandas for averaging)
    step = PipelineStep(
        name="calculate_positivity_ratio",
        input="reviews",
        operations=["filter_by_movie", "add_positivity"]
    )
    
    # Define output
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    output_file = os.path.join(intermediate_dir, "q4_results.json")
    
    output = PipelineOutput(
        type="file",
        path=output_file,
        intermediate_dir=intermediate_dir
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        name="q4_positivity_ratio",
        datasets={"reviews": dataset},
        operations=[movie_filter, add_positivity],
        steps=[step],
        output=output,
        default_model=model_name,
    )
    
    cost = pipeline.run(max_threads=max_threads)
    
    # Load results and calculate average using pandas
    results_df = pd.read_json(output_file)
    
    # Calculate positivity ratio (average of 0/1 values)
    positivity_ratio = results_df['positivity'].mean()
    
    # Return as DataFrame with single row
    final_df = pd.DataFrame([{'positivity_ratio': positivity_ratio}])
    
    return final_df, cost
