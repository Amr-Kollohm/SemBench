"""
DocETL Query 6: Find pairs of reviews with opposite sentiment

This query performs a semantic join to find review pairs with opposite sentiment.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, EquijoinOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q6 using DocETL: Find pairs of reviews with opposite sentiment.
    
    Args:
        data_dir: Path to the data directory containing Reviews.csv
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)
    
    Returns:
        Tuple of (DataFrame with review pairs, cost)
    """
    # Parameters are passed directly from the runner
    # No fallback needed since runner always provides these values
    
    # Define input dataset
    input_file = os.path.join(data_dir, 'Reviews.csv')
    
    # For self-joins, we need to use pandas to filter and create two dataset references
    # This is the correct approach for DocETL joins which require separate dataset names
    all_reviews = pd.read_csv(input_file)
    filtered_reviews = all_reviews[all_reviews['id'] == 'ant_man_and_the_wasp_quantumania']
    
    # Note: Unlike FilterOp, EquijoinOp doesn't have early stopping.
    # It will compare all pairs and return matches. We take head(10) at the end.
    
    # Create temporary files for left and right datasets (same data, self-join)
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    left_file = os.path.join(intermediate_dir, "q6_left.csv")
    right_file = os.path.join(intermediate_dir, "q6_right.csv")
    
    filtered_reviews.to_csv(left_file, index=False)
    filtered_reviews.to_csv(right_file, index=False)
    
    # Define datasets for the join
    left_dataset = Dataset(
        type="file",
        path=left_file,
        source="csv"
    )
    
    right_dataset = Dataset(
        type="file",
        path=right_file,
        source="csv"
    )
    
    # Semantic join operation for opposite sentiment
    # Note: No early stopping available - will compare all pairs
    join_op = EquijoinOp(
        name="join_opposite_sentiment",
        type="equijoin",
        left="reviews_left",
        right="reviews_right",
        comparison_prompt="These two movie reviews express opposite sentiments - one is positive and the other is negative.\n\nReview 1: {{ left.reviewText }}\nReview 2: {{ right.reviewText }}",
        output={
            "schema": {
                "opposite_sentiment": "bool"
            }
        }
    )
    
    # Define pipeline step for the join
    step = PipelineStep(
        name="join_reviews",
        operations=[{"join_opposite_sentiment": {"left": "reviews_left", "right": "reviews_right"}}]
    )
    
    # Define output
    output_file = os.path.join(intermediate_dir, "q6_results.json")
    
    output = PipelineOutput(
        type="file",
        path=output_file,
        intermediate_dir=intermediate_dir
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        name="q6_opposite_sentiment_pairs",
        datasets={"reviews_left": left_dataset, "reviews_right": right_dataset},
        operations=[join_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )
    
    cost = pipeline.run(max_threads=max_threads)
    
    # Load results and project to match Palimpzest output format
    results_df = pd.read_json(output_file)
    
    # Select only the ID columns (matching Palimpzest: movieId, reviewId, reviewId_right)
    projected_df = results_df[['id_left', 'reviewId_left', 'reviewId_right']].copy()
    projected_df.rename(columns={'id_left': 'movieId', 'reviewId_left': 'reviewId'}, inplace=True)
    
    return projected_df.head(10), cost
