"""
DocETL Query 9: Score from 1 to 5 how much did the reviewer like the movie

This query scores reviews on a 1-5 scale for movie 'ant_man_and_the_wasp_quantumania'.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, MapOp, CodeFilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q9 using DocETL: Score reviews from 1-5.
    
    Args:
        data_dir: Path to the data directory containing Reviews.csv
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)
    
    Returns:
        Tuple of (DataFrame with reviewId and reviewScore, cost)
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
        code="def transform(doc):\n    return doc.get('id') == 'ant_man_and_the_wasp_quantumania'"
    )
    
    # Add review score
    add_score = MapOp(
        name="add_review_score",
        type="map",
        prompt="""Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {{ input.reviewText }}

Only provide the score number (1-5) with no other comments.""",
        output={
            "schema": {
                "reviewScore": "int"
            }
        }
    )
    
    # Define pipeline step
    step = PipelineStep(
        name="score_reviews",
        input="reviews",
        operations=["filter_by_movie", "add_review_score"]
    )
    
    # Define output
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    output_file = os.path.join(intermediate_dir, "q9_results.json")
    
    output = PipelineOutput(
        type="file",
        path=output_file,
        intermediate_dir=intermediate_dir
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        name="q9_review_scores",
        datasets={"reviews": dataset},
        operations=[movie_filter, add_score],
        steps=[step],
        output=output,
        default_model=model_name,
    )
    
    cost = pipeline.run(max_threads=max_threads)
    
    # Load results
    results_df = pd.read_json(output_file)
    
    return results_df[['reviewId', 'reviewScore']], cost
