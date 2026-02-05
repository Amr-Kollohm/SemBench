"""
DocETL Query 10: Rank movies based on review scores

This query scores all reviews from 1-5, then groups by movie and calculates average score.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q10 using DocETL: Rank movies by average review score.
    
    Args:
        data_dir: Path to the data directory containing Reviews.csv
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)
    
    Returns:
        Tuple of (DataFrame with movieId and average movieScore, cost)
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
    
    # Add review score to all reviews
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
    
    # Define pipeline step (no LLM-powered groupby, we'll use pandas for grouping/averaging)
    step = PipelineStep(
        name="rank_movies",
        input="reviews",
        operations=["add_review_score"]
    )
    
    # Define output
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    output_file = os.path.join(intermediate_dir, "q10_results.json")
    
    output = PipelineOutput(
        type="file",
        path=output_file,
        intermediate_dir=intermediate_dir
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        name="q10_movie_rankings",
        datasets={"reviews": dataset},
        operations=[add_score],
        steps=[step],
        output=output,
        default_model=model_name,
    )
    
    cost = pipeline.run(max_threads=max_threads)
    
    # Load results and group by movieId using pandas
    results_df = pd.read_json(output_file)
    
    # Group by 'id' (movieId) and calculate average reviewScore
    movie_scores = results_df.groupby('id')['reviewScore'].mean().reset_index()
    movie_scores.columns = ['movieId', 'movieScore']
    
    return movie_scores[['movieId', 'movieScore']], cost
