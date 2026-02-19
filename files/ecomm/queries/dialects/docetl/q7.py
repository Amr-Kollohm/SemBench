"""
DocETL Query 7: Find pairs of products of the same category and brand.

Text-to-text semantic join on product descriptions, pre-filtered to price <= 500.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, EquijoinOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q7 using DocETL: Find product pairs of same category and brand.

    Args:
        data_dir: Path to the data directory containing styles_details.parquet
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with combined product IDs, cost)
    """
    # Load and flatten parquet data for DocETL
    styles_df = pd.read_parquet(os.path.join(data_dir, "styles_details.parquet"))
    styles_df = styles_df.rename(columns={"id": "product_id"})

    # Pre-filter: price <= 500
    styles_df = styles_df[styles_df["price"] <= 500]

    # Extract description text from nested productDescriptors dict
    styles_df["description"] = styles_df["productDescriptors"].apply(
        lambda x: (x.get("description") or {}).get("value", "") if isinstance(x, dict) else ""
    )

    # Keep only needed columns
    styles_df = styles_df[["product_id", "productDisplayName", "description"]]

    # Save as CSV for DocETL (two copies for self-join)
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)

    left_file = os.path.join(intermediate_dir, "q7_left.csv")
    right_file = os.path.join(intermediate_dir, "q7_right.csv")
    styles_df.to_csv(left_file, index=False)
    styles_df.to_csv(right_file, index=False)

    # Define datasets for the join
    left_dataset = Dataset(type="file", path=left_file, source="csv")
    right_dataset = Dataset(type="file", path=right_file, source="csv")

    # Semantic join: same category and same brand
    join_op = EquijoinOp(
        name="join_same_category_brand",
        type="equijoin",
        left="products_left",
        right="products_right",
        comparison_prompt=(
            "You will be given two product descriptions. "
            "Do both product descriptions describe products of the same category "
            "from the same brand, e.g., both are t-shirts from Adidas?\n\n"
            "Product 1 title: {{ left.productDisplayName }}\n"
            "Product 1 description: {{ left.description }}\n\n"
            "Product 2 title: {{ right.productDisplayName }}\n"
            "Product 2 description: {{ right.description }}"
        ),
        output={"schema": {"same_category_brand": "bool"}},
    )

    # Pipeline step for the join
    step = PipelineStep(
        name="join_products",
        operations=[
            {"join_same_category_brand": {"left": "products_left", "right": "products_right"}}
        ],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q7_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q7_same_category_brand",
        datasets={"products_left": left_dataset, "products_right": right_dataset},
        operations=[join_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results and generate combined IDs
    results_df = pd.read_json(output_file)

    # Combine IDs to match Palimpzest output format: "id1-id2"
    results_df["id"] = (
        results_df["product_id_left"].astype(str)
        + "-"
        + results_df["product_id_right"].astype(str)
    )

    return results_df[["id"]], cost
