"""
DocETL Query 3: Extract brand name from product description.

Semantic map on textual data (productDisplayName + productDescriptors).
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q3 using DocETL: Extract brand name from product description.

    Args:
        data_dir: Path to the data directory containing styles_details.parquet
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with product ID and brand category, cost)
    """
    # Load and flatten parquet data for DocETL
    styles_df = pd.read_parquet(os.path.join(data_dir, "styles_details.parquet"))
    styles_df = styles_df.rename(columns={"id": "product_id"})

    # Extract description text from nested productDescriptors dict
    styles_df["description"] = styles_df["productDescriptors"].apply(
        lambda x: (x.get("description") or {}).get("value", "") if isinstance(x, dict) else ""
    )

    # Keep only needed columns
    styles_df = styles_df[["product_id", "productDisplayName", "description"]]

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q3_input.csv")
    styles_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Semantic map: extract brand name
    map_op = MapOp(
        name="extract_brand",
        type="map",
        prompt=(
            "Extract the brand name from the following product description and title. "
            "Only return the brand name, nothing else.\n\n"
            "Product title: {{ input.productDisplayName }}\n"
            "Product description: {{ input.description }}"
        ),
        output={"schema": {"category": "str"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="map_brands",
        input="products",
        operations=["extract_brand"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q3_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q3_extract_brand",
        datasets={"products": dataset},
        operations=[map_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results
    results_df = pd.read_json(output_file)
    return results_df[["product_id", "category"]].rename(columns={"product_id": "id"}), cost
