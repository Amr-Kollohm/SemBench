"""
DocETL Query 5: Classify apparel products into categories.

Semantic classification on text data (productDisplayName + productDescriptors).
Pre-filtered to Apparel masterCategory, excluding Saree, Apparel Set,
Loungewear and Nightwear.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q5 using DocETL: Classify apparel products.

    Categories:
        Dress, Bottomwear, Socks, Topwear, Innerwear

    Args:
        data_dir: Path to the data directory containing styles_details.parquet
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with product ID and category, cost)
    """
    # Load and flatten parquet data for DocETL
    styles_df = pd.read_parquet(os.path.join(data_dir, "styles_details.parquet"))
    styles_df = styles_df.rename(columns={"id": "product_id"})

    # Pre-filter: masterCategory == 'Apparel' and subCategory not in excluded list
    excluded_subcategories = {"Saree", "Apparel Set", "Loungewear and Nightwear"}
    styles_df = styles_df[
        styles_df.apply(
            lambda row: (
                isinstance(row["masterCategory"], dict)
                and row["masterCategory"].get("typeName") == "Apparel"
                and isinstance(row["subCategory"], dict)
                and row["subCategory"].get("typeName") not in excluded_subcategories
            ),
            axis=1,
        )
    ]

    # Extract description text from nested productDescriptors dict
    styles_df["description"] = styles_df["productDescriptors"].apply(
        lambda x: (x.get("description") or {}).get("value", "") if isinstance(x, dict) else ""
    )

    # Keep only needed columns
    styles_df = styles_df[["product_id", "productDisplayName", "description"]]

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "..", "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q5_input.csv")
    styles_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Semantic map: classify product into category
    classify_op = MapOp(
        name="classify_product",
        type="map",
        prompt=(
            "You are given a description of a product. Your task is to classify the product "
            "into one of the following categories:\n"
            "(1) Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down "
            "over the legs, and often consist of a bodice attached to a skirt.\n"
            "(2) Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, "
            "such as trousers, jeans, skirts, shorts, and leggings.\n"
            "(3) Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, "
            "designed to provide comfort and warmth.\n"
            "(4) Topwear: Topwear refers to clothing worn on the upper part of the body, "
            "such as shirts, blouses, t-shirts, and jackets.\n"
            "(5) Innerwear: Innerwear refers to clothing worn beneath outer garments, "
            "typically close to the skin, such as underwear, bras, and undershirts.\n\n"
            "When classifying the product, only output the category name, nothing more.\n\n"
            "Product title: {{ input.productDisplayName }}\n"
            "Product description: {{ input.description }}"
        ),
        output={"schema": {"category": "str"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="classify_products",
        input="products",
        operations=["classify_product"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q5_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q5_classify_apparel",
        datasets={"products": dataset},
        operations=[classify_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results
    results_df = pd.read_json(output_file)
    return results_df[["product_id", "category"]].rename(columns={"product_id": "id"}), cost
