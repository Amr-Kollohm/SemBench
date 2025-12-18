import os
import pandas as pd
import palimpzest as pz


def run(config_builder, data_dir: str, validator=None):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    ).rename(
        columns={"id": "prod_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column
    styles_details = styles_details[styles_details["price"] <= 500]
    styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)

    # Join data (1 sem_join operation)
    styles_details = styles_details.sem_join(
        styles_details,
        """
        You will be given two product descriptions.
        Do both product descriptions describe products of the same category from the
        same brand, e.g., both are t-shirts from Adidas?
        """,
        depends_on=[
            "productDisplayName",
            "productDescriptors",
            "productDisplayName_right",
            "productDescriptors_right",
        ],
    )

    # Generate joined identifiers
    styles_details = styles_details.add_columns(
        udf=lambda row: {
            "product_id": str(row["prod_id"]) + "-" + str(row["prod_id_right"])
        },
        cols=[
            {"name": "product_id", "type": str, "description": "Combined ID"}
        ],
    )
    styles_details = styles_details.project(["product_id"])

    output = styles_details.optimize_and_run(config=config_builder(num_semantic_ops=1), validator=validator)
    return output
