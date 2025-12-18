import os
import pandas as pd
import palimpzest as pz


def run(config_builder, data_dir: str, validator=None):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    ).rename(
        columns={"id": "product_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column
    styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)

    # Filter data (1 sem_filter operation)
    styles_details = styles_details.sem_filter(
        "The product is a backpack from Reebok",
        depends_on=["productDisplayName", "productDescriptors"],
    )
    styles_details = styles_details.project(["product_id"])

    output = styles_details.optimize_and_run(config=config_builder(num_semantic_ops=1), validator=validator)
    return output
