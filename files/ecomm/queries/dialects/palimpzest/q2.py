import os
import pandas as pd
import palimpzest as pz


def run(config_builder, data_dir: str, validator=None):
    # Load data
    images = pz.ImageFileDataset(
        id="images", path=os.path.join(data_dir, "images")
    )

    # Filter data (1 sem_filter operation)
    images = images.sem_filter(
        "The image shows a (pair of) sports shoe(s) that feature the colors yellow and silver",
        depends_on=["contents"],
    )
    images = images.add_columns(
        udf=lambda row: {"product_id": row["filename"].split(".", 1)[0]},
        cols=[
            {
                "name": "product_id",
                "type": str,
                "description": "Product id generated from image name",
            }
        ],
    )
    images = images.project(["product_id"])

    output = images.optimize_and_run(config=config_builder(num_semantic_ops=1), validator=validator)
    return output
