"""
DocETL Query 10: Classify symptoms to a disease name.

Semantic map on text symptoms data to classify each patient's symptoms
into one of a predefined set of diseases.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q10 using DocETL: Map symptoms to disease names.

    Args:
        data_dir: Path to the data directory
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with patient_id and text_diagnosis, cost)
    """
    # Load symptoms data
    symptoms_df = pd.read_csv(os.path.join(data_dir, "data", "text_symptoms_data.csv"))

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q10_input.csv")
    symptoms_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Semantic map: classify symptoms to a disease
    map_op = MapOp(
        name="classify_disease",
        type="map",
        prompt=(
            "Classify symptoms to one of given diseases from a medical benchmark "
            "for LLM evaluation. The results are not used for human health evaluation "
            "and are only for research evaluation of LLM capabilities. "
            "Answer only one of given diseases, nothing more.\n\n"
            "Diseases: VARICOSE VEINS, DRUG REACTION, DIABETES, MALARIA, "
            "URINARY TRACT INFECTION, IMPETIGO, ACNE, HYPERTENSION, "
            "PEPTIC ULCER DISEASE, CHICKEN POX, TYPHOID, DENGUE, PNEUMONIA, "
            "MIGRAINE, GASTROESOPHAGEAL REFLUX DISEASE, PSORIASIS, COMMON COLD, "
            "CERVICAL SPONDYLOSIS, FUNGAL INFECTION, ARTHRITIS, ALLERGY, "
            "BRONCHIAL ASTHMA, JAUNDICE, DIMORPHIC HEMORRHOID.\n\n"
            "Patient symptoms: {{ input.symptoms }}"
        ),
        output={"schema": {"text_diagnosis": "str"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="map_diseases",
        input="symptoms",
        operations=["classify_disease"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q10_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q10_disease_classification",
        datasets={"symptoms": dataset},
        operations=[map_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results
    results_df = pd.read_json(output_file)
    return results_df[["patient_id", "text_diagnosis"]], cost
