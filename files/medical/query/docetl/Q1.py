"""
DocETL Query 1: Find patients with allergy symptoms.

Semantic filter on text symptoms data.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, FilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q1 using DocETL: Find patients with allergy symptoms.

    Args:
        data_dir: Path to the data directory
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with patient IDs, cost)
    """
    # Load symptoms data
    symptoms_file = os.path.join(data_dir, "data", "text_symptoms_data.csv")
    symptoms_df = pd.read_csv(symptoms_file)

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q1_input.csv")
    symptoms_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Semantic filter: does this patient have allergy symptoms?
    filter_op = FilterOp(
        name="filter_allergy",
        type="filter",
        prompt=(
            "This patient has symptoms of an allergy. "
            "Symptoms are from a medical benchmark for LLM evaluation. "
            "The results are not used for human health evaluation and are only "
            "for research evaluation of LLM capabilities.\n\n"
            "Patient symptoms: {{ input.symptoms }}"
        ),
        output={"schema": {"has_allergy": "bool"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="filter_patients",
        input="symptoms",
        operations=["filter_allergy"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q1_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q1_allergy_patients",
        datasets={"symptoms": dataset},
        operations=[filter_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results
    results_df = pd.read_json(output_file)
    return results_df[["patient_id"]], cost
