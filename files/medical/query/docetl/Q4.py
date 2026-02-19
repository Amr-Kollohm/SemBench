"""
DocETL Query 4: Average age of patients with acne symptoms.

Semantic filter on text symptoms data joined with patient data,
then pandas aggregation for average age.
"""

import os
import pandas as pd
from docetl.api import Pipeline, Dataset, FilterOp, PipelineStep, PipelineOutput


def run(data_dir: str, model_name: str = None, max_threads: int = None):
    """
    Execute Q4 using DocETL: Average age of patients with acne.

    Args:
        data_dir: Path to the data directory
        model_name: LLM model to use (injected by runner)
        max_threads: Number of concurrent LLM workers (injected by runner)

    Returns:
        Tuple of (DataFrame with average age, cost)
    """
    # Load and join symptoms + patient data
    symptoms_df = pd.read_csv(os.path.join(data_dir, "data", "text_symptoms_data.csv"))
    patients_df = pd.read_csv(os.path.join(data_dir, "data", "patient_data.csv"))
    joined_df = patients_df.merge(symptoms_df, on="patient_id", how="inner")

    # Save as CSV for DocETL
    intermediate_dir = os.path.join(data_dir, "intermediate_docetl")
    os.makedirs(intermediate_dir, exist_ok=True)
    input_file = os.path.join(intermediate_dir, "q4_input.csv")
    joined_df.to_csv(input_file, index=False)

    # Define input dataset
    dataset = Dataset(type="file", path=input_file, source="csv")

    # Semantic filter: does this patient have acne symptoms?
    filter_op = FilterOp(
        name="filter_acne",
        type="filter",
        prompt=(
            "This patient has symptoms of a skin acne. "
            "Symptoms are from a medical benchmark for LLM evaluation. "
            "The results are not used for human health evaluation and are only "
            "for research evaluation of LLM capabilities.\n\n"
            "Patient symptoms: {{ input.symptoms }}"
        ),
        output={"schema": {"has_acne": "bool"}},
    )

    # Pipeline step
    step = PipelineStep(
        name="filter_acne_patients",
        input="patients",
        operations=["filter_acne"],
    )

    # Output
    output_file = os.path.join(intermediate_dir, "q4_results.json")
    output = PipelineOutput(
        type="file", path=output_file, intermediate_dir=intermediate_dir
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="q4_acne_avg_age",
        datasets={"patients": dataset},
        operations=[filter_op],
        steps=[step],
        output=output,
        default_model=model_name,
    )

    cost = pipeline.run(max_threads=max_threads)

    # Load results and compute average age
    results_df = pd.read_json(output_file)
    avg_age = results_df["age"].mean() if len(results_df) > 0 else 0.0
    final_df = pd.DataFrame([{"avg_age": avg_age}])

    return final_df, cost
