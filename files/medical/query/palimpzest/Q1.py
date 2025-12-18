import os
import pandas as pd
import palimpzest as pz


def run(config_builder, data_dir: str, scale_factor: int = 11112, validator=None):
    # Load data
    symptoms_text = pd.read_csv(os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"))
    symptoms_text = pz.MemoryDataset(id="symptoms", vals=symptoms_text)

    # Filter data (1 sem_filter operation)
    symptoms_text = symptoms_text.sem_filter('This patient has symptoms of an allergy. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.', depends_on=['symptoms'])
    symptoms_text = symptoms_text.project(['patient_id'])
    
    output = symptoms_text.optimize_and_run(config=config_builder(num_semantic_ops=1), validator=validator)
    
    return output
