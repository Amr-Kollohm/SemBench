"""
Palimpzest system runner implementation.
Placeholder required by the current structure of the benchmarking framework.
"""

from pathlib import Path
import sys
import time
import types
import traceback

from codecarbon import EmissionsTracker
from runner.generic_runner import GenericQueryMetric, GenericRunner

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_palimpzest_runner.generic_palimpzest_runner import (
    GenericPalimpzestRunner,
)


class PalimpzestRunner(GenericPalimpzestRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case, scale_factor, model_name, concurrent_llm_worker
        )

    def _discover_queries(self):
        # Match default implementation from GenericRunner
        return GenericRunner._discover_queries(self)

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        try:
            # The queries in Palimpzeset are Python files with a run() function.
            # Load its contents, create a module, invoke the run() function.
            query_text = self.scenario_handler.get_query_text(
                query_id, self.get_system_name()
            )
            query_module = types.ModuleType(f"q{query_id}_module")
            exec(query_text, query_module.__dict__)

            # Create config builder lambda
            config_builder = lambda num_semantic_ops: self.palimpzest_config(num_semantic_ops)

            # Start codecarbon tracking
            tracker = EmissionsTracker(log_level="error")
            tracker.start()

            start_time = time.time()
            results = query_module.run(
                config_builder, self.scenario_handler.get_data_dir(), self.validator
            )
            execution_time = time.time() - start_time

            # Stop tracker and get emissions data
            emissions_data = tracker.stop()

            # Store carbon metrics from tracker's final emissions
            # codecarbon returns emissions in kg CO2eq
            if emissions_data is not None:
                metric.carbon_produced = emissions_data

            # Get more detailed metrics from tracker's final values
            if hasattr(tracker, '_total_energy') and tracker._total_energy:
                metric.energy_consumed = tracker._total_energy.kWh
            if hasattr(tracker, '_total_co2') and tracker._total_co2:
                metric.carbon_produced = tracker._total_co2.kg
            if hasattr(tracker, '_total_water') and tracker._total_water:
                metric.water_consumed = tracker._total_water.litres

            # Store results in metric
            metric.execution_time = execution_time
            metric.results = results.to_df().rename(
                columns={"product_id": "id"}
            )
            metric.status = "success"
            metric.money_cost = results.execution_stats.total_execution_cost
            
            # Get token usage from execution stats
            if hasattr(results, 'execution_stats') and results.execution_stats:
                total_tokens = 0
                for plan_id, plan_stats in results.execution_stats.plan_stats.items():
                    for op_id, op_stats in plan_stats.operator_stats.items():
                        if hasattr(op_stats, 'total_input_tokens') and op_stats.total_input_tokens:
                            total_tokens += op_stats.total_input_tokens
                        if hasattr(op_stats, 'total_output_tokens') and op_stats.total_output_tokens:
                            total_tokens += op_stats.total_output_tokens
                if total_tokens > 0:
                    metric.token_usage = total_tokens

        except Exception as e:
            metric.status = "failed"
            metric.error = str(e)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        return metric
