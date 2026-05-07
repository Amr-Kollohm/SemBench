"""
Generic DocETL runner base class.


"""

import re
import time
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from overrides import override

from runner.generic_runner import GenericQueryMetric, GenericRunner


class GenericDocETLRunner(GenericRunner):
    """GenericRunner for DocETL system."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker: int = 20,
        skip_setup: bool = False,
    ):
        """Initialize DocETL runner."""
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

    @override
    def get_system_name(self) -> str:
        return "docetl"

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query and return metrics.

        Scenario-specific query methods should return either:
        - DataFrame
        - dict with keys:
          - "results": DataFrame
          - "stats": {"cost": float, "token_usage": {model: {prompt_tokens, completion_tokens}}}
        """
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        try:
            query_fn = self._discover_query_impl(query_id)

            start_time = time.time()
            result = query_fn()
            metric.execution_time = time.time() - start_time
            metric.status = "success"

            if isinstance(result, dict):
                metric.results = self._normalize_results(
                    result.get("results", pd.DataFrame())
                )
                self._update_usage_from_stats(metric, result.get("stats"))
            else:
                metric.results = self._normalize_results(result)

        except Exception as e:
            metric.status = "failed"
            metric.error = str(e)
            metric.results = self._get_empty_results_dataframe(query_id)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()

        return metric

    def _normalize_results(self, results: Any) -> pd.DataFrame:
        """Normalize query output to a pandas DataFrame."""
        if isinstance(results, pd.DataFrame):
            return results
        if results is None:
            return pd.DataFrame()
        if isinstance(results, list):
            return pd.DataFrame(results)
        if isinstance(results, dict):
            return pd.DataFrame([results])
        return pd.DataFrame(results)

    def _update_usage_from_stats(
        self, metric: GenericQueryMetric, stats: Optional[Dict[str, Any]]
    ) -> None:
        """Populate token and cost fields from DocETL-style stats."""
        if not stats:
            metric.token_usage = 0
            metric.money_cost = 0.0
            return

        token_usage = stats.get("token_usage", {})
        total_prompt = 0
        total_completion = 0

        for model_usage in token_usage.values():
            total_prompt += int(model_usage.get("prompt_tokens", 0) or 0)
            total_completion += int(
                model_usage.get("completion_tokens", 0) or 0
            )

        metric.token_usage = total_prompt + total_completion
        metric.money_cost = float(stats.get("cost", 0.0) or 0.0)

    def _discover_queries(self) -> List[int]:
        """
        Discover available queries from `_execute_q*` methods.

        This keeps behavior aligned with code-based runners like LOTUS.
        """
        pattern = re.compile(r"_execute_q(\d+)$")
        query_ids: List[int] = []

        for attr_name in dir(self):
            match = pattern.match(attr_name)
            if match:
                attr = getattr(self, attr_name, None)
                if callable(attr):
                    query_ids.append(int(match.group(1)))

        return sorted(query_ids)
