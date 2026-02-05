"""
Generic DocETL system runner implementation.

DocETL is a system for processing documents using LLMs and declarative pipelines.
This runner provides the base implementation that can be extended by scenarios.

Note: DocETL can only process text or text files (no images or audio).
Reference: https://ucbepic.github.io/docetl/tutorial-pythonapi/
"""

import time
import types
import traceback
from overrides import override
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from runner.generic_runner import GenericRunner, GenericQueryMetric


class GenericDocETLRunner(GenericRunner):
    """Generic runner for DocETL system."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker: int = 20,
        skip_setup: bool = False,
    ):
        """
        Initialize DocETL runner.

        Args:
            use_case: The use case to run (e.g., 'ecomm', 'medical')
            scale_factor: Scale factor for data
            model_name: LLM model to use (default: gpt-4o-mini)
            concurrent_llm_worker: Number of concurrent workers
            skip_setup: Skip scenario setup if True
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        
        # DocETL-specific initialization
        self._initialize_docetl()

    @override
    def get_system_name(self) -> str:
        """Return the system name."""
        return "docetl"

    def _initialize_docetl(self):
        """
        Initialize DocETL system.
        
        This method sets up the DocETL environment and any necessary
        configurations.
        """
        try:
            # Import DocETL to verify it's available
            import docetl  # noqa: F401
            from docetl.api import Pipeline  # noqa: F401
            
            # Store model configuration
            self.model_config = {
                "model": self.model_name,
            }
            
            print(f"DocETL initialized with model: {self.model_name}")
            
        except ImportError as e:
            raise RuntimeError(
                f"DocETL not installed. Please install with: pip install docetl\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DocETL: {e}")

    @override
    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query using DocETL.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)

        Returns:
            GenericQueryMetric object containing results and metrics
        """
        metric = GenericQueryMetric(query_id=query_id, status="pending")
        
        try:
            # Get query from scenario handler
            if self.scenario_handler is not None:
                query_text = self.scenario_handler.get_query_text(
                    query_id, self.get_system_name()
                )
            else:
                # Fallback to default query loading
                query_text = self.get_query_text(query_id, "natural_language")
            
            # Execute the query
            start_time = time.time()
            results_df, cost = self._execute_docetl_query(query_id, query_text)
            execution_time = time.time() - start_time
            
            # Store results in metric
            metric.execution_time = execution_time
            metric.results = results_df
            metric.status = "success"
            
            # Store cost if available
            if cost is not None:
                metric.money_cost = cost
            
            print(f"  Q{query_id} completed in {execution_time:.2f}s, cost: ${cost:.4f}")
            
        except Exception as e:
            metric.status = "failed"
            metric.error = str(e)
            print(f"  Error executing Q{query_id}: {type(e).__name__}: {e}")
            raise
        
        return metric

    def _execute_docetl_query(
        self, query_id: int, query_text: str
    ) -> tuple[pd.DataFrame, Optional[float]]:
        """
        Execute a DocETL query and return results as DataFrame.
        
        DocETL queries are Python files that define pipelines using the DocETL API.
        The query should define a run() function that returns (DataFrame, cost).

        Args:
            query_id: Query ID
            query_text: Python code that defines and runs a DocETL pipeline

        Returns:
            Tuple of (DataFrame with query results, cost in USD)
        """
        # Create a module to execute the query code
        query_module = types.ModuleType(f"q{query_id}_module")
        
        # Inject the data directory path for the query to use
        if self.scenario_handler is not None:
            query_module.__dict__["DATA_DIR"] = str(
                self.scenario_handler.get_data_dir()
            )
        else:
            query_module.__dict__["DATA_DIR"] = str(self.data_path)
        
        # Inject configuration similar to Palimpzest's palimpzest_config()
        # Model name for the pipeline's default_model parameter
        query_module.__dict__["MODEL_NAME"] = self.model_name
        # Max threads for parallel execution (concurrent LLM workers)
        query_module.__dict__["MAX_THREADS"] = self.concurrent_llm_worker
        
        # Execute the query code
        exec(query_text, query_module.__dict__)
        
        # The query code should define a run() function that returns (DataFrame, cost)
        if hasattr(query_module, "run"):
            result = query_module.run(
                query_module.__dict__["DATA_DIR"],
                model_name=self.model_name,
                max_threads=self.concurrent_llm_worker
            )
            
            # Handle different return types
            if isinstance(result, tuple) and len(result) == 2:
                results_df, cost = result
            elif isinstance(result, pd.DataFrame):
                # Backward compatibility: if only DataFrame is returned
                results_df = result
                cost = None
            else:
                raise ValueError(
                    f"Query {query_id} run() function must return either:\n"
                    f"  - (DataFrame, cost) tuple, or\n"
                    f"  - DataFrame only\n"
                    f"Got {type(result)}"
                )
            
            if not isinstance(results_df, pd.DataFrame):
                raise ValueError(
                    f"Query {query_id} run() must return DataFrame as first element, "
                    f"got {type(results_df)}"
                )
            
            return results_df, cost
        else:
            raise ValueError(
                f"Query {query_id} must define a run(data_dir) function that "
                f"returns (DataFrame, cost) or just DataFrame"
            )
