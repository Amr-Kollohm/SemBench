"""
DocETL runner implementation for the e-commerce scenario.

This extends the GenericDocETLRunner to provide ecomm-specific functionality.
"""

from pathlib import Path
import sys

# Add parent directories to path to import generic runner
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_docetl_runner.generic_docetl_runner import GenericDocETLRunner


class DocETLRunner(GenericDocETLRunner):
    """
    DocETL runner for e-commerce scenario.

    This class inherits all functionality from GenericDocETLRunner.
    Override methods here if you need ecomm-specific behavior.
    """

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker: int = 20,
        skip_setup: bool = False,
    ):
        """
        Initialize DocETL runner for e-commerce scenario.

        Args:
            use_case: Should be 'ecomm'
            scale_factor: Scale factor for data
            model_name: LLM model to use
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
