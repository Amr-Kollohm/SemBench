"""
EcoLogits environmental impact tracker for LLM API calls.

Uses litellm's callback mechanism to intercept all LLM calls and compute
environmental impacts (energy, GHG emissions, etc.) using EcoLogits.
"""

import threading
from dataclasses import dataclass, field

try:
    from ecologits.tracers.litellm_tracer import litellm_match_model
    from ecologits.tracers.utils import llm_impacts
    from ecologits.utils.range_value import RangeValue

    ECOLOGITS_AVAILABLE = True
except ImportError:
    ECOLOGITS_AVAILABLE = False

try:
    from litellm.integrations.custom_logger import CustomLogger

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


def _extract_value(val):
    """Extract a float from an EcoLogits impact value (float or RangeValue)."""
    if val is None:
        return 0.0
    if isinstance(val, RangeValue):
        return val.mean
    return float(val)


@dataclass
class EnvironmentalMetrics:
    """Accumulated environmental impact metrics for a query."""

    energy_consumed: float = 0.0  # kWh
    ghg_emissions: float = 0.0    # kgCO2eq
    adpe: float = 0.0             # kgSbeq (Abiotic Depletion Potential)
    pe: float = 0.0               # MJ (Primary Energy)

    def to_dict(self):
        return {
            "energy_consumed": self.energy_consumed,
            "ghg_emissions": self.ghg_emissions,
            "adpe": self.adpe,
            "pe": self.pe,
        }


class EcoLogitsTracker(CustomLogger if LITELLM_AVAILABLE else object):
    """
    Litellm callback that computes per-request environmental impacts
    using EcoLogits and accumulates them for the current query.
    """

    def __init__(self):
        if LITELLM_AVAILABLE:
            super().__init__()
        self._lock = threading.Lock()
        self._metrics = EnvironmentalMetrics()

    def reset(self):
        """Reset accumulated metrics before a new query."""
        with self._lock:
            self._metrics = EnvironmentalMetrics()

    def get_results(self) -> EnvironmentalMetrics:
        """Return a copy of the accumulated metrics."""
        with self._lock:
            return EnvironmentalMetrics(
                energy_consumed=self._metrics.energy_consumed,
                ghg_emissions=self._metrics.ghg_emissions,
                adpe=self._metrics.adpe,
                pe=self._metrics.pe,
            )

    def _accumulate_impacts(self, model, start_time, end_time, response_obj):
        """Compute and accumulate environmental impacts for a single LLM call."""
        if not ECOLOGITS_AVAILABLE:
            return

        model_match = litellm_match_model(model)
        if model_match is None:
            return

        output_tokens = 0
        if hasattr(response_obj, "usage") and response_obj.usage is not None:
            output_tokens = getattr(
                response_obj.usage, "completion_tokens", 0
            ) or 0

        request_latency = (end_time - start_time).total_seconds()
        if request_latency <= 0:
            return

        impacts = llm_impacts(
            provider=model_match[0],
            model_name=model_match[1],
            output_token_count=output_tokens,
            request_latency=request_latency,
        )
        if impacts is None:
            return

        with self._lock:
            if impacts.energy and impacts.energy.value is not None:
                self._metrics.energy_consumed += _extract_value(
                    impacts.energy.value
                )
            if impacts.gwp and impacts.gwp.value is not None:
                self._metrics.ghg_emissions += _extract_value(
                    impacts.gwp.value
                )
            if impacts.adpe and impacts.adpe.value is not None:
                self._metrics.adpe += _extract_value(impacts.adpe.value)
            if impacts.pe and impacts.pe.value is not None:
                self._metrics.pe += _extract_value(impacts.pe.value)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        model = kwargs.get("model", "")
        self._accumulate_impacts(model, start_time, end_time, response_obj)

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        model = kwargs.get("model", "")
        self._accumulate_impacts(model, start_time, end_time, response_obj)


def create_tracker():
    """
    Create and register an EcoLogitsTracker with litellm.

    Returns the tracker instance, or None if dependencies are unavailable.
    """
    if not ECOLOGITS_AVAILABLE:
        print(
            "Warning: ecologits not installed. "
            "Environmental metrics will not be tracked."
        )
        return None

    if not LITELLM_AVAILABLE:
        print(
            "Warning: litellm not installed. "
            "Environmental metrics will not be tracked."
        )
        return None

    import litellm

    # Prevent duplicate registration
    for cb in litellm.callbacks:
        if isinstance(cb, EcoLogitsTracker):
            return cb

    tracker = EcoLogitsTracker()
    litellm.callbacks.append(tracker)
    return tracker
