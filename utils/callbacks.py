"""LangChain callback handlers for token tracking.

Copied from refactor/utils/callbacks.py.
"""

import logging
import threading
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger("Columbus.token_measurer")


class TokenMeasurerCallbackHandler(BaseCallbackHandler):
    """Log LLM token usage per call."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    logger.info(
                        "LLM Tokens: [Input: %d, Output: %d, Total: %d]",
                        usage.get("input_tokens", 0),
                        usage.get("output_tokens", 0),
                        usage.get("total_tokens", 0),
                    )


class TokenAccumulatorCallbackHandler(BaseCallbackHandler):
    """Thread-safe accumulator for total token usage across a pipeline run."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self._estimated_inputs = {}
        self._lock = threading.Lock()

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        est_tokens = sum(max(1, len(p) // 4) for p in prompts)
        with self._lock:
            if run_id:
                self._estimated_inputs[str(run_id)] = est_tokens
            else:
                self.input_tokens += est_tokens
                self.total_tokens += est_tokens

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        has_actual = False
        actual_in = 0
        actual_out = 0
        estimated_out = sum(
            max(1, len(gen.text) // 4)
            for generations in response.generations
            for gen in generations
        )

        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    actual_in += usage.get("input_tokens", 0)
                    actual_out += usage.get("output_tokens", 0)
                    has_actual = True

        with self._lock:
            if has_actual:
                self.input_tokens += actual_in
                self.output_tokens += actual_out
                self.total_tokens += (actual_in + actual_out)
                if run_id and str(run_id) in self._estimated_inputs:
                    del self._estimated_inputs[str(run_id)]
            else:
                if run_id and str(run_id) in self._estimated_inputs:
                    est_in = self._estimated_inputs.pop(str(run_id))
                else:
                    est_in = 0
                self.input_tokens += est_in
                self.output_tokens += estimated_out
                self.total_tokens += (est_in + estimated_out)
