"""Public API for the `library` package."""

from .contraction import adjust_embeddings
from .benchmarks import (
    run_fixed_point_benchmark,
    run_llm_benchmark,
    FORCE_MOCK_MODE,
)

__all__ = [
    "adjust_embeddings",
    "run_fixed_point_benchmark",
    "run_llm_benchmark",
    "FORCE_MOCK_MODE",
]

