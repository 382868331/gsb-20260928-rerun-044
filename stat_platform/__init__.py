"""Batch-recorded statistical experiment platform."""

from .adjustment import adjust_p_values
from .models import Dataset, PermutationResult, TestResult, TestSpec
from .permutation import GroupedPermutationTest
from .runner import PlatformRunner

__all__ = [
    "Dataset",
    "GroupedPermutationTest",
    "PermutationResult",
    "PlatformRunner",
    "TestResult",
    "TestSpec",
    "adjust_p_values",
]
