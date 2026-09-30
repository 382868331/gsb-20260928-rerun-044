"""Batch-recorded statistical experiment platform."""

from .adjustment import adjust, bonferroni, holm
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
    "adjust",
    "bonferroni",
    "holm",
]
