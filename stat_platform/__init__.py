"""Batch-recorded statistical experiment platform."""

from .adjustment import adjust, bonferroni, holm
from .models import Dataset, PermutationResult, TestResult, TestSpec
from .permutation import GroupedPermutationTest
from .sequential import SequentialDesign, finite_population_risks
from .runner import PlatformRunner

__all__ = [
    "Dataset",
    "GroupedPermutationTest",
    "PermutationResult",
    "PlatformRunner",
    "SequentialDesign",
    "TestResult",
    "TestSpec",
    "adjust",
    "bonferroni",
    "finite_population_risks",
    "holm",
]
