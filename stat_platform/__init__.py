"""Batch-recorded statistical experiment platform."""

from .adjustment import adjust, bonferroni, holm
from .models import Dataset, PermutationResult, TestResult, TestSpec
from .permutation import GroupedPermutationTest
from .sequential import SequentialDesign, build_design, crossing_probabilities, default_looks
from .runner import PlatformRunner

__all__ = [
    "Dataset",
    "GroupedPermutationTest",
    "SequentialDesign",
    "build_design",
    "crossing_probabilities",
    "default_looks",
    "PermutationResult",
    "PlatformRunner",
    "TestResult",
    "TestSpec",
    "adjust",
    "bonferroni",
    "holm",
]
