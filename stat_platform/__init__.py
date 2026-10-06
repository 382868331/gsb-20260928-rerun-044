"""Batch-recorded statistical experiment platform."""

from .adjustment import adjust, bonferroni, holm
from .allocation import AllocationStudy, AllocationTarget, study_allocations
from .hierarchy import HypothesisGate, path_risk_summaries, validate_gates
from .models import Dataset, PermutationResult, TestResult, TestSpec
from .permutation import GroupedPermutationTest
from .sequential import SequentialDesign, build_design, crossing_probabilities, default_looks
from .runner import PlatformRunner

__all__ = [
    "AllocationStudy",
    "AllocationTarget",
    "Dataset",
    "GroupedPermutationTest",
    "HypothesisGate",
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
    "path_risk_summaries",
    "study_allocations",
    "validate_gates",
]
