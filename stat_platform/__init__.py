"""Batch-recorded statistical experiment platform."""

from .adjustment import adjust, bonferroni, holm
from .allocation import gates_from_plan, simulate_scheme, study_schemes
from .hierarchy import HypothesisGate, path_risk_summaries, validate_gates
from .models import Dataset, PermutationResult, TestResult, TestSpec
from .permutation import GroupedPermutationTest
from .sequential import SequentialDesign, build_design, crossing_probabilities, default_looks
from .runner import PlatformRunner

__all__ = [
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
    "gates_from_plan",
    "holm",
    "path_risk_summaries",
    "simulate_scheme",
    "study_schemes",
    "validate_gates",
]
