"""Batch-recorded statistical experiment platform."""

from .adjustment import adjust, bonferroni, holm
from .gatekeeping import GateNode, GatePlan, plan_from_positions
from .models import Dataset, PermutationResult, TestResult, TestSpec
from .permutation import GroupedPermutationTest
from .sequential import SequentialDesign, build_design, crossing_probabilities, default_looks
from .runner import PlatformRunner

__all__ = [
    "Dataset",
    "GateNode",
    "GatePlan",
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
    "plan_from_positions",
]
