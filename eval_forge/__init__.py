"""eval.forge: an evaluation harness that knows how much a number can be trusted.

Cases and scorers, a crash-safe runner with repeats and concurrency, confidence intervals, slices by tag, a
non-inferiority gate that says "inconclusive" when the suite is too small to tell, dataset hygiene (stable splits,
de-duplication, leakage check), and a way to measure a model grader against human labels.
"""
from __future__ import annotations

from .compare import Comparison, Gate, SignificanceGate, compare, gate, significance_gate
from .core import Case, Report, Result, Suite, run
from .data import dedupe, dump_cases, load_cases, overlap, split
from .judge import Agreement, agreement, render_rubric, rubric
from .scorers import (
                      all_of,
                      any_of,
                      contains,
                      edit_similarity,
                      exact,
                      exact_normalized,
                      f1_tokens,
                      jaccard,
                      json_equal,
                      json_fields,
                      json_valid,
                      normalize,
                      not_contains,
                      numeric_close,
                      parse_json,
                      regex,
                      weighted,
)

__version__ = "0.2.0"

__all__ = [
    "Case", "Suite", "Report", "Result", "run", "Gate", "gate", "SignificanceGate", "significance_gate", "Comparison", "compare",
    "exact", "contains", "regex", "numeric_close", "exact_normalized", "not_contains", "f1_tokens", "jaccard", "edit_similarity",
    "json_valid", "json_fields", "json_equal", "parse_json", "normalize", "all_of", "any_of", "weighted",
    "load_cases", "dump_cases", "split", "dedupe", "overlap", "rubric", "render_rubric", "agreement", "Agreement", "__version__",
]
