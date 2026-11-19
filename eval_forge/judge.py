"""Model-graded scoring, and the check that the grader can be trusted. The judge is any function from prompt text to text."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from . import stats
from .core import Case
from .scorers import Scorer

Judge = Callable[[str], str]
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def render_rubric(case: Case, output: Any, criteria: str, scale: int = 5) -> str:
    return (f"You are grading an answer. Criteria:\n{criteria}\n\nQuestion:\n{case.input}\n\nReference answer:\n{case.expected}\n\n"
            f"Answer to grade:\n{output}\n\nReply with a single integer from 1 (fails the criteria) to {scale} (fully meets them). "
            "Reply with the number only.")


def rubric(judge: Judge, criteria: str, scale: int = 5) -> Scorer:
    """A scorer that asks a judge to rate the output from 1 to ``scale`` and maps the rating to 0..1. A reply with no number
    in range raises, so the case is recorded as an error and scores zero rather than being silently graded."""
    if scale < 2:
        raise ValueError("scale must be at least 2")

    def scorer(case: Case, output: Any) -> float:
        reply = judge(render_rubric(case, output, criteria, scale))
        for token in _NUMBER.findall(reply):
            value = float(token)
            if 1 <= value <= scale:
                return (value - 1) / (scale - 1)
        raise ValueError(f"judge reply has no rating from 1 to {scale}: {reply[:60]!r}")
    return scorer


@dataclass(frozen=True)
class Agreement:
    n: int
    accuracy: float  # share of items where the judge's pass/fail matches the human's
    kappa: float  # chance-corrected agreement on pass/fail
    spearman: float  # rank agreement on the raw scores
    bias: float  # mean judge score minus mean human score: positive means the judge is too generous
    false_pass: int  # the judge passed what the human failed
    false_fail: int


def agreement(judge_scores: Sequence[float], human_scores: Sequence[float], threshold: float = 0.5) -> Agreement:
    """How far to trust a model grader: compare its scores with human labels on the same items before using it to gate releases.
    A kappa under about 0.6 means the grader and the humans often disagree."""
    if len(judge_scores) != len(human_scores) or len(judge_scores) < 2:
        raise ValueError("need two equal-length lists of at least two scores")
    j = [s >= threshold for s in judge_scores]
    h = [s >= threshold for s in human_scores]
    n = len(j)
    return Agreement(n, sum(a == b for a, b in zip(j, h, strict=True)) / n, stats.cohens_kappa(j, h),
                     stats.spearman(judge_scores, human_scores), stats.mean(judge_scores) - stats.mean(human_scores),
                     sum(a and not b for a, b in zip(j, h, strict=True)), sum(b and not a for a, b in zip(j, h, strict=True)))
