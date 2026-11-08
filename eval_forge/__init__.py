"""eval.forge: a small evaluation harness with scorers, suites, and a regression gate."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean

__version__ = "0.1.0"


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    expected: object = None


def exact(case, output) -> float:
    return 1.0 if str(output).strip() == str(case.expected).strip() else 0.0


def contains(case, output) -> float:
    return 1.0 if str(case.expected).lower() in str(output).lower() else 0.0


def regex(pattern: str):
    compiled = re.compile(pattern)
    return lambda case, output: 1.0 if compiled.search(str(output)) else 0.0


def numeric_close(tolerance: float = 1e-6):
    def scorer(case, output) -> float:
        try:
            return 1.0 if abs(float(output) - float(case.expected)) <= tolerance else 0.0
        except (TypeError, ValueError):
            return 0.0
    return scorer


@dataclass
class Suite:
    name: str
    cases: list
    scorers: dict = field(default_factory=lambda: {"exact": exact})


@dataclass
class Report:
    suite: str
    scores: dict  # scorer -> {case id -> score}
    errors: dict  # case id -> error text

    def mean(self, scorer: str) -> float:
        values = list(self.scores[scorer].values())
        return mean(values) if values else 0.0

    def pass_rate(self, scorer: str, threshold: float = 1.0) -> float:
        values = list(self.scores[scorer].values())
        return sum(v >= threshold for v in values) / len(values) if values else 0.0


def run(suite: Suite, system) -> Report:
    """Run system(case.input) over every case. A crash scores zero and is recorded."""
    scores = {name: {} for name in suite.scorers}
    errors: dict = {}
    for case in suite.cases:
        try:
            output = system(case.input)
        except Exception as exc:
            errors[case.id] = f"{type(exc).__name__}: {exc}"
            for name in suite.scorers:
                scores[name][case.id] = 0.0
            continue
        for name, scorer in suite.scorers.items():
            scores[name][case.id] = float(scorer(case, output))
    return Report(suite.name, scores, errors)


@dataclass
class Gate:
    ok: bool
    regressions: list


def gate(baseline: Report, candidate: Report, max_drop: float = 0.02) -> Gate:
    """Fail when any scorer's mean drops by more than max_drop against the baseline."""
    regressions = []
    for scorer in baseline.scores:
        if scorer not in candidate.scores:
            regressions.append((scorer, "missing", None))
            continue
        drop = baseline.mean(scorer) - candidate.mean(scorer)
        if drop > max_drop:
            regressions.append((scorer, round(drop, 4), candidate.mean(scorer)))
    return Gate(not regressions, regressions)
