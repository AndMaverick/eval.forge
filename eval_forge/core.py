"""Cases, suites, the runner and its report."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import stats


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    expected: Any = None  # one answer, or a list of equally acceptable references
    tags: tuple[str, ...] = ()  # slices to report on: "arabic", "long", "refund"
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)  # not scored, kept for your own analysis

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "input": self.input, "expected": self.expected, "tags": list(self.tags), "meta": self.meta}


@dataclass(frozen=True)
class Result:
    """Optionally return this from the system under test to report cost and tokens alongside the output."""

    output: Any
    cost: float = 0.0
    tokens: int = 0


def _default_scorers() -> dict[str, Callable[[Case, Any], float]]:
    from .scorers import exact
    return {"exact": exact}


@dataclass
class Suite:
    name: str
    cases: list[Case]
    scorers: dict[str, Callable[[Case, Any], float]] = field(default_factory=_default_scorers)

    def __post_init__(self) -> None:
        ids = [c.id for c in self.cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate case ids: {', '.join(dupes)}")

    def digest(self) -> str:
        """Fingerprint of the cases and the scorer names. Two reports can only be compared fairly if this matches."""
        payload = json.dumps({"cases": [[c.id, c.input, c.expected, list(c.tags)] for c in self.cases], "scorers": sorted(self.scorers)},
                             sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Report:
    suite: str
    scores: dict[str, dict[str, float]]  # scorer -> {case id -> score}; with repeats, the mean over repeats
    errors: dict[str, str]  # case id -> error text
    outputs: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, tuple[str, ...]] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    repeats: int = 1
    flaky: list[str] = field(default_factory=list)  # cases whose score changed between repeats
    digest: str = ""

    def mean(self, scorer: str) -> float:
        values = list(self.scores[scorer].values())
        return stats.mean(values) if values else 0.0

    def pass_rate(self, scorer: str, threshold: float = 1.0) -> float:
        values = list(self.scores[scorer].values())
        return sum(v >= threshold for v in values) / len(values) if values else 0.0

    def interval(self, scorer: str, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
        """A confidence interval on the mean: Wilson when every score is 0 or 1, bootstrap otherwise."""
        values = list(self.scores[scorer].values())
        if values and all(v in (0.0, 1.0) for v in values):
            return stats.wilson(sum(values), len(values), confidence)
        return stats.bootstrap_ci(values, confidence, seed=seed)

    def by_tag(self, scorer: str, min_cases: int = 1) -> dict[str, tuple[float, int]]:
        """Mean score and case count per tag, worst first. Slices smaller than ``min_cases`` are left out."""
        groups: dict[str, list[float]] = {}
        for cid, score in self.scores[scorer].items():
            for t in self.tags.get(cid, ()):
                groups.setdefault(t, []).append(score)
        rows = {t: (stats.mean(v), len(v)) for t, v in groups.items() if len(v) >= min_cases}
        return dict(sorted(rows.items(), key=lambda kv: kv[1][0]))

    def failures(self, scorer: str, threshold: float = 1.0) -> list[str]:
        return [cid for cid, s in self.scores[scorer].items() if s < threshold]

    def to_dict(self) -> dict[str, Any]:
        return {"suite": self.suite, "digest": self.digest, "scores": self.scores, "errors": self.errors, "outputs": self.outputs,
                "tags": {k: list(v) for k, v in self.tags.items()}, "latency": self.latency, "cost": self.cost, "repeats": self.repeats,
                "flaky": self.flaky}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Report:
        return cls(data["suite"], data["scores"], data.get("errors", {}), data.get("outputs", {}),
                   {k: tuple(v) for k, v in data.get("tags", {}).items()}, data.get("latency", {}), data.get("cost", {}),
                   data.get("repeats", 1), data.get("flaky", []), data.get("digest", ""))


def run(suite: Suite, system: Callable[[str], Any], *, repeats: int = 1, retries: int = 0, workers: int = 1,
        keep_outputs: bool = False, clock: Callable[[], float] = time.perf_counter) -> Report:
    """Run ``system(case.input)`` over every case and score the outputs.

    * A crash scores zero and is recorded, and the run continues. A scorer that raises is recorded the same way.
    * ``retries`` re-calls the system after a crash. ``repeats`` runs every case several times for a system that is not
      deterministic and averages the scores, listing the cases whose score moved as ``flaky``.
    * ``workers`` above 1 runs calls concurrently. Results keep case order, so a report is the same either way.
    """
    if repeats < 1 or retries < 0 or workers < 1:
        raise ValueError("repeats and workers must be at least 1 and retries at least 0")

    def one(case: Case) -> tuple[dict[str, float], Any, str | None, float, float]:
        started = clock()
        error: str | None = None
        output: Any = None
        cost = 0.0
        for _attempt in range(retries + 1):
            try:
                output = system(case.input)
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        latency = clock() - started
        if error is not None:
            return dict.fromkeys(suite.scorers, 0.0), None, error, latency, cost
        if isinstance(output, Result):
            cost, output = output.cost, output.output
        scores: dict[str, float] = {}
        for name, scorer in suite.scorers.items():
            try:
                scores[name] = float(scorer(case, output))
            except Exception as exc:
                scores[name] = 0.0
                error = error or f"scorer {name}: {type(exc).__name__}: {exc}"
        return scores, output, error, latency, cost

    jobs = [c for c in suite.cases for _ in range(repeats)]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, jobs))
    else:
        results = [one(c) for c in jobs]
    scores: dict[str, dict[str, float]] = {name: {} for name in suite.scorers}
    errors: dict[str, str] = {}
    outputs: dict[str, Any] = {}
    latency: dict[str, float] = {}
    cost: dict[str, float] = {}
    flaky: list[str] = []
    for i, case in enumerate(suite.cases):
        chunk = results[i * repeats:(i + 1) * repeats]
        for name in suite.scorers:
            per = [r[0][name] for r in chunk]
            scores[name][case.id] = stats.mean(per)
            if len(set(per)) > 1 and case.id not in flaky:
                flaky.append(case.id)
        err = next((r[2] for r in chunk if r[2]), None)
        if err:
            errors[case.id] = err
        if keep_outputs:
            outputs[case.id] = chunk[-1][1]
        latency[case.id] = stats.mean([r[3] for r in chunk])
        cost[case.id] = stats.mean([r[4] for r in chunk])
    return Report(suite.name, scores, errors, outputs, {c.id: c.tags for c in suite.cases}, latency, cost, repeats, flaky, suite.digest())
