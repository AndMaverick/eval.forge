"""Comparing two runs honestly: paired differences, intervals, and a gate that knows when it cannot tell."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import stats
from .core import Report


@dataclass
class Gate:
    ok: bool
    regressions: list[tuple[str, object, object]]


def gate(baseline: Report, candidate: Report, max_drop: float = 0.02) -> Gate:
    """Fail when any scorer's mean drops by more than ``max_drop`` against the baseline.

    Simple and strict. It ignores noise: on a small suite a drop of 0.02 can be luck either way. For a gate that
    accounts for sample size, use ``significance_gate``.
    """
    regressions: list[tuple[str, object, object]] = []
    for scorer in baseline.scores:
        if scorer not in candidate.scores:
            regressions.append((scorer, "missing", None))
            continue
        drop = baseline.mean(scorer) - candidate.mean(scorer)
        if drop > max_drop:
            regressions.append((scorer, round(drop, 4), candidate.mean(scorer)))
    return Gate(not regressions, regressions)


@dataclass
class Comparison:
    scorer: str
    n: int
    baseline_mean: float
    candidate_mean: float
    diff: float  # candidate minus baseline
    lower: float  # one-sided bound at 1 - alpha: the true diff is above this
    upper: float
    p_value: float  # that the true difference is zero (McNemar for pass/fail scores, permutation test otherwise)
    verdict: str  # "pass", "fail" or "inconclusive" against the margin
    improved: list[str] = field(default_factory=list)  # case ids that got better
    regressed: list[str] = field(default_factory=list)  # case ids that got worse
    cases_needed: int | None = None  # paired cases that would settle an inconclusive result


def compare(baseline: Report, candidate: Report, scorer: str, margin: float = 0.02, alpha: float = 0.05, seed: int = 0,
            require_same_suite: bool = True) -> Comparison:
    """Is the candidate no worse than the baseline by more than ``margin``? Answered with a one-sided non-inferiority test
    on the paired per-case differences.

    * **pass**: the candidate's lower bound is above ``-margin``. It is shown to be within the margin.
    * **fail**: the upper bound is below ``-margin``. It is shown to be worse than the margin allows.
    * **inconclusive**: the data cannot tell. ``cases_needed`` says how many cases would.

    Reports must come from the same suite (same cases and scorers) or ``ValueError`` is raised: comparing runs on
    different test sets is a classic way to fool yourself.
    """
    if require_same_suite and baseline.digest and candidate.digest and baseline.digest != candidate.digest:
        raise ValueError("these reports come from different suites (different cases or scorers), so they cannot be compared")
    if scorer not in baseline.scores or scorer not in candidate.scores:
        raise KeyError(scorer)
    ids = [i for i in baseline.scores[scorer] if i in candidate.scores[scorer]]
    if not ids:
        raise ValueError("the reports share no cases")
    a = [baseline.scores[scorer][i] for i in ids]
    b = [candidate.scores[scorer][i] for i in ids]
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    lower, upper = stats.bootstrap_lower(diffs, alpha, seed=seed)
    binary = all(v in (0.0, 1.0) for v in a + b)
    if binary:
        p = stats.mcnemar_exact(sum(x == 1.0 and y == 0.0 for x, y in zip(a, b, strict=True)), sum(x == 0.0 and y == 1.0 for x, y in zip(a, b, strict=True)))
    else:
        p = stats.permutation_p(diffs, seed=seed)
    verdict = "pass" if lower > -margin else "fail" if upper < -margin else "inconclusive"
    needed = stats.cases_needed(stats.sd(diffs), margin, alpha) if verdict == "inconclusive" and stats.sd(diffs) > 0 else None
    return Comparison(scorer, len(ids), stats.mean(a), stats.mean(b), stats.mean(diffs), lower, upper, p, verdict,
                      [i for i, d in zip(ids, diffs, strict=True) if d > 0], [i for i, d in zip(ids, diffs, strict=True) if d < 0], needed)


@dataclass
class SignificanceGate:
    ok: bool
    comparisons: dict[str, Comparison]

    @property
    def verdict(self) -> str:
        verdicts = {c.verdict for c in self.comparisons.values()}
        return "fail" if "fail" in verdicts else "inconclusive" if "inconclusive" in verdicts else "pass"


def significance_gate(baseline: Report, candidate: Report, margin: float = 0.02, alpha: float = 0.05, strict: bool = False,
                      scorers: list[str] | None = None, seed: int = 0) -> SignificanceGate:
    """Non-inferiority on every scorer. ``ok`` is true when every scorer passes, or, unless ``strict``, is merely inconclusive.
    A suite too small to tell never blocks a release by luck, and with ``strict`` it never lets one through by luck."""
    comps = {s: compare(baseline, candidate, s, margin, alpha, seed) for s in (scorers or list(baseline.scores))}
    g = SignificanceGate(False, comps)
    g.ok = g.verdict == "pass" or (g.verdict == "inconclusive" and not strict)
    return g
