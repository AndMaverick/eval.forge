"""Statistics for evaluation: intervals, paired tests, agreement. Standard library only, seeded where random."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import NormalDist

_NORMAL = NormalDist()


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(math.fsum((v - m) ** 2 for v in values) / (len(values) - 1))


def wilson(successes: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Behaves at 0 and n, unlike the textbook normal interval."""
    if n <= 0:
        return 0.0, 1.0
    z = _NORMAL.inv_cdf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95, resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(values, k=n)) / n for _ in range(resamples))
    lo = means[int((1 - confidence) / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 + confidence) / 2 * resamples))]
    return lo, hi


def bootstrap_lower(values: Sequence[float], alpha: float = 0.05, resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """One-sided bounds at level 1 - alpha: (lower bound, upper bound), each holding with probability 1 - alpha on its own."""
    n = len(values)
    if n == 0:
        return -math.inf, math.inf
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return means[int(alpha * resamples)], means[min(resamples - 1, int((1 - alpha) * resamples))]


def permutation_p(diffs: Sequence[float], resamples: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test that the mean paired difference is zero. Exact when it is cheap to be."""
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 1.0
    observed = abs(math.fsum(nonzero))
    m = len(nonzero)
    if m <= 14:  # enumerate every sign pattern
        hits = 0
        for mask in range(1 << m):
            s = math.fsum(d if mask >> i & 1 else -d for i, d in enumerate(nonzero))
            hits += abs(s) >= observed - 1e-12
        return hits / (1 << m)
    rng = random.Random(seed)
    hits = sum(abs(math.fsum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - 1e-12 for _ in range(resamples))
    return (hits + 1) / (resamples + 1)


def mcnemar_exact(base_only: int, cand_only: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs of two binary results: cases only the baseline passed and
    cases only the candidate passed. Small p means the two systems really differ."""
    n = base_only + cand_only
    if n == 0:
        return 1.0
    k = min(base_only, cand_only)
    tail = math.fsum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def cases_needed(sd_diff: float, margin: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired cases needed to show a candidate is within ``margin`` of a baseline when the true difference is zero,
    given the standard deviation of per-case differences. Use it to size a suite before trusting a gate."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    z = _NORMAL.inv_cdf(1 - alpha) + _NORMAL.inv_cdf(power)
    return max(2, math.ceil((z * sd_diff / margin) ** 2))


def cohens_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    """Chance-corrected agreement between two raters over the same items. 1 is perfect, 0 is chance."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((sum(x == lab for x in a) / n) * (sum(y == lab for y in b) / n) for lab in labels)
    return 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation with ties handled by average ranks. Zero when either side has no variation."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equal-length lists of at least two values")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = mean(ra), mean(rb)
    num = math.fsum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = math.sqrt(math.fsum((x - ma) ** 2 for x in ra) * math.fsum((y - mb) ** 2 for y in rb))
    return 0.0 if den == 0 else num / den
