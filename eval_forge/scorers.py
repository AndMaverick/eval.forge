"""Scorers: functions of (case, output) that return a number, usually between 0 and 1. Any function with that shape works."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from .core import Case

Scorer = Callable[[Case, Any], float]
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def refs(case: Case) -> list[str]:
    """The acceptable answers: ``expected`` may be one value or a list of equally good references."""
    exp = case.expected
    return [str(e) for e in exp] if isinstance(exp, (list, tuple)) else [str(exp)]


def normalize(text: object) -> str:
    """NFKC, case-folded, punctuation removed, whitespace collapsed. For comparing answers, not for display."""
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(_PUNCT.sub(" ", folded).split())


def exact(case: Case, output: Any) -> float:
    return 1.0 if any(str(output).strip() == r.strip() for r in refs(case)) else 0.0


def contains(case: Case, output: Any) -> float:
    return 1.0 if any(r.lower() in str(output).lower() for r in refs(case)) else 0.0


def regex(pattern: str) -> Scorer:
    compiled = re.compile(pattern)
    return lambda case, output: 1.0 if compiled.search(str(output)) else 0.0


def numeric_close(tolerance: float = 1e-6) -> Scorer:
    def scorer(case: Case, output: Any) -> float:
        try:
            return 1.0 if abs(float(output) - float(case.expected)) <= tolerance else 0.0
        except (TypeError, ValueError):
            return 0.0
    return scorer


def exact_normalized(case: Case, output: Any) -> float:
    return 1.0 if normalize(output) in {normalize(r) for r in refs(case)} else 0.0


def not_contains(*banned: str) -> Scorer:
    """1.0 unless the output contains any banned string. For leaks, refusals that should not happen, forbidden claims."""
    low = [b.lower() for b in banned]
    return lambda case, output: 0.0 if any(b in str(output).lower() for b in low) else 1.0


def _tokens(text: object) -> list[str]:
    return normalize(text).split()


def f1_tokens(case: Case, output: Any) -> float:
    """Token-overlap F1 against the best reference (the SQuAD measure). Rewards partial answers."""
    pred = _tokens(output)
    best = 0.0
    for r in refs(case):
        gold = _tokens(r)
        common = sum((Counter(pred) & Counter(gold)).values())
        if not pred or not gold:
            best = max(best, 1.0 if pred == gold else 0.0)
        elif common:
            p, rec = common / len(pred), common / len(gold)
            best = max(best, 2 * p * rec / (p + rec))
    return best


def jaccard(case: Case, output: Any) -> float:
    pred = set(_tokens(output))
    best = 0.0
    for r in refs(case):
        gold = set(_tokens(r))
        union = pred | gold
        best = max(best, len(pred & gold) / len(union) if union else 1.0)
    return best


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(case: Case, output: Any) -> float:
    """1 minus the normalised edit distance to the closest reference. 1.0 is identical."""
    out = str(output)
    best = 0.0
    for r in refs(case):
        longest = max(len(out), len(r))
        best = max(best, 1.0 if longest == 0 else 1 - _levenshtein(out, r) / longest)
    return best


def parse_json(output: Any) -> Any:
    """Parse JSON from a model's output, accepting a fenced block or surrounding prose. Raises ``ValueError`` if there is none."""
    if isinstance(output, (dict, list)):
        return output
    text = str(output).strip()
    for candidate in [text, *_FENCE.findall(text)]:
        try:
            return json.loads(candidate.strip())
        except ValueError:
            continue
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    raise ValueError("no JSON found")


def json_valid(case: Case, output: Any) -> float:
    try:
        parse_json(output)
        return 1.0
    except ValueError:
        return 0.0


def json_fields(required: Sequence[str]) -> Scorer:
    """The fraction of required top-level keys present in the output's JSON object. 0.0 if it is not a JSON object."""
    def scorer(case: Case, output: Any) -> float:
        try:
            data = parse_json(output)
        except ValueError:
            return 0.0
        if not isinstance(data, dict) or not required:
            return 0.0
        return sum(k in data for k in required) / len(required)
    return scorer


def json_equal(case: Case, output: Any) -> float:
    """1.0 when the output's JSON equals the expected JSON (a dict, list, or JSON string), ignoring key order."""
    try:
        got = parse_json(output)
        want = case.expected if isinstance(case.expected, (dict, list)) else json.loads(str(case.expected))
    except ValueError:
        return 0.0
    return 1.0 if got == want else 0.0


def all_of(*scorers: Scorer) -> Scorer:
    """The minimum of several scorers: every condition must hold."""
    return lambda case, output: min(s(case, output) for s in scorers)


def any_of(*scorers: Scorer) -> Scorer:
    return lambda case, output: max(s(case, output) for s in scorers)


def weighted(parts: Sequence[tuple[Scorer, float]]) -> Scorer:
    """A weighted average of scorers. Weights need not sum to 1."""
    total = sum(w for _, w in parts)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return lambda case, output: sum(s(case, output) * w for s, w in parts) / total


BY_NAME: dict[str, Scorer] = {
    "exact": exact, "contains": contains, "exact_normalized": exact_normalized, "f1": f1_tokens, "jaccard": jaccard,
    "edit": edit_similarity, "json_valid": json_valid, "json_equal": json_equal,
}
