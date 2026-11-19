"""Datasets: reading and writing cases, stable splits, de-duplication, and a leakage check."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .core import Case
from .scorers import normalize


def load_cases(text: str) -> list[Case]:
    """Cases from JSONL: one object per line with ``id``, ``input``, and optionally ``expected``, ``tags``, ``meta``."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out.append(Case(str(d["id"]), str(d["input"]), d.get("expected"), tuple(d.get("tags", ())), dict(d.get("meta", {}))))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"line {n}: not a valid case ({exc})") from None
    return out


def dump_cases(cases: Iterable[Case]) -> str:
    return "".join(json.dumps(c.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for c in cases)


def split(cases: list[Case], test_fraction: float = 0.2, seed: str = "") -> tuple[list[Case], list[Case]]:
    """Split by hashing each id with the seed. The assignment of a case never changes when other cases are added or removed,
    so a held-out set stays held out as the dataset grows. Returns (train, test)."""
    if not 0 <= test_fraction <= 1:
        raise ValueError("test_fraction must be between 0 and 1")
    train: list[Case] = []
    test: list[Case] = []
    for c in cases:
        bucket = int(hashlib.sha256(f"{seed}:{c.id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (test if bucket < test_fraction else train).append(c)
    return train, test


def dedupe(cases: list[Case]) -> tuple[list[Case], list[tuple[str, str]]]:
    """Drop cases whose normalised input repeats an earlier one. Returns (kept, [(dropped id, kept id)])."""
    seen: dict[str, str] = {}
    kept, dropped = [], []
    for c in cases:
        key = normalize(c.input)
        if key in seen:
            dropped.append((c.id, seen[key]))
        else:
            seen[key] = c.id
            kept.append(c)
    return kept, dropped


def _grams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = normalize(text).split()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def overlap(corpus: Iterable[str], cases: list[Case], n: int = 8) -> dict[str, int]:
    """Which evaluation cases appear in a text corpus (training data, few-shot examples, a prompt): the count of shared
    n-grams of words for each case that shares any. A case with overlap has leaked, and its score overstates ability.
    Inputs and expected answers both count. Cases shorter than ``n`` words are matched whole."""
    if n < 1:
        raise ValueError("n must be at least 1")
    corpus_texts = [normalize(t) for t in corpus]
    index: set[tuple[str, ...]] = set()
    for t in corpus_texts:
        toks = t.split()
        index.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    blob = " ".join(corpus_texts)
    hits: dict[str, int] = {}
    for c in cases:
        found = 0
        for text in [c.input, *([str(e) for e in c.expected] if isinstance(c.expected, (list, tuple)) else [str(c.expected)] if c.expected is not None else [])]:
            toks = normalize(text).split()
            if 0 < len(toks) < n:
                found += 1 if " ".join(toks) in blob else 0
            else:
                found += len(_grams(text, n) & index)
        if found:
            hits[c.id] = found
    return hits
