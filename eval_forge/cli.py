"""``python -m eval_forge``: score outputs you already have, compare runs, check for leakage. Local only."""
from __future__ import annotations

import argparse
import json
import sys

from .compare import significance_gate
from .core import Report, Suite, run
from .data import load_cases, overlap
from .scorers import BY_NAME


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _load_report(path: str) -> Report:
    return Report.from_dict(json.loads(_read(path)))


def _show(report: Report, scorer: str | None) -> None:
    names = [scorer] if scorer else list(report.scores)
    print(f"{report.suite}: {len(next(iter(report.scores.values()), {}))} cases, repeats {report.repeats}, errors {len(report.errors)}")
    for name in names:
        lo, hi = report.interval(name)
        print(f"  {name:<18} mean {report.mean(name):.4f}   95% interval [{lo:.4f}, {hi:.4f}]   pass rate {report.pass_rate(name):.4f}")
        for tag, (m, n) in list(report.by_tag(name, min_cases=2).items())[:5]:
            print(f"      slice {tag:<14} {m:.3f}  (n={n})")
    if report.flaky:
        print(f"  flaky cases: {', '.join(report.flaky[:10])}")
    total = sum(report.cost.values())
    if total:
        print(f"  total cost {total:.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_forge", description="Evaluation reports with honest uncertainty.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score a JSONL of outputs against a JSONL of cases")
    sc.add_argument("cases")
    sc.add_argument("outputs", help="JSONL with id, output")
    sc.add_argument("--scorers", default="exact", help="comma-separated: " + ", ".join(BY_NAME))
    sc.add_argument("--out", help="write the report JSON here")
    sh = sub.add_parser("show")
    sh.add_argument("report")
    sh.add_argument("--scorer")
    cp = sub.add_parser("compare", help="is the candidate no worse than the baseline by more than the margin")
    cp.add_argument("baseline")
    cp.add_argument("candidate")
    cp.add_argument("--scorer", action="append", help="repeat to check several (default: all)")
    cp.add_argument("--margin", type=float, default=0.02)
    cp.add_argument("--alpha", type=float, default=0.05)
    cp.add_argument("--strict", action="store_true", help="an inconclusive result fails the gate")
    ov = sub.add_parser("overlap", help="find evaluation cases that appear in a text corpus")
    ov.add_argument("cases")
    ov.add_argument("corpus", nargs="+")
    ov.add_argument("--n", type=int, default=8)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "score":
            cases = load_cases(_read(args.cases))
            outputs = {}
            for line in _read(args.outputs).splitlines():
                if line.strip():
                    d = json.loads(line)
                    outputs[str(d["id"])] = d["output"]
            names = [n for n in args.scorers.split(",") if n]
            unknown = [n for n in names if n not in BY_NAME]
            if unknown:
                print(f"error: unknown scorer {unknown[0]!r}", file=sys.stderr)
                return 2
            suite = Suite(args.cases, cases, {n: BY_NAME[n] for n in names})
            by_input = {c.input: outputs[c.id] for c in cases if c.id in outputs}

            def system(text: str) -> object:
                if text not in by_input:
                    raise LookupError("no output for this case")
                return by_input[text]

            report = run(suite, system)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fh:
                    json.dump(report.to_dict(), fh, indent=1, default=str)
            _show(report, None)
            return 0
        if args.cmd == "show":
            _show(_load_report(args.report), args.scorer)
            return 0
        if args.cmd == "compare":
            g = significance_gate(_load_report(args.baseline), _load_report(args.candidate), args.margin, args.alpha, args.strict, args.scorer)
            for name, c in g.comparisons.items():
                print(f"{name:<18} {c.baseline_mean:.4f} -> {c.candidate_mean:.4f}  diff {c.diff:+.4f}  bound [{c.lower:+.4f}, {c.upper:+.4f}]  "
                      f"p {c.p_value:.3f}  {c.verdict.upper()}  (regressed {len(c.regressed)}, improved {len(c.improved)})")
                if c.cases_needed:
                    print(f"    inconclusive at n={c.n}: roughly {c.cases_needed} paired cases would settle it")
            print(f"verdict: {g.verdict}" + ("" if g.ok else "  (gate fails)"))
            return 0 if g.ok else 1
        hits = overlap([_read(p) for p in args.corpus], load_cases(_read(args.cases)), args.n)
        for cid, count in hits.items():
            print(f"{cid}: {count} shared {args.n}-gram(s)")
        print(f"{len(hits)} case(s) overlap the corpus")
        return 1 if hits else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
