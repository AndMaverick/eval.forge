# eval.forge

An evaluation harness that knows how much a number can be trusted.

Most eval setups end in a single score and a hopeful comparison: v2 got 0.83, v1 got 0.85, ship it or don't. With 40 cases that difference is noise, and with 4,000 it may be real, and the harness never tells you which. eval.forge reports an interval on every score, gates changes with a noninferiority test that can answer **pass**, **fail** or **inconclusive** (and says how many more cases would settle it), refuses to compare runs from different test sets, and checks your grader and your dataset for the two classic ways evals lie. Standard library only.

```
pip install git+https://github.com/AndMaverick/eval.forge
```

## What it does

```python
from eval_forge import Case, Suite, contains, exact, run, compare

suite = Suite("support", [
    Case("r1", "refund window?", "30 days", tags=("refund",)),
    Case("a1", "ما هي مدة الاسترداد؟", ["30 يوما", "30 days"], tags=("arabic",)),
], {"exact": exact, "contains": contains})

baseline = run(suite, current_system, repeats=3, workers=8)
candidate = run(suite, new_system, repeats=3, workers=8)

result = compare(baseline, candidate, "exact", margin=0.02)
print(result.verdict)        # pass | fail | inconclusive
print(result.cases_needed)   # when inconclusive: roughly how many paired cases would settle it
```

- **Intervals, not bare means.** Wilson interval for pass/fail scores, bootstrap otherwise. `report.interval("exact")`.
- **Noninferiority gate.** "Is the candidate no worse than the baseline by more than the margin?" is tested on the paired differences per case. *Pass* means shown within the margin, *fail* means shown worse, *inconclusive* means the data cannot tell. A small suite never blocks a release by luck, and with `strict` it never lets one through by luck.
- **McNemar for pass/fail, permutation test otherwise**, with the regressed and improved case ids listed.
- **Slices.** Tag cases (`arabic`, `long`, `refund`) and `report.by_tag("exact")` shows the worst slice first, with its case count, so an average that hides a collapse in one segment does not.
- **Same suite or no comparison.** Every report carries a digest of the cases and scorer names. Comparing runs from different suites raises instead of returning a number.
- **Crash safe, repeatable.** A crash or a scorer error scores zero, is recorded, and the run continues. `retries` calls again after a crash. `repeats` averages a nondeterministic system and lists the flaky cases. `workers` runs concurrently with results in case order.
- **Cost and tokens.** Return `Result(output, cost, tokens)` from the system under test and the report carries them.
- **Scorers.** `exact`, `exact_normalized`, `contains`, `not_contains`, `regex`, `numeric_close`, `f1_tokens`, `jaccard`, `edit_similarity`, `json_valid`, `json_fields`, `json_equal`, and `all_of` / `any_of` / `weighted` to combine them. A scorer is any function of (case, output) returning 0 to 1.
- **Model grading you can audit.** `rubric(judge, criteria)` turns any text in, text out function into a scorer, and a reply with no usable rating is an error rather than a silent grade. `agreement(judge_scores, human_scores)` measures the grader against human labels (accuracy, kappa, Spearman, bias, false passes) before you let it gate anything.
- **Dataset hygiene.** `split` is stable under hashing, so a held out set stays held out as the dataset grows. `dedupe` drops repeated inputs. `overlap` finds evaluation cases that appear in a corpus (training data, few shot examples, a prompt), because a leaked case overstates ability.

## Command line

Score outputs you already have, then compare two runs. The exit code is the gate: `0` pass, `1` fail, `2` error. An inconclusive result exits `0` unless you pass `--strict`.

```
$ python -m eval_forge score examples/cases.jsonl examples/v1_outputs.jsonl --out v1.json
examples/cases.jsonl: 120 cases, repeats 1, errors 0
  exact              mean 0.9000   95% interval [0.8333, 0.9419]   pass rate 0.9000
      slice long           0.875  (n=40)
      slice refund         0.900  (n=40)
      slice arabic         0.925  (n=40)

$ python -m eval_forge compare v1.json v2.json
exact              0.9000 -> 0.7583  diff -0.1417  bound [-0.2167, -0.0667]  p 0.006  FAIL  (regressed 26, improved 9)
verdict: fail  (gate fails)
```

`show REPORT` prints a saved report, and `overlap CASES CORPUS...` lists cases that leaked into a corpus (exit `1` if any did). The files in `examples/` produce the output above.

## Honest limits

- A noninferiority test answers a narrow question about the cases you have. It cannot tell you the suite is representative of production traffic. Slices help, but choosing the cases is still on you.
- The bootstrap bound assumes the cases are a fair sample. Very small suites (under about 20 cases) give wide, unstable bounds, which is what *inconclusive* is for.
- Model graded scores inherit the grader's bias. Run `agreement` against human labels first. A kappa under about 0.6 means the grader and people often disagree.
- `overlap` finds shared word n grams, so it catches verbatim leakage and not paraphrase.
- It does not call any model. You supply the system under test and, if you use one, the judge.

## Pairs with

[prompt.vault](https://github.com/AndMaverick/prompt.vault): use the gate as the deploy check so a prompt only reaches prod after it holds the line.

## Tests

```
python -m unittest discover -s tests -t . -v
```

62 tests, with the statistics checked against worked by hand values. Ruff and mypy strict run in CI on Python 3.10 to 3.13.

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**If this is your problem.** If a system you run can fail in ways that matter, the controls belong in the architecture, not in the postmortem. I am happy to talk through yours. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=eval.forge).
