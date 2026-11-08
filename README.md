# eval.forge

The smallest evaluation harness that still does the job: cases, pluggable scorers, a suite runner that survives crashes, and a regression gate you can put in front of any model or prompt change. Standard library only.

## Use

```python
from eval_forge import Case, Suite, contains, exact, gate, run

suite = Suite("support", [Case("1", "refund window?", "30 days")],
              {"exact": exact, "contains": contains})

baseline = run(suite, current_system)
candidate = run(suite, new_system)

result = gate(baseline, candidate, max_drop=0.02)
assert result.ok, result.regressions
```

## Design

- **Scorers.** `exact`, `contains`, `regex(pattern)`, and `numeric_close(tolerance)`. A scorer is any function of (case, output) returning a number, so rubric or model-graded scorers plug in unchanged.
- **Crash safe.** If the system under test raises, that case scores zero, the error is recorded in the report, and the run continues. A flaky dependency should not hide a regression.
- **Regression gate.** Compares each scorer's mean against a baseline and reports exactly which scorers dropped and by how much. A scorer missing from the candidate counts as a regression.
- **Pairs with** [prompt.vault](https://github.com/AndMaverick/prompt.vault): use the gate as the deploy check so a prompt only reaches prod after it holds the line.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).
