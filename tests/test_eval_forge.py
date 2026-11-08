import unittest

from eval_forge import Case, Suite, contains, exact, gate, numeric_close, regex, run

CASES = [Case("1", "2+2", "4"), Case("2", "capital of France", "Paris"), Case("3", "3*3", "9")]


def good(text):
    return {"2+2": "4", "capital of France": "The capital is Paris", "3*3": "9"}[text]


def weaker(text):
    return {"2+2": "4", "capital of France": "Lyon", "3*3": "8"}[text]


class ScorerTests(unittest.TestCase):
    def test_exact_and_contains(self):
        case = Case("x", "q", "Paris")
        self.assertEqual(exact(case, " Paris "), 1.0)
        self.assertEqual(contains(case, "It is paris."), 1.0)
        self.assertEqual(contains(case, "Lyon"), 0.0)

    def test_regex_and_numeric(self):
        self.assertEqual(regex(r"^\d+$")(Case("x", "q"), "42"), 1.0)
        self.assertEqual(numeric_close(0.01)(Case("x", "q", "3.14"), "3.141"), 1.0)
        self.assertEqual(numeric_close()(Case("x", "q", "3"), "abc"), 0.0)


class HarnessTests(unittest.TestCase):
    def suite(self):
        return Suite("math", CASES, {"exact": exact, "contains": contains})

    def test_run_scores_every_case(self):
        report = run(self.suite(), good)
        self.assertEqual(report.mean("contains"), 1.0)
        self.assertAlmostEqual(report.mean("exact"), 2 / 3)

    def test_crash_scores_zero_and_is_recorded(self):
        def broken(text):
            raise RuntimeError("boom")

        report = run(self.suite(), broken)
        self.assertEqual(report.mean("exact"), 0.0)
        self.assertEqual(len(report.errors), 3)

    def test_gate_passes_when_equal_and_fails_on_regression(self):
        baseline = run(self.suite(), good)
        self.assertTrue(gate(baseline, run(self.suite(), good)).ok)
        result = gate(baseline, run(self.suite(), weaker))
        self.assertFalse(result.ok)
        self.assertEqual({r[0] for r in result.regressions}, {"exact", "contains"})

    def test_pass_rate(self):
        report = run(self.suite(), weaker)
        self.assertAlmostEqual(report.pass_rate("exact"), 1 / 3)


if __name__ == "__main__":
    unittest.main()
