import unittest

from eval_forge import (
    Case,
    Report,
    Result,
    Suite,
    compare,
    contains,
    exact,
    gate,
    run,
    significance_gate,
)


def suite(n=40, tags=True):
    cases = [Case(f"c{i}", f"q{i}", f"a{i}", ("even",) if i % 2 == 0 else ("odd",)) for i in range(n)]
    return Suite("demo", cases, {"exact": exact})


def answer_with(wrong: set[int]):
    def system(text: str) -> str:
        i = int(text[1:])
        return "wrong" if i in wrong else f"a{i}"
    return system


class RunTests(unittest.TestCase):
    def test_basic_scores_and_slices(self):
        r = run(suite(10), answer_with({1, 3}))
        self.assertAlmostEqual(r.mean("exact"), 0.8)
        self.assertEqual(r.failures("exact"), ["c1", "c3"])
        slices = r.by_tag("exact")
        self.assertEqual(slices["odd"], (0.6, 5))
        self.assertEqual(slices["even"], (1.0, 5))
        self.assertEqual(list(slices)[0], "odd")  # worst first

    def test_crash_scores_zero_and_run_continues(self):
        def system(text):
            if text == "q2":
                raise RuntimeError("boom")
            return "a" + text[1:]
        r = run(suite(5), system)
        self.assertEqual(r.scores["exact"]["c2"], 0.0)
        self.assertIn("RuntimeError: boom", r.errors["c2"])
        self.assertEqual(r.scores["exact"]["c3"], 1.0)

    def test_retries_recover_a_flaky_call(self):
        calls = {"n": 0}

        def system(text):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("slow")
            return "a" + text[1:]
        r = run(suite(2), system, retries=1)
        self.assertEqual(r.mean("exact"), 1.0)
        self.assertEqual(r.errors, {})

    def test_scorer_error_is_recorded_not_raised(self):
        def bad(case, output):
            raise ValueError("bad scorer")
        r = run(Suite("s", [Case("a", "q", "x")], {"bad": bad}), lambda t: "x")
        self.assertEqual(r.scores["bad"]["a"], 0.0)
        self.assertIn("scorer bad", r.errors["a"])

    def test_repeats_average_and_flag_flaky(self):
        state = {"n": 0}

        def system(text):
            state["n"] += 1
            return "a0" if state["n"] % 2 else "nope"
        r = run(Suite("s", [Case("c0", "q0", "a0")], {"exact": exact}), system, repeats=4)
        self.assertAlmostEqual(r.scores["exact"]["c0"], 0.5)
        self.assertEqual(r.flaky, ["c0"])

    def test_workers_give_the_same_report_as_serial(self):
        s = suite(30)
        a = run(s, answer_with({4, 9}), workers=1)
        b = run(s, answer_with({4, 9}), workers=6)
        self.assertEqual(a.scores, b.scores)

    def test_result_wrapper_carries_cost(self):
        r = run(Suite("s", [Case("a", "q", "x")], {"exact": exact}), lambda t: Result("x", cost=0.25, tokens=10))
        self.assertEqual(r.cost["a"], 0.25)
        self.assertEqual(r.mean("exact"), 1.0)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            Suite("s", [Case("a", "q"), Case("a", "r")])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            run(suite(2), answer_with(set()), repeats=0)

    def test_interval_wilson_for_binary_bootstrap_otherwise(self):
        r = run(suite(20), answer_with({0, 1, 2, 3}))
        lo, hi = r.interval("exact")
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)
        s = Suite("s", [Case(str(i), f"q{i}", "a b c d") for i in range(10)], {"f": lambda c, o: 0.5})
        lo2, hi2 = run(s, lambda t: "x").interval("f")
        self.assertAlmostEqual(lo2, 0.5)

    def test_report_round_trip(self):
        r = run(suite(6), answer_with({2}), keep_outputs=True)
        r2 = Report.from_dict(r.to_dict())
        self.assertEqual(r2.scores, r.scores)
        self.assertEqual(r2.digest, r.digest)
        self.assertEqual(r2.tags, r.tags)

    def test_digest_changes_with_cases_and_scorers(self):
        a = suite(5)
        b = suite(6)
        c = Suite("demo", a.cases, {"exact": exact, "contains": contains})
        self.assertEqual(a.digest(), suite(5).digest())
        self.assertNotEqual(a.digest(), b.digest())
        self.assertNotEqual(a.digest(), c.digest())


class CompareTests(unittest.TestCase):
    def test_identical_runs_pass(self):
        s = suite(200)
        a = run(s, answer_with({1, 2, 3}))
        c = compare(a, a, "exact", margin=0.02)
        self.assertEqual(c.verdict, "pass")
        self.assertEqual(c.diff, 0.0)

    def test_large_regression_fails(self):
        s = suite(200)
        base = run(s, answer_with(set()))
        cand = run(s, answer_with(set(range(0, 200, 2))))  # half wrong
        c = compare(base, cand, "exact", margin=0.02)
        self.assertEqual(c.verdict, "fail")
        self.assertEqual(len(c.regressed), 100)
        self.assertLess(c.p_value, 0.001)

    def test_small_suite_is_inconclusive_with_a_number_to_fix_it(self):
        s = suite(20)
        base = run(s, answer_with(set()))
        cand = run(s, answer_with({3}))  # one regression in twenty
        c = compare(base, cand, "exact", margin=0.02)
        self.assertEqual(c.verdict, "inconclusive")
        self.assertIsNotNone(c.cases_needed)
        self.assertGreater(c.cases_needed or 0, 20)

    def test_improvement_passes(self):
        s = suite(100)
        base = run(s, answer_with(set(range(30))))
        cand = run(s, answer_with(set(range(10))))
        c = compare(base, cand, "exact")
        self.assertEqual(c.verdict, "pass")
        self.assertEqual(len(c.improved), 20)

    def test_different_suites_refuse_to_compare(self):
        a = run(suite(10), answer_with(set()))
        b = run(suite(11), answer_with(set()))
        with self.assertRaises(ValueError):
            compare(a, b, "exact")
        compare(a, b, "exact", require_same_suite=False)  # explicit opt-out still works

    def test_unknown_scorer(self):
        a = run(suite(5), answer_with(set()))
        with self.assertRaises(KeyError):
            compare(a, a, "nope")

    def test_compare_is_deterministic(self):
        s = suite(60)
        a, b = run(s, answer_with({1})), run(s, answer_with({1, 5, 7, 9}))
        self.assertEqual(compare(a, b, "exact", seed=2), compare(a, b, "exact", seed=2))

    def test_significance_gate_strict_versus_lenient(self):
        s = suite(20)
        base = run(s, answer_with(set()))
        cand = run(s, answer_with({3}))
        lenient = significance_gate(base, cand)
        strict = significance_gate(base, cand, strict=True)
        self.assertEqual(lenient.verdict, "inconclusive")
        self.assertTrue(lenient.ok)
        self.assertFalse(strict.ok)
        bad = significance_gate(run(suite(200), answer_with(set())), run(suite(200), answer_with(set(range(0, 200, 2)))))
        self.assertFalse(bad.ok)
        self.assertEqual(bad.verdict, "fail")

    def test_legacy_gate_still_works(self):
        s = suite(10)
        base = run(s, answer_with(set()))
        cand = run(s, answer_with({1, 2}))
        self.assertFalse(gate(base, cand, 0.05).ok)
        self.assertTrue(gate(base, base, 0.0).ok)


if __name__ == "__main__":
    unittest.main()
