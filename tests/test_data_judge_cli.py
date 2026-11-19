import contextlib
import io
import json
import os
import tempfile
import unittest

from eval_forge import Case, Suite, agreement, dedupe, dump_cases, load_cases, overlap, rubric, run, split
from eval_forge.cli import main


class DataTests(unittest.TestCase):
    def cases(self, n=200):
        return [Case(f"id{i}", f"question number {i}", f"a{i}") for i in range(n)]

    def test_split_is_stable_and_disjoint(self):
        cs = self.cases()
        tr, te = split(cs, 0.2, seed="s")
        self.assertEqual(len(tr) + len(te), len(cs))
        self.assertFalse({c.id for c in tr} & {c.id for c in te})
        self.assertAlmostEqual(len(te) / len(cs), 0.2, delta=0.07)
        # adding cases never moves an existing case between train and test
        tr2, te2 = split(cs + [Case("new", "extra")], 0.2, seed="s")
        self.assertTrue({c.id for c in te} <= {c.id for c in te2})
        self.assertNotEqual({c.id for c in split(cs, 0.2, seed="other")[1]}, {c.id for c in te})

    def test_split_bounds(self):
        with self.assertRaises(ValueError):
            split(self.cases(3), 1.5)
        self.assertEqual(split(self.cases(5), 0.0)[1], [])

    def test_dedupe_normalises(self):
        cs = [Case("a", "What is 2+2?"), Case("b", "what is 2+2"), Case("c", "other")]
        kept, dropped = dedupe(cs)
        self.assertEqual([c.id for c in kept], ["a", "c"])
        self.assertEqual(dropped, [("b", "a")])

    def test_round_trip_keeps_unicode_and_fields(self):
        cs = [Case("1", "ما هي عاصمة الإمارات؟", ["أبوظبي", "Abu Dhabi"], ("arabic",), {"src": "x"})]
        back = load_cases(dump_cases(cs))
        self.assertEqual(back[0].input, cs[0].input)
        self.assertEqual(back[0].expected, cs[0].expected)
        self.assertEqual(back[0].tags, ("arabic",))
        self.assertEqual(back[0].meta, {"src": "x"})

    def test_load_reports_the_bad_line(self):
        with self.assertRaises(ValueError) as cm:
            load_cases('{"id": "1", "input": "ok"}\n{"input": "no id"}\n')
        self.assertIn("line 2", str(cm.exception))

    def test_overlap_finds_leaked_cases_only(self):
        corpus = ["Training notes: the capital of the united arab emirates is abu dhabi and it is large"]
        cs = [Case("leak", "what is the capital of the united arab emirates is abu dhabi and it is large"),
              Case("clean", "completely different question about tax residency rules in germany today")]
        hits = overlap(corpus, cs, n=8)
        self.assertIn("leak", hits)
        self.assertNotIn("clean", hits)
        with self.assertRaises(ValueError):
            overlap(corpus, cs, n=0)


class JudgeTests(unittest.TestCase):
    def test_rubric_maps_rating_to_unit_interval(self):
        s = rubric(lambda prompt: "I give it 4", "be polite", scale=5)
        self.assertAlmostEqual(s(Case("a", "q", "r"), "out"), 0.75)
        self.assertEqual(rubric(lambda p: "1", "x")(Case("a", "q"), "o"), 0.0)
        self.assertEqual(rubric(lambda p: "5", "x")(Case("a", "q"), "o"), 1.0)

    def test_unparseable_judge_reply_is_an_error_not_a_grade(self):
        suite = Suite("s", [Case("a", "q", "r")], {"j": rubric(lambda p: "no idea", "x")})
        r = run(suite, lambda t: "out")
        self.assertEqual(r.scores["j"]["a"], 0.0)
        self.assertIn("a", r.errors)

    def test_out_of_range_numbers_are_skipped(self):
        self.assertAlmostEqual(rubric(lambda p: "score 9 of 10, so 3", "x", 5)(Case("a", "q"), "o"), 0.5)

    def test_agreement_on_a_perfect_and_a_generous_judge(self):
        human = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        perfect = agreement(human, human)
        self.assertEqual(perfect.accuracy, 1.0)
        self.assertAlmostEqual(perfect.kappa, 1.0)
        generous = agreement([1.0] * 8, human)
        self.assertEqual(generous.false_pass, 3)
        self.assertEqual(generous.false_fail, 0)
        self.assertGreater(generous.bias, 0)
        self.assertLess(generous.kappa, 0.1)

    def test_agreement_input_checks(self):
        with self.assertRaises(ValueError):
            agreement([1.0], [1.0])
        with self.assertRaises(ValueError):
            agreement([1.0, 0.0], [1.0])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def write(self, name, text):
        with open(self.path(name), "w", encoding="utf-8") as fh:
            fh.write(text)
        return self.path(name)

    def invoke(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def make_report(self, name, wrong, n=200):
        cases = "".join(json.dumps({"id": f"c{i}", "input": f"q{i}", "expected": f"a{i}", "tags": ["t"]}) + "\n" for i in range(n))
        outputs = "".join(json.dumps({"id": f"c{i}", "output": "x" if i in wrong else f"a{i}"}) + "\n" for i in range(n))
        cp, op, rp = self.write("cases.jsonl", cases), self.write(f"{name}.out.jsonl", outputs), self.path(f"{name}.json")
        code, out, _ = self.invoke("score", cp, op, "--scorers", "exact", "--out", rp)
        self.assertEqual(code, 0)
        return rp, out

    def test_score_show_and_compare(self):
        base, out = self.make_report("base", set())
        self.assertIn("exact", out)
        same, _ = self.make_report("same", set())
        bad, _ = self.make_report("bad", set(range(0, 200, 2)))
        code, out, _ = self.invoke("compare", base, same)
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)
        code, out, _ = self.invoke("compare", base, bad)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)
        code, out, _ = self.invoke("show", bad)
        self.assertEqual(code, 0)
        self.assertIn("0.5000", out)

    def test_inconclusive_exit_codes(self):
        base, _ = self.make_report("b2", set(), n=20)
        cand, _ = self.make_report("c2", {3}, n=20)
        code, out, _ = self.invoke("compare", base, cand)
        self.assertEqual(code, 0)
        self.assertIn("INCONCLUSIVE", out)
        code, _, _ = self.invoke("compare", base, cand, "--strict")
        self.assertEqual(code, 1)

    def test_compare_refuses_different_suites(self):
        a, _ = self.make_report("a3", set(), n=20)
        b, _ = self.make_report("b3", set(), n=21)
        code, _, err = self.invoke("compare", a, b)
        self.assertEqual(code, 2)
        self.assertIn("different suites", err)

    def test_overlap_command(self):
        cases = self.write("oc.jsonl", json.dumps({"id": "leak", "input": "one two three four five six seven eight nine"}) + "\n")
        corpus = self.write("corpus.txt", "intro one two three four five six seven eight nine outro")
        code, out, _ = self.invoke("overlap", cases, corpus)
        self.assertEqual(code, 1)
        self.assertIn("leak", out)

    def test_errors_are_clean(self):
        self.assertEqual(self.invoke("show", self.path("missing.json"))[0], 2)
        cp = self.write("c.jsonl", '{"id": "1", "input": "q"}\n')
        op = self.write("o.jsonl", '{"id": "1", "output": "x"}\n')
        code, _, err = self.invoke("score", cp, op, "--scorers", "nope")
        self.assertEqual(code, 2)
        self.assertIn("unknown scorer", err)


if __name__ == "__main__":
    unittest.main()
