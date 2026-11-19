import unittest

from eval_forge import (
    Case,
    all_of,
    any_of,
    contains,
    edit_similarity,
    exact,
    exact_normalized,
    f1_tokens,
    jaccard,
    json_equal,
    json_fields,
    json_valid,
    not_contains,
    numeric_close,
    regex,
    weighted,
)


def c(expected, input_="q"):
    return Case("x", input_, expected)


class ScorerTests(unittest.TestCase):
    def test_exact_and_multiple_references(self):
        self.assertEqual(exact(c("Paris"), "Paris"), 1.0)
        self.assertEqual(exact(c("Paris"), "paris"), 0.0)
        self.assertEqual(exact(c(["Paris", "Lutetia"]), "Lutetia"), 1.0)

    def test_normalized_ignores_case_space_and_punctuation(self):
        self.assertEqual(exact_normalized(c("The answer is 42."), "  the ANSWER is 42 "), 1.0)

    def test_contains_and_not_contains(self):
        self.assertEqual(contains(c("42"), "it is 42, I think"), 1.0)
        self.assertEqual(contains(c("43"), "it is 42"), 0.0)
        self.assertEqual(not_contains("password")(c(None), "all good"), 1.0)
        self.assertEqual(not_contains("password")(c(None), "my Password is"), 0.0)

    def test_regex(self):
        self.assertEqual(regex(r"^\d{3}$")(c(None), "123"), 1.0)
        self.assertEqual(regex(r"^\d{3}$")(c(None), "12a"), 0.0)

    def test_numeric_close(self):
        s = numeric_close(0.01)
        self.assertEqual(s(c(3.14), "3.145"), 1.0)
        self.assertEqual(s(c(3.14), "3.2"), 0.0)
        self.assertEqual(s(c(3.14), "pi"), 0.0)

    def test_f1_is_hand_checkable(self):
        # prediction 3 tokens, reference 4 tokens, 2 shared: p=2/3, r=2/4, f1 = 4/7
        self.assertAlmostEqual(f1_tokens(c("the cat sat down"), "the cat ran"), 4 / 7, places=6)
        self.assertEqual(f1_tokens(c("a b"), "a b"), 1.0)
        self.assertEqual(f1_tokens(c("a b"), "x y"), 0.0)

    def test_jaccard(self):
        self.assertAlmostEqual(jaccard(c("a b c"), "b c d"), 2 / 4)

    def test_edit_similarity(self):
        self.assertEqual(edit_similarity(c("kitten"), "kitten"), 1.0)
        # kitten -> sitting is distance 3 over length 7
        self.assertAlmostEqual(edit_similarity(c("kitten"), "sitting"), 1 - 3 / 7, places=6)

    def test_json_scorers(self):
        self.assertEqual(json_valid(c(None), '{"a": 1}'), 1.0)
        self.assertEqual(json_valid(c(None), "not json"), 0.0)
        self.assertEqual(json_valid(c(None), '```json\n{"a": 1}\n```'), 1.0)
        self.assertEqual(json_fields(["a", "b"])(c(None), '{"a": 1, "b": 2}'), 1.0)
        self.assertEqual(json_fields(["a", "b"])(c(None), '{"a": 1}'), 0.5)
        self.assertEqual(json_equal(c({"a": 1, "b": [1, 2]}), '{"b": [1, 2], "a": 1}'), 1.0)
        self.assertEqual(json_equal(c({"a": 1}), '{"a": 2}'), 0.0)

    def test_combinators(self):
        yes = lambda case, out: 1.0  # noqa: E731
        no = lambda case, out: 0.0  # noqa: E731
        self.assertEqual(all_of(yes, no)(c(None), ""), 0.0)
        self.assertEqual(any_of(yes, no)(c(None), ""), 1.0)
        self.assertAlmostEqual(weighted([(yes, 3), (no, 1)])(c(None), ""), 0.75)


if __name__ == "__main__":
    unittest.main()
