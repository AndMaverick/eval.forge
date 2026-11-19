import math
import unittest

from eval_forge import stats


class StatsTests(unittest.TestCase):
    def test_wilson_matches_hand_worked_value(self):
        # 8 of 10 at 95%: the textbook Wilson interval is about [0.490, 0.943]
        lo, hi = stats.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)

    def test_wilson_never_leaves_unit_interval(self):
        for k in (0, 1, 9, 10):
            lo, hi = stats.wilson(k, 10)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
        self.assertEqual(stats.wilson(0, 0), (0.0, 1.0))

    def test_mcnemar_exact(self):
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        # 0 vs 5 discordant pairs: two-sided exact p = 2 * (1/32) = 0.0625
        self.assertAlmostEqual(stats.mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(stats.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact(1, 6), stats.mcnemar_exact(6, 1))

    def test_bootstrap_is_seeded_and_brackets_the_mean(self):
        values = [0.2, 0.4, 0.5, 0.9, 1.0, 0.7, 0.6, 0.3]
        a = stats.bootstrap_ci(values, seed=3)
        self.assertEqual(a, stats.bootstrap_ci(values, seed=3))
        self.assertLess(a[0], stats.mean(values))
        self.assertGreater(a[1], stats.mean(values))

    def test_constant_values_give_a_point_interval(self):
        lo, hi = stats.bootstrap_ci([0.5] * 20)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_permutation_detects_a_consistent_shift_and_not_noise(self):
        self.assertLess(stats.permutation_p([0.1] * 12, seed=1), 0.01)
        self.assertGreater(stats.permutation_p([0.1, -0.1] * 6, seed=1), 0.5)

    def test_kappa(self):
        same = [True, False, True, False, True, True]
        self.assertAlmostEqual(stats.cohens_kappa(same, same), 1.0)
        # opposite labels on a balanced set: perfectly anti-correlated
        self.assertAlmostEqual(stats.cohens_kappa([True, False] * 5, [False, True] * 5), -1.0)

    def test_spearman(self):
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        # ties are handled with average ranks, so this stays finite
        self.assertTrue(math.isfinite(stats.spearman([1, 1, 2, 3], [1, 2, 2, 3])))

    def test_cases_needed_grows_as_margin_shrinks(self):
        a = stats.cases_needed(0.3, 0.05)
        b = stats.cases_needed(0.3, 0.02)
        self.assertGreater(b, a)
        self.assertGreater(a, 10)


if __name__ == "__main__":
    unittest.main()
