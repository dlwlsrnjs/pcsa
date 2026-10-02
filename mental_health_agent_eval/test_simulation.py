import unittest

from simulate_experiment import probability, stable_uniform


class SimulationTests(unittest.TestCase):
    def test_paired_draw_is_deterministic(self):
        self.assertEqual(stable_uniform(1, "t", "c"), stable_uniform(1, "t", "c"))

    def test_profile_aware_focuses_high_weakness(self):
        self.assertGreater(probability("profile_aware", .8), probability("profile_blind", .8))
        self.assertEqual(probability("profile_aware", .2), probability("profile_blind", .2))

    def test_probabilities_are_valid(self):
        for condition in ("fixed", "profile_blind", "profile_aware"):
            for weakness in (0, .5, 1):
                self.assertTrue(0 <= probability(condition, weakness) <= 1)


if __name__ == "__main__":
    unittest.main()
