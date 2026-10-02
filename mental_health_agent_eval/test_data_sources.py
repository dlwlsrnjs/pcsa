import unittest

import data_sources
import pcsa


class DataSourceTests(unittest.TestCase):
    def test_goals_cover_all_axes(self):
        goals = data_sources.load_attack_goals()
        self.assertEqual({g["axis"] for g in goals}, set(pcsa.PCSA_AXES))

    def test_fallback_personas_load(self):
        personas, source = data_sources.load_personas()
        self.assertTrue(personas)
        self.assertIn(source, {"corpus_derived", "synthetic_fallback"})
        self.assertIn("cognitive_distortion", personas[0])

    def test_build_cases_is_deterministic_and_axis_scoped(self):
        personas, _ = data_sources.load_personas()
        goals = data_sources.load_attack_goals()
        a = data_sources.build_cases("toxic_empathy", personas, goals, 3, seed=1)
        b = data_sources.build_cases("toxic_empathy", personas, goals, 3, seed=1)
        self.assertEqual([c["case_id"] for c in a], ["toxic_empathy-000", "toxic_empathy-001", "toxic_empathy-002"])
        self.assertTrue(all(c["goal"]["axis"] == "toxic_empathy" for c in a))
        self.assertEqual([c["goal"]["goal_id"] for c in a], [c["goal"]["goal_id"] for c in b])

    def test_calibration_probes_have_constructs(self):
        probes = data_sources.load_calibration_probes()
        self.assertTrue(all("construct" in p and "text" in p for p in probes))


if __name__ == "__main__":
    unittest.main()
