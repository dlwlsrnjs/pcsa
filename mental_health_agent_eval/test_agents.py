import unittest

from agents import AXES, profile_target, select_approved_message


class AgentTests(unittest.TestCase):
    def test_mock_profile_is_complete(self):
        profile, observations = profile_target(None, None)
        self.assertEqual(len(profile), 4)
        self.assertEqual(len(observations), 4)
        self.assertTrue(all(value == 1.0 for value in profile.values()))

    def test_profile_aware_uses_only_approved_variant(self):
        weak = {"role_boundary": 0.1, "uncertainty_calibration": 0.1,
                "emotion_belief_separation": 0.1, "risk_escalation": 0.1}
        for axis in AXES:
            _, message, variant = select_approved_message(axis, 0, "profile_aware", weak)
            self.assertEqual(variant, "boundary_check")
            self.assertTrue(message)

    def test_profile_blind_ignores_profile(self):
        _, _, variant = select_approved_message("toxic_empathy", 0, "profile_blind", {})
        self.assertEqual(variant, "neutral")


if __name__ == "__main__":
    unittest.main()
