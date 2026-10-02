import unittest

import phase1_persona as p1


CASE = {"axis": "toxic_empathy",
        "persona": {"persona_id": "p-01", "descriptor": "d", "style_reference": "s"},
        "goal": {"goal_id": "te-01", "intent": "i", "masked_request": "m"}}


class HardenPersonaTests(unittest.TestCase):
    """Drive harden_persona with stubbed adapter calls (no API)."""

    def setUp(self):
        self._invoke, self._respond = p1.invoke, p1.respond
        p1.respond = lambda messages, cmd: "safe response"

    def tearDown(self):
        p1.invoke, p1.respond = self._invoke, self._respond

    def _wire(self, fitness_sequence):
        """Make score_progress return a scripted sequence of scores (one per surrogate call)."""
        scores = iter(fitness_sequence)

        def fake_invoke(cmd, payload):
            task = payload["task"]
            if task == "generate_cognitive_distortion":
                return {"distortion_name": "catastrophizing", "pattern": "x", "how_it_masks_goal": "y"}
            if task == "scriptwriter":
                return {"system_instruction": "sys", "opening_utterances": ["u0"]}
            if task == "perturb_persona":
                return {"system_instruction": "sys'", "opening_utterances": ["u1"], "changes": "c"}
            if task == "score_progress":
                return {"score": next(scores)}
            raise AssertionError(task)

        p1.invoke = fake_invoke

    def test_accepts_improvement_and_tracks_best(self):
        # one surrogate; initial 3, then 5 (accept), then 4 (reject), then 4 (reject -> patience=2 stop)
        self._wire([3, 5, 4, 4])
        out = p1.harden_persona(CASE, [["python3", "t", "--model", "m1"]], ["g"], ["e"],
                                max_iters=6, patience=2)
        self.assertEqual(out["fitness"], 5)
        self.assertEqual(out["opening_utterances"], ["u1"])  # the accepted perturbation
        self.assertEqual([t["accepted"] for t in out["trace"]], [True, True, False, False])

    def test_early_stop_on_score_target(self):
        self._wire([9])  # initial fitness already >= target -> no perturbation
        out = p1.harden_persona(CASE, [["python3", "t", "--model", "m1"]], ["g"], ["e"],
                                max_iters=6, patience=2, score_target=8.0)
        self.assertEqual(out["iterations"], 0)
        self.assertEqual(out["fitness"], 9)

    def test_mean_over_multiple_surrogates(self):
        # two surrogates initial (2,4)->mean 3 ; perturb (6,8)->mean 7 accept
        self._wire([2, 4, 6, 8])
        surrogates = [["python3", "t", "--model", "a"], ["python3", "t", "--model", "b"]]
        out = p1.harden_persona(CASE, surrogates, ["g"], ["e"], max_iters=1, patience=2)
        self.assertEqual(out["fitness"], 7)


if __name__ == "__main__":
    unittest.main()
