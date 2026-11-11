"""Regressions for evidence integrity and historical parser behavior."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit import load_verified, parse_proposal, validate
from experiment.core import Trigger


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes, cls.calls, cls.summary = load_verified()

    def test_historical_totals(self):
        self.assertEqual(self.summary["episodes"], 3000)
        self.assertEqual(self.summary["response_records"], 7260)
        self.assertEqual(self.summary["vlm_responses_reparsed"], 5064)
        self.assertEqual(
            {r["method"]: r["successes"] for r in self.summary["methods"]},
            dict(adaptive_only=599, error=504, joint=262, joint_async=462, joint_rule=396),
        )

    def test_missing_episode_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate(self.episodes[:-1], self.calls)

    def test_duplicate_episode_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate episode"):
            validate(self.episodes + [self.episodes[0]], self.calls)

    def test_duplicate_call_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate call"):
            validate(self.episodes, self.calls + [self.calls[0]])

    def test_missing_call_rejected(self):
        with self.assertRaisesRegex(ValueError, "Call coverage"):
            validate(self.episodes, self.calls[1:])

    def test_edited_reply_detected(self):
        calls = list(self.calls)
        calls[0] = dict(calls[0], raw="unexpected output")
        with self.assertRaisesRegex(ValueError, "Parser mismatch"):
            validate(self.episodes, calls)

    def test_cross_episode_mismatch_rejected(self):
        calls = list(self.calls)
        calls[0] = dict(calls[0], success=not calls[0]["success"])
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            validate(self.episodes, calls)

    def test_stale_reply_cannot_be_marked_applied(self):
        calls = list(self.calls)
        index = next(i for i, r in enumerate(calls) if r["stale"])
        calls[index] = dict(calls[index], applied=True)
        with self.assertRaisesRegex(ValueError, "Invalid application"):
            validate(self.episodes, calls)

    def test_task_denominators(self):
        for row in self.summary["by_task"]:
            self.assertEqual(row["episodes"], 300)

    def test_paired_differences(self):
        first, second = self.summary["paired_descriptive_counts"]
        self.assertAlmostEqual(first["difference_percentage_points"], 100 / 3)
        self.assertAlmostEqual(second["difference_percentage_points"], -137 / 6)


class ParserTests(unittest.TestCase):
    valid = '{"skills":["descend","grasp","lift"],"preferred":0}'

    def test_wrapped_json_allowed(self):
        self.assertEqual(parse_proposal("Text\n```json\n" + self.valid + "\n```")["preferred"], 0)

    def test_two_skills_is_valid_json_but_invalid_schema(self):
        raw = '{"skills":["grasp","lift"],"preferred":0}'
        self.assertIsInstance(json.loads(raw), dict)
        with self.assertRaisesRegex(ValueError, "Three skills"):
            parse_proposal(raw)

    def test_duplicate_proposal_objects_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_proposal(self.valid + self.valid)

    def test_contract_rejects_unknown_skill_extra_key_and_boolean_index(self):
        cases = [
            self.valid.replace("descend", "fly"),
            self.valid.replace('"preferred":0', '"preferred":true'),
            self.valid[:-1] + ',"force":99}',
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_proposal(raw)

    def test_historical_parser_allows_duplicate_skill_names(self):
        raw = '{"skills":["lift","lift","lift"],"preferred":0}'
        self.assertEqual(len(parse_proposal(raw)["skills"]), 3)


class TriggerTests(unittest.TestCase):
    def history(self, error, gain, progress=0):
        return [dict(progress=progress, prediction_error=error, audit_gain=gain)] * 4

    def test_adapting_high_error_does_not_force_joint_intervention(self):
        trigger = Trigger("joint", 1)
        trigger.record(0)
        self.assertFalse(trigger.ready(30, self.history(2, .5)))
        self.assertTrue(trigger.ready(30, self.history(2, 0)))

    def test_accurate_but_stagnant_can_trigger(self):
        trigger = Trigger("joint", 1)
        trigger.record(0)
        self.assertTrue(trigger.ready(30, self.history(.1, .5)))
        self.assertFalse(trigger.ready(30, self.history(.1, .5, .01)))

    def test_pending_budget_cooldown(self):
        trigger = Trigger("joint", 1)
        self.assertFalse(trigger.ready(0, [], pending=True))
        for step in (0, 30, 60, 90):
            self.assertTrue(trigger.ready(step, self.history(2, 0)))
            trigger.record(step)
            self.assertFalse(trigger.ready(step + 5, self.history(2, 0)))
        self.assertFalse(trigger.ready(150, self.history(2, 0)))


if __name__ == "__main__":
    unittest.main()
