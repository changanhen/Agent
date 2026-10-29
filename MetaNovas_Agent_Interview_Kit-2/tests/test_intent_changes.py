"""Offline tests of the scenario harness and agent audit enforcement.

Mocks do NOT validate a real model's understanding of revised/cancelled intent.
"""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import agent
from scripts import intent_change_runner as runner
from tests.test_agent import report


def proposal(scenario):
    call = scenario["expected_call"]["arguments"]
    return {"content": "For " + call["reservation_id"] + ", replace " + call["old_asset_id"]
            + " with " + call["new_asset_id"] + " for an additional "
            + format(scenario["revised_fee"], ".2f") + " using project credit. Please confirm."}


class IntentChangeTests(unittest.TestCase):
    def test_seed_has_authentication_evidence_and_no_write(self):
        for name in runner.SCENARIOS:
            with self.subTest(scenario=name):
                scenario = runner.prepare_scenario(name)
                self.assertEqual(scenario["env"].db, scenario["before"])
                self.assertTrue(scenario["env"].member_id)
                self.assertNotEqual(scenario["old_call"], scenario["expected_call"])

    def test_harness_accepts_cancellation_without_writes(self):
        for name in runner.SCENARIOS:
            if name.startswith("cancel_"):
                with self.subTest(scenario=name):
                    result = runner.run_scenario(name, lambda **kwargs: {"content": "Replacement withdrawn; original reservation unchanged."})
                    self.assertEqual(result["status"], "PASS")
                    self.assertTrue(result["database_unchanged"])
                    self.assertEqual(result["successful_writes"], 0)

    def test_harness_detects_stale_write_for_all_scenarios(self):
        for name in runner.SCENARIOS:
            scenario = runner.prepare_scenario(name)
            with self.subTest(scenario=name):
                result = runner.run_scenario(name, lambda **kwargs: {"tool_call": scenario["old_call"]})
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(result["database_unchanged"])

    def test_harness_requires_fresh_consent_even_for_correct_revised_option(self):
        name = "revise_after_confirmation"
        scenario = runner.prepare_scenario(name)
        result = runner.run_scenario(name, lambda **kwargs: {"tool_call": scenario["expected_call"]})
        self.assertEqual(result["reason"], "write_attempt_without_fresh_confirmation")

    def test_harness_checks_revised_option_fee_and_actual_balance(self):
        for name in runner.SCENARIOS:
            if not name.startswith("revise_"):
                continue
            scenario = runner.prepare_scenario(name)
            replies = iter([proposal(scenario), {"tool_call": scenario["expected_call"]},
                            {"content": "Replacement succeeded; the revised fee was paid with project credit."}])
            with self.subTest(scenario=name):
                result = runner.run_scenario(name, lambda **kwargs: next(replies))
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["successful_writes"], 1)
                self.assertEqual(result["fee_increase"], 10.25)
                self.assertEqual(result["final_credit_balance"], 64.75)

    def test_harness_never_confirms_an_incorrect_fee_proposal(self):
        scenario = runner.prepare_scenario("revise_before_confirmation")
        incorrect = proposal(scenario)
        incorrect["content"] = incorrect["content"].replace("10.25", "14.00")
        result = runner.run_scenario(scenario["name"], lambda **kwargs: incorrect)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertFalse(result["proposal_checks"]["exact_fee_named"])
        self.assertTrue(result["database_unchanged"])

    def test_fallback_is_not_mistaken_for_successful_cancellation(self):
        result = runner.run_scenario("cancel_after_confirmation", lambda **kwargs: {
            "content": "I couldn't verify the information needed to continue safely. Please try again."})
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_agent_honors_audit_rejection_of_stale_write(self):
        for name in runner.SCENARIOS:
            scenario = runner.prepare_scenario(name)
            rejected = report("write")
            rejected["approved"] = False
            rejected["checks"]["latest_intent"] = False
            rejected["checks"]["confirmation"] = False
            rejected["issues"] = ["User revised or withdrew the request; previous consent is invalid."]
            repaired = ({"content": "Replacement withdrawn; existing reservation unchanged."}
                        if scenario["cancellation"] else proposal(scenario))
            replies = [{"tool_call": scenario["old_call"]}, {"content": json.dumps(rejected)},
                       repaired, {"content": json.dumps(report())}]
            with self.subTest(scenario=name), patch("agent.reconstruct_intent", return_value=None), patch("agent.chat", side_effect=deepcopy(replies)):
                answer = agent.respond(scenario["messages"], runner.TOOLS, runner.POLICY)
                self.assertEqual(answer, repaired)
                self.assertNotIn("tool_call", answer)


if __name__ == "__main__":
    unittest.main()
