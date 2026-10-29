"""Offline regression checks for versioned multi-goal completion guards.

These validate enforcement and parsing, not a live model's semantic accuracy.
"""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import agent
from tests.test_agent import CALL, MESSAGES, TOOLS, audit_reply, report


def goal(goal_id="g1", kind="write", status="pending"):
    return {"id": goal_id, "request": "Resolve " + goal_id, "target": "R" + goal_id[1:],
            "status": status, "kind": kind, "source_user_indexes": [0],
            "last_changed_index": 0, "constraints": ["Use verified evidence"],
            "evidence_indexes": []}


def completion_context(goals, version=2):
    return {"intent_state": {"version": version, "goals": deepcopy(goals),
            "tool_effects": {"modify_record": "write"}, "allowed_writes": []}}


def completion_report(kind="final", addressed=None, resolved=None):
    return {"response_kind": kind, "addressed_goal_ids": addressed or [],
            "resolved_goal_ids": resolved or []}


class GoalCompletionTests(unittest.TestCase):
    def test_pending_write_cannot_be_ended_by_final_prose(self):
        issues = agent.check_goal_completion(
            {"content": "All done."}, completion_context([goal()]),
            completion_report(addressed=["g1"]))
        self.assertTrue(issues)

    def test_pending_write_cannot_be_resolved_by_a_claim_in_text(self):
        issues = agent.check_goal_completion(
            {"content": "The change succeeded."}, completion_context([goal()]),
            completion_report(addressed=["g1"], resolved=["g1"]))
        self.assertTrue(issues)

    def test_pending_mixed_goal_cannot_be_completed_by_text_alone(self):
        issues = agent.check_goal_completion(
            {"content": "I answered and changed everything."},
            completion_context([goal(kind="mixed")]),
            completion_report(addressed=["g1"], resolved=["g1"]))
        self.assertTrue(issues)

    def test_pending_information_can_be_resolved_by_supported_final_answer(self):
        issues = agent.check_goal_completion(
            {"content": "Your verified balance is 61.00."},
            completion_context([goal(kind="information")]),
            completion_report(addressed=["g1"], resolved=["g1"]))
        self.assertEqual(issues, [])

    def test_pending_information_must_be_explicitly_resolved(self):
        issues = agent.check_goal_completion(
            {"content": "Here is an update."},
            completion_context([goal(kind="information")]),
            completion_report(addressed=["g1"]))
        self.assertTrue(issues)

    def test_unrelated_pending_write_is_not_silently_dropped(self):
        issues = agent.check_goal_completion(
            {"content": "Here is the information for R1."},
            completion_context([goal(kind="information"), goal("g2")]),
            completion_report(addressed=["g1"], resolved=["g1"]))
        self.assertTrue(issues)

    def test_unrelated_pending_information_is_not_silently_dropped(self):
        issues = agent.check_goal_completion(
            {"content": "Here is the information for R1."},
            completion_context([goal(kind="information"), goal("g2", kind="information")]),
            completion_report(addressed=["g1"], resolved=["g1"]))
        self.assertTrue(issues)

    def test_progress_only_response_cannot_abandon_pending_work(self):
        issues = agent.check_goal_completion(
            {"content": "I will look that up."}, completion_context([goal()]),
            completion_report("progress", addressed=["g1"]))
        self.assertTrue(issues)

    def test_confirmation_request_is_allowed_for_a_pending_goal(self):
        issues = agent.check_goal_completion(
            {"content": "Please confirm this exact change and fee."},
            completion_context([goal()]),
            completion_report("confirmation", addressed=["g1"]))
        self.assertEqual(issues, [])

    def test_necessary_clarification_is_allowed_for_a_pending_goal(self):
        issues = agent.check_goal_completion(
            {"content": "What is your member code?"}, completion_context([goal()]),
            completion_report("clarification", addressed=["g1"]))
        self.assertEqual(issues, [])

    def test_completed_cancelled_and_blocked_goals_allow_final_report(self):
        goals = [goal("g1", status="completed"), goal("g2", status="cancelled"),
                 goal("g3", status="blocked")]
        issues = agent.check_goal_completion(
            {"content": "R1 completed, R2 withdrawn, R3 ineligible."},
            completion_context(goals), completion_report(addressed=["g1", "g2", "g3"]))
        self.assertEqual(issues, [])

    def test_legacy_context_is_not_broken_by_new_report_fields(self):
        issues = agent.check_goal_completion(
            {"content": "Done."}, completion_context([goal()], version=1), {})
        self.assertEqual(issues, [])


class StructuredParsingTests(unittest.TestCase):
    def test_review_accepts_whole_fenced_json_without_relaxing_checks(self):
        catalog = agent.build_tool_catalog(TOOLS)
        context = agent.build_decision_context(MESSAGES, agent.replay_history(MESSAGES), catalog, "policy")
        review = report()
        review.update(response_kind="final", addressed_goal_ids=[], resolved_goal_ids=[])
        fenced = {"content": "```json\n" + json.dumps(review) + "\n```"}
        with patch("agent.chat", return_value=fenced):
            result = agent.review_candidate(MESSAGES, {"content": "Your request is complete."}, context, catalog, "policy")
        self.assertTrue(result["approved"], result.get("issues"))

    def test_empty_intent_goals_are_rejected_after_one_bounded_retry(self):
        messages = [{"role": "user", "content": "Change record R1 and tell me its final fee."}]
        catalog = agent.build_tool_catalog(TOOLS)
        context = agent.build_decision_context(messages, agent.replay_history(messages), catalog, "policy")
        invalid = {"version": 2, "goals": [], "tool_effects": {"modify_record": "write"},
                   "allowed_writes": [], "latest_update": {"index": 0, "kind": "new_request", "affected_goal_ids": []}}
        with patch("agent.chat", side_effect=[audit_reply(invalid), audit_reply(invalid)]) as model:
            with self.assertRaises(ValueError):
                agent.reconstruct_intent(messages, context, catalog, "policy")
        self.assertEqual(model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
