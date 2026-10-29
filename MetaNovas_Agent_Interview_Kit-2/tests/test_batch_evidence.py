"""Offline tests of actual pre-write batch evidence/proposal enforcement."""

import unittest
from copy import deepcopy

import agent
from tests.test_agent import CALL, tool_message


class BatchEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "user", "content": "Review and change R1 and R2 as eligible."},
            tool_message({"name": "read_record", "arguments": {"record_id": "R1"}}, "a"),
            {"role": "tool", "tool_call_id": "a", "content": '{"record_id":"R1","status":"pending"}'},
            tool_message({"name": "read_record", "arguments": {"record_id": "R2"}}, "b"),
            {"role": "tool", "tool_call_id": "b", "content": '{"record_id":"R2","status":"collected"}'},
            {"role": "assistant", "content": "R1: pending, change; R2: collected, no action. Confirm only R1?"},
            {"role": "user", "content": "Yes, only R1."},
        ]
        self.context = agent.replay_history(self.messages)
        self.context["intent_state"] = {"tool_effects": {"modify_record": "write", "read_record": "read"},
            "batch_plan": {"required": True, "inventory_complete": True, "start_index": 0,
                "records": [{"target": "R1", "verified_status": "pending", "read_index": 2,
                             "decision": "write", "plan_index": 5},
                            {"target": "R2", "verified_status": "collected", "read_index": 4,
                             "decision": "no_action", "plan_index": 5}]}}

    def test_verified_complete_plan_can_reach_first_write(self):
        self.assertEqual(agent.check_batch_evidence({"tool_call": CALL}, self.messages, self.context), [])

    def test_inventory_must_be_complete_before_first_write(self):
        self.context["intent_state"]["batch_plan"]["inventory_complete"] = False
        self.assertTrue(agent.check_batch_evidence({"tool_call": CALL}, self.messages, self.context))

    def test_record_status_must_match_cited_tool_evidence(self):
        self.context["intent_state"]["batch_plan"]["records"][1]["verified_status"] = "pending"
        self.assertTrue(agent.check_batch_evidence({"tool_call": CALL}, self.messages, self.context))

    def test_reading_profile_is_not_a_substitute_for_record_status_evidence(self):
        self.messages[4]["content"] = '{"record_ids":["R1","R2"]}'
        self.assertTrue(agent.check_batch_evidence({"tool_call": CALL}, self.messages, self.context))

    def test_no_action_decision_is_required_for_ineligible_records(self):
        self.context["intent_state"]["batch_plan"]["records"][1]["decision"] = "needs_evidence"
        self.assertTrue(agent.check_batch_evidence({"tool_call": CALL}, self.messages, self.context))

    def test_remaining_write_does_not_need_a_new_entire_batch_plan(self):
        messages = deepcopy(self.messages) + [tool_message(CALL, "write")]
        history = agent.replay_history(messages)
        history["intent_state"] = self.context["intent_state"]
        self.assertEqual(agent.check_batch_evidence({"tool_call": CALL}, messages, history), [])

    def test_local_reasoning_metadata_is_canonicalized(self):
        messages = [{"role": "assistant", "tool_call": CALL, "_reasoning_content": "private"},
                    {"role": "tool", "content": '{"success":true}'}]
        history = agent.replay_history(messages)
        normalized = agent.normalize_model_history(messages, history)
        self.assertNotIn("_reasoning_content", normalized[0])
        self.assertEqual(normalized[0]["reasoning_content"], "private")


if __name__ == "__main__":
    unittest.main()
