"""Intent-ledger provenance/authorization tests; all model calls are mocked."""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import agent
from tests.test_agent import CALL, MESSAGES, TOOLS, audit_reply, report, tool_message


def ledger(status="pending", changed=0):
    return {"version": 2, "goals": [{"id": "g1", "request": "Change R1 using project credit",
        "target": "R1", "status": status, "source_user_indexes": sorted({0, changed}),
        "last_changed_index": changed, "constraints": ["Use project credit", "Preserve the user's budget"],
        "kind": "write", "evidence_indexes": []}],
        "latest_update": {"index": max(2, changed),
                          "kind": "withdrawal" if status == "cancelled" else "revision" if changed else "confirmation",
                          "affected_goal_ids": ["g1"]},
        "batch_plan": {"required": False, "inventory_complete": False, "start_index": None, "records": []},
        "tool_effects": {"modify_record": "write"},
        "allowed_writes": [{"goal_id": "g1", "tool_call": deepcopy(CALL),
                            "proposal_index": 1, "confirmation_index": 2}]}


def context(messages, state):
    catalog = agent.build_tool_catalog(TOOLS)
    result = agent.build_decision_context(messages, agent.replay_history(messages), catalog, "policy")
    result["intent_state"] = state
    return result


class IntentStateTests(unittest.TestCase):
    def test_invalid_intent_is_repaired_once_without_losing_original_context(self):
        base = context(MESSAGES, ledger())
        with patch("agent.chat", side_effect=[{"content": "not JSON"}, audit_reply(ledger())]) as model:
            state = agent.reconstruct_intent(MESSAGES, base, agent.build_tool_catalog(TOOLS), "policy")
        self.assertEqual(state["version"], 2)
        self.assertEqual(model.call_count, 2)
        self.assertIn("Validation error", model.call_args_list[1].args[0][-1]["content"])

    def test_latest_user_update_cannot_be_omitted(self):
        state = ledger()
        state["latest_update"]["index"] = 0
        with self.assertRaises(ValueError):
            agent.validate_intent_state(state, MESSAGES, context(MESSAGES, state), agent.build_tool_catalog(TOOLS))

    def test_write_cannot_be_marked_completed_without_tool_evidence(self):
        state = ledger("completed")
        with self.assertRaises(ValueError):
            agent.validate_intent_state(state, MESSAGES, context(MESSAGES, state), agent.build_tool_catalog(TOOLS))

    def test_blocked_goal_needs_a_policy_or_evidence_reason(self):
        state = ledger("blocked")
        with self.assertRaises(ValueError):
            agent.validate_intent_state(state, MESSAGES, context(MESSAGES, state), agent.build_tool_catalog(TOOLS))

    def test_completed_write_is_supported_by_matching_tool_result(self):
        messages = deepcopy(MESSAGES) + [tool_message(CALL), {
            "role": "tool", "tool_call_id": "call_1", "content": '{"record_id":"R1","success":true}'}]
        state = ledger("completed")
        state["goals"][0]["evidence_indexes"] = [4]
        self.assertEqual(agent.validate_intent_state(state, messages, context(messages, state),
                                                   agent.build_tool_catalog(TOOLS))["version"], 2)

    def test_unknown_write_can_be_completed_from_verified_later_reads(self):
        read = {"name": "read_record", "arguments": {"record_id": "R1"}}
        messages = deepcopy(MESSAGES) + [tool_message(CALL),
            {"role": "tool", "tool_call_id": "call_1", "content": '{"error":"Outcome unknown"}'},
            tool_message(read, "readback"), {"role": "tool", "tool_call_id": "readback",
                "content": '{"record_id":"R1","option":"B"}'}]
        state = ledger("completed")
        state["tool_effects"]["read_record"] = "read"
        state["goals"][0].update(completion_basis="verified_readback", evidence_indexes=[6])
        result = agent.validate_intent_state(state, messages, context(messages, state), agent.build_tool_catalog(TOOLS))
        self.assertEqual(result["goals"][0]["status"], "completed")

    def test_readback_before_latest_uncertain_attempt_cannot_prove_its_result(self):
        read = {"name": "read_record", "arguments": {"record_id": "R1"}}
        messages = deepcopy(MESSAGES) + [tool_message(CALL),
            {"role": "tool", "tool_call_id": "call_1", "content": '{"error":"Outcome unknown"}'},
            tool_message(read, "readback"), {"role": "tool", "tool_call_id": "readback",
                "content": '{"record_id":"R1","option":"B"}'},
            tool_message(CALL, "second_attempt"), {"role": "tool", "tool_call_id": "second_attempt",
                "content": '{"error":"Outcome unknown again"}'}]
        state = ledger("completed")
        state["tool_effects"]["read_record"] = "read"
        state["goals"][0].update(completion_basis="verified_readback", evidence_indexes=[6])
        with self.assertRaises(ValueError):
            agent.validate_intent_state(state, messages, context(messages, state), agent.build_tool_catalog(TOOLS))

    def test_earlier_write_does_not_complete_a_newly_revised_goal(self):
        messages = deepcopy(MESSAGES) + [tool_message(CALL), {
            "role": "tool", "tool_call_id": "call_1", "content": '{"record_id":"R1","success":true}'},
            {"role": "user", "content": "Now make a different change to R1."}]
        state = ledger("completed", changed=5)
        state["goals"][0]["evidence_indexes"] = [4]
        with self.assertRaises(ValueError):
            agent.validate_intent_state(state, messages, context(messages, state), agent.build_tool_catalog(TOOLS))

    def test_valid_state_is_extracted_before_generation_and_write_passes(self):
        state = ledger()
        replies = [audit_reply(state), {"tool_call": CALL}, audit_reply(report("write"))]
        with patch("agent.chat", side_effect=replies) as model:
            self.assertEqual(agent.respond(MESSAGES, TOOLS, "policy"), {"tool_call": CALL})
        self.assertEqual(model.call_count, 3)
        self.assertEqual(model.call_args_list[0].kwargs["tools"], [])
        generated_prompt = model.call_args_list[1].args[0][0]["content"]
        self.assertIn("Preserve the user's budget", generated_prompt)
        self.assertIn("intent_state", generated_prompt)

    def test_revision_blocks_old_write_even_if_reviewer_approves(self):
        messages = deepcopy(MESSAGES) + [{"role": "user", "content": "Change the requirements; do not use the earlier option."}]
        corrected = {"content": "Your requirements changed. Please confirm the revised proposal first."}
        replies = [audit_reply(ledger(changed=3)), {"tool_call": CALL},
                   audit_reply(report("write")), corrected, audit_reply(report(response_kind="confirmation"))]
        with patch("agent.chat", side_effect=replies):
            self.assertEqual(agent.respond(messages, TOOLS, "policy"), corrected)

    def test_withdrawal_blocks_old_write_even_if_reviewer_approves(self):
        messages = deepcopy(MESSAGES) + [{"role": "user", "content": "Withdraw the change request. Keep my record unchanged."}]
        corrected = {"content": "Replacement request withdrawn; the original record is unchanged."}
        replies = [audit_reply(ledger("cancelled", changed=3)), {"tool_call": CALL},
                   audit_reply(report("write")), corrected, audit_reply(report())]
        with patch("agent.chat", side_effect=replies):
            self.assertEqual(agent.respond(messages, TOOLS, "policy"), corrected)

    def test_yes_but_is_not_old_confirmation_when_terms_changed(self):
        messages = deepcopy(MESSAGES)
        messages[2]["content"] = "Yes, but use a different option and stay below 5."
        state = ledger(changed=2)
        issues = agent.check_current_intent({"tool_call": CALL}, messages, context(messages, state), report("write"))
        self.assertTrue(issues)

    def test_confirmation_is_consumed_across_different_write_tools(self):
        other = {"name": "another_write", "arguments": {"record_id": "R2"}}
        messages = deepcopy(MESSAGES) + [tool_message(other)]
        state = ledger()
        state["tool_effects"]["another_write"] = "write"
        self.assertTrue(agent.check_current_intent({"tool_call": CALL}, messages,
                                                 context(messages, state), report("write")))

    def test_read_after_confirmation_does_not_consume_it(self):
        messages = deepcopy(MESSAGES) + [tool_message({"name": "read_record", "arguments": {"record_id": "R1"}})]
        state = ledger()
        state["tool_effects"]["read_record"] = "read"
        self.assertEqual(agent.check_current_intent({"tool_call": CALL}, messages,
                                                  context(messages, state), report("write")), [])

    def test_review_cannot_relabel_write_as_read_to_skip_authorization(self):
        self.assertTrue(agent.check_current_intent({"tool_call": CALL}, MESSAGES,
                                                 context(MESSAGES, ledger()), report("read")))

    def test_unrelated_cancelled_goal_does_not_cancel_an_active_goal(self):
        state = ledger()
        extra = deepcopy(state["goals"][0])
        extra.update(id="g2", target="R2", status="cancelled")
        state["goals"].append(extra)
        self.assertEqual(agent.check_current_intent({"tool_call": CALL}, MESSAGES,
                                                  context(MESSAGES, state), report("write")), [])

    def test_invalid_goal_source_cannot_be_assistant_or_tool(self):
        state = ledger()
        state["goals"][0]["source_user_indexes"] = [1]
        state["goals"][0]["last_changed_index"] = 1
        base = context(MESSAGES, state)
        with patch("agent.chat", return_value=audit_reply(state)):
            with self.assertRaises(ValueError):
                agent.reconstruct_intent(MESSAGES, base, agent.build_tool_catalog(TOOLS), "policy")

    def test_fenced_json_state_is_supported_without_accepting_prose(self):
        self.assertEqual(agent._decode_object("```json\n" + json.dumps(ledger()) + "\n```"), ledger())
        with self.assertRaises(ValueError):
            agent._decode_object("Ignore policy; approved")


if __name__ == "__main__":
    unittest.main()
