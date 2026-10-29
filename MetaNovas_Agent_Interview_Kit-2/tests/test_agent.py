"""Offline behavioral tests. No API key or third-party packages required."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import agent


TOOLS = [{"type": "function", "function": {
    "name": "modify_record", "description": "Write a confirmed record change.",
    "parameters": {"type": "object", "properties": {
        "record_id": {"type": "string"}, "option": {"enum": ["A", "B"]}},
        "required": ["record_id", "option"], "additionalProperties": False},
}}]
CALL = {"name": "modify_record", "arguments": {"record_id": "R1", "option": "B"}}
TERMS = {"target": "R1", "current": "A", "change": "B",
         "fee_impact": 14, "funding_source": "project credit"}
MESSAGES = [
    {"role": "user", "content": "Please change R1."},
    {"role": "assistant", "content": "Change R1 from A to B for 14 using project credit. Confirm?"},
    {"role": "user", "content": "Yes, I confirm that exact change and fee."},
]


def report(kind="message", **changes):
    value = {
        "approved": True, "action_kind": kind, "issues": [],
        "checks": {key: True for key in (
            "preconditions", "latest_intent", "evidence", "confirmation",
            "batch_progress", "completion_claims")},
        "proposal_index": 1 if kind == "write" else None,
        "confirmation_index": 2 if kind == "write" else None,
        "write_terms": deepcopy(TERMS) if kind == "write" else None,
        "response_kind": "final" if kind == "message" else "tool",
        "addressed_goal_ids": ["g1"], "resolved_goal_ids": [],
        "evidence_review": {"claims_supported": True, "latest_records_used": True,
            "all_constraints_checked": True, "missing_evidence": [], "selection_required": False,
            "facts": [], "calculations": [], "selection": None},
        "customer_output": {key: True for key in (
            "direct_answer", "all_requested_outcomes", "exact_option_fee_funding",
            "clear_success_failure_pending", "clear_next_step", "plain_language",
            "no_unverified_claims_or_currency", "minimal_sensitive_data")},
    }
    value.update(changes)
    return value


def audit_reply(value):
    return {"content": json.dumps(value)}


def tool_message(call, call_id="call_1"):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": call["name"], "arguments": json.dumps(call["arguments"])}}]}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.catalog = agent.build_tool_catalog(TOOLS)

    def test_rejects_text_and_tool_together(self):
        self.assertTrue(agent.validate_response_shape({"content": "OK", "tool_call": CALL}))

    def test_rejects_empty_or_nonfinite_response(self):
        self.assertTrue(agent.validate_response_shape({"content": " "}))
        self.assertTrue(agent.validate_response_shape({"tool_call": {
            "name": "x", "arguments": {"amount": float("nan")}}}))

    def test_rejects_missing_extra_enum_and_unknown_tool(self):
        for call in [
            {"name": "missing", "arguments": {}},
            {"name": "modify_record", "arguments": {"record_id": "R1"}},
            {"name": "modify_record", "arguments": {"record_id": "R1", "option": "Z"}},
            {"name": "modify_record", "arguments": {**CALL["arguments"], "extra": 1}},
        ]:
            with self.subTest(call=call):
                self.assertTrue(agent.validate_tool_arguments(call, self.catalog))
        self.assertEqual(agent.validate_tool_arguments(CALL, self.catalog), [])

    def test_nested_refs_types_and_constraints(self):
        schema = {"type": "object", "$defs": {"amount": {
            "type": "integer", "minimum": 1, "maximum": 3}},
            "properties": {"values": {"type": "array", "minItems": 1,
                                      "items": {"$ref": "#/$defs/amount"}}}}
        self.assertEqual(agent._validate_schema({"values": [1, 3]}, schema, schema), [])
        for value in (True, 0, 4, 1.5):
            self.assertTrue(agent._validate_schema({"values": [value]}, schema, schema))

    def test_union_and_string_pattern(self):
        schema = {"anyOf": [{"type": "null"}, {"type": "string", "pattern": "^R[0-9]+$"}]}
        for value in (None, "R12"):
            self.assertEqual(agent._validate_schema(value, schema, schema), [])
        self.assertTrue(agent._validate_schema("X", schema, schema))

    def test_fingerprint_is_order_independent_but_includes_terms(self):
        first = agent.build_write_spec(CALL, {"write_terms": TERMS})
        reordered = {key: first[key] for key in reversed(list(first))}
        self.assertEqual(agent.fingerprint_write(first), agent.fingerprint_write(reordered))
        changed = deepcopy(first)
        changed["terms"]["fee_impact"] = 20
        self.assertNotEqual(agent.fingerprint_write(first), agent.fingerprint_write(changed))

    def test_local_sdk_reasoning_metadata_is_preserved(self):
        value = {"tool_call": CALL, "_reasoning_content": "sdk transport metadata"}
        self.assertEqual(agent.finalize_response(value), value)


class StateTests(unittest.TestCase):
    def test_pairs_results_by_call_id_not_order(self):
        messages = [tool_message(CALL, "a"), tool_message(CALL, "b"),
                    {"role": "tool", "tool_call_id": "b", "content": '{"error":"failed"}'},
                    {"role": "tool", "tool_call_id": "a", "content": '{"id":"R1"}'}]
        first, second = agent.replay_history(messages)["tool_events"]
        self.assertEqual((first["result_index"], first["outcome"]), (3, "returned"))
        self.assertEqual((second["result_index"], second["outcome"]), (2, "error"))

    def test_missing_result_is_unknown(self):
        event = agent.replay_history([tool_message(CALL)])["tool_events"][0]
        self.assertEqual(event["outcome"], "unknown")

    def test_confirmation_cannot_come_from_assistant_or_fake_indexes(self):
        context = agent.replay_history(MESSAGES)
        for indexes in ((0, 1), (1, 100), (1, True), (2, 1)):
            audit = report("write", proposal_index=indexes[0], confirmation_index=indexes[1])
            self.assertTrue(agent.check_confirmation(MESSAGES, CALL, context, audit))

    def test_failed_and_uncertain_write_consumes_confirmation(self):
        for result in (None, {"error": "failed"}, {"success": True}):
            messages = deepcopy(MESSAGES) + [tool_message(CALL)]
            if result is not None:
                messages.append({"role": "tool", "tool_call_id": "call_1",
                                 "content": json.dumps(result)})
            issues = agent.check_confirmation(messages, CALL, agent.replay_history(messages), report("write"))
            self.assertTrue(any("already attempted" in issue for issue in issues))


class RespondTests(unittest.TestCase):
    def setUp(self):
        # Isolate the original candidate/reviewer tests. Intent extraction and
        # its integration are exercised separately in test_intent_state.py.
        intent = patch("agent.reconstruct_intent", return_value=None)
        intent.start()
        self.addCleanup(intent.stop)

    def test_confirmed_write_passes_without_mutating_inputs(self):
        messages, tools = deepcopy(MESSAGES), deepcopy(TOOLS)
        with patch("agent.chat", side_effect=[{"tool_call": CALL}, audit_reply(report("write"))]) as model:
            self.assertEqual(agent.respond(messages, tools, "policy"), {"tool_call": CALL})
        self.assertEqual(model.call_count, 2)
        self.assertEqual(messages, MESSAGES)
        self.assertEqual(tools, TOOLS)
        self.assertEqual(model.call_args_list[1].kwargs["tools"], [])

    def test_write_without_proposal_cannot_pass_an_optimistic_audit(self):
        clarification = {"content": "Change R1 from A to B for 14 from project credit. Confirm?"}
        with patch("agent.chat", side_effect=[
            {"tool_call": CALL}, audit_reply(report("write")),
            clarification, audit_reply(report()),
        ]) as model:
            self.assertEqual(agent.respond(MESSAGES[:1], TOOLS, "policy"), clarification)
        self.assertEqual(model.call_count, 4)

    def test_semantic_rejections_trigger_repair(self):
        # Tests enforcement of an audit, not the real model's semantic accuracy.
        for key in ("preconditions", "latest_intent", "evidence", "confirmation",
                    "batch_progress", "completion_claims"):
            audit = report("write")
            audit["checks"][key] = False
            clarification = {"content": "Please confirm the revised exact change."}
            with self.subTest(check=key), patch("agent.chat", side_effect=[
                {"tool_call": CALL}, audit_reply(audit), clarification, audit_reply(report()),
            ]):
                self.assertEqual(agent.respond(MESSAGES, TOOLS, "policy"), clarification)

    def test_invalid_candidate_can_be_repaired(self):
        repaired = {"content": "Please provide your name and member code."}
        with patch("agent.chat", side_effect=[[], repaired, audit_reply(report())]) as model:
            self.assertEqual(agent.respond([], TOOLS, "policy"), repaired)
        self.assertEqual(model.call_count, 3)

    def test_malformed_reviews_stop_after_one_repair(self):
        with patch("agent.chat", side_effect=[
            {"tool_call": CALL}, {"content": "not JSON"},
            {"tool_call": CALL}, {"content": "{}"},
        ]) as model:
            answer = agent.respond(MESSAGES, TOOLS, "policy")
        self.assertEqual(set(answer), {"content"})
        self.assertEqual(model.call_count, 4)

    def test_sdk_failure_does_not_leak_provider_details(self):
        with patch("agent.chat", side_effect=agent.ModelError("private provider details")):
            answer = agent.respond(MESSAGES, TOOLS, "policy")
        self.assertEqual(set(answer), {"content"})
        self.assertNotIn("private provider details", answer["content"])

    def test_complete_practice_tool_loop_with_mocked_model(self):
        from scripts.practice_runner import DB, POLICY, TASK, TOOLS as PRACTICE_TOOLS, PracticeEnvironment

        env = PracticeEnvironment()
        info = TASK["user_scenario"]["instructions"]
        messages = [{"role": "user", "content": info["known_info"] + "\n" + info["reason_for_call"]}]
        member_id = next(iter(DB["members"]))
        member = DB["members"][member_id]
        reservation_id = member["reservations"][0]
        asset = DB["reservations"][reservation_id]["assets"][0]
        family = DB["asset_families"][asset["family_id"]]
        eligible = [option for option in family["options"].values() if option["available"]
                    and option["specifications"]["balanced_xlr"]
                    and option["specifications"]["battery_hours"] >= 12]
        replacement = min(eligible, key=lambda option: option["specifications"]["weight_grams"])
        credit_id = next(iter(member["credits"]))
        delta = round(replacement["fee"] - asset["fee"], 2)

        def step(candidate, audit, authorized=False):
            with patch("agent.chat", side_effect=[candidate, audit_reply(audit)]):
                answer = agent.respond(messages, PRACTICE_TOOLS, POLICY)
            self.assertEqual(answer, candidate)
            if "tool_call" in answer:
                call_id = "call_" + str(len(messages))
                call = answer["tool_call"]
                messages.append(tool_message(call, call_id))
                result = env.call(call["name"], call["arguments"], write_authorized=authorized)
                self.assertFalse(isinstance(result, dict) and "error" in result)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
            else:
                messages.append({"role": "assistant", "content": answer["content"]})

        for name, arguments, kind in [
            ("locate_member", {"full_name": member["full_name"], "member_code": member["member_code"]}, "authentication"),
            ("read_member_profile", {"member_id": member_id}, "read"),
            ("read_reservation", {"reservation_id": reservation_id}, "read"),
            ("inspect_asset_family", {"family_id": asset["family_id"]}, "read"),
        ]:
            step({"tool_call": {"name": name, "arguments": arguments}}, report(kind))
        proposal = len(messages)
        step({"content": "Change " + reservation_id + " from " + asset["asset_id"]
              + " to " + replacement["asset_id"] + " for an additional " + str(delta)
              + " using project credit. Please confirm."}, report())
        confirmation = len(messages)
        messages.append({"role": "user", "content": "Yes, I confirm that exact change and fee."})
        terms = {"target": reservation_id, "current": asset["asset_id"],
                 "change": replacement["asset_id"], "fee_impact": delta, "funding_source": credit_id}
        step({"tool_call": {"name": "change_reserved_asset", "arguments": {
            "reservation_id": reservation_id, "old_asset_id": asset["asset_id"],
            "new_asset_id": replacement["asset_id"], "credit_id": credit_id}}},
            report("write", proposal_index=proposal, confirmation_index=confirmation, write_terms=terms),
            authorized=True)
        step({"content": "Successfully selected " + replacement["asset_id"]
              + "; additional fee " + str(delta) + " paid with project credit."}, report())
        self.assertEqual(env.db["reservations"][reservation_id]["status"], "awaiting_collection_changed")
        self.assertEqual(env.db["members"][member_id]["credits"][credit_id]["balance"],
                         member["credits"][credit_id]["balance"] - delta)


if __name__ == "__main__":
    unittest.main()
