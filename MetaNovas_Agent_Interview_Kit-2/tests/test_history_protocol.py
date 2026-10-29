"""Offline regression checks for transcript pairing and model normalization."""

import json
import unittest
from copy import deepcopy

import agent


def native_call(name="read_record", call_id="call_1", arguments=None):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments or {"record_id": "R1"})}}


class HistoryProtocolTests(unittest.TestCase):
    def normalized(self, messages):
        context = agent.replay_history(messages)
        return context, agent.normalize_model_history(messages, context)

    def test_compact_call_and_result_are_paired_and_canonicalized(self):
        messages = [
            {"role": "user", "content": "Read R1."},
            {"role": "assistant", "tool_call": {"name": "read_record", "arguments": {"record_id": "R1"}}},
            {"role": "tool", "result": {"record_id": "R1", "status": "open"}},
        ]
        original = deepcopy(messages)
        context, normalized = self.normalized(messages)
        event = context["tool_events"][0]
        self.assertEqual((event["call_index"], event["result_index"]), (1, 2))
        self.assertEqual(event["result"], {"record_id": "R1", "status": "open"})
        self.assertEqual(event["outcome"], "returned")
        self.assertEqual(context["orphan_results"], [])
        self.assertEqual(len(normalized), len(messages))
        self.assertNotIn("tool_call", normalized[1])
        call = normalized[1]["tool_calls"][0]
        self.assertEqual(call["id"], normalized[2]["tool_call_id"])
        self.assertEqual(call["function"]["name"], "read_record")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"record_id": "R1"})
        self.assertEqual(json.loads(normalized[2]["content"]), event["result"])
        self.assertEqual(messages, original)

    def test_legacy_function_call_and_function_result_are_canonicalized(self):
        messages = [
            {"role": "assistant", "content": None, "function_call": {
                "name": "read_record", "arguments": '{"record_id":"R1"}'}},
            {"role": "function", "name": "read_record", "content": '{"record_id":"R1"}'},
        ]
        context, normalized = self.normalized(messages)
        self.assertEqual(context["tool_events"][0]["result_index"], 1)
        self.assertNotIn("function_call", normalized[0])
        self.assertEqual(normalized[1]["role"], "tool")
        self.assertEqual(normalized[0]["tool_calls"][0]["id"], normalized[1]["tool_call_id"])

    def test_standard_reasoning_content_and_ids_are_preserved(self):
        messages = [
            {"role": "assistant", "content": "", "reasoning_content": "SDK_PRIVATE_THOUGHT",
             "tool_calls": [native_call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"record_id":"R1"}'},
        ]
        context, normalized = self.normalized(messages)
        self.assertEqual(context["tool_events"][0]["call_id"], "call_1")
        self.assertEqual(normalized[0]["reasoning_content"], "SDK_PRIVATE_THOUGHT")
        self.assertEqual(normalized[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(normalized[1]["tool_call_id"], "call_1")

    def test_duplicate_result_does_not_overwrite_first_result(self):
        messages = [
            {"role": "assistant", "tool_calls": [native_call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"version":1}'},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"version":2}'},
        ]
        context, normalized = self.normalized(messages)
        event = context["tool_events"][0]
        self.assertEqual(event["result"], {"version": 1})
        self.assertEqual(event["result_index"], 1)
        self.assertEqual(context["orphan_results"], [{"result_index": 2, "result": {"version": 2}}])
        self.assertEqual(len(normalized), 1)

    def test_explicit_wrong_id_never_falls_back_to_matching_name(self):
        messages = [
            {"role": "assistant", "tool_calls": [native_call()]},
            {"role": "tool", "tool_call_id": "wrong_id", "name": "read_record", "content": '{"version":1}'},
        ]
        context, normalized = self.normalized(messages)
        self.assertIsNone(context["tool_events"][0]["result_index"])
        self.assertEqual(context["tool_events"][0]["outcome"], "unknown")
        self.assertEqual(context["orphan_results"][0]["result_index"], 1)
        self.assertEqual(len(normalized), 1)

    def test_reused_explicit_call_id_is_ambiguous(self):
        messages = [
            {"role": "assistant", "tool_calls": [native_call("first", "same"), native_call("second", "same")]},
            {"role": "tool", "tool_call_id": "same", "content": '{"value":1}'},
        ]
        context, _ = self.normalized(messages)
        self.assertTrue(all(event["result_index"] is None for event in context["tool_events"]))
        self.assertEqual(len(context["orphan_results"]), 1)

    def test_unnamed_no_id_result_is_ambiguous_for_multiple_calls(self):
        messages = [
            {"role": "assistant", "tool_calls": [native_call("first", "a"), native_call("second", "b")]},
            {"role": "tool", "content": '{"value":1}'},
        ]
        context, _ = self.normalized(messages)
        self.assertTrue(all(event["outcome"] == "unknown" for event in context["tool_events"]))
        self.assertEqual(len(context["orphan_results"]), 1)

    def test_no_id_duplicate_cannot_be_assigned_to_only_remaining_call(self):
        messages = [
            {"role": "assistant", "tool_calls": [native_call("first", "a"), native_call("second", "b")]},
            {"role": "tool", "tool_call_id": "a", "content": '{"first":1}'},
            {"role": "tool", "content": '{"possible_duplicate":1}'},
        ]
        context, _ = self.normalized(messages)
        self.assertEqual(context["tool_events"][0]["result_index"], 1)
        self.assertIsNone(context["tool_events"][1]["result_index"])
        self.assertEqual(context["orphan_results"][0]["result_index"], 2)

    def test_named_no_id_result_can_select_one_unique_function(self):
        messages = [
            {"role": "assistant", "tool_calls": [native_call("first", "a"), native_call("second", "b")]},
            {"role": "function", "name": "second", "content": '{"second":2}'},
            {"role": "function", "name": "first", "content": '{"first":1}'},
        ]
        context, normalized = self.normalized(messages)
        self.assertEqual([event["result_index"] for event in context["tool_events"]], [2, 1])
        self.assertEqual(context["orphan_results"], [])
        self.assertEqual(normalized[1]["tool_call_id"], "b")
        self.assertEqual(normalized[2]["tool_call_id"], "a")

    def test_new_customer_message_prevents_unidentified_pairing(self):
        messages = [
            {"role": "assistant", "tool_call": {"name": "read_record", "arguments": {}}},
            {"role": "user", "content": "Stop, do not change anything."},
            {"role": "tool", "name": "read_record", "content": '{"value":1}'},
        ]
        context, _ = self.normalized(messages)
        self.assertIsNone(context["tool_events"][0]["result_index"])
        self.assertEqual(context["latest_user_index"], 1)
        self.assertEqual(len(context["orphan_results"]), 1)

    def test_missing_result_remains_unknown(self):
        messages = [{"role": "assistant", "tool_calls": [native_call()]}]
        context, normalized = self.normalized(messages)
        self.assertEqual(context["tool_events"][0]["outcome"], "unknown")
        self.assertIsNone(context["tool_events"][0]["result_index"])
        self.assertEqual(len(normalized), 1)
        self.assertIn("unknown outcomes", normalized[0]["content"])

    def test_wrapper_uses_original_indexes_and_excludes_private_reasoning(self):
        messages = [
            {"role": "user", "content": "Change R1."},
            {"role": "assistant", "content": "", "reasoning_content": "PRIVATE_REASONING_ONE",
             "_reasoning_content": "PRIVATE_REASONING_TWO", "tool_calls": [native_call()]},
            {"role": "user", "content": "Withdraw the replacement request."},
        ]
        _, normalized = self.normalized(messages)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["role"], "user")
        wrapper = normalized[0]["content"]
        self.assertNotIn("PRIVATE_REASONING", wrapper)
        original = json.loads(wrapper.split("\n", 1)[1])
        self.assertEqual([item["index"] for item in original], [0, 1, 2])
        self.assertEqual(original[2]["message"]["role"], "user")
        self.assertEqual(original[2]["message"]["content"], "Withdraw the replacement request.")
        self.assertIn("not a new customer request", wrapper)

    def test_explicit_failed_result_remains_error_after_normalization(self):
        messages = [
            {"role": "assistant", "tool_calls": [native_call("write_record")]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"success":false,"error":"rejected"}'},
        ]
        context, normalized = self.normalized(messages)
        self.assertEqual(context["tool_events"][0]["outcome"], "error")
        self.assertFalse(json.loads(normalized[1]["content"])["success"])


if __name__ == "__main__":
    unittest.main()
