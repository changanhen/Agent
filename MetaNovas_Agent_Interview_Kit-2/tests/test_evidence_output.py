"""Source, optimum, arithmetic and customer-output regression checks.

These exercise real guards with fictional tool evidence; they do not score a
real model's understanding of the policy or prose quality.
"""

import unittest
from copy import deepcopy

import agent
from scripts.intent_change_runner import prepare_scenario, append_tool
from scripts.practice_runner import TOOLS
from tests.test_agent import report


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        case = prepare_scenario("revise_before_confirmation")
        self.messages = case["messages"][:-1]  # Original 12-hour requirement.
        self.context = agent.replay_history(self.messages)
        self.context["intent_state"] = {"version": 2, "tool_effects": {
            "locate_member": "authentication", "read_member_profile": "read",
            "read_reservation": "read", "inspect_asset_family": "read",
            "compute_amount": "other", "change_reserved_asset": "write"}}
        self.candidate = {"content": "Please confirm the exact option and fee."}
        self.report = report(response_kind="confirmation")
        self.options = case["before"]["asset_families"][next(iter(case["before"]["asset_families"]))]["options"]
        self.selected = case["old_call"]["arguments"]["new_asset_id"]
        self.write = {"tool_call": case["old_call"]}
        self.policy = "Only project credit may be used. There is no cancellation fee."

    def evaluate(self):
        return agent.check_evidence_judgment(self.candidate, self.messages, self.context, self.report, self.policy)

    def selection(self):
        return {"requirements": ["available", "balanced XLR", "at least 12 battery hours"],
            "complete_set": True, "source_indexes": [8], "temporal": "current",
            "objective": "min", "objective_kind": "number",
            "objective_field": "specifications.weight_grams", "selected_id": self.selected,
            "candidates": [{"id": key, "checks": [value["available"], value["specifications"]["balanced_xlr"],
                value["specifications"]["battery_hours"] >= 12],
                "objective_value": value["specifications"]["weight_grams"]} for key, value in self.options.items()]}

    def add_selection(self):
        self.report["evidence_review"].update(selection_required=True, selection=self.selection())

    def fact(self, value, indexes, **fields):
        row = {"value": value, "source_indexes": indexes, "origin": "tool", "temporal": "current", "value_kind": "value"}
        row.update(fields)
        self.report["evidence_review"]["facts"] = [row]

    def calculation(self, inputs, result, operation="subtract", **fields):
        row = {"operation": operation, "inputs": inputs, "result": result, "source_indexes": [4, 6, 8]}
        row.update(fields)
        self.report["evidence_review"]["calculations"] = [row]

    def test_valid_lightest_eligible_option_passes(self):
        self.add_selection()
        self.assertEqual(self.evaluate(), [])

    def test_unavailable_option_is_rejected_even_if_lighter(self):
        self.add_selection()
        self.report["evidence_review"]["selection"]["selected_id"] = next(key for key, value in self.options.items() if not value["available"])
        self.assertTrue(self.evaluate())

    def test_short_battery_option_is_rejected(self):
        self.add_selection()
        self.report["evidence_review"]["selection"]["selected_id"] = next(key for key, value in self.options.items()
            if value["available"] and value["specifications"]["balanced_xlr"] and value["specifications"]["battery_hours"] < 12)
        self.assertTrue(self.evaluate())

    def test_eligible_but_heavier_option_is_rejected(self):
        self.add_selection()
        matrix = self.report["evidence_review"]["selection"]
        matrix["selected_id"] = next(key for key, value in self.options.items()
            if value["available"] and value["specifications"]["battery_hours"] >= 18)
        self.assertTrue(self.evaluate())

    def test_incomplete_comparison_and_unknown_checks_are_rejected(self):
        self.add_selection()
        matrix = self.report["evidence_review"]["selection"]
        matrix["complete_set"] = False
        self.assertTrue(self.evaluate())
        matrix["complete_set"] = True
        next(row for row in matrix["candidates"] if row["id"] == self.selected)["checks"][0] = None
        self.assertTrue(self.evaluate())

    def test_invented_option_and_objective_are_rejected(self):
        self.add_selection()
        matrix = self.report["evidence_review"]["selection"]
        matrix["candidates"][0]["id"] = "NOT-IN-DATABASE"
        self.assertTrue(self.evaluate())
        matrix["candidates"] = self.selection()["candidates"]
        next(row for row in matrix["candidates"] if row["id"] == self.selected)["objective_value"] = 1
        self.assertTrue(self.evaluate())

    def test_price_cannot_be_substituted_for_weight_objective(self):
        self.add_selection()
        matrix = self.report["evidence_review"]["selection"]
        matrix["selected_id"] = min((key for key, value in self.options.items() if value["available"]
            and value["specifications"]["balanced_xlr"] and value["specifications"]["battery_hours"] >= 12),
            key=lambda key: self.options[key]["fee"])
        for row in matrix["candidates"]:
            row["objective_value"] = self.options[row["id"]]["fee"]
        self.assertTrue(self.evaluate())

    def test_write_must_match_compared_selection(self):
        self.add_selection()
        self.candidate = deepcopy(self.write)
        self.assertEqual(self.evaluate(), [])
        alternative = next(key for key in self.options if key != self.selected)
        self.candidate["tool_call"]["arguments"]["new_asset_id"] = alternative
        self.assertTrue(self.evaluate())

    def test_unknown_reference_is_rejected_before_dispatch(self):
        call = deepcopy(self.write["tool_call"])
        call["arguments"]["new_asset_id"] = "MADE-UP-OPTION"
        self.assertTrue(agent.check_argument_evidence(call, self.messages, agent.build_tool_catalog(TOOLS)))
        self.assertEqual(agent.check_argument_evidence(self.write["tool_call"], self.messages, agent.build_tool_catalog(TOOLS)), [])

    def test_valid_fee_difference_and_derived_fact_pass(self):
        self.calculation([53.5, 39.5], 14)
        self.fact(14, [6, 8], origin="derived")
        self.assertEqual(self.evaluate(), [])

    def test_prior_verified_result_can_feed_balance_calculation(self):
        self.report["evidence_review"]["calculations"] = [
            {"operation": "subtract", "inputs": [53.5, 39.5], "result": 14, "source_indexes": [6, 8]},
            {"operation": "subtract", "inputs": [75, 14], "result": 61, "source_indexes": [4]}]
        self.fact(61, [4, 6, 8], origin="derived")
        self.assertEqual(self.evaluate(), [])

    def test_valid_unit_conversion_passes_but_invented_factor_fails(self):
        self.calculation([365, 0.001], 0.365, operation="multiply", constants=[
            {"basis": "unit_conversion", "value": 0.001, "from_unit": "g", "to_unit": "kg"}])
        self.assertEqual(self.evaluate(), [])
        self.report["evidence_review"]["calculations"][0]["constants"][0]["value"] = 9998
        self.assertTrue(self.evaluate())

    def test_rounding_is_explicit_and_exact(self):
        self.calculation([53.5, 39.5], 1.35, operation="divide", round_places=2, rounding="half_even")
        self.assertEqual(self.evaluate(), [])
        self.report["evidence_review"]["calculations"][0]["result"] = 1.36
        self.assertTrue(self.evaluate())

    def test_division_by_zero_and_boolean_inputs_are_rejected(self):
        self.calculation([75, 0], 0, operation="divide", constants=[{"basis": "identity", "value": 0}])
        self.assertTrue(self.evaluate())
        self.calculation([True, 1], 2, operation="sum", constants=[{"basis": "identity", "value": 1}])
        self.assertTrue(self.evaluate())

    def test_policy_constant_needs_exact_policy_and_matching_number(self):
        self.policy += " The fixed adjustment is 2."
        self.calculation([75, 2], 73, constants=[{"basis": "policy", "value": 2,
            "policy_quote": "The fixed adjustment is 2."}])
        self.assertEqual(self.evaluate(), [])
        self.report["evidence_review"]["calculations"][0]["constants"][0]["value"] = 3
        self.assertTrue(self.evaluate())

    def test_current_selection_cannot_reuse_option_data_after_a_write(self):
        self.add_selection()
        append_tool(self.messages, self.write["tool_call"], {"success": True})
        state = self.context["intent_state"]
        self.context = agent.replay_history(self.messages)
        self.context["intent_state"] = state
        self.assertTrue(self.evaluate())
        self.candidate = {"content": "The selected option was the lightest eligible option when changed."}
        self.report["response_kind"] = "final"
        self.report["evidence_review"]["selection"]["temporal"] = "historical"
        self.assertEqual(self.evaluate(), [])

    def test_wrong_arithmetic_and_unchecked_derived_amount_are_rejected(self):
        self.calculation([53.5, 39.5], 15)
        self.assertTrue(self.evaluate())
        self.report["evidence_review"]["calculations"] = []
        self.fact(14, [6, 8], origin="derived")
        self.assertTrue(self.evaluate())

    def test_correct_arithmetic_cannot_use_invented_input_numbers(self):
        self.calculation([9999, 9998], 1)
        self.assertTrue(self.evaluate())

    def test_wrong_role_and_absent_source_are_rejected(self):
        self.fact(75, [0])
        self.assertTrue(self.evaluate())
        self.fact(75, [999])
        self.assertTrue(self.evaluate())

    def test_claimed_value_must_occur_in_sources(self):
        self.fact(9999, [4])
        self.assertTrue(self.evaluate())
        self.fact(75, [4])
        self.assertEqual(self.evaluate(), [])

    def test_boolean_false_and_status_field_name_are_not_availability_proof(self):
        self.assertFalse(agent.contains_evidence_value(False, 0))
        self.assertFalse(agent.contains_evidence_value("unavailable", "available", False))
        self.fact("available", [8])
        self.assertTrue(self.evaluate())

    def test_reference_dictionary_keys_are_valid_but_not_ordinary_fact_values(self):
        self.assertTrue(agent.contains_evidence_value({"OPT-1": {"weight": 1}}, "OPT-1"))
        self.assertFalse(agent.contains_evidence_value({"available": False}, "available", False))

    def test_newer_read_invalidates_older_current_claim(self):
        append_tool(self.messages, {"name": "read_member_profile", "arguments": {"member_id": next(iter(self.context["tool_events"][1]["arguments"].values()))}}, {"balance": 60})
        old_state = self.context["intent_state"]
        self.context = agent.replay_history(self.messages)
        self.context["intent_state"] = old_state
        self.fact(75, [4])
        self.assertTrue(self.evaluate())
        self.fact(75, [4], temporal="historical")
        self.assertEqual(self.evaluate(), [])

    def test_transaction_does_not_revoke_verified_identity(self):
        append_tool(self.messages, self.write["tool_call"], {"success": True})
        state = self.context["intent_state"]
        self.context = agent.replay_history(self.messages)
        self.context["intent_state"] = state
        self.assertTrue(agent.evidence_is_current(2, self.context))
        self.assertFalse(agent.evidence_is_current(6, self.context))

    def test_policy_fact_requires_actual_policy_quote(self):
        self.fact(0, [], origin="policy", policy_quote="There is no cancellation fee.")
        self.assertEqual(self.evaluate(), [])
        self.report["evidence_review"]["facts"][0]["policy_quote"] = "All changes are free."
        self.assertTrue(self.evaluate())

    def test_chinese_and_english_reference_punctuation(self):
        self.assertTrue(agent.contains_evidence_value("My code is 588701.", "588701"))
        self.assertTrue(agent.contains_evidence_value("我的会员码是588701。", "588701"))
        self.assertFalse(agent.contains_evidence_value("Reference RS-12345 is active.", "12345"))


class CustomerOutputTests(unittest.TestCase):
    def setUp(self):
        self.context = {"intent_state": {"version": 2}}

    def test_supported_direct_complete_reply_passes(self):
        self.assertEqual(agent.check_customer_output({"content": "Your replacement succeeded; additional fee 14 from project credit."}, self.context, report()), [])

    def test_each_output_quality_failure_is_enforced(self):
        for key in report()["customer_output"]:
            value = report()
            value["customer_output"][key] = False
            with self.subTest(check=key):
                issues = agent.check_customer_output({"content": "Done."}, self.context, value)
                self.assertTrue(any(key in issue for issue in issues))

    def test_missing_output_audit_is_not_silently_approved(self):
        self.assertTrue(agent.check_customer_output({"content": "Done."}, self.context, {}))

    def test_tool_calls_do_not_need_customer_facing_prose(self):
        self.assertEqual(agent.check_customer_output({"tool_call": {"name": "read", "arguments": {}}}, self.context, {}), [])


if __name__ == "__main__":
    unittest.main()
