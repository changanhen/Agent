"""Check that the uploaded file works with just stdlib and the host SDK."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import agent
from tests.test_agent import report


class SingleFileSubmissionTests(unittest.TestCase):
    def test_only_agent_and_platform_sdk_are_needed(self):
        state = {"goals": [{"id": "g1", "request": "Greet the customer", "target": None,
            "kind": "information", "status": "pending", "source_user_indexes": [0],
            "last_changed_index": 0, "constraints": [], "evidence_indexes": []}],
            "tool_effects": {}, "allowed_writes": [],
            "latest_update": {"index": 0, "kind": "new_request", "affected_goal_ids": ["g1"]},
            "batch_plan": {"required": False, "inventory_complete": False, "start_index": None, "records": []}}
        review = report(resolved_goal_ids=["g1"])
        replies = [{"content": json.dumps(state)}, {"content": "Hello."}, {"content": json.dumps(review)}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path(agent.__file__).read_text(encoding="utf-8-sig")
            (root / "agent.py").write_text(source, encoding="utf-8")
            sdk = ("class ModelError(RuntimeError): pass\n"
                   + "replies = iter(" + repr(replies) + ")\n"
                   + "def chat(*args, **kwargs): return next(replies)\n")
            (root / "interview_sdk.py").write_text(sdk, encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", "-c",
                "import agent,json; print(json.dumps(agent.respond([{'role':'user','content':'Hello'}], [], 'Be helpful.')))"],
                cwd=directory, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"content": "Hello."})
            self.assertLess(len(source.encode("utf-8")), 150_000)


if __name__ == "__main__":
    unittest.main()
