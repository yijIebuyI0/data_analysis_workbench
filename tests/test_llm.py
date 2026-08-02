from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.llm import ANALYSIS_TOOL, SYSTEM_PROMPT, generate_strategy


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                type="function_call",
                name="get_analysis_summary",
                call_id="call-analysis-001",
                arguments="{}",
            )
            return SimpleNamespace(output=[tool_call], output_text="")
        return SimpleNamespace(output=[], output_text="## 核心发现\n收入主要来自老客。")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class LlmTests(unittest.TestCase):
    def test_strategy_uses_real_function_call_output(self) -> None:
        client = FakeClient()
        context = {"dataset_overview": {"rows": 10}, "user_aggregation": {"description": "按客群求和"}}
        result = generate_strategy("请给出增长建议", context, "test-key", client=client)
        self.assertIn("核心发现", result)
        self.assertEqual(len(client.responses.calls), 2)
        first, second = client.responses.calls
        self.assertEqual(first["instructions"], SYSTEM_PROMPT)
        self.assertEqual(first["tools"], [ANALYSIS_TOOL])
        self.assertEqual(first["tool_choice"]["name"], "get_analysis_summary")
        tool_outputs = [item for item in second["input"] if isinstance(item, dict) and item.get("type") == "function_call_output"]
        self.assertEqual(len(tool_outputs), 1)
        self.assertEqual(tool_outputs[0]["call_id"], "call-analysis-001")
        self.assertEqual(json.loads(tool_outputs[0]["output"]), context)
        self.assertEqual(second["tool_choice"], "none")


if __name__ == "__main__":
    unittest.main()
