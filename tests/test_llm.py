from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.llm import ANALYSIS_TOOL, SYSTEM_PROMPT, generate_strategy


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="call-analysis-001",
                type="function",
                function=SimpleNamespace(name="get_analysis_summary", arguments="{}"),
            )
            message = SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        message = SimpleNamespace(role="assistant", content="## 核心发现\n收入主要来自老客。", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


class LlmTests(unittest.TestCase):
    def test_strategy_uses_deepseek_tool_message(self) -> None:
        client = FakeClient()
        context = {"dataset_overview": {"rows": 10}, "user_aggregation": {"description": "按客群求和"}}
        result = generate_strategy("请给出增长建议", context, "test-key", client=client)
        self.assertIn("核心发现", result)
        self.assertEqual(len(client.chat.completions.calls), 2)
        first, second = client.chat.completions.calls
        self.assertEqual(first["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(first["tools"], [ANALYSIS_TOOL])
        self.assertEqual(first["tool_choice"]["function"]["name"], "get_analysis_summary")
        self.assertEqual(first["extra_body"], {"thinking": {"type": "disabled"}})
        tool_outputs = [item for item in second["messages"] if isinstance(item, dict) and item.get("role") == "tool"]
        self.assertEqual(len(tool_outputs), 1)
        self.assertEqual(tool_outputs[0]["tool_call_id"], "call-analysis-001")
        self.assertEqual(json.loads(tool_outputs[0]["content"]), context)
        self.assertEqual(second["tool_choice"], "none")


if __name__ == "__main__":
    unittest.main()
