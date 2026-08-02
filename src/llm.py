from __future__ import annotations

import json
import os
from typing import Any


DEFAULT_MODEL = "gpt-5.6-terra"
SYSTEM_PROMPT = """你是一名严谨的数据分析专家和业务策略顾问。

工作规则：
1. 必须先读取 get_analysis_summary 工具返回的结构化统计，再回答用户。
2. 只基于工具数据和用户提供的业务背景进行推断，不得编造指标、因果关系或行业事实。
3. 明确区分“数据事实”“合理推断”和“仍需验证的假设”。
4. 输出包含：核心发现、业务解释、可执行策略、建议验证指标、局限性。
5. 建议要具体但克制，避免空泛口号；若上下文不足，明确指出还需要哪些字段。
6. 使用中文，标题清晰，优先给出最重要的三项建议。
"""

ANALYSIS_TOOL = {
    "type": "function",
    "name": "get_analysis_summary",
    "description": "读取已清洗数据的描述性统计、清洗记录和用户选择的聚合分析结果。",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True,
}


class StrategyGenerationError(RuntimeError):
    """Raised when the strategy generation flow cannot complete."""


def resolve_api_key(user_key: str | None = None) -> str | None:
    return (user_key or "").strip() or os.getenv("OPENAI_API_KEY")


def generate_strategy(
    user_prompt: str,
    analysis_context: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> str:
    if not user_prompt.strip():
        raise StrategyGenerationError("请先填写业务问题。")
    if not api_key:
        raise StrategyGenerationError("未配置 OpenAI API Key。")
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise StrategyGenerationError("缺少 openai 依赖，请先安装 requirements.txt。") from exc
        client = OpenAI(api_key=api_key)

    input_items: list[Any] = [{"role": "user", "content": user_prompt.strip()}]
    try:
        first_response = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            instructions=SYSTEM_PROMPT,
            input=input_items,
            tools=[ANALYSIS_TOOL],
            tool_choice={"type": "function", "name": "get_analysis_summary"},
        )
        input_items.extend(first_response.output)
        tool_calls = [item for item in first_response.output if getattr(item, "type", None) == "function_call"]
        if not tool_calls:
            raise StrategyGenerationError("模型没有请求分析工具，请稍后重试。")
        context_json = json.dumps(analysis_context, ensure_ascii=False, default=str)
        for call in tool_calls:
            if getattr(call, "name", None) != "get_analysis_summary":
                raise StrategyGenerationError(f"模型请求了未授权工具：{getattr(call, 'name', '')}")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": context_json,
                }
            )
        final_response = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            instructions=SYSTEM_PROMPT,
            input=input_items,
            tools=[ANALYSIS_TOOL],
            tool_choice="none",
        )
    except StrategyGenerationError:
        raise
    except Exception as exc:
        raise StrategyGenerationError(f"调用模型失败：{exc}") from exc

    output = getattr(final_response, "output_text", "").strip()
    if not output:
        raise StrategyGenerationError("模型未返回可展示的策略内容。")
    return output
