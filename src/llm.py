from __future__ import annotations

import json
import os
from typing import Any


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
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
    "function": {
        "name": "get_analysis_summary",
        "description": "读取已清洗数据的描述性统计、清洗记录和用户选择的聚合分析结果。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


class StrategyGenerationError(RuntimeError):
    """Raised when the strategy generation flow cannot complete."""


def resolve_api_key(user_key: str | None = None) -> str | None:
    return (user_key or "").strip() or os.getenv("DEEPSEEK_API_KEY")


def generate_strategy(
    user_prompt: str,
    analysis_context: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    client: Any | None = None,
) -> str:
    if not user_prompt.strip():
        raise StrategyGenerationError("请先填写业务问题。")
    if not api_key:
        raise StrategyGenerationError("未配置 DeepSeek API Key。")
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise StrategyGenerationError("缺少 openai 依赖，请先安装 requirements.txt。") from exc
        client = OpenAI(api_key=api_key, base_url=base_url)

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt.strip()},
    ]
    try:
        first_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[ANALYSIS_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_analysis_summary"}},
            extra_body={"thinking": {"type": "disabled"}},
        )
        assistant_message = first_response.choices[0].message
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        if not tool_calls:
            raise StrategyGenerationError("模型没有请求分析工具，请稍后重试。")
        messages.append(assistant_message)
        context_json = json.dumps(analysis_context, ensure_ascii=False, default=str)
        for call in tool_calls:
            function = getattr(call, "function", None)
            function_name = getattr(function, "name", "")
            if function_name != "get_analysis_summary":
                raise StrategyGenerationError(f"模型请求了未授权工具：{function_name}")
            try:
                arguments = json.loads(getattr(function, "arguments", "{}") or "{}")
            except json.JSONDecodeError as exc:
                raise StrategyGenerationError("模型返回了无法解析的工具参数。") from exc
            if arguments:
                raise StrategyGenerationError("分析摘要工具不接受参数，已拒绝本次调用。")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": context_json,
                }
            )
        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[ANALYSIS_TOOL],
            tool_choice="none",
            extra_body={"thinking": {"type": "disabled"}},
        )
    except StrategyGenerationError:
        raise
    except Exception as exc:
        raise StrategyGenerationError(f"调用 DeepSeek 失败：{exc}") from exc

    output = (final_response.choices[0].message.content or "").strip()
    if not output:
        raise StrategyGenerationError("模型未返回可展示的策略内容。")
    return output
