"""Thin Anthropic client wrapper: tool-call structured output + usage tracking."""

from dataclasses import dataclass
from typing import Any, Type, TypeVar

import anthropic
from pydantic import BaseModel

from src import config

T = TypeVar("T", bound=BaseModel)

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "fill in your key."
            )
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response: Any) -> None:
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens


def structured_call(
    system: str,
    user: str,
    output_model: Type[T],
    tool_name: str,
    tool_description: str,
    usage: Usage | None = None,
    model: str | None = None,
) -> T:
    """Force the model to answer via a single tool call whose input schema is
    the given Pydantic model; parse and return the validated instance."""
    client = get_client()
    response = client.messages.create(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": output_model.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
    )
    if usage is not None:
        usage.add(response)
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return output_model.model_validate(block.input)
    raise RuntimeError(f"Model did not return the expected {tool_name} tool call.")


def text_call(
    system: str,
    user: str,
    usage: Usage | None = None,
    model: str | None = None,
) -> str:
    client = get_client()
    response = client.messages.create(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if usage is not None:
        usage.add(response)
    return "".join(block.text for block in response.content if block.type == "text")
