"""Thin Google Gemini client wrapper: JSON-schema structured output + usage
tracking.

Structured output is obtained by forcing a JSON response
(``response_mime_type="application/json"``) and embedding the target model's
JSON Schema in the system instruction, then validating the returned JSON with
Pydantic. This prompt-plus-schema approach is used instead of Gemini's
constrained ``response_schema`` decoding because our contracts include
free-form ``dict`` fields (e.g. ``ProblemIntake.entities``), which Gemini's
constrained-decoding schema does not accept (it requires every OBJECT to
declare properties). The Validator + retry loop downstream compensates for the
softer guarantee on the modeling spec.
"""

import json
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from src import config

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill "
                "in your Google Gemini API key (GOOGLE_API_KEY also works)."
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response: Any) -> None:
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            self.input_tokens += getattr(meta, "prompt_token_count", 0) or 0
            self.output_tokens += getattr(meta, "candidates_token_count", 0) or 0


def structured_call(
    system: str,
    user: str,
    output_model: Type[T],
    tool_name: str,
    tool_description: str,
    usage: Usage | None = None,
    model: str | None = None,
) -> T:
    """Force a JSON response matching ``output_model`` and return the validated
    instance. ``tool_name``/``tool_description`` are folded into the instruction
    to describe the expected output object."""
    schema = json.dumps(output_model.model_json_schema(), indent=2)
    system_instruction = (
        f"{system}\n\n"
        f"Your task ({tool_name}): {tool_description}\n"
        "Respond with a SINGLE JSON object and nothing else — no prose, no "
        "markdown, no code fences. It MUST conform to this JSON Schema:\n"
        f"{schema}"
    )
    client = get_client()
    response = client.models.generate_content(
        model=model or config.GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            max_output_tokens=config.MAX_TOKENS,
            response_mime_type="application/json",
        ),
    )
    if usage is not None:
        usage.add(response)
    text = response.text
    if not text or not text.strip():
        raise RuntimeError(
            f"Gemini returned no content for {tool_name}; the response may have "
            "been truncated or blocked."
        )
    return output_model.model_validate_json(text)


def text_call(
    system: str,
    user: str,
    usage: Usage | None = None,
    model: str | None = None,
) -> str:
    client = get_client()
    response = client.models.generate_content(
        model=model or config.GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=config.MAX_TOKENS,
        ),
    )
    if usage is not None:
        usage.add(response)
    return response.text or ""
