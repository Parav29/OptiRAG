"""Thin LLM client wrapper over NVIDIA NIM's OpenAI-compatible endpoint:
JSON-schema structured output + usage tracking + rate limiting.

Structured output is obtained by embedding the target model's JSON Schema in
the system prompt, requesting JSON, and validating the returned text with
Pydantic. We also request OpenAI-style ``response_format={"type":
"json_object"}`` when the endpoint accepts it, and fall back to defensive JSON
extraction (stripping code fences / surrounding prose) otherwise, so the layer
is robust across models that do not honour JSON mode. The Validator + retry
loop downstream compensates for any residual softness on the modeling spec.

The provider is isolated here: every agent calls ``structured_call`` /
``text_call``, so pointing at a different OpenAI-compatible endpoint is just a
base-URL + key change in ``src.config``.
"""

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from src import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: OpenAI | None = None

# --- client-side throttle so we stay under the free-tier rate limit ---------
_rate_lock = threading.Lock()
_last_call = [0.0]


def _throttle() -> None:
    """Block until at least (60 / LLM_CALLS_PER_MIN) seconds have elapsed since
    the previous call, so bursts never exceed the configured rate."""
    if config.LLM_CALLS_PER_MIN <= 0:
        return
    min_interval = 60.0 / config.LLM_CALLS_PER_MIN
    with _rate_lock:
        wait = _last_call[0] + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _is_transient(exc: Exception) -> bool:
    """True for errors worth retrying: rate limits (429) and transient server
    overload (503/500), as opposed to permanent errors like a bad API key or a
    malformed request (400/401/403), which should surface immediately."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, 500, 503):
        return True
    text = str(exc).upper()
    return any(marker in text for marker in (
        "RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503", "500",
        "INTERNAL", "OVERLOADED", "RATE LIMIT", "TIMEOUT", "TIMED OUT", "CONNECT",
    )) or ("QUOTA" in text and "EXCEED" in text)


def _retry_delay_hint(exc: Exception) -> float | None:
    """Honour a server-suggested retry delay if the error carries one."""
    match = re.search(r"retry.?(?:delay|after)['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)",
                      str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and fill "
                "in your NVIDIA NIM API key (get one at https://build.nvidia.com)."
            )
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response: Any) -> None:
        meta = getattr(response, "usage", None)
        if meta is not None:
            self.input_tokens += getattr(meta, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(meta, "completion_tokens", 0) or 0


def _chat(model: str, system: str, user: str, json_mode: bool):
    """One chat-completion call with throttling and backoff on transient errors.

    Tries OpenAI-style JSON mode first (when requested); if the endpoint or
    model rejects that parameter, retries once without it."""
    client = get_client()
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=config.MAX_TOKENS,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(config.LLM_MAX_RETRIES):
        _throttle()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - inspect to decide retry
            last_exc = exc
            # Some models reject response_format; drop it and retry immediately.
            if json_mode and "response_format" in kwargs and _rejects_json_mode(exc):
                logger.info("Model %s rejects JSON mode; retrying without it.", model)
                kwargs.pop("response_format")
                continue
            if not _is_transient(exc) or attempt == config.LLM_MAX_RETRIES - 1:
                raise
            backoff = _retry_delay_hint(exc) or min(
                config.LLM_MAX_BACKOFF_S, 2.0 * (2 ** attempt)
            )
            backoff += random.uniform(0, 1.0)  # jitter
            logger.warning(
                "Transient error (attempt %d/%d); backing off %.1fs: %s",
                attempt + 1, config.LLM_MAX_RETRIES, backoff, str(exc)[:120],
            )
            time.sleep(backoff)
    raise last_exc  # pragma: no cover


def _rejects_json_mode(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text or (
        "400" in text and "json" in text
    )


def _extract_json(text: str) -> str:
    """Pull a single JSON object out of a model response that may be wrapped in
    markdown fences or surrounded by prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


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
    instance. ``tool_name``/``tool_description`` describe the expected object."""
    schema = json.dumps(output_model.model_json_schema(), indent=2)
    system_instruction = (
        f"{system}\n\n"
        f"Your task ({tool_name}): {tool_description}\n"
        "Respond with a SINGLE JSON object and nothing else — no prose, no "
        "markdown, no code fences. It MUST conform to this JSON Schema:\n"
        f"{schema}"
    )
    response = _chat(
        model=model or config.LLM_MODEL,
        system=system_instruction,
        user=user,
        json_mode=True,
    )
    if usage is not None:
        usage.add(response)
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError(
            f"Model returned no content for {tool_name}; the response may have "
            "been truncated or filtered."
        )
    return output_model.model_validate_json(_extract_json(text))


def text_call(
    system: str,
    user: str,
    usage: Usage | None = None,
    model: str | None = None,
) -> str:
    response = _chat(
        model=model or config.LLM_MODEL,
        system=system,
        user=user,
        json_mode=False,
    )
    if usage is not None:
        usage.add(response)
    return response.choices[0].message.content or ""
