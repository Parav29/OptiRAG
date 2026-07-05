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
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from src import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None

# --- client-side throttle so we stay under the free-tier rate limit ---------
_rate_lock = threading.Lock()
_last_call = [0.0]


def _throttle() -> None:
    """Block until at least (60 / GEMINI_CALLS_PER_MIN) seconds have elapsed
    since the previous call, so bursts never exceed the configured rate."""
    if config.GEMINI_CALLS_PER_MIN <= 0:
        return
    min_interval = 60.0 / config.GEMINI_CALLS_PER_MIN
    with _rate_lock:
        wait = _last_call[0] + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _is_transient(exc: Exception) -> bool:
    """True for errors worth retrying: rate limits (429) and transient server
    overload (503/500), as opposed to permanent errors like a bad API key or
    a malformed request (400/403), which should surface immediately."""
    text = str(exc).upper()
    code = getattr(exc, "code", None)
    if code in (429, 503, 500):
        return True
    return any(marker in text for marker in (
        "RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503",
        "INTERNAL", "OVERLOADED",
    )) or ("QUOTA" in text and "EXCEED" in text)


def _retry_delay_hint(exc: Exception) -> float | None:
    """Honour a server-suggested retry delay if the error carries one."""
    match = re.search(r"retry.?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)",
                      str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _generate(model: str, contents: str, gen_config: "types.GenerateContentConfig"):
    """One Gemini call with throttling and exponential backoff on transient
    errors (429 rate limit, 503/500 server overload)."""
    client = get_client()
    last_exc: Exception | None = None
    for attempt in range(config.GEMINI_MAX_RETRIES):
        _throttle()
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=gen_config
            )
        except Exception as exc:  # noqa: BLE001 - inspect to decide retry
            last_exc = exc
            if not _is_transient(exc) or attempt == config.GEMINI_MAX_RETRIES - 1:
                raise
            backoff = _retry_delay_hint(exc) or min(
                config.GEMINI_MAX_BACKOFF_S, 2.0 * (2 ** attempt)
            )
            backoff += random.uniform(0, 1.0)  # jitter
            logger.warning(
                "Transient error (attempt %d/%d); backing off %.1fs: %s",
                attempt + 1, config.GEMINI_MAX_RETRIES, backoff,
                str(exc)[:120],
            )
            time.sleep(backoff)
    raise last_exc  # pragma: no cover


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
    response = _generate(
        model=model or config.GEMINI_MODEL,
        contents=user,
        gen_config=types.GenerateContentConfig(
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
    response = _generate(
        model=model or config.GEMINI_MODEL,
        contents=user,
        gen_config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=config.MAX_TOKENS,
        ),
    )
    if usage is not None:
        usage.add(response)
    return response.text or ""
