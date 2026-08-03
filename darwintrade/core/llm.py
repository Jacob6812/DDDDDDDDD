from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

_TOKEN_LIMIT_KWARGS = {"max_tokens", "max_completion_tokens", "max_output_tokens"}
_INVALID_JSON_ESCAPE = re.compile(r'\\([^"\\/bfnrtu])')


def _looks_like_bad_param(exc: Exception, param: str) -> bool:
    """Heuristic: did the provider reject a specific request parameter?
    Matches the common OpenAI-compatible error shapes for unknown/unsupported
    params so we can drop the param and retry rather than fail the whole call."""
    msg = str(exc).lower()
    if param.lower() not in msg:
        return False
    return any(
        marker in msg
        for marker in (
            "unknown", "unsupported", "unexpected", "not supported",
            "invalid", "unrecognized", "unrecognised", "extra fields",
            "additional propert", "does not support",
        )
    )


def _looks_like_json_schema_rejected(exc: Exception) -> bool:
    """Detect providers that accept `response_format` but not the `json_schema`
    variant (e.g. DeepSeek returns 400 'This response_format type is unavailable
    now'). Such providers still honor the simpler `json_object` mode, so we can
    downgrade and retry rather than fail the whole run."""
    msg = str(exc).lower()
    if "response_format" not in msg and "json_schema" not in msg and "json schema" not in msg:
        return False
    return any(
        marker in msg
        for marker in (
            "unavailable", "not available", "unknown", "unsupported",
            "unexpected", "not supported", "invalid", "unrecognized",
            "unrecognised", "does not support",
        )
    )


def _invoke_with_retries(
    operation: Callable[[], Any],
    *,
    retries: int,
    retry_interval_seconds: float,
    caller_tag: str = "unknown",
) -> Any:
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= retries:
                raise
            msg = str(exc).replace("\n", " ")[:500]
            logger.warning(
                "LLM retry caller=%s attempt=%s/%s exc=%s msg=%s",
                caller_tag, attempt + 2, retries + 1, exc.__class__.__name__, msg,
            )
            time.sleep(retry_interval_seconds)
    raise RuntimeError("retry loop exhausted")
def _with_validation_feedback(user_prompt: str, exc: Exception | None) -> str:
    if exc is None:
        return user_prompt
    feedback = {
        "previous_response_validation_error": str(exc).replace("\n", " ")[:1200],
        "repair_instruction": (
            "Return a fresh JSON object satisfying the output_contract. "
            "Do not repeat structures named in previous_response_validation_error."
        ),
    }
    return user_prompt + "\n\n" + json.dumps({"validation_feedback_for_retry": feedback}, ensure_ascii=True)


def build_json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def _is_json_schema_format(response_format: dict[str, Any] | None) -> bool:
    return bool(response_format) and response_format.get("type") == "json_schema"


def _resolve_response_format(
    system_prompt: str,
    response_format: dict[str, Any] | None,
    *,
    downgraded: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Return the (system_prompt, response_format) to send.

    When `downgraded` is True and the caller asked for a `json_schema` format,
    swap to the more broadly supported `json_object` mode and append the schema
    to the system prompt so the model still knows the required shape."""
    if not (downgraded and _is_json_schema_format(response_format)):
        return system_prompt, response_format
    schema = response_format["json_schema"].get("schema", {})
    schema_text = json.dumps(schema, ensure_ascii=True)
    augmented = (
        f"{system_prompt}\n\n"
        "Respond with a single JSON object that conforms exactly to this JSON schema "
        f"(no markdown, no commentary):\n{schema_text}"
    )
    return augmented, {"type": "json_object"}


def normalize_json_text(raw: str, *, repair_invalid_escapes: bool = False) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:].strip()
    if repair_invalid_escapes:
        text = _INVALID_JSON_ESCAPE.sub(r'\1', text)
    return text


def parse_json_dict(raw: str, *, repair_invalid_escapes: bool = False) -> dict[str, Any]:
    text = normalize_json_text(raw, repair_invalid_escapes=repair_invalid_escapes)
    try:
        parsed = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("LLM did not return valid JSON") from None
        parsed = json.loads(text[start:end + 1])
    return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}


class LLMClient:
    """Thin OpenAI-compatible LLM client with retry and JSON validation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 100,
        retry_interval_seconds: float = 3.0,
        request_timeout_seconds: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_interval_seconds = retry_interval_seconds
        # Per-request hard timeout passed to both OpenAI() and every
        # chat.completions.create call. Under high symbol-parallelism the
        # provider queues responses, so a single request can legitimately
        # take minutes; 600s gives slow requests room to complete instead of
        # timing out and re-queuing (which compounds the queue). Overridable
        # via LLM_REQUEST_TIMEOUT env. SDK-level retries are disabled below so
        # retry policy stays owned by _invoke_with_retries.
        if request_timeout_seconds is None:
            env_to = os.getenv("LLM_REQUEST_TIMEOUT", "").strip()
            try:
                request_timeout_seconds = float(env_to) if env_to else 600.0
            except ValueError:
                request_timeout_seconds = 600.0
        self.request_timeout_seconds = float(request_timeout_seconds)
        # Deterministic sampling. temperature=0 alone does NOT guarantee
        # reproducible outputs on most serving stacks (MoE routing, batching,
        # float non-associativity), so two backtest runs got DIFFERENT analyst
        # signals for the same date/inputs — making baseline-vs-ablation
        # comparisons measure LLM sampling noise, not the ablated component.
        # A fixed seed makes runs reproducible on providers that honor it; on
        # providers that reject the param, the call auto-retries without it.
        if seed is None:
            env_seed = os.getenv("LLM_SEED", "").strip()
            seed = int(env_seed) if env_seed.lstrip("-").isdigit() else 42
        self.seed = seed
        # Flipped to True once a provider rejects `seed`, so we stop sending it.
        self._seed_unsupported = False
        # Flipped to True once a provider rejects the `json_schema` response
        # format (e.g. DeepSeek), so we fall back to `json_object` + in-prompt
        # schema for the rest of the run.
        self._json_schema_unsupported = False

    def _get_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.base_url or None,
            api_key=self.api_key or "no-key",
            timeout=self.request_timeout_seconds,
            # Disable SDK-level retries: retry policy is owned by
            # _invoke_with_retries (up to max_retries, retry_interval_seconds).
            # Leaving the SDK's default 2 retries on top would double-count
            # attempts and inflate wall-clock under load.
            max_retries=0,
        )

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        caller_tag: str = "unknown",
        response_format: dict[str, Any] | None = None,
        validator: Callable[[str], Any] | None = None,
    ) -> str:
        last_exc: Exception | None = None

        def _call() -> str:
            nonlocal last_exc
            effective_prompt = _with_validation_feedback(user_prompt, last_exc) if validator else user_prompt
            effective_system, effective_rf = _resolve_response_format(
                system_prompt, response_format, downgraded=self._json_schema_unsupported
            )
            kwargs: dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": effective_prompt},
                ],
            }
            if effective_rf is not None:
                kwargs["response_format"] = effective_rf
            if self.seed is not None and not self._seed_unsupported:
                kwargs["seed"] = self.seed
            # Per-call timeout override (also set at client level). Explicit
            # here so a slow provider under load gets the full budget instead
            # of any lower SDK default leaking through.
            kwargs["timeout"] = self.request_timeout_seconds
            try:
                response = self._get_client().chat.completions.create(**kwargs)
            except Exception as exc:
                # Some OpenAI-compatible providers reject unknown params like
                # `seed`. Detect that once, disable it, and retry without it so
                # the run still proceeds (just without reproducibility).
                if "seed" in kwargs and _looks_like_bad_param(exc, "seed"):
                    self._seed_unsupported = True
                    kwargs.pop("seed", None)
                    logger.warning("LLM provider rejected `seed`; continuing without it")
                    response = self._get_client().chat.completions.create(**kwargs)
                # Some providers (e.g. DeepSeek) reject the `json_schema`
                # response format but accept `json_object`. Detect that once,
                # downgrade to `json_object` with the schema folded into the
                # system prompt, and retry so structured calls still succeed.
                elif (
                    not self._json_schema_unsupported
                    and _is_json_schema_format(response_format)
                    and _looks_like_json_schema_rejected(exc)
                ):
                    self._json_schema_unsupported = True
                    logger.warning(
                        "LLM provider rejected `json_schema` response_format; "
                        "falling back to `json_object` + in-prompt schema"
                    )
                    effective_system, effective_rf = _resolve_response_format(
                        system_prompt, response_format, downgraded=True
                    )
                    kwargs["messages"][0]["content"] = effective_system
                    kwargs["response_format"] = effective_rf
                    response = self._get_client().chat.completions.create(**kwargs)
                else:
                    raise
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("LLM returned empty content")
            if validator is not None:
                try:
                    validator(content)
                except Exception as exc:
                    last_exc = exc
                    raise
            return content

        return _invoke_with_retries(
            _call,
            retries=self.max_retries,
            retry_interval_seconds=self.retry_interval_seconds,
            caller_tag=caller_tag,
        )

    def invoke_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        caller_tag: str = "unknown",
        schema_name: str = "response",
        schema: dict[str, Any] | None = None,
        domain_validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Call the LLM with optional JSON-schema enforcement.

        `domain_validator` runs AFTER parsing and may raise to force a retry —
        useful when the schema accepts a string field but the application
        rejects values that don't match a runtime whitelist (e.g. unknown
        optimizer names, made-up tickers). The raised exception's text is
        echoed back to the LLM on the next attempt via `_with_validation_feedback`.
        """
        response_format = build_json_schema_response_format(schema_name, schema) if schema else None

        def _validate(content: str) -> None:
            parsed = parse_json_dict(content)
            if domain_validator is not None:
                domain_validator(parsed)

        raw = self.invoke(
            system_prompt,
            user_prompt,
            caller_tag=caller_tag,
            response_format=response_format,
            validator=_validate,
        )
        return parse_json_dict(raw)


# alias used by existing code
DarwinTradeLLMClient = LLMClient

__all__ = ["LLMClient", "DarwinTradeLLMClient", "build_json_schema_response_format", "parse_json_dict", "normalize_json_text"]
