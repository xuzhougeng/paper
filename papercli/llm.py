"""OpenAI-compatible LLM client with retry and structured output support."""

import json
from typing import TYPE_CHECKING, Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from papercli.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """LLM-related error with diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        raw_responses: list[str] | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message)
        self.model = model
        self.base_url = base_url
        self.raw_responses = raw_responses or []
        self.original_exception = original_exception

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.base_url:
            parts.append(f"Base URL: {self.base_url}")
        if self.original_exception:
            parts.append(f"Original error: {type(self.original_exception).__name__}: {self.original_exception}")
        if self.raw_responses:
            parts.append("Raw LLM response(s):")
            for i, resp in enumerate(self.raw_responses, 1):
                truncated = resp[:800] + "..." if len(resp) > 800 else resp
                parts.append(f"  [{i}] {truncated}")
        return "\n".join(parts)


class LLMClient:
    """OpenAI-compatible LLM client with support for different models per stage."""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.get_llm_api_key(),
                base_url=self.settings.llm.base_url,
                timeout=self.settings.llm.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client to avoid loop shutdown errors."""
        if self._client is None:
            return
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            result = close_fn()
            if hasattr(result, "__await__"):
                await result
            self._client = None
            return
        aclose_fn = getattr(self._client, "aclose", None)
        if callable(aclose_fn):
            result = aclose_fn()
            if hasattr(result, "__await__"):
                await result
        self._client = None

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2000,
    ) -> str:
        """
        Get a completion from the LLM.

        Args:
            prompt: User prompt
            model: Model to use (defaults to intent_model)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            The completion text
        """
        model = model or self.settings.get_intent_model()

        response_text = await self._call_api_text(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response_text

    async def complete_json(
        self,
        prompt: str,
        response_model: type[T],
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2000,
        retry_on_parse_error: bool = True,
    ) -> T:
        """
        Get a structured JSON response from the LLM.

        Args:
            prompt: User prompt
            response_model: Pydantic model for response validation
            model: Model to use
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            retry_on_parse_error: Whether to retry with stricter prompt on parse failure

        Returns:
            Parsed response as the specified Pydantic model

        Raises:
            LLMError: If the LLM response cannot be parsed as valid JSON or
                      doesn't match the schema, with diagnostic context.
        """
        model = model or self.settings.get_intent_model()
        base_url = self.settings.llm.base_url

        # Build a compact "shape" instruction rather than embedding full JSON Schema.
        # Some models/proxies may echo the schema itself, causing huge/incomplete outputs.
        shape_instruction = self._build_json_shape_instruction(response_model)
        json_instruction = (
            "\n\nReturn ONLY a JSON object.\n"
            "- No markdown, no code fences, no commentary.\n"
            "- Do NOT return a JSON Schema (no keys like: description, title, type, properties, items, anyOf, allOf, oneOf, required, $defs, $ref).\n"
            + shape_instruction
        )

        full_system = (system_prompt or "") + json_instruction

        # Track raw responses for diagnostics
        raw_responses: list[str] = []
        first_error: Exception | None = None

        try:
            response_text = await self.complete(
                prompt=prompt,
                model=model,
                system_prompt=full_system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_responses.append(response_text)

            # Parse JSON from response
            parsed = self._extract_json(response_text)
            parsed = self._normalize_parsed_json(parsed)
            return response_model.model_validate(parsed)

        except (json.JSONDecodeError, ValueError) as e:
            first_error = e
            if not retry_on_parse_error:
                raise LLMError(
                    f"Failed to parse LLM response as JSON: {e}",
                    model=model,
                    base_url=base_url,
                    raw_responses=raw_responses,
                    original_exception=e,
                )

            # Retry with stricter prompt
            try:
                stricter_system = (
                    (system_prompt or "")
                    + "\n\nYour previous response was invalid (not a JSON instance object matching the required keys). "
                    + "You MUST respond with ONLY a single JSON object (no markdown, no code fences, no prose). "
                    + "Do NOT return schema metadata (description/title/type/properties/etc)."
                    + shape_instruction
                )
                response_text = await self.complete(
                    prompt=prompt
                    + "\n\nReminder: return ONLY a JSON object with actual values for the fields.",
                    model=model,
                    system_prompt=stricter_system,
                    temperature=1.0,
                    max_tokens=max_tokens,
                )
                raw_responses.append(response_text)

                parsed = self._extract_json(response_text)
                parsed = self._normalize_parsed_json(parsed)
                return response_model.model_validate(parsed)

            except (json.JSONDecodeError, ValueError) as retry_error:
                # Both attempts failed - raise with full context
                raise LLMError(
                    f"Failed to parse LLM response as JSON after retry. "
                    f"First error: {first_error}; Retry error: {retry_error}",
                    model=model,
                    base_url=base_url,
                    raw_responses=raw_responses,
                    original_exception=retry_error,
                )
        except Exception as e:
            # API errors or other exceptions
            raise LLMError(
                f"LLM API call failed: {e}",
                model=model,
                base_url=base_url,
                raw_responses=raw_responses,
                original_exception=e,
            )

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_api_text(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Call the LLM and return plain text.

        Strategy:
        - For models starting with "gpt", prefer the Responses API when available.
        - If the server doesn't support Responses (common for some proxies), fall back to Chat Completions.
        - For Chat Completions, try max_completion_tokens first for gpt* models, then fall back to max_tokens.
        """
        base_url = self.settings.llm.base_url

        model_is_gpt = model.lower().startswith("gpt")

        # Prefer Responses API for gpt* models when the SDK supports it.
        if model_is_gpt and hasattr(self.client, "responses"):
            try:
                resp = await self.client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    instructions=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                text = self._extract_text_from_responses(resp)
                return text
            except Exception as e:
                # If Responses isn't supported by the server/proxy, fall back to chat.completions.
                msg = str(e).lower()
                # We treat these as "endpoint not available / provider doesn't implement Responses".
                if (
                    "404" not in msg
                    and "not found" not in msg
                    and "unknown url" not in msg
                    and "unsupported" not in msg
                    and "not supported" not in msg
                ):
                    # For non-endpoint errors, re-raise so tenacity can retry.
                    raise

        # Chat Completions (proxy-friendly)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Many gpt* models prefer max_completion_tokens; many proxies only accept max_tokens.
        use_max_completion_tokens = model_is_gpt
        try:
            if use_max_completion_tokens:
                resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
            else:
                resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return resp.choices[0].message.content or ""
        except Exception as e:
            # Parameter compatibility fallback: if the server rejects the chosen token param, retry once.
            msg = str(e).lower()
            if use_max_completion_tokens and ("max_completion_tokens" in msg or "unsupported" in msg):
                resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            if (not use_max_completion_tokens) and ("max_tokens" in msg or "unsupported" in msg):
                resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            # If the model likely requires /v1/responses but we're on a proxy, raise a clearer error.
            if ("not supported" in msg or "unsupported" in msg) and "chat/completions" in msg:
                raise LLMError(
                    f"LLM API call failed (model may require /v1/responses; base_url={base_url}). {e}",
                    model=model,
                    base_url=base_url,
                    original_exception=e,
                )
            raise

    def _extract_text_from_responses(self, resp: Any) -> str:
        """
        Extract text from a Responses API response in a tolerant way.

        SDKs expose a convenience property `output_text`, but we keep fallbacks for robustness.
        """
        try:
            text = getattr(resp, "output_text", None)
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass

        # Fallback: walk the output structure
        try:
            output = getattr(resp, "output", None)
            if isinstance(output, list):
                parts: list[str] = []
                for item in output:
                    content = getattr(item, "content", None)
                    if isinstance(content, list):
                        for c in content:
                            # Prefer `text` / `value` fields if present
                            for key in ("text", "value", "content"):
                                v = getattr(c, key, None)
                                if isinstance(v, str) and v:
                                    parts.append(v)
                                    break
                joined = "\n".join(p for p in parts if p)
                if joined.strip():
                    return joined
        except Exception:
            pass

        return ""

    def _looks_like_schema_dict(self, d: dict) -> bool:
        """Heuristic to detect JSON Schema-ish dicts returned by some models/proxies."""
        schema_keys = {
            "type",
            "title",
            "description",
            "properties",
            "items",
            "required",
            "anyOf",
            "allOf",
            "oneOf",
            "$defs",
            "$ref",
        }
        return any(k in d for k in schema_keys)

    def _build_json_shape_instruction(self, response_model: type[T]) -> str:
        """
        Build a compact instruction that describes the expected JSON instance shape.

        We avoid embedding the full JSON Schema because some models/proxies echo it back.
        """
        schema = response_model.model_json_schema()
        required = schema.get("required", []) or []
        props: dict[str, Any] = schema.get("properties", {}) or {}

        required_str = ", ".join(required) if required else "(none)"
        optional = [k for k in props.keys() if k not in set(required)]
        optional_str = ", ".join(optional) if optional else "(none)"

        example: dict[str, Any] = {}
        # Put required fields first for readability
        for key in required + [k for k in props.keys() if k not in required]:
            info = props.get(key, {})
            example[key] = self._example_value_for_schema(info, is_required=key in required)

        example_json = json.dumps(example, ensure_ascii=False, indent=2)
        return (
            f"\nThe JSON object MUST include required keys: {required_str}\n"
            f"Optional keys: {optional_str}\n"
            "Example (shape only; replace values with real content):\n"
            f"{example_json}\n"
        )

    def _example_value_for_schema(self, info: dict[str, Any], *, is_required: bool) -> Any:
        """Create a small example value for a field based on its JSON schema snippet."""
        # anyOf often used for Optional[T] (e.g. string|null)
        if "anyOf" in info and isinstance(info["anyOf"], list):
            # Prefer a non-null type for required, null for optional
            if not is_required:
                return None
            non_null = next((x for x in info["anyOf"] if x.get("type") != "null"), {})
            return self._example_value_for_schema(non_null, is_required=True)

        t = info.get("type")
        if t == "string":
            return "..."
        if t == "integer":
            return 0
        if t == "number":
            return 0.0
        if t == "boolean":
            return False
        if t == "array":
            return []
        if t == "object":
            return {}

        # Fallback: keep it simple
        return None

    def _normalize_schema_echo_value(self, v: Any) -> tuple[bool, Any]:
        """
        Normalize values that look like schema-echo structures.

        Returns:
            (include, normalized_value)
            - include=False means "skip this key" (likely schema metadata, not data)
        """
        if isinstance(v, dict):
            # Common schema-echo pattern: {"type": "...", "value": ...}
            if "value" in v:
                return True, v["value"]
            # If it still looks like schema, skip to avoid validation errors.
            if self._looks_like_schema_dict(v):
                return False, None
        return True, v

    def _normalize_parsed_json(self, parsed: Any) -> Any:
        """
        Normalize parsed JSON before Pydantic validation.

        Some LLMs/proxies return a schema-like wrapper:
          {"description": "...", "properties": {"field": {"value": ...}}}
        or:
          {"description": "...", "properties": {"field": ...}}

        This function unwraps such responses into a plain object:
          {"field": ...}
        """
        if not isinstance(parsed, dict):
            return parsed

        # Unwrap top-level schema echo wrapper
        if (
            "properties" in parsed
            and isinstance(parsed.get("properties"), dict)
            and self._looks_like_schema_dict(parsed)
        ):
            out: dict[str, Any] = {}
            for k, v in parsed["properties"].items():
                include, normalized = self._normalize_schema_echo_value(v)
                if include:
                    out[k] = normalized
            return out

        # Unwrap per-field {"value": ...} shapes, and drop schema-ish nested dicts.
        out: dict[str, Any] = {}
        changed = False
        for k, v in parsed.items():
            include, normalized = self._normalize_schema_echo_value(v)
            if not include:
                changed = True
                continue
            if normalized is not v:
                changed = True
            out[k] = normalized
        return out if changed else parsed

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        # Try to extract from markdown code block
        if "```" in text:
            # Find JSON block
            import re

            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found", text, 0)

    async def intent_completion(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Get a structured completion using the intent model."""
        return await self.complete_json(
            prompt=prompt,
            response_model=response_model,
            model=self.settings.get_intent_model(),
            system_prompt=system_prompt,
        )

    async def eval_completion(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Get a structured completion using the eval model."""
        return await self.complete_json(
            prompt=prompt,
            response_model=response_model,
            model=self.settings.get_eval_model(),
            system_prompt=system_prompt,
        )

