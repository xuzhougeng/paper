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
    """LLM-related error."""

    pass


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

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
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

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._call_api(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""

    async def complete_json(
        self,
        prompt: str,
        response_model: type[T],
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
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
        """
        model = model or self.settings.get_intent_model()

        # Build schema description for the prompt
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        json_instruction = f"\n\nRespond with valid JSON matching this schema:\n```json\n{schema_json}\n```\nReturn ONLY the JSON object, no other text."

        full_system = (system_prompt or "") + json_instruction

        try:
            response_text = await self.complete(
                prompt=prompt,
                model=model,
                system_prompt=full_system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Parse JSON from response
            parsed = self._extract_json(response_text)
            return response_model.model_validate(parsed)

        except (json.JSONDecodeError, ValueError) as e:
            if retry_on_parse_error:
                # Retry with stricter prompt
                stricter_system = (
                    (system_prompt or "")
                    + f"\n\nYou MUST respond with ONLY a valid JSON object. No markdown, no explanation, just JSON.\nSchema:\n{schema_json}"
                )
                response_text = await self.complete(
                    prompt=prompt + "\n\nRemember: Return ONLY valid JSON, nothing else.",
                    model=model,
                    system_prompt=stricter_system,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                parsed = self._extract_json(response_text)
                return response_model.model_validate(parsed)
            raise LLMError(f"Failed to parse LLM response as JSON: {e}")

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_api(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """Call the OpenAI API with retry logic."""
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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

