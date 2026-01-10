"""Gemini API client for text generation and image generation."""

import base64
import json
from typing import TYPE_CHECKING, Any, Literal

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from papercli.config import Settings


class GeminiError(Exception):
    """Gemini API error with diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        status_code: int | None = None,
        raw_response: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message)
        self.model = model
        self.base_url = base_url
        self.status_code = status_code
        self.raw_response = raw_response
        self.original_exception = original_exception

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.base_url:
            parts.append(f"Base URL: {self.base_url}")
        if self.status_code:
            parts.append(f"Status code: {self.status_code}")
        if self.original_exception:
            parts.append(
                f"Original error: {type(self.original_exception).__name__}: {self.original_exception}"
            )
        if self.raw_response:
            truncated = self.raw_response[:800] + "..." if len(self.raw_response) > 800 else self.raw_response
            parts.append(f"Raw response: {truncated}")
        return "\n".join(parts)


class GeminiClient:
    """Async client for Gemini API (text and image generation)."""

    # Fallback sequences for aspect ratio and image size
    ASPECT_RATIO_FALLBACKS = ["16:9", "4:3", "1:1"]
    IMAGE_SIZE_FALLBACKS = ["1K", "2K"]  # 4K may not be supported on all models

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialize the httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.gemini.timeout),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_url(self, model: str) -> str:
        """Build the generateContent URL for a model."""
        base_url = self.settings.gemini.base_url.rstrip("/")
        return f"{base_url}/models/{model}:generateContent"

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Gemini API request."""
        api_key = self.settings.get_gemini_api_key()
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate text using Gemini text model.

        Args:
            prompt: User prompt
            model: Model to use (defaults to text_model from config)
            system_instruction: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated text

        Raises:
            GeminiError: If the API call fails
        """
        model = model or self.settings.gemini.text_model
        base_url = self.settings.gemini.base_url
        url = self._build_url(model)

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        try:
            response = await self.client.post(url, headers=self._get_headers(), json=payload)
            response.raise_for_status()
            data = response.json()
            return self._extract_text_from_response(data)
        except httpx.HTTPStatusError as e:
            raise GeminiError(
                f"Gemini API request failed: {e.response.status_code}",
                model=model,
                base_url=base_url,
                status_code=e.response.status_code,
                raw_response=e.response.text,
                original_exception=e,
            )
        except Exception as e:
            raise GeminiError(
                f"Gemini API call failed: {e}",
                model=model,
                base_url=base_url,
                original_exception=e,
            )

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        aspect_ratio: str = "16:9",
        image_size: str = "1K",
        temperature: float = 1.0,
    ) -> tuple[bytes, str]:
        """
        Generate an image using Gemini image model.

        Args:
            prompt: Image generation prompt
            model: Model to use (defaults to image_model from config)
            aspect_ratio: Aspect ratio (16:9, 4:3, 1:1)
            image_size: Image size (1K, 2K, 4K)
            temperature: Sampling temperature

        Returns:
            Tuple of (image_bytes, mime_type)

        Raises:
            GeminiError: If the API call fails after all fallbacks
        """
        model = model or self.settings.gemini.image_model

        # Try with requested aspect ratio first, then fallbacks
        aspect_ratios_to_try = [aspect_ratio] + [
            ar for ar in self.ASPECT_RATIO_FALLBACKS if ar != aspect_ratio
        ]
        image_sizes_to_try = [image_size] + [
            sz for sz in self.IMAGE_SIZE_FALLBACKS if sz != image_size
        ]

        last_error: Exception | None = None

        for ar in aspect_ratios_to_try:
            for sz in image_sizes_to_try:
                try:
                    return await self._generate_image_with_config(
                        prompt=prompt,
                        model=model,
                        aspect_ratio=ar,
                        image_size=sz,
                        temperature=temperature,
                    )
                except GeminiError as e:
                    last_error = e
                    # If it's a parameter-related error, try next config
                    if e.status_code in (400, 422):
                        continue
                    # For other errors (auth, rate limit, etc.), raise immediately
                    raise

        # All fallbacks exhausted
        raise GeminiError(
            f"Image generation failed after trying all fallback configurations. "
            f"Last error: {last_error}",
            model=model,
            base_url=self.settings.gemini.base_url,
            original_exception=last_error,
        )

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _generate_image_with_config(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        image_size: str,
        temperature: float,
    ) -> tuple[bytes, str]:
        """Internal method to generate image with specific configuration."""
        base_url = self.settings.gemini.base_url
        url = self._build_url(model)

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                },
            },
        }

        try:
            response = await self.client.post(url, headers=self._get_headers(), json=payload)
            response.raise_for_status()
            data = response.json()
            return self._extract_image_from_response(data, model, base_url)
        except httpx.HTTPStatusError as e:
            raise GeminiError(
                f"Gemini API request failed: {e.response.status_code}",
                model=model,
                base_url=base_url,
                status_code=e.response.status_code,
                raw_response=e.response.text,
                original_exception=e,
            )
        except Exception as e:
            if isinstance(e, GeminiError):
                raise
            raise GeminiError(
                f"Gemini API call failed: {e}",
                model=model,
                base_url=base_url,
                original_exception=e,
            )

    def _extract_text_from_response(self, data: dict[str, Any]) -> str:
        """Extract text content from Gemini response."""
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            text_parts = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])

            if not text_parts:
                raise ValueError("No text content in response")

            return "\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to extract text from response: {e}") from e

    def _extract_image_from_response(
        self, data: dict[str, Any], model: str, base_url: str
    ) -> tuple[bytes, str]:
        """
        Extract image data from Gemini response.

        Handles various field name conventions:
        - inlineData / inline_data
        - mimeType / mime_type
        - data (base64 encoded)
        """
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise GeminiError(
                    "No candidates in response",
                    model=model,
                    base_url=base_url,
                    raw_response=json.dumps(data)[:500],
                )

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                # Try different field name conventions
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data:
                    # Extract mime type
                    mime_type = (
                        inline_data.get("mimeType")
                        or inline_data.get("mime_type")
                        or "image/png"
                    )

                    # Extract base64 data
                    b64_data = inline_data.get("data")
                    if b64_data:
                        image_bytes = base64.b64decode(b64_data)
                        return image_bytes, mime_type

            raise GeminiError(
                "No image data found in response",
                model=model,
                base_url=base_url,
                raw_response=json.dumps(data)[:500],
            )

        except GeminiError:
            raise
        except Exception as e:
            raise GeminiError(
                f"Failed to extract image from response: {e}",
                model=model,
                base_url=base_url,
                raw_response=json.dumps(data)[:500] if data else None,
                original_exception=e,
            )

    async def generate_text_json(
        self,
        prompt: str,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Generate text and parse as JSON.

        Args:
            prompt: User prompt (should request JSON output)
            model: Model to use
            system_instruction: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON dict

        Raises:
            GeminiError: If API call fails or response is not valid JSON
        """
        text = await self.generate_text(
            prompt=prompt,
            model=model,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Try to extract JSON from response
        text = text.strip()

        # Handle markdown code blocks
        if "```" in text:
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

        raise GeminiError(
            f"Failed to parse Gemini response as JSON",
            model=model or self.settings.gemini.text_model,
            base_url=self.settings.gemini.base_url,
            raw_response=text[:500],
        )

