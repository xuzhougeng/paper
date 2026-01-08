"""Doc2X v2 API client for PDF parsing."""

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import httpx

if TYPE_CHECKING:
    from papercli.config import Settings


# Doc2X error codes with actionable messages
ERROR_CODE_MESSAGES = {
    "parse_task_limit_exceeded": "Task limit reached. Wait for previous tasks to complete.",
    "parse_concurrency_limit": "Page concurrency limit reached. Wait for previous tasks to complete.",
    "parse_quota_limit": "Insufficient parsing quota. Check your Doc2X account.",
    "parse_error": "Parse error occurred. Retry shortly or contact Doc2X support.",
    "parse_create_task_error": "Failed to create task. Retry shortly.",
    "parse_status_not_found": "Status not found or expired. The UID may be invalid.",
    "parse_file_too_large": "File too large. Max allowed: 300MB. Consider splitting the PDF.",
    "parse_page_limit_exceeded": "Too many pages. Max allowed: 2000 pages. Consider splitting the PDF.",
    "parse_file_lock": "File locked for 24h due to repeated failures. Re-export the PDF and retry.",
    "parse_file_not_pdf": "File is not a valid PDF.",
    "parse_file_invalid": "Invalid or malformed PDF file.",
    "parse_timeout": "Processing exceeded 15 minutes. Consider splitting the PDF.",
}


class Doc2XError(Exception):
    """Doc2X API error with diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        uid: str | None = None,
        status: str | None = None,
        error_code: str | None = None,
        detail: str | None = None,
    ):
        super().__init__(message)
        self.uid = uid
        self.status = status
        self.error_code = error_code
        self.detail = detail

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.uid:
            parts.append(f"UID: {self.uid}")
        if self.status:
            parts.append(f"Status: {self.status}")
        if self.error_code:
            parts.append(f"Error code: {self.error_code}")
            if hint := ERROR_CODE_MESSAGES.get(self.error_code):
                parts.append(f"Hint: {hint}")
        if self.detail:
            parts.append(f"Detail: {self.detail}")
        return "\n".join(parts)


class Doc2XClient:
    """Async client for Doc2X v2 PDF parsing API."""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        return self.settings.doc2x.base_url.rstrip("/")

    @property
    def api_key(self) -> str:
        return self.settings.get_doc2x_api_key()

    @property
    def timeout(self) -> float:
        return self.settings.doc2x.timeout

    @property
    def poll_interval(self) -> float:
        return self.settings.doc2x.poll_interval

    @property
    def max_wait(self) -> float:
        return self.settings.doc2x.max_wait

    def _get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def preupload(self) -> tuple[str, str]:
        """
        Request a pre-signed upload URL from Doc2X.

        Returns:
            Tuple of (uid, upload_url)

        Raises:
            Doc2XError: If the request fails
        """
        client = await self._get_client()
        url = f"{self.base_url}/api/v2/parse/preupload"

        try:
            response = await client.post(url, headers=self._get_headers())
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise Doc2XError(
                    "Rate limited. Too many concurrent requests.",
                    error_code="rate_limit",
                ) from e
            raise Doc2XError(f"Preupload request failed: {e}") from e
        except httpx.RequestError as e:
            raise Doc2XError(f"Network error during preupload: {e}") from e

        data = response.json()
        if data.get("code") != "success":
            raise Doc2XError(
                f"Preupload failed: {data}",
                error_code=data.get("code"),
            )

        result = data.get("data", {})
        uid = result.get("uid")
        upload_url = result.get("url")

        if not uid or not upload_url:
            raise Doc2XError(f"Invalid preupload response: missing uid or url. Data: {data}")

        return uid, upload_url

    async def upload_pdf(self, upload_url: str, pdf_path: Path) -> None:
        """
        Upload a PDF file to the pre-signed URL.

        Args:
            upload_url: The pre-signed upload URL from preupload()
            pdf_path: Path to the PDF file

        Raises:
            Doc2XError: If upload fails
            FileNotFoundError: If PDF file doesn't exist
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        client = await self._get_client()

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            # PUT the file directly to the OSS URL (no auth header needed)
            response = await client.put(upload_url, content=pdf_bytes)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise Doc2XError(
                f"PDF upload failed with status {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise Doc2XError(f"Network error during PDF upload: {e}") from e

    async def get_parse_status(self, uid: str) -> dict[str, Any]:
        """
        Get the current parse status for a task.

        Args:
            uid: The task UID from preupload()

        Returns:
            Status dict with keys: status, progress, result, detail

        Raises:
            Doc2XError: If the request fails
        """
        client = await self._get_client()
        url = f"{self.base_url}/api/v2/parse/status"

        try:
            response = await client.get(
                url,
                params={"uid": uid},
                headers=self._get_headers(),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise Doc2XError(
                f"Status request failed: {e}",
                uid=uid,
            ) from e
        except httpx.RequestError as e:
            raise Doc2XError(
                f"Network error getting status: {e}",
                uid=uid,
            ) from e

        data = response.json()
        if data.get("code") != "success":
            error_code = data.get("code")
            raise Doc2XError(
                f"Status check failed: {data}",
                uid=uid,
                error_code=error_code,
            )

        return data.get("data", {})

    async def poll_until_complete(
        self,
        uid: str,
        poll_interval: float | None = None,
        max_wait: float | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        """
        Poll the parse status until completion or failure.

        Args:
            uid: The task UID
            poll_interval: Seconds between polls (default from settings)
            max_wait: Maximum wait time in seconds (default from settings)
            on_progress: Optional callback(progress: int) for progress updates

        Returns:
            The final result dict from Doc2X

        Raises:
            Doc2XError: If parsing fails or times out
        """
        poll_interval = poll_interval or self.poll_interval
        max_wait = max_wait or self.max_wait

        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > max_wait:
                raise Doc2XError(
                    f"Polling timed out after {max_wait:.0f}s",
                    uid=uid,
                    status="timeout",
                )

            status_data = await self.get_parse_status(uid)
            status = status_data.get("status")

            if status == "success":
                return status_data.get("result", {})

            if status == "failed":
                detail = status_data.get("detail", "Unknown error")
                # Try to extract error code from detail
                error_code = None
                for code in ERROR_CODE_MESSAGES:
                    if code in str(detail).lower():
                        error_code = code
                        break
                raise Doc2XError(
                    f"Parse failed: {detail}",
                    uid=uid,
                    status=status,
                    error_code=error_code,
                    detail=detail,
                )

            if status == "processing":
                progress = status_data.get("progress", 0)
                if on_progress:
                    on_progress(progress)

            await asyncio.sleep(poll_interval)

    async def parse_pdf(
        self,
        pdf_path: Path,
        poll_interval: float | None = None,
        max_wait: float | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Complete workflow: preupload -> upload -> poll -> return result.

        Args:
            pdf_path: Path to the PDF file
            poll_interval: Seconds between status polls
            max_wait: Maximum wait time in seconds
            on_progress: Optional callback for progress updates

        Returns:
            Tuple of (uid, result_dict)

        Raises:
            Doc2XError: If any step fails
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)

        # Step 1: Get upload URL
        uid, upload_url = await self.preupload()

        # Step 2: Upload the PDF
        await self.upload_pdf(upload_url, pdf_path)

        # Step 3: Poll until complete
        # Note: Doc2X docs say there may be a brief delay after upload
        # before status becomes available, so first poll may show nothing
        result = await self.poll_until_complete(
            uid=uid,
            poll_interval=poll_interval,
            max_wait=max_wait,
            on_progress=on_progress,
        )

        return uid, result

