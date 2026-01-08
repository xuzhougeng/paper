"""JSONL conversion helpers for Doc2X parse results."""

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse, parse_qs

import httpx

# Pattern to match Doc2X CDN image URLs
DOC2X_IMAGE_URL_PATTERN = re.compile(
    r'https://cdn\.noedgeai\.com/[a-f0-9\-]+_\d+\.(jpg|png|jpeg|gif|webp)(?:\?[^"\'>\s]*)?',
    re.IGNORECASE,
)


def extract_page_text(page_obj: dict[str, Any]) -> str:
    """
    Best-effort text extraction from a Doc2X page object.

    Handles various possible structures:
    - page_obj["md"] or page_obj["text"] as direct text
    - page_obj["blocks"] list with block["text"] or block["content"]

    Args:
        page_obj: A single page object from Doc2X result

    Returns:
        Extracted text as a string (may be empty)
    """
    # Direct text fields
    if "md" in page_obj and isinstance(page_obj["md"], str):
        return page_obj["md"]

    if "text" in page_obj and isinstance(page_obj["text"], str):
        return page_obj["text"]

    # Try extracting from blocks
    blocks = page_obj.get("blocks", [])
    if isinstance(blocks, list):
        text_parts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            # Try various text field names
            for key in ("text", "content", "md", "value"):
                if key in block and isinstance(block[key], str):
                    text_parts.append(block[key])
                    break
        if text_parts:
            return "\n".join(text_parts)

    # Fallback: try to serialize any content
    if "content" in page_obj:
        content = page_obj["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content if item)

    return ""


def result_to_page_records(
    result: Any,
    uid: str,
    source_path: str,
    include_raw: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Convert Doc2X parse result to page-level records.

    Handles various result shapes:
    - {"pages": [...]} - dict with pages list
    - [page1, page2, ...] - direct list of pages
    - Single page dict with no pages key

    Args:
        result: The result from Doc2X parse (can be dict or list)
        uid: The Doc2X task UID
        source_path: Original PDF file path
        include_raw: Whether to include raw page data

    Yields:
        Page record dicts suitable for JSONL output
    """
    pages: list[dict[str, Any]] = []

    if isinstance(result, dict):
        # Check for pages array
        if "pages" in result and isinstance(result["pages"], list):
            pages = result["pages"]
        elif "result" in result and isinstance(result["result"], list):
            # Sometimes result is nested
            pages = result["result"]
        else:
            # Treat the whole result as a single "page"
            pages = [result]
    elif isinstance(result, list):
        pages = result
    else:
        # Fallback: wrap in list
        pages = [{"content": str(result)}] if result else []

    for idx, page_obj in enumerate(pages):
        if not isinstance(page_obj, dict):
            page_obj = {"content": str(page_obj)}

        record = {
            "doc2x_uid": uid,
            "source_path": source_path,
            "page_index": idx,
            "page_no": idx + 1,
            "text": extract_page_text(page_obj),
        }

        if include_raw:
            record["raw_page"] = page_obj

        yield record


def result_to_jsonl(
    result: Any,
    uid: str,
    source_path: str,
    include_raw: bool = False,
) -> str:
    """
    Convert Doc2X parse result to JSONL string.

    Each line is a JSON object representing one page.

    Args:
        result: The result from Doc2X parse
        uid: The Doc2X task UID
        source_path: Original PDF file path
        include_raw: Whether to include raw page data

    Returns:
        JSONL string (one JSON object per line)
    """
    lines = []
    for record in result_to_page_records(result, uid, source_path, include_raw):
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


def write_jsonl(
    result: Any,
    uid: str,
    source_path: str,
    output_path: str,
    include_raw: bool = False,
) -> int:
    """
    Write Doc2X parse result to a JSONL file.

    Args:
        result: The result from Doc2X parse
        uid: The Doc2X task UID
        source_path: Original PDF file path
        output_path: Path to output JSONL file
        include_raw: Whether to include raw page data

    Returns:
        Number of page records written
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in result_to_page_records(result, uid, source_path, include_raw):
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


# =============================================================================
# Image Download and URL Replacement
# =============================================================================


def find_image_urls(text: str) -> list[str]:
    """
    Find all Doc2X CDN image URLs in the given text.

    Args:
        text: Text content that may contain image URLs

    Returns:
        List of unique image URLs found
    """
    urls = DOC2X_IMAGE_URL_PATTERN.findall(text)
    # findall returns tuples when there are groups, we need the full match
    full_matches = DOC2X_IMAGE_URL_PATTERN.finditer(text)
    return list(dict.fromkeys(m.group(0) for m in full_matches))  # Preserve order, remove dupes


def generate_image_filename(url: str, uid: str) -> str:
    """
    Generate a local filename for an image URL.

    The filename is based on the URL structure:
    - {uid}_{page_idx}.{ext} -> {uid}_p{page_idx}.{ext}

    Args:
        url: The image URL
        uid: The Doc2X task UID

    Returns:
        Local filename (without directory)
    """
    parsed = urlparse(url)
    path = parsed.path  # e.g., /019b9d84-fe68-773d-9652-7f7dbdb9cc5d_0.jpg

    # Extract filename from path
    filename = path.split("/")[-1]

    # Try to parse the Doc2X naming convention: {uid}_{page_idx}.{ext}
    match = re.match(r"([a-f0-9\-]+)_(\d+)\.(\w+)$", filename, re.IGNORECASE)
    if match:
        file_uid, page_idx, ext = match.groups()
        # Use a cleaner naming: {uid}_p{page_idx}.{ext}
        return f"{file_uid}_p{page_idx}.{ext}"

    # Fallback: use hash of URL for unique filename
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    ext = filename.split(".")[-1] if "." in filename else "jpg"
    return f"{uid}_{url_hash}.{ext}"


async def download_image(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    timeout: float = 30.0,
) -> bool:
    """
    Download a single image from URL to local path.

    Args:
        client: httpx async client
        url: Image URL to download
        output_path: Local path to save the image
        timeout: Download timeout in seconds

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()

        # Write the image content
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return True
    except Exception:
        return False


async def download_images(
    urls: list[str],
    image_dir: Path,
    uid: str,
    max_concurrent: int = 5,
    timeout: float = 30.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """
    Download multiple images concurrently.

    Args:
        urls: List of image URLs to download
        image_dir: Directory to save images
        uid: Doc2X task UID (for generating filenames)
        max_concurrent: Maximum concurrent downloads
        timeout: Download timeout per image
        on_progress: Optional callback(downloaded: int, total: int)

    Returns:
        Dict mapping original URL -> local file path (only for successful downloads)
    """
    if not urls:
        return {}

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    url_to_local: dict[str, str] = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    downloaded_count = 0
    total = len(urls)

    async def download_one(url: str) -> tuple[str, str | None]:
        nonlocal downloaded_count
        async with semaphore:
            filename = generate_image_filename(url, uid)
            local_path = image_dir / filename

            # Skip if already exists
            if local_path.exists():
                downloaded_count += 1
                if on_progress:
                    on_progress(downloaded_count, total)
                return url, str(local_path)

            async with httpx.AsyncClient(follow_redirects=True) as client:
                success = await download_image(client, url, local_path, timeout)

            downloaded_count += 1
            if on_progress:
                on_progress(downloaded_count, total)

            if success:
                return url, str(local_path)
            return url, None

    # Download all images concurrently
    tasks = [download_one(url) for url in urls]
    results = await asyncio.gather(*tasks)

    for url, local_path in results:
        if local_path:
            url_to_local[url] = local_path

    return url_to_local


def replace_image_urls(text: str, url_mapping: dict[str, str]) -> str:
    """
    Replace image URLs in text with local file paths.

    Args:
        text: Original text with image URLs
        url_mapping: Dict mapping original URL -> local path

    Returns:
        Text with URLs replaced by local paths
    """
    result = text
    for url, local_path in url_mapping.items():
        result = result.replace(url, local_path)
    return result


def collect_all_image_urls(result: Any) -> list[str]:
    """
    Collect all image URLs from a Doc2X parse result.

    Args:
        result: The result from Doc2X parse

    Returns:
        List of unique image URLs found across all pages
    """
    all_urls: list[str] = []

    pages: list[dict[str, Any]] = []
    if isinstance(result, dict):
        if "pages" in result and isinstance(result["pages"], list):
            pages = result["pages"]
        elif "result" in result and isinstance(result["result"], list):
            pages = result["result"]
        else:
            pages = [result]
    elif isinstance(result, list):
        pages = result

    for page_obj in pages:
        if not isinstance(page_obj, dict):
            continue
        # Check md field
        if "md" in page_obj and isinstance(page_obj["md"], str):
            all_urls.extend(find_image_urls(page_obj["md"]))
        # Check text field
        if "text" in page_obj and isinstance(page_obj["text"], str):
            all_urls.extend(find_image_urls(page_obj["text"]))

    # Remove duplicates while preserving order
    return list(dict.fromkeys(all_urls))


def replace_urls_in_result(result: Any, url_mapping: dict[str, str]) -> Any:
    """
    Replace image URLs in the Doc2X result with local paths.

    Args:
        result: The result from Doc2X parse
        url_mapping: Dict mapping original URL -> local path

    Returns:
        Modified result with URLs replaced
    """
    if not url_mapping:
        return result

    if isinstance(result, dict):
        new_result = {}
        for key, value in result.items():
            if key in ("md", "text") and isinstance(value, str):
                new_result[key] = replace_image_urls(value, url_mapping)
            elif key == "pages" and isinstance(value, list):
                new_result[key] = [replace_urls_in_result(p, url_mapping) for p in value]
            elif key == "result" and isinstance(value, list):
                new_result[key] = [replace_urls_in_result(p, url_mapping) for p in value]
            elif isinstance(value, dict):
                new_result[key] = replace_urls_in_result(value, url_mapping)
            else:
                new_result[key] = value
        return new_result
    elif isinstance(result, list):
        return [replace_urls_in_result(item, url_mapping) for item in result]
    else:
        return result

