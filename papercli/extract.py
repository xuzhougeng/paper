"""JSONL conversion helpers for Doc2X parse results."""

import json
from typing import Any, Iterator


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

