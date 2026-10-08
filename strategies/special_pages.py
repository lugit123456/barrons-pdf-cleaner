from __future__ import annotations

import re
from typing import Any


PAGE_RULE_VERSION = "publication_special_pages_v2"
PAGE_TYPE_NORMAL = "normal"
PAGE_TYPE_CONTENTS = "contents"
PAGE_TYPE_POLITICS = "world_this_week_politics"
PAGE_TYPE_BUSINESS = "world_this_week_business"
PAGE_TYPE_UTILITY = "utility"

SPECIAL_PAGE_TYPES = {
    PAGE_TYPE_CONTENTS,
    PAGE_TYPE_POLITICS,
    PAGE_TYPE_BUSINESS,
    PAGE_TYPE_UTILITY,
}


def classify_native_page(
    blocks: list[dict[str, Any]],
    page_height: float,
    publication_type: str = "",
) -> str:
    """Classify publication-specific utility pages from coordinate-backed headings."""
    publication = str(publication_type or "").strip().upper()
    candidates = []
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            continue
        y0 = float(bbox[1])
        top_limit = min(180.0, page_height * 0.24) if publication == "BARRONS" else min(100.0, page_height * 0.16)
        if y0 > top_limit:
            continue
        min_font_size = 8 if publication == "BARRONS" else 13
        if float(block.get("max_font_size") or 0) < min_font_size:
            continue
        candidates.append(str(block.get("text") or ""))
    return classify_page_heading("\n".join(candidates), publication)


def classify_ocr_page(page_text: str, publication_type: str = "") -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    return classify_page_heading("\n".join(lines[:16]), publication_type)


def classify_page_heading(text: str, publication_type: str = "") -> str:
    normalized_lines = [
        _normalize_heading(line)
        for line in text.splitlines()
        if _normalize_heading(line)
    ]
    joined = " ".join(normalized_lines[:6])
    if any(line == "contents" for line in normalized_lines[:6]):
        return PAGE_TYPE_CONTENTS
    if str(publication_type or "").strip().upper() == "BARRONS":
        if any(line == "index" for line in normalized_lines[:8]):
            return PAGE_TYPE_UTILITY
        if any(line == "data" or line.startswith("data barrons com data") for line in normalized_lines[:8]):
            return PAGE_TYPE_UTILITY
        if any(line == "winners losers" for line in normalized_lines[:8]):
            return PAGE_TYPE_UTILITY
    if "the world this week politics" in joined:
        return PAGE_TYPE_POLITICS
    if "the world this week business" in joined:
        return PAGE_TYPE_BUSINESS
    return PAGE_TYPE_NORMAL


def build_contents_page_result(page_number: int, parser: str) -> dict[str, Any]:
    return {
        "page": page_number,
        "page_type": PAGE_TYPE_CONTENTS,
        "page_rule_version": PAGE_RULE_VERSION,
        "parser": parser,
        "skipped": True,
        "skip_reason": "Contents page",
        "articles": [],
    }


def build_utility_page_result(page_number: int, parser: str) -> dict[str, Any]:
    return {
        "page": page_number,
        "page_type": PAGE_TYPE_UTILITY,
        "page_rule_version": PAGE_RULE_VERSION,
        "parser": parser,
        "skipped": True,
        "skip_reason": "Publication utility/data page",
        "articles": [],
    }


def build_native_world_page_result(
    *,
    page_number: int,
    page_type: str,
    parser: str,
    blocks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    source_stats: dict[str, Any],
) -> dict[str, Any]:
    body_blocks = [
        block
        for block in blocks
        if _keep_world_page_block(block, page_type, page_height)
    ]
    body_blocks.sort(
        key=lambda block: (
            _column_index(block, page_width),
            float((block.get("bbox") or [0, 0])[1]),
            float((block.get("bbox") or [0])[0]),
        )
    )
    content = "\n\n".join(
        _flatten_block_lines(str(block.get("text") or ""))
        for block in body_blocks
        if str(block.get("text") or "").strip()
    ).strip()
    title, category = world_page_labels(page_type)
    article = {
        "title": title,
        "category": category,
        "content_markdown": content,
        "content_raw": content,
        "images": [
            str(image.get("rel_path"))
            for image in images
            if isinstance(image, dict) and image.get("rel_path")
        ],
        "source_block_ids": [str(block.get("id")) for block in body_blocks],
        "page": page_number,
        "special_page_type": page_type,
        "bypass_min_article_words": True,
    }
    return {
        "page": page_number,
        "page_type": page_type,
        "page_rule_version": PAGE_RULE_VERSION,
        "parser": parser,
        "source_stats": source_stats,
        "articles": [article] if content else [],
    }


def build_ocr_world_page_result(
    *,
    page_number: int,
    page_type: str,
    page_text: str,
    image_paths: list[str],
) -> dict[str, Any]:
    title, category = world_page_labels(page_type)
    content = _strip_ocr_page_chrome(page_text, page_type)
    article = {
        "title": title,
        "category": category,
        "content_markdown": content,
        "content_raw": page_text.strip(),
        "images": list(dict.fromkeys(image_paths)),
        "page": page_number,
        "special_page_type": page_type,
        "bypass_min_article_words": True,
    }
    return {
        "page": page_number,
        "page_type": page_type,
        "page_rule_version": PAGE_RULE_VERSION,
        "articles": [article] if content else [],
    }


def world_page_labels(page_type: str) -> tuple[str, str]:
    if page_type == PAGE_TYPE_POLITICS:
        return "The world this week Politics", "The world this week / Politics"
    if page_type == PAGE_TYPE_BUSINESS:
        return "The world this week Business", "The world this week / Business"
    raise ValueError(f"Unsupported world page type: {page_type}")


def cache_matches_page_type(cache_data: dict[str, Any], expected_type: str) -> bool:
    cached_type = str(cache_data.get("page_type") or PAGE_TYPE_NORMAL)
    if expected_type in SPECIAL_PAGE_TYPES:
        return (
            cached_type == expected_type
            and cache_data.get("page_rule_version") == PAGE_RULE_VERSION
        )
    return cached_type not in SPECIAL_PAGE_TYPES


def _keep_world_page_block(
    block: dict[str, Any],
    page_type: str,
    page_height: float,
) -> bool:
    text = str(block.get("text") or "").strip()
    bbox = block.get("bbox") or [0, 0, 0, 0]
    if not text or len(bbox) < 4:
        return False
    y0, y1 = float(bbox[1]), float(bbox[3])
    if y0 < 72 or y1 > page_height - 12:
        return False
    normalized = _normalize_heading(text)
    title, _ = world_page_labels(page_type)
    if normalized == _normalize_heading(title):
        return False
    if re.fullmatch(r"\d+ the economist july \d+(?:st|nd|rd|th) \d{4}", normalized):
        return False
    return float(block.get("max_font_size") or 0) >= 5


def _column_index(block: dict[str, Any], page_width: float) -> int:
    bbox = block.get("bbox") or [0, 0, 0, 0]
    x0 = max(float(bbox[0]), 0.0)
    column_width = max(page_width / 4.0, 1.0)
    return min(int(x0 / column_width), 3)


def _strip_ocr_page_chrome(page_text: str, page_type: str) -> str:
    title, _ = world_page_labels(page_type)
    title_key = _normalize_heading(title)
    retained = []
    for line in page_text.splitlines():
        stripped = line.strip()
        normalized = _normalize_heading(stripped)
        if not stripped or normalized == title_key:
            continue
        if re.fullmatch(r"\d+ the economist july \d+(?:st|nd|rd|th) \d{4}", normalized):
            continue
        retained.append(stripped)
    return "\n".join(retained).strip()


def _flatten_block_lines(text: str) -> str:
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _normalize_heading(value: str) -> str:
    value = value.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
