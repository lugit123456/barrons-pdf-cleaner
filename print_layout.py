from __future__ import annotations

import re
from typing import Any, Literal


PRINT_LAYOUT_VERSION = 5

_WSJ_SECTIONS = (
    "PAGE ONE",
    "U.S. NEWS",
    "WORLD NEWS",
    "OPINION",
    "SPORTS",
    "BUSINESS & FINANCE",
    "TECHNOLOGY",
    "PERSONAL JOURNAL",
    "ARTS IN REVIEW",
    "MARKETS DIGEST",
    "HEARD ON THE STREET",
    "EXCHANGE",
)

_FT_SECTIONS = (
    "NATIONAL",
    "INTERNATIONAL",
    "COMPANIES & MARKETS",
    "UK COMPANIES",
    "MARKETS",
    "OPINION",
    "LIFE & ARTS",
    "HOUSE & HOME",
    "MONEY",
    "FT WEEKEND",
)

_TE_SECTIONS = (
    "THE WORLD THIS WEEK",
    "LEADERS",
    "LETTERS",
    "BRIEFING",
    "UNITED STATES",
    "THE AMERICAS",
    "ASIA",
    "CHINA",
    "MIDDLE EAST & AFRICA",
    "EUROPE",
    "BRITAIN",
    "INTERNATIONAL",
    "BUSINESS",
    "FINANCE & ECONOMICS",
    "SCIENCE & TECHNOLOGY",
    "CULTURE",
    "OBITUARY",
)

_BARRONS_SECTIONS = (
    "UP & DOWN WALL STREET",
    "STREETWISE",
    "REVIEW & PREVIEW",
    "INCOME INVESTING",
    "FUNDS QUARTERLY",
    "THE ECONOMY",
    "TECH TRADER",
    "MARKET WEEK",
    "INTERNATIONAL TRADER",
    "THE STRIKING PRICE",
    "INSIDE SCOOP",
    "WINNERS & LOSERS",
    "MARKET VIEW",
    "OTHER VOICES",
    "RETIREMENT MAILBAG",
)

_WHATS_NEWS_HEADING = re.compile(r"(?i)what['\u2019]?s\s+news")
_WHATS_NEWS_GROUP = re.compile(
    r"(?i)^(Business\s*&\s*Finance|Worldwide)$"
)
_WHATS_NEWS_BOUNDARY = re.compile(
    r"(?i)^(Contents|Index|Markets|Opinion|Weather|Corrections?)\b"
)
_FT_BRIEFING_HEADING = re.compile(r"(?i)^Briefing$")
_FT_BRIEFING_BOUNDARY = re.compile(
    r"(?i)^(World Markets|Subscribe|For the latest news|Contents|Index)\b"
)

_FT_WEEKEND_MASTHEAD = re.compile(r"(?i)\bFT\s+Weekend\b")
_WSJ_WEEKEND_MASTHEAD = re.compile(
    r"(?i)\b(?:The\s+)?Wall\s+Street\s+Journal\s+Weekend\b"
)
_WEEKEND_DATE_LINE = re.compile(
    r"(?i)\bSaturday\b(?:\s*/\s*Sunday\b|[\s\S]{0,80}\bSunday\b)"
)


def detect_weekend_issue(
    publication_type: str,
    header_text: str = "",
    header_blocks: list[dict[str, Any]] | None = None,
) -> bool:
    """Identify FT/WSJ weekend editions from trusted first-page header text."""
    publication = str(publication_type or "").strip().upper()
    if publication not in {"FT", "WSJ"}:
        return False
    text_parts = [str(header_text or "")]
    text_parts.extend(str(block.get("text") or "") for block in (header_blocks or []))
    normalized = re.sub(r"\s+", " ", "\n".join(text_parts)).strip()
    if publication == "FT" and _FT_WEEKEND_MASTHEAD.search(normalized):
        return True
    if publication == "WSJ" and _WSJ_WEEKEND_MASTHEAD.search(normalized):
        return True
    return bool(_WEEKEND_DATE_LINE.search(normalized))


def extract_print_layout(
    publication_type: str,
    pdf_page: int,
    header_text: str = "",
    header_blocks: list[dict[str, Any]] | None = None,
    page_width: float | None = None,
) -> dict[str, str | None]:
    """Extract print metadata only from a trusted page-header crop or blocks."""
    publication = str(publication_type or "").strip().upper()
    candidates = _header_candidates(header_text, header_blocks, page_width=page_width)

    if publication == "WSJ":
        label = _extract_wsj_page(candidates)
        section = _find_section(candidates, _WSJ_SECTIONS)
        if label == "A1":
            section = "PAGE ONE"
    elif publication in {"FT", "TE", "BARRONS"}:
        label = _extract_numeric_page(publication, candidates)
        if publication == "FT":
            sections = _FT_SECTIONS
        elif publication == "TE":
            sections = _TE_SECTIONS
        else:
            sections = _BARRONS_SECTIONS
        utility_section = (
            _find_section(candidates, ("CONTENTS", "INDEX", "DATA"))
            if publication == "BARRONS"
            else None
        )
        section = utility_section or _find_section(candidates, sections)
    else:
        label = None
        section = None

    return {
        "print_page_label": label,
        "print_section": section,
        "print_page_source": "header" if label or section else None,
    }


def extract_whats_news(
    module_text: str = "",
    *,
    blocks: list[dict[str, Any]] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Parse a cropped/coordinate-bounded WSJ What's News module.

    Full-page linear text is deliberately unsupported: callers must provide a
    left-column crop or blocks with page coordinates. Items must begin with a
    visible bullet, otherwise the extractor returns no inferred data.
    """
    if blocks:
        module_text = "\n".join(
            str(block.get("text") or "")
            for block in select_whats_news_blocks(
                blocks,
                page_width=page_width,
                page_height=page_height,
            )
        )
    normalized = str(module_text or "").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines()]
    content_start = _find_whats_news_content_start(lines)
    if content_start is None:
        return {"groups": []}

    groups: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None
    pending: list[str] = []
    pending_started_with_bullet = False

    def finish_pending() -> None:
        nonlocal pending, pending_started_with_bullet
        if not current_group or not pending or not pending_started_with_bullet:
            pending = []
            pending_started_with_bullet = False
            return
        match = re.search(r"(?i)^(.*?)(?:[,.;:]?\s+)([A-Z]\d{1,3})[.]?$", " ".join(pending))
        if match:
            _append_whats_news_item(
                current_group["items"],
                [match.group(1)],
                match.group(2),
            )
        pending = []
        pending_started_with_bullet = False

    for line in lines[content_start:]:
        if not line:
            continue
        if _WHATS_NEWS_BOUNDARY.fullmatch(line):
            finish_pending()
            break
        group_name = _whats_news_group_name(line)
        if group_name:
            finish_pending()
            current_group = {"name": group_name, "items": []}
            groups.append(current_group)
            continue
        if current_group is None:
            continue

        bullet = re.match(r"^[\ue013\u2022\u25cf\u25aa\u25e6]\s*(.*)$", line)
        if bullet:
            finish_pending()
            pending = [bullet.group(1).strip()]
            pending_started_with_bullet = True
        elif pending_started_with_bullet:
            pending.append(line)

        if pending_started_with_bullet and re.search(r"(?i)\b[A-Z]\d{1,3}[.]?$", " ".join(pending)):
            finish_pending()

    finish_pending()
    groups = [group for group in groups if group["items"]]
    return {"groups": groups}


def extract_ft_briefing(
    module_text: str = "",
    *,
    blocks: list[dict[str, Any]] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Parse the coordinate-cropped Briefing directory on a regular FT front page."""
    if blocks:
        module_text = "\n".join(
            str(block.get("text") or "")
            for block in select_ft_briefing_blocks(
                blocks,
                page_width=page_width,
                page_height=page_height,
            )
        )
    normalized = str(module_text or "").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines()]
    start = next(
        (index + 1 for index, line in enumerate(lines) if _FT_BRIEFING_HEADING.fullmatch(line)),
        None,
    )
    if start is None:
        return {"groups": []}

    items: list[dict[str, Any]] = []
    pending: list[str] = []

    def finish_pending() -> None:
        nonlocal pending
        if not pending:
            return
        text = re.sub(r"\s+", " ", " ".join(pending)).strip(" -\u2013\u2014")
        page_match = re.search(r"(?i)\bPAGE\s*([0-9]{1,3})\b", text)
        if text:
            items.append(
                {
                    "text": text,
                    "target_print_page_label": page_match.group(1) if page_match else None,
                    "target_article_id": None,
                }
            )
        pending = []

    for line in lines[start:]:
        if not line:
            continue
        if _FT_BRIEFING_BOUNDARY.match(line):
            finish_pending()
            break
        bullet = re.match(
            r"^[^A-Za-z]{0,4}[>\u00bb\u2022\u25b6\u25b8\u25ba]\s*(.*)$",
            line,
        )
        if bullet:
            finish_pending()
            pending = [bullet.group(1).strip()]
        elif pending:
            pending.append(line)
    finish_pending()
    return {"groups": [{"name": "Briefing", "items": items}]} if items else {"groups": []}


def extract_front_page_directory(
    publication_type: str,
    module_text: str = "",
    *,
    blocks: list[dict[str, Any]] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    publication = str(publication_type or "").strip().upper()
    if publication == "WSJ":
        return extract_whats_news(
            module_text,
            blocks=blocks,
            page_width=page_width,
            page_height=page_height,
        )
    if publication == "FT":
        return extract_ft_briefing(
            module_text,
            blocks=blocks,
            page_width=page_width,
            page_height=page_height,
        )
    return {"groups": []}


def _find_whats_news_content_start(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if _WHATS_NEWS_HEADING.fullmatch(line):
            return index + 1
        for span in (2, 3):
            if index + span > len(lines):
                continue
            combined = " ".join(lines[index : index + span])
            if _WHATS_NEWS_HEADING.fullmatch(combined):
                return index + span
    return None


def _whats_news_group_name(line: str) -> str | None:
    match = _WHATS_NEWS_GROUP.fullmatch(line) or re.match(
        r"(?i)^(Business\s*&\s*Finance|Worldwide)"
        r"(?:\s+(?:WSJ|THE\s*WALL\s*STREET\s*JOURNAL).*)?$",
        line,
    )
    if not match:
        return None
    return "Business & Finance" if match.group(1).lower().startswith("business") else "Worldwide"


def _block_contains_whats_news_group(block: dict[str, Any]) -> bool:
    return any(
        _whats_news_group_name(re.sub(r"\s+", " ", line).strip())
        for line in str(block.get("text") or "").splitlines()
    )


def select_header_blocks(
    blocks: list[dict[str, Any]],
    page_height: float,
) -> list[dict[str, Any]]:
    """Return coordinate-backed blocks in the physical page-header band."""
    limit = min(125.0, max(float(page_height) * 0.16, 70.0))
    return sorted(
        (
            block
            for block in blocks
            if _valid_bbox_block(block)
            and float(block["bbox"][1]) <= limit
            and float(block["bbox"][3]) <= limit + 35.0
        ),
        key=_block_sort_key,
    )


def select_whats_news_blocks(
    blocks: list[dict[str, Any]],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[dict[str, Any]]:
    """Return only coordinate-confirmed blocks in the WSJ What's News column."""
    valid = [block for block in blocks if _valid_bbox_block(block)]
    if not valid:
        return []
    width = float(page_width or max(float(block["bbox"][2]) for block in valid))
    height = float(page_height or max(float(block["bbox"][3]) for block in valid))
    headings = [
        block
        for block in valid
        if _WHATS_NEWS_HEADING.search(str(block.get("text") or ""))
        and float(block["bbox"][0]) <= width * 0.36
    ]
    if not headings:
        return []

    heading = min(headings, key=lambda block: float(block["bbox"][1]))
    heading_x1 = float(heading["bbox"][2])
    start_y = float(heading["bbox"][1])
    right_column_starts = [
        float(block["bbox"][0])
        for block in valid
        if float(block["bbox"][0]) > heading_x1 + max(6.0, width * 0.01)
        and float(block["bbox"][3]) >= start_y - 4.0
    ]
    next_column_x = min(right_column_starts, default=heading_x1 + width * 0.04)
    left_limit = min(
        width * 0.28,
        heading_x1 + width * 0.04,
        (heading_x1 + next_column_x) / 2.0,
    )
    bottom_limit = height * 0.98
    selected: list[dict[str, Any]] = []
    for block in valid:
        x0, y0, x1, y1 = (float(value) for value in block["bbox"][:4])
        intersects_module = y1 >= start_y - 4.0 and y0 <= bottom_limit
        inside_left_column = x0 <= left_limit and (
            x1 <= left_limit + width * 0.03
            or block is heading
            or _block_contains_whats_news_group(block)
        )
        if not intersects_module or not inside_left_column:
            continue
        selected.append(block)

    # A tall block can start above the standalone What's News heading while its
    # lower lines contain Business & Finance. Keep it, but serialize the actual
    # heading first so the parser sees both groups in their logical order.
    selected.sort(
        key=lambda block: (
            0 if block is heading else 1,
            max(float(block["bbox"][1]), start_y),
            float(block["bbox"][0]),
        )
    )
    bounded: list[dict[str, Any]] = []
    for block in selected:
        text = str(block.get("text") or "").strip()
        normalized_first_line = re.sub(r"\s+", " ", text.splitlines()[0]).strip()
        if bounded and _WHATS_NEWS_BOUNDARY.match(normalized_first_line):
            break
        bounded.append(block)
    return bounded


def select_ft_briefing_blocks(
    blocks: list[dict[str, Any]],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[dict[str, Any]]:
    """Return coordinate-confirmed blocks in the FT front-page Briefing rail."""
    valid = [block for block in blocks if _valid_bbox_block(block)]
    if not valid:
        return []
    width = float(page_width or max(float(block["bbox"][2]) for block in valid))
    height = float(page_height or max(float(block["bbox"][3]) for block in valid))
    headings = [
        block
        for block in valid
        if any(
            _FT_BRIEFING_HEADING.fullmatch(re.sub(r"\s+", " ", line).strip())
            for line in str(block.get("text") or "").splitlines()
        )
        and float(block["bbox"][0]) >= width * 0.65
        and float(block["bbox"][1]) <= height * 0.4
    ]
    if not headings:
        return []

    heading = min(headings, key=lambda block: float(block["bbox"][1]))
    start_y = float(heading["bbox"][1])
    left_limit = max(width * 0.72, float(heading["bbox"][0]) - width * 0.03)
    selected = [
        block
        for block in valid
        if float(block["bbox"][0]) >= left_limit
        and float(block["bbox"][1]) >= start_y - 4.0
        and float(block["bbox"][3]) <= height * 0.78
    ]
    selected.sort(key=_block_sort_key)
    bounded: list[dict[str, Any]] = []
    for block in selected:
        first_line = re.sub(
            r"\s+", " ", str(block.get("text") or "").splitlines()[0]
        ).strip()
        if bounded and _FT_BRIEFING_BOUNDARY.match(first_line):
            break
        bounded.append(block)
    return bounded


def select_front_page_directory_blocks(
    blocks: list[dict[str, Any]],
    publication_type: str,
    *,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[dict[str, Any]]:
    publication = str(publication_type or "").strip().upper()
    if publication == "WSJ":
        return select_whats_news_blocks(
            blocks,
            page_width=page_width,
            page_height=page_height,
        )
    if publication == "FT":
        return select_ft_briefing_blocks(
            blocks,
            page_width=page_width,
            page_height=page_height,
        )
    return []


def build_issue_pages(page_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for page_order, result in enumerate(
        sorted(page_results, key=lambda item: int(item.get("page") or 0)),
        start=1,
    ):
        pdf_page = int(result.get("page") or 0)
        if pdf_page <= 0:
            continue
        pages.append(
            {
                "pdf_page": pdf_page,
                "page_order": page_order,
                "print_page_label": result.get("print_page_label"),
                "print_section": result.get("print_section"),
                "print_page_source": result.get("print_page_source"),
            }
        )
    return pages


def build_front_page(
    page_results: list[dict[str, Any]],
    publication_type: str | None = None,
) -> dict[str, Any] | None:
    publication = str(publication_type or "").strip().upper()
    if publication not in {"WSJ", "FT"}:
        return None
    expected_label = "A1" if publication == "WSJ" else "1"
    directory_name = "What’s News" if publication == "WSJ" else "Briefing"
    for result in sorted(page_results, key=lambda item: int(item.get("page") or 0)):
        if str(result.get("print_page_label") or "").upper() != expected_label:
            continue
        whats_news = result.get("whats_news")
        if isinstance(whats_news, dict) and whats_news.get("groups"):
            return {
                "pdf_page": int(result.get("page") or 0),
                "print_page_label": expected_label,
                "print_section": result.get("print_section"),
                "directory_name": directory_name,
                "whats_news": whats_news,
            }
    return None


def enrich_page_result(
    result: dict[str, Any],
    publication_type: str,
    pdf_page: int,
    header_text: str = "",
    header_blocks: list[dict[str, Any]] | None = None,
    *,
    whats_news_text: str = "",
    whats_news_blocks: list[dict[str, Any]] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
    existing_metadata: Literal["replace", "merge"] = "replace",
) -> dict[str, Any]:
    if existing_metadata not in {"replace", "merge"}:
        raise ValueError(f"Unsupported existing_metadata mode: {existing_metadata}")
    enriched = dict(result)
    publication = str(publication_type or "").strip().upper()
    metadata_attempted = bool(
        str(header_text or "").strip()
        or any(_valid_bbox_block(block) for block in (header_blocks or []))
    )
    whats_news_attempted = bool(
        str(whats_news_text or "").strip()
        or any(_valid_bbox_block(block) for block in (whats_news_blocks or []))
    )
    attempted = metadata_attempted or whats_news_attempted

    if pdf_page == 1 and publication in {"FT", "WSJ"} and metadata_attempted:
        enriched["is_weekend"] = detect_weekend_issue(
            publication,
            header_text=header_text,
            header_blocks=header_blocks,
        )
    if attempted and existing_metadata == "replace":
        for key in (
            "print_page_label",
            "print_section",
            "print_page_source",
            "whats_news",
            "print_layout_status",
            "print_layout_version",
        ):
            enriched.pop(key, None)

    layout = extract_print_layout(
        publication_type,
        pdf_page,
        header_text=header_text,
        header_blocks=header_blocks,
        page_width=page_width,
    )
    layout_found = bool(layout.get("print_page_label") or layout.get("print_section"))
    if metadata_attempted:
        if existing_metadata == "replace":
            enriched.update(layout)
        elif layout_found:
            for key in ("print_page_label", "print_section"):
                if layout.get(key):
                    enriched[key] = layout[key]
            enriched["print_page_source"] = layout["print_page_source"]
    if attempted and existing_metadata == "replace":
        enriched["print_layout_version"] = PRINT_LAYOUT_VERSION
        enriched["print_layout_status"] = (
            "found" if layout_found else "not_found"
        )
    elif layout_found:
        enriched["print_layout_version"] = PRINT_LAYOUT_VERSION
        enriched["print_layout_status"] = "found"

    if publication not in {"WSJ", "FT"}:
        enriched.pop("whats_news", None)
        return enriched

    whats_news = {"groups": []}
    if pdf_page == 1 and whats_news_attempted:
        whats_news = extract_front_page_directory(
            publication,
            whats_news_text,
            blocks=whats_news_blocks,
            page_width=page_width,
            page_height=page_height,
        )
        if whats_news["groups"]:
            enriched["whats_news"] = whats_news
            enriched["print_layout_version"] = PRINT_LAYOUT_VERSION

    expected_label = "A1" if publication == "WSJ" else "1"
    if pdf_page == 1 and whats_news["groups"]:
        if not enriched.get("print_page_label"):
            enriched["print_page_label"] = expected_label
            enriched["print_page_source"] = "derived"
        if not enriched.get("print_section"):
            enriched["print_section"] = "PAGE ONE"
        enriched["print_layout_status"] = "found"

    is_supported_front = (
        pdf_page == 1
        and str(enriched.get("print_page_label") or "").upper() == expected_label
    )
    if not is_supported_front:
        enriched.pop("whats_news", None)
    return enriched


def _header_candidates(
    header_text: str,
    header_blocks: list[dict[str, Any]] | None,
    *,
    page_width: float | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    valid_blocks = [
        block for block in (header_blocks or []) if _valid_bbox_block(block)
    ]
    resolved_page_width = float(page_width or 0.0)
    if header_blocks:
        for block in sorted(
            valid_blocks,
            key=_block_sort_key,
        ):
            raw_text = str(block.get("text") or "")
            lines = raw_text.splitlines()
            for index, line in enumerate(lines):
                normalized = re.sub(r"\s+", " ", line).strip()
                if not normalized or _looks_like_reference_line(normalized):
                    continue
                if _is_split_reference_label(lines, index, normalized):
                    continue
                candidates.append(
                    {
                        "text": normalized,
                        "block_text": re.sub(r"\s+", " ", raw_text).strip(),
                        "bbox": block["bbox"],
                        "page_width": resolved_page_width,
                        "standalone_block": bool(
                            re.fullmatch(r"[A-Z]?\d{1,3}", raw_text.strip(), re.I)
                        ),
                    }
                )
    text_lines = str(header_text or "").splitlines()
    for index, line in enumerate(text_lines):
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized or _looks_like_reference_line(normalized):
            continue
        if _is_split_reference_label(text_lines, index, normalized):
            continue
        candidates.append(
            {
                "text": normalized,
                "block_text": normalized,
                "bbox": None,
                "page_width": None,
                "standalone_block": True,
            }
        )
    return candidates


def _extract_wsj_page(candidates: list[dict[str, Any]]) -> str | None:
    section_pattern = "|".join(re.escape(section) for section in _WSJ_SECTIONS)
    for candidate in candidates:
        line = candidate["text"]
        if candidate.get("bbox") and not _candidate_is_at_page_edge(candidate):
            continue
        standalone = re.fullmatch(r"(?i)([A-Z]\d{1,3})", line)
        if standalone:
            return standalone.group(1).upper()
        match = re.match(r"(?i)^([A-Z]\d{1,3})\s*\|", line)
        if match:
            return match.group(1).upper()
        match = re.fullmatch(
            rf"(?i)([A-Z]\d{{1,3}})\s+(?:{section_pattern})\.??",
            line,
        )
        if match:
            return match.group(1).upper()
        match = re.search(r"(?i)(?:^|\|)\s*([A-Z]\d{1,3})\s*$", line)
        if match and "|" in line:
            return match.group(1).upper()
        match = re.fullmatch(
            r"(?i)(?:THE\s+)?WALL\s+STREET\s+JOURNAL[.,]?\s*\|?\s*([A-Z]\d{1,3})",
            line,
        )
        if match:
            return match.group(1).upper()
    return None


def _extract_numeric_page(
    publication: str,
    candidates: list[dict[str, Any]],
) -> str | None:
    publication_names = {
        "FT": "FINANCIAL TIMES",
        "TE": "THE ECONOMIST",
        "BARRONS": "BARRON(?:['\u2019]S|S)",
    }
    publication_name = publication_names.get(publication, re.escape(publication))
    for candidate in candidates:
        line = candidate["text"]
        if (
            re.fullmatch(r"\d{1,3}", line)
            and (
                candidate.get("standalone_block")
                or (
                    publication == "BARRONS"
                    and re.search(
                        r"(?i)\bBARRON(?:['\u2019]S|S)\b",
                        str(candidate.get("block_text") or ""),
                    )
                )
            )
            and _candidate_is_at_page_edge(candidate)
        ):
            return line
        match = re.search(
            rf"(?:^|\s)(\d{{1,3}})\s+(?:{publication_name})\b|"
            rf"\b(?:{publication_name})\b\s+(\d{{1,3}})(?:\s|$)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            return next(group for group in match.groups() if group)
    return None


def _find_section(
    candidates: list[dict[str, Any]],
    sections: tuple[str, ...],
) -> str | None:
    normalized = " ".join(candidate["text"].upper() for candidate in candidates)
    comparable = re.sub(r"\s*&\s*", "&", normalized)
    for section in sorted(sections, key=len, reverse=True):
        needle = re.sub(r"\s*&\s*", "&", section)
        if re.search(rf"(?<![A-Z]){re.escape(needle)}(?![A-Z])", comparable):
            return section
    return None


def _looks_like_reference_line(line: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(continued\s+(?:on|from)|turn\s+to|see)\s+(?:page\s+)?[A-Z]?\d+\b",
            line,
        )
    )


def _is_split_reference_label(
    lines: list[str],
    index: int,
    normalized: str,
) -> bool:
    if index <= 0 or not re.fullmatch(r"(?i)[A-Z]\d{1,3}", normalized):
        return False
    previous = re.sub(r"\s+", " ", lines[index - 1]).strip()
    return bool(
        re.search(
            r"(?i)(?:continued\s+(?:on|from)|turn\s+to|see)(?:\s+page)?\s*$",
            previous,
        )
    )


def _candidate_is_at_page_edge(candidate: dict[str, Any]) -> bool:
    bbox = candidate.get("bbox")
    page_width = float(candidate.get("page_width") or 0.0)
    if not bbox or page_width <= 0:
        return False
    x0, _y0, x1, _y1 = (float(value) for value in bbox[:4])
    return x0 <= page_width * 0.1 or x1 >= page_width * 0.9


def _valid_bbox_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and str(block.get("text") or "").strip() != ""
        and isinstance(block.get("bbox"), (list, tuple))
        and len(block["bbox"]) >= 4
    )


def _block_sort_key(block: dict[str, Any]) -> tuple[float, float]:
    return float(block["bbox"][1]), float(block["bbox"][0])


def _append_whats_news_item(
    items: list[dict[str, Any]],
    fragments: list[str],
    page_label: str,
) -> None:
    item_text = re.sub(r"\s+", " ", " ".join(fragments)).strip(" -\u2013\u2014")
    if not item_text:
        return
    items.append(
        {
            "text": item_text,
            "target_print_page_label": page_label.upper(),
            "target_article_id": None,
        }
    )
