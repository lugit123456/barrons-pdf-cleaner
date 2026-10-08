from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any


GLOSSARY_VERSION = 2
ZH_ENGLISH_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Za-z][A-Za-z0-9]*(?:[.'’_-][A-Za-z0-9]+)*"
    r"(?:(?:[ \t]+|,\s*)[A-Za-z&][A-Za-z0-9]*(?:[.'’_-][A-Za-z0-9]+)*){0,7})"
    r"(?![A-Za-z0-9_])"
)
GLOSSARY_TYPES = {
    "person",
    "organization",
    "company",
    "policy_law",
    "event",
    "place_context",
    "work",
    "proper_concept",
    "acronym",
}
GENERIC_TERMS = {
    "business", "ceo", "company", "democracy", "economy", "government",
    "globalization", "inflation", "market", "president", "sovereignty", "us", "uk",
}


def enrich_article_glossary(
    article: dict[str, Any],
    chat_completion_json: Callable[..., Any],
) -> dict[str, Any]:
    """Add validated glossary entries without changing the article text."""
    if not needs_glossary_refresh(article):
        return article

    paragraphs = article_paragraphs(article)
    if not paragraphs:
        return _mark_glossary_complete(article, [], [])

    configured_max_terms = max(int(os.getenv("LLM_GLOSSARY_MAX_TERMS", "12")), 1)
    max_input_chars = max(int(os.getenv("LLM_GLOSSARY_MAX_INPUT_CHARS", "24000")), 2000)
    prompt_paragraphs = _paragraphs_for_prompt(paragraphs, max_input_chars)
    if not prompt_paragraphs:
        return _mark_glossary_complete(article, [], [])
    max_candidates = max(int(os.getenv("LLM_GLOSSARY_MAX_ZH_CANDIDATES", "16")), 1)
    candidate_hints = extract_zh_english_candidates(
        prompt_paragraphs,
        max_candidates=max_candidates,
    )

    title = str(article.get("title") or "Untitled").strip()
    prompt = _build_glossary_prompt(
        title,
        prompt_paragraphs,
        configured_max_terms,
        candidate_hints=candidate_hints,
    )
    data = chat_completion_json(
        operation=f"文章术语解读：{title}",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=int(os.getenv("LLM_GLOSSARY_MAX_TOKENS", "5000")),
        model_env="OPENAI_GLOSSARY_MODEL",
        max_retries=int(os.getenv("LLM_GLOSSARY_MAX_RETRIES", "2")),
    )
    entries, annotations = normalize_glossary_response(
        data,
        paragraphs=paragraphs,
        max_terms=configured_max_terms,
    )
    return _mark_glossary_complete(article, entries, annotations)


def needs_glossary_refresh(article: dict[str, Any]) -> bool:
    try:
        version = int(article.get("glossary_version") or 0)
    except (TypeError, ValueError):
        version = 0
    return not article.get("glossary_analysis_complete") or version < GLOSSARY_VERSION


def article_paragraphs(article: dict[str, Any]) -> list[dict[str, str]]:
    raw_paragraphs = article.get("paragraphs")
    paragraphs: list[dict[str, str]] = []
    if isinstance(raw_paragraphs, list):
        for raw in raw_paragraphs:
            if not isinstance(raw, dict):
                continue
            en_text = str(raw.get("en_text") or raw.get("en_html") or "").strip()
            zh_text = str(raw.get("zh_text") or "").strip()
            role = str(raw.get("role") or "body").strip() or "body"
            if en_text or zh_text:
                paragraphs.append(
                    {"en_text": en_text, "zh_text": zh_text, "role": role}
                )
    if paragraphs:
        return paragraphs

    content = str(
        article.get("content_markdown") or article.get("content_raw") or ""
    ).strip()
    return [
        {"en_text": part.strip(), "zh_text": "", "role": "body"}
        for part in re.split(r"\n\s*\n", content)
        if part.strip()
    ]


def normalize_glossary_response(
    data: Any,
    *,
    paragraphs: list[dict[str, str]],
    max_terms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(data, dict):
        raw_terms = data.get("terms", data.get("glossary", data.get("items", [])))
    elif isinstance(data, list):
        raw_terms = data
    else:
        raw_terms = []
    if not isinstance(raw_terms, list):
        return [], []

    entries: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_locations: set[tuple[str, int, str, int]] = set()

    for raw_term in raw_terms:
        if len(entries) >= max_terms:
            break
        if not isinstance(raw_term, dict):
            continue
        term = _clean_inline_text(
            raw_term.get("term")
            or raw_term.get("canonical_term")
            or raw_term.get("name")
        )
        term_type = str(raw_term.get("type") or "proper_concept").strip().lower()
        if term_type not in GLOSSARY_TYPES:
            term_type = "proper_concept"
        description = _clean_description(
            raw_term.get("description_zh")
            or raw_term.get("explanation_zh")
            or raw_term.get("description")
        )
        if not _is_valuable_proper_term(term, term_type) or len(description) < 60:
            continue
        description = _truncate_description(description)
        glossary_id = build_glossary_id(term, term_type)
        if glossary_id in seen_ids:
            continue

        raw_occurrences = raw_term.get("occurrences") or raw_term.get("locations") or []
        if isinstance(raw_occurrences, dict):
            raw_occurrences = [raw_occurrences]
        validated_occurrences = _normalize_occurrences(
            raw_occurrences,
            term=term,
            paragraphs=paragraphs,
            glossary_id=glossary_id,
            seen_locations=seen_locations,
        )
        if not validated_occurrences:
            validated_occurrences = _find_first_occurrence(
                term,
                paragraphs,
                glossary_id,
                seen_locations,
            )
        if not validated_occurrences:
            continue

        seen_ids.add(glossary_id)
        entries.append(
            {
                "id": glossary_id,
                "term": term,
                "term_zh": _clean_inline_text(
                    raw_term.get("term_zh") or raw_term.get("name_zh")
                ),
                "type": term_type,
                "description_zh": description,
                "version": GLOSSARY_VERSION,
            }
        )
        annotations.extend(validated_occurrences[:2])

    return entries, annotations


def normalize_existing_glossary_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        raw_entries = [
            {"id": key, **entry}
            for key, entry in value.items()
            if isinstance(entry, dict)
        ]
    elif isinstance(value, list):
        raw_entries = [entry for entry in value if isinstance(entry, dict)]
    else:
        raw_entries = []

    normalized = []
    for raw in raw_entries:
        term = _clean_inline_text(raw.get("term") or raw.get("name"))
        term_type = str(raw.get("type") or "proper_concept").strip().lower()
        if term_type not in GLOSSARY_TYPES:
            term_type = "proper_concept"
        description = _clean_description(
            raw.get("description_zh")
            or raw.get("explanation_zh")
            or raw.get("description")
        )
        if not _is_valuable_proper_term(term, term_type) or not description:
            continue
        normalized.append(
            {
                "id": str(raw.get("id") or build_glossary_id(term, term_type)),
                "term": term,
                "term_zh": _clean_inline_text(raw.get("term_zh") or raw.get("name_zh")),
                "type": term_type,
                "description_zh": description[:200].rstrip(),
                "version": int(raw.get("version") or GLOSSARY_VERSION),
            }
        )
    return normalized


def normalize_term_annotations(value: Any, paragraph_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    seen: set[tuple[str, str, int, str, int]] = set()
    for raw in value:
        if not isinstance(raw, dict):
            continue
        glossary_id = str(raw.get("glossary_id") or "").strip()
        surface = _clean_inline_text(raw.get("surface") or raw.get("text"))
        try:
            paragraph_index = int(raw.get("paragraph_index") or 0)
            occurrence = max(int(raw.get("occurrence") or 1), 1)
        except (TypeError, ValueError):
            continue
        text_field = _normalize_text_field(raw.get("text_field"))
        if (
            not glossary_id
            or not surface
            or text_field != "zh_text"
            or paragraph_index < 1
            or paragraph_index > paragraph_count
        ):
            continue
        key = (
            glossary_id,
            text_field,
            paragraph_index,
            surface.lower(),
            occurrence,
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "glossary_id": glossary_id,
                "paragraph_index": paragraph_index,
                "text_field": text_field,
                "surface": surface,
                "occurrence": occurrence,
            }
        )
    return normalized


def build_glossary_id(term: str, term_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
    if not slug:
        slug = "term"
    return f"{term_type}_{slug[:80]}"


def _paragraphs_for_prompt(
    paragraphs: list[dict[str, str]],
    max_input_chars: int,
) -> list[tuple[int, str, str]]:
    result = []
    used = 0
    for index, paragraph in enumerate(paragraphs, start=1):
        en_text = paragraph.get("en_text", "").strip()
        zh_text = paragraph.get("zh_text", "").strip()
        if not en_text and not zh_text:
            continue
        en_text, used = _take_prompt_text(en_text, used, max_input_chars)
        zh_text, used = _take_prompt_text(zh_text, used, max_input_chars)
        if en_text or zh_text:
            result.append((index, en_text, zh_text))
        if used >= max_input_chars:
            break
    return result


def _build_glossary_prompt(
    title: str,
    paragraphs: list[tuple[int, str, str]],
    max_terms: int,
    *,
    candidate_hints: list[dict[str, Any]] | None = None,
) -> str:
    paragraph_text = "\n\n".join(
        "\n".join(
            line
            for line in (
                f"[P{index}.EN] {en_text}" if en_text else "",
                f"[P{index}.ZH] {zh_text}" if zh_text else "",
            )
            if line
        )
        for index, en_text, zh_text in paragraphs
    )
    candidates = candidate_hints or []
    candidate_text = "\n".join(
        f'- [{item["candidate_id"]}] P{item["paragraph_index"]}.ZH: '
        f'{item["surface"]}'
        for item in candidates
    )
    candidate_instruction = ""
    if candidates:
        candidate_instruction = f"""
The following English strings were extracted from the Chinese column as possible proper-name hints. They are not mandatory. Select them only when their explanation adds concrete background for this article; omit ordinary words, obvious references, or names with no useful article context.

Possible Chinese-column candidates:
{candidate_text}
""".strip()
    return f"""
You are a senior English-Chinese translator and global political-economic background editor. Analyze the bilingual article and select at most {max_terms} English-language proper terms that genuinely need contextual explanation for a Chinese reader.

Only annotate an exact English substring that remains visible in [P<number>.ZH]. Readers use the Chinese translation, so never create an annotation in [P<number>.EN].

Strict negative filter: never annotate ordinary English vocabulary, generic abstract concepts, or common roles. In particular, never explain sovereignty, democracy, inflation, globalization, President, US, CEO, or similar common words.

Allowed categories only: a specific person; a named organization or company; a named law, policy, or financial/industry mechanism; a named event; a historical or geographic reference with specific context; a named book, report, film, publication, project, or acronym. A bare ordinary word is never a proper term.

Every explanation must state both (1) who or what it is and (2) its role or relevant background in this article. Avoid dictionary definitions and invented facts.

{candidate_instruction}

For every selected term:
- term must be the canonical English name.
- term_zh is its conventional Chinese name, or an empty string when none exists.
- type must be one of: person, organization, company, policy_law, event, place_context, work, proper_concept, acronym.
- description_zh must be an objective Chinese introduction of roughly 100-200 Chinese characters. Explain identity/meaning and why it matters in this article. Do not invent facts.
- occurrences may contain only the first useful occurrence in the Chinese column.
- paragraph_index is the integer from the [P<number>.ZH] marker.
- text_field must always be "zh_text".
- surface must copy the exact visible English substring from that Chinese column. Do not translate or normalize it.

Return strict JSON only. Do not modify, quote, summarize, or reproduce the article body outside surface.

Article title: {title}

{paragraph_text}

Return JSON ONLY:
{{
  "terms": [
    {{
      "term": "Jerome Powell",
      "term_zh": "杰罗姆·鲍威尔",
      "type": "person",
      "description_zh": "100-200字中文介绍",
      "occurrences": [
        {{"paragraph_index": 3, "text_field": "zh_text", "surface": "Jerome Powell", "occurrence": 1}}
      ]
    }}
  ]
}}
""".strip()


def extract_zh_english_candidates(
    paragraphs: list[tuple[int, str, str]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates = []
    seen: set[str] = set()
    for paragraph_index, _en_text, zh_text in paragraphs:
        for match in ZH_ENGLISH_CANDIDATE_RE.finditer(zh_text):
            surface = _clean_inline_text(match.group(1)).strip(" ,.;:!?()[]{}")
            if not _is_named_english_candidate(surface):
                continue
            key = surface.casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "candidate_id": f"ZH{len(candidates) + 1}",
                    "paragraph_index": paragraph_index,
                    "text_field": "zh_text",
                    "surface": surface,
                }
            )
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _is_named_english_candidate(surface: str) -> bool:
    if len(surface) < 2 or not re.search(r"[A-Za-z]", surface):
        return False
    return _is_valuable_proper_term(surface, "proper_concept")


def _missing_required_candidates(
    candidates: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    covered = {
        (
            int(annotation.get("paragraph_index") or 0),
            str(annotation.get("text_field") or "en_text"),
            str(annotation.get("surface") or "").casefold(),
        )
        for annotation in annotations
    }
    return [
        candidate
        for candidate in candidates
        if (
            candidate["paragraph_index"],
            "zh_text",
            candidate["surface"].casefold(),
        ) not in covered
    ]


def _merge_glossary_results(
    entries: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    extra_entries: list[dict[str, Any]],
    extra_annotations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entry_map = {str(entry.get("id") or ""): entry for entry in entries}
    for entry in extra_entries:
        entry_map[str(entry.get("id") or "")] = entry

    merged_annotations = []
    seen_annotations = set()
    for annotation in annotations + extra_annotations:
        key = (
            annotation.get("glossary_id"),
            annotation.get("paragraph_index"),
            annotation.get("text_field"),
            str(annotation.get("surface") or "").casefold(),
            annotation.get("occurrence"),
        )
        if key in seen_annotations:
            continue
        seen_annotations.add(key)
        merged_annotations.append(annotation)
    return list(entry_map.values()), merged_annotations


def _normalize_occurrences(
    value: Any,
    *,
    term: str,
    paragraphs: list[dict[str, str]],
    glossary_id: str,
    seen_locations: set[tuple[str, int, str, int]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        try:
            paragraph_index = int(raw.get("paragraph_index") or 0)
            occurrence = max(int(raw.get("occurrence") or 1), 1)
        except (TypeError, ValueError):
            continue
        if paragraph_index < 1 or paragraph_index > len(paragraphs):
            continue
        surface = _clean_inline_text(raw.get("surface") or term)
        text_field = _occurrence_text_field(
            raw,
            paragraphs[paragraph_index - 1],
            surface,
            occurrence,
        )
        paragraph = paragraphs[paragraph_index - 1].get(text_field, "")
        actual_surface = _actual_occurrence_surface(paragraph, surface, occurrence)
        if not actual_surface:
            continue
        location_key = (
            text_field,
            paragraph_index,
            actual_surface.lower(),
            occurrence,
        )
        if location_key in seen_locations:
            continue
        seen_locations.add(location_key)
        normalized.append(
            {
                "glossary_id": glossary_id,
                "paragraph_index": paragraph_index,
                "text_field": text_field,
                "surface": actual_surface,
                "occurrence": occurrence,
            }
        )
    return normalized


def _find_first_occurrence(
    term: str,
    paragraphs: list[dict[str, str]],
    glossary_id: str,
    seen_locations: set[tuple[str, int, str, int]],
) -> list[dict[str, Any]]:
    return _find_term_in_columns(
        term,
        paragraphs,
        glossary_id,
        seen_locations,
        fields=("zh_text",),
    )


def _add_missing_column_occurrences(
    term: str,
    paragraphs: list[dict[str, str]],
    glossary_id: str,
    seen_locations: set[tuple[str, int, str, int]],
    occurrences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    present_fields = {item.get("text_field") for item in occurrences}
    missing_fields = tuple(
        field for field in ("zh_text", "en_text") if field not in present_fields
    )
    if not missing_fields:
        return occurrences
    counterparts = _find_term_in_columns(
        term,
        paragraphs,
        glossary_id,
        seen_locations,
        fields=missing_fields,
    )
    combined = occurrences + counterparts
    return sorted(
        combined,
        key=lambda item: (item.get("text_field") != "zh_text", item["paragraph_index"]),
    )


def _find_term_in_columns(
    term: str,
    paragraphs: list[dict[str, str]],
    glossary_id: str,
    seen_locations: set[tuple[str, int, str, int]],
    *,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    found = []
    for text_field in fields:
        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            actual_surface = _actual_occurrence_surface(
                paragraph.get(text_field, ""), term, 1
            )
            if not actual_surface:
                continue
            location_key = (text_field, paragraph_index, actual_surface.lower(), 1)
            if location_key in seen_locations:
                break
            seen_locations.add(location_key)
            found.append(
                {
                    "glossary_id": glossary_id,
                    "paragraph_index": paragraph_index,
                    "text_field": text_field,
                    "surface": actual_surface,
                    "occurrence": 1,
                }
            )
            break
    return found


def _occurrence_text_field(
    occurrence: dict[str, Any],
    paragraph: dict[str, str],
    surface: str,
    occurrence_index: int,
) -> str:
    raw_field = occurrence.get("text_field") or occurrence.get("column")
    if raw_field and _normalize_text_field(raw_field) != "zh_text":
        return ""
    if _actual_occurrence_surface(paragraph.get("zh_text", ""), surface, occurrence_index):
        return "zh_text"
    return ""


def _normalize_text_field(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"zh", "zh_text", "chinese", "translation"}:
        return "zh_text"
    return ""


def _is_valuable_proper_term(term: str, term_type: str) -> bool:
    normalized = _clean_inline_text(term)
    lowered = normalized.casefold()
    if len(normalized) < 2 or lowered in GENERIC_TERMS or not re.search(r"[A-Za-z]", normalized):
        return False
    if term_type == "acronym":
        return bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{1,11}", normalized)) and lowered not in GENERIC_TERMS
    if re.fullmatch(r"[a-z][a-z-]*", normalized):
        return False
    return bool(
        re.search(r"[A-Z]", normalized)
        or re.search(r"\d", normalized)
        or "." in normalized
    )


def _take_prompt_text(
    text: str,
    used: int,
    max_input_chars: int,
) -> tuple[str, int]:
    if not text or used >= max_input_chars:
        return "", used
    remaining = max_input_chars - used
    if len(text) > remaining:
        text = text[:remaining].rsplit(" ", 1)[0].strip() or text[:remaining].strip()
    return text, used + len(text)


def _actual_occurrence_surface(text: str, surface: str, occurrence: int) -> str:
    if not text or not surface:
        return ""
    lowered_text = text.lower()
    lowered_surface = surface.lower()
    start = 0
    found = -1
    for _ in range(max(occurrence, 1)):
        found = lowered_text.find(lowered_surface, start)
        while found >= 0 and not _has_term_boundaries(text, found, len(surface)):
            found = lowered_text.find(lowered_surface, found + 1)
        if found < 0:
            return ""
        start = found + len(surface)
    return text[found : found + len(surface)]


def _has_term_boundaries(text: str, start: int, length: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after_index = start + length
    after = text[after_index] if after_index < len(text) else ""
    first = text[start] if start < len(text) else ""
    last = text[after_index - 1] if after_index > 0 else ""
    if _is_ascii_word_char(first) and _is_ascii_word_char(before):
        return False
    if _is_ascii_word_char(last) and _is_ascii_word_char(after):
        return False
    return True


def _is_ascii_word_char(value: str) -> bool:
    return bool(value and re.fullmatch(r"[A-Za-z0-9_]", value))


def _mark_glossary_complete(
    article: dict[str, Any],
    entries: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(article)
    enriched["glossary_entries"] = entries
    enriched["term_annotations"] = annotations
    enriched["glossary_analysis_complete"] = True
    enriched["glossary_version"] = GLOSSARY_VERSION
    return enriched


def _clean_inline_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().strip("`*_#")


def _clean_description(value: Any) -> str:
    description = re.sub(r"\s+", " ", str(value or "")).strip()
    return description.replace('"', "「").replace("'", "’")


def _truncate_description(description: str, limit: int = 200) -> str:
    if len(description) <= limit:
        return description
    candidate = description[:limit]
    sentence_end = max(candidate.rfind(mark) for mark in "。！？")
    if sentence_end >= 99:
        return candidate[: sentence_end + 1]
    return candidate[: limit - 1].rstrip("，,；;：: ") + "…"
