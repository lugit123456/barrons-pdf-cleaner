from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


@dataclass(frozen=True)
class PdfMetadata:
    publication_type: str
    publication_date: str


def parse_pdf_metadata(pdf_path: str) -> PdfMetadata:
    """Parse publication metadata from filename, falling back to mtime."""
    path = Path(pdf_path)
    filename = path.name
    publication_type = _parse_publication_type(filename)
    publication_date = _parse_publication_date(filename)

    if publication_type and publication_date:
        return PdfMetadata(publication_type, publication_date)

    fallback_date = datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
    return PdfMetadata(publication_type or "General", publication_date or fallback_date)


def _parse_publication_type(filename: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", filename.lower()).strip()

    if re.search(r"\bbarron(?:s)?\b", normalized):
        return "BARRONS"
    if "wall street journal" in normalized or re.search(r"\bwsj\b", normalized):
        return "WSJ"
    if "financial times" in normalized or re.search(r"\bft\b", normalized):
        return "FT"
    if "the economist" in normalized or re.search(r"\beconomist\b", normalized):
        return "TE"
    return None


def _parse_publication_date(filename: str) -> str | None:
    iso_match = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", filename)
    if iso_match:
        year, month, day = (int(part) for part in iso_match.groups())
        return _safe_iso_date(year, month, day)

    month_match = re.search(
        r"([A-Za-z]+)[-_ ]+(\d{1,2})[-_, ]+(20\d{2})",
        filename,
        flags=re.IGNORECASE,
    )
    if month_match:
        month_name, day, year = month_match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return _safe_iso_date(int(year), month, int(day))

    return None


def _safe_iso_date(year: int, month: int, day: int) -> str | None:
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return None
