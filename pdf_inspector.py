from __future__ import annotations


def is_vector_pdf(
    pdf_path: str,
    max_check_pages: int = 3,
    char_threshold: int = 100,
) -> bool:
    """Inspect real PDF contents and decide whether native text is present."""
    total_chars = _count_chars_with_pymupdf(pdf_path, max_check_pages)
    if total_chars is None:
        total_chars = _count_chars_with_pypdf(pdf_path, max_check_pages)
    return total_chars > char_threshold


def _count_printable_chars(text: str) -> int:
    return sum(1 for char in text if char.isprintable() and not char.isspace())


def _count_chars_with_pymupdf(pdf_path: str, max_check_pages: int) -> int | None:
    try:
        import fitz
    except ImportError:
        return None

    total = 0
    fitz.TOOLS.mupdf_display_errors(False)
    try:
        with fitz.open(pdf_path) as doc:
            for page_idx in _sample_page_indices(len(doc), max_check_pages):
                page = doc[page_idx]
                total += _count_printable_chars(page.get_text("text") or "")
        return total
    except Exception:
        return None
    finally:
        fitz.TOOLS.mupdf_display_errors(True)


def _count_chars_with_pypdf(pdf_path: str, max_check_pages: int) -> int:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    total = 0
    for page_idx in _sample_page_indices(len(reader.pages), max_check_pages):
        total += _count_printable_chars(reader.pages[page_idx].extract_text() or "")
    return total


def _sample_page_indices(page_count: int, max_check_pages: int) -> list[int]:
    """Sample across the document so scanned covers do not hide native inner pages."""
    sample_count = min(max(page_count, 0), max(max_check_pages, 1))
    if sample_count <= 0:
        return []
    if sample_count == 1:
        return [0]
    last_index = page_count - 1
    return sorted(
        {
            round(position * last_index / (sample_count - 1))
            for position in range(sample_count)
        }
    )
