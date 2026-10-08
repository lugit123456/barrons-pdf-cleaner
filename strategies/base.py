from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from metadata_parser import PdfMetadata


@dataclass(frozen=True)
class ParseContext:
    pdf_path: Path
    target_output_dir: Path
    image_dir: Path
    metadata: PdfMetadata
    original_filename: str
    article_writer: Callable[[list[dict[str, Any]], str], list[Path]] | None = None
    selected_pages: set[int] | None = None


@dataclass(frozen=True)
class ParseResult:
    body_markdown: str
    engine_name: str
    articles: list[dict[str, Any]] | None = None
    complete: bool = True
    failed_pages: tuple[int, ...] = ()
    pages: list[dict[str, Any]] | None = None
    front_page: dict[str, Any] | None = None
    is_weekend: bool = False


class ParseStrategy(Protocol):
    engine_name: str

    def parse(self, context: ParseContext) -> ParseResult:
        ...
