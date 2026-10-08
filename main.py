from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from database_writer import build_pdf_id, write_pdf_database
from metadata_parser import PdfMetadata, parse_pdf_metadata
from pdf_inspector import is_vector_pdf
from pdf_repair import prepare_pdf_for_parsing
from processing_state import ProcessingStateDB, compute_sha256
from strategies import HybridPdfStrategy, ParseContext, ParseResult, ScannedPdfStrategy


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
NORMALIZED_IMAGE_RE = re.compile(r"^page_\d+_fig_\d+\.[A-Za-z0-9]+$")


@dataclass(frozen=True)
class ProcessOutcome:
    output_markdown: Path
    target_dir: Path
    pdf_id: str
    article_count: int
    complete: bool = True
    failed_pages: tuple[int, ...] = ()


def main() -> None:
    load_dotenv(os.getenv("DOTENV_PATH") or None)
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    selected_pages = _parse_pages_arg(args.pages)
    state_db = ProcessingStateDB(Path(args.state_db))
    imported = state_db.import_existing_outputs(output_dir)
    if imported:
        print(f"🔄 已将 {imported} 份历史输出迁移到抓取记录库：{state_db.db_path}")

    if args.status:
        _print_processing_status(state_db)
        return

    if not input_dir.exists():
        print(f"❌ 输入目录不存在：{input_dir}")
        return

    pdf_files = sorted(input_dir.glob("*.pdf"))
    selectors = _normalize_file_selectors(args.file)
    if selectors:
        pdf_files = [path for path in pdf_files if _matches_file_selector(path, selectors)]
    if not pdf_files:
        detail = f"，筛选条件：{', '.join(selectors)}" if selectors else ""
        print(f"⚠️ 未在输入目录找到 PDF：{input_dir}{detail}")
        return

    print(f"🚀 开始批量解析，共 {len(pdf_files)} 个 PDF。")
    for pdf_path in pdf_files:
        try:
            metadata = parse_pdf_metadata(str(pdf_path))
            if args.publication:
                metadata = PdfMetadata(args.publication.strip(), metadata.publication_date)
            file_sha = compute_sha256(pdf_path)
            if not args.force and selected_pages is None:
                processed = state_db.find_processed(
                    file_sha,
                    metadata.publication_type,
                    metadata.publication_date,
                    pdf_path.name,
                )
                if processed:
                    output_complete, incomplete_reason = _existing_output_is_complete(
                        output_dir,
                        metadata,
                        pdf_path,
                    )
                    if output_complete:
                        print(
                            f"\n⏭️ 已抓取，跳过：{pdf_path.name} "
                            f"({processed['publication_type']} / {processed['publication_date']} / "
                            f"{processed['article_count']} 篇；使用 --force 强制重跑)"
                        )
                        continue

                    print(
                        f"\n♻️ 状态库虽已记录，但输出不完整，自动续跑：{pdf_path.name}"
                        f"（{incomplete_reason}）"
                    )

            outcome = process_pdf(
                pdf_path=pdf_path,
                output_root=output_dir,
                max_check_pages=args.max_check_pages,
                char_threshold=args.char_threshold,
                selected_pages=selected_pages,
                cache_enabled=not args.no_cache,
                database_enabled=not args.no_database,
                metadata=metadata,
            )
            if (
                selected_pages is None
                and not args.no_database
                and outcome.article_count > 0
                and outcome.complete
            ):
                state_db.mark_processed(
                    sha256=file_sha,
                    filename=pdf_path.name,
                    size=pdf_path.stat().st_size,
                    source_path=str(pdf_path.resolve()),
                    publication_type=metadata.publication_type,
                    publication_date=metadata.publication_date,
                    pdf_id=outcome.pdf_id,
                    output_path=outcome.target_dir.relative_to(output_dir).as_posix(),
                    article_count=outcome.article_count,
                )
                print(f"📚 已写入抓取记录库：{state_db.db_path}")
            elif selected_pages is None and not outcome.complete:
                failed = ", ".join(str(page) for page in outcome.failed_pages) or "未知"
                print(f"⚠️ 本次仍有失败页（{failed}），不写入完成状态，下次将继续续跑。")
            elif selected_pages is None and outcome.article_count <= 0:
                print("⚠️ 本次未生成有效文章，不写入抓取记录库，下次仍会重试。")
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            print(f"❌ 解析失败：{pdf_path.name}，错误：{exc}")

    print("🎉 批处理完成。")


def _existing_output_is_complete(
    output_root: Path,
    metadata: PdfMetadata,
    pdf_path: Path,
) -> tuple[bool, str]:
    cache_dir = (
        output_root
        / metadata.publication_type
        / metadata.publication_date
        / "cache_json"
    )
    cache_files = list(cache_dir.glob("page_*.json")) if cache_dir.exists() else []
    if not cache_files:
        # Legacy or explicit --no-cache outputs have no page checkpoints. Keep
        # trusting their database/state record instead of forcing a full rerun.
        return True, ""

    page_count = _source_pdf_page_count(pdf_path)
    if page_count <= 0:
        return False, "无法确认源 PDF 页数"

    valid_pages: set[int] = set()
    failed_pages: list[int] = []
    pending_compile_pages: list[int] = []
    require_compiled_articles = os.getenv("LLM_COMPILE_ARTICLES", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    for cache_file in cache_files:
        match = re.fullmatch(r"page_(\d+)\.json", cache_file.name)
        if not match:
            continue
        page_number = int(match.group(1))
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failed_pages.append(page_number)
            continue
        if not isinstance(payload, dict) or payload.get("error"):
            failed_pages.append(page_number)
            continue
        if require_compiled_articles:
            articles = payload.get("articles") or []
            if any(
                isinstance(article, dict)
                and str(article.get("content_markdown") or "").strip()
                and not article.get("compiled_article")
                for article in articles
            ):
                pending_compile_pages.append(page_number)
        valid_pages.add(page_number)

    missing_pages = [
        page_number
        for page_number in range(1, page_count + 1)
        if page_number not in valid_pages
    ]
    if failed_pages:
        preview = ",".join(str(page) for page in sorted(set(failed_pages))[:8])
        return False, f"失败缓存页：{preview}"
    if pending_compile_pages:
        preview = ",".join(
            str(page) for page in sorted(set(pending_compile_pages))[:8]
        )
        return False, f"存在未完成翻译/解读的页：{preview}"
    if missing_pages:
        preview = ",".join(str(page) for page in missing_pages[:8])
        suffix = "..." if len(missing_pages) > 8 else ""
        return False, f"缺少 {len(missing_pages)}/{page_count} 个页缓存：{preview}{suffix}"
    return True, ""


def _source_pdf_page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path), strict=False).pages)
    except Exception:
        pass

    try:
        import fitz

        fitz.TOOLS.mupdf_display_errors(False)
        try:
            with fitz.open(pdf_path) as document:
                return len(document)
        finally:
            fitz.TOOLS.mupdf_display_errors(True)
    except Exception:
        return 0


def process_pdf(
    pdf_path: Path,
    output_root: Path,
    max_check_pages: int = 3,
    char_threshold: int = 100,
    selected_pages: set[int] | None = None,
    cache_enabled: bool = True,
    database_enabled: bool = True,
    metadata: PdfMetadata | None = None,
) -> ProcessOutcome:
    print(f"\n📄 正在处理：{pdf_path.name}")
    metadata = metadata or parse_pdf_metadata(str(pdf_path))
    print(f"🏷️ 元数据：{metadata.publication_type} / {metadata.publication_date}")

    target_dir = output_root / metadata.publication_type / metadata.publication_date
    image_dir = target_dir / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    prepared_pdf = prepare_pdf_for_parsing(pdf_path)
    parsing_pdf_path = prepared_pdf.parsing_path
    _write_cover_image(parsing_pdf_path, target_dir / "cover.jpg")
    article_writer = ArticleMarkdownWriter(
        articles_dir=target_dir / "articles",
        publication_type=metadata.publication_type,
        publication_date=metadata.publication_date,
        original_filename=pdf_path.name,
        selected_pages=selected_pages,
    )

    vector_pdf = is_vector_pdf(
        str(parsing_pdf_path),
        max_check_pages=max_check_pages,
        char_threshold=char_threshold,
    )
    strategy = (
        HybridPdfStrategy(cache_enabled=cache_enabled)
        if vector_pdf
        else ScannedPdfStrategy(cache_enabled=cache_enabled)
    )
    pdf_type = "矢量/混合 PDF" if vector_pdf else "纯图片扫描件 PDF"
    print(f"🧭 类型检测：{pdf_type}，解析引擎：{strategy.engine_name}")

    context = ParseContext(
        pdf_path=parsing_pdf_path,
        target_output_dir=target_dir,
        image_dir=image_dir,
        metadata=metadata,
        original_filename=pdf_path.name,
        article_writer=article_writer.write,
        selected_pages=selected_pages,
    )
    try:
        result = strategy.parse(context)
    except Exception as exc:
        if _is_required_article_split_error(exc):
            raise
        print(f"⚠️ 解析引擎异常，仍将基于已提取图片生成 Markdown：{exc}")
        result = ParseResult(
            body_markdown=_render_asset_index_markdown(image_dir, parse_error=exc),
            engine_name=f"{strategy.engine_name}_ImageFallback",
            articles=[],
            complete=False,
        )

    image_mapping = _normalize_image_assets(image_dir)
    body_markdown = _apply_image_mapping_to_markdown(
        result.body_markdown,
        image_mapping,
    )
    raw_articles = (
        _split_markdown_into_articles(body_markdown)
        if result.articles is None
        else result.articles
    )
    articles = _apply_image_mapping_to_articles(raw_articles, image_mapping)

    if selected_pages:
        page_label = "-".join(str(page) for page in sorted(selected_pages))
        output_md = target_dir / f"{pdf_path.stem}.pages-{page_label}.md"
    else:
        output_md = target_dir / f"{pdf_path.stem}.md"
    output_md.write_text(
        _render_final_markdown(
            publication_type=metadata.publication_type,
            publication_date=metadata.publication_date,
            original_filename=pdf_path.name,
            engine_name=result.engine_name,
            body=body_markdown,
        ),
        encoding="utf-8",
    )
    article_paths = article_writer.rewrite_all(articles, result.engine_name)
    if database_enabled and not selected_pages:
        database_path, index_path = write_pdf_database(
            output_root=output_root,
            target_dir=target_dir,
            metadata=metadata,
            original_filename=pdf_path.name,
            articles=articles,
            pages=result.pages,
            front_page=result.front_page,
            is_weekend=result.is_weekend,
        )
        print(f"🗄️ 已导出数据库：{database_path}")
        print(f"🧭 已更新数据库索引：{index_path}")
    elif database_enabled and selected_pages:
        print("🧪 指定页测试不覆盖整期 database.js；完整运行时再统一更新数据库。")
    print(f"✅ 已导出 Markdown：{output_md}")
    print(f"📰 已导出单篇文章：{len(article_paths)} 篇，目录：{target_dir / 'articles'}")
    print(f"🖼️ 图片目录：{image_dir}")
    return ProcessOutcome(
        output_markdown=output_md,
        target_dir=target_dir,
        pdf_id=build_pdf_id(metadata, pdf_path.name),
        article_count=len(articles),
        complete=result.complete,
        failed_pages=result.failed_pages,
    )


def _render_final_markdown(
    publication_type: str,
    publication_date: str,
    original_filename: str,
    engine_name: str,
    body: str,
) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    return f"""# {publication_type} - {publication_date}

> **源文件**：`{original_filename}`
> **解析时间**：`{timestamp}`
> **解析引擎**：`{engine_name}`

---

{body.strip()}
"""


def _normalize_image_assets(image_dir: Path) -> dict[str, str]:
    if not image_dir.exists():
        return {}

    mapping: dict[str, str] = {}
    used_names = {
        image_path.name
        for image_path in image_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    }
    next_fig = 1

    for image_path in sorted(image_dir.iterdir(), key=lambda path: path.name):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if NORMALIZED_IMAGE_RE.match(image_path.name):
            continue

        target_name, next_fig = _next_available_image_name(
            used_names,
            image_path.suffix.lower(),
            next_fig,
        )
        temp_path = image_path.with_name(f".{image_path.name}.normalizing")
        target_path = image_path.with_name(target_name)
        image_path.rename(temp_path)
        temp_path.rename(target_path)

        used_names.remove(image_path.name)
        used_names.add(target_name)
        mapping[f"images/{image_path.name}"] = f"images/{target_name}"

    for old_rel_path, new_rel_path in mapping.items():
        print(f"  🔁 图片规范化：{old_rel_path} -> {new_rel_path}")
    return mapping


def _apply_image_mapping_to_markdown(
    markdown: str,
    image_mapping: dict[str, str],
) -> str:
    for old_rel_path, new_rel_path in image_mapping.items():
        markdown = markdown.replace(old_rel_path, new_rel_path)
    return markdown


def _apply_image_mapping_to_articles(
    articles: list[dict[str, Any]],
    image_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    normalized_articles = []
    for article in articles:
        normalized = dict(article)
        content = normalized.get("content_markdown")
        if isinstance(content, str):
            normalized["content_markdown"] = _apply_image_mapping_to_markdown(
                content,
                image_mapping,
            )

        images = normalized.get("images") or []
        normalized["images"] = [
            image_mapping.get(image, image)
            for image in images
            if isinstance(image, str) and image
        ]
        normalized_articles.append(normalized)
    return normalized_articles


def _next_available_image_name(
    used_names: set[str],
    suffix: str,
    start_at: int,
) -> tuple[str, int]:
    index = start_at
    while True:
        candidate = f"page_1_fig_{index}{suffix}"
        if candidate not in used_names:
            return candidate, index + 1
        index += 1


def _render_asset_index_markdown(
    image_dir: Path,
    parse_error: Exception | None = None,
) -> str:
    image_paths = [
        image_path
        for image_path in sorted(image_dir.iterdir(), key=lambda path: path.name)
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ] if image_dir.exists() else []

    sections = ["## Extracted Image Assets"]
    if parse_error:
        sections.append(f"_正文解析未完成：{parse_error}_")

    if not image_paths:
        sections.append("_未提取到图片资产。_")
    for image_path in image_paths:
        sections.append(f"![]({image_dir.name}/{image_path.name})")

    return "\n\n".join(sections)


def _is_required_article_split_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "不会退回一页一篇" in message
        or "视觉 LLM 也未返回文章列表" in message
        or "无法完成 LLM 文章级拆分" in message
    )


class ArticleMarkdownWriter:
    def __init__(
        self,
        articles_dir: Path,
        publication_type: str,
        publication_date: str,
        original_filename: str,
        selected_pages: set[int] | None = None,
    ) -> None:
        self.articles_dir = articles_dir
        self.publication_type = publication_type
        self.publication_date = publication_date
        self.original_filename = original_filename
        self.selected_pages = selected_pages
        self.used_names: set[str] = set()
        self.next_index = 1
        self.written_paths: list[Path] = []
        self.cleanup_done = False
        self.log_writes = True

    def write(
        self,
        articles: list[dict[str, Any]],
        engine_name: str,
    ) -> list[Path]:
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        if not articles:
            return []
        self._cleanup_existing_outputs()
        written_now = []

        for article in articles:
            index = self.next_index
            title = str(article.get("title") or f"Article {index}").strip()
            if self.publication_type == "BARRONS" and not self.selected_pages:
                filename = _unique_archive_article_filename(
                    self.publication_date,
                    index,
                    self.used_names,
                )
            else:
                filename = _unique_article_output_filename(article, index, title, self.used_names)
            output_path = self.articles_dir / filename
            output_path.write_text(
                _render_article_markdown(
                    article=article,
                    title=title,
                    index=index,
                    publication_type=self.publication_type,
                    publication_date=self.publication_date,
                    original_filename=self.original_filename,
                    engine_name=engine_name,
                ),
                encoding="utf-8",
            )
            if self.log_writes:
                print(f"  📝 已生成文章 Markdown：{output_path}")
            self.written_paths.append(output_path)
            written_now.append(output_path)
            self.next_index += 1

        return written_now

    def rewrite_all(
        self,
        articles: list[dict[str, Any]],
        engine_name: str,
    ) -> list[Path]:
        """Replace temporary per-page outputs with the final compiled article set."""
        self.used_names.clear()
        self.next_index = 1
        self.written_paths.clear()
        self.cleanup_done = False
        self._cleanup_existing_outputs()
        previous_log_writes = self.log_writes
        self.log_writes = False
        try:
            return self.write(articles, engine_name)
        finally:
            self.log_writes = previous_log_writes

    def _cleanup_existing_outputs(self) -> None:
        if self.cleanup_done:
            return
        self.cleanup_done = True
        if not self.articles_dir.exists():
            return

        if self.selected_pages:
            prefixes = {f"P{page:02d}_" for page in self.selected_pages}
            targets = [
                path
                for path in self.articles_dir.iterdir()
                if any(path.name.startswith(prefix) for prefix in prefixes)
            ]
        else:
            targets = list(self.articles_dir.iterdir())

        for path in targets:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _write_article_markdowns(
    articles: list[dict[str, Any]],
    articles_dir: Path,
    publication_type: str,
    publication_date: str,
    original_filename: str,
    engine_name: str,
) -> list[Path]:
    writer = ArticleMarkdownWriter(
        articles_dir=articles_dir,
        publication_type=publication_type,
        publication_date=publication_date,
        original_filename=original_filename,
    )
    writer.write(articles, engine_name)
    return writer.written_paths


def _render_article_markdown(
    article: dict[str, Any],
    title: str,
    index: int,
    publication_type: str,
    publication_date: str,
    original_filename: str,
    engine_name: str,
) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    category = article.get("category")
    page = article.get("page")
    print_page_label = article.get("print_page_label")
    print_section = article.get("print_section")
    images = article.get("images") or []
    title_zh = str(article.get("title_zh") or "").strip()
    summary_md = str(article.get("summary_md") or "").strip()
    content = str(article.get("content_markdown") or "").strip()
    content = _rewrite_article_relative_image_refs(content)

    metadata_lines = [
        f"> **源文件**：`{original_filename}`  ",
        f"> **报刊**：`{publication_type}`  ",
        f"> **出版日期**：`{publication_date}`  ",
        f"> **解析时间**：`{timestamp}`  ",
        f"> **解析引擎**：`{engine_name}`  ",
        f"> **文章序号**：`{index}`  ",
    ]
    if page is not None:
        metadata_lines.append(f"> **PDF 页码**：`{page}`  ")
    if print_page_label:
        metadata_lines.append(f"> **印刷版号**：`{print_page_label}`  ")
    if print_section:
        metadata_lines.append(f"> **版面栏目**：`{print_section}`  ")
    if category:
        metadata_lines.append(f"> **分类**：`{category}`  ")
    if title_zh:
        metadata_lines.append(f"> **中文标题**：{title_zh}  ")

    sections = [f"# {title}", "\n".join(metadata_lines), "---"]
    if summary_md:
        sections.append("## 中文解读\n\n" + summary_md)
    for image in images:
        sections.append(f"![{title}]({_article_relative_image_path(image)})")
    if content:
        if summary_md:
            sections.append("## English Article")
        sections.append(content)

    return "\n\n".join(sections).strip() + "\n"


def _write_cover_image(pdf_path: Path, output_path: Path) -> None:
    fitz = None
    try:
        import fitz

        fitz.TOOLS.mupdf_display_errors(False)
        with fitz.open(pdf_path) as document:
            if not document:
                return
            scale = 130 / 72
            pixmap = document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pixmap.save(output_path)
        print(f"  🗞️ 已生成本期封面：{output_path}")
    except Exception as exc:
        print(f"  ⚠️ 封面生成失败，将使用第 1 页文章图片：{exc}")
    finally:
        if fitz is not None:
            fitz.TOOLS.mupdf_display_errors(True)


def _unique_article_filename(
    index: int,
    title: str,
    used_names: set[str],
) -> str:
    slug = _slugify(title) or "article"
    base_name = f"{index:03d}-{slug}.md"
    candidate = base_name
    suffix = 2
    while candidate in used_names:
        candidate = f"{index:03d}-{slug}-{suffix}.md"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _unique_article_output_filename(
    article: dict[str, Any],
    index: int,
    title: str,
    used_names: set[str],
) -> str:
    page = _safe_int(article.get("page"), 0)
    page_article_index = _safe_int(article.get("page_article_index"), index)
    safe_title = _safe_article_title(title)
    candidate = f"P{page:02d}_{page_article_index:02d}_{safe_title}.md"
    suffix = 2
    while candidate in used_names:
        candidate = f"P{page:02d}_{page_article_index:02d}_{safe_title}_{suffix}.md"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _unique_archive_article_filename(
    publication_date: str,
    index: int,
    used_names: set[str],
) -> str:
    base_name = f"art_{publication_date}_{index:03d}"
    candidate = f"{base_name}.md"
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}.md"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_article_title(title: str) -> str:
    safe_title = "".join(
        char for char in title if char.isalnum() or char in (" ", "_", "-")
    ).strip()
    return safe_title[:30] or "Article"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80].strip("-")


def _article_relative_image_path(image_path: str) -> str:
    if image_path.startswith("../"):
        return image_path
    if image_path.startswith("images/"):
        return f"../{image_path}"
    return image_path


def _rewrite_article_relative_image_refs(markdown: str) -> str:
    markdown = markdown.replace("(images/", "(../images/")
    markdown = markdown.replace('src="images/', 'src="../images/')
    markdown = markdown.replace("src='images/", "src='../images/")
    return markdown


def _split_markdown_into_articles(body_markdown: str) -> list[dict[str, Any]]:
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+?)\s*$", flags=re.MULTILINE)
    matches = list(heading_pattern.finditer(body_markdown))
    if not matches:
        return [
            {
                "title": "Extracted Content",
                "category": "Markdown Fallback",
                "content_markdown": body_markdown.strip(),
                "images": [],
            }
        ] if body_markdown.strip() else []

    articles = []
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body_markdown)
        content = body_markdown[start:end].strip()
        articles.append(
            {
                "title": title,
                "category": "Markdown Heading",
                "content_markdown": content,
                "images": [],
            }
        )
    return articles


def _normalize_file_selectors(raw_selectors: list[str] | None) -> list[str]:
    selectors = []
    for raw in raw_selectors or []:
        selectors.extend(part.strip() for part in raw.split(",") if part.strip())
    return selectors


def _matches_file_selector(path: Path, selectors: list[str]) -> bool:
    filename = path.name.lower()
    stem = path.stem.lower()
    for selector in selectors:
        lowered = selector.lower()
        if filename == lowered or stem == lowered:
            return True
        if fnmatch.fnmatch(filename, lowered) or fnmatch.fnmatch(stem, lowered):
            return True
    return False


def _print_processing_status(state_db: ProcessingStateDB) -> None:
    records = state_db.list_all()
    print(f"📚 抓取记录库：{state_db.db_path}，共 {len(records)} 份 PDF")
    for record in records:
        processed_at = datetime.fromtimestamp(record["processed_at"]).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"  ✅ {record['publication_type']:<10} {record['publication_date']}  "
            f"{record['filename']}  {record['article_count']} 篇  {processed_at}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通用报纸 PDF 自动化批量解析系统",
    )
    parser.add_argument(
        "--input-dir",
        default=os.getenv("INPUT_DIR", "./input_pdfs"),
        help="待解析 PDF 输入目录",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "./output_results"),
        help="统一输出目录",
    )
    parser.add_argument(
        "--max-check-pages",
        type=int,
        default=3,
        help="PDF 类型探测读取页数",
    )
    parser.add_argument(
        "--char-threshold",
        type=int,
        default=100,
        help="判定矢量 PDF 的原生文本字符阈值",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="仅处理指定页，支持 2 或 2-3,9",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="忽略已有 cache_json，强制重新解析",
    )
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="只生成 Markdown，不写 database.js/database_index.js",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="只抓取指定文件；可重复传入，支持文件名、stem 和通配符",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略抓取记录库，强制重新解析",
    )
    parser.add_argument(
        "--publication",
        default=None,
        help="覆盖报刊类型，用于自定义来源目录，例如 Economist",
    )
    parser.add_argument(
        "--state-db",
        default=os.getenv("STATE_DB_PATH", "./.state/processed_pdfs.sqlite3"),
        help="SQLite 抓取记录库路径",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="显示已抓取 PDF 记录后退出",
    )
    return parser.parse_args()


def _parse_pages_arg(raw_pages: str | None) -> set[int] | None:
    if not raw_pages:
        return None

    pages: set[int] = set()
    for part in raw_pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))

    return pages or None


if __name__ == "__main__":
    main()
