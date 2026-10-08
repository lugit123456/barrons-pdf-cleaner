from __future__ import annotations

import io
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import wordninja
from print_layout import (
    PRINT_LAYOUT_VERSION,
    build_front_page,
    build_issue_pages,
    enrich_page_result,
    select_header_blocks,
    select_front_page_directory_blocks,
)

from .base import ParseContext, ParseResult
from .special_pages import (
    PAGE_TYPE_BUSINESS,
    PAGE_TYPE_CONTENTS,
    PAGE_TYPE_NORMAL,
    PAGE_TYPE_POLITICS,
    PAGE_TYPE_UTILITY,
    build_contents_page_result,
    build_utility_page_result,
    build_native_world_page_result,
    cache_matches_page_type,
    classify_page_heading,
    classify_native_page,
)
from .strategy_scanned import (
    ScannedPdfStrategy,
    _chat_completion_json,
    _clean_articles,
    _compact_error,
    _compile_articles,
    _env_bool,
    _flatten_page_articles,
    _pending_compile_pages,
    _render_pages_to_markdown,
    retry_deferred_compiles,
)


CACHE_VERSION = "hybrid_native_blocks_v2"
FALLBACK_CACHE_VERSION = "hybrid_page_fallback_v1"


class HybridPdfStrategy:
    """Article-level parser for PDFs containing a usable native text layer."""

    engine_name = "Strategy_A_NativeText_LLM"

    def __init__(
        self,
        cache_enabled: bool | None = None,
        max_retries: int | None = None,
    ) -> None:
        load_dotenv()
        self.cache_enabled = (
            _env_bool("ENABLE_JSON_CACHE", True)
            if cache_enabled is None
            else cache_enabled
        )
        self.max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.scanned_fallback = ScannedPdfStrategy(
            cache_enabled=self.cache_enabled,
            max_retries=self.max_retries,
        )

    def parse(self, context: ParseContext) -> ParseResult:
        import fitz

        context.target_output_dir.mkdir(parents=True, exist_ok=True)
        context.image_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = context.target_output_dir / "cache_json"
        if self.cache_enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)

        page_results: list[dict[str, Any]] = []
        prior_articles: list[dict[str, Any]] = []
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.reset_mupdf_warnings()
        try:
            with fitz.open(context.pdf_path) as document:
                for page_number in range(1, len(document) + 1):
                    if context.selected_pages and page_number not in context.selected_pages:
                        continue
                    try:
                        page_result = self._process_page(
                            document=document,
                            page_number=page_number,
                            image_dir=context.image_dir,
                            cache_dir=cache_dir,
                            prior_articles=prior_articles[-60:],
                            publication_type=context.metadata.publication_type,
                        )
                    except Exception as exc:
                        print(
                            f"  ⚠️ 第 {page_number} 页原生 PDF 结构异常，"
                            f"跳过并继续后续页面：{_compact_error(exc)}"
                        )
                        page_result = {
                            "page": page_number,
                            "articles": [],
                            "error": str(exc),
                            "parser": CACHE_VERSION,
                        }
                    page_results.append(page_result)
                    for article in page_results[-1].get("articles") or []:
                        if isinstance(article, dict) and article.get("title"):
                            prior_articles.append(
                                {"page": page_number, "title": str(article["title"])}
                            )
        finally:
            fitz.TOOLS.mupdf_display_errors(True)

        articles = _flatten_page_articles(page_results)
        articles = _merge_cross_page_continuations(articles)
        articles = _clean_articles(articles)
        articles = _reindex_articles(articles)
        articles = _compile_articles_in_order(articles, image_dir=context.image_dir)
        compiled_pages = _group_articles_by_page(articles)
        self._persist_compiled_page_caches(
            cache_dir=cache_dir,
            page_results=page_results,
            compiled_pages=compiled_pages,
        )
        checkpoint_pages = {
            int(page.get("page") or 0): page
            for page in compiled_pages
        }

        def checkpoint_article(_: int, article: dict[str, Any]) -> None:
            if not self.cache_enabled:
                return
            page_number = int(article.get("page") or 0)
            page_article_index = int(article.get("page_article_index") or 0)
            page = checkpoint_pages.get(page_number)
            if page is None or page_article_index <= 0:
                return
            page_articles = page.get("articles") or []
            if page_article_index > len(page_articles):
                return
            page_articles[page_article_index - 1] = article
            page["articles"] = page_articles
            source_page = next(
                (
                    item
                    for item in page_results
                    if int(item.get("page") or 0) == page_number
                ),
                {"page": page_number, "parser": CACHE_VERSION},
            )
            updated = dict(source_page)
            updated["articles"] = page_articles
            self._write_cache(cache_dir / f"page_{page_number}.json", updated)

        articles = retry_deferred_compiles(
            articles,
            image_dir=context.image_dir,
            on_article_updated=checkpoint_article,
        )
        articles = _reindex_articles(articles)
        compiled_pages = _group_articles_by_page(articles)
        self._persist_compiled_page_caches(
            cache_dir=cache_dir,
            page_results=page_results,
            compiled_pages=compiled_pages,
        )

        if context.article_writer:
            context.article_writer(articles, self.engine_name)

        failed_page_numbers = {
            int(page.get("page") or 0)
            for page in page_results
            if page.get("error") and int(page.get("page") or 0) > 0
        }
        failed_page_numbers.update(_pending_compile_pages(articles))
        failed_pages = tuple(sorted(failed_page_numbers))

        return ParseResult(
            body_markdown=_render_pages_to_markdown(compiled_pages),
            engine_name=self.engine_name,
            articles=articles,
            pages=build_issue_pages(page_results),
            front_page=build_front_page(
                page_results, context.metadata.publication_type
            ),
            is_weekend=any(
                int(page.get("page") or 0) == 1
                and page.get("is_weekend") is True
                for page in page_results
            ),
            complete=not failed_pages,
            failed_pages=failed_pages,
        )

    def _persist_compiled_page_caches(
        self,
        cache_dir: Path,
        page_results: list[dict[str, Any]],
        compiled_pages: list[dict[str, Any]],
    ) -> None:
        if not self.cache_enabled:
            return
        compiled_by_page = {
            int(page.get("page") or 0): page.get("articles") or []
            for page in compiled_pages
        }
        for page_result in page_results:
            page_number = int(page_result.get("page") or 0)
            if page_number <= 0 or page_result.get("error"):
                continue
            updated = dict(page_result)
            updated["articles"] = compiled_by_page.get(page_number, [])
            self._write_cache(cache_dir / f"page_{page_number}.json", updated)

    def _process_page(
        self,
        document: Any,
        page_number: int,
        image_dir: Path,
        cache_dir: Path,
        prior_articles: list[dict[str, Any]],
        publication_type: str,
    ) -> dict[str, Any]:
        cache_file = cache_dir / f"page_{page_number}.json"
        page = document[page_number - 1]
        page_type = _classify_native_page_header(page)
        cached = self._load_cache(
            cache_file,
            image_dir,
            expected_page_type=page_type,
            publication_type=publication_type,
        )
        if cached is not None:
            needs_metadata = (
                cached.get("print_layout_version") != PRINT_LAYOUT_VERSION
                or any(
                    key not in cached
                    for key in (
                        "print_page_label",
                        "print_section",
                        "print_page_source",
                    )
                )
            )
            if needs_metadata:
                metadata_blocks, _ = _extract_native_text_blocks(page)
                directory_blocks = _native_whats_news_blocks(
                    metadata_blocks,
                    publication_type,
                    page_number,
                    float(page.rect.width),
                    float(page.rect.height),
                )
                cached = enrich_page_result(
                    cached,
                    publication_type,
                    page_number,
                    header_blocks=select_header_blocks(
                        metadata_blocks, float(page.rect.height)
                    ),
                    whats_news_blocks=metadata_blocks,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                )
                cached["articles"] = _filter_articles_from_block_region(
                    cached.get("articles") or [], directory_blocks
                )
                self._write_cache(cache_file, cached)
            print(f"⚡ 第 {page_number} 页读取原生文字结构缓存。")
            return cached

        blocks, stats = _extract_native_text_blocks(page)
        directory_blocks = _native_whats_news_blocks(
            blocks,
            publication_type,
            page_number,
            float(page.rect.width),
            float(page.rect.height),
        )
        excluded_objects = {id(block) for block in directory_blocks}
        article_blocks = [block for block in blocks if id(block) not in excluded_objects]

        def with_layout(
            result: dict[str, Any],
            *,
            existing_metadata: str = "replace",
        ) -> dict[str, Any]:
            return enrich_page_result(
                result,
                publication_type,
                page_number,
                header_blocks=select_header_blocks(blocks, float(page.rect.height)),
                whats_news_blocks=blocks,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                existing_metadata=existing_metadata,
            )

        page_type = classify_native_page(
            blocks,
            float(page.rect.height),
            publication_type,
        )
        is_barrons = str(publication_type or "").strip().upper() == "BARRONS"
        if page_type == PAGE_TYPE_CONTENTS:
            print(f"  ⏭️ 第 {page_number} 页识别为 Contents，跳过目录页。")
            result = build_contents_page_result(page_number, CACHE_VERSION)
            result["source_stats"] = stats
            result = with_layout(result)
            self._write_cache(cache_file, result)
            return result

        if page_type == PAGE_TYPE_UTILITY:
            print(f"  ⏭️ 第 {page_number} 页识别为 Barron's Index/Data，跳过非正文页。")
            result = build_utility_page_result(page_number, CACHE_VERSION)
            result["source_stats"] = stats
            result = with_layout(result)
            self._write_cache(cache_file, result)
            return result

        if (
            is_barrons
            and int(stats.get("words") or 0)
            < int(os.getenv("BARRONS_MIN_ARTICLE_PAGE_WORDS", "300"))
        ):
            result = with_layout({
                "page": page_number,
                "page_type": PAGE_TYPE_UTILITY,
                "parser": CACHE_VERSION,
                "source_stats": stats,
                "skipped": True,
                "skip_reason": "Barron's short non-article page",
                "articles": [],
            })
            print(
                f"  ⏭️ 第 {page_number} 页不足 Barron's 正文页词数门槛，"
                "跳过非正文页。"
            )
            self._write_cache(cache_file, result)
            return result

        if not _has_usable_native_text(stats):
            print(
                f"  🔀 第 {page_number} 页原生文字层质量不足"
                f"（chars={stats['chars']}, words={stats['words']}），切换 OCR/视觉解析。"
            )
            result = self._run_scanned_fallback(
                page,
                page_number,
                image_dir,
                cache_dir,
                publication_type,
            )
            result["parser"] = FALLBACK_CACHE_VERSION
            result["source_stats"] = stats
            result = with_layout(result, existing_metadata="merge")
            self._write_cache(cache_file, result)
            return result

        images = _extract_page_images(document, page, page_number, image_dir)
        print(
            f"  🧾 第 {page_number} 页读取原生文字："
            f"{stats['blocks']} blocks / {stats['words']} words；提取 {len(images)} 张原图。"
        )

        if page_type in {PAGE_TYPE_POLITICS, PAGE_TYPE_BUSINESS}:
            label = "Politics" if page_type == PAGE_TYPE_POLITICS else "Business"
            print(f"  🧩 第 {page_number} 页识别为 The world this week / {label}，整页合并为一篇。")
            result = build_native_world_page_result(
                page_number=page_number,
                page_type=page_type,
                parser=CACHE_VERSION,
                blocks=article_blocks,
                images=images,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                source_stats=stats,
            )
            result = with_layout(result)
            self._write_cache(cache_file, result)
            return result

        articles: list[dict[str, Any]] = []
        if is_barrons and _env_bool("BARRONS_LOCAL_SPLIT_FIRST", True):
            articles = _build_local_layout_articles(
                page_number=page_number,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                blocks=article_blocks,
                images=images,
                publication_type=publication_type,
                prior_articles=prior_articles,
            )
            if articles:
                print(
                    f"  🧩 第 {page_number} 页使用 Barron's native 坐标规则"
                    "直接恢复单篇版面。"
                )

        if not articles:
            try:
                articles = self._identify_native_articles(
                    page_number=page_number,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                    blocks=article_blocks,
                    images=images,
                    prior_articles=prior_articles,
                )
            except Exception as exc:
                print(f"  ⚠️ 第 {page_number} 页原生文字文章识别失败：{_compact_error(exc)}")

        if (
            not articles
            and is_barrons
        ):
            articles = _build_local_layout_articles(
                page_number=page_number,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                blocks=article_blocks,
                images=images,
                publication_type=publication_type,
                prior_articles=prior_articles,
            )
            if articles:
                print(
                    f"  🧩 第 {page_number} 页使用 Barron's native 坐标规则"
                    f"恢复 {len(articles)} 篇候选文章。"
                )

        if not articles and _env_bool("HYBRID_ENABLE_MINERU_FALLBACK", True):
            articles = self._run_mineru_fallback(
                document=document,
                page_number=page_number,
                image_dir=image_dir,
            )

        if not articles:
            articles = _build_local_layout_articles(
                page_number=page_number,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                blocks=article_blocks,
                images=images,
                publication_type=publication_type,
                prior_articles=prior_articles,
            )
            if articles:
                print(f"  🧩 第 {page_number} 页使用本地坐标规则恢复 {len(articles)} 篇候选文章。")

        if not articles:
            print(f"  🔀 第 {page_number} 页结构识别无结果，切换 OCR/视觉解析。")
            result = self._run_scanned_fallback(
                page,
                page_number,
                image_dir,
                cache_dir,
                publication_type,
            )
            result["parser"] = FALLBACK_CACHE_VERSION
            result["source_stats"] = stats
            result = with_layout(result, existing_metadata="merge")
            self._write_cache(cache_file, result)
            return result

        articles = _filter_articles_from_block_region(articles, directory_blocks)
        result = with_layout({
            "page": page_number,
            "parser": CACHE_VERSION,
            "source_stats": stats,
            "articles": articles,
        })
        self._write_cache(cache_file, result)
        return result

    def _identify_native_articles(
        self,
        page_number: int,
        page_width: float,
        page_height: float,
        blocks: list[dict[str, Any]],
        images: list[dict[str, Any]],
        prior_articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = _build_native_article_prompt(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            blocks=blocks,
            images=images,
            prior_articles=prior_articles,
        )
        print(f"  🤖 正在调用 LLM 识别第 {page_number} 页原生文字文章结构...")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                data = _chat_completion_json(
                    operation=f"第 {page_number} 页原生文字文章识别",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=int(os.getenv("LLM_NATIVE_SPLIT_MAX_TOKENS", "6000")),
                    max_retries=1,
                )
                articles = _assemble_native_articles(data, blocks, images, page_number)
                if articles:
                    return articles
                raise ValueError("LLM 未返回带有效 content_block_ids 的文章")
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    print(
                        f"  🔁 第 {page_number} 页原生文字识别第 "
                        f"{attempt}/{self.max_retries} 次失败，准备重试：{_compact_error(exc)}"
                    )
                    time.sleep(min(attempt * 2, 6))
        raise RuntimeError(str(last_error) if last_error else "原生文字文章识别失败")

    def _run_mineru_fallback(
        self,
        document: Any,
        page_number: int,
        image_dir: Path,
    ) -> list[dict[str, Any]]:
        try:
            import fitz

            from .strategy_vector import parse_pdf_bytes_to_markdown

            single_page = fitz.open()
            single_page.insert_pdf(
                document,
                from_page=page_number - 1,
                to_page=page_number - 1,
            )
            pdf_bytes = single_page.tobytes()
            single_page.close()
            print(f"  🧭 第 {page_number} 页改用 MinerU 版面分析作为降级方案...")
            markdown = parse_pdf_bytes_to_markdown(pdf_bytes, image_dir)
            articles = _articles_from_mineru_markdown(markdown, page_number)
            if articles:
                print(f"  🧭 MinerU 为第 {page_number} 页恢复 {len(articles)} 篇候选文章。")
            return articles
        except Exception as exc:
            print(f"  ⚠️ 第 {page_number} 页 MinerU 降级失败：{_compact_error(exc)}")
            return []

    def _run_scanned_fallback(
        self,
        page: Any,
        page_number: int,
        image_dir: Path,
        cache_dir: Path,
        publication_type: str,
    ) -> dict[str, Any]:
        page_image = _render_page_to_pil(page)
        return self.scanned_fallback.process_page_image(
            page_number,
            page_image,
            image_dir,
            cache_dir / "ocr_fallback",
            publication_type=publication_type,
        )

    def _load_cache(
        self,
        cache_file: Path,
        image_dir: Path,
        expected_page_type: str = PAGE_TYPE_NORMAL,
        publication_type: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.cache_enabled or not cache_file.exists():
            return None
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if cached.get("parser") not in {CACHE_VERSION, FALLBACK_CACHE_VERSION}:
            return None
        if cached.get("error"):
            cache_file.unlink(missing_ok=True)
            return None
        if (
            str(publication_type or "").strip().upper() == "BARRONS"
            and cached.get("empty_page")
            and int((cached.get("source_stats") or {}).get("words") or 0) >= 80
        ):
            print(
                f"♻️ 第 {cached.get('page') or '?'} 页旧缓存遗漏 Barron's 正文，重建本页。"
            )
            return None
        if not cache_matches_page_type(cached, expected_page_type):
            print(f"♻️ 第 {cached.get('page') or '?'} 页页面规则已更新，重建本页缓存。")
            return None
        if not _cached_images_exist(cached, image_dir):
            return None
        return cached

    def _write_cache(self, cache_file: Path, result: dict[str, Any]) -> None:
        if not self.cache_enabled or result.get("error"):
            return
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        temp_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_file.replace(cache_file)


def _extract_native_text_blocks(page: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    page_dict = page.get_text("dict", sort=True)
    blocks: list[dict[str, Any]] = []
    font_sizes: list[float] = []

    for raw_block in page_dict.get("blocks") or []:
        if raw_block.get("type") != 0:
            continue
        lines: list[str] = []
        block_sizes: list[float] = []
        fonts: list[str] = []
        for raw_line in raw_block.get("lines") or []:
            line_text = "".join(
                str(span.get("text") or "")
                for span in raw_line.get("spans") or []
            ).strip()
            if line_text:
                lines.append(line_text)
            for span in raw_line.get("spans") or []:
                size = float(span.get("size") or 0)
                if size > 0:
                    block_sizes.append(size)
                    font_sizes.append(size)
                font = str(span.get("font") or "").strip()
                if font:
                    fonts.append(font)
        text = "\n".join(lines).strip()
        if not text:
            continue
        bbox = [round(float(value), 2) for value in raw_block.get("bbox", (0, 0, 0, 0))]
        block_id = f"b{len(blocks) + 1:03d}"
        blocks.append(
            {
                "id": block_id,
                "bbox": bbox,
                "text": text,
                "max_font_size": round(max(block_sizes, default=0), 2),
                "median_font_size": round(statistics.median(block_sizes), 2) if block_sizes else 0,
                "bold": any(re.search(r"bold|black|semibold|demi", font, re.I) for font in fonts),
            }
        )

    combined = "\n".join(block["text"] for block in blocks)
    printable = [char for char in combined if char.isprintable() and not char.isspace()]
    alpha_numeric = [char for char in printable if char.isalnum()]
    replacement_count = combined.count("\ufffd")
    stats = {
        "blocks": len(blocks),
        "chars": len(printable),
        "words": len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", combined)),
        "alpha_numeric_ratio": round(len(alpha_numeric) / max(len(printable), 1), 4),
        "replacement_ratio": round(replacement_count / max(len(printable), 1), 4),
        "median_font_size": round(statistics.median(font_sizes), 2) if font_sizes else 0,
    }
    return blocks, stats


def _classify_native_page_header(page: Any) -> str:
    try:
        header_text = page.get_text(
            "text",
            clip=(0, 0, float(page.rect.width), min(float(page.rect.height), 110.0)),
            sort=True,
        )
    except Exception:
        return PAGE_TYPE_NORMAL
    return classify_page_heading(str(header_text or ""))


def _has_usable_native_text(stats: dict[str, Any]) -> bool:
    return (
        int(stats.get("chars") or 0) >= int(os.getenv("NATIVE_PAGE_MIN_CHARS", "200"))
        and int(stats.get("words") or 0) >= int(os.getenv("NATIVE_PAGE_MIN_WORDS", "40"))
        and float(stats.get("alpha_numeric_ratio") or 0) >= float(os.getenv("NATIVE_MIN_ALNUM_RATIO", "0.55"))
        and float(stats.get("replacement_ratio") or 0) <= float(os.getenv("NATIVE_MAX_REPLACEMENT_RATIO", "0.02"))
    )


def _extract_page_images(
    document: Any,
    page: Any,
    page_number: int,
    image_dir: Path,
) -> list[dict[str, Any]]:
    from PIL import Image

    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    min_area_ratio = float(os.getenv("NATIVE_IMAGE_MIN_AREA_RATIO", "0.0025"))
    min_width = float(os.getenv("NATIVE_IMAGE_MIN_WIDTH", "45"))
    min_height = float(os.getenv("NATIVE_IMAGE_MIN_HEIGHT", "30"))
    images: list[dict[str, Any]] = []
    seen_xrefs: set[int] = set()

    try:
        image_infos = page.get_images(full=True)
    except Exception as exc:
        print(f"  ⚠️ 第 {page_number} 页图片列表读取失败：{_compact_error(exc)}")
        return images

    for image_info in image_infos:
        xref = int(image_info[0])
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            rects = [rect for rect in page.get_image_rects(xref) if not rect.is_empty]
        except Exception as exc:
            print(f"  ⚠️ 第 {page_number} 页图片对象 {xref} 坐标读取失败：{_compact_error(exc)}")
            continue
        if not rects:
            continue
        rect = max(rects, key=lambda value: value.width * value.height)
        if rect.width < min_width or rect.height < min_height:
            continue
        if (rect.width * rect.height) / page_area < min_area_ratio:
            continue

        image_index = len(images) + 1
        filename = f"page_{page_number}_fig_{image_index}.jpg"
        output_path = image_dir / filename
        pixel_width = 0
        pixel_height = 0
        try:
            extracted = document.extract_image(xref)
            source = Image.open(io.BytesIO(extracted["image"]))
            source.load()
            if source.mode in {"RGBA", "LA"}:
                rgba = source.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
            else:
                rgb = source.convert("RGB")
            rgb.save(output_path, format="JPEG", quality=95, optimize=True)
            pixel_width, pixel_height = int(rgb.width), int(rgb.height)
        except Exception as exc:
            try:
                import fitz

                dpi = max(int(os.getenv("NATIVE_IMAGE_FALLBACK_DPI", "180")), 72)
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(dpi / 72, dpi / 72),
                    clip=rect,
                    alpha=False,
                )
                pixmap.save(output_path)
                pixel_width, pixel_height = int(pixmap.width), int(pixmap.height)
                print(
                    f"  🩹 第 {page_number} 页图片对象 {xref} 无法直接导出，"
                    "已改用页面区域栅格化。"
                )
            except Exception as fallback_exc:
                print(
                    f"  ⚠️ 第 {page_number} 页图片对象 {xref} 导出失败："
                    f"{_compact_error(exc)}；栅格回退也失败：{_compact_error(fallback_exc)}"
                )
                continue

        images.append(
            {
                "id": f"img{image_index:02d}",
                "bbox": [round(float(value), 2) for value in rect],
                "rel_path": f"images/{filename}",
                "pixel_width": pixel_width,
                "pixel_height": pixel_height,
            }
        )
    return images


def _render_page_to_pil(page: Any) -> Any:
    import fitz
    from PIL import Image

    dpi = int(os.getenv("HYBRID_FALLBACK_DPI", "180"))
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _build_native_article_prompt(
    page_number: int,
    page_width: float,
    page_height: float,
    blocks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    prior_articles: list[dict[str, Any]],
) -> str:
    prompt_blocks = [
        {
            "id": block["id"],
            "bbox": block["bbox"],
            "max_font_size": block["max_font_size"],
            "bold": block["bold"],
            "text": block["text"],
        }
        for block in blocks
    ]
    prompt_images = [
        {"id": image["id"], "bbox": image["bbox"], "path": image["rel_path"]}
        for image in images
    ]
    return f"""
Analyze newspaper page {page_number} using native PDF text blocks and their coordinates.
The coordinate origin is the top-left. Page size: {page_width:.1f} x {page_height:.1f}.

TEXT BLOCKS:
{json.dumps(prompt_blocks, ensure_ascii=False)}

IMAGES:
{json.dumps(prompt_images, ensure_ascii=False)}

PRIOR ARTICLE TITLE CANDIDATES FROM EARLIER PAGES:
{json.dumps(prior_articles, ensure_ascii=False)}

Identify every complete or continued reporting article on this page.
Use layout coordinates, font size, headings, bylines, column alignment, and meaning to determine article boundaries and reading order.

Rules:
- Do not copy or rewrite article bodies in the response.
- Return block IDs only. The program will assemble exact source text locally.
- content_block_ids must be in newspaper reading order and must exclude the main title block.
- Include subtitle, deck, byline and body blocks when they belong to the article.
- A body may continue through several narrow columns. Keep all its column blocks in order.
- Keep separate news articles separate, even when they share a page.
- Mark a continuation with is_continuation=true. Match continuation_of to one of the prior article titles when the page says "Continued from..." or uses a shortened continuation headline.
- Assign only image IDs that genuinely belong to that article.
- Exclude mastheads, market tickers, What's News or FT Briefing summaries, tables, ads, service listings, corrections, puzzles, weather, photo-only captions, and page-reference teasers.
- Exclude market data, FINANCIAL TIMES SHARE SERVICE, and MANAGED FUNDS SERVICE.
- Return valid JSON only. Do not return Markdown fences, explanations, or <think> blocks.

Return JSON exactly in this shape:
{{
  "articles": [
    {{
      "title": "Article title",
      "category": "Section or desk",
      "title_block_ids": ["b001"],
      "content_block_ids": ["b002", "b003"],
      "image_ids": ["img01"],
      "is_continuation": false,
      "continuation_of": ""
    }}
  ]
}}
"""


def _assemble_native_articles(
    data: Any,
    blocks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    page_number: int,
) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        raw_articles = data.get("articles", data.get("items", []))
    elif isinstance(data, list):
        raw_articles = data
    else:
        raw_articles = []

    block_map = {block["id"]: block for block in blocks}
    image_map = {image["id"]: image["rel_path"] for image in images}
    assembled: list[dict[str, Any]] = []

    for raw_article in raw_articles:
        if not isinstance(raw_article, dict):
            continue
        title_ids = _normalize_id_list(raw_article.get("title_block_ids"), block_map)
        content_ids = _normalize_id_list(
            raw_article.get("content_block_ids")
            or raw_article.get("body_block_ids")
            or raw_article.get("block_ids"),
            block_map,
        )
        title = re.sub(r"\s+", " ", str(raw_article.get("title") or "")).strip()
        if not title and title_ids:
            title = " ".join(block_map[block_id]["text"] for block_id in title_ids)
            title = re.sub(r"\s+", " ", title).strip()
        if not title or not content_ids:
            continue

        title_key = _normalize_title(title)
        content_parts = []
        retained_ids = []
        for block_id in content_ids:
            block_text = block_map[block_id]["text"].strip()
            if not block_text:
                continue
            if _normalize_title(block_text) == title_key:
                continue
            content_parts.append(_flatten_block_lines(block_text))
            retained_ids.append(block_id)
        content = "\n\n".join(part for part in content_parts if part).strip()
        if not content:
            continue

        requested_images = raw_article.get("image_ids") or raw_article.get("images") or []
        article_images = []
        for image_id in requested_images if isinstance(requested_images, list) else []:
            value = str(image_id).strip()
            path = image_map.get(value, value if value.startswith("images/") else "")
            if path and path not in article_images:
                article_images.append(path)

        assembled.append(
            {
                "title": title,
                "category": str(raw_article.get("category") or "News").strip() or "News",
                "content_markdown": content,
                "images": article_images,
                "source_block_ids": retained_ids,
                "title_block_ids": title_ids,
                "is_continuation": bool(raw_article.get("is_continuation")),
                "continuation_of": str(raw_article.get("continuation_of") or "").strip(),
                "page": page_number,
            }
        )

    if len(assembled) == 1 and images and not assembled[0]["images"]:
        assembled[0]["images"] = [image["rel_path"] for image in images]
    return assembled


def _native_whats_news_blocks(
    blocks: list[dict[str, Any]],
    publication_type: str,
    page_number: int,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    if (
        str(publication_type or "").strip().upper() not in {"WSJ", "FT"}
        or page_number != 1
    ):
        return []
    return select_front_page_directory_blocks(
        blocks,
        publication_type,
        page_width=page_width,
        page_height=page_height,
    )


def _filter_articles_from_block_region(
    articles: list[dict[str, Any]],
    region_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    region_ids = {
        str(block.get("id") or "").strip()
        for block in region_blocks
        if str(block.get("id") or "").strip()
    }
    if not region_ids:
        return articles
    kept: list[dict[str, Any]] = []
    for article in articles:
        source_ids = {
            str(block_id).strip()
            for block_id in article.get("source_block_ids") or []
            if str(block_id).strip()
        }
        directory_share = len(source_ids & region_ids) / max(1, len(source_ids))
        if source_ids and directory_share >= 0.8:
            print(
                f"  🧹 丢弃非正文文章：{article.get('title') or 'Untitled'}"
                "（头版目录坐标区域）"
            )
            continue
        kept.append(article)
    return kept


def _normalize_id_list(value: Any, valid_items: dict[str, Any]) -> list[str]:
    if isinstance(value, str):
        value = re.findall(r"[A-Za-z]+\d+", value)
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        item_id = str(item).strip()
        if item_id in valid_items and item_id not in result:
            result.append(item_id)
    return result


def _flatten_block_lines(text: str) -> str:
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _articles_from_mineru_markdown(markdown: str, page_number: int) -> list[dict[str, Any]]:
    if not markdown or not markdown.strip():
        return []
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+?)\s*$", flags=re.MULTILINE)
    headings = list(heading_pattern.finditer(markdown))
    if not headings:
        return []

    articles = []
    for index, heading in enumerate(headings):
        title = heading.group(2).strip()
        if re.fullmatch(r"Page\s+\d+", title, flags=re.I):
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        content = markdown[heading.end() : end].strip()
        word_count = len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", content))
        if word_count < 40:
            continue
        image_paths = re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", content)
        articles.append(
            {
                "title": title,
                "category": "MinerU Layout Fallback",
                "content_markdown": content,
                "images": list(dict.fromkeys(image_paths)),
                "page": page_number,
            }
        )
    return articles


def _build_local_layout_articles(
    page_number: int,
    page_width: float,
    page_height: float,
    blocks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    publication_type: str = "",
    prior_articles: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not blocks:
        return []
    if str(publication_type or "").strip().upper() == "BARRONS":
        recovered = _build_barrons_local_article(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            blocks=blocks,
            images=images,
            prior_articles=prior_articles or [],
        )
        return [recovered] if recovered else []
    body_sizes = [
        float(block.get("median_font_size") or 0)
        for block in blocks
        if block.get("median_font_size")
    ]
    body_size = statistics.median(body_sizes) if body_sizes else 8.0
    title_threshold = max(float(os.getenv("NATIVE_LOCAL_TITLE_MIN_SIZE", "11")), body_size * 1.45)
    headings = [
        block for block in blocks
        if _looks_like_heading(block, title_threshold, page_height)
    ]
    if not headings:
        return []

    heading_ids = {heading["id"] for heading in headings}
    assignments: dict[str, list[dict[str, Any]]] = {heading["id"]: [] for heading in headings}
    for block in blocks:
        if block["id"] in heading_ids:
            continue
        candidates = []
        for heading in headings:
            score = _block_assignment_score(block, heading, page_width)
            if score is not None:
                candidates.append((score, heading))
        if candidates:
            _, heading = min(candidates, key=lambda item: item[0])
            assignments[heading["id"]].append(block)

    articles = []
    for heading in headings:
        assigned = assignments[heading["id"]]
        assigned.sort(key=lambda block: (round(float(block["bbox"][0]) / 20), block["bbox"][1]))
        content = "\n\n".join(_flatten_block_lines(block["text"]) for block in assigned).strip()
        if len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", content)) < 50:
            continue
        title = _flatten_block_lines(heading["text"])
        article_images = _nearest_images_for_heading(heading, images, page_width, page_height)
        articles.append(
            {
                "title": title,
                "category": "Native Layout Fallback",
                "content_markdown": content,
                "images": article_images,
                "source_block_ids": [block["id"] for block in assigned],
                "title_block_ids": [heading["id"]],
                "page": page_number,
            }
        )
    return articles


def _build_barrons_local_article(
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    blocks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    prior_articles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Recover a Barron's article when the native split model returns no JSON.

    Barron's often places the headline and the first decorative drop cap in one
    native text block. The generic heading heuristic rejects that mixed block,
    so this fallback uses font size and coordinates to separate it locally.
    """
    complex_sections = {
        "REVIEW & PREVIEW", "MARKET WEEK", "MARKET VIEW", "RETIREMENT MAILBAG",
        "INSIDE SCOOP",
    }
    top_section = ""
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        text = _flatten_block_lines(str(block.get("text") or ""))
        if (
            len(bbox) >= 4
            and float(bbox[1]) < page_height * 0.14
            and float(block.get("median_font_size") or 0) >= 18
            and 1 <= len(text.split()) <= 8
            and text.upper() == text
        ):
            top_section = text
            break
    if top_section in complex_sections:
        return _build_barrons_complex_article(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            blocks=blocks,
            images=images,
            section=top_section,
        )

    headline_candidates: list[tuple[float, dict[str, Any], str]] = []
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            continue
        y0 = float(bbox[1])
        width = float(bbox[2]) - float(bbox[0])
        if y0 < page_height * 0.035 or y0 > page_height * 0.72:
            continue
        median_size = float(block.get("median_font_size") or 0)
        max_size = float(block.get("max_font_size") or 0)
        if (
            width < page_width * 0.18
            or median_size < 20
            or (median_size < 24 and max_size < 40)
        ):
            continue
        title = _barrons_headline_text(str(block.get("text") or ""))
        words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", title)
        if not 2 <= len(words) <= 24:
            continue
        if title.upper() == title and len(words) <= 8:
            continue
        normalized_title = _normalize_title(title)
        if not re.match(r"[A-Za-z0-9]", title) or any(
            phrase in normalized_title
            for phrase in {
                "daily real estate obsession",
                "mansionglobal com newsletters",
            }
        ):
            continue
        score = median_size + width / page_width * 10
        headline_candidates.append((score, block, title))
    if len(headline_candidates) != 1:
        return _build_barrons_continuation_article(
            page_number=page_number,
            page_height=page_height,
            blocks=blocks,
            images=images,
            prior_articles=prior_articles,
        ) if not headline_candidates else None

    _, headline_block, title = max(headline_candidates, key=lambda item: item[0])
    body_blocks = []
    for block in blocks:
        if block is headline_block:
            continue
        bbox = block.get("bbox") or [0, 0, 0, 0]
        text = str(block.get("text") or "").strip()
        words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)
        max_size = float(block.get("max_font_size") or 0)
        width = float(bbox[2]) - float(bbox[0]) if len(bbox) >= 4 else 0
        if (
            len(bbox) < 4
            or (
                len(words) < 35
                and not (
                    len(words) >= 20
                    and max_size >= 40
                    and width <= page_width * 0.24
                )
            )
        ):
            continue
        if _is_barrons_promotion_block(text):
            continue
        if float(bbox[1]) < page_height * 0.035:
            continue
        if float(block.get("median_font_size") or 0) < 7.5:
            continue
        body_blocks.append(block)
    if not body_blocks:
        return None

    body_blocks = _sort_barrons_body_blocks(body_blocks)
    content_parts = _barrons_body_content_parts(body_blocks)
    dropcap = _barrons_leading_dropcap(blocks, body_blocks)
    if dropcap and content_parts:
        content_parts[0] = _prepend_dropcap(content_parts[0], dropcap)
    content = "\n\n".join(content_parts).strip()
    if len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", content)) < 80:
        return None

    category = "Barron's"
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        text = _flatten_block_lines(str(block.get("text") or ""))
        if (
            len(bbox) >= 4
            and float(bbox[1]) < page_height * 0.14
            and float(block.get("max_font_size") or 0) >= 14
            and 2 <= len(text.split()) <= 8
            and text.upper() == text
        ):
            category = text
            break

    return {
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
        "title_block_ids": [str(headline_block.get("id"))],
        "page": page_number,
        "local_recovery": "barrons_native_layout_v1",
    }


def _build_barrons_complex_article(
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    blocks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    section: str,
) -> dict[str, Any] | None:
    if section == "MARKET VIEW":
        body_blocks = _barrons_body_blocks(blocks, page_height)
        content = "\n\n".join(
            _flatten_block_lines(str(block.get("text") or ""))
            for block in body_blocks
        ).strip()
        if len(re.findall(r"[A-Za-z0-9]+", content)) < 300:
            return None
        return _barrons_article_record(
            page_number, "Market View", section, content, body_blocks, images,
            recovery="barrons_market_view_v1",
        )

    candidates = []
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        title = _barrons_headline_text(str(block.get("text") or ""))
        words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", title)
        if (
            len(bbox) >= 4
            and float(block.get("median_font_size") or 0) >= 20
            and float(block.get("max_font_size") or 0) >= 24
            and 2 <= len(words) <= 24
            and title.upper() != title
        ):
            candidates.append(block)
    if not candidates:
        return None
    if section == "RETIREMENT MAILBAG":
        candidates.sort(key=lambda block: float((block.get("bbox") or [0])[0]))
        headline_block = candidates[0]
    else:
        headline_block = max(
            candidates,
            key=lambda block: float(block.get("median_font_size") or 0),
        )
    title = _barrons_headline_text(str(headline_block.get("text") or ""))
    hx0, hy0, hx1, hy1 = [float(value) for value in headline_block["bbox"]]
    if section == "RETIREMENT MAILBAG":
        x_limit = page_width * 0.52
    elif section == "REVIEW & PREVIEW":
        x_limit = page_width * 0.56
    else:
        x_limit = page_width
    body_blocks = [
        block for block in _barrons_body_blocks(blocks, page_height, x_limit=x_limit)
        if block is not headline_block
        and (
            section == "MARKET WEEK"
            or float((block.get("bbox") or [0, 0])[1]) >= hy0 - 4
        )
    ]
    if section == "INSIDE SCOOP":
        x_limit = page_width * 0.55
        following_headlines = [
            float(block["bbox"][1])
            for block in blocks
            if block is not headline_block
            and len(block.get("bbox") or []) >= 4
            and float(block["bbox"][0]) < x_limit
            and float(block["bbox"][1]) > hy1 + 20
            and float(block.get("max_font_size") or 0) >= 16
            and 2 <= len(
                re.findall(r"[A-Za-z0-9]+", str(block.get("text") or ""))
            ) <= 20
        ]
        lower_bound = min(following_headlines, default=page_height)
        body_blocks = [
            block for block in body_blocks
            if float(block["bbox"][0]) < x_limit
            and float(block["bbox"][1]) < lower_bound
        ]
    content = "\n\n".join(_barrons_body_content_parts(body_blocks)).strip()
    dropcap = _barrons_leading_dropcap(blocks, body_blocks)
    if dropcap and content:
        content = _prepend_dropcap(content, dropcap)
    if not content and section == "MARKET WEEK":
        content = title
    if len(re.findall(r"[A-Za-z0-9]+", content)) < 40 and section != "MARKET WEEK":
        return None
    return _barrons_article_record(
        page_number, title, section, content, body_blocks, images,
        recovery="barrons_complex_layout_v1",
        bypass_min_words=section == "MARKET WEEK",
    )


def _build_barrons_continuation_article(
    *,
    page_number: int,
    page_height: float,
    blocks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    prior_articles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not prior_articles:
        return None
    body_blocks = _barrons_body_blocks(blocks, page_height, max_median_size=12)
    content = "\n\n".join(
        _flatten_block_lines(str(block.get("text") or ""))
        for block in body_blocks
    ).strip()
    if len(re.findall(r"[A-Za-z0-9]+", content)) < 300:
        return None
    title = str(prior_articles[-1].get("title") or "").strip()
    if not title:
        return None
    article = _barrons_article_record(
        page_number, title, "Barron's", content, body_blocks, images,
        recovery="barrons_continuation_layout_v1",
    )
    article["is_continuation"] = True
    article["continuation_of"] = title
    return article


def _barrons_body_blocks(
    blocks: list[dict[str, Any]],
    page_height: float,
    *,
    x_limit: float | None = None,
    max_median_size: float | None = None,
) -> list[dict[str, Any]]:
    retained = []
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        text = str(block.get("text") or "").strip()
        words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)
        median_size = float(block.get("median_font_size") or 0)
        max_size = float(block.get("max_font_size") or 0)
        width = float(bbox[2]) - float(bbox[0]) if len(bbox) >= 4 else 0
        if (
            len(bbox) < 4
            or median_size < 7.5
            or (
                len(words) < 35
                and not (len(words) >= 20 and max_size >= 40 and width <= 180)
            )
        ):
            continue
        if _is_barrons_promotion_block(text):
            continue
        if max_median_size is not None and median_size > max_median_size:
            continue
        if float(bbox[1]) < page_height * 0.035:
            continue
        if x_limit is not None and float(bbox[0]) >= x_limit:
            continue
        retained.append(block)
    return _sort_barrons_body_blocks(retained)


def _sort_barrons_body_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for block in sorted(blocks, key=lambda item: float(item["bbox"][0])):
        x0 = float(block["bbox"][0])
        column = next(
            (item for item in columns if abs(x0 - float(item["x"])) <= 32),
            None,
        )
        if column is None:
            column = {"x": x0, "blocks": []}
            columns.append(column)
        column["blocks"].append(block)

    ordered = []
    for column in sorted(columns, key=lambda item: float(item["x"])):
        ordered.extend(sorted(column["blocks"], key=lambda item: float(item["bbox"][1])))
    return ordered


def _barrons_body_content_parts(
    body_blocks: list[dict[str, Any]],
) -> list[str]:
    parts = [
        _flatten_block_lines(str(block.get("text") or ""))
        for block in body_blocks
    ]
    for index in range(len(parts) - 1):
        if float(body_blocks[index].get("max_font_size") or 0) < 30:
            continue
        dropcap = _barrons_dropcap(str(body_blocks[index].get("text") or ""))
        if not dropcap or not re.match(r"[a-z]", parts[index + 1]):
            continue
        parts[index] = re.sub(rf"\s+{re.escape(dropcap)}$", "", parts[index]).strip()
        parts[index + 1] = _prepend_dropcap(parts[index + 1], dropcap)
    return [part for part in parts if part]


def _barrons_leading_dropcap(
    blocks: list[dict[str, Any]],
    body_blocks: list[dict[str, Any]],
) -> str:
    if not body_blocks:
        return ""
    first_bbox = body_blocks[0].get("bbox") or [0, 0, 0, 0]
    first_x = float(first_bbox[0])
    first_y = float(first_bbox[1])
    candidates = []
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4 or float(block.get("max_font_size") or 0) < 30:
            continue
        dropcap = _barrons_dropcap(str(block.get("text") or ""))
        if not dropcap:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if not x0 - 20 <= first_x <= x1 + 20:
            continue
        if y0 > first_y + 40 or y1 < first_y - 180:
            continue
        vertical_gap = 0 if y0 <= first_y <= y1 else min(abs(first_y - y0), abs(first_y - y1))
        candidates.append((vertical_gap, abs(first_x - x0), dropcap))
    return min(candidates)[2] if candidates else ""


def _prepend_dropcap(content: str, dropcap: str) -> str:
    stripped = content.lstrip()
    if not stripped or stripped.upper().startswith(dropcap.upper()):
        return content
    first_word = re.match(r"[A-Za-z]+(?:['’][A-Za-z]+)?", stripped)
    if first_word:
        combined = dropcap + first_word.group(0)
        normalized = re.sub(r"[^A-Za-z]", "", combined)
        if len(wordninja.split(normalized)) > 1:
            return f"{dropcap} {content}"
    return f"{dropcap}{content}"


def _barrons_article_record(
    page_number: int,
    title: str,
    category: str,
    content: str,
    body_blocks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    *,
    recovery: str,
    bypass_min_words: bool = False,
) -> dict[str, Any]:
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
        "local_recovery": recovery,
    }
    if bypass_min_words:
        article["bypass_min_article_words"] = True
    return article


def _barrons_headline_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    while lines and re.fullmatch(r"[A-Z]", lines[-1]):
        lines.pop()
    headline = re.sub(r"-\s+", "-", " ".join(lines)).strip()
    headline = re.sub(r"(?<=[,;:])(?=[A-Za-z])", " ", headline)
    headline = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", headline)
    headline = re.sub(r"(?<=[A-Za-z])(?=\$)", " ", headline)
    headline = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", headline)

    for acronym in ("CEO", "ETF", "SEC", "AI"):
        headline = re.sub(
            rf"(?<![A-Z])({acronym})(?=[A-Z][a-z])",
            r"\1 ",
            headline,
        )
        headline = re.sub(rf"(?<=[a-z])({acronym})", r" \1", headline)

    headline = re.sub(
        r"[A-Za-z]{2,}",
        lambda match: _split_glued_headline_token(match.group(0)),
        headline,
    )
    return re.sub(r"\s+", " ", headline).strip()


def _split_glued_headline_token(token: str) -> str:
    parts = wordninja.split(token)
    if len(parts) < 2 or "".join(parts).lower() != token.lower():
        return token

    case_boundaries = len(re.findall(r"(?<=[a-z])(?=[A-Z])", token))
    join_words = {"a", "an", "and", "at", "for", "in", "of", "on", "the", "to"}
    has_embedded_join_word = any(part.lower() in join_words for part in parts[1:])
    if case_boundaries == 0 and not has_embedded_join_word:
        return token

    restored = []
    offset = 0
    for part in parts:
        restored.append(token[offset : offset + len(part)])
        offset += len(part)
    return " ".join(restored)


def _is_barrons_promotion_block(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return (
        "in a weekly podcast by barron" in normalized
        and "subscribe to barron" in normalized
        and "favorite listening app" in normalized
    )


def _barrons_dropcap(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines and re.fullmatch(r"[A-Z]", lines[-1]) else ""


def _looks_like_heading(block: dict[str, Any], threshold: float, page_height: float) -> bool:
    text = _flatten_block_lines(str(block.get("text") or ""))
    bbox = block.get("bbox") or [0, 0, 0, 0]
    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)
    if float(block.get("max_font_size") or 0) < threshold:
        return False
    if bbox[1] < 110 or bbox[1] > page_height - 45:
        return False
    if not 3 <= len(words) <= 24 or len(text) > 190:
        return False
    normalized = _normalize_title(text)
    blocked = {
        "what s news",
        "business finance",
        "the wall street journal",
        "corrections amplifications",
    }
    return normalized not in blocked and not re.search(r"\.{4,}\s*[A-Z]?\d+$", text)


def _block_assignment_score(
    block: dict[str, Any],
    heading: dict[str, Any],
    page_width: float,
) -> float | None:
    bx0, by0, bx1, _ = [float(value) for value in block["bbox"]]
    hx0, _, hx1, hy1 = [float(value) for value in heading["bbox"]]
    if by0 < hy1 - 2:
        return None
    overlap = max(0.0, min(bx1, hx1) - max(bx0, hx0))
    overlap_ratio = overlap / max(min(bx1 - bx0, hx1 - hx0), 1.0)
    block_center = (bx0 + bx1) / 2
    heading_center = (hx0 + hx1) / 2
    wide_heading_contains = (hx1 - hx0) >= page_width * 0.25 and hx0 - 8 <= block_center <= hx1 + 8
    if overlap_ratio < 0.12 and not wide_heading_contains:
        return None
    vertical_gap = by0 - hy1
    if vertical_gap > 900:
        return None
    return vertical_gap + abs(block_center - heading_center) * 0.18 - overlap_ratio * 60


def _nearest_images_for_heading(
    heading: dict[str, Any],
    images: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> list[str]:
    if not images:
        return []
    hx0, hy0, hx1, hy1 = [float(value) for value in heading["bbox"]]
    hcenter = ((hx0 + hx1) / 2, (hy0 + hy1) / 2)
    ranked = []
    for image in images:
        ix0, iy0, ix1, iy1 = [float(value) for value in image["bbox"]]
        icenter = ((ix0 + ix1) / 2, (iy0 + iy1) / 2)
        distance = abs(icenter[0] - hcenter[0]) / page_width + abs(icenter[1] - hcenter[1]) / page_height
        ranked.append((distance, image["rel_path"]))
    ranked.sort()
    return [path for distance, path in ranked[:1] if distance < 0.45]


def _merge_cross_page_continuations(
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for article in articles:
        title = str(article.get("title") or "").strip()
        continuation_of = str(article.get("continuation_of") or "").strip()
        is_continuation = bool(article.get("is_continuation")) or bool(continuation_of)
        exact_title_target = _find_exact_title_on_other_page(merged, article)
        if exact_title_target is not None:
            is_continuation = True
            continuation_of = str(merged[exact_title_target].get("title") or title)
        if re.search(r"\bcontinued\b", title, flags=re.I):
            is_continuation = True
            if not continuation_of:
                continuation_of = _strip_continued_marker(title)

        target_index = None
        if is_continuation:
            target_index = (
                exact_title_target
                if exact_title_target is not None
                else _find_continuation_target(merged, continuation_of or title)
            )
        if target_index is None:
            normalized = dict(article)
            if is_continuation:
                normalized["title"] = continuation_of or _strip_continued_marker(title) or title
            merged.append(normalized)
            continue

        base = dict(merged[target_index])
        continuation_content = str(article.get("content_markdown") or "").strip()
        base_content = str(base.get("content_markdown") or "").strip()
        if continuation_content and not _text_mostly_overlaps(continuation_content, base_content):
            base["content_markdown"] = "\n\n".join(
                part for part in [base_content, continuation_content] if part
            )
        base["images"] = list(
            dict.fromkeys([*(base.get("images") or []), *(article.get("images") or [])])
        )
        base["source_pages"] = list(
            dict.fromkeys(
                [
                    *(base.get("source_pages") or [base.get("page")]),
                    *(article.get("source_pages") or [article.get("page")]),
                ]
            )
        )
        base["source_block_ids"] = [
            *(base.get("source_block_ids") or []),
            *(article.get("source_block_ids") or []),
        ]
        merged[target_index] = base
        print(f"  🧷 合并跨页续文：第 {article.get('page')} 页 -> {base.get('title')}")
    return merged


def _find_exact_title_on_other_page(
    articles: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> int | None:
    candidate_key = _normalize_title(str(candidate.get("title") or ""))
    candidate_page = candidate.get("page")
    if not candidate_key:
        return None
    for index in range(len(articles) - 1, -1, -1):
        article = articles[index]
        if article.get("page") == candidate_page:
            continue
        if _normalize_title(str(article.get("title") or "")) == candidate_key:
            return index
    return None


def _find_continuation_target(
    articles: list[dict[str, Any]],
    continuation_title: str,
) -> int | None:
    target_key = _normalize_title(_strip_continued_marker(continuation_title))
    if not target_key:
        return None
    best: tuple[float, int] | None = None
    for index, article in enumerate(articles[-20:], start=max(0, len(articles) - 20)):
        candidate = _normalize_title(str(article.get("title") or ""))
        if not candidate:
            continue
        if candidate == target_key:
            return index
        target_words = set(target_key.split())
        candidate_words = set(candidate.split())
        word_overlap = len(target_words & candidate_words) / max(min(len(target_words), len(candidate_words)), 1)
        similarity = max(SequenceMatcher(None, target_key, candidate).ratio(), word_overlap)
        if similarity >= 0.72 and (best is None or similarity > best[0]):
            best = (similarity, index)
    return best[1] if best else None


def _strip_continued_marker(title: str) -> str:
    title = re.sub(r"\s*\(?\s*continued\s*\)?\s*$", "", title, flags=re.I)
    title = re.sub(r"^continued\s*[:\-]\s*", "", title, flags=re.I)
    return re.sub(r"\s+", " ", title).strip()


def _text_mostly_overlaps(fragment: str, existing: str) -> bool:
    fragment_tokens = re.findall(r"[a-z0-9]+", fragment.lower())
    existing_tokens = set(re.findall(r"[a-z0-9]+", existing.lower()))
    if not fragment_tokens or not existing_tokens:
        return False
    return sum(token in existing_tokens for token in fragment_tokens) / len(fragment_tokens) >= 0.75


def _compile_articles_in_order(
    articles: list[dict[str, Any]],
    image_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if not articles:
        return []

    workers = min(
        max(int(os.getenv("LLM_ARTICLE_WORKERS", "3")), 1),
        len(articles),
    )

    def compile_one(article: dict[str, Any]) -> dict[str, Any]:
        page_number = int(article.get("page") or 1)
        result = _compile_articles(page_number, [article], image_dir=image_dir)
        return result[0] if result else article

    if workers == 1:
        return [compile_one(article) for article in articles]
    print(f"  ⚙️ 并发编译 {len(articles)} 篇原生 PDF 文章（workers={workers}）。")
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="native-article",
    ) as executor:
        return list(executor.map(compile_one, articles))


def _reindex_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    page_counts: dict[int, int] = {}
    reindexed = []
    for article_index, article in enumerate(articles, start=1):
        normalized = dict(article)
        page_number = int(normalized.get("page") or 1)
        page_counts[page_number] = page_counts.get(page_number, 0) + 1
        normalized["page"] = page_number
        normalized["page_article_index"] = page_counts[page_number]
        normalized["article_index"] = article_index
        reindexed.append(normalized)
    return reindexed


def _group_articles_by_page(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for article in articles:
        page_number = int(article.get("page") or 1)
        grouped.setdefault(page_number, []).append(article)
    pages = []
    for page_number in sorted(grouped):
        page_articles = grouped[page_number]
        page = {"page": page_number, "articles": page_articles}
        if page_articles:
            for field in (
                "print_page_label",
                "print_section",
                "print_page_source",
            ):
                page[field] = page_articles[0].get(field)
        pages.append(page)
    return pages


def _cached_images_exist(cache_data: dict[str, Any], image_dir: Path) -> bool:
    for article in cache_data.get("articles") or []:
        if not isinstance(article, dict):
            continue
        for image_path in article.get("images") or []:
            filename = Path(str(image_path)).name
            if filename and not (image_dir / filename).exists():
                return False
    return True


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
