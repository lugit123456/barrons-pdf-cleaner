from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
from collections.abc import Callable
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import wordninja

from glossary_enricher import enrich_article_glossary, needs_glossary_refresh
from print_layout import (
    PRINT_LAYOUT_VERSION,
    build_front_page,
    build_issue_pages,
    enrich_page_result,
    extract_front_page_directory,
)

from .base import ParseContext, ParseResult
from .special_pages import (
    PAGE_TYPE_BUSINESS,
    PAGE_TYPE_CONTENTS,
    PAGE_TYPE_POLITICS,
    PAGE_TYPE_UTILITY,
    build_contents_page_result,
    build_utility_page_result,
    build_ocr_world_page_result,
    classify_ocr_page,
)


BLOCKED_ARTICLE_TITLES = {
    "market data",
    "financial times share service",
    "managed funds service",
    "what s news",
    "whats news",
}
MIN_ARTICLE_WORDS = int(os.getenv("MIN_ARTICLE_WORDS", "80"))
_LLM_SEMAPHORE_LOCK = threading.Lock()
_LLM_SEMAPHORE: threading.BoundedSemaphore | None = None
_LLM_SEMAPHORE_SIZE = 0


class ScannedPdfStrategy:
    """Scanned PDF parser using FT demo image extraction, OCR, and LLM grouping."""

    engine_name = "Strategy_B_OCR_LLM"

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
        self.require_llm_article_split = _env_bool(
            "REQUIRE_LLM_ARTICLE_SPLIT",
            True,
        )

    def parse(self, context: ParseContext) -> ParseResult:
        from pdf2image import convert_from_path

        context.target_output_dir.mkdir(parents=True, exist_ok=True)
        context.image_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = context.target_output_dir / "cache_json"
        if self.cache_enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)

        page_count = _pdf_page_count(context.pdf_path)
        page_numbers = [
            page_number
            for page_number in range(1, page_count + 1)
            if not context.selected_pages or page_number in context.selected_pages
        ]
        work_page_numbers = sorted(
            page_numbers,
            key=lambda page_number: (
                self.cache_enabled and (cache_dir / f"page_{page_number}.json").exists(),
                page_number,
            ),
        )
        uncached_count = sum(
            1
            for page_number in page_numbers
            if not self.cache_enabled or not (cache_dir / f"page_{page_number}.json").exists()
        )
        all_pages: list[dict[str, Any]] = []
        page_workers = min(
            max(int(os.getenv("PDF_PAGE_WORKERS", "2")), 1),
            max(len(page_numbers), 1),
        )
        render_dpi = max(int(os.getenv("SCANNED_RENDER_DPI", "150")), 72)
        print(
            f"  ⚙️ 扫描页流水线：{page_count} 页，页面并发 {page_workers}，"
            f"LLM 最大并发 {_llm_max_concurrency()}；优先处理 {uncached_count} 个未缓存页。"
        )

        def process_page(page_idx: int) -> dict[str, Any]:
            started = time.monotonic()
            try:
                cached = self._load_cached_page(
                    page_idx,
                    context.image_dir,
                    cache_dir,
                    context.metadata.publication_type,
                )
                if cached is not None:
                    return cached
                pages = convert_from_path(
                    str(context.pdf_path),
                    dpi=render_dpi,
                    first_page=page_idx,
                    last_page=page_idx,
                    thread_count=1,
                )
                if not pages:
                    raise RuntimeError("Poppler 未返回页面图像")
                return self.process_page_image(
                    page_idx,
                    pages[0],
                    context.image_dir,
                    cache_dir,
                    check_cache=False,
                    publication_type=context.metadata.publication_type,
                )
            except Exception as exc:
                print(f"  ⚠️ 第 {page_idx} 页处理失败，跳过并继续：{_compact_error(exc)}")
                return {"page": page_idx, "articles": [], "error": str(exc)}
            finally:
                elapsed = time.monotonic() - started
                print(f"  ⏱️ 第 {page_idx} 页处理耗时：{elapsed:.1f}s")

        if page_workers == 1:
            page_iterator = map(process_page, work_page_numbers)
            executor = None
        else:
            executor = ThreadPoolExecutor(
                max_workers=page_workers,
                thread_name_prefix="pdf-page",
            )
            page_iterator = executor.map(process_page, work_page_numbers)
        try:
            for page_result in page_iterator:
                all_pages.append(page_result)
                if context.article_writer:
                    context.article_writer(
                        _flatten_page_articles([page_result]),
                        self.engine_name,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

        all_pages.sort(key=lambda page: int(page.get("page") or 0))
        articles = _flatten_page_articles(all_pages)

        def checkpoint_article(_: int, article: dict[str, Any]) -> None:
            if not self.cache_enabled:
                return
            page_number = int(article.get("page") or 0)
            page_article_index = int(article.get("page_article_index") or 0)
            page = next(
                (
                    item
                    for item in all_pages
                    if int(item.get("page") or 0) == page_number
                ),
                None,
            )
            if page is None or page_article_index <= 0:
                return
            page_articles = page.get("articles") or []
            if page_article_index > len(page_articles):
                return
            page_articles[page_article_index - 1] = article
            page["articles"] = page_articles
            _write_json_cache(cache_dir / f"page_{page_number}.json", page)

        articles = retry_deferred_compiles(
            articles,
            image_dir=context.image_dir,
            on_article_updated=checkpoint_article,
        )
        _replace_page_articles(all_pages, articles)
        if self.cache_enabled:
            for page_result in all_pages:
                page_number = int(page_result.get("page") or 0)
                if page_number > 0 and not page_result.get("error"):
                    _write_json_cache(cache_dir / f"page_{page_number}.json", page_result)

        failed_page_numbers = {
            int(page.get("page") or 0)
            for page in all_pages
            if page.get("error") and int(page.get("page") or 0) > 0
        }
        failed_page_numbers.update(_pending_compile_pages(articles))
        failed_pages = tuple(sorted(failed_page_numbers))

        return ParseResult(
            body_markdown=_render_pages_to_markdown(all_pages),
            engine_name=self.engine_name,
            articles=articles,
            pages=build_issue_pages(all_pages),
            front_page=build_front_page(
                all_pages, context.metadata.publication_type
            ),
            is_weekend=any(
                int(page.get("page") or 0) == 1
                and page.get("is_weekend") is True
                for page in all_pages
            ),
            complete=not failed_pages,
            failed_pages=failed_pages,
        )

    def _load_cached_page(
        self,
        page_idx: int,
        image_dir: Path,
        cache_dir: Path,
        publication_type: str | None = None,
    ) -> dict[str, Any] | None:
        cache_file = cache_dir / f"page_{page_idx}.json"
        if not self.cache_enabled or not cache_file.exists():
            return None
        cached_result = _read_json_cache(cache_file)
        if cached_result is None:
            print(f"♻️ 第 {page_idx} 页缓存不完整，重新处理本页。")
            return None
        if cached_result.get("error"):
            cache_file.unlink(missing_ok=True)
            print(f"♻️ 第 {page_idx} 页缓存记录了失败状态，重新处理本页。")
            return None
        metadata_enriched = False
        header_blocks = cached_result.get("header_blocks")
        whats_news_blocks = cached_result.get("whats_news_blocks")
        if (
            str(publication_type or "").strip().upper() == "FT"
            and page_idx == 1
            and cached_result.get("print_layout_version") != PRINT_LAYOUT_VERSION
            and not str(cached_result.get("whats_news_text") or "").strip()
        ):
            print("♻️ FT 第一页缓存缺少 Briefing 区域，重新读取页面图像。")
            return None
        has_cached_layout_input = bool(
            str(cached_result.get("header_text") or "").strip()
            or str(cached_result.get("whats_news_text") or "").strip()
            or (isinstance(header_blocks, list) and header_blocks)
            or (isinstance(whats_news_blocks, list) and whats_news_blocks)
        )
        if (
            has_cached_layout_input
            and cached_result.get("print_layout_version") != PRINT_LAYOUT_VERSION
        ):
            cached_result = enrich_page_result(
                cached_result,
                publication_type or "",
                page_idx,
                header_text=str(cached_result.get("header_text") or ""),
                header_blocks=header_blocks if isinstance(header_blocks, list) else None,
                whats_news_text=str(cached_result.get("whats_news_text") or ""),
                whats_news_blocks=(
                    whats_news_blocks if isinstance(whats_news_blocks, list) else None
                ),
                page_width=cached_result.get("page_width"),
                page_height=cached_result.get("page_height"),
            )
            metadata_enriched = True
        if self.require_llm_article_split and _is_page_level_fallback_cache(cached_result):
            print(f"♻️ 第 {page_idx} 页缓存是一页 fallback，重新调用 LLM 拆文章。")
            return None
        if _cache_needs_reparse(cached_result):
            print(f"♻️ 第 {page_idx} 页缓存含未编译短正文，重新 OCR + LLM 修复。")
            return None

        cleaned_result = _clean_page_result(cached_result)
        cleaned_result["articles"] = _merge_duplicate_titles(
            cleaned_result.get("articles") or []
        )
        cleaned_result["articles"] = _compile_articles(
            page_idx,
            cleaned_result.get("articles") or [],
            image_dir=image_dir,
        )
        if metadata_enriched or cleaned_result != cached_result:
            _write_json_cache(cache_file, cleaned_result)
        print(f"⚡ 第 {page_idx} 页读取本地 JSON 缓存。")
        return cleaned_result

    def process_page_image(
        self,
        page_idx: int,
        page_img: Any,
        image_dir: Path,
        cache_dir: Path,
        check_cache: bool = True,
        publication_type: str | None = None,
    ) -> dict[str, Any]:
        """Run the scanned-page pipeline for one rendered PDF page."""
        crop_images_from_scan_page, slice_text_by_anchors = _load_ft_demo_functions()
        cache_dir.mkdir(parents=True, exist_ok=True)
        if check_cache:
            cached = self._load_cached_page(
                page_idx, image_dir, cache_dir, publication_type
            )
            if cached is not None:
                return cached
        return self._process_single_page(
            page_idx,
            page_img,
            image_dir,
            cache_dir,
            crop_images_from_scan_page,
            slice_text_by_anchors,
            publication_type,
        )

    def _process_single_page(
        self,
        page_idx: int,
        page_img: Any,
        image_dir: Path,
        cache_dir: Path,
        crop_images_from_scan_page: Any,
        slice_text_by_anchors: Any,
        publication_type: str | None = None,
    ) -> dict[str, Any]:
        import pytesseract

        cache_file = cache_dir / f"page_{page_idx}.json"

        page_figs = crop_images_from_scan_page(page_img, page_idx, str(image_dir))
        print(f"  🔍 正在对第 {page_idx} 页进行本地 OCR 识别...")
        page_text = pytesseract.image_to_string(page_img, lang="eng")
        page_width, page_height = page_img.size
        header_crop = page_img.crop(
            (0, 0, page_width, max(1, int(page_height * 0.18)))
        )
        header_text = pytesseract.image_to_string(header_crop, lang="eng")
        whats_news_text = ""
        publication = str(publication_type or "").strip().upper()
        if publication == "WSJ" and page_idx == 1:
            whats_news_crop = page_img.crop(
                (0, 0, max(1, int(page_width * 0.19)), max(1, int(page_height * 0.94)))
            )
            whats_news_text = pytesseract.image_to_string(whats_news_crop, lang="eng")
        elif publication == "FT" and page_idx == 1:
            heading_crop = page_img.crop(
                (
                    int(page_width * 0.81),
                    int(page_height * 0.115),
                    int(page_width * 0.995),
                    int(page_height * 0.19),
                )
            )
            heading_text = pytesseract.image_to_string(
                heading_crop,
                lang="eng",
                config="--psm 6",
            )
            briefing_crop = page_img.crop(
                (
                    int(page_width * 0.81),
                    int(page_height * 0.11),
                    int(page_width * 0.995),
                    int(page_height * 0.68),
                )
            )
            briefing_text = pytesseract.image_to_string(
                briefing_crop,
                lang="eng",
                config="--psm 6",
            )
            if re.search(r"(?i)\bBriefing\b", f"{heading_text}\n{briefing_text}"):
                whats_news_text = f"Briefing\n{briefing_text}"
        whats_news = extract_front_page_directory(
            publication,
            whats_news_text,
        )
        article_page_text = _strip_whats_news_module_text(
            page_text,
            whats_news_text,
            whats_news,
        )

        def with_layout(result: dict[str, Any]) -> dict[str, Any]:
            result = dict(result)
            result["header_text"] = header_text
            if whats_news_text:
                result["whats_news_text"] = whats_news_text
            enriched = enrich_page_result(
                result,
                publication_type or "",
                page_idx,
                header_text=header_text,
                whats_news_text=whats_news_text,
            )
            if self.cache_enabled and not enriched.get("error"):
                _write_json_cache(cache_file, enriched)
            return enriched

        page_type = classify_ocr_page(page_text, publication)
        if page_type == PAGE_TYPE_CONTENTS:
            print(f"  ⏭️ 第 {page_idx} 页识别为 Contents，跳过目录页。")
            result = build_contents_page_result(page_idx, "scanned_ocr_special_v1")
            if self.cache_enabled:
                _write_json_cache(cache_file, result)
            return with_layout(result)
        if page_type == PAGE_TYPE_UTILITY:
            print(f"  ⏭️ 第 {page_idx} 页识别为 Barron's Index/Data，跳过非正文页。")
            result = build_utility_page_result(page_idx, "scanned_ocr_special_v2")
            if self.cache_enabled:
                _write_json_cache(cache_file, result)
            return with_layout(result)
        if page_type in {PAGE_TYPE_POLITICS, PAGE_TYPE_BUSINESS}:
            label = "Politics" if page_type == PAGE_TYPE_POLITICS else "Business"
            print(f"  🧩 第 {page_idx} 页识别为 The world this week / {label}，整页合并为一篇。")
            result = build_ocr_world_page_result(
                page_number=page_idx,
                page_type=page_type,
                page_text=page_text,
                image_paths=_figure_paths(page_figs),
            )
            result["parser"] = "scanned_ocr_special_v1"
            result["articles"] = _compile_articles(
                page_idx,
                result.get("articles") or [],
                image_dir=image_dir,
            )
            if self.cache_enabled:
                _write_json_cache(cache_file, result)
            return with_layout(result)

        if not page_text.strip():
            print(f"  🤖 第 {page_idx} 页 OCR 为空，改用视觉 LLM 直接抽取文章...")
            articles = _filter_articles_against_whats_news(
                _clean_articles(
                    _identify_articles_from_page_image(page_idx, page_img, page_figs)
                ),
                whats_news,
            )
            if articles:
                result = {"page": page_idx, "articles": articles}
                if self.cache_enabled:
                    _write_json_cache(cache_file, result)
                return with_layout(result)
            if self.require_llm_article_split:
                print(f"  💤 第 {page_idx} 页 OCR 与视觉识别均未发现报道，记录为空页。")
                result = {
                    "page": page_idx,
                    "articles": [],
                    "empty_page": True,
                    "empty_reason": "OCR 为空，视觉 LLM 未识别到报道",
                }
                if self.cache_enabled:
                    _write_json_cache(cache_file, result)
                return with_layout(result)
            return with_layout({
                "page": page_idx,
                "articles": _build_image_only_articles(page_idx, page_figs),
                "raw_text": "",
            })

        try:
            articles = self._identify_articles(page_idx, article_page_text, page_figs)
        except Exception as exc:
            print(f"⚠️ 第 {page_idx} 页 LLM 文章识别失败，跳过本页继续后续页面：{exc}")
            return with_layout(
                {"page": page_idx, "articles": [], "error": str(exc)}
            )

        if not articles:
            print(f"  🧹 第 {page_idx} 页 LLM 未返回可保存文章。")
            if self.require_llm_article_split:
                result = {
                    "page": page_idx,
                    "articles": [],
                    "empty_page": True,
                    "empty_reason": "OCR 有文字，但 LLM 未识别到可保存报道",
                }
                if self.cache_enabled:
                    _write_json_cache(cache_file, result)
                return with_layout(result)
            articles = _build_ocr_fallback_articles(page_idx, article_page_text, page_figs)

        for article in articles:
            if article.get("content_markdown"):
                continue

            full_body = slice_text_by_anchors(
                article_page_text,
                article.get("start_anchor"),
                article.get("end_anchor"),
            )
            article["content_markdown"] = (
                full_body
                if full_body
                else f"{article.get('start_anchor', '')} ... {article.get('end_anchor', '')}".strip()
            )
            article.pop("start_anchor", None)
            article.pop("end_anchor", None)

        articles = _repair_short_articles(page_idx, article_page_text, articles)
        articles = _review_and_merge_page_fragments(page_idx, article_page_text, articles)
        articles = _merge_continued_articles(articles)
        articles = _merge_related_article_fragments(articles)
        articles = _merge_duplicate_titles(articles)
        articles = _clean_articles(articles)
        articles = _filter_articles_against_whats_news(articles, whats_news)
        articles = _compile_articles(page_idx, articles, image_dir=image_dir)
        result = {"page": page_idx, "articles": articles}
        if self.cache_enabled:
            _write_json_cache(cache_file, result)
        return with_layout(result)

    def _identify_articles(
        self,
        page_idx: int,
        page_text: str,
        page_figs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "3000"))
        prompt = _build_article_prompt(page_idx, page_text, page_figs)
        print(f"  🤖 正在调用 LLM 识别第 {page_idx} 页文章结构...")

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                content = _chat_completion_text(
                    operation=f"第 {page_idx} 页文章识别",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    max_retries=1,
                )
                return _extract_articles_from_json(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    print(
                        f"  🔁 第 {page_idx} 页文章识别第 {attempt}/{self.max_retries} 次失败，"
                        f"准备重试：{_compact_error(exc)}"
                    )
                    time.sleep(min(2 * attempt, 5))

        raise RuntimeError(f"第 {page_idx} 页 LLM 解析异常: {last_error}")


def _load_ft_demo_functions() -> tuple[Any, Any]:
    from parse_ft_pipeline import crop_images_from_scan_page, slice_text_by_anchors

    return crop_images_from_scan_page, slice_text_by_anchors


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        from pdf2image import pdfinfo_from_path

        page_count = int(pdfinfo_from_path(str(pdf_path)).get("Pages") or 0)
        if page_count > 0:
            return page_count
    except Exception:
        pass

    from pypdf import PdfReader

    page_count = len(PdfReader(str(pdf_path), strict=False).pages)
    if page_count <= 0:
        raise RuntimeError("PDF 不包含可解析页面")
    return page_count


def _read_json_cache(cache_file: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json_cache(cache_file: Path, data: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    temp_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(cache_file)


def _llm_max_concurrency() -> int:
    return max(int(os.getenv("LLM_MAX_CONCURRENT_REQUESTS", "3")), 1)


def _llm_request_semaphore() -> threading.BoundedSemaphore:
    global _LLM_SEMAPHORE, _LLM_SEMAPHORE_SIZE
    requested_size = _llm_max_concurrency()
    with _LLM_SEMAPHORE_LOCK:
        if _LLM_SEMAPHORE is None:
            _LLM_SEMAPHORE = threading.BoundedSemaphore(requested_size)
            _LLM_SEMAPHORE_SIZE = requested_size
        return _LLM_SEMAPHORE


def _openai_client() -> Any:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，请在 .env 中配置后重试。")
    timeout = float(os.getenv("OPENAI_TIMEOUT", os.getenv("LLM_TIMEOUT", "180")))

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_URL", "https://sub2.yz.rs/v1"),
        timeout=timeout,
        max_retries=0,
    )


def _chat_completion_text(
    operation: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    model_env: str = "OPENAI_MODEL",
    temperature: float | None = 0.1,
    max_retries: int | None = None,
) -> str:
    client = _openai_client()
    model_names = _llm_model_candidates(model_env)
    retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "3"))
    last_error: Exception | None = None
    if not model_names:
        raise RuntimeError("所有候选 LLM 模型在本次运行中均已失败，跳过本次 LLM 调用")

    for model_index, model_name in enumerate(model_names, start=1):
        if model_index > 1:
            print(f"  🔄 {operation}：切换备用模型 {model_name}")

        token_param = "max_tokens"
        include_temperature = temperature is not None
        include_extra_body = True
        include_response_format = _env_bool("LLM_JSON_MODE", True)
        attempt = 1
        while attempt <= retries:
            try:
                response = _create_chat_completion_once(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    token_param=token_param,
                    temperature=temperature,
                    include_temperature=include_temperature,
                    include_extra_body=include_extra_body,
                    include_response_format=include_response_format,
                )
                text = _completion_response_text(response)
                if not text.strip():
                    raise RuntimeError(
                        f"LLM 返回为空，{_completion_response_summary(response)}"
                    )
                return text
            except Exception as exc:
                if include_temperature and _is_unsupported_parameter(exc, "temperature"):
                    include_temperature = False
                    last_error = exc
                    print(f"  🔧 {operation}：当前模型不接受 temperature，已自动去掉后重试。")
                    continue
                if token_param == "max_tokens" and _is_unsupported_parameter(exc, "max_tokens"):
                    token_param = "max_completion_tokens"
                    last_error = exc
                    print(f"  🔧 {operation}：当前模型改用 max_completion_tokens 后重试。")
                    continue
                if include_extra_body and _is_unsupported_extra_body(exc):
                    include_extra_body = False
                    last_error = exc
                    print(f"  🔧 {operation}：当前接口不接受 thinking/reasoning 参数，已自动去掉后重试。")
                    continue
                if include_response_format and _is_unsupported_parameter(exc, "response_format"):
                    include_response_format = False
                    last_error = exc
                    print(f"  🔧 {operation}：当前接口不接受 JSON mode，已自动去掉后重试。")
                    continue

                last_error = exc
                if attempt < retries:
                    print(
                        f"  🔁 {operation} 使用 {model_name} 第 {attempt}/{retries} 次调用失败，"
                        f"准备重试：{_compact_error(exc)}"
                    )
                    time.sleep(min(2 * attempt, 8))
                attempt += 1

    raise RuntimeError(str(last_error) if last_error else "LLM 调用失败")


def _chat_completion_json(
    operation: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    model_env: str = "OPENAI_MODEL",
    temperature: float | None = 0.1,
    max_retries: int | None = None,
) -> Any:
    retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "3"))
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            content = _chat_completion_text(
                operation=operation,
                messages=messages,
                max_tokens=max_tokens,
                model_env=model_env,
                temperature=temperature,
                max_retries=1,
            )
            return _loads_llm_json(content)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(
                    f"  🔁 {operation} 第 {attempt}/{retries} 次未得到有效 JSON，"
                    f"准备重试：{_compact_error(exc)}"
                )
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(str(last_error) if last_error else "LLM JSON 解析失败")


def _llm_model_candidates(model_env: str) -> list[str]:
    primary = os.getenv(model_env) or os.getenv("OPENAI_MODEL", "gpt-4o")
    fallback_env = os.getenv(f"{model_env}_FALLBACK_MODELS")
    if fallback_env is None:
        fallback_env = os.getenv("OPENAI_FALLBACK_MODELS", "")
    candidates = [primary]
    candidates.extend(
        model.strip()
        for model in fallback_env.split(",")
        if model.strip()
    )

    unique_candidates = []
    seen = set()
    for model in candidates:
        if model not in seen:
            seen.add(model)
            unique_candidates.append(model)

    return unique_candidates


def _is_transient_llm_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    message = str(exc).lower()
    transient_markers = (
        "connection error",
        "request timed out",
        "timed out",
        "timeout",
        "502",
        "503",
        "504",
        "upstream",
        "temporarily unavailable",
        "llm 返回为空",
    )
    return any(marker in message for marker in transient_markers)


def _create_chat_completion_once(
    client: Any,
    model_name: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    token_param: str,
    temperature: float | None,
    include_temperature: bool,
    include_extra_body: bool,
    include_response_format: bool,
) -> Any:
    params: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        token_param: max_tokens,
    }
    if include_temperature and temperature is not None:
        params["temperature"] = temperature
    if include_response_format:
        params["response_format"] = {"type": "json_object"}
    extra_body = _llm_extra_body(model_name) if include_extra_body else {}
    if extra_body:
        params["extra_body"] = extra_body
    with _llm_request_semaphore():
        return client.chat.completions.create(**params)


def _llm_extra_body(model_name: str) -> dict[str, Any]:
    extra_body: dict[str, Any] = {}
    raw_extra_body = os.getenv("OPENAI_EXTRA_BODY", "").strip()
    if raw_extra_body:
        try:
            parsed = json.loads(raw_extra_body)
            if isinstance(parsed, dict):
                extra_body.update(parsed)
        except json.JSONDecodeError as exc:
            print(f"  ⚠️ OPENAI_EXTRA_BODY 不是合法 JSON，已忽略：{exc}")

    if _env_bool("MINIMAX_REASONING_SPLIT", _looks_like_minimax_model(model_name)):
        extra_body.setdefault("reasoning_split", True)

    if _env_bool("MINIMAX_DISABLE_THINKING", _looks_like_minimax_model(model_name)):
        extra_body.setdefault("thinking", {"type": "disabled"})
        extra_body.setdefault("enable_thinking", False)

    return extra_body


def _looks_like_minimax_model(model_name: str) -> bool:
    normalized = model_name.lower()
    return "minimax" in normalized or re.search(r"\bm[23](?:[\._-]|$)", normalized) is not None


def _is_unsupported_parameter(exc: Exception, parameter: str) -> bool:
    message = str(exc).lower()
    parameter = parameter.lower()
    return (
        parameter in message
        and (
            "unsupported" in message
            or "not support" in message
            or "does not support" in message
            or "unrecognized" in message
            or "unknown parameter" in message
            or "invalid" in message
        )
    )


def _is_unsupported_extra_body(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        any(
            parameter in message
            for parameter in (
                "reasoning_split",
                "thinking",
                "enable_thinking",
                "extra_body",
            )
        )
        and (
            "unsupported" in message
            or "not support" in message
            or "does not support" in message
            or "unrecognized" in message
            or "unknown parameter" in message
            or "invalid" in message
        )
    )


def _completion_response_text(response: Any) -> str:
    texts: list[str] = []
    choices = getattr(response, "choices", None) or []
    for choice in choices:
        message = getattr(choice, "message", None) or getattr(choice, "delta", None)
        texts.extend(_message_text_parts(message))
        choice_text = getattr(choice, "text", None)
        if isinstance(choice_text, str):
            texts.append(choice_text)

    if not texts and hasattr(response, "output_text"):
        output_text = getattr(response, "output_text")
        if isinstance(output_text, str):
            texts.append(output_text)

    if not texts and hasattr(response, "model_dump"):
        data = response.model_dump()
        texts.extend(_response_dict_text_parts(data))

    return "\n".join(text.strip() for text in texts if text and text.strip()).strip()


def _message_text_parts(message: Any) -> list[str]:
    if message is None:
        return []

    if isinstance(message, dict):
        content = message.get("content")
        tool_calls = message.get("tool_calls") or []
    else:
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None) or []

    texts = _content_text_parts(content)
    for tool_call in tool_calls:
        function = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        arguments = (
            function.get("arguments")
            if isinstance(function, dict)
            else getattr(function, "arguments", None)
        )
        if isinstance(arguments, str) and arguments.strip():
            texts.append(arguments)
    return texts


def _content_text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                for key in ("text", "content", "arguments"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        texts.append(value)
                    elif isinstance(value, list):
                        texts.extend(_content_text_parts(value))
        return texts
    if isinstance(content, dict):
        return _content_text_parts([content])
    return []


def _response_dict_text_parts(data: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or choice.get("delta") or {}
        texts.extend(_message_text_parts(message))
        text = choice.get("text")
        if isinstance(text, str):
            texts.append(text)
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        texts.append(output_text)
    return texts


def _completion_response_summary(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None) or []
        finish_reasons = [
            str(getattr(choice, "finish_reason", "") or "")
            for choice in choices
        ]
        return f"choices={len(choices)}, finish_reason={','.join(finish_reasons) or 'unknown'}"
    except Exception:
        return "response summary unavailable"


def _compact_error(exc: Exception, limit: int = 220) -> str:
    message = re.sub(r"\s+", " ", str(exc)).strip()
    if len(message) > limit:
        return message[: limit - 3] + "..."
    return message


def _build_article_prompt(
    page_idx: int,
    page_text: str,
    page_figs: list[dict[str, Any]],
) -> str:
    return f"""
Analyze the newspaper Page {page_idx} OCR text below:

--- OCR TEXT START ---
{page_text}
--- OCR TEXT END ---

Image paths on this page: {json.dumps(page_figs, ensure_ascii=False)}

Task: Identify all distinct articles and locate their boundaries in the OCR text.
Do NOT merge multiple news items into one article. A front page can contain many short articles.
If one article is visually split into multiple columns or regions, return it once with anchors spanning all of its text.
Never return the same title twice. Repeated titles on one page usually indicate fragments of the same report.
Do NOT return:
- What's News / What’s News summaries, FT Briefing summaries, or their directory bullets
- market data tables
- FINANCIAL TIMES SHARE SERVICE
- MANAGED FUNDS SERVICE
- ads, fund listings, stock lists, service directories, crosswords, puzzles, weather, mastheads
- very short teasers that only point to another page, such as "Big Read, page 21"
For each article, provide:
1. title
2. category
3. start_anchor: exact first 10-15 words of that article body as they appear in OCR text
4. end_anchor: exact last 10-15 words of that article body as they appear in OCR text
5. images: array of matching image rel_path values, such as "images/page_{page_idx}_fig_1.jpg"

IMPORTANT OUTPUT RULES:
- Return the final JSON object only.
- Do not output analysis, reasoning, notes, explanations, or Markdown.
- Do not output <think> or </think> tags.
- If you have internal reasoning, keep it hidden and output only the JSON.

Return JSON ONLY:
{{
  "articles": [
    {{
      "title": "Article Title",
      "category": "Section Name",
      "start_anchor": "exact first 10-15 words...",
      "end_anchor": "exact last 10-15 words...",
      "images": []
    }}
  ]
}}
"""


def _extract_articles_from_json(content: str) -> list[dict[str, Any]]:
    data = _loads_llm_json(content)
    if isinstance(data, list):
        articles = data
    elif isinstance(data, dict):
        articles = data.get("articles", data.get("items", []))
    else:
        articles = []
    return [article for article in articles if isinstance(article, dict)]


def _loads_llm_json(content: str) -> Any:
    content = (content or "").strip().lstrip("\ufeff")
    if not content:
        raise ValueError("LLM 返回为空，无法解析 JSON")

    errors: list[str] = []
    for source in _json_parse_sources(content):
        for candidate in _json_candidates(source):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))

    preview = re.sub(r"\s+", " ", content)[:300]
    if _looks_like_thinking_only_response(content):
        raise ValueError(
            "LLM 只返回了 <think> 思考过程，没有返回最终 JSON。"
            "请增大 LLM_MAX_TOKENS，或换用非 thinking 模型/在模型参数里关闭 thinking。"
            f"（preview={preview!r}）"
        )
    raise ValueError(
        "LLM 未返回可解析 JSON"
        f"（preview={preview!r}; errors={'; '.join(errors[-2:]) or 'none'}）"
    )


def _json_parse_sources(content: str) -> list[str]:
    stripped = content.strip()
    without_thinking = _strip_thinking_blocks(stripped).strip()
    sources = []
    if without_thinking:
        sources.append(without_thinking)
    if stripped and stripped != without_thinking:
        sources.append(stripped)
    return sources or [stripped]


def _strip_thinking_blocks(content: str) -> str:
    content = re.sub(
        r"<think\b[^>]*>[\s\S]*?</think>",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()
    end_match = re.search(r"</think>", content, flags=re.IGNORECASE)
    if end_match:
        content = content[end_match.end() :].strip()
    return content


def _looks_like_thinking_only_response(content: str) -> bool:
    if not re.search(r"<think\b", content, flags=re.IGNORECASE):
        return False
    without_thinking = _strip_thinking_blocks(content)
    return not _json_candidates(without_thinking)


def _json_candidates(content: str) -> list[str]:
    candidates: list[str] = []
    stripped = content.strip()
    if stripped.startswith(("{", "[")):
        candidates.append(stripped)

    for match in re.finditer(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", content):
        candidate = match.group(1).strip()
        if candidate:
            candidates.append(candidate)

    for match in re.finditer(r"[\{\[]", content):
        start = match.start()
        candidate = _balanced_json_substring(content, start)
        if candidate:
            candidates.append(candidate.strip())

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def _balanced_json_substring(content: str, start: int) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                return ""
            stack.pop()
            if not stack:
                return content[start : index + 1]
    return ""


def _clean_page_result(page_result: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(page_result)
    cleaned["articles"] = _filter_articles_against_whats_news(
        _clean_articles(page_result.get("articles") or []),
        page_result.get("whats_news"),
    )
    return cleaned


def _clean_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for article in articles:
        reason = _article_reject_reason(article)
        if reason:
            title = article.get("title") or "Untitled"
            print(f"  🧹 丢弃非正文文章：{title}（{reason}）")
            continue
        cleaned.append(article)
    return cleaned


def _repair_short_articles(
    page_idx: int,
    page_text: str,
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not _env_bool("LLM_COMPILE_ARTICLES", True):
        return articles

    repaired_articles = []
    for article in articles:
        reason = _article_reject_reason(article)
        if not reason or "正文过短" not in reason:
            repaired_articles.append(article)
            continue
        if _is_unrepairable_article(article):
            repaired_articles.append(article)
            continue

        repaired_content = _repair_article_content_with_llm(
            page_idx=page_idx,
            page_text=page_text,
            article=article,
        )
        if repaired_content:
            repaired = dict(article)
            repaired["content_markdown"] = repaired_content
            repaired_articles.append(repaired)
        else:
            repaired_articles.append(article)
    return repaired_articles


def _review_and_merge_page_fragments(
    page_idx: int,
    page_text: str,
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ask the LLM to confirm suspicious same-page fragments before local merging."""
    if len(articles) < 2:
        return articles

    candidates = []
    for index, article in enumerate(articles, start=1):
        content = str(article.get("content_markdown") or "")
        candidates.append(
            {
                "index": index,
                "title": str(article.get("title") or ""),
                "category": str(article.get("category") or ""),
                "word_count": len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", content)),
                "start": re.sub(r"\s+", " ", content)[:180],
                "end": re.sub(r"\s+", " ", content)[-180:],
            }
        )
    if not _has_suspicious_fragment_pair(candidates):
        return articles

    prompt = f"""
You are checking article boundaries on newspaper page {page_idx}.
Several OCR candidates may be fragments of one complete article. Decide only definite merges.
Do not merge separate reports just because they share a topic or section.
Use the candidate titles, categories, starts, ends, and the raw page text for context.

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}

RAW PAGE TEXT:
{page_text}

Return JSON only, with 1-based candidate indexes. If no definite merge exists, return an empty list.
{{"merge_groups": [[1, 2]]}}
"""
    try:
        data = _chat_completion_json(
            operation=f"第 {page_idx} 页文章完整性复核",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(os.getenv("LLM_FRAGMENT_REVIEW_MAX_TOKENS", "1800")),
            max_retries=int(os.getenv("LLM_FRAGMENT_REVIEW_MAX_RETRIES", "2")),
        )
        groups = data.get("merge_groups", []) if isinstance(data, dict) else []
        return _merge_index_groups(articles, groups)
    except Exception as exc:
        print(f"  ⚠️ 第 {page_idx} 页文章完整性复核失败，使用本地规则：{_compact_error(exc)}")
        return _merge_duplicate_titles(articles)


def _has_suspicious_fragment_pair(candidates: list[dict[str, Any]]) -> bool:
    normalized = [re.sub(r"[^a-z0-9]+", " ", item["title"].lower()).strip() for item in candidates]
    if len(normalized) != len(set(normalized)):
        return True
    for index, current in enumerate(normalized):
        current_words = set(current.split())
        for other in normalized[index + 1:]:
            other_words = set(other.split())
            overlap = len(current_words & other_words) / max(min(len(current_words), len(other_words)), 1)
            if overlap >= 0.75:
                return True
    return False


def _merge_index_groups(
    articles: list[dict[str, Any]],
    raw_groups: Any,
) -> list[dict[str, Any]]:
    if not isinstance(raw_groups, list):
        return articles
    consumed: set[int] = set()
    merged_by_first: dict[int, dict[str, Any]] = {}
    for raw_group in raw_groups:
        if not isinstance(raw_group, list):
            continue
        indexes = sorted({int(value) - 1 for value in raw_group if str(value).isdigit()})
        indexes = [index for index in indexes if 0 <= index < len(articles)]
        if len(indexes) < 2 or consumed.intersection(indexes):
            continue
        merged_by_first[indexes[0]] = _merge_article_fragment_group(
            [articles[index] for index in indexes]
        )
        consumed.update(indexes)

    if not consumed:
        return articles
    result = []
    for index, article in enumerate(articles):
        if index in merged_by_first:
            result.append(merged_by_first[index])
        if index not in consumed:
            result.append(article)
    return result


def _merge_duplicate_titles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for article in articles:
        title_key = re.sub(r"[^a-z0-9]+", " ", str(article.get("title") or "").lower()).strip()
        title_key = title_key or f"untitled-{len(order)}"
        if title_key not in groups:
            order.append(title_key)
        groups.setdefault(title_key, []).append(article)

    merged = []
    for title_key in order:
        group = groups[title_key]
        merged.append(_merge_article_fragment_group(group) if len(group) > 1 else group[0])
        if len(group) > 1:
            print(f"  🧷 同页合并重复标题：{group[0].get('title')}")
    return merged


def _is_unrepairable_article(article: dict[str, Any]) -> bool:
    title = str(article.get("title") or "")
    category = str(article.get("category") or "")
    content = str(article.get("content_markdown") or "")
    if _normalize_label(title) in BLOCKED_ARTICLE_TITLES:
        return True
    if _normalize_label(category) in BLOCKED_ARTICLE_TITLES:
        return True
    return _looks_like_market_or_service_block(title, category, content)


def _repair_article_content_with_llm(
    page_idx: int,
    page_text: str,
    article: dict[str, Any],
) -> str:
    title = article.get("title") or "Untitled"
    category = article.get("category") or "Unknown"
    current_content = article.get("content_markdown") or ""
    prompt = f"""
The article candidate below was extracted too short from newspaper page {page_idx}.
Use the full OCR text to recover the complete article body for this exact article.

Title: {title}
Category: {category}
Current short content:
{current_content}

--- FULL PAGE OCR TEXT START ---
{page_text}
--- FULL PAGE OCR TEXT END ---

IMPORTANT OUTPUT RULES:
- Return the final JSON object only.
- Do not output analysis, reasoning, notes, explanations, or Markdown.
- Do not output <think> or </think> tags.
- If you have internal reasoning, keep it hidden and output only the JSON.

Return JSON ONLY:
{{
  "content_markdown": "complete recovered article body, or empty string if this is only a teaser/page reference/non-article"
}}
"""
    try:
        print(f"  🛠️ 正在用 LLM 修复短正文：{title}")
        data = _chat_completion_json(
            operation=f"短正文修复：{title}",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(os.getenv("LLM_REPAIR_MAX_TOKENS", "3000")),
            max_retries=int(os.getenv("LLM_REPAIR_MAX_RETRIES", os.getenv("LLM_MAX_RETRIES", "3"))),
        )
        repaired = str(data.get("content_markdown") or "").strip()
        if repaired and len(repaired) > len(str(current_content)):
            return repaired
    except Exception as exc:
        print(f"  ⚠️ 短正文修复失败：{title}，错误：{exc}")
    return ""


def _merge_related_article_fragments(
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_by_signature: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for index, article in enumerate(articles):
        signature = _article_fragment_signature(article)
        if signature is None:
            continue
        grouped_by_signature.setdefault(signature, []).append(index)

    merged_by_first_index: dict[int, dict[str, Any]] = {}
    skipped_indexes: set[int] = set()
    for indexes in grouped_by_signature.values():
        if len(indexes) < 2:
            continue
        group = [articles[index] for index in indexes]
        merged = _merge_article_fragment_group(group)
        merged_by_first_index[indexes[0]] = merged
        skipped_indexes.update(indexes[1:])
        titles = "、".join(str(article.get("title") or "Untitled") for article in group[1:])
        print(f"  🧩 合并同图同栏文章碎片：{group[0].get('title') or 'Untitled'} <- {titles}")

    merged_articles = []
    for index, article in enumerate(articles):
        if index in skipped_indexes:
            continue
        merged_articles.append(merged_by_first_index.get(index, article))
    return merged_articles


def _merge_continued_articles(
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_articles: list[dict[str, Any]] = []
    title_to_index: dict[str, int] = {}

    for article in articles:
        title = str(article.get("title") or "").strip()
        base_title = _continued_base_title(title)
        title_key = _normalize_label(base_title)
        if not title_key:
            merged_articles.append(article)
            continue

        if _is_continued_title(title) and title_key in title_to_index:
            target_index = title_to_index[title_key]
            merged_articles[target_index] = _append_article_continuation(
                merged_articles[target_index],
                article,
            )
            print(f"  🧷 合并续文：{title} -> {merged_articles[target_index].get('title')}")
            continue

        normalized = dict(article)
        if _is_continued_title(title):
            normalized["title"] = base_title
        title_to_index.setdefault(title_key, len(merged_articles))
        merged_articles.append(normalized)

    return merged_articles


def _is_continued_title(title: str) -> bool:
    return bool(re.search(r"\bcontinued\b", title, flags=re.IGNORECASE))


def _continued_base_title(title: str) -> str:
    title = re.sub(r"\s*\(?\s*continued\s*\)?\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*[-–—:]\s*continued\s*$", "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title).strip()


def _append_article_continuation(
    article: dict[str, Any],
    continuation: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(article)
    current_content = str(article.get("content_markdown") or "").strip()
    continuation_content = str(continuation.get("content_markdown") or "").strip()
    if continuation_content and _content_overlap_ratio(continuation_content, current_content) < 0.75:
        merged["content_markdown"] = "\n\n".join(
            part for part in [current_content, continuation_content] if part
        )

    images = []
    seen_images = set()
    for source in (article, continuation):
        for image in source.get("images") or []:
            if image and image not in seen_images:
                seen_images.add(image)
                images.append(image)
    merged["images"] = images
    return merged


def _article_fragment_signature(
    article: dict[str, Any],
) -> tuple[str, tuple[str, ...]] | None:
    images = tuple(sorted(str(image) for image in (article.get("images") or []) if image))
    if not images:
        return None
    category = _normalize_label(str(article.get("category") or ""))
    if not category:
        return None
    return category, images


def _merge_article_fragment_group(
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    merged = dict(group[0])
    merged_images = []
    seen_images = set()
    for article in group:
        for image in article.get("images") or []:
            if image and image not in seen_images:
                seen_images.add(image)
                merged_images.append(image)

    merged_content = str(merged.get("content_markdown") or "").strip()
    for fragment in group[1:]:
        fragment_content = str(fragment.get("content_markdown") or "").strip()
        if not fragment_content:
            continue
        if _content_overlap_ratio(fragment_content, merged_content) >= 0.65:
            continue

        fragment_title = str(fragment.get("title") or "").strip()
        if fragment_title and _normalize_label(fragment_title) != _normalize_label(str(merged.get("title") or "")):
            addition = f"#### {fragment_title}\n\n{fragment_content}"
        else:
            addition = fragment_content
        merged_content = "\n\n".join(part for part in [merged_content, addition] if part)

    merged["images"] = merged_images
    merged["content_markdown"] = merged_content
    return merged


def _content_overlap_ratio(fragment_content: str, merged_content: str) -> float:
    fragment_tokens = _content_tokens(fragment_content)
    if not fragment_tokens:
        return 0.0
    merged_tokens = set(_content_tokens(merged_content))
    if not merged_tokens:
        return 0.0
    overlap_count = sum(1 for token in fragment_tokens if token in merged_tokens)
    return overlap_count / len(fragment_tokens)


def _content_tokens(content: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", content)
    ]


def _compile_articles(
    page_idx: int,
    articles: list[dict[str, Any]],
    image_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if not articles:
        return []

    compile_enabled = _env_bool("LLM_COMPILE_ARTICLES", True)
    analyze_images = image_dir is not None and _env_bool(
        "LLM_ANALYZE_ARTICLE_IMAGES",
        True,
    )
    analyze_glossary = _env_bool("LLM_GLOSSARY_ENABLED", True)

    def needs_processing(article: dict[str, Any]) -> bool:
        if compile_enabled:
            if (
                not article.get("compiled_article")
                and str(article.get("content_markdown") or "").strip()
            ):
                return True
        elif not article.get("formatted_markdown"):
            return True
        if analyze_images and article.get("images"):
            if not (
                article.get("image_analysis_complete")
                or isinstance(article.get("image_insights"), list)
            ):
                return True
        if (
            analyze_glossary
            and bool(article.get("compiled_article"))
            and str(article.get("content_markdown") or "").strip()
            and needs_glossary_refresh(article)
        ):
            return True
        return False

    pending_count = sum(needs_processing(article) for article in articles)
    if pending_count == 0:
        return articles

    article_workers = min(
        max(int(os.getenv("LLM_ARTICLE_WORKERS", "3")), 1),
        pending_count,
    )

    def compile_one(article: dict[str, Any]) -> dict[str, Any]:
        if not needs_processing(article):
            return article
        if compile_enabled:
            if article.get("compiled_article"):
                compiled = article
            elif str(article.get("content_markdown") or "").strip():
                compiled = _compile_article_with_llm(page_idx=page_idx, article=article)
            else:
                compiled = article
        else:
            compiled = (
                article
                if article.get("formatted_markdown")
                else _format_article_locally(article)
            )
        if analyze_images and image_dir is not None:
            compiled = _analyze_article_images(compiled, image_dir)
        if (
            analyze_glossary
            and bool(compiled.get("compiled_article"))
            and str(compiled.get("content_markdown") or "").strip()
            and needs_glossary_refresh(compiled)
        ):
            try:
                print(f"  📖 正在用 LLM 整理术语：{compiled.get('title') or 'Untitled'}")
                compiled = enrich_article_glossary(compiled, _chat_completion_json)
            except Exception as exc:
                print(
                    "  ⚠️ 文章术语解读失败，保留正文并稍后重试："
                    f"{compiled.get('title') or 'Untitled'}，错误：{_compact_error(exc)}"
                )
        return compiled

    if article_workers == 1:
        return [compile_one(article) for article in articles]
    print(f"  ⚙️ 第 {page_idx} 页并发处理 {pending_count} 篇待完成文章（workers={article_workers}）。")
    with ThreadPoolExecutor(
        max_workers=article_workers,
        thread_name_prefix=f"article-p{page_idx}",
    ) as executor:
        return list(executor.map(compile_one, articles))


def retry_deferred_compiles(
    articles: list[dict[str, Any]],
    image_dir: Path | None = None,
    on_article_updated: Callable[[int, dict[str, Any]], None] | None = None,
    include_glossary: bool | None = None,
) -> list[dict[str, Any]]:
    """Retry only failed article compilation after all PDF pages are extracted."""
    if (
        not articles
        or not _env_bool("LLM_COMPILE_ARTICLES", True)
        or not _env_bool("DEFERRED_COMPILE_ENABLED", True)
    ):
        return articles

    result = [dict(article) for article in articles]
    rounds = max(int(os.getenv("DEFERRED_COMPILE_ROUNDS", "3")), 0)
    retries_per_round = max(
        int(os.getenv("DEFERRED_COMPILE_RETRIES_PER_ROUND", "1")),
        1,
    )
    workers = max(int(os.getenv("DEFERRED_COMPILE_WORKERS", "2")), 1)
    base_cooldown = max(float(os.getenv("DEFERRED_COMPILE_COOLDOWN_SECONDS", "30")), 0.0)
    analyze_glossary = (
        _env_bool("DEFERRED_COMPILE_GLOSSARY_ENABLED", False)
        if include_glossary is None
        else include_glossary
    ) and _env_bool("LLM_GLOSSARY_ENABLED", True)
    analyze_images = image_dir is not None and _env_bool(
        "LLM_ANALYZE_ARTICLE_IMAGES",
        True,
    )

    pending = _pending_compile_indexes(result)
    if not pending or rounds == 0:
        return result
    print(f"  🕒 页面抓取完成，延迟重试 {len(pending)} 篇编译失败文章。")

    for round_index in range(1, rounds + 1):
        pending = _pending_compile_indexes(result)
        if not pending:
            break
        cooldown_multiplier = 1 if round_index == 1 else 3 * (2 ** (round_index - 2))
        cooldown = base_cooldown * cooldown_multiplier
        if cooldown > 0:
            print(
                f"  ⏳ 延迟编译第 {round_index}/{rounds} 轮将在 "
                f"{cooldown:g}s 后开始，等待接口限流恢复。"
            )
            time.sleep(cooldown)
        print(
            f"  🔄 开始延迟编译第 {round_index}/{rounds} 轮："
            f"{len(pending)} 篇，workers={min(workers, len(pending))}。"
        )

        def retry_one(index: int) -> tuple[int, dict[str, Any]]:
            article = result[index]
            page_number = int(article.get("page") or 1)
            compiled = _compile_article_with_llm(
                page_idx=page_number,
                article=article,
                max_retries=retries_per_round,
                deferred=True,
            )
            if analyze_images and image_dir is not None:
                compiled = _analyze_article_images(compiled, image_dir)
            if (
                analyze_glossary
                and compiled.get("compiled_article")
                and needs_glossary_refresh(compiled)
            ):
                try:
                    print(f"  📖 正在用 LLM 整理术语：{compiled.get('title') or 'Untitled'}")
                    compiled = enrich_article_glossary(compiled, _chat_completion_json)
                except Exception as exc:
                    print(
                        "  ⚠️ 文章术语解读失败，保留正文并稍后重试："
                        f"{compiled.get('title') or 'Untitled'}，错误：{_compact_error(exc)}"
                    )
            return index, compiled

        completed_count = 0
        succeeded_count = 0
        round_started = time.monotonic()

        def record_completed(index: int, compiled: dict[str, Any]) -> None:
            nonlocal completed_count, succeeded_count
            result[index] = compiled
            completed_count += 1
            if compiled.get("compiled_article"):
                succeeded_count += 1
            if on_article_updated is not None:
                try:
                    on_article_updated(index, compiled)
                except Exception as exc:
                    print(f"  ⚠️ 延迟编译 checkpoint 写入失败：{_compact_error(exc)}")
            elapsed = time.monotonic() - round_started
            print(
                f"  📌 延迟编译进度 {completed_count}/{len(pending)}，"
                f"本轮成功 {succeeded_count} 篇，耗时 {elapsed:.1f}s。"
            )

        if workers == 1 or len(pending) == 1:
            for index in pending:
                completed_index, compiled = retry_one(index)
                record_completed(completed_index, compiled)
        else:
            with ThreadPoolExecutor(
                max_workers=min(workers, len(pending)),
                thread_name_prefix="deferred-compile",
            ) as executor:
                futures = {
                    executor.submit(retry_one, index): index
                    for index in pending
                }
                for future in as_completed(futures):
                    completed_index, compiled = future.result()
                    record_completed(completed_index, compiled)

    remaining = _pending_compile_indexes(result)
    if remaining:
        print(f"  ⚠️ 延迟编译结束后仍有 {len(remaining)} 篇失败，本期保持未完成状态。")
    else:
        print("  ✅ 延迟编译队列已全部完成。")
    return result


def _pending_compile_indexes(articles: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, article in enumerate(articles)
        if str(article.get("content_markdown") or "").strip()
        and not article.get("compiled_article")
    ]


def _pending_compile_pages(articles: list[dict[str, Any]]) -> set[int]:
    if not _env_bool("LLM_COMPILE_ARTICLES", True):
        return set()
    return {
        int(article.get("page") or 0)
        for article in articles
        if str(article.get("content_markdown") or "").strip()
        and not article.get("compiled_article")
        and int(article.get("page") or 0) > 0
    }


def _replace_page_articles(
    pages: list[dict[str, Any]],
    articles: list[dict[str, Any]],
) -> None:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for article in articles:
        grouped.setdefault(int(article.get("page") or 0), []).append(article)
    for page in pages:
        page_number = int(page.get("page") or 0)
        if not page.get("error"):
            page["articles"] = grouped.get(page_number, [])


def _analyze_article_images(article: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    image_paths = [str(path) for path in article.get("images") or [] if str(path).strip()]
    if not image_paths or not _env_bool("LLM_ANALYZE_ARTICLE_IMAGES", True):
        return article
    if article.get("image_analysis_complete") or isinstance(article.get("image_insights"), list):
        return article

    image_paths = image_paths[: int(os.getenv("LLM_MAX_IMAGES_PER_ARTICLE", "6"))]
    content_parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"""
Analyze the following newspaper article images in relation to the article.
Title: {article.get('title') or ''}
Article context:
{str(article.get('content_markdown') or '')[:4500]}

For each supplied image, decide whether it is relevant to this article. You may set keep=false for an unrelated or purely decorative image.
If image_type is chart, description must be 80-200 Chinese characters and start exactly with "📊 图表:". Include only 3-5 high-value numbers or turning points that are clearly readable, mark uncertain data as "（数据模糊）", and state the broader trend.
If image_type is image, description must be 20-60 Chinese characters, objective and concise. Omit the description for a decorative image.
Return strict JSON only. image_type must be exactly "chart" or "image". Do not use English double quotes inside description; use 「」 or 『』 instead.

Return:
{{"images":[{{"path":"images/page_1_fig_1.jpg","image_type":"image","keep":true,"description":"图片说明"}}]}}
""",
        }
    ]
    usable_paths: list[str] = []
    for path in image_paths:
        local_path = image_dir / Path(path).name
        if not local_path.exists():
            continue
        try:
            image_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
        except OSError:
            continue
        usable_paths.append(path)
        content_parts.append(
            {
                "type": "text",
                "text": f"Image path: {path}",
            }
        )
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high",
                },
            }
        )
    if not usable_paths:
        return article

    try:
        data = _chat_completion_json(
            operation=f"文章图片解读：{article.get('title') or 'Untitled'}",
            messages=[{"role": "user", "content": content_parts}],
            max_tokens=int(os.getenv("LLM_IMAGE_ANALYSIS_MAX_TOKENS", "2400")),
            model_env="OPENAI_VISION_MODEL",
            max_retries=int(os.getenv("LLM_IMAGE_ANALYSIS_MAX_RETRIES", "2")),
        )
        raw_items = data.get("images", []) if isinstance(data, dict) else []
        if not isinstance(raw_items, list):
            return article
        insights = []
        kept_paths = []
        reviewed_count = 0
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            path = str(raw_item.get("path") or "").strip()
            if path not in usable_paths:
                continue
            reviewed_count += 1
            if raw_item.get("keep", True) is False:
                continue
            image_type = str(raw_item.get("image_type") or "image").strip().lower()
            if image_type not in {"chart", "image"}:
                image_type = "image"
            description = str(raw_item.get("description") or "").strip().replace('"', "「")
            if description:
                if image_type == "chart" and not description.startswith("📊 图表:"):
                    description = "📊 图表: " + description
            insights.append(
                {"path": path, "image_type": image_type, "description": description}
            )
            kept_paths.append(path)
        if reviewed_count:
            normalized = dict(article)
            normalized["images"] = kept_paths
            normalized["image_insights"] = insights
            normalized["image_analysis_complete"] = True
            return normalized
    except Exception as exc:
        print(f"  ⚠️ 文章图片解读失败，保留原图片：{article.get('title') or 'Untitled'}，错误：{_compact_error(exc)}")
    return article


def _mark_formatted(article: dict[str, Any]) -> dict[str, Any]:
    marked = dict(article)
    marked["formatted_markdown"] = True
    return marked


def _format_article_locally(article: dict[str, Any]) -> dict[str, Any]:
    formatted = dict(article)
    cleaned_content = _simple_paragraph_cleanup(
        str(article.get("content_markdown") or "")
    )
    if str(article.get("local_recovery") or "").startswith("barrons_"):
        cleaned_content = _strip_barrons_end_mark(cleaned_content)
    formatted["content_markdown"] = cleaned_content
    formatted["formatted_markdown"] = True
    formatted["compiled_article"] = False
    formatted.setdefault("compile_status", "pending")
    return formatted


def _compile_article_with_llm(
    page_idx: int,
    article: dict[str, Any],
    max_retries: int | None = None,
    deferred: bool = False,
) -> dict[str, Any]:
    title = article.get("title") or "Untitled"
    category = article.get("category") or "Unknown"
    content = str(article.get("content_markdown") or "").strip()
    prompt = f"""
Compile the extracted OCR article into a structured newspaper article record.

Rules:
- Preserve the article's meaning and facts. Do not add new facts.
- Do not summarize.
- Repair OCR hyphenation caused by newspaper columns, e.g. "govern- ment" -> "government".
- Repair extraction artifacts that insert spaces between letters inside one word, e.g. "c o m m i t t e d" -> "committed".
- Merge visual line breaks that are only caused by narrow newspaper columns.
- Insert paragraph breaks where the meaning changes.
- Preserve bylines, subheadings, bullet-like lines, and italic page references when they are part of the article.
- Remove page-navigation markers such as "Please turn to page A4" and "Continued from Page One" after article fragments have been merged.
- Remove obvious newspaper boilerplate, copyright notices, subscription ads, printer information, and unrelated snippets if they slipped into the body.
- Produce Chinese analysis based only on the article.
- Translate each paragraph into fluent, natural Chinese that reads like it was originally written in Chinese
- Focus on meaning and natural expression, not literal word-for-word translation
- For proper nouns (people, organizations, companies, laws, policies, events, works, acronyms) that need context, provide both Chinese name and English original at first mention, e.g. 美国证券交易委员会（SEC）
- For well-known Chinese translations (like 美联储 for Federal Reserve, 华尔街 for Wall Street), use the standard Chinese term
- Preserve the original article's tone and style - if it's analytical, the Chinese should be analytical; if it's narrative, the Chinese should be narrative
- Do NOT translate common English words into Chinese when they are part of a proper noun (e.g., keep "Apple Inc." as "苹果公司（Apple Inc.）", not "苹果公司（苹果公司）")
- Return JSON only.
- Do not output analysis, reasoning, notes, explanations, Markdown fences, or <think> tags outside the JSON.

Chinese summary requirements:
- Write a cohesive, flowing Chinese summary of 400-500 characters (not exceeding 600 characters)
- Naturally integrate the core message, key arguments, supporting evidence, and potential implications
- Use smooth, professional Chinese prose - do NOT use bullet points, numbered lists, or section headers
- Write as if explaining the article to a knowledgeable Chinese reader
- The summary should read like a well-written newspaper analysis piece, not a structured outline
- Aim for depth and insight rather than brevity - provide substantive analysis

Page: {page_idx}
Title: {title}
Category: {category}

--- RAW ARTICLE TEXT START ---
{content}
--- RAW ARTICLE TEXT END ---

Return JSON ONLY:
{{
  "title_zh": "中文标题",
  "summary_md": "中文结构化摘要 Markdown",
  "content_markdown": "formatted English Markdown with paragraphs separated by blank lines",
  "paragraphs": [
    {{
      "en_text": "English paragraph text",
      "zh_text": "中文翻译",
      "role": "body"
    }}
  ]
}}
"""
    try:
        print(f"  🧠 正在用 LLM 编译文章结构：{title}")
        data = _compile_article_data_with_retries(
            operation=f"文章结构编译：{title}",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_compile_token_budget(content),
            max_retries=max_retries or int(
                os.getenv(
                    "LLM_COMPILE_INITIAL_RETRIES",
                    os.getenv("LLM_COMPILE_MAX_RETRIES", "1"),
                )
            ),
        )
        compiled = dict(article)
        compiled["title_zh"] = str(data.get("title_zh") or "").strip()
        compiled["summary_md"] = str(data.get("summary_md") or "").strip()
        compiled["content_markdown"] = str(data.get("content_markdown") or "").strip() or _simple_paragraph_cleanup(content)
        compiled["paragraphs"] = _normalize_compiled_paragraphs(
            data.get("paragraphs"),
            compiled["content_markdown"],
        )
        for key in (
            "glossary_entries",
            "term_annotations",
            "glossary_analysis_complete",
            "glossary_version",
        ):
            compiled.pop(key, None)
        compiled["glossary_invalidated"] = True
        compiled["formatted_markdown"] = True
        compiled["compiled_article"] = True
        compiled["compile_status"] = "complete"
        compiled["compile_attempts"] = int(article.get("compile_attempts") or 0) + 1
        compiled.pop("last_compile_error", None)
        compiled.pop("next_retry_at", None)
        return compiled
    except Exception as exc:
        phase = "延迟编译仍失败" if deferred else "文章结构编译失败，已加入延迟重试队列"
        print(f"  ⚠️ {phase}：{title}，错误：{_compact_error(exc)}")
        failed = _format_article_locally(article)
        failed["compile_status"] = "pending"
        failed["compile_attempts"] = int(article.get("compile_attempts") or 0) + 1
        failed["last_compile_error"] = _compact_error(exc, limit=500)
        failed["next_retry_at"] = int(
            time.time()
            + max(float(os.getenv("DEFERRED_COMPILE_COOLDOWN_SECONDS", "30")), 0.0)
        )
        return failed


def _compile_token_budget(content: str) -> int:
    configured_max = max(int(os.getenv("LLM_COMPILE_MAX_TOKENS", "7000")), 1000)
    configured_min = max(int(os.getenv("LLM_COMPILE_MIN_TOKENS", "5000")), 1000)
    word_count = len(
        re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", content)
    )
    estimated = int(3500 + word_count * 3.5)
    return min(configured_max, max(configured_min, estimated))


def _compile_article_data_with_retries(
    operation: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            data = _chat_completion_json(
                operation=operation,
                messages=messages,
                max_tokens=max_tokens,
                max_retries=1,
            )
            data = _coerce_compiled_article_data(data)
            if isinstance(data, dict):
                quality_issue = _compiled_data_quality_issue(data)
                if quality_issue:
                    raise ValueError(quality_issue)
                return data
            raise ValueError(f"文章结构编译 JSON 不是 object（type={type(data).__name__}）")
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                print(
                    f"  🔁 {operation} 第 {attempt}/{max_retries} 次结构不合格，"
                    f"准备重试：{_compact_error(exc)}"
                )
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(str(last_error) if last_error else "文章结构编译 JSON 无效")


def _compiled_data_quality_issue(data: dict[str, Any]) -> str | None:
    if not _env_bool("STRICT_LLM_COMPILE", True):
        return None

    if not str(data.get("title_zh") or "").strip():
        return "文章结构编译缺少 title_zh"
    if not str(data.get("summary_md") or "").strip():
        return "文章结构编译缺少 summary_md"

    paragraphs = data.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        return "文章结构编译缺少 paragraphs"

    valid_paragraphs = [
        paragraph for paragraph in paragraphs
        if isinstance(paragraph, dict)
        and (str(paragraph.get("en_text") or "").strip() or str(paragraph.get("zh_text") or "").strip())
    ]
    if not valid_paragraphs:
        return "文章结构编译 paragraphs 为空"

    zh_count = sum(
        1 for paragraph in valid_paragraphs
        if str(paragraph.get("zh_text") or "").strip()
    )
    min_ratio = float(os.getenv("STRICT_LLM_COMPILE_MIN_ZH_RATIO", "0.6"))
    if zh_count / len(valid_paragraphs) < min_ratio:
        return f"文章结构编译中文段落覆盖不足（{zh_count}/{len(valid_paragraphs)}）"
    return None


def _coerce_compiled_article_data(data: Any) -> Any:
    if isinstance(data, dict):
        for key in ("article", "data", "result", "record"):
            nested = data.get(key)
            if isinstance(nested, dict):
                return nested
        articles = data.get("articles")
        if isinstance(articles, list) and articles and isinstance(articles[0], dict):
            return articles[0]
        return data

    if isinstance(data, list):
        dict_items = [item for item in data if isinstance(item, dict)]
        if not dict_items:
            return data
        for item in dict_items:
            if any(key in item for key in ("title_zh", "summary_md", "content_markdown")):
                return item
        if all("en_text" in item or "zh_text" in item for item in dict_items):
            content_markdown = "\n\n".join(
                str(item.get("en_text") or "").strip()
                for item in dict_items
                if str(item.get("en_text") or "").strip()
            )
            return {
                "content_markdown": content_markdown,
                "paragraphs": dict_items,
            }
    return data


def _normalize_compiled_paragraphs(raw_paragraphs: Any, content_markdown: str) -> list[dict[str, str]]:
    if isinstance(raw_paragraphs, list):
        paragraphs = []
        for paragraph in raw_paragraphs:
            if not isinstance(paragraph, dict):
                continue
            en_text = str(paragraph.get("en_text") or "").strip()
            zh_text = str(paragraph.get("zh_text") or "").strip()
            role = str(paragraph.get("role") or "body").strip() or "body"
            if en_text or zh_text:
                paragraphs.append({"en_text": en_text, "zh_text": zh_text, "role": role})
        if paragraphs:
            return paragraphs

    return [
        {"en_text": paragraph, "zh_text": "", "role": "body"}
        for paragraph in re.split(r"\n\s*\n", content_markdown)
        if paragraph.strip()
    ]


def _simple_paragraph_cleanup(content: str) -> str:
    content = re.sub(
        r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b",
        lambda match: re.sub(r"\s+", "", match.group(0)),
        content,
    )
    content = re.sub(
        r"\b(?:please\s*)?turn\s*to\s*page\s*[A-Z]?\d+\b",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"\bcontinued\s*from\s*page\s*(?:one|[A-Z]?\d+)\b",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"(\w+)-\s+(\w+)", _repair_line_break_hyphen, content)
    content = re.sub(r"\s+", " ", content).strip()
    sentence_boundary = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z“‘\"'])")
    sentences = sentence_boundary.split(content)
    paragraphs = []
    current = []
    for sentence in sentences:
        current.append(sentence)
        if sum(len(part.split()) for part in current) >= 90:
            paragraphs.append(" ".join(current).strip())
            current = []
    if current:
        paragraphs.append(" ".join(current).strip())
    return "\n\n".join(paragraphs)


def _strip_barrons_end_mark(content: str) -> str:
    return re.sub(r"\s+B\s*$", "", content).rstrip()


def _repair_line_break_hyphen(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    combined = left + right
    return combined if len(wordninja.split(combined)) == 1 else f"{left}-{right}"


def _article_reject_reason(article: dict[str, Any]) -> str | None:
    title = str(article.get("title") or "").strip()
    category = str(article.get("category") or "").strip()
    content = str(article.get("content_markdown") or "").strip()
    normalized_title = _normalize_label(title)

    if article.get("bypass_min_article_words"):
        return None if content else "正文为空"

    if normalized_title in BLOCKED_ARTICLE_TITLES:
        return "标题黑名单"
    if _looks_like_advertisement(title, category, content):
        return "疑似广告/推广内容"
    if not content:
        return "正文为空"

    word_count = len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", content))
    min_words = int(os.getenv("MIN_ARTICLE_WORDS", str(MIN_ARTICLE_WORDS)))
    if word_count < min_words:
        return f"正文过短（{word_count} words < {min_words}）"
    if _looks_like_page_reference_teaser(title, content):
        return "疑似跳转短讯/目录摘要"
    if _looks_like_market_or_service_block(title, category, content):
        return "疑似行情/服务信息"
    return None


def _strip_whats_news_module_text(
    page_text: str,
    module_text: str,
    whats_news: dict[str, Any] | None,
) -> str:
    """Remove one coordinate-cropped directory occurrence from full-page OCR."""
    if not _whats_news_items(whats_news):
        return page_text
    if module_text and module_text in page_text:
        return page_text.replace(module_text, "", 1)

    module_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in str(module_text or "").splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ]
    remaining = Counter(module_lines)
    output: list[str] = []
    for line in page_text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        exact = normalized if remaining.get(normalized, 0) else None
        fuzzy = None
        if not exact and len(normalized) >= 18:
            fuzzy = next(
                (
                    candidate
                    for candidate, count in remaining.items()
                    if count > 0
                    and abs(len(candidate) - len(normalized)) <= max(8, len(candidate) // 5)
                    and SequenceMatcher(None, normalized, candidate).ratio() >= 0.9
                ),
                None,
            )
        matched = exact or fuzzy
        if matched:
            remaining[matched] -= 1
            continue
        output.append(line)
    return "\n".join(output)


def _filter_articles_against_whats_news(
    articles: list[dict[str, Any]],
    whats_news: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    item_tokens = [_summary_tokens(item["text"]) for item in _whats_news_items(whats_news)]
    item_tokens = [tokens for tokens in item_tokens if len(tokens) >= 4]
    if not item_tokens:
        return articles

    directory_counts: Counter[str] = Counter()
    for tokens in item_tokens:
        directory_counts.update(tokens)
    kept: list[dict[str, Any]] = []
    for article in articles:
        content_tokens = _summary_tokens(str(article.get("content_markdown") or ""))
        content_counts = Counter(content_tokens)
        matched = sum((content_counts & directory_counts).values())
        content_coverage = matched / max(1, len(content_tokens))
        best_item_coverage = max(
            sum((content_counts & Counter(tokens)).values()) / len(tokens)
            for tokens in item_tokens
        )
        mostly_directory = (
            len(content_tokens) >= 4
            and content_coverage >= 0.65
            and best_item_coverage >= 0.6
        )
        if mostly_directory:
            print(
                f"  🧹 丢弃非正文文章：{article.get('title') or 'Untitled'}"
                "（头版目录摘要）"
            )
            continue
        kept.append(article)
    return kept


def _whats_news_items(whats_news: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(whats_news, dict):
        return []
    return [
        item
        for group in whats_news.get("groups") or []
        if isinstance(group, dict)
        for item in group.get("items") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]


def _summary_tokens(value: str) -> list[str]:
    normalized = re.sub(
        r"(?i)(?<![A-Z0-9])[A-Z]\d{1,3}(?![A-Z0-9])",
        " ",
        value,
    )
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", normalized.lower())


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _looks_like_page_reference_teaser(title: str, content: str) -> bool:
    compact = re.sub(r"\s+", " ", content).strip()
    title_words = set(_normalize_label(title).split())
    content_words = set(_normalize_label(compact).split())
    mostly_title = bool(title_words) and len(title_words & content_words) >= max(2, len(title_words) // 2)
    has_page_ref = bool(
        re.search(r"\b(?:page|pages|p)\s*\d+\b", compact, flags=re.IGNORECASE)
    )
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", compact))
    return has_page_ref and mostly_title and word_count < max(120, int(os.getenv("MIN_ARTICLE_WORDS", str(MIN_ARTICLE_WORDS))) * 2)


def _looks_like_market_or_service_block(title: str, category: str, content: str) -> bool:
    haystack = _normalize_label(" ".join([title, category, content[:500]]))
    blocked_phrases = {
        "market data",
        "financial times share service",
        "managed funds service",
        "share service",
        "funds service",
        "stock market data",
    }
    return any(phrase in haystack for phrase in blocked_phrases)


def _looks_like_advertisement(title: str, category: str, content: str) -> bool:
    normalized_category = _normalize_label(category)
    if normalized_category in {
        "advertisement",
        "advertising",
        "sponsored content",
        "paid post",
        "brand content",
    }:
        return True

    haystack = _normalize_label(" ".join([title, content[:1200]]))
    ad_markers = {
        "now at etrade",
        "check out etrade com",
        "call today for a free",
        "promotional offer",
        "terms and conditions apply",
        "paid advertisement",
        "terms conditions restrictions and capacity constraints",
        "mansionglobal com newsletters",
        "your new daily real estate obsession",
    }
    return any(marker in haystack for marker in ad_markers)


def _identify_articles_from_page_image(
    page_idx: int,
    page_img: Any,
    page_figs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt = f"""
You are a professional OCR and newspaper layout analysis assistant.
Analyze this financial newspaper page image and identify ALL distinct news articles/items on the page.
Do NOT merge separate articles.
Do NOT return market data, FINANCIAL TIMES SHARE SERVICE, MANAGED FUNDS SERVICE,
ads, fund listings, stock lists, service directories, crosswords, puzzles, weather,
mastheads, What's News / What’s News or FT Briefing summary bullets, or very short page-reference teasers.

Available cropped image assets on this page:
{json.dumps(page_figs, ensure_ascii=False)}

For each article return:
1. title
2. category
3. content_markdown: complete article text in clean Markdown
   - Use readable paragraphs separated by blank lines.
   - Merge visual line breaks caused by newspaper columns.
   - Do not summarize.
4. images: array of matching image rel_path values, such as "images/page_{page_idx}_fig_1.jpg"

IMPORTANT OUTPUT RULES:
- Return the final JSON object only.
- Do not output analysis, reasoning, notes, explanations, or Markdown.
- Do not output <think> or </think> tags.
- If you have internal reasoning, keep it hidden and output only the JSON.

Return JSON ONLY:
{{
  "articles": [
    {{
      "title": "Article Title",
      "category": "Section Name",
      "content_markdown": "Full article text...",
      "images": []
    }}
  ]
}}
"""

    content = _chat_completion_text(
        operation=f"第 {page_idx} 页视觉文章识别",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{_page_image_to_base64(page_img)}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=int(os.getenv("VISION_LLM_MAX_TOKENS", "6000")),
        model_env="OPENAI_VISION_MODEL",
        max_retries=int(os.getenv("VISION_LLM_MAX_RETRIES", os.getenv("LLM_MAX_RETRIES", "3"))),
    )
    return _extract_articles_from_json(content)


def _page_image_to_base64(page_img: Any) -> str:
    buffer = io.BytesIO()
    page_img.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _render_pages_to_markdown(all_pages: list[dict[str, Any]]) -> str:
    sections = []
    for page in all_pages:
        articles = page.get("articles", [])
        if not articles:
            continue

        page_heading = f"Page {page.get('page')}"
        if page.get("print_page_label"):
            page_heading += f" | {page['print_page_label']}"
        if page.get("print_section"):
            page_heading += f" · {page['print_section']}"
        sections.append(f"## {page_heading}")
        for article in articles:
            title = article.get("title") or "Untitled Article"
            category = article.get("category")
            images = article.get("images") or []
            content = article.get("content_markdown") or ""

            sections.append(f"### {title}")
            if category:
                sections.append(f"**Category**: {category}")
            for image in images:
                sections.append(f"![{title}]({image})")
            if content:
                sections.append(content)

    return "\n\n".join(sections).strip() or "_未提取到有效文章。_"


def _flatten_page_articles(all_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    article_index = 1
    for page in all_pages:
        page_number = page.get("page")
        for page_article_index, article in enumerate(page.get("articles", []), start=1):
            if not isinstance(article, dict):
                continue

            flattened_article = dict(article)
            flattened_article["page"] = page_number
            flattened_article["page_article_index"] = page_article_index
            flattened_article["article_index"] = article_index
            for field in (
                "print_page_label",
                "print_section",
                "print_page_source",
            ):
                flattened_article[field] = page.get(field)
            flattened_article.setdefault("source_pages", [page_number])
            flattened.append(flattened_article)
            article_index += 1
    return flattened


def _build_image_only_articles(
    page_idx: int,
    page_figs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not page_figs:
        return [
            {
                "title": f"Page {page_idx}",
                "category": "Image-only Page",
                "images": [],
                "content_markdown": "_本页未识别到文本或图片资产。_",
            }
        ]

    return [
        {
            "title": f"Page {page_idx} Image Assets",
            "category": "Image-only Page",
            "images": _figure_paths(page_figs),
            "content_markdown": "_本页未识别到可用文本，已保留提取出的图片/图表资产。_",
        }
    ]


def _build_ocr_fallback_articles(
    page_idx: int,
    page_text: str,
    page_figs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "title": _guess_title_from_ocr(page_text) or f"Page {page_idx} OCR Text",
            "category": "OCR Fallback",
            "images": _figure_paths(page_figs),
            "content_markdown": page_text.strip(),
        }
    ]


def _figure_paths(page_figs: list[dict[str, Any]]) -> list[str]:
    return [
        fig["rel_path"]
        for fig in page_figs
        if isinstance(fig, dict) and fig.get("rel_path")
    ]


def _guess_title_from_ocr(page_text: str) -> str | None:
    for line in page_text.splitlines():
        title = re.sub(r"\s+", " ", line).strip()
        if len(title) >= 8:
            return title[:120]
    return None


def _is_page_level_fallback_cache(page_result: dict[str, Any]) -> bool:
    articles = page_result.get("articles") or []
    if len(articles) != 1 or not isinstance(articles[0], dict):
        return False

    category = str(articles[0].get("category") or "")
    title = str(articles[0].get("title") or "")
    fallback_title = bool(
        re.fullmatch(
            r"Page\s+\d+(?:\s+(?:OCR Text|Image Assets))?",
            title,
            flags=re.IGNORECASE,
        )
    )
    return category in {"OCR Fallback", "Image-only Page"} or fallback_title


def _cache_needs_reparse(page_result: dict[str, Any]) -> bool:
    articles = page_result.get("articles") or []
    for article in articles:
        if not isinstance(article, dict):
            continue
        if article.get("compiled_article"):
            continue
        reason = _article_reject_reason(article)
        if reason and "正文过短" in reason:
            return True
    return False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
