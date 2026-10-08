from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from glossary_enricher import (
    GLOSSARY_VERSION,
    normalize_existing_glossary_entries,
    normalize_term_annotations,
)
from metadata_parser import PdfMetadata


DATABASE_INDEX_FILENAME = "database_index.js"
DATABASE_FILENAME = "database.js"
_UNSET = object()

_WHATS_NEWS_STOP_WORDS = {
    "about", "after", "again", "against", "also", "among", "and", "are",
    "because", "been", "before", "being", "between", "but", "can", "could",
    "did", "does", "for", "from", "had", "has", "have", "her", "his",
    "into", "its", "more", "new", "not", "over", "said", "that", "the",
    "their", "them", "they", "this", "through", "under", "was", "were",
    "will", "with", "would", "year", "years",
}

# 标准分类列表
STANDARD_CATEGORIES = [
    "UP & DOWN WALL STREET", "STREETWISE", "REVIEW & PREVIEW",
    "INCOME INVESTING", "FUNDS QUARTERLY", "THE ECONOMY", "TECH TRADER",
    "MARKET WEEK", "INTERNATIONAL TRADER", "THE STRIKING PRICE",
    "INSIDE SCOOP", "WINNERS & LOSERS", "MARKET VIEW", "OTHER VOICES",
    "RETIREMENT MAILBAG",
    "U.S. NEWS", "WORLD NEWS", "PERSONAL JOURNAL", "SPORTS", "ARTS IN REVIEW",
    "OPINION", "BUSINESS & FINANCE", "TECHNOLOGY", "MARKETS DIGEST",
    "CLOSED-END FUNDS", "HEARD ON THE STREET", "NEW PRIME MINISTER",
    "INTERNATIONAL", "COMPANIES & MARKETS", "UK COMPANIES", "MARKET DATA",
    "FINANCIAL TIMES SHARE SERVICE", "MANAGED FUNDS SERVICE", "ARTS",
    "FT Big Read: Global Trade", "The FT View",
    "BUSINESS", "MARKETS", "EXCHANGE", "ANALYSIS", "REVIEW", "REVIEW & OUTLOOK",
    "NATIONAL", "LIFE & ARTS", "STYLE & FASHION", "ADVENTURE & TRAVEL",
    "EATING & DRINKING", "GEAR & GADGETS", "BOOKS", "CULTURE & ENTERTAINMENT",
    "TAX REPORT", "WELLNESS & BEAUTY", "SHORTCUTS: BUSINESS",
    "LETTERS TO THE EDITOR", "OBITUARY", "UK Politics / Economy",
    "Technology / Business", "House & Home", "Money", "FT Weekend",
    "FT Globetrotter", "Letters", "Property Gallery", "Online learning",
    "UK POLITICS", "GLOBAL", "ASIA", "CHINA", "EUROPE",
    "BRITAIN", "Leaders", "Briefing", "BUTTONWOOD", "FREE EXCHANGE",
    "THE AMERICAS", "MIDDLE EAST & AFRICA",
    "SCIENCE & TECHNOLOGY", "FINANCE & ECONOMICS", "CULTURE",
    "THE WORLD THIS WEEK", "BY INVITATION"
]


def _load_category_dict(dict_path: Path) -> dict[str, str]:
    """加载分类中文字典"""
    if not dict_path.exists():
        return {}
    try:
        data = json.loads(dict_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_category_dict(dict_path: Path, category_dict: dict[str, str]) -> None:
    """保存分类中文字典"""
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    dict_path.write_text(
        json.dumps(category_dict, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def _normalize_category(raw: str) -> str:
    """清理category字段，匹配标准分类"""
    if not raw:
        return "General"

    category = raw.strip()
    category_upper = category.upper()

    # 1. 精确匹配（忽略大小写）
    for std in STANDARD_CATEGORIES:
        if category_upper == std.upper():
            return std

    # 2. 前缀匹配：如 "OPINION / AMERICAS" -> "OPINION"
    for std in STANDARD_CATEGORIES:
        std_upper = std.upper()
        if category_upper.startswith(std_upper + " /") or category_upper.startswith(std_upper + "/"):
            return std
        # 处理 " - " 的情况（如 "ARTS IN REVIEW - ART REVIEW"）
        if " - " in category_upper and category_upper.startswith(std_upper + " -"):
            return std
        # 处理 " — " 的情况（如 "Personal Journal — Your Health"）
        if " — " in category_upper and category_upper.startswith(std_upper + " —"):
            return std

    # 3. 保留原值
    return category


def _get_category_zh(category: str, category_dict: dict[str, str]) -> str:
    """获取分类的中文名称"""
    # 精确匹配
    if category in category_dict:
        return category_dict[category]

    # 忽略大小写匹配
    category_upper = category.upper()
    for key, value in category_dict.items():
        if key.upper() == category_upper:
            return value

    # 返回空字符串，需要翻译
    return ""


def _translate_category(category: str) -> str:
    """调用LLM翻译分类名称（简化版本）"""
    simple_translations = {
        "General": "综合",
        "News": "新闻",
        "Business": "商业",
        "Technology": "科技",
        "Sports": "体育",
        "Arts": "艺术",
        "Opinion": "观点",
        "International": "国际",
        "National": "国内",
        "Politics": "政治",
        "Economy": "经济",
        "Finance": "金融",
        "Markets": "市场",
        "Culture": "文化",
        "Lifestyle": "生活方式",
        "Travel": "旅行",
        "Food": "美食",
        "Health": "健康",
        "Science": "科学",
        "Education": "教育",
        "Personal Journal": "个人专栏",
        "World News": "世界新闻",
        "U.S. News": "美国新闻",
        "Business & Finance": "商业金融",
        "Arts in Review": "艺术评论",
        "Life & Arts": "生活艺术",
        "Style & Fashion": "时尚",
        "Adventure & Travel": "旅行探险",
        "Eating & Drink": "餐饮",
        "Gear & Gadgets": "装备",
        "Books": "书评",
        "Culture & Entertainment": "文化娱乐",
        "Tax Report": "税务报告",
        "Wellness & Beauty": "健康美容",
        "Heard on the Street": "华尔街见闻",
        "Markets Digest": "市场摘要",
        "Closed-End Funds": "封闭式基金",
        "FT Big Read": "FT深度",
        "The FT View": "FT社论",
        "UK Politics / Economy": "英国政治经济",
        "Technology / Business": "科技商业",
        "House & Home": "家居",
        "Money": "理财",
        "FT Weekend": "FT周末",
        "FT Globetrotter": "FT环球旅行",
        "Letters": "读者来信",
        "Property Gallery": "房产",
        "Online learning": "在线学习",
        "UK Politics": "英国政治",
        "UK Companies": "英国公司",
        "Companies & Markets": "公司与市场",
        "Market Data": "市场数据",
        "Financial Times Share Service": "金融时报股票服务",
        "Managed Funds Service": "管理基金服务",
        "Global": "全球",
        "Asia": "亚洲",
        "China": "中国",
        "Europe": "欧洲",
        "Britain": "英国",
        "Leaders": "社论",
        "Briefing": "简报",
        "Buttonwood": "梧桐树",
        "Free Exchange": "自由交流",
        "The Americas": "美洲",
        "Middle East & Africa": "中东非洲",
        "Science & Technology": "科技",
        "Finance & Economics": "财经",
        "The World This Week": "本周世界",
        "By Invitation": "特邀",
        "Worldwide": "世界新闻",
        "Review": "评论",
        "Review & Outlook": "评论展望",
        "Exchange": "外汇",
        "Analysis": "分析",
        "Obituary": "讣告",
    }
    return simple_translations.get(category, category)


def write_pdf_database(
    output_root: Path,
    target_dir: Path,
    metadata: PdfMetadata,
    original_filename: str,
    articles: list[dict[str, Any]],
    is_initial_processing: bool = True,
    pages: list[dict[str, Any]] | None | object = _UNSET,
    front_page: dict[str, Any] | None | object = _UNSET,
    is_weekend: bool | object = _UNSET,
) -> tuple[Path, Path]:
    economist_compatible = metadata.publication_type == "BARRONS"
    pdf_id = build_pdf_id(metadata, original_filename)
    database_path = target_dir / DATABASE_FILENAME
    database_rel_path = database_path.relative_to(output_root).as_posix()
    existing_payload = read_pdf_database(database_path)
    existing_articles = {
        str(article.get("id") or ""): article
        for article in (existing_payload or {}).get("articles") or []
        if isinstance(article, dict) and article.get("id")
    }
    glossary_entries = {
        entry["id"]: entry
        for entry in normalize_existing_glossary_entries(
            (existing_payload or {}).get("glossary")
        )
    }

    # 加载分类字典
    category_dict_path = output_root.parent / "category_dict.json"
    category_dict = _load_category_dict(category_dict_path)
    dict_modified = False

    database_articles = []
    for index, article in enumerate(articles, start=1):
        database_article = build_database_article(
            article=article,
            article_index=index,
            pdf_id=pdf_id,
            metadata=metadata,
            original_filename=original_filename,
            category_dict=category_dict,
        )

        # 检查是否需要翻译新分类
        category = database_article["category"]
        category_zh = database_article.get("category_zh", "")
        if not economist_compatible and not category_zh and category != "General":
            category_zh = _translate_category(category)
            category_dict[category] = category_zh
            database_article["category_zh"] = category_zh
            dict_modified = True

        previous = existing_articles.get(database_article["id"])
        has_current_glossary = bool(article.get("glossary_analysis_complete")) or (
            "term_annotations" in article
        )
        if (
            previous
            and not has_current_glossary
            and not article.get("glossary_invalidated")
        ):
            database_article["term_annotations"] = normalize_term_annotations(
                previous.get("term_annotations"),
                len(database_article["paragraphs"]),
            )
            database_article["glossary_analysis_complete"] = bool(
                previous.get("glossary_analysis_complete")
            )
            database_article["glossary_version"] = int(
                previous.get("glossary_version") or 0
            )
        for entry in normalize_existing_glossary_entries(
            article.get("glossary_entries")
        ):
            glossary_entries[entry["id"]] = entry
        database_articles.append(database_article)

    # 保存更新后的字典
    if dict_modified:
        _save_category_dict(category_dict_path, category_dict)

    referenced_glossary_ids = {
        str(annotation.get("glossary_id") or "")
        for article in database_articles
        for annotation in article.get("term_annotations") or []
        if isinstance(annotation, dict) and annotation.get("glossary_id")
    }
    database_glossary = {
        glossary_id: (
            dict(glossary_entries[glossary_id])
            if economist_compatible
            else {
                key: value for key, value in glossary_entries[glossary_id].items()
                if key != "id"
            }
        )
        for glossary_id in sorted(referenced_glossary_ids)
        if glossary_id in glossary_entries
    }
    source_pages = (
        (existing_payload or {}).get("pages") or []
        if pages is _UNSET
        else pages
    )
    database_pages = []
    for page in source_pages or []:
        normalized_page = dict(page)
        pdf_page = safe_int(normalized_page.get("pdf_page"), 0)
        normalized_page["article_ids"] = [
            article["id"]
            for article in database_articles
            if pdf_page in {
                safe_int(source_page, 0)
                for source_page in (
                    article.get("source_pages") or [article.get("page")]
                )
            }
        ]
        database_pages.append(normalized_page)

    database_front_page = (
        (existing_payload or {}).get("front_page")
        if front_page is _UNSET
        else front_page
    )
    database_front_page = _link_whats_news_articles(
        database_front_page,
        database_articles,
    )

    database_is_weekend = is_weekend
    if database_is_weekend is _UNSET and existing_payload is not None:
        if "is_weekend" in existing_payload:
            database_is_weekend = existing_payload["is_weekend"]

    payload = {
        "id": pdf_id,
        "publication_type": metadata.publication_type,
        "publication_date": metadata.publication_date,
        "original_filename": original_filename,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cover_image": find_cover_image(target_dir),
        "article_count": len(articles),
        "glossary_version": GLOSSARY_VERSION,
        "glossary": database_glossary,
        "articles": database_articles,
    }
    if not economist_compatible:
        payload["pages"] = database_pages
        payload["front_page"] = database_front_page
        # Keep articles last in the legacy schema while Barron's follows the
        # Economist weekly archive field order exactly.
        payload["articles"] = payload.pop("articles")
    if not economist_compatible and database_is_weekend is not _UNSET:
        payload["is_weekend"] = database_is_weekend is True
    database_text = (
        "window.paper_databases = window.paper_databases || {};\n"
        f"window.paper_databases[{json.dumps(pdf_id, ensure_ascii=False)}] = "
        f"{json.dumps(payload, ensure_ascii=False, indent=2)};\n"
    )
    _atomic_write_text(database_path, database_text)

    index_path = update_database_index(
        output_root=output_root,
        pdf_id=pdf_id,
        metadata=metadata,
        original_filename=original_filename,
        database_rel_path=database_rel_path,
        articles=payload["articles"],
        is_initial_processing=is_initial_processing,
        category_dict=category_dict,
        has_front_page=bool(database_front_page),
        is_weekend=database_is_weekend,
    )
    return database_path, index_path


def _link_whats_news_articles(
    front_page: dict[str, Any] | None,
    articles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Attach bilingual article titles to What's News directory entries."""
    if not isinstance(front_page, dict):
        return front_page
    linked = copy.deepcopy(front_page)
    groups = (linked.get("whats_news") or {}).get("groups") or []
    if not isinstance(groups, list):
        return linked

    articles_by_id = {
        str(article.get("id") or ""): article
        for article in articles
        if isinstance(article, dict) and article.get("id")
    }
    used_article_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        for item in group.get("items") or []:
            if not isinstance(item, dict):
                continue
            target = articles_by_id.get(str(item.get("target_article_id") or ""))
            if target is None:
                target = _match_whats_news_article(
                    item,
                    articles,
                    used_article_ids,
                )
            if target is None:
                continue
            target_id = str(target.get("id") or "")
            if target_id:
                item["target_article_id"] = target_id
                used_article_ids.add(target_id)
            if not str(item.get("target_print_page_label") or "").strip():
                inferred_label = target.get("print_page_label")
                if (
                    not inferred_label
                    and str(target.get("publication_type") or "").strip().upper() == "FT"
                    and safe_int(target.get("page"), 0) > 0
                ):
                    inferred_label = str(safe_int(target.get("page"), 0))
                item["target_print_page_label"] = inferred_label
            item["title"] = str(target.get("title") or "").strip()
            item["title_zh"] = str(target.get("title_zh") or "").strip()
    return linked


def _match_whats_news_article(
    item: dict[str, Any],
    articles: list[dict[str, Any]],
    used_article_ids: set[str],
) -> dict[str, Any] | None:
    target_label = str(item.get("target_print_page_label") or "").strip().upper()
    candidates = [
        article
        for article in articles
        if str(article.get("id") or "") not in used_article_ids
        and (
            not target_label
            or str(article.get("print_page_label") or "").strip().upper() == target_label
            or (
                str(article.get("publication_type") or "").strip().upper() == "FT"
                and target_label.isdigit()
                and safe_int(article.get("page"), 0) == int(target_label)
            )
        )
    ]
    query_tokens = _whats_news_tokens(item.get("text"))
    if not target_label:
        candidates = [
            article
            for article in candidates
            if len(query_tokens & _whats_news_tokens(article.get("title"))) >= 2
        ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    scored = [
        (_whats_news_match_score(query_tokens, article), article)
        for article in candidates
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_article = scored[0]
    threshold = 14 if not target_label else 3
    return best_article if best_score >= threshold else None


def _whats_news_match_score(
    query_tokens: set[str],
    article: dict[str, Any],
) -> int:
    if not query_tokens:
        return 0
    title_tokens = _whats_news_tokens(article.get("title"))
    body_tokens = _whats_news_tokens(
        article.get("content_markdown") or article.get("content_raw")
    )
    title_overlap = len(query_tokens & title_tokens)
    body_overlap = len(query_tokens & body_tokens)
    if title_overlap == 0 and body_overlap < 3:
        return 0
    return title_overlap * 7 + body_overlap


def _whats_news_tokens(value: Any) -> set[str]:
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", str(value or "").lower())
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", text):
        if len(token) < 3 or token in _WHATS_NEWS_STOP_WORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def build_pdf_id(metadata: PdfMetadata, original_filename: str) -> str:
    if metadata.publication_type == "BARRONS":
        return f"BARRONS_{metadata.publication_date}_barrons-weekly"
    return "_".join(
        [
            metadata.publication_type,
            metadata.publication_date,
            slugify(Path(original_filename).stem),
        ]
    )


def build_database_article(
    article: dict[str, Any],
    article_index: int,
    pdf_id: str,
    metadata: PdfMetadata,
    original_filename: str,
    category_dict: dict[str, str] | None = None,
) -> dict[str, Any]:
    title = str(article.get("title") or f"Article {article_index}").strip()
    page = safe_int(article.get("page"), 0)
    page_article_index = safe_int(article.get("page_article_index"), article_index)
    economist_compatible = metadata.publication_type == "BARRONS"
    if economist_compatible:
        article_id = f"art_{metadata.publication_date}_{article_index:03d}"
        markdown_path = f"articles/{article_id}.md"
    else:
        article_id = f"{pdf_id}_p{page:02d}_{page_article_index:02d}"
        markdown_path = (
            "articles/"
            f"P{page:02d}_{page_article_index:02d}_{safe_article_title(title)}.md"
        )

    paragraphs = build_paragraphs(article, article_id)
    compiled_article = _is_compiled_article(article, paragraphs)
    annotations = normalize_term_annotations(
        article.get("term_annotations"),
        len(paragraphs),
    )

    # 处理 category
    article_category = str(article.get("category") or "").strip()
    article_category_is_standard = any(
        article_category.upper() == standard.upper()
        for standard in STANDARD_CATEGORIES
    )
    raw_category = article_category
    if (
        economist_compatible
        and not article_category_is_standard
        and article.get("print_section")
    ):
        raw_category = str(article["print_section"])
    raw_category = raw_category or "General"
    category = _normalize_category(raw_category)
    category_zh = ""
    if category_dict:
        category_zh = _get_category_zh(category, category_dict)

    common_tail = {
        "title": title,
        "title_zh": article.get("title_zh") or "",
        "markdown_path": markdown_path,
        "summary_md": article.get("summary_md") or "",
        "compiled_article": compiled_article,
        "compile_status": article.get("compile_status") or (
            "complete" if compiled_article else "pending"
        ),
        "content_markdown": article.get("content_markdown") or "",
        "content_raw": article.get("content_raw") or article.get("content_markdown") or "",
        "paragraphs": paragraphs,
        "images": article.get("images") or [],
        "image_insights": article.get("image_insights") or [],
        "term_annotations": annotations,
        "glossary_analysis_complete": bool(
            article.get("glossary_analysis_complete")
        ),
        "glossary_version": int(article.get("glossary_version") or 0),
    }
    common_header = {
        "id": article_id,
        "publication_type": metadata.publication_type,
        "publication_date": metadata.publication_date,
        "source_pdf": original_filename,
        "page": page,
        "page_article_index": page_article_index,
    }
    if economist_compatible:
        return {**common_header, "category": category, **common_tail}
    return {
        **common_header,
        "print_page_label": article.get("print_page_label"),
        "print_section": article.get("print_section"),
        "print_page_source": article.get("print_page_source"),
        "source_pages": article.get("source_pages") or [page],
        "category": category,
        "category_zh": category_zh,
        **common_tail,
    }


def _is_compiled_article(
    article: dict[str, Any],
    paragraphs: list[dict[str, Any]],
) -> bool:
    if article.get("compiled_article"):
        return True
    if not str(article.get("title_zh") or "").strip():
        return False
    if not str(article.get("summary_md") or "").strip():
        return False
    return any(str(paragraph.get("zh_text") or "").strip() for paragraph in paragraphs)


def build_paragraphs(article: dict[str, Any], article_id: str) -> list[dict[str, Any]]:
    raw_paragraphs = article.get("paragraphs")
    if isinstance(raw_paragraphs, list):
        paragraphs = []
        for index, raw in enumerate(raw_paragraphs, start=1):
            if not isinstance(raw, dict):
                continue
            en_text = str(raw.get("en_text") or raw.get("en_html") or "").strip()
            zh_text = str(raw.get("zh_text") or "").strip()
            role = str(raw.get("role") or "body").strip() or "body"
            if not en_text and not zh_text:
                continue
            paragraphs.append(
                {
                    "para_id": str(raw.get("para_id") or f"{article_id}_p{index}"),
                    "en_text": en_text,
                    "zh_text": zh_text,
                    "role": role,
                }
            )
        if paragraphs:
            return paragraphs

    content = str(article.get("content_markdown") or "").strip()
    content_paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content)
        if paragraph.strip()
    ]
    return [
        {
            "para_id": f"{article_id}_p{index}",
            "en_text": paragraph,
            "zh_text": "",
            "role": "body",
        }
        for index, paragraph in enumerate(content_paragraphs, start=1)
    ]


def update_database_index(
    output_root: Path,
    pdf_id: str,
    metadata: PdfMetadata,
    original_filename: str,
    database_rel_path: str,
    articles: list[dict[str, Any]],
    is_initial_processing: bool = True,
    category_dict: dict[str, str] | None = None,
    has_front_page: bool = False,
    is_weekend: bool | object = _UNSET,
) -> Path:
    index_path = output_root / DATABASE_INDEX_FILENAME
    index_items = read_database_index(index_path)

    # 提取 sections 并生成中文映射
    sections = sorted({str(article.get("category") or "General") for article in articles})
    sections_zh = {}
    if category_dict:
        for section in sections:
            zh = _get_category_zh(section, category_dict)
            if zh:
                sections_zh[section] = zh

    item = {
        "id": pdf_id,
        "publication_type": metadata.publication_type,
        "publication_date": metadata.publication_date,
        "original_filename": original_filename,
        "database_path": database_rel_path,
        "cover_image": find_cover_image(output_root / Path(database_rel_path).parent),
        "article_count": len(articles),
        "sections": sections,
        "titles": [article.get("title") for article in articles if article.get("title")],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if metadata.publication_type != "BARRONS":
        item["sections_zh"] = sections_zh
        item["is_initial_processing"] = is_initial_processing
        item["has_front_page"] = has_front_page

    if metadata.publication_type != "BARRONS" and is_weekend is _UNSET:
        existing_item = next(
            (existing for existing in index_items if existing.get("id") == pdf_id),
            None,
        )
        if existing_item is not None and "is_weekend" in existing_item:
            item["is_weekend"] = existing_item["is_weekend"] is True
    elif metadata.publication_type != "BARRONS":
        item["is_weekend"] = is_weekend is True

    merged = [existing for existing in index_items if existing.get("id") != pdf_id]
    merged.append(item)
    merged.sort(key=lambda value: (value.get("publication_date", ""), value.get("publication_type", ""), value.get("original_filename", "")))

    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        index_path,
        "window.paper_db_index = "
        f"{json.dumps(merged, ensure_ascii=False, indent=2)};\n",
    )
    return index_path


def find_cover_image(target_dir: Path) -> str:
    cover_path = target_dir / "cover.jpg"
    if cover_path.exists():
        return "cover.jpg"
    image_dir = target_dir / "images"
    if image_dir.exists():
        candidates = sorted(image_dir.glob("page_1_fig_*"))
        if candidates:
            return f"images/{candidates[0].name}"
    return ""


def read_database_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []
    text = index_path.read_text(encoding="utf-8")
    match = re.search(r"window\.paper_db_index\s*=\s*([\s\S]*?);\s*$", text)
    if not match:
        return []
    data = json.loads(match.group(1))
    return data if isinstance(data, list) else []


def read_pdf_database(database_path: Path) -> dict[str, Any] | None:
    if not database_path.exists():
        return None
    try:
        text = database_path.read_text(encoding="utf-8")
        match = re.search(
            r"window\.paper_databases\[[^\]]+\]\s*=\s*([\s\S]*?);\s*$",
            text,
        )
        data = json.loads(match.group(1)) if match else None
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "paper"


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_article_title(title: str) -> str:
    safe_title = "".join(
        char for char in title if char.isalnum() or char in (" ", "_", "-")
    ).strip()
    return safe_title[:30] or "Article"
