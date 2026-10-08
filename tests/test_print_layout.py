from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from database_writer import read_database_index, read_pdf_database, write_pdf_database
from metadata_parser import PdfMetadata
from print_layout import (
    PRINT_LAYOUT_VERSION,
    build_front_page,
    build_issue_pages,
    detect_weekend_issue,
    enrich_page_result,
    extract_ft_briefing,
    extract_print_layout,
    extract_whats_news,
    select_ft_briefing_blocks,
    select_whats_news_blocks,
)
from strategies.base import ParseResult
from strategies.strategy_hybrid import _filter_articles_from_block_region
from strategies.strategy_scanned import (
    ScannedPdfStrategy,
    _clean_articles,
    _filter_articles_against_whats_news,
    _strip_whats_news_module_text,
)
from strategies.strategy_scanned import _flatten_page_articles


class PrintLayoutTests(unittest.TestCase):
    def test_detects_ft_weekend_masthead_and_date_line(self) -> None:
        self.assertTrue(detect_weekend_issue("FT", "FT Weekend"))
        self.assertTrue(
            detect_weekend_issue(
                "FT", "SATURDAY 25 JULY / SUNDAY 26 JULY 2026"
            )
        )

    def test_detects_wsj_weekend_masthead_and_date_line(self) -> None:
        self.assertTrue(
            detect_weekend_issue("WSJ", "THE WALL STREET JOURNAL WEEKEND")
        )
        self.assertTrue(
            detect_weekend_issue(
                "WSJ", "SATURDAY/SUNDAY, AUGUST 8 - 9, 2026"
            )
        )

    def test_regular_and_unsupported_issues_are_not_weekend(self) -> None:
        self.assertFalse(
            detect_weekend_issue("WSJ", "TUESDAY, AUGUST 4, 2026")
        )
        self.assertFalse(detect_weekend_issue("TE", "SATURDAY/SUNDAY"))

    def test_first_page_enrichment_records_explicit_weekend_boolean(self) -> None:
        weekend = enrich_page_result(
            {"page": 1},
            "FT",
            1,
            header_text="FT Weekend\nSATURDAY 25 JULY / SUNDAY 26 JULY 2026",
        )
        regular = enrich_page_result(
            {"page": 1},
            "WSJ",
            1,
            header_text="THE WALL STREET JOURNAL\nTUESDAY, AUGUST 4, 2026",
        )
        self.assertTrue(weekend["is_weekend"])
        self.assertFalse(regular["is_weekend"])

    def test_wsj_page_labels_and_shared_section(self) -> None:
        for label in ("A2", "A3"):
            layout = extract_print_layout(
                "WSJ",
                int(label[1:]),
                f"THE WALL STREET JOURNAL\n{label}\nU.S. NEWS\nTariffs reshape trade",
            )
            self.assertEqual(layout["print_page_label"], label)
            self.assertEqual(layout["print_section"], "U.S. NEWS")
            self.assertEqual(layout["print_page_source"], "header")

    def test_wsj_b_section_is_not_derived_from_pdf_page(self) -> None:
        layout = extract_print_layout(
            "WSJ",
            19,
            "B1\nBUSINESS&FINANCE\nTHE WALL STREET JOURNAL",
        )
        self.assertEqual(layout["print_page_label"], "B1")
        self.assertEqual(layout["print_section"], "BUSINESS & FINANCE")

    def test_wsj_page_label_at_start_of_dated_header_line(self) -> None:
        layout = extract_print_layout(
            "WSJ",
            2,
            header_blocks=[
                {
                    "text": "A2 | Tuesday, August 4, 2026\nTHE WALL STREET JOURNAL.",
                    "bbox": [14.47, 18.42, 608.72, 28.83],
                },
                {"text": "U.S. NEWS", "bbox": [256.42, 30.33, 366.77, 62.94]},
            ],
            page_width=630,
        )
        self.assertEqual(layout["print_page_label"], "A2")
        self.assertEqual(layout["print_section"], "U.S. NEWS")
        self.assertEqual(layout["print_page_source"], "header")

    def test_wsj_exchange_section_on_real_b5_header(self) -> None:
        layout = extract_print_layout(
            "WSJ", 19, "THE WALL STREET JOURNAL\nB5\nEXCHANGE"
        )
        self.assertEqual(layout["print_page_label"], "B5")
        self.assertEqual(layout["print_section"], "EXCHANGE")

    def test_ft_and_economist_keep_numeric_page_labels(self) -> None:
        ft = extract_print_layout("FT", 2, "2 FINANCIAL TIMES\nNATIONAL\nNews")
        te = extract_print_layout(
            "TE", 4, "4 The Economist July 25th 2026\nThe world this week"
        )
        self.assertEqual(ft["print_page_label"], "2")
        self.assertEqual(ft["print_section"], "NATIONAL")
        self.assertEqual(te["print_page_label"], "4")
        self.assertEqual(te["print_section"], "THE WORLD THIS WEEK")
        self.assertNotEqual(ft["print_page_label"], "A2")

    def test_native_blocks_prioritize_the_header(self) -> None:
        blocks = [
            {"text": "A6 U.S. NEWS", "bbox": [10, 10, 200, 30]},
            {"text": "A1 appears as an article reference", "bbox": [10, 400, 300, 430]},
        ]
        layout = extract_print_layout(
            "WSJ", 6, "", header_blocks=blocks[:1], page_width=600
        )
        self.assertEqual(layout["print_page_label"], "A6")
        self.assertEqual(layout["print_section"], "U.S. NEWS")

    def test_continued_reference_is_not_a_page_header(self) -> None:
        layout = extract_print_layout(
            "WSJ", 6, header_text="Story continued on A6\nMore reporting"
        )
        self.assertIsNone(layout["print_page_label"])
        self.assertIsNone(layout["print_section"])

        split = extract_print_layout(
            "WSJ", 6, header_text="Story continued on\nA6\nMore reporting"
        )
        self.assertIsNone(split["print_page_label"])

    def test_short_sentence_with_page_reference_is_not_a_header(self) -> None:
        layout = extract_print_layout(
            "WSJ", 6, header_text="Markets closed at A6.\nMore reporting"
        )
        self.assertIsNone(layout["print_page_label"])

    def test_ft_date_is_not_used_as_print_page(self) -> None:
        layout = extract_print_layout(
            "FT", 25, header_text="FINANCIAL TIMES\nJuly 25 2026\nNATIONAL"
        )
        self.assertIsNone(layout["print_page_label"])
        self.assertEqual(layout["print_section"], "NATIONAL")

    def test_split_dates_are_not_numeric_print_pages(self) -> None:
        for publication, masthead in (("FT", "FINANCIAL TIMES"), ("TE", "THE ECONOMIST")):
            layout = extract_print_layout(
                publication,
                25,
                header_text=f"{masthead}\nJuly\n25\n2026",
            )
            self.assertIsNone(layout["print_page_label"])

    def test_standalone_numeric_block_must_be_at_page_edge(self) -> None:
        layout = extract_print_layout(
            "FT",
            25,
            header_blocks=[{"text": "25", "bbox": [250, 20, 270, 35]}],
            page_width=634,
        )
        self.assertIsNone(layout["print_page_label"])


class WhatsNewsTests(unittest.TestCase):
    def test_weekend_tall_group_block_keeps_both_groups_without_right_column(self) -> None:
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "wsj_2026_07_25_page1_tall_block.json").read_text(encoding="utf-8")
        )
        selected = select_whats_news_blocks(
            fixture["blocks"],
            page_width=fixture["page_width"],
            page_height=fixture["page_height"],
        )
        self.assertNotIn("right", [block["id"] for block in selected])
        result = extract_whats_news(
            blocks=fixture["blocks"],
            page_width=fixture["page_width"],
            page_height=fixture["page_height"],
        )
        self.assertEqual([group["name"] for group in result["groups"]], ["Business & Finance", "Worldwide"])
        directory_article = {"title": "Paramount", "source_block_ids": ["b013"]}
        real_article = {"title": "AI Chips", "source_block_ids": ["right"]}
        self.assertEqual(
            _filter_articles_from_block_region(
                [directory_article, real_article], selected
            ),
            [real_article],
        )
    def test_real_native_blocks_use_narrow_left_column(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "wsj_2026_07_27_page1_native_blocks.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        result = extract_whats_news(
            blocks=fixture["blocks"],
            page_width=fixture["page_width"],
            page_height=fixture["page_height"],
        )

        self.assertEqual(
            [group["name"] for group in result["groups"]],
            ["Business & Finance", "Worldwide"],
        )
        items = [item for group in result["groups"] for item in group["items"]]
        self.assertGreaterEqual(len(items), 6)
        item_text = " ".join(item["text"] for item in items)
        self.assertIn("North Korean leader Kim", item_text)
        self.assertNotIn("Tour de France Champion", item_text)
        self.assertNotIn("CONTENTS", item_text)

        enriched = enrich_page_result(
            {"page": 1},
            "WSJ",
            1,
            whats_news_blocks=fixture["blocks"],
            page_width=fixture["page_width"],
            page_height=fixture["page_height"],
        )
        self.assertEqual(enriched["print_page_label"], "A1")
        self.assertEqual(enriched["print_page_source"], "derived")
        self.assertIsNotNone(build_front_page([enriched], "WSJ"))

    def test_extracts_groups_wrapped_items_and_page_references(self) -> None:
        text = """
What's News
Business & Finance
• Nvidia is in talks to invest in OpenAI, a deal that could
reshape the industry. A1
• Hiring slowed sharply. B1.
Worldwide
• Iran's government faced new pressure, A6
• Canada opened a bridge.
A3
"""
        result = extract_whats_news(text)
        self.assertEqual(
            [group["name"] for group in result["groups"]],
            ["Business & Finance", "Worldwide"],
        )
        self.assertEqual(
            [
                item["target_print_page_label"]
                for group in result["groups"]
                for item in group["items"]
            ],
            ["A1", "B1", "A6", "A3"],
        )
        self.assertIsNone(result["groups"][0]["items"][0]["target_article_id"])

    def test_does_not_treat_regular_section_as_whats_news(self) -> None:
        result = extract_whats_news("Business & Finance\nA market report. B1")
        self.assertEqual(result, {"groups": []})

    def test_ft_does_not_treat_wsj_directory_as_briefing(self) -> None:
        result = enrich_page_result(
            {"page": 1},
            "FT",
            1,
            header_text="1 FINANCIAL TIMES",
            whats_news_text=(
                "What's News\nWorldwide\n"
                "• This is a summary item that points elsewhere. A6"
            ),
        )
        self.assertNotIn("whats_news", result)

    def test_ft_briefing_builds_front_page_directory(self) -> None:
        briefing = """
Briefing
> Oil falls as Bessent fuels hopes for a Hormuz deal
Treasury secretary Scott Bessent signalled that the US might be close to a deal. — PAGE 6
> Public contracts shake-up
Onerous procurement rules are to be stripped back. — PAGE 2
> OpenAI bites into Apple
The AI start-up accused the iPhone maker of waging a personal lawsuit. — PAGE 7
"""
        enriched = enrich_page_result(
            {"page": 1},
            "FT",
            1,
            header_text="FINANCIAL TIMES",
            whats_news_text=briefing,
        )
        self.assertEqual(enriched["print_page_label"], "1")
        self.assertEqual(enriched["print_section"], "PAGE ONE")
        self.assertEqual(
            [item["target_print_page_label"] for item in enriched["whats_news"]["groups"][0]["items"]],
            ["6", "2", "7"],
        )
        front_page = build_front_page([enriched], "FT")
        assert front_page is not None
        self.assertEqual(front_page["directory_name"], "Briefing")
        self.assertEqual(front_page["print_page_label"], "1")

    def test_ft_briefing_uses_only_coordinate_confirmed_right_rail(self) -> None:
        blocks = [
            {"id": "article", "text": "Briefing the cabinet on policy", "bbox": [80, 180, 420, 260]},
            {"id": "heading", "text": "Briefing", "bbox": [500, 100, 580, 125]},
            {"id": "item", "text": "> Oil prices fall after talks. — PAGE 6", "bbox": [500, 130, 590, 220]},
        ]
        selected = select_ft_briefing_blocks(
            blocks,
            page_width=600,
            page_height=800,
        )
        self.assertEqual([block["id"] for block in selected], ["heading", "item"])
        parsed = extract_ft_briefing(
            blocks=blocks,
            page_width=600,
            page_height=800,
        )
        self.assertEqual(
            parsed["groups"][0]["items"][0]["target_print_page_label"],
            "6",
        )

    def test_ft_weekend_without_briefing_keeps_page_navigation(self) -> None:
        weekend_sidebar = """
Lunch with the FT
Met Office's Penny Endersby
LIFE & ARTS
Forever chemicals
A quest to detoxify your life
FT WEEKEND MAGAZINE
"""
        enriched = enrich_page_result(
            {"page": 1},
            "FT",
            1,
            header_text="FT Weekend",
            whats_news_text=weekend_sidebar,
        )
        self.assertNotIn("whats_news", enriched)
        self.assertIsNone(build_front_page([enriched], "FT"))

    def test_stale_non_wsj_cache_cannot_build_front_page(self) -> None:
        stale = {
            "page": 1,
            "print_page_label": "A1",
            "whats_news": {"groups": [{"name": "Worldwide", "items": [{}]}]},
        }
        self.assertIsNone(build_front_page([stale], "FT"))

    def test_only_wsj_pdf_page_one_can_derive_a1(self) -> None:
        directory = "What's News\nWorldwide\n• A directory summary points inside. A6"
        self.assertIsNone(
            enrich_page_result(
                {"page": 2}, "WSJ", 2, whats_news_text=directory
            ).get("print_page_label")
        )
        self.assertIsNone(
            enrich_page_result(
                {"page": 1}, "FT", 1, whats_news_text=directory
            ).get("print_page_label")
        )

    def test_full_page_text_without_bullets_is_not_consumed(self) -> None:
        result = extract_whats_news(
            "What's News\nWorldwide\nMain headline story continues here. A6\nContents"
        )
        self.assertEqual(result, {"groups": []})

    def test_scanned_cleaner_excludes_whats_news(self) -> None:
        articles = _clean_articles(
            [{"title": "What's News", "category": "Worldwide", "content_markdown": "word " * 100}]
        )
        self.assertEqual(articles, [])

    def test_scanned_directory_match_filters_single_item_but_keeps_real_body(self) -> None:
        whats_news = {"groups": [{"name": "Worldwide", "items": [{"text": "Diplomacy resumed after officials met in Geneva"}]}]}
        directory = {"title": "Diplomacy Resumes", "category": "Worldwide", "content_markdown": "Diplomacy resumed after officials met in Geneva. A6"}
        real_article = {
            "title": "Diplomacy Resumes",
            "category": "Business & Finance",
            "content_markdown": "Diplomacy resumed after officials met in Geneva. " + "A full reported article with interviews and original detail. " * 30,
        }
        self.assertEqual(_filter_articles_against_whats_news([directory, real_article], whats_news), [real_article])

    def test_scanned_removes_cropped_module_before_llm(self) -> None:
        module = "What's News\nWorldwide\n• Diplomacy resumed after officials met. A6"
        parsed = extract_whats_news(module)
        page = module + "\nMain Headline\nA full reported article follows."
        cleaned = _strip_whats_news_module_text(page, module, parsed)
        self.assertNotIn("Diplomacy resumed", cleaned)
        self.assertIn("Main Headline", cleaned)

    def test_empty_native_metadata_does_not_drop_fallback_whats_news(self) -> None:
        existing = {
            "page": 1,
            "print_page_label": "A1",
            "print_page_source": "derived",
            "whats_news": {"groups": [{"name": "Worldwide", "items": [{"text": "Summary"}]}]},
        }
        enriched = enrich_page_result(
            existing,
            "WSJ",
            1,
            header_blocks=[],
            whats_news_blocks=[],
        )
        self.assertEqual(enriched["whats_news"], existing["whats_news"])

    def test_unrelated_native_blocks_merge_without_dropping_fallback_layout(self) -> None:
        existing = {
            "page": 1,
            "print_page_label": "A1",
            "print_section": "PAGE ONE",
            "print_page_source": "derived",
            "print_layout_version": PRINT_LAYOUT_VERSION,
            "print_layout_status": "found",
            "whats_news": {
                "groups": [
                    {"name": "Worldwide", "items": [{"text": "Summary"}]}
                ]
            },
        }
        unrelated = [{"text": "An unrelated native article block", "bbox": [150, 180, 500, 260]}]
        enriched = enrich_page_result(
            existing,
            "WSJ",
            1,
            header_blocks=unrelated,
            whats_news_blocks=unrelated,
            page_width=600,
            page_height=800,
            existing_metadata="merge",
        )
        self.assertEqual(enriched["print_page_label"], "A1")
        self.assertEqual(enriched["print_section"], "PAGE ONE")
        self.assertEqual(enriched["whats_news"], existing["whats_news"])

    def test_scanned_legacy_cache_without_header_stays_unattempted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            image_dir = root / "images"
            cache_dir.mkdir()
            image_dir.mkdir()
            cache_path = cache_dir / "page_2.json"
            cache_path.write_text('{"page": 2, "articles": []}', encoding="utf-8")
            cached = ScannedPdfStrategy(cache_enabled=True)._load_cached_page(
                2, image_dir, cache_dir, "FT"
            )
            persisted = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached is not None
        self.assertNotIn("print_page_label", cached)
        self.assertNotIn("print_page_label", persisted)
        self.assertNotIn("print_layout_version", persisted)

    def test_scanned_stale_ft_date_cache_clears_wrong_page_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir, image_dir = root / "cache", root / "images"
            cache_dir.mkdir(); image_dir.mkdir()
            cache_path = cache_dir / "page_25.json"
            cache_path.write_text(json.dumps({"page": 25, "articles": [], "header_text": "FINANCIAL TIMES\nJuly 25 2026\nNATIONAL", "print_page_label": "25", "print_section": "NATIONAL", "print_page_source": "header"}), encoding="utf-8")
            cached = ScannedPdfStrategy(cache_enabled=True)._load_cached_page(25, image_dir, cache_dir, "FT")
            persisted = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached is not None
        self.assertIsNone(cached["print_page_label"])
        self.assertEqual(cached["print_layout_status"], "found")
        self.assertEqual(cached["print_layout_version"], PRINT_LAYOUT_VERSION)
        self.assertEqual(persisted, cached)

    def test_scanned_stale_a6_cache_is_cleared_when_header_has_only_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir, image_dir = root / "cache", root / "images"
            cache_dir.mkdir(); image_dir.mkdir()
            cache_path = cache_dir / "page_6.json"
            cache_path.write_text(json.dumps({"page": 6, "articles": [], "header_text": "Story continued on A6\nMore reporting", "print_page_label": "A6", "print_section": "U.S. NEWS", "print_page_source": "header", "whats_news": {"groups": [{"name": "Worldwide", "items": []}]}}), encoding="utf-8")
            cached = ScannedPdfStrategy(cache_enabled=True)._load_cached_page(6, image_dir, cache_dir, "WSJ")
        assert cached is not None
        self.assertIsNone(cached["print_page_label"])
        self.assertIsNone(cached["print_section"])
        self.assertNotIn("whats_news", cached)
        self.assertEqual(cached["print_layout_status"], "not_found")

    def test_page_layout_propagates_to_articles_and_issue_metadata(self) -> None:
        page = enrich_page_result(
            {"page": 2, "articles": [{"title": "Trade", "category": "News"}]},
            "WSJ",
            2,
            "A2\nU.S. NEWS\nTrade",
        )
        article = _flatten_page_articles([page])[0]
        self.assertEqual(article["print_page_label"], "A2")
        self.assertEqual(article["print_section"], "U.S. NEWS")
        self.assertEqual(article["source_pages"], [2])
        self.assertEqual(build_issue_pages([page])[0]["print_page_label"], "A2")
        self.assertIsNone(build_front_page([page], "WSJ"))


class PrintLayoutDatabaseTests(unittest.TestCase):
    def test_ft_briefing_links_numeric_target_when_header_label_is_missing(self) -> None:
        metadata = PdfMetadata("FT", "2026-08-05")
        pages = [
            {"pdf_page": 1, "print_page_label": "1", "print_section": "PAGE ONE"},
            {"pdf_page": 6, "print_page_label": None, "print_section": "INTERNATIONAL"},
            {"pdf_page": 7, "print_page_label": None, "print_section": "COMPANIES & MARKETS"},
        ]
        front_page = {
            "pdf_page": 1,
            "print_page_label": "1",
            "directory_name": "Briefing",
            "whats_news": {
                "groups": [{
                    "name": "Briefing",
                    "items": [
                        {
                            "text": "Oil falls as Bessent fuels hopes for a Hormuz deal.",
                            "target_print_page_label": "6",
                            "target_article_id": None,
                        },
                        {
                            "text": "OpenAI bites into Apple after a trade secrets lawsuit.",
                            "target_print_page_label": None,
                            "target_article_id": None,
                        },
                    ],
                }],
            },
        }
        articles = [
            {
                "page": 6,
                "page_article_index": 1,
                "title": "Oil falls as Bessent fuels hopes for a Hormuz deal",
                "title_zh": "贝森特推动谈判希望，油价下跌",
                "category": "International",
                "print_page_label": None,
                "source_pages": [6],
                "content_markdown": "Oil prices fell after the latest diplomatic signals.",
            },
            {
                "page": 7,
                "page_article_index": 1,
                "title": "OpenAI Bites Into Apple",
                "title_zh": "OpenAI向苹果发难",
                "category": "Companies & Markets",
                "print_page_label": None,
                "source_pages": [7],
                "content_markdown": "The trade secrets lawsuit became unusually personal.",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output_results"
            issue_dir = output_root / "FT" / metadata.publication_date
            issue_dir.mkdir(parents=True)
            database_path, _ = write_pdf_database(
                output_root,
                issue_dir,
                metadata,
                "test.pdf",
                articles,
                pages=pages,
                front_page=front_page,
            )
            payload = read_pdf_database(database_path)

        assert payload is not None
        items = payload["front_page"]["whats_news"]["groups"][0]["items"]
        self.assertEqual(items[0]["target_article_id"], payload["articles"][0]["id"])
        self.assertEqual(items[0]["title_zh"], articles[0]["title_zh"])
        self.assertEqual(items[1]["target_article_id"], payload["articles"][1]["id"])
        self.assertEqual(items[1]["target_print_page_label"], "7")

    def test_database_roundtrip_preserves_layout_and_front_page(self) -> None:
        metadata = PdfMetadata("WSJ", "2026-07-25")
        pages = [
            {
                "pdf_page": 1,
                "page_order": 1,
                "print_page_label": "A1",
                "print_section": "PAGE ONE",
                "print_page_source": "header",
            }
        ]
        front_page = {
            "pdf_page": 1,
            "print_page_label": "A1",
            "print_section": "PAGE ONE",
            "whats_news": {
                "groups": [
                    {
                        "name": "Worldwide",
                        "items": [
                            {
                                "text": "A report continues inside.",
                                "target_print_page_label": "A6",
                                "target_article_id": None,
                            }
                        ],
                    }
                ]
            },
        }
        article = {
            "page": 1,
            "page_article_index": 1,
            "title": "Lead story",
            "category": "News",
            "print_page_label": "A1",
            "print_section": "PAGE ONE",
            "print_page_source": "header",
            "source_pages": [1, 6],
            "content_markdown": "Body",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            issue_dir = output_root / "WSJ" / metadata.publication_date
            issue_dir.mkdir(parents=True)
            database_path, _ = write_pdf_database(
                output_root,
                issue_dir,
                metadata,
                "test.pdf",
                [article],
                pages=pages,
                front_page=front_page,
                is_weekend=True,
            )
            payload = read_pdf_database(database_path)
            index = read_database_index(output_root / "database_index.js")

        assert payload is not None
        self.assertEqual(payload["pages"][0]["print_page_label"], "A1")
        self.assertEqual(
            payload["pages"][0]["article_ids"],
            [payload["articles"][0]["id"]],
        )
        self.assertEqual(payload["front_page"], front_page)
        self.assertTrue(index[0]["has_front_page"])
        self.assertTrue(payload["is_weekend"])
        self.assertTrue(index[0]["is_weekend"])
        self.assertEqual(payload["articles"][0]["print_page_label"], "A1")
        self.assertEqual(payload["articles"][0]["source_pages"], [1, 6])

    def test_database_links_whats_news_items_to_bilingual_article_titles(self) -> None:
        metadata = PdfMetadata("WSJ", "2026-08-05")
        pages = [
            {"pdf_page": 1, "print_page_label": "A1", "print_section": "PAGE ONE"},
            {"pdf_page": 17, "print_page_label": "B1", "print_section": "BUSINESS & FINANCE"},
            {"pdf_page": 18, "print_page_label": "B4", "print_section": "TECHNOLOGY"},
        ]
        front_page = {
            "pdf_page": 1,
            "print_page_label": "A1",
            "whats_news": {
                "groups": [{
                    "name": "Business & Finance",
                    "items": [
                        {
                            "text": "The Saudi Arabian Oil Company delivered bumper profits despite war disruptions.",
                            "target_print_page_label": "B1",
                            "target_article_id": None,
                        },
                        {
                            "text": "OpenAI bites into Apple after the AI start-up accused the iPhone maker.",
                            "target_print_page_label": None,
                            "target_article_id": None,
                        },
                    ],
                }],
            },
        }
        articles = [
            {
                "page": 17,
                "page_article_index": 1,
                "title": "Bilt Cards Hit by Another Snafu",
                "title_zh": "Bilt信用卡再出差错",
                "category": "Business & Finance",
                "print_page_label": "B1",
                "source_pages": [17],
                "content_markdown": "Cardholders received incorrect collection notices.",
            },
            {
                "page": 17,
                "page_article_index": 2,
                "title": "Aramco's Profit Hits $33 Billion Despite War Strain",
                "title_zh": "战争压力下沙特阿美利润达330亿美元",
                "category": "Business & Finance",
                "print_page_label": "B1",
                "source_pages": [17],
                "content_markdown": "The Saudi Arabian Oil Company reported bumper profits despite disruptions.",
            },
            {
                "page": 18,
                "page_article_index": 1,
                "title": "OpenAI Bites Into Apple",
                "title_zh": "OpenAI向苹果发难",
                "category": "Technology",
                "print_page_label": "B4",
                "source_pages": [18],
                "content_markdown": "The AI start-up accused the iPhone maker in a lawsuit.",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output_results"
            issue_dir = output_root / "WSJ" / metadata.publication_date
            issue_dir.mkdir(parents=True)
            database_path, _ = write_pdf_database(
                output_root,
                issue_dir,
                metadata,
                "test.pdf",
                articles,
                pages=pages,
                front_page=front_page,
            )
            payload = read_pdf_database(database_path)

        assert payload is not None
        item = payload["front_page"]["whats_news"]["groups"][0]["items"][0]
        self.assertEqual(item["title"], articles[1]["title"])
        self.assertEqual(item["title_zh"], articles[1]["title_zh"])
        self.assertEqual(item["target_article_id"], payload["articles"][1]["id"])
        inferred = payload["front_page"]["whats_news"]["groups"][0]["items"][1]
        self.assertEqual(inferred["target_article_id"], payload["articles"][2]["id"])
        self.assertEqual(inferred["target_print_page_label"], "B4")

    def test_writer_preserves_existing_metadata_when_arguments_are_none(self) -> None:
        metadata = PdfMetadata("WSJ", "2026-07-25")
        pages = [{"pdf_page": 1, "page_order": 1, "print_page_label": "A1"}]
        front_page = {"pdf_page": 1, "print_page_label": "A1", "whats_news": {"groups": []}}
        article = {"page": 1, "page_article_index": 1, "title": "Lead", "content_markdown": "Body"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            issue = root / "WSJ" / metadata.publication_date
            issue.mkdir(parents=True)
            path, _ = write_pdf_database(root, issue, metadata, "test.pdf", [article], pages=pages, front_page=front_page)
            write_pdf_database(root, issue, metadata, "test.pdf", [article])
            payload = read_pdf_database(path)
        assert payload is not None
        self.assertEqual(payload["pages"][0]["print_page_label"], "A1")
        self.assertEqual(payload["front_page"], front_page)

    def test_writer_does_not_add_weekend_field_when_it_is_unknown(self) -> None:
        metadata = PdfMetadata("WSJ", "2026-07-25")
        article = {
            "page": 1,
            "page_article_index": 1,
            "title": "Lead",
            "content_markdown": "Body",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            issue = root / "WSJ" / metadata.publication_date
            issue.mkdir(parents=True)
            path, _ = write_pdf_database(
                root, issue, metadata, "historical.pdf", [article]
            )
            payload = read_pdf_database(path)
            index = read_database_index(root / "database_index.js")
        assert payload is not None
        self.assertNotIn("is_weekend", payload)
        self.assertNotIn("is_weekend", index[0])

    def test_writer_explicit_none_clears_existing_metadata(self) -> None:
        metadata = PdfMetadata("WSJ", "2026-07-25")
        pages = [{"pdf_page": 1, "page_order": 1, "print_page_label": "A1"}]
        front_page = {"pdf_page": 1, "print_page_label": "A1"}
        article = {"page": 1, "page_article_index": 1, "title": "Lead", "content_markdown": "Body"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            issue = root / "WSJ" / metadata.publication_date
            issue.mkdir(parents=True)
            path, _ = write_pdf_database(
                root, issue, metadata, "test.pdf", [article],
                pages=pages, front_page=front_page,
            )
            write_pdf_database(
                root, issue, metadata, "test.pdf", [article],
                pages=None, front_page=None,
            )
            payload = read_pdf_database(path)
        assert payload is not None
        self.assertEqual(payload["pages"], [])
        self.assertIsNone(payload["front_page"])

    def test_page_article_ids_include_continued_source_pages(self) -> None:
        metadata = PdfMetadata("WSJ", "2026-07-25")
        pages = [{"pdf_page": 1}, {"pdf_page": 6}]
        article = {
            "page": 1,
            "page_article_index": 1,
            "title": "Continued report",
            "source_pages": [1, 6],
            "content_markdown": "Body",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            issue = root / "WSJ" / metadata.publication_date
            issue.mkdir(parents=True)
            path, _ = write_pdf_database(root, issue, metadata, "test.pdf", [article], pages=pages)
            payload = read_pdf_database(path)
        assert payload is not None
        article_id = payload["articles"][0]["id"]
        self.assertEqual(payload["pages"][0]["article_ids"], [article_id])
        self.assertEqual(payload["pages"][1]["article_ids"], [article_id])


class ParseResultCompatibilityTests(unittest.TestCase):
    def test_old_positional_arguments_keep_complete_and_failed_pages(self) -> None:
        result = ParseResult("body", "engine", [], False, (7,))
        self.assertFalse(result.complete)
        self.assertEqual(result.failed_pages, (7,))
        self.assertIsNone(result.pages)
        self.assertIsNone(result.front_page)


if __name__ == "__main__":
    unittest.main()
