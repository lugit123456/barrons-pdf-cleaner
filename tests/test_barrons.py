from __future__ import annotations

import json
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from database_writer import (
    build_database_article,
    read_database_index,
    read_pdf_database,
    write_pdf_database,
)
from metadata_parser import PdfMetadata
from print_layout import extract_print_layout
from scripts.watch_and_process import discover_candidates, load_config
from strategies.special_pages import (
    PAGE_TYPE_NORMAL,
    PAGE_TYPE_UTILITY,
    classify_native_page,
)
from strategies.strategy_hybrid import (
    _barrons_headline_text,
    _build_local_layout_articles,
)
from strategies.strategy_scanned import (
    _repair_short_articles,
    _simple_paragraph_cleanup,
)


class BarronsMetadataTests(unittest.TestCase):
    def test_print_layout_extracts_barrons_page_and_section(self) -> None:
        result = extract_print_layout(
            "BARRONS",
            7,
            header_text="October 5, 2026 BARRON'S 7\nUP & DOWN WALL STREET",
        )
        self.assertEqual(result["print_page_label"], "7")
        self.assertEqual(result["print_section"], "UP & DOWN WALL STREET")

    def test_multiline_barrons_header_extracts_page_number(self) -> None:
        result = extract_print_layout(
            "BARRONS",
            7,
            header_blocks=[
                {
                    "text": "October 5, 2026\nBARRON’S\n7",
                    "bbox": [22.5, 14.4, 734.8, 23.9],
                }
            ],
            page_width=756,
        )
        self.assertEqual(result["print_page_label"], "7")

    def test_barrons_data_page_is_utility(self) -> None:
        blocks = [
            {
                "text": "DATA BARRONS.COM/DATA",
                "bbox": [20, 29, 680, 44],
                "max_font_size": 11.7,
            }
        ]
        self.assertEqual(
            classify_native_page(blocks, 765, "BARRONS"),
            PAGE_TYPE_UTILITY,
        )

    def test_article_heading_is_not_utility(self) -> None:
        blocks = [
            {
                "text": "UP & DOWN WALL STREET",
                "bbox": [20, 35, 300, 80],
                "max_font_size": 40,
            }
        ]
        self.assertEqual(
            classify_native_page(blocks, 765, "BARRONS"),
            PAGE_TYPE_NORMAL,
        )


class WatcherSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "incoming"
        self.input_dir.mkdir()
        self.config_path = self.root / "config.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_pdf(self, filename: str, mtime: float) -> Path:
        path = self.input_dir / filename
        path.write_bytes(b"%PDF-1.7\n" + b"x" * 1100 + b"\n%%EOF\n")
        path.touch()
        path.chmod(0o644)
        import os

        os.utime(path, (mtime, mtime))
        return path

    def write_config(self, selection_mode: str) -> dict:
        payload = {
            "input_dir": str(self.input_dir),
            "output_dir": "./out",
            "state_db": "./state.sqlite3",
            "pattern": "*Barron*.pdf",
            "selection_mode": selection_mode,
            "stable_seconds": 0,
            "min_pdf_bytes": 1024,
        }
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")
        return load_config(self.config_path)

    def test_latest_uses_publication_date_before_mtime(self) -> None:
        now = time.time()
        self.write_pdf("Barron's - September 28 2026.pdf", now)
        newest = self.write_pdf("Barron's - October 5 2026.pdf", now - 500)
        config = self.write_config("latest")

        selected = discover_candidates(config, config_path=self.config_path)

        self.assertEqual(selected, [newest])

    def test_all_unprocessed_returns_newest_first(self) -> None:
        now = time.time()
        older = self.write_pdf("Barron's - September 28 2026.pdf", now)
        newest = self.write_pdf("Barron's - October 5 2026.pdf", now - 500)
        config = self.write_config("all_unprocessed")

        selected = discover_candidates(config, config_path=self.config_path)

        self.assertEqual(selected, [newest, older])

    def test_original_content_mode_is_the_default(self) -> None:
        config = self.write_config("latest")
        self.assertEqual(config["content_mode"], "original")


class DatabaseSchemaTests(unittest.TestCase):
    REFERENCE_OUTPUT = Path(
        "/Users/luzhe/Desktop/code/agent_skills/"
        "economist_weekly_archiver_skill/output_results"
    )

    def _reference_issue(self) -> dict:
        path = self.REFERENCE_OUTPUT / "TE" / "2026-08-08" / "database.js"
        if not path.exists():
            self.skipTest(f"Economist reference output is unavailable: {path}")
        text = path.read_text(encoding="utf-8")
        match = re.search(
            r"window\.paper_databases\[[^\]]+\]\s*=\s*([\s\S]*?);\s*$",
            text,
        )
        self.assertIsNotNone(match)
        return json.loads(match.group(1))

    def test_article_schema_matches_economist_weekly(self) -> None:
        metadata = PdfMetadata("BARRONS", "2026-10-05")
        article = build_database_article(
            {
                "title": "Sample",
                "category": "Sample title fragment J",
                "print_section": "MARKET VIEW",
                "page": 7,
                "page_article_index": 1,
                "content_markdown": "A complete sample paragraph.",
                "paragraphs": [
                    {
                        "en_text": "A complete sample paragraph.",
                        "zh_text": "一段完整的示例文字。",
                        "role": "body",
                    }
                ],
                "title_zh": "示例",
                "summary_md": "摘要",
                "compiled_article": True,
            },
            1,
            "BARRONS_2026-10-05_sample",
            metadata,
            "Barron's - October 5 2026.pdf",
            {"MARKET VIEW": "市场观点"},
        )
        self.assertEqual(
            list(article),
            [
                "id", "publication_type", "publication_date", "source_pdf",
                "page", "page_article_index", "category", "title", "title_zh",
                "markdown_path", "summary_md", "compiled_article",
                "compile_status", "content_markdown", "content_raw",
                "paragraphs", "images", "image_insights",
                "term_annotations", "glossary_analysis_complete",
                "glossary_version",
            ],
        )
        self.assertEqual(article["id"], "art_2026-10-05_001")
        self.assertEqual(article["category"], "MARKET VIEW")
        self.assertEqual(
            article["markdown_path"],
            "articles/art_2026-10-05_001.md",
        )

    def test_standard_article_category_wins_over_print_section(self) -> None:
        metadata = PdfMetadata("BARRONS", "2026-10-05")
        article = build_database_article(
            {
                "title": "The Market Looks Ahead to Earnings Season",
                "category": "MARKET WEEK",
                "print_section": "WINNERS & LOSERS",
                "page": 29,
                "content_markdown": "A complete article body.",
            },
            1,
            "BARRONS_2026-10-05_sample",
            metadata,
            "Barron's - October 5 2026.pdf",
        )
        self.assertEqual(article["category"], "MARKET WEEK")

    def test_barrons_output_keys_match_real_economist_output(self) -> None:
        reference = self._reference_issue()
        metadata = PdfMetadata("BARRONS", "2026-10-05")
        article = {
            "title": "Sample",
            "category": "MARKET VIEW",
            "page": 7,
            "page_article_index": 1,
            "content_markdown": "Test Person wrote a complete sample paragraph.",
            "paragraphs": [{
                "en_text": "Test Person wrote a complete sample paragraph.",
                "zh_text": "Test Person 写了一段完整的示例文字。",
                "role": "body",
            }],
            "title_zh": "示例",
            "summary_md": "摘要",
            "compiled_article": True,
            "glossary_analysis_complete": True,
            "glossary_version": 2,
            "glossary_entries": [{
                "id": "person-test-person",
                "term": "Test Person",
                "term_zh": "测试人物",
                "type": "person",
                "description_zh": "用于验证真实 Economist glossary 数据结构的测试人物。",
                "version": 2,
            }],
            "term_annotations": [{
                "glossary_id": "person-test-person",
                "paragraph_index": 1,
                "text_field": "zh_text",
                "surface": "Test Person",
                "occurrence": 1,
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            issue_dir = output_root / "BARRONS" / metadata.publication_date
            issue_dir.mkdir(parents=True)
            database_path, index_path = write_pdf_database(
                output_root,
                issue_dir,
                metadata,
                "Barron's - October 5 2026.pdf",
                [article],
            )
            actual = read_pdf_database(database_path)
            actual_index = read_database_index(index_path)[0]

        assert actual is not None
        reference_article = reference["articles"][0]
        self.assertEqual(list(actual), list(reference))
        self.assertEqual(list(actual["articles"][0]), list(reference_article))
        self.assertEqual(
            list(actual["articles"][0]["paragraphs"][0]),
            list(reference_article["paragraphs"][0]),
        )
        self.assertEqual(
            list(next(iter(actual["glossary"].values()))),
            list(next(iter(reference["glossary"].values()))),
        )

        reference_index_path = self.REFERENCE_OUTPUT / "database_index.js"
        index_text = reference_index_path.read_text(encoding="utf-8")
        index_match = re.search(
            r"window\.paper_db_index\s*=\s*([\s\S]*?);\s*$",
            index_text,
        )
        self.assertIsNotNone(index_match)
        reference_index = json.loads(index_match.group(1))[0]
        self.assertEqual(list(actual_index), list(reference_index))


class BarronsLocalRecoveryTests(unittest.TestCase):
    def test_body_columns_and_dropcaps_follow_print_reading_order(self) -> None:
        blocks = [
            {
                "id": "title",
                "bbox": [20, 35, 400, 140],
                "text": "AIWillTransformMedicine.\nThisStockStandstoGain.",
                "max_font_size": 44,
                "median_font_size": 44,
            },
            {
                "id": "deck",
                "bbox": [20, 160, 320, 335],
                "text": "A useful deck.\nA",
                "max_font_size": 94,
                "median_font_size": 9.8,
            },
            {
                "id": "column-1",
                "bbox": [20, 240, 170, 745],
                "text": "rtificial " + " ".join(["first"] * 50),
                "max_font_size": 8.8,
                "median_font_size": 8.8,
            },
            {
                "id": "column-2",
                "bbox": [182, 218, 332, 487],
                "text": "Second " + " ".join(["column"] * 50),
                "max_font_size": 8.8,
                "median_font_size": 8.8,
            },
            {
                "id": "column-3",
                "bbox": [583, 46, 733, 174],
                "text": "Third " + " ".join(["column"] * 30) + "\nI",
                "max_font_size": 51,
                "median_font_size": 8.8,
            },
            {
                "id": "column-4",
                "bbox": [583, 122, 733, 745],
                "text": "n its " + " ".join(["final"] * 50),
                "max_font_size": 8.8,
                "median_font_size": 8.8,
            },
        ]
        article = _build_local_layout_articles(
            12, 756, 765, blocks, [], publication_type="BARRONS"
        )[0]
        self.assertTrue(article["content_markdown"].startswith("Artificial"))
        self.assertLess(
            article["content_markdown"].index("first"),
            article["content_markdown"].index("Second"),
        )
        self.assertIn("In its", article["content_markdown"])

    def test_repairs_real_glued_barrons_headlines(self) -> None:
        cases = {
            "Overseasto TheRescue": "Overseas to The Rescue",
            "TheHottestSectorAfter AIIsHavingaMoment": (
                "The Hottest Sector After AI Is Having a Moment"
            ),
            "AIWillTransformMedicine. ThisStockStandstoGain.": (
                "AI Will Transform Medicine. This Stock Stands to Gain."
            ),
            "AllegedPonziScheme LeftInvestorsintheLurch": (
                "Alleged Ponzi Scheme Left Investors in the Lurch"
            ),
            "AIAgentsGot CashtoTrade Stocks.What TheyDidNext.": (
                "AI Agents Got Cash to Trade Stocks. What They Did Next."
            ),
            "ADividedStock MarketCouldSet UpaSeriousRally": (
                "A Divided Stock Market Could Set Up a Serious Rally"
            ),
            "FocusedFundsCanHelpYou NavigateaTop-HeavyMarket": (
                "Focused Funds Can Help You Navigate a Top-Heavy Market"
            ),
            "BitcoinFundsBatteredBond FundsintheThirdQuarter": (
                "Bitcoin Funds Battered Bond Funds in the Third Quarter"
            ),
            "TheMarket Looks Aheadto Earnings Season": (
                "The Market Looks Ahead to Earnings Season"
            ),
            "MetaCEOZuckerberg Sells$21.3MillionofStock": (
                "Meta CEO Zuckerberg Sells $21.3 Million of Stock"
            ),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(_barrons_headline_text(source), expected)

    def test_known_company_name_is_not_over_segmented(self) -> None:
        self.assertEqual(
            _barrons_headline_text("Coke,Mondelez, AndOther StaplesBargains"),
            "Coke, Mondelez, And Other Staples Bargains",
        )

    def test_mixed_headline_and_dropcap_block_is_recovered(self) -> None:
        blocks = [
            {
                "id": "b001",
                "bbox": [20, 38, 330, 85],
                "text": "UP & DOWN WALL STREET",
                "max_font_size": 40,
                "median_font_size": 40,
            },
            {
                "id": "b002",
                "bbox": [20, 107, 405, 382],
                "text": "The New Math of AI:\nAre Those Trillion-\nDollar Numbers Real?\nW",
                "max_font_size": 96,
                "median_font_size": 39,
            },
            {
                "id": "b003",
                "bbox": [20, 283, 171, 746],
                "text": " ".join(["body"] * 90),
                "max_font_size": 8.8,
                "median_font_size": 8.8,
            },
        ]
        articles = _build_local_layout_articles(
            7, 756, 765, blocks, [], publication_type="BARRONS"
        )
        self.assertEqual(len(articles), 1)
        self.assertEqual(
            articles[0]["title"],
            "The New Math of AI: Are Those Trillion-Dollar Numbers Real?",
        )
        self.assertEqual(articles[0]["category"], "UP & DOWN WALL STREET")
        self.assertTrue(articles[0]["content_markdown"].startswith("W body"))

    def test_top_headline_wins_over_large_dropcap_deck(self) -> None:
        blocks = [
            {
                "id": "title",
                "bbox": [21, 37, 374, 142],
                "text": "The Hottest Sector After AI Is Having a Moment",
                "max_font_size": 44,
                "median_font_size": 44,
            },
            {
                "id": "deck",
                "bbox": [21, 162, 320, 334],
                "text": "Quantum computing is becoming a commercial market. "
                + " ".join(["context"] * 40)
                + "\nA",
                "max_font_size": 93,
                "median_font_size": 9.7,
            },
            {
                "id": "body",
                "bbox": [21, 241, 168, 700],
                "text": " ".join(["body"] * 100),
                "max_font_size": 8.7,
                "median_font_size": 8.7,
            },
        ]
        articles = _build_local_layout_articles(
            10, 756, 765, blocks, [], publication_type="BARRONS"
        )
        self.assertEqual(
            articles[0]["title"],
            "The Hottest Sector After AI Is Having a Moment",
        )

    def test_complex_section_extracts_primary_article_locally(self) -> None:
        blocks = [
            {
                "id": "section",
                "bbox": [22, 38, 253, 85],
                "text": "REVIEW & PREVIEW",
                "max_font_size": 40,
                "median_font_size": 40,
            },
            {
                "id": "title",
                "bbox": [22, 321, 207, 414],
                "text": "Overseas to the Rescue",
                "max_font_size": 38,
                "median_font_size": 38,
            },
            {
                "id": "body",
                "bbox": [22, 420, 171, 740],
                "text": " ".join(["body"] * 100),
                "max_font_size": 8.8,
                "median_font_size": 8.8,
            },
        ]
        articles = _build_local_layout_articles(
            9, 756, 765, blocks, [], publication_type="BARRONS"
        )
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["title"], "Overseas to the Rescue")
        self.assertEqual(articles[0]["category"], "REVIEW & PREVIEW")

    def test_real_estate_promotion_is_not_an_article_headline(self) -> None:
        blocks = [
            {
                "id": "promo",
                "bbox": [405, 145, 701, 220],
                "text": "Your New Daily Real Estate Obsession",
                "max_font_size": 30,
                "median_font_size": 30,
            },
            {
                "id": "body",
                "bbox": [22, 230, 172, 661],
                "text": " ".join(["body"] * 100),
                "max_font_size": 8.8,
                "median_font_size": 8.8,
            },
        ]
        articles = _build_local_layout_articles(
            17, 756, 765, blocks, [], publication_type="BARRONS"
        )
        self.assertEqual(articles, [])

    def test_streetwise_subscription_block_is_excluded(self) -> None:
        blocks = [
            {
                "id": "title",
                "bbox": [20, 38, 500, 100],
                "text": "How to Pretend To Understand Persistent Agents",
                "max_font_size": 40,
                "median_font_size": 40,
            },
            {
                "id": "body",
                "bbox": [20, 120, 170, 700],
                "text": " ".join(["body"] * 100),
                "max_font_size": 8.8,
                "median_font_size": 8.8,
            },
            {
                "id": "promo",
                "bbox": [340, 630, 565, 735],
                "text": (
                    "In a weekly podcast by Barron's, columnist Jack Hough looks at "
                    "the companies, people, and trends you should be watching. This is "
                    "Wall Street like you've never heard before. Subscribe to Barron's "
                    "Streetwise on Spotify, Apple Podcasts, or your favorite listening app."
                ),
                "max_font_size": 8.8,
                "median_font_size": 8.8,
            },
        ]
        articles = _build_local_layout_articles(
            8, 756, 765, blocks, [], publication_type="BARRONS"
        )
        self.assertEqual(len(articles), 1)
        self.assertNotIn("Subscribe to Barron's", articles[0]["content_markdown"])


class OriginalOnlyTests(unittest.TestCase):
    def test_local_cleanup_preserves_compounds_and_repairs_split_words(self) -> None:
        cleaned = _simple_paragraph_cleanup(
            "The govern- ment reviewed trillion- dollar artificial- intelligence plans."
        )
        self.assertEqual(
            cleaned,
            "The government reviewed trillion-dollar artificial-intelligence plans.",
        )

    def test_short_article_repair_is_disabled_with_article_compilation(self) -> None:
        articles = [{
            "title": "Short candidate",
            "category": "History",
            "content_markdown": "too short",
        }]
        with patch.dict("os.environ", {"LLM_COMPILE_ARTICLES": "false"}), patch(
            "strategies.strategy_scanned._repair_article_content_with_llm"
        ) as repair:
            result = _repair_short_articles(5, "raw page", articles)
        self.assertEqual(result, articles)
        repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
