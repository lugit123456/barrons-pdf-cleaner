from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from database_writer import read_pdf_database, write_pdf_database
from glossary_enricher import (
    GLOSSARY_VERSION,
    enrich_article_glossary,
    extract_zh_english_candidates,
    normalize_glossary_response,
    normalize_term_annotations,
)
from metadata_parser import PdfMetadata


DESCRIPTION = (
    "杰罗姆·鲍威尔是美国联邦储备委员会主席，负责主持货币政策决策并代表美联储"
    "对外沟通。他关于利率、通胀与金融环境的表态会影响全球资产定价，因此是理解"
    "文章政策背景和市场反应的重要人物。"
)


class GlossaryEnricherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.article = {
            "title": "Powell and policy",
            "page": 1,
            "page_article_index": 1,
            "content_markdown": "Jerome Powell discussed monetary policy.",
            "paragraphs": [
                {
                    "en_text": "Jerome Powell discussed monetary policy.",
                    "zh_text": "杰罗姆·鲍威尔（Jerome Powell）讨论了货币政策。",
                    "role": "body",
                }
            ],
        }

    def test_enrichment_uses_exact_existing_surface(self) -> None:
        def fake_chat(**_: object) -> dict[str, object]:
            return {
                "terms": [
                    {
                        "term": "Jerome Powell",
                        "term_zh": "杰罗姆·鲍威尔",
                        "type": "person",
                        "description_zh": DESCRIPTION,
                        "occurrences": [
                            {"paragraph_index": 1, "surface": "jerome powell"}
                        ],
                    },
                ]
            }

        enriched = enrich_article_glossary(self.article, fake_chat)

        self.assertTrue(enriched["glossary_analysis_complete"])
        self.assertEqual(
            enriched["term_annotations"][0]["surface"],
            "Jerome Powell",
        )
        self.assertEqual(enriched["glossary_entries"][0]["type"], "person")

    def test_nonexistent_surface_is_dropped(self) -> None:
        entries, annotations = normalize_glossary_response(
            {
                "terms": [
                    {
                        "term": "Not In Article",
                        "type": "proper_concept",
                        "description_zh": DESCRIPTION,
                        "occurrences": [
                            {"paragraph_index": 1, "surface": "Not In Article"}
                        ],
                    }
                ]
            },
            paragraphs=self.article["paragraphs"],
            max_terms=8,
        )

        self.assertEqual(entries, [])
        self.assertEqual(annotations, [])

    def test_chinese_column_english_term_is_annotated(self) -> None:
        article = {
            **self.article,
            "paragraphs": [
                {
                    "en_text": "The historian published a major new book.",
                    "zh_text": "历史学家Manu Pillai出版了新作《Gods, Guns and Missionaries》。",
                    "role": "body",
                }
            ],
        }

        def fake_chat(**kwargs: object) -> dict[str, object]:
            prompt = kwargs["messages"][0]["content"]  # type: ignore[index]
            self.assertIn("[P1.ZH]", prompt)
            return {
                "terms": [
                    {
                        "term": "Manu Pillai",
                        "term_zh": "马努·皮莱",
                        "type": "person",
                        "description_zh": DESCRIPTION,
                        "occurrences": [
                            {
                                "paragraph_index": 1,
                                "text_field": "zh_text",
                                "surface": "Manu Pillai",
                            }
                        ],
                    },
                    {
                        "term": "Gods, Guns and Missionaries",
                        "term_zh": "《神、枪与传教士》",
                        "type": "work",
                        "description_zh": DESCRIPTION,
                        "occurrences": [
                            {
                                "paragraph_index": 1,
                                "text_field": "zh_text",
                                "surface": "Gods, Guns and Missionaries",
                            }
                        ],
                    },
                ]
            }

        enriched = enrich_article_glossary(article, fake_chat)

        self.assertEqual(enriched["glossary_version"], GLOSSARY_VERSION)
        self.assertEqual(
            enriched["term_annotations"][0]["text_field"],
            "zh_text",
        )
        self.assertEqual(enriched["term_annotations"][0]["surface"], "Manu Pillai")
        self.assertEqual(enriched["glossary_entries"][1]["type"], "work")

    def test_extracts_named_english_candidates_from_chinese_text(self) -> None:
        zh_text = (
            "旅游网站Booking.com发布调查。非营利组织DarkSky International负责认证。"
            "位于印度村庄Maan的Astro Retreat游客激增。沙漠中的Al Ula也在发展旅游。"
            "天文学家John Barentine支持这一趋势，DarkSky总监Ruskin Hartley也表示赞同。"
        )
        candidates = extract_zh_english_candidates(
            [(2, "", zh_text)],
            max_candidates=16,
        )

        self.assertEqual(
            [candidate["surface"] for candidate in candidates],
            [
                "Booking.com",
                "DarkSky International",
                "Maan",
                "Astro Retreat",
                "Al Ula",
                "John Barentine",
                "DarkSky",
                "Ruskin Hartley",
            ],
        )
        self.assertTrue(
            all(candidate["text_field"] == "zh_text" for candidate in candidates)
        )

    def test_candidate_hints_do_not_force_follow_up(self) -> None:
        article = {
            **self.article,
            "paragraphs": [
                {
                    "en_text": "The travel survey mentioned a certification group.",
                    "zh_text": "旅游网站Booking.com发布调查，组织DarkSky International负责认证。",
                    "role": "body",
                }
            ],
        }
        calls = []

        def fake_chat(**kwargs: object) -> dict[str, object]:
            prompt = kwargs["messages"][0]["content"]  # type: ignore[index]
            calls.append(prompt)
            if len(calls) == 1:
                return {
                    "terms": [
                        {
                            "term": "DarkSky International",
                            "term_zh": "国际暗夜协会",
                            "type": "organization",
                            "description_zh": DESCRIPTION,
                            "occurrences": [
                                {
                                    "paragraph_index": 1,
                                    "text_field": "zh_text",
                                    "surface": "DarkSky International",
                                }
                            ],
                        }
                    ]
                }
        enriched = enrich_article_glossary(article, fake_chat)

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            {annotation["surface"] for annotation in enriched["term_annotations"]},
            {"DarkSky International"},
        )

    def test_matching_term_is_only_added_to_chinese_column(self) -> None:
        paragraphs = [
            {
                "en_text": "Manu Pillai published a major new book.",
                "zh_text": "历史学家 Manu Pillai 出版了一部重要新作。",
                "role": "body",
            }
        ]
        entries, annotations = normalize_glossary_response(
            {
                "terms": [
                    {
                        "term": "Manu Pillai",
                        "term_zh": "马努·皮莱",
                        "type": "person",
                        "description_zh": DESCRIPTION,
                        "occurrences": [
                            {
                                "paragraph_index": 1,
                                "text_field": "en_text",
                                "surface": "Manu Pillai",
                            }
                        ],
                    }
                ]
            },
            paragraphs=paragraphs,
            max_terms=8,
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(
            [annotation["text_field"] for annotation in annotations],
            ["zh_text"],
        )

    def test_legacy_annotation_without_chinese_column_is_dropped(self) -> None:
        annotations = normalize_term_annotations(
            [
                {
                    "glossary_id": "person_manu-pillai",
                    "paragraph_index": 1,
                    "surface": "Manu Pillai",
                    "occurrence": 1,
                }
            ],
            paragraph_count=1,
        )

        self.assertEqual(annotations, [])

    def test_ordinary_abstract_word_is_rejected(self) -> None:
        paragraphs = [
            {
                "en_text": "Sovereignty was discussed.",
                "zh_text": "文章讨论了 sovereignty。",
                "role": "body",
            }
        ]
        entries, annotations = normalize_glossary_response(
            {
                "terms": [
                    {
                        "term": "sovereignty",
                        "type": "proper_concept",
                        "description_zh": DESCRIPTION,
                        "occurrences": [
                            {
                                "paragraph_index": 1,
                                "text_field": "zh_text",
                                "surface": "sovereignty",
                            }
                        ],
                    }
                ]
            },
            paragraphs=paragraphs,
            max_terms=8,
        )

        self.assertEqual(entries, [])
        self.assertEqual(annotations, [])

    def test_database_rebuild_preserves_existing_glossary(self) -> None:
        def fake_chat(**_: object) -> dict[str, object]:
            return {
                "terms": [
                    {
                        "term": "Jerome Powell",
                        "term_zh": "杰罗姆·鲍威尔",
                        "type": "person",
                        "description_zh": DESCRIPTION,
                        "occurrences": [
                            {"paragraph_index": 1, "surface": "Jerome Powell"}
                        ],
                    }
                ]
            }

        enriched = enrich_article_glossary(self.article, fake_chat)
        metadata = PdfMetadata("WSJ", "2026-07-25")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            issue_dir = output_root / "WSJ" / "2026-07-25"
            issue_dir.mkdir(parents=True)
            write_pdf_database(
                output_root,
                issue_dir,
                metadata,
                "test.pdf",
                [enriched],
            )
            write_pdf_database(
                output_root,
                issue_dir,
                metadata,
                "test.pdf",
                [self.article],
            )
            payload = read_pdf_database(issue_dir / "database.js")

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIn("person_jerome-powell", payload["glossary"])
        self.assertEqual(len(payload["articles"][0]["term_annotations"]), 1)
        self.assertTrue(payload["articles"][0]["paragraphs"][0]["para_id"])


if __name__ == "__main__":
    unittest.main()
