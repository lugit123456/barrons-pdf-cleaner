from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from strategies.special_pages import (
    PAGE_TYPE_BUSINESS,
    PAGE_TYPE_CONTENTS,
    PAGE_TYPE_POLITICS,
    build_native_world_page_result,
    classify_native_page,
)
from strategies.strategy_scanned import (
    _article_reject_reason,
    _compile_token_budget,
    _compile_article_with_llm,
    _pending_compile_pages,
    _request_article_translation,
    _summary_length_bounds,
    _translation_prompt,
    _validate_summary_style,
    retry_deferred_compiles,
)


def block(
    block_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    size: float = 9,
) -> dict[str, object]:
    return {
        "id": block_id,
        "text": text,
        "bbox": [x0, y0, x1, y1],
        "max_font_size": size,
        "median_font_size": size,
    }


class SpecialPageTests(unittest.TestCase):
    def test_classifies_contents_from_prominent_top_heading(self) -> None:
        blocks = [
            block("b1", "4 The Economist July 25th 2026", 30, 24, 530, 32, 8),
            block("b2", "Contents", 31, 46, 103, 65, 18),
            block("b3", "Business", 50, 470, 100, 485, 9),
        ]
        self.assertEqual(classify_native_page(blocks, 756), PAGE_TYPE_CONTENTS)

    def test_world_pages_are_classified_and_assembled_in_column_order(self) -> None:
        blocks = [
            block("b1", "The world this week Politics", 31, 46, 233, 65, 18),
            block("b2", "First item begins.", 31, 200, 148, 450),
            block("b3", "First item continues.", 160, 83, 277, 112),
            block("b4", "Second item.", 160, 123, 277, 230),
            block("b5", "Third item.", 289, 83, 406, 200),
        ]
        self.assertEqual(classify_native_page(blocks, 756), PAGE_TYPE_POLITICS)
        result = build_native_world_page_result(
            page_number=6,
            page_type=PAGE_TYPE_POLITICS,
            parser="test",
            blocks=blocks,
            images=[{"rel_path": "images/page_6_fig_1.jpg"}],
            page_width=567,
            page_height=756,
            source_stats={"words": 12},
        )
        article = result["articles"][0]
        self.assertEqual(article["category"], "The world this week / Politics")
        self.assertEqual(
            article["content_markdown"].split("\n\n"),
            [
                "First item begins.",
                "First item continues.",
                "Second item.",
                "Third item.",
            ],
        )
        self.assertIsNone(_article_reject_reason(article))

    def test_business_heading_is_detected(self) -> None:
        blocks = [
            block("b1", "The world this week Business", 31, 46, 245, 65, 18),
        ]
        self.assertEqual(classify_native_page(blocks, 756), PAGE_TYPE_BUSINESS)


class DeferredCompileTests(unittest.TestCase):
    def test_failed_article_is_retried_without_losing_image_analysis(self) -> None:
        article = {
            "title": "Staying aloft",
            "page": 56,
            "content_markdown": "A sufficiently long extracted article body.",
            "compiled_article": False,
            "compile_status": "pending",
            "image_analysis_complete": True,
            "image_insights": [{"path": "images/page_56_fig_1.jpg"}],
        }

        def compile_success(**kwargs: object) -> dict[str, object]:
            source = dict(kwargs["article"])  # type: ignore[arg-type]
            source.update(
                {
                    "compiled_article": True,
                    "compile_status": "complete",
                    "title_zh": "保持飞行",
                    "summary_md": "### 一句话核心主旨\n完成",
                    "paragraphs": [
                        {"en_text": "English.", "zh_text": "中文。", "role": "body"}
                    ],
                }
            )
            return source

        env = {
            "LLM_COMPILE_ARTICLES": "true",
            "DEFERRED_COMPILE_ENABLED": "true",
            "DEFERRED_COMPILE_ROUNDS": "3",
            "DEFERRED_COMPILE_COOLDOWN_SECONDS": "0",
            "DEFERRED_COMPILE_WORKERS": "1",
            "LLM_ANALYZE_ARTICLE_IMAGES": "false",
            "LLM_GLOSSARY_ENABLED": "false",
        }
        checkpoints = []
        with patch.dict(os.environ, env, clear=False), patch(
            "strategies.strategy_scanned._compile_article_with_llm",
            side_effect=compile_success,
        ) as mocked:
            result = retry_deferred_compiles(
                [article],
                on_article_updated=lambda index, value: checkpoints.append(
                    (index, value["compile_status"])
                ),
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertTrue(result[0]["compiled_article"])
        self.assertTrue(result[0]["image_analysis_complete"])
        self.assertEqual(_pending_compile_pages(result), set())
        self.assertEqual(checkpoints, [(0, "complete")])

    def test_remaining_failure_marks_page_pending(self) -> None:
        article = {
            "title": "Still pending",
            "page": 9,
            "content_markdown": "Extracted body.",
            "compiled_article": False,
        }
        env = {
            "LLM_COMPILE_ARTICLES": "true",
            "DEFERRED_COMPILE_ENABLED": "true",
            "DEFERRED_COMPILE_ROUNDS": "1",
            "DEFERRED_COMPILE_COOLDOWN_SECONDS": "0",
            "LLM_ANALYZE_ARTICLE_IMAGES": "false",
            "LLM_GLOSSARY_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "strategies.strategy_scanned._compile_article_with_llm",
            return_value=article,
        ):
            result = retry_deferred_compiles([article])
        self.assertEqual(_pending_compile_pages(result), {9})

    def test_compile_token_budget_scales_with_article_length(self) -> None:
        env = {
            "LLM_COMPILE_MIN_TOKENS": "5000",
            "LLM_COMPILE_MAX_TOKENS": "16000",
        }
        with patch.dict(os.environ, env, clear=False):
            short_budget = _compile_token_budget("word " * 120)
            typical_budget = _compile_token_budget("word " * 1000)
            long_budget = _compile_token_budget("word " * 6000)
        self.assertEqual(short_budget, 5000)
        self.assertEqual(typical_budget, 7000)
        self.assertEqual(long_budget, 16000)

    def test_successful_recompile_invalidates_old_glossary_positions(self) -> None:
        article = {
            "title": "Recompiled",
            "content_markdown": "Old local paragraph.",
            "glossary_analysis_complete": True,
            "glossary_version": 3,
            "glossary_entries": [{"id": "old"}],
            "term_annotations": [{"glossary_id": "old", "paragraph_index": 1}],
        }
        with patch(
            "strategies.strategy_scanned._request_article_translation",
            return_value=[{
                "en_text": "Old local paragraph.",
                "zh_text": "旧的本地段落。",
                "role": "body",
            }],
        ), patch(
            "strategies.strategy_scanned._request_article_summary",
            return_value=("重新编译", "第一段。\n\n第二段。\n\n第三段。"),
        ):
            compiled = _compile_article_with_llm(1, article)
        self.assertTrue(compiled["compiled_article"])
        self.assertTrue(compiled["glossary_invalidated"])
        self.assertEqual(compiled["content_markdown"], "Old local paragraph.")
        self.assertEqual(compiled["paragraphs"][0]["zh_text"], "旧的本地段落。")
        self.assertNotIn("term_annotations", compiled)
        self.assertNotIn("glossary_entries", compiled)

    def test_translation_prompt_requires_natural_financial_chinese(self) -> None:
        prompt = _translation_prompt(
            "A Stock Has More Upside",
            "MARKET VIEW",
            [{"en_text": "Analysts raised guidance.", "role": "body"}],
        )
        self.assertIn("不逐词照搬英文语序", prompt)
        self.assertIn("upside、downside、guidance", prompt)
        self.assertIn("不把预测或分析师观点写成确定事实", prompt)
        self.assertIn("段落顺序和数量完全一致", prompt)

    def test_translation_keeps_source_english_and_requires_one_to_one_output(self) -> None:
        source = [
            {"en_text": "First exact paragraph.", "zh_text": "", "role": "body"},
            {"en_text": "Second exact paragraph.", "zh_text": "", "role": "crosshead"},
        ]
        response = {
            "paragraphs": [
                {"zh_text": "第一段准确译文。"},
                {"zh_text": "第二段准确译文。"},
            ]
        }
        with patch(
            "strategies.strategy_scanned._chat_completion_json",
            return_value=response,
        ):
            translated = _request_article_translation(
                title="Test",
                category="MARKET VIEW",
                source_paragraphs=source,
                max_retries=1,
            )
        self.assertEqual(
            [paragraph["en_text"] for paragraph in translated],
            ["First exact paragraph.", "Second exact paragraph."],
        )
        self.assertEqual(translated[1]["role"], "crosshead")

        with patch(
            "strategies.strategy_scanned._chat_completion_json",
            return_value={"paragraphs": [{"zh_text": "只有一段。"}]},
        ):
            with self.assertRaisesRegex(RuntimeError, "数量不合格"):
                _request_article_translation(
                    title="Test",
                    category="MARKET VIEW",
                    source_paragraphs=source,
                    max_retries=1,
                )

    def test_summary_length_scales_with_source_article(self) -> None:
        self.assertEqual(
            _summary_length_bounds([{"en_text": "word " * 500}]),
            (420, 650),
        )
        self.assertEqual(
            _summary_length_bounds([{"en_text": "word " * 1200}]),
            (520, 800),
        )
        self.assertEqual(
            _summary_length_bounds([{"en_text": "word " * 2000}]),
            (620, 1000),
        )

    def test_summary_style_rejects_outlines_and_long_sentences(self) -> None:
        with self.assertRaisesRegex(ValueError, "不得使用标题"):
            _validate_summary_style("# 第一段\n\n第二段。\n\n第三段。")
        with self.assertRaisesRegex(ValueError, "过长句子"):
            _validate_summary_style(
                ("很" * 91) + "。\n\n第二段。\n\n第三段。"
            )


if __name__ == "__main__":
    unittest.main()
