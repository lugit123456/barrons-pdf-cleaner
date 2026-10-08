from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import _existing_output_is_complete
from metadata_parser import PdfMetadata


class CompletionStateTests(unittest.TestCase):
    def test_uncompiled_article_keeps_issue_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "TE" / "2026-07-25" / "cache_json"
            cache_dir.mkdir(parents=True)
            (cache_dir / "page_1.json").write_text(
                json.dumps(
                    {
                        "page": 1,
                        "articles": [
                            {
                                "title": "Pending",
                                "content_markdown": "Extracted article body.",
                                "compiled_article": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            env = {"LLM_COMPILE_ARTICLES": "true"}
            with patch.dict(os.environ, env, clear=False), patch(
                "main._source_pdf_page_count",
                return_value=1,
            ):
                complete, reason = _existing_output_is_complete(
                    root,
                    PdfMetadata("TE", "2026-07-25"),
                    root / "input.pdf",
                )
            self.assertFalse(complete)
            self.assertIn("未完成翻译", reason)

    def test_skipped_contents_cache_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "TE" / "2026-07-25" / "cache_json"
            cache_dir.mkdir(parents=True)
            (cache_dir / "page_1.json").write_text(
                json.dumps(
                    {
                        "page": 1,
                        "page_type": "contents",
                        "skipped": True,
                        "articles": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"LLM_COMPILE_ARTICLES": "true"},
                clear=False,
            ), patch("main._source_pdf_page_count", return_value=1):
                complete, reason = _existing_output_is_complete(
                    root,
                    PdfMetadata("TE", "2026-07-25"),
                    root / "input.pdf",
                )
            self.assertTrue(complete)
            self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
